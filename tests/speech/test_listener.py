"""Background listener: trigger → record request → deliver → speak reply."""

from __future__ import annotations

import time

import numpy as np
import pytest

from openjarvis.speech import wakeword as ww
from openjarvis.speech.clap import ClapDetector
from openjarvis.speech.features import read_wav, write_wav
from openjarvis.speech.listener import (
    CommandRecorder,
    ListenerOptions,
    VoiceListener,
)

from ._synth import HEY_JARVIS, OTHER, OTHER2, SR, room, take


class FakeApi:
    def __init__(self, *, text="what time is it", reply="It is noon."):
        self.text = text
        self.reply = reply
        self.states: list[str] = []
        self.commands: list[bytes] = []
        self.synthesized: list[str] = []
        self.subscribers = 1

    def healthy(self):
        return True

    def get(self, path, timeout=5.0):
        if path.startswith("/v1/voice/status"):
            return {"subscribers": self.subscribers, "states": {}}
        if path.startswith("/v1/voice/reply/"):
            return {"text": self.reply}
        raise AssertionError(path)

    def post_json(self, path, data, timeout=10.0):
        assert path == "/v1/voice/state"
        self.states.append(data["state"])
        return {"ok": True}

    def post_wav(self, path, wav, timeout=120.0):
        assert path == "/v1/voice/command"
        self.commands.append(wav)
        return {"text": self.text, "delivered": 1, "id": "cmd1"}

    def synthesize(self, text, timeout=120.0):
        self.synthesized.append(text)
        return write_wav(np.zeros(1600, dtype=np.float32))


class FakeSpeaker:
    def __init__(self):
        self.played: list[int] = []

    def play(self, samples, rate, wait=False):
        self.played.append(len(samples))


def _clap(rng, amplitude=0.8):
    t = np.arange(int(0.12 * SR)) / SR
    burst = rng.normal(0, 1, t.size) * amplitude * np.exp(-t / 0.015)
    return (np.clip(burst, -1, 1) + room(0.12, rng)).astype(np.float32)


def _feed(listener, audio):
    for start in range(0, audio.size - 256, 256):
        listener.process(audio[start : start + 256])


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def _listener(api, speaker, **kwargs):
    opened = []
    listener = VoiceListener(
        api,
        options=ListenerOptions(),
        speaker=speaker,
        open_ui=lambda: opened.append(1) or "launched",
        log=lambda message: None,
        **kwargs,
    )
    return listener, opened


def test_double_clap_records_request_and_speaks_reply(rng):
    api, speaker = FakeApi(), FakeSpeaker()
    listener, opened = _listener(api, speaker, clap_detector=ClapDetector())
    listener._ui_open = True
    listener._last_poll = time.monotonic() + 3600  # no background status polls
    audio = np.concatenate(
        [
            room(1.0, rng),
            _clap(rng),
            room(0.3, rng),
            _clap(rng),
            room(1.2, rng),  # pattern confirmation + chirp
            take(OTHER, rng),  # the request
            room(2.0, rng),  # pause ends the recording
        ]
    )
    _feed(listener, audio)
    assert _wait(lambda: api.synthesized == ["It is noon."])
    assert len(api.commands) == 1
    recorded, _ = read_wav(api.commands[0])
    assert 0.8 * SR < recorded.size < 4.5 * SR
    assert _wait(lambda: api.states[-1:] == ["idle"])
    assert "listening" in api.states and "transcribing" in api.states
    assert "speaking" in api.states
    assert opened == []  # a page was already connected
    listener.close()


def test_wake_word_opens_ui_when_no_page_is_connected(rng, tmp_path):
    for _ in range(6):
        ww.add_sample(write_wav(take(HEY_JARVIS, rng)), "positive", root=tmp_path)
    for segments in (OTHER, OTHER2):
        ww.add_sample(write_wav(take(segments, rng)), "negative", root=tmp_path)
    detector = ww.WakeWordDetector(ww.train(root=tmp_path))
    api, speaker = FakeApi(), FakeSpeaker()
    api.subscribers = 0
    listener, opened = _listener(api, speaker, wake_detector=detector)
    listener._ui_open = False
    listener._last_poll = time.monotonic() + 3600  # keep "no page" for the test
    audio = np.concatenate(
        [
            room(1.0, rng),
            take(HEY_JARVIS, rng),
            room(0.6, rng),
            take(OTHER2, rng),
            room(2.0, rng),
        ]
    )
    _feed(listener, audio)
    assert opened == [1]
    assert _wait(lambda: len(api.commands) == 1)
    listener.close()


def test_open_only_mode_does_not_record(rng):
    api, speaker = FakeApi(), FakeSpeaker()
    listener = VoiceListener(
        api,
        options=ListenerOptions(converse=False),
        clap_detector=ClapDetector(),
        speaker=speaker,
        log=lambda m: None,
    )
    audio = np.concatenate(
        [room(1.0, rng), _clap(rng), room(0.3, rng), _clap(rng), room(2.0, rng)]
    )
    _feed(listener, audio)
    time.sleep(0.2)
    assert api.commands == [] and "listening" not in api.states
    assert speaker.played  # confirmation chirp
    listener.close()


def test_command_recorder_outcomes(rng):
    silent = CommandRecorder(0.003, start_timeout=1.0)
    outcome = None
    audio = room(2.0, rng)
    for start in range(0, audio.size - 256, 256):
        outcome = silent.feed(audio[start : start + 256]) or outcome
        if outcome:
            break
    assert outcome == "no-speech"

    speech = CommandRecorder(0.003, silence_seconds=0.5)
    audio = np.concatenate([room(0.3, rng), take(OTHER, rng), room(1.0, rng)])
    for start in range(0, audio.size - 256, 256):
        outcome = speech.feed(audio[start : start + 256])
        if outcome:
            break
    assert outcome == "done"
    assert speech.wav().startswith(b"RIFF")


def test_confirmed_wake_word_is_learned_and_actions_spoken_at_once(rng, tmp_path):
    for _ in range(6):
        ww.add_sample(write_wav(take(HEY_JARVIS, rng)), "positive", root=tmp_path)
    detector = ww.WakeWordDetector(ww.train(root=tmp_path))

    class ActionApi(FakeApi):
        def post_wav(self, path, wav, timeout=120.0):
            self.commands.append(wav)
            return {
                "text": "sube el volumen",
                "delivered": 0,
                "id": "a1",
                "action": {"reply": "Subí el volumen.", "ok": True},
            }

        def get(self, path, timeout=5.0):
            if path.startswith("/v1/voice/reply/"):
                raise AssertionError("actions must not wait for the model")
            return super().get(path, timeout)

    api, speaker = ActionApi(), FakeSpeaker()
    learned = []
    listener, _ = _listener(
        api, speaker, wake_detector=detector, learner=lambda a: learned.append(a)
    )
    listener._ui_open = True
    listener._last_poll = time.monotonic() + 3600
    audio = np.concatenate(
        [
            room(1.0, rng),
            take(HEY_JARVIS, rng),
            room(0.6, rng),
            take(OTHER2, rng),
            room(2.0, rng),
        ]
    )
    _feed(listener, audio)
    assert _wait(lambda: api.synthesized == ["Subí el volumen."])
    assert len(learned) == 1 and learned[0] is not None
    listener.close()
