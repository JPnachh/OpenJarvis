"""Tests for the hands-free voice loop and the double-clap detector."""

from __future__ import annotations

import numpy as np
import pytest

from openjarvis.speech.clap import ClapDetector
from openjarvis.speech.voice_daemon import (
    FRAME,
    SAMPLE_RATE,
    VoiceDaemon,
    is_cancel,
    record_utterance,
    strip_markdown,
    strip_wake_phrase,
)

SR = SAMPLE_RATE


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.int16)


def clap() -> np.ndarray:
    n = int(0.06 * SR)
    rng = np.random.default_rng(3)
    env = np.exp(-np.arange(n) / (0.008 * SR))
    return (rng.standard_normal(n) * env * 22000).astype(np.int16)


def speech(sec: float) -> np.ndarray:
    """Loud, sustained, speech-like signal (stays loud, so not a clap)."""
    t = np.arange(int(sec * SR)) / SR
    return (
        9000 * np.sin(2 * np.pi * 180 * t) * (1 + 0.4 * np.sin(2 * np.pi * 4 * t))
    ).astype(np.int16)


def frames_of(audio: np.ndarray):
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        yield audio[i : i + FRAME]


def detects(audio: np.ndarray) -> bool:
    det = ClapDetector()
    return any(det.process(f) for f in frames_of(audio))


# ----------------------------------------------------------------- clap


def test_double_clap_triggers():
    assert detects(
        np.concatenate([silence(1), clap(), silence(0.3), clap(), silence(1)])
    )


def test_single_clap_does_not_trigger():
    assert not detects(np.concatenate([silence(1), clap(), silence(2)]))


def test_claps_too_far_apart_do_not_trigger():
    assert not detects(
        np.concatenate([silence(1), clap(), silence(1.2), clap(), silence(1)])
    )


def test_sustained_speech_does_not_trigger():
    assert not detects(np.concatenate([silence(1), speech(3), silence(1)]))


def test_cooldown_prevents_immediate_retrigger():
    det = ClapDetector()
    audio = np.concatenate(
        [
            silence(1),
            clap(),
            silence(0.3),
            clap(),
            silence(0.3),
            clap(),
            silence(0.3),
            clap(),
            silence(1),
        ]
    )
    assert sum(det.process(f) for f in frames_of(audio)) == 1


# ------------------------------------------------------------ text helpers


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("Hey Jarvis, dime la hora", "dime la hora"),
        ("oye jarvis ¿qué tal?", "¿qué tal?"),
        ("Ey, Jarvis. Hola", "Hola"),
        ("dime algo", "dime algo"),
    ],
)
def test_strip_wake_phrase(raw, clean):
    assert strip_wake_phrase(raw) == clean


def test_is_cancel_ignores_accents_and_punctuation():
    assert (
        is_cancel("Olvídalo.")
        and is_cancel("¡Para!")
        and not is_cancel("para qué sirve")
    )


def test_strip_markdown_for_speech():
    out = strip_markdown(
        "# Título\n- **uno** y `dos`\n```py\nx=1\n```\nFin [web](http://a.b)"
    )
    assert out == "Título uno y dos Fin web"


# --------------------------------------------------------------- recording


def test_record_returns_none_when_nobody_speaks():
    assert record_utterance(frames_of(silence(10)), startup_s=1.0) is None


def test_record_stops_after_silence():
    audio = np.concatenate([silence(0.3), speech(1.0), silence(3)])
    wav = record_utterance(frames_of(audio), silence_s=0.8)
    assert wav is not None and wav[:4] == b"RIFF"
    assert len(wav) < 2 * len(audio)  # stopped before consuming everything


# ------------------------------------------------------------------ daemon


class _FireOnce:
    """Fake detector that fires when it sees the marker frame (value 7)."""

    def __init__(self):
        self.resets = 0

    def process(self, frame):
        return bool(frame[0] == 7)

    def reset(self):
        self.resets += 1


class _Source:
    def __init__(self, audio):
        self._audio = audio
        self.flushes = 0

    def flush(self):
        self.flushes += 1

    def __iter__(self):
        return frames_of(self._audio)


def _marker() -> np.ndarray:
    m = np.zeros(FRAME, dtype=np.int16)
    m[0] = 7
    return m


def _daemon(transcripts, replies, **kw):
    said, asked = [], []
    texts = iter(transcripts)

    d = VoiceDaemon(
        transcribe=lambda wav: next(texts, ""),
        ask=lambda t: (asked.append(t), replies.pop(0))[1],
        say=said.append,
        log=lambda m: None,
        wake=_FireOnce(),
        **kw,
    )
    return d, said, asked


def test_wake_turn_asks_and_speaks():
    audio = np.concatenate(
        [np.zeros(FRAME * 6, dtype=np.int16), _marker(), speech(1), silence(13)]
    )
    d, said, asked = _daemon(
        ["Hey Jarvis, qué hora es"], ["Son las tres."], max_turns=1
    )
    d.run(_Source(audio))
    assert asked == ["qué hora es"]
    assert said == ["Son las tres."]
    assert d.wake.resets == 1


def test_follow_up_without_wake_word_then_returns_to_idle():
    audio = np.concatenate(
        [_marker(), speech(1), silence(2), speech(1), silence(2), silence(12)]
    )
    d, said, asked = _daemon(["uno", "dos"], ["r1", "r2"], follow_up_s=6.0)
    d.run(_Source(audio))
    assert asked == ["uno", "dos"]
    assert said == ["r1", "r2"]


def test_cancel_word_skips_the_server():
    audio = np.concatenate([_marker(), speech(1), silence(3), silence(10)])
    d, said, asked = _daemon(["cancela"], [], max_turns=1)
    d.run(_Source(audio))
    assert asked == [] and said == []


def test_no_trigger_means_no_turn():
    d, said, asked = _daemon(["x"], [])
    d.run(_Source(np.concatenate([speech(2), silence(2)])))
    assert asked == [] and said == []
