"""Clap pattern detection on synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from openjarvis.speech.clap import (
    BLOCK_SIZE,
    SAMPLE_RATE,
    ClapDetector,
    block_stats,
    sensitivity_to_peak,
)

_rng = np.random.default_rng(0)


def _room(seconds: float, level: float = 0.005) -> np.ndarray:
    return _rng.normal(0, level, int(seconds * SAMPLE_RATE))


def _clap(amplitude: float = 0.8, decay: float = 0.015, seconds: float = 0.12):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    burst = _rng.normal(0, 1, t.size) * amplitude * np.exp(-t / decay)
    return np.clip(burst, -1, 1) + _room(seconds)


def _speech(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    # Noise with a 4 Hz syllable envelope: loud, with dips but no silence.
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = amplitude * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return np.clip(_rng.normal(0, 1, t.size) * envelope, -1, 1)


def _claps(count: int, gap: float, **kwargs) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for i in range(count):
        parts.append(_clap(**kwargs))
        if i < count - 1:
            parts.append(_room(gap))
    return parts


def _triggers(signal: np.ndarray, **kwargs) -> list[float]:
    detector = ClapDetector(**kwargs)
    hits = []
    for start in range(0, signal.size - BLOCK_SIZE, BLOCK_SIZE):
        if detector.process(*block_stats(signal[start : start + BLOCK_SIZE])):
            hits.append(start / SAMPLE_RATE)
    return hits


def _scene(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([_room(1.0), *parts, _room(1.5)])


def test_double_clap_triggers_once() -> None:
    assert len(_triggers(_scene(*_claps(2, 0.3)))) == 1


def test_single_clap_does_not_trigger() -> None:
    assert _triggers(_scene(_clap())) == []


def test_claps_too_far_apart_do_not_trigger() -> None:
    assert _triggers(_scene(*_claps(2, 1.6))) == []


@pytest.mark.parametrize("amplitude", [0.3, 0.8])
def test_speech_does_not_trigger(amplitude: float) -> None:
    assert _triggers(_scene(_speech(3.0, amplitude))) == []


def test_steady_beat_does_not_trigger() -> None:
    # Music or footsteps: many evenly spaced hits are a rhythm, not a signal.
    assert _triggers(_scene(*_claps(8, 0.38))) == []


def test_extra_clap_cancels_a_two_clap_pattern() -> None:
    assert _triggers(_scene(*_claps(3, 0.3))) == []


def test_three_clap_pattern_when_configured() -> None:
    assert len(_triggers(_scene(*_claps(3, 0.3)), claps_required=3)) == 1


def test_room_reverb_still_counts_as_a_clap() -> None:
    assert len(_triggers(_scene(*_claps(2, 0.3, decay=0.05, seconds=0.3)))) == 1


def test_two_separate_patterns_trigger_twice() -> None:
    signal = _scene(*_claps(2, 0.3), _room(2.0), *_claps(2, 0.4))
    assert len(_triggers(signal)) == 2


def test_soft_claps_need_more_sensitivity() -> None:
    soft = _scene(*_claps(2, 0.3, amplitude=0.12))
    assert _triggers(soft, min_peak=sensitivity_to_peak(0.0)) == []
    assert len(_triggers(soft, min_peak=sensitivity_to_peak(1.0))) == 1


def test_block_stats() -> None:
    peak, rms = block_stats(np.array([0.5, -0.5, 0.5, -0.5]))
    assert peak == pytest.approx(0.5)
    assert rms == pytest.approx(0.5)
    assert block_stats(np.array([])) == (0.0, 0.0)


def test_listen_for_claps_streams_microphone_blocks(monkeypatch) -> None:
    import sys
    import threading
    import time
    import types

    from openjarvis.speech import clap as clap_mod

    signal = _scene(*_claps(2, 0.3)).astype(np.float32)
    done = threading.Event()

    class FakeStream:
        def __init__(self, *, callback, blocksize, **_):
            self.callback = callback
            self.blocksize = blocksize

        def __enter__(self):
            def feed():
                for i in range(0, signal.size - self.blocksize, self.blocksize):
                    block = signal[i : i + self.blocksize].reshape(-1, 1)
                    self.callback(block, self.blocksize, None, None)
                done.set()

            threading.Thread(target=feed, daemon=True).start()
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeStream)
    )
    hits = []
    deadline = time.monotonic() + 10
    clap_mod.listen_for_claps(
        lambda: hits.append(1),
        ClapDetector(),
        # Stop once the pattern fired, or give up rather than hang the suite.
        should_stop=lambda: bool(hits) or time.monotonic() > deadline,
    )
    assert done.wait(5)
    assert hits == [1]
