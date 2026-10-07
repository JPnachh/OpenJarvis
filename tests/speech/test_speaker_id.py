"""Speaker voiceprint logic, with a fake embedding extractor (no model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from openjarvis.speech.speaker_id import SAMPLE_RATE, SpeakerVerifier


class _FakeStream:
    def __init__(self):
        self.wave = None

    def accept_waveform(self, sample_rate, waveform):
        self.wave = waveform

    def input_finished(self):
        pass


class _FakeExtractor:
    """Embedding = the audio's mean level pattern: sign decides the 'speaker'."""

    def create_stream(self):
        return _FakeStream()

    def compute(self, stream):
        base = (
            np.array([1.0, 0.2, 0.1, 0.0])
            if stream.wave.mean() >= 0
            else np.array([0.0, 0.1, 0.2, 1.0])
        )
        rng = np.random.default_rng(int(abs(stream.wave[:50].sum() * 1000)) % 1000)
        return (base + 0.05 * rng.standard_normal(4)).tolist()


def clip(sign: float, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    return (sign * 3000 + 100 * rng.standard_normal(n)).astype(np.int16)


@pytest.fixture
def verifier(tmp_path):
    return SpeakerVerifier(tmp_path / "vp.npz", extractor=_FakeExtractor())


def test_not_enrolled_accepts_everyone(verifier):
    assert verifier.verify(clip(-1)) == (True, None)


def test_enroll_then_accept_owner_reject_stranger(verifier):
    info = verifier.enroll([clip(+1, seed=i) for i in range(5)])
    assert info["samples"] == 5
    assert verifier.verify(clip(+1, seed=99))[0] is True
    assert verifier.verify(clip(-1, seed=98))[0] is False


def test_enroll_needs_three_usable_samples(verifier):
    with pytest.raises(ValueError):
        verifier.enroll([clip(+1), clip(+1), clip(+1, seconds=0.2)])


def test_short_audio_is_not_rejected(verifier):
    verifier.enroll([clip(+1, seed=i) for i in range(4)])
    assert verifier.verify(clip(-1, seconds=0.3)) == (True, None)


def test_voiceprint_persists_and_can_be_cleared(tmp_path):
    path = tmp_path / "vp.npz"
    a = SpeakerVerifier(path, extractor=_FakeExtractor())
    a.enroll([clip(+1, seed=i) for i in range(4)])
    b = SpeakerVerifier(path, extractor=_FakeExtractor())
    assert b.enrolled and b.threshold == pytest.approx(a.threshold)
    b.clear()
    assert not SpeakerVerifier(path, extractor=_FakeExtractor()).enrolled


def test_adapt_only_on_confident_match(verifier):
    verifier.enroll([clip(+1, seed=i) for i in range(4)])
    before = verifier.centroid.copy()
    verifier.adapt(clip(+1, seed=7), score=verifier.threshold)  # not confident
    assert np.allclose(before, verifier.centroid)
    verifier.adapt(clip(+1, seed=7), score=verifier.threshold + 0.3)
    assert not np.allclose(before, verifier.centroid)
