"""Personal wake word: training and live detection on synthetic speech."""

from __future__ import annotations

import numpy as np
import pytest

from openjarvis.speech import wakeword as ww
from openjarvis.speech.features import (
    dtw_distance,
    mfcc,
    read_wav,
    resample,
    trim_silence,
    write_wav,
)

from ._synth import HEY_JARVIS, OTHER, OTHER2, SR, room, take


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def trained(tmp_path, rng):
    for _ in range(6):
        ww.add_sample(write_wav(take(HEY_JARVIS, rng)), "positive", root=tmp_path)
    for segments in (OTHER, OTHER2, OTHER + OTHER2):
        ww.add_sample(write_wav(take(segments, rng)), "negative", root=tmp_path)
    return ww.train(root=tmp_path)


def _stream(detector, audio):
    hits = []
    for start in range(0, audio.size - 256, 256):
        if detector.process(audio[start : start + 256]):
            hits.append(start / SR)
    return hits


def test_features_roundtrip_and_distance(rng):
    x = take(HEY_JARVIS, rng)
    decoded, rate = read_wav(write_wav(x))
    assert rate == SR
    assert np.allclose(decoded, x, atol=1e-3)
    a = mfcc(x)
    assert a.shape[1] == 13
    assert dtw_distance(a, a) == 0
    assert dtw_distance(a, mfcc(take(HEY_JARVIS, rng))) < dtw_distance(
        a, mfcc(take(OTHER, rng))
    )


def test_trim_and_resample(rng):
    x = np.concatenate([room(0.5, rng), take(HEY_JARVIS, rng), room(0.5, rng)])
    trimmed = trim_silence(x)
    assert 0.6 * SR < trimmed.size < 1.0 * SR
    assert resample(x, SR, 8000).size == pytest.approx(x.size / 2, abs=2)


def test_training_profile_is_saved_and_separates(trained, tmp_path):
    loaded = ww.WakeWordProfile.load(tmp_path)
    assert loaded is not None
    assert len(loaded.templates) == 6
    assert loaded.threshold == pytest.approx(trained.threshold)
    stats = loaded.stats
    assert stats["positives"] == 6 and stats["negatives"] == 3
    assert stats["nearest_negative"] > stats["own_distance_max"]
    assert stats["own_distance_max"] < loaded.threshold < stats["nearest_negative"]
    # Synthetic phrases share their sounds, so they never rate "good".
    assert stats["quality"] in ("good", "fair", "poor")
    assert stats["nearest_impostor"] <= stats["nearest_negative"]
    info = ww.status(tmp_path)
    assert info["trained"] and info["positives"] == 6
    assert info["profile_path"].endswith("profile.json")


def test_detector_fires_on_the_phrase_only(trained, rng):
    detector = ww.WakeWordDetector(trained)
    audio = np.concatenate(
        [
            room(1.0, rng),
            take(OTHER, rng),
            room(0.5, rng),
            take(HEY_JARVIS, rng),  # ends at ~2.6 s
            take(OTHER2, rng),  # the request right after the wake word
            room(1.0, rng),
            take(OTHER2 + OTHER, rng),
            room(1.0, rng),
        ]
    )
    hits = _stream(detector, audio)
    assert len(hits) == 1
    assert 2.0 < hits[0] < 3.2


def test_detector_ignores_room_noise(trained, rng):
    detector = ww.WakeWordDetector(trained)
    assert _stream(detector, room(5.0, rng, level=0.02)) == []


def test_pause_suppresses_detection(trained, rng):
    detector = ww.WakeWordDetector(trained)
    detector.pause(10.0)
    audio = np.concatenate([room(0.5, rng), take(HEY_JARVIS, rng), room(1.0, rng)])
    assert _stream(detector, audio) == []


def test_training_needs_enough_samples(tmp_path, rng):
    ww.add_sample(write_wav(take(HEY_JARVIS, rng)), "positive", root=tmp_path)
    with pytest.raises(ww.TrainingError, match="at least 3"):
        ww.train(root=tmp_path)


def test_rejects_silent_or_long_samples(tmp_path, rng):
    with pytest.raises(ValueError, match="silent|too short"):
        ww.add_sample(
            write_wav(np.zeros(SR, dtype=np.float32)), "positive", root=tmp_path
        )
    long = np.concatenate([take(OTHER, rng) for _ in range(4)])
    with pytest.raises(ValueError, match="long"):
        ww.add_sample(write_wav(long), "positive", root=tmp_path)


def test_sample_management(tmp_path, rng):
    sample = ww.add_sample(write_wav(take(HEY_JARVIS, rng)), "positive", root=tmp_path)
    assert ww.sample_audio(sample.id, root=tmp_path).startswith(b"RIFF")
    assert ww.sample_audio("../profile", root=tmp_path) is None
    assert ww.delete_sample(sample.id, root=tmp_path)
    assert not ww.delete_sample(sample.id, root=tmp_path)
    assert ww.list_samples(tmp_path) == []
