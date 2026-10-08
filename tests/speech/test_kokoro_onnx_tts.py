"""Jarvis's local voice: Kokoro-82M through kokoro-onnx."""

from __future__ import annotations

import numpy as np
import pytest

from openjarvis.speech import kokoro_onnx_tts as kk
from openjarvis.speech.features import read_wav


class FakeEngine:
    def __init__(self, broken_first=False):
        self.calls = []
        self.broken_first = broken_first

    def create(self, text, voice, speed=1.0, lang="en-us", trim=True):
        self.calls.append((text, voice, lang, trim))
        if self.broken_first and trim:
            return np.array([np.nan, np.inf], dtype=np.float32), 24000
        return np.full(2400, 0.25, dtype=np.float32), 24000


def _backend(engine, tmp_path):
    backend = kk.KokoroOnnxTTSBackend(model_folder=str(tmp_path))
    backend._engine = engine
    return backend


def test_voice_follows_the_language(tmp_path):
    engine = FakeEngine()
    backend = _backend(engine, tmp_path)
    assert backend.synthesize("Hola, ¿qué tal?").voice_id == "em_alex"
    assert engine.calls[-1][2] == "es"
    assert backend.synthesize("Good evening, sir.").voice_id == "bm_george"
    assert engine.calls[-1][2] == "en-gb"
    # A configured voice applies to text in its own language only.
    assert (
        backend.synthesize("Hello there.", voice_id="af_heart").voice_id == "af_heart"
    )
    assert backend.synthesize("Hola amigo.", voice_id="af_heart").voice_id == "em_alex"


def test_sentences_are_synthesised_one_by_one_without_markdown(tmp_path):
    engine = FakeEngine()
    result = _backend(engine, tmp_path).synthesize(
        "**Hola.** Tienes dos reuniones. ¿Algo más?"
    )
    assert [c[0] for c in engine.calls] == [
        "Hola.",
        "Tienes dos reuniones.",
        "¿Algo más?",
    ]
    audio, rate = read_wav(result.audio)
    assert rate == 24000
    # Three sentences of 0.1 s plus two pauses.
    assert result.duration_seconds == pytest.approx(0.3 + 2 * 0.22, abs=0.01)
    assert audio.size == round(result.duration_seconds * 24000)


def test_broken_samples_are_retried_and_never_reach_the_speaker(tmp_path):
    engine = FakeEngine(broken_first=True)
    result = _backend(engine, tmp_path).synthesize("Hola.")
    assert engine.calls[-1][3] is False  # retried without trimming
    audio, _ = read_wav(result.audio)
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0


def test_to_wav_sanitises_and_normalises():
    audio, _ = read_wav(kk.to_wav(np.array([np.nan, 4.0, -2.0, np.inf], np.float32)))
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() == pytest.approx(1.0, abs=1e-3)


def test_long_sentences_are_split():
    text = "palabra, " * 120 + "fin."
    pieces = kk.split_sentences(text)
    assert len(pieces) > 1
    assert all(len(p) <= kk._MAX_CHARS + 1 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


def test_health_needs_the_downloaded_model(tmp_path):
    pytest.importorskip("kokoro_onnx")
    backend = kk.KokoroOnnxTTSBackend(model_folder=str(tmp_path))
    assert not backend.health()
    (tmp_path / kk.MODEL_FILE).write_bytes(b"0" * 2_000_000)
    (tmp_path / kk.VOICES_FILE).write_bytes(b"0")
    assert backend.health()


def test_download_writes_files_atomically(tmp_path, monkeypatch):
    import httpx

    class FakeStream:
        def __init__(self, url):
            self.url = url
            self.headers = {"content-length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(httpx, "stream", lambda method, url, **kw: FakeStream(url))
    seen = []
    kk.download(lambda name, done, total: seen.append((name, done, total)), tmp_path)
    assert (tmp_path / kk.MODEL_FILE).read_bytes() == b"abcdef"
    assert (tmp_path / kk.VOICES_FILE).read_bytes() == b"abcdef"
    assert not list(tmp_path.glob("*.part"))
    assert seen[-1] == (kk.VOICES_FILE, 6, 6)


@pytest.mark.skipif(not kk.installed(), reason="Kokoro model not downloaded")
def test_real_model_speaks_spanish_cleanly():
    backend = kk.KokoroOnnxTTSBackend()
    result = backend.synthesize("Hola, soy Jarvis. Mañana va a llover.")
    audio, _ = read_wav(result.audio)
    assert 1.5 < result.duration_seconds < 8
    assert np.isfinite(audio).all() and 0.1 < np.abs(audio).max() <= 1.0
