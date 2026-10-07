"""Tests for the ElevenLabs TTS backend (HTTP mocked)."""

from __future__ import annotations

import httpx
import pytest
import respx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.elevenlabs_tts import (
    DEFAULT_VOICE_ID,
    ElevenLabsTTSBackend,
)


def test_registered():
    TTSRegistry.register_value("elevenlabs", ElevenLabsTTSBackend)
    assert TTSRegistry.contains("elevenlabs")


def test_health_requires_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTSBackend().health() is False
    assert ElevenLabsTTSBackend(api_key="k").health() is True


def test_synthesize_without_key_raises(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ElevenLabsTTSBackend().synthesize("hi")


@respx.mock
def test_synthesize_posts_to_voice_endpoint():
    route = respx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}"
    ).mock(return_value=httpx.Response(200, content=b"AUDIO"))
    result = ElevenLabsTTSBackend(api_key="k").synthesize("hola", speed=1.1)
    assert result.audio == b"AUDIO"
    assert result.voice_id == DEFAULT_VOICE_ID
    req = route.calls[0].request
    assert req.headers["xi-api-key"] == "k"
    assert req.url.params["output_format"] == "mp3_44100_128"


@respx.mock
def test_wav_output_is_wrapped_pcm():
    import io
    import wave

    route = respx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}"
    ).mock(return_value=httpx.Response(200, content=b"\x00\x01" * 100))
    result = ElevenLabsTTSBackend(api_key="k").synthesize("hi", output_format="wav")
    assert route.calls[0].request.url.params["output_format"] == "pcm_24000"
    with wave.open(io.BytesIO(result.audio)) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() == 100
    assert result.sample_rate == 24000
