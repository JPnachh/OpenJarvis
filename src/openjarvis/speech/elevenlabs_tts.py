"""ElevenLabs text-to-speech backend.

Uses the ElevenLabs REST API. Requires the ELEVENLABS_API_KEY environment
variable (or ``api_key=``). Voice IDs are ElevenLabs voice IDs; see
``available_voices()`` or https://elevenlabs.io/app/voice-library.
"""

from __future__ import annotations

import os
from typing import List

import httpx

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

_ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Premade "George" voice; override with speech.voice_id in config.toml.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

_FORMATS = {
    "mp3": "mp3_44100_128",
    "pcm": "pcm_24000",
    "ulaw": "ulaw_8000",
}


def _elevenlabs_synthesize(
    api_key: str,
    text: str,
    voice_id: str,
    model: str = "eleven_multilingual_v2",
    output_format: str = "mp3",
    speed: float = 1.0,
) -> bytes:
    """Call the ElevenLabs TTS API and return raw audio bytes."""
    resp = httpx.post(
        f"{_ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
        params={"output_format": _FORMATS.get(output_format, output_format)},
        headers={"xi-api-key": api_key},
        json={
            "text": text,
            "model_id": model,
            "voice_settings": {"speed": speed},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.content


@TTSRegistry.register("elevenlabs")
class ElevenLabsTTSBackend(TTSBackend):
    """ElevenLabs TTS backend — cloud synthesis."""

    backend_id = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "eleven_multilingual_v2",
    ) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._model = model

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> TTSResult:
        if not self._api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")

        voice = voice_id or DEFAULT_VOICE_ID
        audio = _elevenlabs_synthesize(
            self._api_key,
            text,
            voice_id=voice,
            model=self._model,
            output_format=output_format,
            speed=speed,
        )
        return TTSResult(
            audio=audio,
            format=output_format,
            voice_id=voice,
            sample_rate=24000 if output_format == "pcm" else 44100,
            metadata={"backend": "elevenlabs", "model": self._model},
        )

    def available_voices(self) -> List[str]:
        if not self._api_key:
            return []
        try:
            resp = httpx.get(
                f"{_ELEVENLABS_API_BASE}/voices",
                headers={"xi-api-key": self._api_key},
                timeout=30.0,
            )
            resp.raise_for_status()
            return [v["voice_id"] for v in resp.json().get("voices", [])]
        except Exception:
            return []

    def health(self) -> bool:
        return bool(self._api_key)
