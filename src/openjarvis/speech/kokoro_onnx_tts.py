"""Kokoro-82M on onnxruntime: Jarvis's local voice, Spanish and English.

Same model as the ``kokoro`` backend, without PyTorch: ``kokoro-onnx`` runs
on onnxruntime (already installed for Whisper) and bundles espeak-ng for
phonemes, so Spanish works on Windows with nothing else to install. The
fp16 model (about 177 MB) plus voices (about 28 MB) download once into
``~/.openjarvis/models/kokoro`` (``jarvis voice install``).

The voice follows the language of the text: a male Spanish voice
(``em_alex``) for Spanish, a British butler (``bm_george``) for English,
unless the user configured a voice of that language.
"""

from __future__ import annotations

import io
import re
import threading
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend, TTSResult

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)
# fp16: clean audio and about 4x faster than real time on a laptop CPU. The
# int8 file is smaller but produced NaN/overflowing samples (noise) and ran
# slower in testing, so it is not used.
MODEL_FILE = "kokoro-v1.0.fp16.onnx"
VOICES_FILE = "voices-v1.0.bin"
SAMPLE_RATE = 24000

DEFAULT_VOICES = {"es": "em_alex", "en": "bm_george"}
_LANG_FOR_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "i": "it",
    "p": "pt-br",
    "h": "hi",
    "j": "ja",
    "z": "cmn",
}
_SPANISH = re.compile(
    r"[áéíóúñ¿¡]|\b(el|la|los|las|que|de|es|una?|por|para|con|hola|gracias|"
    r"tienes|hoy|mañana|está|son)\b",
    re.IGNORECASE,
)
# Sentence ends (keeping the punctuation) and very long clauses.
_SENTENCES = re.compile(r"(?<=[.!?¡¿…;:])\s+|\n+")
_MAX_CHARS = 350


def model_dir() -> Path:
    return DEFAULT_CONFIG_DIR / "models" / "kokoro"


def model_files(folder: Optional[Path] = None) -> tuple[Path, Path]:
    folder = folder or model_dir()
    return folder / MODEL_FILE, folder / VOICES_FILE


def installed(folder: Optional[Path] = None) -> bool:
    model, voices = model_files(folder)
    return model.is_file() and voices.is_file() and model.stat().st_size > 1_000_000


def download(
    progress: Optional[Callable[[str, int, int], None]] = None,
    folder: Optional[Path] = None,
) -> Path:
    """Fetch the model files once; *progress(name, done, total)* is optional."""
    import httpx

    folder = folder or model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name in (MODEL_FILE, VOICES_FILE):
        target = folder / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with httpx.stream(
            "GET", MODEL_URL + name, follow_redirects=True, timeout=60.0
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            done = 0
            with open(partial, "wb") as fh:
                for chunk in response.iter_bytes(1 << 16):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(name, done, total)
        partial.replace(target)
    return folder


def language_of(text: str) -> str:
    return "es" if _SPANISH.search(text) else "en"


def split_sentences(text: str) -> List[str]:
    """Sentences short enough for one synthesis pass."""
    pieces: List[str] = []
    for sentence in _SENTENCES.split(text):
        sentence = sentence.strip()
        while len(sentence) > _MAX_CHARS:
            cut = sentence.rfind(",", 0, _MAX_CHARS)
            cut = cut if cut > 40 else sentence.rfind(" ", 0, _MAX_CHARS)
            cut = cut if cut > 0 else _MAX_CHARS
            pieces.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    samples = np.nan_to_num(
        np.asarray(samples, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak  # never clip into distortion
    pcm = (samples * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


@TTSRegistry.register("kokoro-onnx")
class KokoroOnnxTTSBackend(TTSBackend):
    """Kokoro-82M through kokoro-onnx; downloads its model on first install."""

    backend_id = "kokoro-onnx"

    def __init__(self, *, model_folder: str = "") -> None:
        self._folder = Path(model_folder) if model_folder else model_dir()
        self._engine = None
        self._lock = threading.Lock()

    def _ensure_engine(self):
        if self._engine is None:
            from kokoro_onnx import Kokoro

            if not installed(self._folder):
                raise RuntimeError(
                    "Kokoro voice model not downloaded. Run: jarvis voice install"
                )
            model, voices = model_files(self._folder)
            self._engine = Kokoro(str(model), str(voices))
        return self._engine

    def health(self) -> bool:
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            return False
        return installed(self._folder)

    def available_voices(self) -> List[str]:
        try:
            return sorted(self._ensure_engine().get_voices())
        except Exception:  # noqa: BLE001
            return []

    def _voice_and_lang(self, text: str, voice_id: str) -> tuple[str, str]:
        lang = language_of(text)
        if voice_id and voice_id[:1] in _LANG_FOR_PREFIX:
            voice_lang = _LANG_FOR_PREFIX[voice_id[:1]]
            # A configured voice is used for text in its own language.
            if voice_lang.startswith(lang):
                return voice_id, voice_lang
        voice = DEFAULT_VOICES[lang]
        return voice, _LANG_FOR_PREFIX[voice[0]]

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str = "",
        speed: float = 1.0,
        output_format: str = "wav",
    ) -> TTSResult:
        from openjarvis.speech.system_tts import speakable

        clean = speakable(text)
        voice, lang = self._voice_and_lang(clean, voice_id)
        engine = self._ensure_engine()
        pause = np.zeros(int(SAMPLE_RATE * 0.22), dtype=np.float32)
        parts: List[np.ndarray] = []
        with self._lock:
            # One sentence at a time: kokoro-onnx's own silence trimming
            # empties multi-sentence input, and short passes keep each one
            # within the model's phoneme limit.
            for sentence in split_sentences(clean):
                samples, _ = engine.create(
                    sentence, voice=voice, speed=float(speed or 1.0), lang=lang
                )
                samples = np.asarray(samples, dtype=np.float32)
                if len(samples) == 0 or not np.isfinite(samples).all():
                    samples, _ = engine.create(
                        sentence,
                        voice=voice,
                        speed=float(speed or 1.0),
                        lang=lang,
                        trim=False,
                    )
                if len(samples):
                    parts.extend([np.asarray(samples, dtype=np.float32), pause])
        audio = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
        return TTSResult(
            audio=to_wav(audio),
            format="wav",
            duration_seconds=len(audio) / SAMPLE_RATE,
            voice_id=voice,
            sample_rate=SAMPLE_RATE,
            metadata={"lang": lang},
        )
