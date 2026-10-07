"""Hands-free voice assistant loop.

idle (listening for "Hey Jarvis" or a double clap)
  -> chime -> record until silence -> transcribe -> ask the Jarvis server
  -> speak the reply -> keep listening for a follow-up for a few seconds.

Audio stays on this machine until the trigger fires. Only the transcribed text
goes to the Jarvis server (which may route it to Claude via ``jarvis-auto``), and
only the reply text goes to the TTS provider.
"""

from __future__ import annotations

import queue
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms

VOICE_SYSTEM_PROMPT = (
    "Estás en modo voz. Responde en español, de forma natural y breve "
    "(máximo 3 frases), sin markdown, sin listas, sin emojis y sin código, "
    "como si hablaras en voz alta."
)

_WAKE_PREFIX = re.compile(
    r"^\s*(?:hey|ey|oye|hola|ok)[\s,]*jarvis[\s,.:;!?\-]*", re.IGNORECASE
)
_CANCEL_WORDS = {"para", "cancela", "cancelar", "olvidalo", "silencio", "nada", "stop"}


def _fold(text: str) -> str:
    d = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in d if not unicodedata.combining(c)).strip(" .,!¡?¿")


def strip_wake_phrase(text: str) -> str:
    return _WAKE_PREFIX.sub("", text, count=1).strip()


def is_cancel(text: str) -> bool:
    return _fold(text) in _CANCEL_WORDS


def strip_markdown(text: str) -> str:
    """Make *text* pleasant to read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- audio sources


class WavSource:
    """Feed a 16 kHz mono WAV file as if it were the microphone (for testing)."""

    def __init__(self, path: str, tail_silence_s: float = 3.0) -> None:
        import soundfile as sf
        from scipy.signal import resample_poly

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            mono = resample_poly(mono, SAMPLE_RATE, sr)
        pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
        tail = np.zeros(int(tail_silence_s * SAMPLE_RATE), dtype=np.int16)
        self._pcm = np.concatenate([pcm, tail])

    def flush(self) -> None:
        pass

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(0, len(self._pcm) - FRAME + 1, FRAME):
            yield self._pcm[i : i + FRAME]


class MicSource:
    """Live microphone via sounddevice (16 kHz mono int16, 80 ms blocks)."""

    def __init__(self, device: Optional[int] = None) -> None:
        self._device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)

    def flush(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def __iter__(self) -> Iterator[np.ndarray]:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            try:
                self._q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME,
            device=self._device,
            callback=callback,
        ):
            while True:
                yield self._q.get()


# ------------------------------------------------------------------ recording


def _rms(frame: np.ndarray) -> float:
    x = frame.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def record_utterance(
    frames: Iterator[np.ndarray],
    *,
    floor: float = 100.0,
    startup_s: float = 6.0,
    silence_s: float = 1.2,
    max_s: float = 20.0,
) -> Optional[bytes]:
    """Collect frames until the speaker stops. Returns WAV bytes, or None.

    Returns None when nobody starts speaking within *startup_s*.
    """
    import io
    import wave

    block_s = FRAME / SAMPLE_RATE
    threshold = max(350.0, floor * 3.0)
    startup_blocks = int(startup_s / block_s)
    silence_blocks = int(silence_s / block_s)
    max_blocks = int(max_s / block_s)

    chunks: list[np.ndarray] = []
    spoke = False
    quiet = 0
    for n, frame in enumerate(frames):
        chunks.append(frame)
        if _rms(frame) > threshold:
            spoke, quiet = True, 0
        elif spoke:
            quiet += 1
            if quiet >= silence_blocks:
                break
        elif n + 1 >= startup_blocks:
            return None
        if n + 1 >= max_blocks:
            break
    if not spoke:
        return None

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.concatenate(chunks).astype(np.int16).tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------- daemon


@dataclass
class VoiceDaemon:
    """Wires detectors, STT, the Jarvis server and TTS into one loop."""

    transcribe: Callable[[bytes], str]
    ask: Callable[[str], Optional[str]]
    say: Callable[[str], None]
    beep: Callable[[], None] = lambda: None
    log: Callable[[str], None] = print
    wake: Any = None  # WakeWordDetector or None
    clap: Any = None  # ClapDetector or None
    follow_up_s: float = 8.0
    startup_s: float = 6.0
    max_turns: Optional[int] = None  # stop after N completed turns (testing)
    turns_done: int = field(default=0, init=False)
    _floor: float = field(default=100.0, init=False)

    def run(self, source: Iterable[np.ndarray]) -> None:
        frames = iter(source)
        self.log(
            "Escuchando… di «Hey Jarvis»"
            + (" o aplaude dos veces." if self.clap else ".")
        )
        for frame in frames:
            rms = _rms(frame)
            if rms < 2000:
                self._floor = 0.98 * self._floor + 0.02 * rms
            trigger = None
            if self.clap is not None and self.clap.process(frame):
                trigger = "aplausos"
            elif self.wake is not None and self.wake.process(frame):
                trigger = "Hey Jarvis"
            if trigger is None:
                continue
            self.log(f"[{trigger}] te escucho")
            self._turn(frames, source)
            for det in (self.wake, self.clap):
                if det is not None:
                    det.reset()
            if self.max_turns is not None and self.turns_done >= self.max_turns:
                return
            self.log("Escuchando…")

    def _turn(self, frames: Iterator[np.ndarray], source: Any) -> None:
        self.beep()
        source.flush()
        startup = self.startup_s
        while True:
            wav = record_utterance(frames, floor=self._floor, startup_s=startup)
            if wav is None:
                return
            text = strip_wake_phrase(self.transcribe(wav))
            if not text or is_cancel(text):
                return
            self.log(f"Tú: {text}")
            reply = self.ask(text)
            if reply:
                self.say(reply)
            source.flush()
            self.turns_done += 1
            if self.max_turns is not None and self.turns_done >= self.max_turns:
                return
            startup = self.follow_up_s  # keep the conversation open briefly
