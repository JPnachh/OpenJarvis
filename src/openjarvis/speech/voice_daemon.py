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
from collections import deque
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
    """Live microphone via sounddevice (16 kHz mono int16, 80 ms blocks).

    Raises ``RuntimeError`` if no audio arrives for ``stall_s`` seconds (sleep /
    unplugged device), so a supervisor can restart the stream.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        gain: float = 1.0,
        stall_s: float = 15.0,
    ) -> None:
        self._device = device
        self._gain = gain
        self._stall_s = stall_s
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
                data = indata[:, 0].copy()
                if self._gain != 1.0:
                    data = np.clip(
                        data.astype(np.float32) * self._gain, -32768, 32767
                    ).astype(np.int16)
                self._q.put_nowait(data)
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
                try:
                    yield self._q.get(timeout=self._stall_s)
                except queue.Empty:
                    raise RuntimeError(
                        f"el micrófono no entregó audio en {self._stall_s:.0f} s"
                    ) from None


# ------------------------------------------------------------------ recording


def _rms(frame: np.ndarray) -> float:
    x = frame.astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def wav_to_pcm(wav: bytes) -> np.ndarray:
    """Decode 16-bit mono WAV bytes (as produced here) into int16 samples."""
    import io
    import wave

    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


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

PRE_FRAMES = 30  # ~2.4 s of audio kept from before the trigger fired


@dataclass
class VoiceDaemon:
    """Wires detectors, STT, the Jarvis server and TTS into one loop.

    ``beep(kind)`` kinds: ``ready`` (listening), ``wake`` (heard you), ``end``
    (conversation over, back to sleep), ``denied`` (voice not recognised).
    ``verifier(audio) -> (accepted, score)`` checks the speaker once per turn;
    ``on_verified(audio, score)`` lets the voiceprint adapt. ``paused()`` mutes
    the listener (privacy switch / enrolment).
    """

    transcribe: Callable[[bytes], str]
    ask: Callable[[str], Optional[str]]
    say: Callable[[str], None]
    beep: Callable[[str], None] = lambda kind="wake": None
    log: Callable[[str], None] = print
    wake: Any = None  # WakeWordDetector or None
    clap: Any = None  # ClapDetector or None
    verifier: Optional[Callable[[np.ndarray], tuple]] = None
    on_verified: Optional[Callable[[np.ndarray, float], None]] = None
    paused: Callable[[], bool] = lambda: False
    follow_up_s: float = 8.0
    startup_s: float = 6.0
    max_turns: Optional[int] = None  # stop after N completed turns (testing)
    turns_done: int = field(default=0, init=False)
    status: Optional[Callable[[float, float], None]] = None  # (level, wake score)
    _floor: float = field(default=100.0, init=False)

    def run(self, source: Iterable[np.ndarray]) -> None:
        frames = iter(source)
        self.log(
            "Escuchando… di «Hey Jarvis»"
            + (" o aplaude dos veces." if self.clap else ".")
        )
        self.beep("ready")
        pre: "deque[np.ndarray]" = deque(maxlen=PRE_FRAMES)
        peak, best, n = 0.0, 0.0, 0
        was_paused = False
        for frame in frames:
            n += 1
            if n % 6 == 0:
                now_paused = bool(self.paused())
                if now_paused != was_paused:
                    self.log("Escucha en pausa." if now_paused else "Escuchando…")
                    if not now_paused:
                        self._reset_detectors()
                    was_paused = now_paused
            if was_paused:
                continue
            rms = _rms(frame)
            if rms < 2000:
                self._floor = 0.98 * self._floor + 0.02 * rms
            pre.append(frame)
            trigger = None
            if self.clap is not None and self.clap.process(frame):
                trigger = "aplausos"
            elif self.wake is not None and self.wake.process(frame):
                trigger = "Hey Jarvis"
            if self.wake is not None:
                best = max(best, getattr(self.wake, "last_score", 0.0))
            if self.status is not None:
                peak = max(peak, rms)
                if n % 6 == 0:  # about every 0.5 s
                    self.status(peak, best)
                    peak, best = 0.0, 0.0
            if trigger is None:
                continue
            self.log(f"[{trigger}] te escucho")
            self._turn(frames, source, trigger, list(pre))
            pre.clear()
            self._reset_detectors()
            if self.max_turns is not None and self.turns_done >= self.max_turns:
                return
            self.log("Escuchando…")

    def _reset_detectors(self) -> None:
        for det in (self.wake, self.clap):
            if det is not None:
                det.reset()

    def _check_speaker(self, wav: bytes, trigger: str, pre: list) -> bool:
        if self.verifier is None:
            return True
        audio = wav_to_pcm(wav)
        if trigger == "Hey Jarvis" and pre:  # include the wake phrase itself
            audio = np.concatenate([*pre, audio])
        accepted, score = self.verifier(audio)
        if not accepted:
            self.log(f"Voz no reconocida (similitud {score:.2f}): ignorado.")
            self.beep("denied")
            return False
        if score is not None:
            self.log(f"Voz reconocida (similitud {score:.2f}).")
            if self.on_verified is not None:
                self.on_verified(audio, score)
        return True

    def _turn(
        self,
        frames: Iterator[np.ndarray],
        source: Any,
        trigger: str = "Hey Jarvis",
        pre: Optional[list] = None,
    ) -> None:
        self.beep("wake")
        source.flush()
        startup = self.startup_s
        first = True
        while True:
            wav = record_utterance(frames, floor=self._floor, startup_s=startup)
            if wav is None:
                break
            if first and not self._check_speaker(wav, trigger, pre or []):
                return
            first = False
            text = strip_wake_phrase(self.transcribe(wav))
            if not text or is_cancel(text):
                break
            self.log(f"Tú: {text}")
            reply = self.ask(text)
            if reply:
                self.say(reply)
            source.flush()
            self.turns_done += 1
            if self.max_turns is not None and self.turns_done >= self.max_turns:
                return
            startup = self.follow_up_s  # keep the conversation open briefly
        self.beep("end")
