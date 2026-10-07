"""Builds and supervises the hands-free voice assistant.

Used both by ``jarvis voice`` (foreground) and by the API server, which starts
the same loop in a background thread when ``[voice_assistant] autostart`` is on,
so voice works whenever Jarvis is running — including right after Windows login.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("openjarvis.voice")

LOCK_PORT = 48765  # single-instance guard (loopback only)

ENROLL_PHRASES = [
    "Hey Jarvis, buenos días, ¿cómo amaneció el día?",
    "Jarvis, dime qué hora es y cómo estará el clima.",
    "Quiero que me ayudes a organizar mis proyectos de esta semana.",
    "Cinco pingüinos caminan despacio por la playa del sur.",
    "Jarvis, busca información sobre inteligencia artificial.",
    "Gracias Jarvis, eso es todo por ahora.",
]


def _home() -> Path:
    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR)


# ----------------------------------------------------------- small state files


def pause_file() -> Path:
    return _home() / "voice.pause"


def is_paused() -> bool:
    return pause_file().exists()


def set_paused(paused: bool) -> None:
    f = pause_file()
    if paused:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("paused")
    else:
        try:
            f.unlink()
        except OSError:
            pass


def device_file() -> Path:
    return _home() / "mic_device.txt"


def saved_device() -> Optional[int]:
    """Microphone remembered by ``jarvis voice --find-mic`` (None if unset)."""
    try:
        return int(device_file().read_text().strip())
    except (OSError, ValueError):
        return None


def acquire_single_instance(port: int = LOCK_PORT) -> Optional[socket.socket]:
    """Bind a loopback port as a lock; None if another listener already runs."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: no shared binding
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        return sock
    except OSError:
        sock.close()
        return None


# ------------------------------------------------------------------ chimes


def make_beeper(enabled: bool = True) -> Callable[[str], None]:
    tones = {
        "ready": [(660, 0.10), (880, 0.14)],  # listening
        "wake": [(988, 0.16)],  # heard you, speak now
        "got": [(1320, 0.05)],  # finished recording, thinking
        "end": [(784, 0.12), (587, 0.12), (392, 0.20)],  # back to sleep
        "denied": [(196, 0.35)],  # voice not recognised
    }

    def beep(kind: str = "wake") -> None:
        if not enabled:
            return
        try:
            import sounddevice as sd

            sr = 24000
            parts = []
            for freq, dur in tones.get(kind, tones["wake"]):
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                parts.append(
                    0.40 * np.sin(2 * np.pi * freq * t) * np.linspace(1, 0.3, t.size)
                )
            sd.play(np.concatenate(parts).astype("float32"), sr)
            sd.wait()
        except Exception:
            logger.debug("beep failed", exc_info=True)

    return beep


SPANISH_DAYS = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
]
SPANISH_MONTHS = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]  # fmt: skip


def spanish_now(now: Optional[Any] = None) -> str:
    """Current local date and time in Spanish (weekday, day, month, year, HH:MM)."""
    import datetime as _dt

    now = now or _dt.datetime.now()
    return (
        f"{SPANISH_DAYS[now.weekday()]} {now.day} de {SPANISH_MONTHS[now.month - 1]} "
        f"de {now.year}, {now:%H:%M}"
    )


def voice_system_prompt(
    base: str, location: str = "", now: Optional[Any] = None
) -> str:
    """Voice-mode prompt plus the real clock (the model has no clock of its own)."""
    parts = [base, f"Fecha y hora actuales del usuario: {spanish_now(now)}."]
    if location:
        parts.append(
            f"El usuario está en {location}; úsalo para clima y búsquedas locales."
        )
    return " ".join(parts)


def models_to_try(model: str, cloud_model: str, cloud_ok: bool) -> list[str]:
    """The requested model first, then Claude as a fallback if it is available."""
    chain = [model]
    if cloud_ok and cloud_model and cloud_model != model:
        chain.append(cloud_model)
    return chain


class HttpEmitter:
    """Posts voice events to the Jarvis server without blocking the audio loop."""

    def __init__(self, server_url: str) -> None:
        import queue as _queue

        self._url = f"{server_url}/v1/voice/events"
        self._q: "_queue.Queue[dict]" = _queue.Queue(maxsize=100)
        self._thread = threading.Thread(
            target=self._run, name="voice-events", daemon=True
        )
        self._thread.start()

    def __call__(self, kind: str, **data: Any) -> None:
        try:
            self._q.put_nowait({"type": kind, **data})
        except Exception:
            pass  # a full queue only means the UI misses an update

    def _run(self) -> None:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            while True:
                event = self._q.get()
                try:
                    client.post(self._url, json=event)
                except Exception:
                    logger.debug("voice event not delivered", exc_info=True)


# ---------------------------------------------------------------- options


@dataclass
class VoiceOptions:
    server_url: str = "http://localhost:8000"
    model: str = "jarvis-auto"
    device: Optional[int] = None
    gain: float = 1.0
    clap: bool = True
    wake: bool = True
    wake_threshold: float = 0.5
    follow_up: float = 8.0
    speaker_verify: bool = True
    speaker_adapt: bool = True
    speaker_threshold: float = 0.0
    idle_reset_min: float = 10.0
    chime: bool = True
    location: str = ""
    play: bool = True
    input_wav: Optional[str] = None
    save_dir: Optional[str] = None
    debug_status: Optional[Callable[[float, float], None]] = None
    max_turns: Optional[int] = None
    emit: Optional[Callable[..., None]] = None


def options_from_config(config: Any, server_url: Optional[str] = None) -> VoiceOptions:
    va = config.voice_assistant
    return VoiceOptions(
        server_url=server_url or f"http://127.0.0.1:{config.server.port}",
        model=va.model,
        device=va.device if va.device >= 0 else saved_device(),
        gain=va.gain,
        clap=va.clap,
        wake=va.wake,
        wake_threshold=va.wake_threshold,
        follow_up=va.follow_up,
        speaker_verify=va.speaker_verify,
        speaker_adapt=va.speaker_adapt,
        speaker_threshold=va.speaker_threshold,
        idle_reset_min=va.idle_reset_min,
        chime=va.chime,
        location=va.location,
    )


def background_status(log: Callable[[str], None], threshold: float):
    """Status callback for the background listener: logs what it hears.

    Logs a line when it hears speech-level sound (so a missed "Hey Jarvis" shows
    its confidence) and a one-minute heartbeat with the loudest level seen, which
    reveals a silent / wrong microphone.
    """
    state = {"last": 0.0, "beat": time.monotonic(), "peak": 0.0, "best": 0.0}

    def status(level: float, wake: float) -> None:
        now = time.monotonic()
        state["peak"] = max(state["peak"], level)
        state["best"] = max(state["best"], wake)
        if level > 800 and now - state["last"] > 2.0:
            state["last"] = now
            log(f"Oí sonido (nivel {int(level)}); «Hey Jarvis» {wake:.2f}/{threshold}")
        if now - state["beat"] >= 60:
            log(
                f"[vivo] último minuto: nivel máx {int(state['peak'])}, "
                f"«Hey Jarvis» máx {state['best']:.2f}/{threshold}"
            )
            state.update(beat=now, peak=0.0, best=0.0)

    return status


# ------------------------------------------------------------------ builder


def build_daemon(config: Any, opts: VoiceOptions, log: Callable[[str], None]):
    """Create the ``(VoiceDaemon, audio_source)`` pair for *opts*."""
    import httpx

    from openjarvis.cli._voice_chat import VoiceSession
    from openjarvis.learning.routing.hybrid_router import cloud_available
    from openjarvis.speech.voice_daemon import (
        VOICE_SYSTEM_PROMPT,
        MicSource,
        VoiceDaemon,
        WavSource,
        strip_markdown,
    )

    emit = opts.emit or HttpEmitter(opts.server_url)
    session = VoiceSession(config)
    stt = session.get_stt_backend()
    if stt is None:
        raise RuntimeError("No hay reconocimiento de voz (extra 'speech').")
    language = getattr(config.speech, "language", "") or None

    history: list[dict] = []
    last_activity = {"t": time.monotonic()}
    saved = {"n": 0}

    def transcribe(wav: bytes) -> str:
        return stt.transcribe(wav, format="wav", language=language).text.strip()

    def ask(text: str) -> Optional[str]:
        now = time.monotonic()
        if (
            opts.idle_reset_min > 0
            and now - last_activity["t"] > opts.idle_reset_min * 60
        ):
            history.clear()  # a new conversation after a long pause
        last_activity["t"] = now
        prompt = voice_system_prompt(VOICE_SYSTEM_PROMPT, opts.location)
        messages = [{"role": "system", "content": prompt}]
        messages += history[-8:] + [{"role": "user", "content": text}]
        data = None
        reply = ""
        chain = models_to_try(
            opts.model, config.hybrid_routing.cloud_model, cloud_available()
        )
        for i, model_name in enumerate(chain):
            try:
                resp = httpx.post(
                    f"{opts.server_url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False,
                        "max_tokens": 400,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = strip_markdown(data["choices"][0]["message"]["content"] or "")
                break
            except Exception as exc:
                log(f"Error con el modelo {model_name}: {exc}")
                if i + 1 < len(chain):
                    log(f"Pruebo con {chain[i + 1]}…")
        if data is None:
            return "No pude conseguir una respuesta del servidor de Jarvis."
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        used = data.get("model", "?")
        log(f"({used}) Jarvis: {reply}")
        emit("turn", user=text, assistant=reply, model=used)
        return reply

    def say(text: str) -> None:
        emit("state", state="speaking")
        backend = session.get_tts_backend()
        if backend is None:
            return
        voice_id, speed = session.voice_for_backend(backend)
        kwargs: dict = {"output_format": "wav", "speed": speed}
        if voice_id:
            kwargs["voice_id"] = voice_id
        result = backend.synthesize(text[:1500], **kwargs)
        if not result.audio:
            return
        if opts.play:
            from openjarvis.speech.voice_io import play_wav

            play_wav(result.audio, sample_rate=result.sample_rate)
        else:
            out = Path(opts.save_dir or ".")
            out.mkdir(parents=True, exist_ok=True)
            saved["n"] += 1
            (out / f"reply_{saved['n']}.wav").write_bytes(result.audio)

    clap_det = None
    if opts.clap:
        from openjarvis.speech.clap import ClapDetector

        clap_det = ClapDetector()
    wake_det = None
    if opts.wake:
        from openjarvis.speech.wake_word import WakeWordDetector

        wake_det = WakeWordDetector(threshold=opts.wake_threshold)
    if clap_det is None and wake_det is None:
        raise RuntimeError("Activa al menos 'wake' o 'clap'.")

    verifier = None
    on_verified = None
    if opts.speaker_verify:
        try:
            from openjarvis.speech.speaker_id import SpeakerVerifier, voiceprint_path

            sv = SpeakerVerifier()

            def apply_threshold() -> None:
                if sv.enrolled and opts.speaker_threshold > 0:
                    sv.threshold = opts.speaker_threshold

            def stamp() -> float:
                try:
                    return voiceprint_path().stat().st_mtime
                except OSError:
                    return 0.0

            seen = {"mtime": stamp()}
            apply_threshold()
            if sv.enrolled:
                sv._get_extractor()  # load now (downloads the model once)
                log(f"Reconocimiento de voz activo (umbral {sv.threshold:.2f}).")
            else:
                log(
                    "Sin huella de voz: responderé a cualquier voz "
                    "(registra la tuya con 'jarvis voice --enroll')."
                )

            def verifier(audio):  # noqa: F811 - hot-reloads a new voiceprint
                m = stamp()
                if m != seen["mtime"]:
                    seen["mtime"] = m
                    sv.load()
                    apply_threshold()
                    if sv.enrolled:
                        log(f"Huella de voz actualizada (umbral {sv.threshold:.2f}).")
                return sv.verify(audio)

            if opts.speaker_adapt:
                on_verified = sv.adapt
        except Exception as exc:
            verifier = None
            log(
                f"No pude cargar el reconocimiento de voz ({exc}); "
                "sigo SIN verificar quién habla."
            )

    daemon = VoiceDaemon(
        transcribe=transcribe,
        ask=ask,
        say=say,
        beep=make_beeper(opts.chime and opts.play and opts.input_wav is None),
        log=log,
        wake=wake_det,
        clap=clap_det,
        verifier=verifier,
        on_verified=on_verified,
        paused=is_paused,
        emit=emit,
        follow_up_s=opts.follow_up,
        max_turns=opts.max_turns,
        status=opts.debug_status
        or (None if opts.input_wav else background_status(log, opts.wake_threshold)),
    )
    source = (
        WavSource(opts.input_wav)
        if opts.input_wav
        else MicSource(opts.device, opts.gain)
    )
    return daemon, source


# ------------------------------------------------------------- enrolment


def enroll_voice(
    opts: VoiceOptions,
    log: Callable[[str], None],
    phrases: Optional[list[str]] = None,
) -> dict:
    """Record the owner reading *phrases* and save a voiceprint."""
    from openjarvis.speech.speaker_id import SpeakerVerifier
    from openjarvis.speech.voice_daemon import MicSource, record_utterance, wav_to_pcm

    phrases = phrases or ENROLL_PHRASES
    verifier = SpeakerVerifier()
    verifier._get_extractor()  # fail early / download the model first

    set_paused(True)  # keep the background listener from reacting to the phrases
    try:
        time.sleep(1.5)
        source = MicSource(opts.device, opts.gain)
        frames = iter(source)
        samples: list[np.ndarray] = []
        for i, phrase in enumerate(phrases, 1):
            source.flush()
            log(
                f"\n({i}/{len(phrases)}) Lee en voz alta, con tu tono normal:"
                f"\n    «{phrase}»"
            )
            wav = record_utterance(frames, startup_s=12.0, silence_s=1.0)
            if wav is None:
                log("    No te oí; salto esta frase.")
                continue
            samples.append(wav_to_pcm(wav))
            log("    Listo.")
        return verifier.enroll(samples)
    finally:
        set_paused(False)


# ------------------------------------------------------------ supervision


def _wait_for_server(url: str, timeout: float = 180.0) -> bool:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{url}/health", timeout=2).raise_for_status()
            return True
        except Exception:
            time.sleep(2)
    return False


def run_forever(config: Any, server_url: Optional[str] = None) -> None:
    """Blocking supervisor: (re)start the listener until the process exits."""
    lock = acquire_single_instance()
    if lock is None:
        logger.info("Voice assistant already running elsewhere; not starting another.")
        return
    opts = options_from_config(config, server_url)
    if not _wait_for_server(opts.server_url):
        logger.warning(
            "Jarvis server not reachable at %s; voice not started.", opts.server_url
        )
        return
    backoff = 5.0
    while True:
        try:
            daemon, source = build_daemon(config, opts, logger.info)
            backoff = 5.0
            daemon.run(source)
        except Exception:
            logger.exception("Voice listener stopped; restarting in %.0fs", backoff)
            time.sleep(backoff)
            backoff = min(60.0, backoff * 2)


def start_background(config: Any, server_url: Optional[str] = None) -> threading.Thread:
    """Start the supervisor in a daemon thread (called by the API server)."""
    try:
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            _home() / "voice.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    except OSError:
        pass
    thread = threading.Thread(
        target=run_forever, args=(config, server_url), name="jarvis-voice", daemon=True
    )
    thread.start()
    return thread
