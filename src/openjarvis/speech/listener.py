"""Background voice listener: claps and "Hey Jarvis" → a spoken conversation.

One microphone stream feeds two triggers, a double clap and the personal
wake word. Either one opens OpenJarvis when it is not open yet, plays a
"listening" chirp, records the request until the user pauses, hands it to
the web UI through the API (``/v1/voice/command``) and finally speaks the
answer aloud. The audio loop never blocks on the network: everything after
the recording runs on a worker thread, and the listener ignores the
microphone while Jarvis talks so it does not hear itself.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from openjarvis.speech.clap import BLOCK_SIZE, SAMPLE_RATE, ClapDetector
from openjarvis.speech.features import read_wav, write_wav

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def resolve_api_key() -> str:
    """Same lookup as ``jarvis serve``: env var, then config.toml."""
    import os

    key = os.environ.get("OPENJARVIS_API_KEY", "")
    if key:
        return key
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return ""
    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    try:
        with open(DEFAULT_CONFIG_DIR / "config.toml", "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, ValueError):
        return ""
    return str(raw.get("server", {}).get("auth", {}).get("api_key", "") or "")


class ApiClient:
    """Tiny stdlib HTTP client for the local OpenJarvis API."""

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float = 5.0,
    ) -> dict:
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + path, data=body, method=method, headers=headers
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
        return json.loads(payload) if payload else {}

    def get(self, path: str, timeout: float = 5.0) -> dict:
        return self._request("GET", path, timeout=timeout)

    def post_json(self, path: str, data: dict, timeout: float = 10.0) -> dict:
        return self._request(
            "POST",
            path,
            body=json.dumps(data).encode(),
            content_type="application/json",
            timeout=timeout,
        )

    def post_wav(self, path: str, wav: bytes, timeout: float = 120.0) -> dict:
        boundary = uuid.uuid4().hex
        body = (
            (
                f"--{boundary}\r\n"
                "Content-Disposition: form-data; "
                'name="file"; filename="command.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + wav
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self._request(
            "POST",
            path,
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
            timeout=timeout,
        )

    def synthesize(self, text: str, timeout: float = 120.0) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/v1/speech/synthesize",
            data=json.dumps({"text": text}).encode(),
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def healthy(self) -> bool:
        try:
            req = urllib.request.Request(self.base_url + "/health")
            with urllib.request.urlopen(req, timeout=1.5):
                return True
        except urllib.error.HTTPError:
            return True  # up, engine maybe down
        except (OSError, ValueError):
            return False


# ---------------------------------------------------------------------------
# Recording the request
# ---------------------------------------------------------------------------


class CommandRecorder:
    """Collect one spoken request: wait for speech, stop after a pause."""

    def __init__(
        self,
        noise_rms: float,
        *,
        start_timeout: float = 6.0,
        silence_seconds: float = 1.2,
        max_seconds: float = 15.0,
    ) -> None:
        self.threshold = max(0.012, noise_rms * 3.5)
        self.start_timeout = start_timeout
        self.silence_seconds = silence_seconds
        self.max_seconds = max_seconds
        self.blocks: list[np.ndarray] = []
        #: Audio of the wake word that started this request, if any.
        self.wake_audio: Optional[np.ndarray] = None
        self.elapsed = 0.0
        self.heard = False
        self._quiet = 0.0

    def feed(self, block: np.ndarray) -> Optional[str]:
        """Add a block; return ``"done"``, ``"no-speech"`` or None to continue."""
        seconds = block.size / SAMPLE_RATE
        self.elapsed += seconds
        self.blocks.append(block)
        rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        if rms >= self.threshold:
            self.heard = True
            self._quiet = 0.0
        else:
            self._quiet += seconds
        if not self.heard:
            if self.elapsed >= self.start_timeout:
                return "no-speech"
            return None
        if self._quiet >= self.silence_seconds or self.elapsed >= self.max_seconds:
            return "done"
        return None

    def wav(self) -> bytes:
        audio = np.concatenate(self.blocks) if self.blocks else np.zeros(1)
        return write_wav(audio)


# ---------------------------------------------------------------------------
# Sounds
# ---------------------------------------------------------------------------


def chirp(kind: str) -> np.ndarray:
    """Short cue tones at 22.05 kHz: ``listen``, ``done`` or ``error``."""
    rate = 22050
    notes = {"listen": (660, 990), "done": (880, 587), "error": (330, 262)}[kind]
    parts = []
    for freq in notes:
        t = np.arange(int(rate * 0.08)) / rate
        envelope = np.minimum(1, t / 0.01) * np.exp(-t * 25)
        parts.append(0.2 * np.sin(2 * np.pi * freq * t) * envelope)
    return np.concatenate(parts).astype(np.float32)


class Speaker:
    """Plays sounds; replaceable in tests."""

    def play(self, samples: np.ndarray, rate: int, wait: bool = False) -> None:
        try:
            import sounddevice as sd

            sd.play(samples, rate)
            if wait:
                sd.wait()
        except Exception:  # noqa: BLE001 - sound is a nicety, never fatal
            logger.debug("Playback failed", exc_info=True)


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


@dataclass
class ListenerOptions:
    claps: bool = True
    wake: bool = True
    converse: bool = True
    speak_replies: bool = True
    sounds: bool = True
    reply_timeout: float = 180.0
    #: Learn from confirmed "Hey Jarvis" hits (keeps the profile improving).
    adapt: bool = True


class VoiceListener:
    """Drive the detectors and the conversation from microphone blocks."""

    def __init__(
        self,
        api: ApiClient,
        *,
        options: ListenerOptions,
        clap_detector: Optional[ClapDetector] = None,
        wake_detector=None,
        open_ui: Callable[[], str] = lambda: "opened",
        speaker: Optional[Speaker] = None,
        log: Callable[[str], None] = print,
        profile_loader: Optional[Callable[[], object]] = None,
        profile_path: Optional[Path] = None,
        learner: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> None:
        self.api = api
        self.options = options
        self.clap_detector = clap_detector
        self.wake_detector = wake_detector
        self.open_ui = open_ui
        self.speaker = speaker or Speaker()
        self.log = log
        self.profile_loader = profile_loader
        self.profile_path = profile_path
        #: Stores a confirmed wake-word hit for training; True if retrained.
        self.learner = learner
        self._profile_mtime = self._mtime()
        self._recorder: Optional[CommandRecorder] = None
        self._busy = threading.Event()  # transcribing / waiting / speaking
        self._noise = 0.003
        self._ui_speaking = False
        self._ui_open = False
        self._last_poll = 0.0
        self._last_profile_check = time.monotonic()
        self._notify: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        notifier = threading.Thread(target=self._notifier, daemon=True)
        notifier.start()
        self._threads.append(notifier)
        # Let open pages know a listener is running.
        self.set_state("idle", "ready")

    # -- state shared with the web UI --------------------------------------

    def _notifier(self) -> None:
        while True:
            state, detail = self._notify.get()
            if state == "__stop__":
                return
            try:
                self.api.post_json(
                    "/v1/voice/state",
                    {"source": "listener", "state": state, "detail": detail},
                    timeout=2.0,
                )
            except (OSError, ValueError):
                pass

    def set_state(self, state: str, detail: Optional[str] = None) -> None:
        self._notify.put((state, detail))

    def _poll_ui(self) -> None:
        def poll() -> None:
            try:
                status = self.api.get("/v1/voice/status", timeout=1.5)
            except (OSError, ValueError):
                self._ui_open = False
                self._ui_speaking = False
                return
            self._ui_open = int(status.get("subscribers") or 0) > 0
            ui = (status.get("states") or {}).get("ui") or {}
            self._ui_speaking = ui.get("state") == "speaking"

        threading.Thread(target=poll, daemon=True).start()

    def _mtime(self) -> float:
        try:
            return self.profile_path.stat().st_mtime if self.profile_path else 0.0
        except OSError:
            return 0.0

    def _maybe_reload_profile(self) -> None:
        if self.profile_loader is None:
            return
        mtime = self._mtime()
        if mtime and mtime != self._profile_mtime:
            self._profile_mtime = mtime
            detector = self.profile_loader()
            if detector is not None:
                self.wake_detector = detector
                self.log("Wake word profile reloaded (retrained).")

    # -- audio loop ----------------------------------------------------------

    def process(self, block: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_poll >= 1.5:
            self._last_poll = now
            self._poll_ui()
        if now - self._last_profile_check >= 5:
            self._last_profile_check = now
            self._maybe_reload_profile()

        rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        if self._recorder is None and rms < self._noise * 2:
            self._noise = 0.98 * self._noise + 0.02 * rms

        if self._recorder is not None:
            outcome = self._recorder.feed(block)
            if outcome is not None:
                recorder, self._recorder = self._recorder, None
                self._finish_recording(recorder, outcome)
            return

        if self._busy.is_set() or self._ui_speaking:
            if self.wake_detector is not None:
                self.wake_detector.pause(1.0)
            return

        peak = float(np.max(np.abs(block))) if block.size else 0.0
        if self.options.claps and self.clap_detector is not None:
            if self.clap_detector.process(peak, rms):
                self._triggered("claps")
                return
        if self.options.wake and self.wake_detector is not None:
            if self.wake_detector.process(block):
                self._triggered("wake word")

    def _triggered(self, what: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        if not self._ui_open:
            outcome = self.open_ui()
            self.log(f"[{stamp}] Heard {what}: {outcome} OpenJarvis.")
        else:
            self.log(f"[{stamp}] Heard {what}.")
        if not self.options.converse:
            if self.options.sounds:
                self.speaker.play(chirp("done"), 22050)
            return
        if self.options.sounds:
            self.speaker.play(chirp("listen"), 22050)
        self.set_state("listening", what)
        self._recorder = CommandRecorder(self._noise)
        if what == "wake word" and self.wake_detector is not None:
            self._recorder.wake_audio = getattr(
                self.wake_detector, "last_match_audio", None
            )
        if self.wake_detector is not None:
            self.wake_detector.reset()

    def _finish_recording(self, recorder: CommandRecorder, outcome: str) -> None:
        if outcome == "no-speech":
            self.log("  ...didn't hear a request; going back to waiting.")
            if self.options.sounds:
                self.speaker.play(chirp("done"), 22050)
            self.set_state("idle", "no-speech")
            return
        if self.options.sounds:
            self.speaker.play(chirp("done"), 22050)
        self.set_state("transcribing")
        self._busy.set()
        worker = threading.Thread(
            target=self._converse,
            args=(recorder.wav(), recorder.wake_audio),
            daemon=True,
        )
        worker.start()
        self._threads.append(worker)

    # -- conversation (worker thread) ---------------------------------------

    def _wait_for_api(self, timeout: float = 90.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.api.healthy():
                return True
            time.sleep(1.0)
        return False

    def _learn(self, wake_audio) -> None:
        if wake_audio is None or not self.options.adapt or self.learner is None:
            return
        try:
            if self.learner(wake_audio):
                self.log("  (learned from this wake word; voice profile updated)")
        except Exception:  # noqa: BLE001 - learning must never break a request
            logger.debug("Wake word learning failed", exc_info=True)

    def _converse(self, wav: bytes, wake_audio=None) -> None:
        try:
            if not self._wait_for_api():
                self.log("  OpenJarvis API is not reachable; request dropped.")
                self._error_cue()
                return
            try:
                result = self.api.post_wav("/v1/voice/command", wav)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = json.loads(exc.read()).get("detail", "")
                except (ValueError, OSError):
                    pass
                self.log(f"  Could not transcribe: {detail or exc}")
                self._error_cue()
                return
            text = str(result.get("text") or "")
            if not text:
                self.log("  ...couldn't make out the words.")
                self._error_cue()
                return
            self.log(f"  You: {text}")
            # A real request followed: the wake word was meant. Learn from it.
            self._learn(wake_audio)
            action = result.get("action")
            if action:
                # Ran directly on this computer; no model, answer at once.
                answer = str(action.get("reply") or "")
                self.log(f"  Jarvis: {answer}")
                if answer and self.options.speak_replies:
                    self._speak(answer)
                return
            if not result.get("delivered"):
                self.log("  (waiting for the OpenJarvis page to open)")
            self.set_state("thinking")
            command_id = result.get("id")
            if not command_id or not self.options.speak_replies:
                return
            # Waiting for the model can take a while; keep hearing triggers
            # meanwhile, and go deaf again only while speaking the answer.
            self._busy.clear()
            try:
                reply = self.api.get(
                    f"/v1/voice/reply/{command_id}?timeout={self.options.reply_timeout}",
                    timeout=self.options.reply_timeout + 10,
                )
            except (OSError, ValueError):
                self.log("  (no reply to speak)")
                return
            answer = str(reply.get("text") or "").strip()
            if not answer:
                return
            self.log(f"  Jarvis: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            self._speak(answer)
        finally:
            self.set_state("idle")
            self._busy.clear()

    def _speak(self, text: str) -> None:
        try:
            audio = self.api.synthesize(text[:4000])
        except (OSError, ValueError):
            self.log("  (voice output is not configured; reply shown in the page)")
            return
        try:
            samples, rate = read_wav(audio)
        except (ValueError, EOFError):
            return
        self._busy.set()
        self.set_state("speaking")
        self.speaker.play(samples, rate, wait=True)

    def _error_cue(self) -> None:
        if self.options.sounds:
            self.speaker.play(chirp("error"), 22050)

    def close(self) -> None:
        self.set_state("offline")
        self._notify.put(("__stop__", None))


def microphone_blocks(
    should_stop: Callable[[], bool], device: Optional[int | str] = None
) -> Iterator[np.ndarray]:
    """Yield float32 blocks from the microphone until *should_stop*."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Listening needs the sounddevice package and PortAudio. "
            "Install with: uv sync --extra desktop"
        ) from exc

    blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=512)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        try:
            blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        )
    except Exception as exc:  # PortAudioError subclasses Exception
        raise RuntimeError(f"Could not open the microphone: {exc}") from exc
    with stream:
        while not should_stop():
            try:
                yield blocks.get(timeout=0.5)
            except queue.Empty:
                continue


def describe_distance(distance: float, threshold: float) -> str:
    if not math.isfinite(distance):
        return "  -  "
    return f"{distance:4.2f}{'*' if distance <= threshold else ' '}"
