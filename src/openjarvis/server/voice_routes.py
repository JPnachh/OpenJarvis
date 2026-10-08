"""Hands-free voice: live voice events, spoken commands, wake-word training.

The background listener (``jarvis listen``) hears "Hey Jarvis" or a double
clap, records the request and posts it here. This router transcribes it and
pushes it to every open web UI over a server-sent event stream; the UI sends
it to the model like a typed message and posts the reply back, which the
listener then speaks aloud. Both sides also publish their state (listening,
transcribing, speaking) so each can show — and respect — what the other is
doing, e.g. the listener ignores the microphone while Jarvis talks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

voice_router = APIRouter(prefix="/v1/voice", tags=["voice"])
wakeword_router = APIRouter(prefix="/v1/wakeword", tags=["voice"])

# A command spoken while no browser is open waits this long for one.
_PENDING_TTL = 90.0
_KEEPALIVE_SECONDS = 15.0
_STATES = {"idle", "listening", "transcribing", "thinking", "speaking", "offline"}


class VoiceHub:
    """In-process fan-out of voice events to connected web UIs."""

    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.states: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.replies: dict[str, str] = {}
        self._reply_events: dict[str, asyncio.Event] = {}

    def publish(self, event: dict[str, Any]) -> int:
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return len(self.subscribers)

    def set_state(self, source: str, state: str, detail: Optional[str]) -> None:
        self.states[source] = {"state": state, "detail": detail, "at": time.time()}
        self.publish(
            {"type": "state", "source": source, "state": state, "detail": detail}
        )

    def command(self, text: str, source: str) -> dict[str, Any]:
        event = {
            "type": "command",
            "id": uuid.uuid4().hex,
            "text": text,
            "source": source,
            "at": time.time(),
        }
        self._reply_events[event["id"]] = asyncio.Event()
        # Forget the oldest unanswered commands so the map cannot grow forever.
        while len(self._reply_events) > 50:
            stale = next(iter(self._reply_events))
            self._reply_events.pop(stale)
            self.replies.pop(stale, None)
        delivered = self.publish(event)
        if delivered == 0:
            self.pending.append(event)
        return {**event, "delivered": delivered}

    def take_pending(self) -> list[dict[str, Any]]:
        now = time.time()
        fresh = [e for e in self.pending if now - e["at"] <= _PENDING_TTL]
        self.pending = []
        return fresh

    def post_reply(self, command_id: str, text: str) -> bool:
        event = self._reply_events.get(command_id)
        if event is None:
            return False
        self.replies[command_id] = text
        event.set()
        return True

    async def wait_reply(self, command_id: str, timeout: float) -> Optional[str]:
        event = self._reply_events.get(command_id)
        if event is None:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        self._reply_events.pop(command_id, None)
        return self.replies.pop(command_id, None)


def get_hub(request: Request) -> VoiceHub:
    hub = getattr(request.app.state, "voice_hub", None)
    if hub is None:
        hub = VoiceHub()
        request.app.state.voice_hub = hub
    return hub


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@voice_router.get("/events")
async def voice_events(request: Request):
    """Server-sent events: voice state changes and spoken commands."""
    hub = get_hub(request)
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    hub.subscribers.add(queue)

    async def stream():
        try:
            yield _sse({"type": "hello", "states": hub.states})
            for event in hub.take_pending():
                yield _sse(event)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), _KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(event)
        finally:
            hub.subscribers.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@voice_router.get("/status")
async def voice_status(request: Request):
    hub = get_hub(request)
    return {"states": hub.states, "subscribers": len(hub.subscribers)}


@voice_router.post("/state")
async def voice_state(request: Request):
    body = await request.json()
    source = str(body.get("source") or "")
    state = str(body.get("state") or "")
    if source not in ("ui", "listener") or state not in _STATES:
        raise HTTPException(status_code=400, detail="Invalid source or state")
    detail = body.get("detail")
    get_hub(request).set_state(source, state, str(detail) if detail else None)
    return {"ok": True}


@voice_router.post("/command")
async def voice_command(request: Request):
    """Deliver a spoken request to the web UI.

    Accepts multipart audio (``file``), transcribed with the configured speech
    backend, or JSON ``{"text": ...}``.
    """
    hub = get_hub(request)
    content_type = request.headers.get("content-type", "")
    source = "listener"
    if content_type.startswith("application/json"):
        body = await request.json()
        text = str(body.get("text") or "").strip()
        source = str(body.get("source") or source)
    else:
        from openjarvis.server.api_routes import _speech_backend

        backend = await _speech_backend(request)
        if backend is None:
            reason = getattr(request.app.state, "speech_unavailable_reason", None)
            raise HTTPException(
                status_code=501, detail=reason or "Speech backend not configured"
            )
        form = await request.form()
        audio = form.get("file")
        if audio is None:
            raise HTTPException(status_code=400, detail="Missing 'file' field")
        data = await audio.read()
        language = form.get("language") or None
        try:
            result = await asyncio.to_thread(
                backend.transcribe, data, format="wav", language=language
            )
        except Exception as exc:
            logger.exception("Voice command transcription failed")
            raise HTTPException(
                status_code=500, detail=f"Speech transcription failed: {exc}"
            ) from exc
        text = (result.text or "").strip()
    if not text:
        return {"text": "", "delivered": 0, "id": None}

    # Direct commands ("sube el volumen", "pon X en Spotify") run right here,
    # in a fraction of a second, without the model. The page still shows the
    # exchange, and the reply is ready at once for the listener to speak.
    action = await _run_action(text)
    if action is not None:
        command_id = uuid.uuid4().hex
        hub._reply_events[command_id] = asyncio.Event()
        hub.post_reply(command_id, action.reply)
        delivered = hub.publish(
            {
                "type": "action",
                "id": command_id,
                "text": text,
                "reply": action.reply,
                "ok": action.ok,
                "intent": action.intent,
            }
        )
        return {
            "text": text,
            "delivered": delivered,
            "id": command_id,
            "action": {"reply": action.reply, "ok": action.ok, "intent": action.intent},
        }

    event = hub.command(text, source)
    return {"text": text, "delivered": event["delivered"], "id": event["id"]}


async def _run_action(text: str):
    from openjarvis.actions.router import handle

    return await asyncio.to_thread(handle, text)


@voice_router.post("/reply")
async def voice_reply(request: Request):
    """The web UI posts the assistant's answer to a spoken command."""
    body = await request.json()
    ok = get_hub(request).post_reply(
        str(body.get("id") or ""), str(body.get("text") or "")
    )
    return {"ok": ok}


@voice_router.get("/reply/{command_id}")
async def voice_wait_reply(command_id: str, request: Request, timeout: float = 120):
    """Long-poll for the answer to a command (used by the listener to speak it)."""
    text = await get_hub(request).wait_reply(command_id, min(max(timeout, 1), 300))
    if text is None:
        raise HTTPException(status_code=404, detail="No reply yet")
    return {"text": text}


# ---------------------------------------------------------------------------
# Wake word training
# ---------------------------------------------------------------------------


@wakeword_router.get("")
async def wakeword_status():
    from openjarvis.speech import wakeword

    return await asyncio.to_thread(wakeword.status)


@wakeword_router.post("/samples")
async def wakeword_add_sample(request: Request):
    """Store one training recording (16-bit PCM WAV; any rate, mono or stereo)."""
    from openjarvis.speech import wakeword

    form = await request.form()
    kind = str(form.get("kind") or "positive")
    audio = form.get("file")
    if audio is None:
        raise HTTPException(status_code=400, detail="Missing 'file' field")
    data = await audio.read()
    try:
        sample = await asyncio.to_thread(wakeword.add_sample, data, kind)
    except (ValueError, EOFError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": sample.id, "kind": sample.kind, "seconds": round(sample.seconds, 2)}


@wakeword_router.get("/samples/{sample_id}/audio")
async def wakeword_sample_audio(sample_id: str):
    from openjarvis.speech import wakeword

    data = wakeword.sample_audio(sample_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    return Response(content=data, media_type="audio/wav")


@wakeword_router.delete("/samples/{sample_id}")
async def wakeword_delete_sample(sample_id: str):
    from openjarvis.speech import wakeword

    if not wakeword.delete_sample(sample_id):
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"ok": True}


@wakeword_router.post("/train")
async def wakeword_train(request: Request):
    from openjarvis.speech import wakeword

    try:
        body = await request.json()
    except ValueError:
        body = {}
    phrase = str((body or {}).get("phrase") or wakeword.DEFAULT_PHRASE)[:60]
    try:
        profile = await asyncio.to_thread(wakeword.train, phrase)
    except wakeword.TrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "phrase": profile.phrase,
        "threshold": profile.threshold,
        "stats": profile.stats,
    }


# ---------------------------------------------------------------------------
# Direct actions and user-taught commands
# ---------------------------------------------------------------------------

actions_router = APIRouter(prefix="/v1/actions", tags=["voice"])

_EXAMPLES = [
    {
        "group": "Volume",
        "phrases": ["sube el volumen", "baja el volumen", "volumen al 40", "silencio"],
    },
    {
        "group": "Music",
        "phrases": [
            "pausa",
            "dale play",
            "siguiente canción",
            "la anterior",
            "¿qué canción es esta?",
        ],
    },
    {
        "group": "Spotify",
        "phrases": [
            "pon Bad Bunny en Spotify",
            "pon la playlist rock en español",
            "pon música de Shakira",
        ],
    },
    {
        "group": "Apps",
        "phrases": ["abre Spotify", "abre YouTube", "abre la calculadora"],
    },
    {
        "group": "Calendar",
        "phrases": [
            "¿qué tengo hoy?",
            "¿qué tengo mañana?",
            "¿cuál es mi próxima reunión?",
            "agenda dentista el viernes a las 10",
        ],
    },
    {"group": "Time", "phrases": ["¿qué hora es?", "¿qué día es hoy?"]},
]


@actions_router.post("/run")
async def actions_run(request: Request):
    """Run *text* if it is a direct command; ``handled: false`` otherwise."""
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")
    action = await _run_action(text)
    if action is None:
        return {"handled": False}
    return {
        "handled": True,
        "reply": action.reply,
        "ok": action.ok,
        "intent": action.intent,
    }


@actions_router.get("/commands")
async def actions_commands():
    from openjarvis.actions.custom import ACTIONS, KEYS, CommandBook, commands_path

    book = CommandBook.load()
    return {
        "path": str(commands_path()),
        "allow_shell": book.allow_shell,
        "actions": list(ACTIONS),
        "keys": list(KEYS),
        "commands": [
            {
                "id": c.id,
                "phrases": c.phrases,
                "action": c.action,
                "target": c.target,
                "reply": c.reply,
            }
            for c in book.commands
        ],
        "examples": _EXAMPLES,
    }


@actions_router.post("/commands")
async def actions_add_command(request: Request):
    from openjarvis.actions.custom import CommandBook, CustomCommand

    body = await request.json()
    phrases = body.get("phrases")
    if isinstance(phrases, str):
        phrases = [p for p in phrases.split("\n")]
    command = CustomCommand(
        phrases=list(phrases or []),
        action=str(body.get("action") or ""),
        target=str(body.get("target") or ""),
        reply=str(body.get("reply") or ""),
    )
    book = CommandBook.load()
    if command.action == "shell" and not book.allow_shell:
        raise HTTPException(
            status_code=400,
            detail=(
                'Shell commands are off. To allow them, set "allow_shell": true '
                "in commands.json yourself."
            ),
        )
    try:
        book.add(command)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    book.save()
    return {"id": command.id}


@actions_router.delete("/commands/{command_id}")
async def actions_delete_command(command_id: str):
    from openjarvis.actions.custom import CommandBook

    book = CommandBook.load()
    if not book.remove(command_id):
        raise HTTPException(status_code=404, detail="Command not found")
    book.save()
    return {"ok": True}
