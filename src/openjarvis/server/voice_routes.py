"""Live voice-assistant events for the web UI.

The hands-free listener (``openjarvis.speech.voice_runtime``) posts state changes
and finished conversation turns here; browsers subscribe over Server-Sent Events
to show *"listening / heard you / thinking / speaking"* and to mirror spoken
exchanges into the chat history.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, Set

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

STATES = {
    "offline",
    "listening",
    "heard",
    "recording",
    "follow_up",
    "thinking",
    "speaking",
    "denied",
    "paused",
}
_MAX_TEXT = 8000


class VoiceHub:
    """In-memory fan-out of voice events with a small replay buffer."""

    def __init__(self) -> None:
        self.session = secrets.token_hex(4)  # changes on server restart
        self._seq = 0
        self.state: Dict[str, Any] = {"state": "offline", "ts": time.time()}
        self.turns: Deque[Dict[str, Any]] = deque(maxlen=30)
        self._subs: Set["asyncio.Queue[Dict[str, Any]]"] = set()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "type": "snapshot",
            "session": self.session,
            "state": self.state,
            "turns": list(self.turns),
        }

    def publish(self, kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self._seq += 1
        event: Dict[str, Any] = {
            "type": kind,
            "id": f"{self.session}-{self._seq}",
            "ts": time.time(),
            **data,
        }
        if kind == "state":
            self.state = {k: event[k] for k in event if k != "type"}
        elif kind == "turn":
            self.turns.append(event)
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._subs.discard(q)  # slow client: it will reconnect
        return event

    def subscribe(self) -> "asyncio.Queue[Dict[str, Any]]":
        q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=200)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Dict[str, Any]]") -> None:
        self._subs.discard(q)


def get_hub(request: Request) -> VoiceHub:
    hub = getattr(request.app.state, "voice_hub", None)
    if hub is None:
        hub = VoiceHub()
        request.app.state.voice_hub = hub
    return hub


router = APIRouter(prefix="/v1/voice", tags=["voice"])


@router.post("/events")
async def post_event(request: Request) -> Dict[str, Any]:
    """Called by the voice listener. Accepts ``state`` and ``turn`` events."""
    body = await request.json()
    kind = str(body.get("type", ""))
    hub = get_hub(request)
    if kind == "state":
        state = str(body.get("state", ""))
        if state not in STATES:
            raise HTTPException(status_code=400, detail=f"Unknown state {state!r}")
        data: Dict[str, Any] = {"state": state}
        if "trigger" in body:
            data["trigger"] = str(body["trigger"])[:40]
    elif kind == "turn":
        data = {
            "user": str(body.get("user", ""))[:_MAX_TEXT],
            "assistant": str(body.get("assistant", ""))[:_MAX_TEXT],
            "model": str(body.get("model", ""))[:120],
        }
        if not data["user"] and not data["assistant"]:
            raise HTTPException(status_code=400, detail="Empty turn")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown event type {kind!r}")
    return {"ok": True, "id": hub.publish(kind, data)["id"]}


@router.get("/state")
async def get_state(request: Request) -> Dict[str, Any]:
    return get_hub(request).snapshot()


@router.get("/events")
async def stream_events(request: Request) -> StreamingResponse:
    hub = get_hub(request)
    queue = hub.subscribe()

    async def gen() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps(hub.snapshot())}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
