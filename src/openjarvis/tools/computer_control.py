"""Tools that let the agent act on this computer: music, volume, apps, calendar.

They wrap :mod:`openjarvis.actions`, the same code that runs direct voice
commands, so "pon algo tranquilo" in a conversation and "pausa" spoken to
Jarvis end up doing the same thing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


def _result(name: str, fn) -> ToolResult:
    try:
        return ToolResult(tool_name=name, content=str(fn()), success=True)
    except Exception as exc:  # noqa: BLE001 - report every failure to the model
        return ToolResult(tool_name=name, content=f"Error: {exc}", success=False)


@ToolRegistry.register("media_control")
class MediaControlTool(BaseTool):
    tool_id = "media_control"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="media_control",
            description=(
                "Control music/media playing on this computer (Spotify when "
                "connected, otherwise the system media keys): play, pause, next, "
                "previous, or report what is playing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "next", "previous", "now_playing"],
                    }
                },
                "required": ["action"],
            },
            category="computer",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions.intents import Intent
        from openjarvis.actions.router import run_intent

        action = str(params.get("action", ""))
        if action not in ("play", "pause", "next", "previous", "now_playing"):
            return ToolResult("media_control", f"Unknown action {action}", False)
        return _result(
            "media_control",
            lambda: run_intent(Intent(f"media.{action}", {}, False)).reply,
        )


@ToolRegistry.register("system_volume")
class SystemVolumeTool(BaseTool):
    tool_id = "system_volume"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_volume",
            description=(
                "Change this computer's speaker volume: set an exact level, "
                "raise or lower it by a step, or mute/unmute."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "up", "down", "mute", "unmute"],
                    },
                    "percent": {
                        "type": "integer",
                        "description": "Level 0-100 for 'set'; step size for up/down.",
                    },
                },
                "required": ["action"],
            },
            category="computer",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions import system

        action = params.get("action")
        percent = int(params.get("percent") or 10)

        def run() -> str:
            if action == "set":
                system.set_volume(percent)
                return f"Volume set to {percent}%."
            if action in ("up", "down"):
                system.change_volume(percent if action == "up" else -percent)
                return f"Volume {action} by {percent}%."
            if action in ("mute", "unmute"):
                system.toggle_mute(action == "mute")
                return f"{action.capitalize()}d."
            raise ValueError(f"Unknown action {action}")

        return _result("system_volume", run)


@ToolRegistry.register("open_app")
class OpenAppTool(BaseTool):
    tool_id = "open_app"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_app",
            description=(
                "Open an application (e.g. Spotify, Chrome, calculator, Word) or a "
                "well-known website (YouTube, Gmail, Google Calendar) or any URL "
                "on this computer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "App/site name or URL"}
                },
                "required": ["name"],
            },
            category="computer",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions import system

        name = str(params.get("name", "")).strip()

        def run() -> str:
            if name.startswith(("http://", "https://")):
                system.open_url(name)
                return f"Opened {name}."
            return f"Opened {system.open_app(name)}."

        return _result("open_app", run)


@ToolRegistry.register("spotify_play")
class SpotifyPlayTool(BaseTool):
    tool_id = "spotify_play"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="spotify_play",
            description=(
                "Search Spotify and start playing the best match: a song, an "
                "artist, an album or a playlist (also moods like 'lofi para "
                "estudiar')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["auto", "track", "artist", "album", "playlist"],
                        "default": "auto",
                    },
                },
                "required": ["query"],
            },
            category="computer",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions.intents import Intent
        from openjarvis.actions.router import run_intent

        query = str(params.get("query", "")).strip()
        kind = str(params.get("kind") or "auto")
        return _result(
            "spotify_play",
            lambda: (
                run_intent(
                    Intent("spotify.play", {"query": query, "kind": kind}, False)
                ).reply
            ),
        )


@ToolRegistry.register("calendar_events")
class CalendarEventsTool(BaseTool):
    tool_id = "calendar_events"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calendar_events",
            description=(
                "List the user's Google Calendar events for a day "
                "('today', 'tomorrow' or YYYY-MM-DD)."
            ),
            parameters={
                "type": "object",
                "properties": {"day": {"type": "string", "default": "today"}},
                "required": [],
            },
            category="productivity",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions import calendar

        raw = str(params.get("day") or "today").strip().lower()

        def run() -> str:
            if raw in ("today", "hoy"):
                day = date.today()
            elif raw in ("tomorrow", "manana", "mañana"):
                day = date.today() + timedelta(days=1)
            else:
                day = date.fromisoformat(raw)
            events = calendar.events_on(day)
            lines = []
            for event in events:
                when = calendar.event_time(event)
                lines.append(
                    f"- {when.strftime('%H:%M') if when else 'all day'}: "
                    f"{event.get('summary') or '(untitled)'}"
                )
            return f"{day.isoformat()}:\n" + ("\n".join(lines) or "No events.")

        return _result("calendar_events", run)


@ToolRegistry.register("calendar_add_event")
class CalendarAddEventTool(BaseTool):
    tool_id = "calendar_add_event"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calendar_add_event",
            description=(
                "Add an event to the user's Google Calendar. Use ISO 8601 local "
                "times (e.g. 2026-10-09T17:00). Omit 'end' for one hour; pass "
                "only a date (YYYY-MM-DD) for an all-day event."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["title", "start"],
            },
            category="productivity",
        )

    def execute(self, **params: Any) -> ToolResult:
        from openjarvis.actions import calendar

        title = str(params.get("title", "")).strip()
        start_raw = str(params.get("start", "")).strip()
        end_raw = str(params.get("end") or "").strip()

        def run() -> str:
            all_day = len(start_raw) == 10
            start = datetime.fromisoformat(start_raw)
            end = datetime.fromisoformat(end_raw) if end_raw else None
            calendar.create_event(title, start, end, all_day=all_day)
            return f"Added '{title}' on {start_raw}."

        return _result("calendar_add_event", run)
