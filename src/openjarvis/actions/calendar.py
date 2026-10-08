"""Read and add Google Calendar events for voice commands and tools.

Reuses the Google connection (``jarvis connect gdrive`` covers Calendar),
whose scope already allows writing events.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

import httpx

API = "https://www.googleapis.com/calendar/v3"


class CalendarError(RuntimeError):
    """User-facing calendar problem."""


def _credentials_path() -> str:
    from openjarvis.connectors.gcalendar import _DEFAULT_CREDENTIALS_PATH
    from openjarvis.connectors.oauth import load_tokens, resolve_google_credentials

    path = resolve_google_credentials(_DEFAULT_CREDENTIALS_PATH)
    if not load_tokens(path):
        raise CalendarError(
            "Google Calendar is not connected. Run: jarvis connect gdrive"
        )
    return path


def _api_events(token: str, time_min: str, time_max: str) -> Dict[str, Any]:
    resp = httpx.get(
        f"{API}/calendars/primary/events",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "singleEvents": "true",
            "orderBy": "startTime",
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": 20,
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def _api_insert(token: str, event: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(
        f"{API}/calendars/primary/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event,
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def _local(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.astimezone()


def events_between(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    from openjarvis.connectors.google_auth import call_with_refresh

    try:
        data = call_with_refresh(
            _api_events,
            _credentials_path(),
            _local(start).isoformat(),
            _local(end).isoformat(),
        )
    except httpx.HTTPError as exc:
        raise CalendarError(f"Google Calendar did not answer: {exc}") from exc
    return list(data.get("items") or [])


def events_on(day: date) -> List[Dict[str, Any]]:
    start = datetime.combine(day, time.min)
    return events_between(start, start + timedelta(days=1))


def next_event(now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    now = now or datetime.now()
    events = events_between(now, now + timedelta(days=14))
    return events[0] if events else None


def create_event(
    title: str,
    start: datetime,
    end: Optional[datetime] = None,
    *,
    all_day: bool = False,
) -> Dict[str, Any]:
    from openjarvis.connectors.google_auth import call_with_refresh

    title = title.strip() or "Event"
    if all_day:
        body = {
            "summary": title,
            "start": {"date": start.date().isoformat()},
            "end": {"date": (start.date() + timedelta(days=1)).isoformat()},
        }
    else:
        end = end or start + timedelta(hours=1)
        body = {
            "summary": title,
            "start": {"dateTime": _local(start).isoformat()},
            "end": {"dateTime": _local(end).isoformat()},
        }
    try:
        return call_with_refresh(_api_insert, _credentials_path(), body)
    except httpx.HTTPError as exc:
        raise CalendarError(f"Google Calendar did not save the event: {exc}") from exc


def event_time(event: Dict[str, Any]) -> Optional[datetime]:
    start = event.get("start") or {}
    raw = start.get("dateTime")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
        except ValueError:
            return None
    return None


def describe(events: List[Dict[str, Any]], spanish: bool) -> str:
    """One spoken sentence per event: time and title."""
    if not events:
        return "No tienes eventos." if spanish else "You have no events."
    parts = []
    for event in events[:6]:
        title = event.get("summary") or ("evento sin título" if spanish else "untitled")
        when = event_time(event)
        if when is None:
            parts.append(f"{title} ({'todo el día' if spanish else 'all day'})")
        else:
            parts.append(f"{when.strftime('%H:%M')} {title}")
    extra = len(events) - 6
    head = (
        f"Tienes {len(events)} evento{'s' if len(events) != 1 else ''}: "
        if spanish
        else f"You have {len(events)} event{'s' if len(events) != 1 else ''}: "
    )
    tail = ""
    if extra > 0:
        tail = f", y {extra} más" if spanish else f", and {extra} more"
    return head + "; ".join(parts) + tail + "."
