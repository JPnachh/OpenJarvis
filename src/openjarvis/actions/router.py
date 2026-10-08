"""Run recognised voice commands directly, without the language model.

``handle(text)`` returns an :class:`ActionResult` with a short spoken reply
when the text is a command (built-in or user-taught), or ``None`` so the
caller sends the text to the model as usual.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from openjarvis.actions import system
from openjarvis.actions.custom import CommandBook, CustomCommand
from openjarvis.actions.intents import Intent, parse

logger = logging.getLogger(__name__)

_MONTHS_ES = (
    "enero febrero marzo abril mayo junio julio agosto septiembre octubre "
    "noviembre diciembre"
).split()
_DAYS_ES = "lunes martes miércoles jueves viernes sábado domingo".split()


@dataclass
class ActionResult:
    intent: str
    reply: str
    ok: bool = True


def _say(es: bool, spanish: str, english: str) -> str:
    return spanish if es else english


def _spoken_date(day: date, es: bool) -> str:
    if es:
        return f"{_DAYS_ES[day.weekday()]} {day.day} de {_MONTHS_ES[day.month - 1]}"
    return (
        day.strftime("%A, %B %-d")
        if sys.platform != "win32"
        else day.strftime("%A, %B %d")
    )


def _spoken_when(start: datetime, all_day: bool, es: bool) -> str:
    today = date.today()
    if start.date() == today:
        day = _say(es, "hoy", "today")
    elif start.date() == today + timedelta(days=1):
        day = _say(es, "mañana", "tomorrow")
    else:
        day = _spoken_date(start.date(), es)
    if all_day:
        return day
    return f"{day} {_say(es, 'a las', 'at')} {start.strftime('%H:%M')}"


# ---------------------------------------------------------------------------
# Media with Spotify first, media keys as fallback
# ---------------------------------------------------------------------------


def _spotify():
    from openjarvis.actions.spotify import SpotifyClient

    client = SpotifyClient()
    return client if client.connected() else None


def _media(action: str, es: bool) -> ActionResult:
    from openjarvis.actions.spotify import SpotifyError

    replies = {
        "pause": ("Pausado.", "Paused."),
        "play": ("Reproduciendo.", "Playing."),
        "next": ("Siguiente canción.", "Next song."),
        "previous": ("Canción anterior.", "Previous song."),
    }
    client = _spotify()
    if client is not None:
        try:
            getattr(client, action)()
            return ActionResult(f"media.{action}", _say(es, *replies[action]))
        except SpotifyError as exc:
            logger.info("Spotify control failed (%s); using media keys", exc)
    system.media(action)
    return ActionResult(f"media.{action}", _say(es, *replies[action]))


def _play_search(query: str, kind: str, es: bool) -> ActionResult:
    from openjarvis.actions.spotify import SpotifyError, open_search_in_app

    client = _spotify()
    if client is not None:
        try:
            playing = client.play_search(query, kind)
            return ActionResult(
                "spotify.play", _say(es, f"Poniendo {playing}.", f"Playing {playing}.")
            )
        except SpotifyError as exc:
            if "couldn't find" in str(exc):
                return ActionResult(
                    "spotify.play",
                    _say(es, f"No encontré {query} en Spotify.", str(exc)),
                    ok=False,
                )
            logger.info("Spotify play failed (%s); opening search", exc)
    open_search_in_app(query)
    return ActionResult(
        "spotify.play",
        _say(
            es,
            f"Abrí {query} en Spotify. Conecta Spotify para que lo reproduzca solo.",
            f"Opened {query} in Spotify. Connect Spotify so I can play it directly.",
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_intent(intent: Intent) -> ActionResult:
    es = intent.spanish
    name, args = intent.name, intent.args

    if name == "volume.up":
        system.change_volume(args.get("step", 10))
        return ActionResult(name, _say(es, "Subí el volumen.", "Volume up."))
    if name == "volume.down":
        system.change_volume(-args.get("step", 10))
        return ActionResult(name, _say(es, "Bajé el volumen.", "Volume down."))
    if name == "volume.set":
        percent = args["percent"]
        system.set_volume(percent)
        return ActionResult(
            name,
            _say(es, f"Volumen al {percent} por ciento.", f"Volume at {percent}%."),
        )
    if name == "volume.mute":
        system.toggle_mute(args.get("mute"))
        muted = args.get("mute", True)
        return ActionResult(
            name,
            _say(
                es,
                "Silenciado." if muted else "Sonido activado.",
                "Muted." if muted else "Sound on.",
            ),
        )
    if name.startswith("media.") and name != "media.now_playing":
        return _media(name.split(".", 1)[1], es)
    if name == "media.now_playing":
        client = _spotify()
        playing = client.now_playing() if client is not None else None
        if not playing:
            return ActionResult(
                name,
                _say(
                    es,
                    "No sé qué está sonando; conecta Spotify para saberlo.",
                    "I can't tell what's playing; connect Spotify for that.",
                ),
                ok=False,
            )
        return ActionResult(name, _say(es, f"Suena {playing}.", f"This is {playing}."))
    if name == "spotify.play":
        return _play_search(args["query"], args.get("kind", "auto"), es)
    if name == "time.now":
        now = datetime.now().strftime("%H:%M")
        return ActionResult(name, _say(es, f"Son las {now}.", f"It's {now}."))
    if name == "time.date":
        today = _spoken_date(date.today(), es)
        return ActionResult(name, _say(es, f"Hoy es {today}.", f"Today is {today}."))
    if name == "calendar.day":
        from openjarvis.actions import calendar

        day = date.today() + timedelta(days=args.get("offset", 0))
        reply = calendar.describe(calendar.events_on(day), es)
        if args.get("offset"):
            reply = _say(es, "Mañana: ", "Tomorrow: ") + reply[0].lower() + reply[1:]
        return ActionResult(name, reply)
    if name == "calendar.next":
        from openjarvis.actions import calendar

        event = calendar.next_event()
        if event is None:
            return ActionResult(
                name,
                _say(
                    es,
                    "No tienes nada en las próximas dos semanas.",
                    "Nothing in the next two weeks.",
                ),
            )
        when = calendar.event_time(event)
        title = event.get("summary") or _say(es, "un evento", "an event")
        if when is None:
            return ActionResult(
                name, _say(es, f"Lo próximo es {title}.", f"Next is {title}.")
            )
        spoken = _spoken_when(when.replace(tzinfo=None), False, es)
        return ActionResult(
            name, _say(es, f"{title}, {spoken}.", f"{title}, {spoken}.")
        )
    if name == "calendar.create":
        from openjarvis.actions import calendar

        start = args["start"]
        calendar.create_event(args["title"], start, all_day=args.get("all_day", False))
        spoken = _spoken_when(start, args.get("all_day", False), es)
        return ActionResult(
            name,
            _say(
                es,
                f"Listo, agendé {args['title']} para {spoken}.",
                f"Done, {args['title']} is on your calendar for {spoken}.",
            ),
        )
    if name == "app.open":
        system.open_app(args["name"])
        spoken = args.get("spoken") or args["name"]
        return ActionResult(name, _say(es, f"Abriendo {spoken}.", f"Opening {spoken}."))
    raise ValueError(f"Unknown intent {name}")


def run_custom(command: CustomCommand, allow_shell: bool) -> ActionResult:
    if command.action == "open":
        target = command.target.strip()
        if target.startswith(("http://", "https://")) or ":" in target[:12]:
            system.open_url(target)
        else:
            system.open_app(target)
    elif command.action == "spotify":
        result = _play_search(command.target, "auto", True)
        if command.reply and result.ok:
            return ActionResult("custom", command.reply)
        return result
    elif command.action == "keys":
        key = command.target
        if key == "volume_up":
            system.change_volume(10)
        elif key == "volume_down":
            system.change_volume(-10)
        elif key == "mute":
            system.toggle_mute(None)
        else:
            system.media(key)
    elif command.action == "shell":
        if not allow_shell:
            return ActionResult(
                "custom",
                "Ese comando ejecuta un programa, pero los comandos de shell están "
                "desactivados (allow_shell en commands.json).",
                ok=False,
            )
        subprocess.Popen(command.target, shell=True)  # noqa: S602 - user opted in
    return ActionResult("custom", command.reply or "Hecho.")


def handle(text: str, *, book: Optional[CommandBook] = None) -> Optional[ActionResult]:
    """Run *text* if it is a command; None means "send it to the model"."""
    book = book if book is not None else CommandBook.load()
    custom = book.match(text)
    try:
        if custom is not None:
            return run_custom(custom, book.allow_shell)
        intent = parse(text)
        if intent is None:
            return None
        return run_intent(intent)
    except Exception as exc:  # noqa: BLE001 - always answer the user
        logger.warning("Voice command failed: %s", exc)
        es = intent_is_spanish(text)
        message = str(exc)
        if es:
            if "not connected" in str(exc) and "Calendar" in str(exc):
                message = (
                    "Google Calendar no está conectado. Conéctalo con: "
                    "jarvis connect gdrive"
                )
            elif "not connected" in str(exc) and "Spotify" in str(exc):
                message = (
                    "Spotify no está conectado. Conéctalo con: jarvis connect spotify"
                )
            return ActionResult("error", f"No pude hacerlo. {message}", ok=False)
        return ActionResult("error", f"I couldn't do that: {message}", ok=False)


def intent_is_spanish(text: str) -> bool:
    from openjarvis.actions.intents import _is_spanish, normalize

    return _is_spanish(normalize(text))
