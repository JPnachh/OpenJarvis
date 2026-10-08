"""Understand short spoken commands (Spanish and English) without a model.

``parse()`` turns "sube el volumen", "pon Bad Bunny en Spotify", "abre
YouTube" or "agenda dentista mañana a las 5" into an :class:`Intent`. It is
deliberately conservative: anything it does not clearly recognise returns
``None`` and goes to the language model instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional


@dataclass
class Intent:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    spanish: bool = True


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, drop wake words and politeness."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[¿?¡!.,;:\"“”«»]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(hey|hola|oye|ok|okay)?\s*jarvis\s*", "", text)
    text = re.sub(r"\s*(por favor|porfa|please)$", "", text)
    text = re.sub(r"^(por favor|porfa|please)\s*", "", text)
    text = re.sub(r"^(puedes|podrias|can you|could you)\s+", "", text)
    return text.strip()


_SPANISH_HINTS = re.compile(
    r"\b(el|la|los|las|de|que|pon|abre|sube|baja|siguiente|anterior|cancion|"
    r"musica|volumen|hoy|manana|agenda|tengo|hora|dia|reproduce|pausa|silencio|"
    r"quita|activa|para|deten|reanuda|salta|pasa|y|en|mi|mis|dale|ponme|abre)\b"
)


_ENGLISH_HINTS = re.compile(
    r"\b(the|turn|up|down|next|song|play|what|whats|my|open|set|volume|skip|"
    r"previous|pause|music|is|it|on|to|louder|quieter|today|tomorrow|do|i|"
    r"have|any|launch|start|schedule|remind|me|stop|resume|track|go|back|time)\b"
)


def _is_spanish(text: str) -> bool:
    """Spanish unless the words are clearly English (the default audience)."""
    if _SPANISH_HINTS.search(text):
        return True
    return not _ENGLISH_HINTS.search(text)


def _restore(original: str, fragment: str) -> str:
    """Recover accents and capitals of *fragment* from the original text."""
    words = re.findall(r"[\w'%-]+", original)
    plain = [normalize(w) or w.lower() for w in words]
    target = fragment.split()
    for start in range(len(plain) - len(target) + 1):
        if plain[start : start + len(target)] == target:
            return " ".join(words[start : start + len(target)])
    return fragment


# ---------------------------------------------------------------------------
# Dates and times for scheduling
# ---------------------------------------------------------------------------

_WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "domingo": 6,
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NUMBER_WORDS = {
    "una": 1,
    "uno": 1,
    "one": 1,
    "dos": 2,
    "two": 2,
    "tres": 3,
    "three": 3,
    "cuatro": 4,
    "four": 4,
    "cinco": 5,
    "five": 5,
    "seis": 6,
    "six": 6,
    "siete": 7,
    "seven": 7,
    "ocho": 8,
    "eight": 8,
    "nueve": 9,
    "nine": 9,
    "diez": 10,
    "ten": 10,
    "once": 11,
    "eleven": 11,
    "doce": 12,
    "twelve": 12,
}
_NUM = r"(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"


def _num(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


@dataclass
class When:
    start: datetime
    all_day: bool
    remainder: str
    duration: Optional[timedelta] = None


def parse_when(text: str, now: Optional[datetime] = None) -> Optional[When]:
    """Find a day and/or time in *text*; return it and the leftover words."""
    now = now or datetime.now()
    rest = f" {text} "
    meridiem: Optional[str] = None

    # Morning/afternoon markers first: "de la manana" is not "tomorrow".
    for pattern, value in (
        (r" (de|por|en) la manana ", "am"),
        (r" in the morning ", "am"),
        (r" (de|por|en) la (tarde|noche) ", "pm"),
        (r" (in the|this) (afternoon|evening) ", "pm"),
        (r" at night ", "pm"),
    ):
        if re.search(pattern, rest):
            meridiem = value
            rest = re.sub(pattern, " ", rest)

    day: Optional[date] = None
    for pattern, delta in (
        (r" pasado manana ", 2),
        (r" day after tomorrow ", 2),
        (r" manana ", 1),
        (r" tomorrow ", 1),
        (r" hoy ", 0),
        (r" today ", 0),
        (r" tonight ", 0),
        (r" esta noche ", 0),
    ):
        if re.search(pattern, rest):
            day = now.date() + timedelta(days=delta)
            if pattern in (r" tonight ", r" esta noche "):
                meridiem = meridiem or "pm"
            rest = re.sub(pattern, " ", rest, count=1)
            break
    if day is None:
        match = re.search(
            r" (el |este |el proximo |on |next |this )?(" + "|".join(_WEEKDAYS) + r") ",
            rest,
        )
        if match:
            target = _WEEKDAYS[match.group(2)]
            ahead = (target - now.weekday()) % 7 or 7
            day = now.date() + timedelta(days=ahead)
            rest = rest.replace(match.group(0), " ", 1)

    relative = re.search(
        r" (en|dentro de|in) " + _NUM + r" (minutos?|horas?|minutes?|hours?) ", rest
    )
    if relative:
        amount = _num(relative.group(2))
        unit = relative.group(3)
        delta = (
            timedelta(hours=amount)
            if unit.startswith("h")
            else timedelta(minutes=amount)
        )
        rest = rest.replace(relative.group(0), " ", 1)
        return When(now + delta, False, " ".join(rest.split()))

    hour = minute = None
    match = re.search(
        r" (a las|a la|at|para las) " + _NUM + r"(?::(\d{2}))?( y media| y cuarto)?"
        r"( ?(am|pm|a m|p m))? ",
        rest,
    )
    if match is None:
        match = re.search(r" (al )?(mediodia|noon) ", rest)
        if match:
            hour, minute = 12, 0
            rest = rest.replace(match.group(0), " ", 1)
    else:
        hour = _num(match.group(2))
        minute = int(match.group(3)) if match.group(3) else 0
        if match.group(4) == " y media":
            minute = 30
        elif match.group(4) == " y cuarto":
            minute = 15
        marker = (match.group(6) or "").replace(" ", "")
        meridiem = marker or meridiem
        rest = rest.replace(match.group(0), " ", 1)

    if hour is not None:
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif meridiem is None and 1 <= hour <= 7:
            hour += 12  # "a las 5" almost always means the afternoon
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        if day is None:
            day = now.date()
            if (
                datetime.combine(day, datetime.min.time()).replace(
                    hour=hour, minute=minute
                )
                <= now
            ):
                day += timedelta(days=1)
        start = datetime.combine(day, datetime.min.time()).replace(
            hour=hour, minute=minute
        )
        return When(start, False, " ".join(rest.split()))
    if day is not None:
        return When(
            datetime.combine(day, datetime.min.time()), True, " ".join(rest.split())
        )
    return None


# ---------------------------------------------------------------------------
# Command patterns
# ---------------------------------------------------------------------------

_VOLUME = r"(el )?(volumen|sonido|audio)"


def _match(pattern: str, text: str) -> Optional[re.Match]:
    return re.fullmatch(pattern, text)


def parse(raw: str, now: Optional[datetime] = None) -> Optional[Intent]:
    """Return the command in *raw*, or None to let the model handle it."""
    text = normalize(raw)
    if not text or len(text) > 120:
        return None
    es = _is_spanish(text)

    # --- volume -------------------------------------------------------------
    m = _match(
        r"(pon|ajusta|sube|baja|deja|cambia)? ?" + _VOLUME + r" (al|a|en) (\d{1,3})"
        r"( por ciento| %|%)?|set (the )?volume to (\d{1,3})( percent|%)?",
        text,
    )
    if m:
        level = next(int(g) for g in (m.group(5), m.group(8)) if g)
        return Intent("volume.set", {"percent": min(100, level)}, es)
    up = (
        r"(sube|subele|subeme|aumenta|subir)( el| le)?( volumen| sonido| audio)?"
        r"( un poco| mas| mucho)?|mas (volumen|sonido)|(turn (it|the volume) up"
        r"|turn up the volume|volume up|louder)( a bit)?"
    )
    if _match(up, text):
        return Intent("volume.up", {"step": 20 if "mucho" in text else 10}, es)
    down = (
        r"(baja|bajale|bajame|disminuye|bajar)( el| le)?( volumen| sonido| audio)?"
        r"( un poco| mas| mucho)?|menos (volumen|sonido)|(turn (it|the volume) down"
        r"|turn down the volume|volume down|quieter)( a bit)?"
    )
    if _match(down, text):
        return Intent("volume.down", {"step": 20 if "mucho" in text else 10}, es)
    if _match(
        r"(silencio|silencia( todo| el sonido)?|quita el (sonido|volumen)|mutea|mute( it)?)",  # noqa: E501
        text,
    ):
        return Intent("volume.mute", {"mute": True}, es)
    if _match(r"(activa|devuelve|pon) el sonido|quita el silencio|unmute", text):
        return Intent("volume.mute", {"mute": False}, es)

    # --- media --------------------------------------------------------------
    if _match(
        r"(pausa|pausar|pausala|pausa (la )?(musica|cancion|spotify)"
        r"|(para|deten|detener|stop) (la )?(musica|cancion|spotify)"
        r"|pause( (the )?(music|song|spotify))?|stop( the)? music)",
        text,
    ):
        return Intent("media.pause", {}, es)
    if _match(
        r"(reanuda|reanudar|continua|sigue|dale play|play|resume|play music"
        r"|reanuda (la )?(musica|cancion)|continua (la )?(musica|cancion)"
        r"|pon (la )?musica|resume (the )?music)",
        text,
    ):
        return Intent("media.play", {}, es)
    if _match(
        r"(siguiente|la siguiente|siguiente cancion|pasa(la| la cancion| de cancion| cancion)?"  # noqa: E501
        r"|salta(la| la cancion| esta cancion| cancion)?|cambia de cancion|otra cancion"
        r"|next( song| track)?|skip( (this|the) (song|track))?)",
        text,
    ):
        return Intent("media.next", {}, es)
    if _match(
        r"(anterior|la anterior|cancion anterior|la cancion anterior|regresa la cancion"
        r"|vuelve a la (cancion )?anterior|previous( song| track)?|go back( a song)?"
        r"|last song)",
        text,
    ):
        return Intent("media.previous", {}, es)
    if _match(
        r"(que (cancion|tema) (es esta|suena|esta sonando)|que suena"
        r"|what('s| is) (this song|playing))",
        text,
    ):
        return Intent("media.now_playing", {}, es)

    # --- spotify search & play ------------------------------------------------
    m = (
        _match(
            r"(pon|ponme|reproduce|toca|play)( la cancion| el tema| la playlist| el album"  # noqa: E501
            r"| musica de| canciones de| algo de| un poco de| the song| the playlist"
            r"| some| music by| songs by)? (.+?)( en| on) spotify",
            text,
        )
        or _match(
            r"(reproduce|ponme|pon|toca)( la cancion| el tema| la playlist| el album"
            r"| musica de| canciones de| algo de| un poco de)(.+)",
            text,
        )
        or _match(r"(reproduce) (.+)", text)
        or _match(
            r"play( the song| the playlist| some| music by| songs by)? (.+)", text
        )
    )
    if m:
        groups = [g for g in m.groups() if g]
        query = groups[-1] if groups[-1] not in (" en", " on") else groups[-2]
        query = re.sub(r"( en| on) spotify$", "", query).strip()
        qualifier = " ".join(groups[1:-1]) if len(groups) > 2 else ""
        kind = (
            "playlist"
            if "playlist" in qualifier
            else "album"
            if "album" in qualifier
            else "artist"
            if re.search(r"musica de|canciones de|music by|songs by", qualifier)
            else "auto"
        )
        if query and query not in ("musica", "music"):
            return Intent(
                "spotify.play", {"query": _restore(raw, query), "kind": kind}, es
            )

    # --- time and date ------------------------------------------------------
    if _match(
        r"(que hora es|dime la hora|la hora|hora|what time is it|what's the time"
        r"|tell me the time)",
        text,
    ):
        return Intent("time.now", {}, es)
    if _match(
        r"(que (dia|fecha) es( hoy)?|a que (dia|fecha) estamos|what day is (it|today)"
        r"|what('s| is) (the date|today's date))",
        text,
    ):
        return Intent("time.date", {}, es)

    # --- calendar -----------------------------------------------------------
    day_query = (
        r"(que tengo|que hay|tengo algo|mis eventos|mis reuniones|mi agenda|agenda)"
        r"( (para|en mi agenda|en el calendario|en mi calendario|de))?"
        r"( (hoy|manana|pasado manana))?"
        r"|what('s| is) (on )?my (calendar|schedule|agenda)( for)?( (today|tomorrow))?"
        r"|(do i have|any) (meetings|events)( (today|tomorrow))?"
    )
    if _match(day_query, text):
        offset = (
            2
            if "pasado manana" in text
            else (1 if re.search(r"manana|tomorrow", text) else 0)
        )
        return Intent("calendar.day", {"offset": offset}, es)
    if _match(
        r"(cual es )?(mi )?(proxima|siguiente) (reunion|cita|evento)"
        r"|what('s| is) my next (meeting|event|appointment)|next meeting",
        text,
    ):
        return Intent("calendar.next", {}, es)
    m = _match(
        r"(agenda(me)?|agendar|programa(me)?|crea (un )?evento|crear (un )?evento"
        r"|anade al calendario|pon en (mi|el) calendario|recuerdame|schedule"
        r"|add to my calendar|create (an )?event|remind me to) (.+)",
        text,
    )
    if m:
        when = parse_when(m.group(m.lastindex), now)
        if when is not None:
            title = re.sub(
                r"^(un evento |una cita |una reunion |el evento |la cita |an event |a meeting )",  # noqa: E501
                lambda x: (
                    "reunion "
                    if "reunion" in x.group(0) or "meeting" in x.group(0)
                    else ""
                ),
                when.remainder,
            ).strip(" -,")
            title = re.sub(r"^(de |que |to |para )", "", title).strip()
            if title:
                title = _restore(raw, title)
                return Intent(
                    "calendar.create",
                    {
                        "title": title[:1].upper() + title[1:],
                        "start": when.start,
                        "all_day": when.all_day,
                    },
                    es,
                )

    # --- open apps and sites --------------------------------------------------
    m = _match(
        r"(abre|abrir|abreme|inicia|iniciar|lanza|ejecuta|open|launch|start)"
        r"( la app| la aplicacion| el programa| la pagina de| la pagina| the app)? (.+)",  # noqa: E501
        text,
    )
    if m:
        target = m.group(3).strip()
        target = re.sub(r"^(el|la|los|las|the|mi|my) ", "", target)
        if target and len(target.split()) <= 4:
            return Intent(
                "app.open", {"name": target, "spoken": _restore(raw, target)}, es
            )

    return None
