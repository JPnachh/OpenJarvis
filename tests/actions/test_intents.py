"""Direct voice commands: understanding Spanish and English phrases."""

from __future__ import annotations

from datetime import datetime

import pytest

from openjarvis.actions.intents import normalize, parse, parse_when

NOW = datetime(2026, 10, 8, 10, 0)  # a Thursday morning


@pytest.mark.parametrize(
    "text, name, args, spanish",
    [
        ("Hey Jarvis, sube el volumen", "volume.up", {"step": 10}, True),
        ("súbele un poco", "volume.up", {"step": 10}, True),
        ("bájale mucho", "volume.down", {"step": 20}, True),
        ("pon el volumen al 40", "volume.set", {"percent": 40}, True),
        ("volumen al 80%", "volume.set", {"percent": 80}, True),
        ("silencio", "volume.mute", {"mute": True}, True),
        ("quita el silencio", "volume.mute", {"mute": False}, True),
        ("pausa la música", "media.pause", {}, True),
        ("dale play", "media.play", {}, True),
        ("siguiente canción", "media.next", {}, True),
        ("pásala", "media.next", {}, True),
        ("la anterior", "media.previous", {}, True),
        ("¿qué canción es esta?", "media.now_playing", {}, True),
        ("¿qué hora es?", "time.now", {}, True),
        ("qué día es hoy", "time.date", {}, True),
        ("¿qué tengo hoy?", "calendar.day", {"offset": 0}, True),
        ("qué tengo para mañana", "calendar.day", {"offset": 1}, True),
        ("cuál es mi próxima reunión", "calendar.next", {}, True),
        ("abre YouTube", "app.open", {"name": "youtube", "spoken": "YouTube"}, True),
        ("turn up the volume", "volume.up", {"step": 10}, False),
        ("next song", "media.next", {}, False),
        ("what's on my calendar tomorrow", "calendar.day", {"offset": 1}, False),
        ("open chrome", "app.open", {"name": "chrome", "spoken": "chrome"}, False),
    ],
)
def test_commands(text, name, args, spanish):
    intent = parse(text, NOW)
    assert intent is not None, text
    assert intent.name == name
    assert intent.args == args
    assert intent.spanish is spanish


@pytest.mark.parametrize(
    "text, query, kind",
    [
        ("pon Bad Bunny en Spotify", "Bad Bunny", "auto"),
        ("reproduce Despacito", "Despacito", "auto"),
        ("pon música de Shakira", "Shakira", "artist"),
        ("pon la playlist rock en español", "rock en español", "playlist"),
        ("play Bohemian Rhapsody", "Bohemian Rhapsody", "auto"),
    ],
)
def test_spotify_requests(text, query, kind):
    intent = parse(text, NOW)
    assert intent.name == "spotify.play"
    assert intent.args == {"query": query, "kind": kind}


@pytest.mark.parametrize(
    "text, title, start, all_day",
    [
        (
            "agenda reunión con Ana mañana a las 5",
            "Reunión con Ana",
            datetime(2026, 10, 9, 17, 0),
            False,
        ),
        (
            "agéndame dentista el viernes a las 10 de la mañana",
            "Dentista",
            datetime(2026, 10, 9, 10, 0),
            False,
        ),
        (
            "recuérdame llamar a mamá en 30 minutos",
            "Llamar a mamá",
            datetime(2026, 10, 8, 10, 30),
            False,
        ),
        (
            "crea un evento cumpleaños de Pedro el sábado",
            "Cumpleaños de Pedro",
            datetime(2026, 10, 10),
            True,
        ),
        ("schedule gym tomorrow at 7 pm", "Gym", datetime(2026, 10, 9, 19, 0), False),
    ],
)
def test_calendar_creation(text, title, start, all_day):
    intent = parse(text, NOW)
    assert intent.name == "calendar.create"
    assert intent.args == {"title": title, "start": start, "all_day": all_day}


@pytest.mark.parametrize(
    "text",
    [
        "explícame la teoría de la relatividad",
        "pon una alarma",
        "qué tiempo hace",
        "abre la puerta del garaje y enciende las luces del jardín ahora mismo",
        "",
    ],
)
def test_everything_else_goes_to_the_model(text):
    assert parse(text, NOW) is None


def test_time_parsing_details():
    assert parse_when("a las 9 y media de la noche", NOW).start == datetime(
        2026, 10, 8, 21, 30
    )
    # A time already past today means tomorrow.
    assert parse_when("a las 9 de la mañana", NOW).start == datetime(2026, 10, 9, 9, 0)
    assert parse_when("al mediodia", NOW).start == datetime(2026, 10, 8, 12, 0)
    assert parse_when("sin fecha", NOW) is None


def test_normalize_strips_wake_words_and_politeness():
    assert normalize("¡Oye Jarvis, pausa la música, por favor!") == "pausa la musica"
