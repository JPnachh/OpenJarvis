"""Running commands: system calls, Spotify fallback, calendar, custom commands."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from openjarvis.actions import router, system
from openjarvis.actions.custom import CommandBook, CustomCommand
from openjarvis.actions.spotify import SpotifyClient, SpotifyError


@pytest.fixture(autouse=True)
def no_spotify(monkeypatch):
    monkeypatch.setattr(router, "_spotify", lambda: None)


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(system, "change_volume", lambda d: log.append(("volume", d)))
    monkeypatch.setattr(system, "set_volume", lambda p: log.append(("set", p)))
    monkeypatch.setattr(system, "toggle_mute", lambda m=None: log.append(("mute", m)))
    monkeypatch.setattr(system, "media", lambda a: log.append(("media", a)))
    monkeypatch.setattr(system, "open_app", lambda n: log.append(("app", n)) or n)
    monkeypatch.setattr(system, "open_url", lambda u: log.append(("url", u)))
    return log


def run(text):
    return router.handle(text, book=CommandBook())


def test_volume_and_media(calls):
    assert run("sube el volumen").reply == "Subí el volumen."
    assert run("turn down the volume").reply == "Volume down."
    assert run("siguiente canción").reply == "Siguiente canción."
    assert run("pausa").ok
    assert calls == [
        ("volume", 10),
        ("volume", -10),
        ("media", "next"),
        ("media", "pause"),
    ]


def test_open_app_reply_keeps_spoken_name(calls):
    assert run("abre YouTube").reply == "Abriendo YouTube."
    assert calls == [("app", "youtube")]


def test_not_a_command_returns_none(calls):
    assert run("cuéntame un chiste") is None
    assert calls == []


def test_failures_become_spoken_errors(monkeypatch):
    def broken(_):
        raise system.ActionError("Install playerctl to control media players")

    monkeypatch.setattr(system, "media", broken)
    result = run("pausa")
    assert not result.ok
    assert result.reply.startswith("No pude hacerlo")


def test_spotify_without_connection_opens_search(monkeypatch, calls):
    opened = []
    monkeypatch.setattr(
        "openjarvis.actions.spotify.open_search_in_app", lambda q: opened.append(q)
    )
    result = run("pon Bad Bunny en Spotify")
    assert opened == ["Bad Bunny"]
    assert "Conecta Spotify" in result.reply


def test_time_answers():
    reply = run("qué hora es").reply
    assert reply.startswith("Son las ")
    assert run("what day is it").reply.startswith("Today is ")


def test_calendar_create(monkeypatch):
    created = {}
    from openjarvis.actions import calendar

    monkeypatch.setattr(
        calendar,
        "create_event",
        lambda title, start, end=None, all_day=False: created.update(
            title=title, start=start, all_day=all_day
        ),
    )
    result = run("agenda dentista mañana a las 10 de la mañana")
    assert created["title"] == "Dentista"
    assert created["start"].hour == 10
    assert result.reply.startswith("Listo, agendé Dentista para mañana a las 10:00")


def test_calendar_day(monkeypatch):
    from openjarvis.actions import calendar

    monkeypatch.setattr(
        calendar,
        "events_on",
        lambda day: [
            {"summary": "Standup", "start": {"dateTime": f"{day}T09:30:00"}},
            {"summary": "Feriado", "start": {"date": str(day)}},
        ],
    )
    reply = run("qué tengo hoy").reply
    assert reply == "Tienes 2 eventos: 09:30 Standup; Feriado (todo el día)."


def test_custom_commands(tmp_path, calls):
    book = CommandBook()
    book.add(
        CustomCommand(
            ["modo trabajo", "work mode"],
            "open",
            "https://calendar.google.com",
            "Modo trabajo listo",
        )
    )
    book.add(CustomCommand(["súbelo todo"], "keys", "volume_up"))
    book.add(CustomCommand(["di hola"], "say", reply="¡Hola!"))
    with pytest.raises(ValueError, match="Already used"):
        book.add(CustomCommand(["Modo Trabajo"], "say", reply="x"))
    path = book.save(tmp_path / "commands.json")
    loaded = CommandBook.load(path)
    assert len(loaded.commands) == 3

    assert router.handle("Oye Jarvis, ¡modo trabajo!", book=loaded).reply == (
        "Modo trabajo listo"
    )
    assert router.handle("subelo todo", book=loaded).reply == "Hecho."
    assert router.handle("di hola", book=loaded).reply == "¡Hola!"
    assert calls == [("url", "https://calendar.google.com"), ("volume", 10)]


def test_shell_commands_need_opt_in(tmp_path):
    (tmp_path / "commands.json").write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "phrases": ["borra todo"],
                        "action": "shell",
                        "target": "echo nope",
                    }
                ]
            }
        )
    )
    book = CommandBook.load(tmp_path / "commands.json")
    with mock.patch("subprocess.Popen") as popen:
        result = router.handle("borra todo", book=book)
    popen.assert_not_called()
    assert not result.ok


def test_invalid_custom_commands_are_rejected():
    with pytest.raises(ValueError):
        CustomCommand([], "open", "x").validate()
    with pytest.raises(ValueError):
        CustomCommand(["x"], "keys", "launch_missiles").validate()


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload
        self.content = b"" if payload is None else json.dumps(payload).encode()

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.get("params"), kwargs.get("json")))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, None, kwargs.get("data")))
        return self.responses.pop(0)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "spotify.json"
    path.write_text(
        json.dumps(
            {
                "access_token": "old",
                "refresh_token": "r",
                "client_id": "id",
                "client_secret": "s",
            }
        )
    )
    return path


def test_spotify_search_plays_first_track(token_file):
    http = FakeHttp(
        [
            FakeResponse(
                200,
                {
                    "tracks": {
                        "items": [
                            {
                                "uri": "spotify:track:1",
                                "name": "Tití Me Preguntó",
                                "artists": [{"name": "Bad Bunny"}],
                            }
                        ]
                    }
                },
            ),
            FakeResponse(204),
        ]
    )
    client = SpotifyClient(token_file, http=http)
    assert client.play_search("titi me pregunto") == "Tití Me Preguntó — Bad Bunny"
    assert http.calls[1][0] == "PUT" and http.calls[1][3] == {
        "uris": ["spotify:track:1"]
    }


def test_spotify_refreshes_expired_token_and_wakes_device(token_file):
    http = FakeHttp(
        [
            FakeResponse(401),
            FakeResponse(200, {"access_token": "new"}),
            FakeResponse(404),  # no active device
            FakeResponse(
                200,
                {
                    "devices": [
                        {"id": "phone", "type": "Smartphone"},
                        {"id": "pc", "type": "Computer"},
                    ]
                },
            ),
            FakeResponse(204),
        ]
    )
    SpotifyClient(token_file, http=http).next()
    assert json.loads(token_file.read_text())["access_token"] == "new"
    assert http.calls[-1][2] == {"device_id": "pc"}


def test_spotify_premium_error_is_explained(token_file):
    http = FakeHttp([FakeResponse(403)])
    with pytest.raises(SpotifyError, match="Premium"):
        SpotifyClient(token_file, http=http).pause()


def test_spotify_not_connected(tmp_path):
    assert not SpotifyClient(tmp_path / "missing.json").connected()
