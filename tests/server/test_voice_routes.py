"""Hands-free voice routes: events, commands, replies and wake-word training."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from openjarvis.server import voice_routes  # noqa: E402
from openjarvis.speech import wakeword  # noqa: E402
from openjarvis.speech._stubs import TranscriptionResult  # noqa: E402
from openjarvis.speech.features import write_wav  # noqa: E402

from ..speech._synth import HEY_JARVIS, OTHER, take  # noqa: E402


@pytest.fixture
def app():
    app = FastAPI()
    backend = MagicMock()
    backend.transcribe.return_value = TranscriptionResult(
        text=" What time is it? ",
        language="en",
        confidence=0.9,
        duration_seconds=1.0,
        segments=[],
    )
    app.state.speech_backend = backend
    app.include_router(voice_routes.voice_router)
    app.include_router(voice_routes.wakeword_router)
    app.include_router(voice_routes.actions_router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


async def _collect(app, count: int) -> list[dict]:
    """Read *count* events from the SSE generator without a real socket."""
    request = MagicMock()
    request.app = app

    async def connected():
        return False

    request.is_disconnected = connected
    response = await voice_routes.voice_events(request)
    events = []
    iterator = response.body_iterator
    while len(events) < count:
        chunk = await asyncio.wait_for(iterator.__anext__(), 2)
        if chunk.startswith("data: "):
            events.append(json.loads(chunk[6:]))
    await iterator.aclose()
    return events


def test_command_without_page_waits_and_is_delivered_on_connect(client, app):
    result = client.post("/v1/voice/command", json={"text": "hello jarvis"}).json()
    assert result["delivered"] == 0 and result["id"]

    events = asyncio.run(_collect(app, 2))
    assert events[0]["type"] == "hello"
    assert events[1]["type"] == "command"
    assert events[1]["text"] == "hello jarvis"
    assert app.state.voice_hub.pending == []


def test_audio_command_is_transcribed(client, app):
    response = client.post(
        "/v1/voice/command",
        files={"file": ("command.wav", b"RIFF....", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "What time is it?"
    app.state.speech_backend.transcribe.assert_called_once()


def test_audio_command_without_backend(app):
    app.state.speech_backend = None
    response = TestClient(app).post(
        "/v1/voice/command", files={"file": ("c.wav", b"x", "audio/wav")}
    )
    assert response.status_code == 501


def test_reply_round_trip(client):
    command_id = client.post("/v1/voice/command", json={"text": "hi"}).json()["id"]
    assert client.post(
        "/v1/voice/reply", json={"id": command_id, "text": "Hello!"}
    ).json() == {"ok": True}
    reply = client.get(f"/v1/voice/reply/{command_id}?timeout=2")
    assert reply.json() == {"text": "Hello!"}


def test_reply_times_out(client):
    command_id = client.post("/v1/voice/command", json={"text": "hi"}).json()["id"]
    assert client.get(f"/v1/voice/reply/{command_id}?timeout=1").status_code == 404


def test_state_is_validated_and_reported(client):
    assert (
        client.post(
            "/v1/voice/state", json={"source": "listener", "state": "listening"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/voice/state", json={"source": "x", "state": "listening"}
        ).status_code
        == 400
    )
    status = client.get("/v1/voice/status").json()
    assert status["states"]["listener"]["state"] == "listening"
    assert status["subscribers"] == 0


def test_wakeword_training_over_http(client, tmp_path, monkeypatch):
    monkeypatch.setattr(wakeword, "wakeword_dir", lambda: tmp_path)
    rng = np.random.default_rng(3)
    for _ in range(4):
        r = client.post(
            "/v1/wakeword/samples",
            data={"kind": "positive"},
            files={"file": ("s.wav", write_wav(take(HEY_JARVIS, rng)), "audio/wav")},
        )
        assert r.status_code == 200, r.text
    client.post(
        "/v1/wakeword/samples",
        data={"kind": "negative"},
        files={"file": ("n.wav", write_wav(take(OTHER, rng)), "audio/wav")},
    )
    bad = client.post(
        "/v1/wakeword/samples",
        data={"kind": "positive"},
        files={"file": ("s.wav", write_wav(np.zeros(16000)), "audio/wav")},
    )
    assert bad.status_code == 400

    status = client.get("/v1/wakeword").json()
    assert status["positives"] == 4 and status["negatives"] == 1
    assert not status["trained"]

    trained = client.post("/v1/wakeword/train", json={"phrase": "Oye Jarvis"}).json()
    assert trained["phrase"] == "Oye Jarvis"
    assert client.get("/v1/wakeword").json()["trained"]

    sample_id = status["samples"][0]["id"]
    audio = client.get(f"/v1/wakeword/samples/{sample_id}/audio")
    assert audio.headers["content-type"] == "audio/wav"
    assert client.delete(f"/v1/wakeword/samples/{sample_id}").status_code == 200
    assert client.delete(f"/v1/wakeword/samples/{sample_id}").status_code == 404


def test_training_without_samples_explains(client, tmp_path, monkeypatch):
    monkeypatch.setattr(wakeword, "wakeword_dir", lambda: tmp_path)
    response = client.post("/v1/wakeword/train", json={})
    assert response.status_code == 400
    assert "at least 3" in response.json()["detail"]


def test_direct_command_runs_without_the_model(client, app, monkeypatch):
    from openjarvis.actions import router as actions_router
    from openjarvis.actions.router import ActionResult

    monkeypatch.setattr(
        actions_router,
        "handle",
        lambda text: (
            ActionResult("volume.up", "Subí el volumen.") if "volumen" in text else None
        ),
    )
    result = client.post("/v1/voice/command", json={"text": "sube el volumen"}).json()
    assert result["action"] == {
        "reply": "Subí el volumen.",
        "ok": True,
        "intent": "volume.up",
    }
    # The listener can collect the spoken reply at once.
    assert client.get(f"/v1/voice/reply/{result['id']}?timeout=1").json() == {
        "text": "Subí el volumen."
    }
    assert app.state.voice_hub.pending == []  # nothing queued for the model

    other = client.post("/v1/voice/command", json={"text": "cuéntame un chiste"})
    assert "action" not in other.json()

    run = client.post("/v1/actions/run", json={"text": "sube el volumen"}).json()
    assert run["handled"] and run["reply"] == "Subí el volumen."
    assert client.post("/v1/actions/run", json={"text": "hola"}).json() == {
        "handled": False
    }


def test_custom_commands_api(client, tmp_path, monkeypatch):
    from openjarvis.actions import custom

    monkeypatch.setattr(custom, "commands_path", lambda: tmp_path / "commands.json")
    info = client.get("/v1/actions/commands").json()
    assert info["commands"] == [] and info["examples"]

    created = client.post(
        "/v1/actions/commands",
        json={
            "phrases": "modo trabajo\nwork mode",
            "action": "open",
            "target": "https://calendar.google.com",
            "reply": "Listo",
        },
    )
    assert created.status_code == 200, created.text
    shell = client.post(
        "/v1/actions/commands",
        json={"phrases": ["x"], "action": "shell", "target": "rm -rf /"},
    )
    assert shell.status_code == 400
    duplicate = client.post(
        "/v1/actions/commands",
        json={"phrases": ["Modo trabajo"], "action": "say", "reply": "hi"},
    )
    assert duplicate.status_code == 400

    commands = client.get("/v1/actions/commands").json()["commands"]
    assert commands[0]["phrases"] == ["modo trabajo", "work mode"]
    assert client.delete(f"/v1/actions/commands/{commands[0]['id']}").status_code == 200
    assert client.get("/v1/actions/commands").json()["commands"] == []
