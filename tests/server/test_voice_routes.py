"""Voice event hub: what the web UI uses to show state and mirror spoken turns."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from openjarvis.core.config import JarvisConfig
from openjarvis.server.app import create_app


class _Engine:
    engine_id = "mock"

    def list_models(self):
        return ["m"]

    def health(self):
        return True


@pytest.fixture
def client():
    cfg = JarvisConfig()
    cfg.analytics.enabled = False
    cfg.traces.enabled = False
    cfg.security.enabled = False
    return TestClient(create_app(_Engine(), "m", config=cfg))


def test_initial_state_is_offline(client):
    snap = client.get("/v1/voice/state").json()
    assert snap["state"]["state"] == "offline" and snap["turns"] == []


def test_state_event_updates_snapshot(client):
    r = client.post(
        "/v1/voice/events",
        json={"type": "state", "state": "heard", "trigger": "Hey Jarvis"},
    )
    assert r.status_code == 200
    state = client.get("/v1/voice/state").json()["state"]
    assert state["state"] == "heard" and state["trigger"] == "Hey Jarvis"


def test_turn_event_is_kept_with_unique_id(client):
    a = client.post(
        "/v1/voice/events",
        json={
            "type": "turn",
            "user": "hola",
            "assistant": "¡hola!",
            "model": "qwen3:8b",
        },
    ).json()
    b = client.post(
        "/v1/voice/events", json={"type": "turn", "user": "otra", "assistant": "ok"}
    ).json()
    assert a["id"] != b["id"]
    turns = client.get("/v1/voice/state").json()["turns"]
    assert [t["user"] for t in turns] == ["hola", "otra"]
    assert turns[0]["model"] == "qwen3:8b" and turns[0]["id"] == a["id"]


@pytest.mark.parametrize(
    "body",
    [
        {"type": "state", "state": "dancing"},
        {"type": "turn", "user": "", "assistant": ""},
        {"type": "nope"},
        {},
    ],
)
def test_invalid_events_are_rejected(client, body):
    assert client.post("/v1/voice/events", json=body).status_code == 400


def test_hub_fans_out_to_subscribers():
    import asyncio

    from openjarvis.server.voice_routes import VoiceHub

    async def go():
        hub = VoiceHub()
        q1, q2 = hub.subscribe(), hub.subscribe()
        hub.publish("state", {"state": "listening"})
        assert (await q1.get())["state"] == "listening"
        assert (await q2.get())["state"] == "listening"
        hub.unsubscribe(q1)
        hub.publish("state", {"state": "speaking"})
        assert q1.empty() and (await q2.get())["state"] == "speaking"

    asyncio.run(go())
