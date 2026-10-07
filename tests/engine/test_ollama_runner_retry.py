"""Ollama's CUDA runner can die while loading a cold model; a retry succeeds."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from openjarvis.core.types import Message, Role
from openjarvis.engine import ollama as ollama_mod
from openjarvis.engine.ollama import OllamaEngine

CRASH = {
    "error": (
        "llama-server process has terminated: exit status 0xc0000409: "
        "CUDA error\nCUDA error: shared object initialization failed"
    )
}
MSGS = [Message(role=Role.USER, content="hola")]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ollama_mod, "_RUNNER_RETRY_DELAY", 0)


def _engine(handler) -> OllamaEngine:
    engine = OllamaEngine(host="http://testhost:11434")
    engine._client = httpx.Client(
        base_url="http://testhost:11434", transport=httpx.MockTransport(handler)
    )
    engine._async_transport = httpx.MockTransport(handler)
    return engine


def _flaky(crashes: int, ok_body: bytes):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= crashes:
            return httpx.Response(500, json=CRASH)
        return httpx.Response(200, content=ok_body)

    return handler, calls


def _chat_ok() -> bytes:
    return json.dumps({"model": "m", "message": {"content": "¡hola!"}}).encode()


def _stream_ok() -> bytes:
    lines = [
        {"message": {"content": "¡ho"}, "done": False},
        {"message": {"content": "la!"}, "done": True, "eval_count": 2},
    ]
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


def test_generate_retries_after_runner_crash():
    handler, calls = _flaky(1, _chat_ok())
    result = _engine(handler).generate(MSGS, model="m")
    assert result["content"] == "¡hola!" and calls["n"] == 2


def test_generate_gives_up_after_the_retry_budget():
    handler, calls = _flaky(99, _chat_ok())
    with pytest.raises(RuntimeError, match="llama-server process has terminated"):
        _engine(handler).generate(MSGS, model="m")
    assert calls["n"] == 1 + ollama_mod._RUNNER_RETRIES


def test_generate_does_not_retry_other_server_errors():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "something else"})

    with pytest.raises(RuntimeError):
        _engine(handler).generate(MSGS, model="m")
    assert calls["n"] == 1


def test_stream_retries_after_runner_crash():
    handler, calls = _flaky(1, _stream_ok())

    async def go():
        return [t async for t in _engine(handler).stream(MSGS, model="m")]

    assert "".join(asyncio.run(go())) == "¡hola!" and calls["n"] == 2


def test_stream_full_retries_after_runner_crash():
    handler, calls = _flaky(2, _stream_ok())

    async def go():
        return [c async for c in _engine(handler).stream_full(MSGS, model="m")]

    chunks = asyncio.run(go())
    assert "".join(c.content or "" for c in chunks) == "¡hola!" and calls["n"] == 3
