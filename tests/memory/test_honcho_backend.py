"""Tests for the Honcho memory backend (SDK mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("honcho")

from openjarvis.core.registry import MemoryRegistry  # noqa: E402
from openjarvis.tools.storage.honcho import HonchoMemory  # noqa: E402


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("HONCHO_API_KEY", "k")
    with patch("honcho.Honcho") as honcho_cls:
        client = honcho_cls.return_value
        client.peer.return_value = MagicMock()
        backend = HonchoMemory()
        yield backend, client


def test_registered():
    MemoryRegistry.register_value("honcho", HonchoMemory)
    assert MemoryRegistry.contains("honcho")


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)
    monkeypatch.delenv("HONCHO_URL", raising=False)
    with pytest.raises(RuntimeError):
        HonchoMemory()


def test_store_returns_honcho_message_id(backend):
    mem, _ = backend
    mem._session.add_messages.return_value = [SimpleNamespace(id="msg_1")]
    assert mem.store("I like tea", source="chat") == "msg_1"


def test_retrieve_combines_dialectic_and_search(backend):
    mem, _ = backend
    mem._user.chat.return_value = SimpleNamespace(content="Likes tea")
    mem._user.search.return_value = [
        SimpleNamespace(content="I like tea", metadata={"source": "chat"})
    ]
    results = mem.retrieve("drinks?", top_k=3)
    assert [r.source for r in results] == ["honcho:dialectic", "honcho:search"]
    assert results[0].content == "Likes tea"
    mem._user.search.assert_called_once_with("drinks?", limit=3)


def test_retrieve_survives_dialectic_failure(backend):
    mem, _ = backend
    mem._user.chat.side_effect = RuntimeError("boom")
    mem._user.search.return_value = [SimpleNamespace(content="x", metadata=None)]
    assert len(mem.retrieve("q")) == 1


def test_delete_unsupported(backend):
    mem, _ = backend
    assert mem.delete("x") is False
