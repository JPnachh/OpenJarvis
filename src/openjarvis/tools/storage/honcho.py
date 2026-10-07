"""Honcho memory backend — long-term user modeling via Plastic Labs' Honcho.

Unlike the other backends, Honcho does not just index text: it reasons over the
messages you store and builds a representation of the user. ``retrieve`` returns
search hits plus (optionally) a synthesized answer from Honcho's dialectic API.

Configuration (environment variables):

* ``HONCHO_API_KEY``      — key from https://app.honcho.dev (managed service)
* ``HONCHO_URL``          — base URL of a self-hosted Honcho (e.g.
  ``http://localhost:8000``); with no auth the API key may be omitted
* ``HONCHO_WORKSPACE_ID`` — workspace name (default ``openjarvis``)

Requires ``pip install honcho-ai``.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from openjarvis.core.events import EventType, get_event_bus
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage._stubs import MemoryBackend, RetrievalResult

logger = logging.getLogger(__name__)


@MemoryRegistry.register("honcho")
class HonchoMemory(MemoryBackend):
    """Memory backend backed by Honcho (managed or self-hosted)."""

    backend_id: str = "honcho"

    def __init__(
        self,
        db_path: str = "",  # unused; accepted for registry-signature parity
        *,
        workspace_id: str = "",
        api_key: str = "",
        base_url: str = "",
        user_peer: str = "user",
        assistant_peer: str = "jarvis",
        session_id: str = "openjarvis",
        dialectic: bool = True,
    ) -> None:
        try:
            from honcho import Honcho
        except ImportError as exc:  # picked up by the optional-import guard
            raise ImportError(
                "Honcho backend requires `pip install honcho-ai`"
            ) from exc

        api_key = api_key or os.environ.get("HONCHO_API_KEY", "")
        base_url = base_url or os.environ.get("HONCHO_URL", "")
        if not api_key and not base_url:
            raise RuntimeError(
                "Set HONCHO_API_KEY (managed) or HONCHO_URL (self-hosted)"
            )

        kwargs: Dict[str, Any] = {
            "workspace_id": workspace_id
            or os.environ.get("HONCHO_WORKSPACE_ID", "openjarvis")
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._honcho = Honcho(**kwargs)
        self._user = self._honcho.peer(user_peer)
        self._assistant = self._honcho.peer(assistant_peer)
        self._session = self._honcho.session(session_id)
        self._session.add_peers([self._user, self._assistant])
        self._dialectic = dialectic

    def store(
        self,
        content: str,
        *,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record *content* as a message from the user peer."""
        meta = dict(metadata or {})
        if source:
            meta["source"] = source
        message = self._user.message(content, metadata=meta or None)
        created = self._session.add_messages([message])
        doc_id = str(getattr(created[0], "id", "") or uuid.uuid4().hex)
        get_event_bus().publish(
            EventType.MEMORY_STORE,
            {"backend": self.backend_id, "doc_id": doc_id, "source": source},
        )
        return doc_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """Return Honcho's synthesized answer (if enabled) plus search hits."""
        if not query.strip():
            return []

        results: List[RetrievalResult] = []

        if self._dialectic:
            try:
                answer = self._user.chat(query)
                answer = getattr(answer, "content", answer)
                if answer:
                    results.append(
                        RetrievalResult(
                            content=str(answer),
                            score=1.0,
                            source="honcho:dialectic",
                            metadata={"kind": "dialectic"},
                        )
                    )
            except Exception:
                logger.warning("Honcho dialectic query failed", exc_info=True)

        try:
            hits = self._user.search(query, limit=top_k)
            for rank, hit in enumerate(list(hits)[:top_k]):
                text = getattr(hit, "content", None) or str(hit)
                results.append(
                    RetrievalResult(
                        content=text,
                        score=1.0 / (rank + 2),
                        source="honcho:search",
                        metadata=dict(getattr(hit, "metadata", None) or {}),
                    )
                )
        except Exception:
            logger.warning("Honcho search failed", exc_info=True)

        get_event_bus().publish(
            EventType.MEMORY_RETRIEVE,
            {"backend": self.backend_id, "query": query, "num_results": len(results)},
        )
        return results[: top_k + (1 if self._dialectic else 0)]

    def delete(self, doc_id: str) -> bool:
        """Honcho keeps derived state; individual deletes are unsupported."""
        logger.warning("HonchoMemory.delete is not supported")
        return False

    def clear(self) -> None:
        """Honcho workspaces are cleared from the Honcho dashboard/API."""
        logger.warning("HonchoMemory.clear is not supported; reset the workspace")


__all__ = ["HonchoMemory"]
