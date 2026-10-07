"""Hybrid local/cloud router behind the virtual model ``jarvis-auto``.

Decides, per message, whether a request runs on the local model (fast, free,
private) or escalates to Claude (slower per token, costs money, stronger).

Rules, in priority order — the first one that matches wins:

1. **Directive prefix** typed by the user: ``/local``, ``/claude``, ``/opus``.
2. **Privacy keywords** (``privado``, ``confidencial``…) -> always local.
3. **Escalation phrases** (``usa claude``, ``piensa a fondo``…) -> cloud.
4. **Long prompt** (>= ``cloud_min_chars``) -> cloud (local context is small).
5. **Complexity score** >= ``cloud_threshold`` -> cloud.
6. Otherwise -> local.

If the cloud is selected but ``ANTHROPIC_API_KEY`` is missing, the request
stays local and the reason says so, instead of failing.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from openjarvis.learning.routing.complexity import score_complexity

AUTO_MODEL_ID = "jarvis-auto"

_DIRECTIVE = re.compile(r"^\s*/(local|claude|opus)\b[ \t]*", re.IGNORECASE)


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of routing one message."""

    model: str
    target: str  # "local" | "cloud" | "heavy"
    reason: str
    tier: str
    score: float
    query: str  # message with any directive prefix removed


def _fold(text: str) -> str:
    """Lowercase and strip accents so keyword matching is forgiving."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _split(csv: str) -> list[str]:
    return [_fold(p.strip()) for p in (csv or "").split(",") if p.strip()]


def _first_hit(folded_query: str, keywords: list[str]) -> Optional[str]:
    for kw in keywords:
        if kw in folded_query:
            return kw
    return None


def cloud_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def decide(
    query: str,
    cfg: Any,
    *,
    local_default: str = "",
    cloud_ok: Optional[bool] = None,
) -> RouteDecision:
    """Route *query* using ``cfg`` (a :class:`HybridRoutingConfig`)."""
    cloud_ok = cloud_available() if cloud_ok is None else cloud_ok
    local_model = cfg.local_model or local_default

    directive = None
    match = _DIRECTIVE.match(query or "")
    if match:
        directive = match.group(1).lower()
        query = query[match.end() :]

    result = score_complexity(query)
    base = dict(tier=result.tier, score=result.score, query=query)

    def local(reason: str) -> RouteDecision:
        return RouteDecision(local_model, "local", reason, **base)

    def cloud(reason: str, heavy: bool = False) -> RouteDecision:
        if not cloud_ok:
            return local(f"{reason}, pero no hay ANTHROPIC_API_KEY: me quedo en local")
        if heavy:
            return RouteDecision(cfg.heavy_model, "heavy", reason, **base)
        return RouteDecision(cfg.cloud_model, "cloud", reason, **base)

    if directive == "local":
        return local("pediste /local")
    if directive == "claude":
        return cloud("pediste /claude")
    if directive == "opus":
        return cloud("pediste /opus (modelo más potente)", heavy=True)

    folded = _fold(query)
    hit = _first_hit(folded, _split(cfg.local_keywords))
    if hit:
        return local(f"contiene '{hit}': se queda en tu PC")
    hit = _first_hit(folded, _split(cfg.cloud_keywords))
    if hit:
        return cloud(f"contiene '{hit}'")
    if len(query) >= cfg.cloud_min_chars:
        return cloud(f"mensaje largo ({len(query)} caracteres)")
    if result.score >= cfg.cloud_threshold:
        return cloud(
            f"complejidad {result.score} >= {cfg.cloud_threshold} ({result.tier})"
        )
    return local(f"complejidad {result.score} ({result.tier}) < {cfg.cloud_threshold}")


__all__ = ["AUTO_MODEL_ID", "RouteDecision", "cloud_available", "decide"]
