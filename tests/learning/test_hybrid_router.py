"""Tests for the local/cloud hybrid router (`jarvis-auto`)."""

from __future__ import annotations

from types import SimpleNamespace

from openjarvis.core.config import HybridRoutingConfig, JarvisConfig
from openjarvis.learning.routing.hybrid_router import AUTO_MODEL_ID, decide
from openjarvis.server.models import ChatCompletionRequest, ChatMessage
from openjarvis.server.routes import _apply_hybrid_routing

CFG = HybridRoutingConfig()
LOCAL = "qwen3:8b"

COMPLEX = (
    "Analiza paso a paso las ventajas y desventajas de migrar este servicio a "
    "microservicios. Primero compara costos, luego evalúa riesgos, y además "
    "explica cómo escribirías el plan. 1. Costos 2. Riesgos 3. Cronograma 4. Equipo"
)


def route(text, **kw):
    kw.setdefault("cloud_ok", True)
    return decide(text, CFG, local_default=LOCAL, **kw)


def test_trivial_stays_local():
    d = route("hola, ¿qué tal?")
    assert (d.target, d.model) == ("local", LOCAL)


def test_complex_escalates_to_claude():
    d = route(COMPLEX)
    assert d.target == "cloud"
    assert d.model == CFG.cloud_model


def test_directives_override_and_are_stripped():
    d = route("/claude hola")
    assert (d.target, d.query) == ("cloud", "hola")
    assert route("/opus hola").model == CFG.heavy_model
    assert route(f"/local {COMPLEX}").target == "local"


def test_privacy_keyword_forces_local_even_if_complex():
    assert route(COMPLEX + " Es información confidencial.").target == "local"


def test_accent_insensitive_escalation_phrase():
    assert route("Piensa a fondo sobre esto").target == "cloud"


def test_long_prompt_escalates():
    assert route("a " * (CFG.cloud_min_chars // 2 + 10)).target == "cloud"


def test_no_api_key_falls_back_to_local_with_reason():
    d = route("/claude hola", cloud_ok=False)
    assert d.target == "local"
    assert "ANTHROPIC_API_KEY" in d.reason


def test_config_section_is_registered():
    assert isinstance(JarvisConfig().hybrid_routing, HybridRoutingConfig)


def _req(text, model=AUTO_MODEL_ID):
    return ChatCompletionRequest(
        model=model, messages=[ChatMessage(role="user", content=text)]
    )


def test_server_hook_rewrites_model_and_strips_directive(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = JarvisConfig()
    cfg.intelligence.default_model = LOCAL
    req = _req("/claude hola")
    decision = _apply_hybrid_routing(req, cfg)
    assert decision.target == "cloud"
    assert req.model == CFG.cloud_model
    assert req.messages[-1].content == "hola"


def test_server_hook_ignores_other_models():
    req = _req("/claude hola", model="qwen3:8b")
    assert _apply_hybrid_routing(req, JarvisConfig()) is None
    assert req.model == "qwen3:8b"
    assert req.messages[-1].content == "/claude hola"


def test_disabled_routing_maps_auto_to_local():
    cfg = JarvisConfig()
    cfg.intelligence.default_model = LOCAL
    cfg.hybrid_routing.enabled = False
    req = _req(COMPLEX)
    assert _apply_hybrid_routing(req, cfg) is None
    assert req.model == LOCAL




def test_models_endpoint_lists_auto_model_first():
    from fastapi.testclient import TestClient

    from openjarvis.server.app import create_app

    engine = SimpleNamespace(
        engine_id="mock",
        list_models=lambda: ["qwen3:8b"],
        health=lambda: True,
    )
    cfg = JarvisConfig()
    cfg.analytics.enabled = False
    cfg.traces.enabled = False
    cfg.security.enabled = False
    app = create_app(engine, "qwen3:8b", config=cfg)
    ids = [m["id"] for m in TestClient(app).get("/v1/models").json()["data"]]
    assert ids[0] == AUTO_MODEL_ID
    assert "qwen3:8b" in ids
