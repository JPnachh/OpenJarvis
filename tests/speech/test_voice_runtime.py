"""Voice runtime helpers: pause switch, single-instance lock, config wiring."""

from __future__ import annotations

import pytest

from openjarvis.core.config import JarvisConfig, VoiceAssistantConfig
from openjarvis.speech import voice_runtime as rt


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr("openjarvis.core.config.DEFAULT_CONFIG_DIR", tmp_path)


def test_pause_switch_round_trip():
    assert not rt.is_paused()
    rt.set_paused(True)
    assert rt.is_paused()
    rt.set_paused(False)
    assert not rt.is_paused()
    rt.set_paused(False)  # idempotent


def test_single_instance_lock():
    first = rt.acquire_single_instance(port=48999)
    assert first is not None
    try:
        assert rt.acquire_single_instance(port=48999) is None
    finally:
        first.close()
    again = rt.acquire_single_instance(port=48999)
    assert again is not None
    again.close()


def test_saved_device_used_when_config_says_auto():
    rt.device_file().write_text("2")
    cfg = JarvisConfig()
    assert cfg.voice_assistant.device == -1
    assert rt.options_from_config(cfg).device == 2
    cfg.voice_assistant.device = 5
    assert rt.options_from_config(cfg).device == 5


def test_options_follow_effective_server_port():
    cfg = JarvisConfig()
    cfg.server.port = 8123
    assert rt.options_from_config(cfg).server_url == "http://127.0.0.1:8123"


def test_voice_config_defaults_are_safe():
    va = VoiceAssistantConfig()
    assert va.autostart is False  # opt-in; the user's config.toml enables it
    assert va.speaker_verify and va.model == "jarvis-auto"


def test_server_starts_voice_only_when_autostart(monkeypatch):
    from fastapi.testclient import TestClient

    from openjarvis.server.app import create_app

    started = []
    monkeypatch.setattr(
        rt, "start_background", lambda cfg, url=None: started.append(cfg)
    )

    class _Engine:
        engine_id = "mock"

        def list_models(self):
            return ["m"]

        def health(self):
            return True

    def make(autostart):
        cfg = JarvisConfig()
        cfg.analytics.enabled = False
        cfg.traces.enabled = False
        cfg.security.enabled = False
        cfg.voice_assistant.autostart = autostart
        return create_app(_Engine(), "m", config=cfg)

    with TestClient(make(False)):
        pass
    assert started == []
    with TestClient(make(True)):
        pass
    assert len(started) == 1


def test_background_status_logs_sound_and_heartbeat(monkeypatch):
    lines = []
    clock = {"t": 1000.0}
    monkeypatch.setattr(rt.time, "monotonic", lambda: clock["t"])
    status = rt.background_status(lines.append, 0.5)

    status(100.0, 0.0)  # quiet: nothing logged
    assert lines == []
    status(2500.0, 0.31)  # speech-level sound that missed the wake word
    assert len(lines) == 1 and "0.31/0.5" in lines[0] and "2500" in lines[0]
    status(2600.0, 0.2)  # throttled (same 2 s window)
    assert len(lines) == 1
    clock["t"] += 61
    status(50.0, 0.0)
    assert "[vivo]" in lines[-1] and "2600" in lines[-1]
