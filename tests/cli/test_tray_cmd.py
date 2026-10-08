"""System tray: status icon, menu actions and autostart."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

PIL = pytest.importorskip("PIL")

from openjarvis.cli import desktop_integration as desk  # noqa: E402
from openjarvis.cli import tray_cmd  # noqa: E402


def _app(**kwargs):
    defaults = dict(
        frontend_port=5173, api_port=None, listen=True, wake_sensitivity=0.5, claps=2
    )
    return tray_cmd.TrayApp(**{**defaults, **kwargs})


def test_icon_shows_state_colour():
    for state, colour in tray_cmd.STATE_COLORS.items():
        image = tray_cmd.tray_image(state)
        assert image.size == (64, 64)
        assert image.getpixel((58, 58))[:3] == colour


def test_state_updates_icon_and_tooltip():
    app = _app()
    app.icon = SimpleNamespace(icon=None, title="")
    app.set_state("listening")
    assert app.icon.title == "OpenJarvis: listening to you"
    assert app.icon.icon.getpixel((58, 58))[:3] == tray_cmd.STATE_COLORS["listening"]


def test_tray_does_not_fight_a_running_listener(monkeypatch):
    app = _app()
    notes = []
    app.icon = SimpleNamespace(notify=lambda message, title: notes.append(message))
    monkeypatch.setattr("openjarvis.cli.clap_cmd._running_listener", lambda: 4242)
    app.start_listening()
    assert not app.listening
    assert "4242" in notes[0]


def test_autostart_toggle(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(desk.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    app = _app()
    assert not app.autostart_installed()
    app.toggle_autostart()
    assert app.autostart_installed()
    assert "-m openjarvis.cli tray" in desk.autostart_path().read_text()
    app.toggle_autostart()
    assert not app.autostart_installed()
