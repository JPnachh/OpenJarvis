"""`jarvis setup-desktop`: shortcut, frontend and autostart in one go."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import desktop_integration as desk
from openjarvis.cli import setup_desktop_cmd


def test_setup_desktop_without_tray_installs_listener(tmp_path, monkeypatch):
    monkeypatch.setattr(desk, "create_desktop_shortcut", lambda: tmp_path / "icon")
    installed = []
    monkeypatch.setattr(
        desk,
        "install_autostart",
        lambda args, subcommand="clap": installed.append(subcommand) or tmp_path,
    )
    monkeypatch.setattr(
        setup_desktop_cmd, "_can_import", lambda name: name == "sounddevice"
    )
    result = CliRunner().invoke(setup_desktop_cmd.setup_desktop, ["--no-frontend"])
    assert result.exit_code == 0, result.output
    assert installed == ["listen"]
    assert "Desktop icon" in result.output and "Hey Jarvis" in result.output


def test_setup_desktop_with_tray_starts_it(tmp_path, monkeypatch):
    monkeypatch.setattr(desk, "create_desktop_shortcut", lambda: tmp_path / "icon")
    installed = []
    monkeypatch.setattr(
        desk,
        "install_autostart",
        lambda args, subcommand="clap": installed.append(subcommand) or tmp_path,
    )
    monkeypatch.setattr(setup_desktop_cmd, "_can_import", lambda name: True)
    monkeypatch.setattr("openjarvis.cli.tray_cmd._PID_FILE", tmp_path / "tray.pid")
    with mock.patch("subprocess.Popen") as popen:
        result = CliRunner().invoke(setup_desktop_cmd.setup_desktop, ["--no-frontend"])
    assert result.exit_code == 0, result.output
    assert installed == ["tray"]
    assert "tray" in popen.call_args.args[0]
