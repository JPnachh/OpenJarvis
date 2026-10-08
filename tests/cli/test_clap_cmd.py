"""Desktop shortcut, login autostart and the clap listener's launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from openjarvis.cli import clap_cmd
from openjarvis.cli import desktop_integration as desk


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(desk.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(desk.shutil, "which", lambda name: None)
    return tmp_path


def test_linux_shortcut_runs_gui_with_this_interpreter(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "linux")
    path = desk.create_desktop_shortcut(home / "Desktop")

    text = path.read_text()
    assert path.name == "openjarvis.desktop"
    assert f"Exec={sys.executable} -m openjarvis.cli gui --pause-on-error" in text
    assert "Terminal=true" in text
    assert (home / ".local/share/applications/openjarvis.desktop").exists()

    removed = desk.remove_desktop_shortcut(home / "Desktop")
    assert path in removed and not path.exists()


def test_macos_shortcut_is_executable_command_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "darwin")
    path = desk.create_desktop_shortcut(home / "Desktop")

    assert path.name == "OpenJarvis.command"
    assert path.stat().st_mode & 0o111
    assert "-m openjarvis.cli gui" in path.read_text()


def test_windows_shortcut_falls_back_to_batch_file(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "win32")
    failed = mock.Mock(returncode=1, stderr="no COM")
    with mock.patch.object(desk.subprocess, "run", return_value=failed):
        path = desk.create_desktop_shortcut(home / "Desktop")

    assert path.name == "OpenJarvis.cmd"
    assert "-m openjarvis.cli gui" in path.read_text()


def test_windows_autostart_is_hidden_startup_script(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "win32")
    path = desk.install_autostart(["--claps", "2"])

    assert path.parent.name == "Startup"
    text = path.read_text()
    assert "openjarvis.cli clap --claps 2" in text
    assert ", 0, False" in text  # window style 0: no console
    assert desk.remove_autostart() == path and not path.exists()


def test_macos_autostart_is_launch_agent(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "darwin")
    path = desk.install_autostart(["--sensitivity", "0.7"])

    text = path.read_text()
    assert path.parent == home / "Library" / "LaunchAgents"
    assert "<key>RunAtLoad</key><true/>" in text
    assert "<string>clap</string>" in text and "<string>0.7</string>" in text


def test_linux_autostart_is_xdg_entry(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desk.sys, "platform", "linux")
    path = desk.install_autostart([])

    assert path == home / ".config" / "autostart" / "openjarvis-clap.desktop"
    assert "-m openjarvis.cli clap" in path.read_text()


def test_opener_focuses_running_gui_instead_of_launching() -> None:
    opener = clap_cmd.JarvisOpener(5173, None)
    with (
        mock.patch.object(clap_cmd, "_port_open", return_value=True),
        mock.patch.object(clap_cmd.webbrowser, "open") as open_browser,
        mock.patch.object(clap_cmd.subprocess, "Popen") as popen,
    ):
        assert opener.open() == "opened"
    open_browser.assert_called_once_with("http://127.0.0.1:5173")
    popen.assert_not_called()


def test_opener_launches_gui_once_while_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clap_cmd, "DEFAULT_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(clap_cmd, "_GUI_LOG", tmp_path / "gui.log")
    process = mock.Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    opener = clap_cmd.JarvisOpener(5180, None)
    with (
        mock.patch.object(clap_cmd, "_port_open", return_value=False),
        mock.patch.object(clap_cmd.subprocess, "Popen", return_value=process) as popen,
    ):
        assert opener.open() == "launched"
        assert opener.open() == "starting"

    popen.assert_called_once()
    command = popen.call_args.args[0]
    assert command[-4:] == ["openjarvis.cli", "gui", "--frontend-port", "5180"]
    assert "--api-port" not in command


def test_clap_stop_without_listener(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clap_cmd, "_PID_FILE", tmp_path / "clap.pid")
    result = CliRunner().invoke(clap_cmd.clap, ["--stop"])
    assert result.exit_code == 0
    assert "No listener is running" in result.output


def test_clap_refuses_second_listener(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "clap.pid"
    pid_file.write_text("4242")
    monkeypatch.setattr(clap_cmd, "_PID_FILE", pid_file)
    monkeypatch.setattr(clap_cmd, "process_alive", lambda pid: pid == 4242)
    monkeypatch.setitem(sys.modules, "sounddevice", mock.Mock())
    result = CliRunner().invoke(clap_cmd.clap, [])
    assert result.exit_code != 0
    assert "already running (PID 4242)" in result.output
