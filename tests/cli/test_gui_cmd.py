"""Launch behavior for the source-checkout graphical command."""

from __future__ import annotations

import os
import socket
import subprocess
import urllib.error
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from openjarvis.cli import gui_cmd


def test_gui_custom_ports_use_project_root_and_same_origin_proxy(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    process = mock.Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.wait.return_value = 0

    with (
        mock.patch.object(gui_cmd, "_frontend_dir", return_value=frontend),
        mock.patch.object(gui_cmd, "_check_frontend_port") as check_port,
        mock.patch.object(gui_cmd, "_frontend_already_running", return_value=False),
        mock.patch.object(gui_cmd, "_ensure_frontend_dependencies") as ensure_deps,
        mock.patch.object(
            gui_cmd.shutil, "which", side_effect=lambda name: f"/bin/{name}"
        ),
        mock.patch.object(
            gui_cmd.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run,
        mock.patch.object(gui_cmd, "_api_kind", return_value=None),
        mock.patch.object(gui_cmd, "_probe_api", return_value="ok"),
        mock.patch.object(gui_cmd.daemon_cmd, "_read_pid", return_value=None),
        mock.patch.object(gui_cmd.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(
            gui_cmd, "_wait_for_port", return_value=True
        ) as wait_for_port,
        mock.patch.object(gui_cmd.webbrowser, "open") as open_browser,
        mock.patch.dict(os.environ, {"VITE_API_URL": "http://stale.example"}),
    ):
        result = CliRunner().invoke(
            gui_cmd.gui,
            ["--frontend-port", "5180", "--api-port", "8123", "--no-browser"],
        )

    assert result.exit_code == 0, result.output
    check_port.assert_called_once_with(5180)
    ensure_deps.assert_called_once_with(frontend, "/bin/npm")
    run.assert_called_once_with(
        [
            "/bin/uv",
            "run",
            "--extra",
            "desktop",
            "jarvis",
            "start",
            "--port",
            "8123",
        ],
        cwd=tmp_path,
        check=False,
    )
    args, kwargs = popen.call_args
    assert args[0] == [
        "/bin/npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "5180",
        "--strictPort",
    ]
    assert kwargs["cwd"] == frontend
    assert kwargs["env"]["OPENJARVIS_VITE_PROXY_TARGET"] == "http://127.0.0.1:8123"
    assert kwargs["env"]["VITE_API_URL"] == ""
    wait_for_port.assert_called_once_with(process, "127.0.0.1", 5180)
    open_browser.assert_not_called()
    process.wait.assert_called_once_with()


def test_gui_rejects_occupied_frontend_port_before_launch(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with (
            mock.patch.object(gui_cmd, "_frontend_dir", return_value=frontend),
            mock.patch.object(gui_cmd.shutil, "which", return_value="/bin/npm"),
            mock.patch.object(gui_cmd, "_ensure_frontend_dependencies") as ensure_deps,
            mock.patch.object(gui_cmd.subprocess, "Popen") as popen,
            mock.patch.object(gui_cmd.subprocess, "run") as run,
        ):
            result = CliRunner().invoke(
                gui_cmd.gui,
                ["--frontend-port", str(port), "--no-server", "--no-browser"],
            )

    assert result.exit_code != 0
    assert f"Frontend port {port} is unavailable" in result.output
    ensure_deps.assert_not_called()
    run.assert_not_called()
    popen.assert_not_called()


def test_wait_for_port_stops_when_vite_exits() -> None:
    process = mock.Mock(spec=subprocess.Popen)
    process.poll.return_value = 1
    with mock.patch.object(gui_cmd.socket, "create_connection") as connect:
        assert not gui_cmd._wait_for_port(process, "127.0.0.1", 5173)
    connect.assert_not_called()


def _launch(
    tmp_path: Path,
    args: list[str],
    *,
    probe: list[str | None],
    pid: int | None = None,
    state: dict | None = None,
    kind: str | None = "auto",
):
    """Run `jarvis gui` with the frontend and API stubbed out."""
    frontend = tmp_path / "frontend"
    frontend.mkdir(exist_ok=True)
    process = mock.Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.wait.return_value = 0
    with (
        mock.patch.object(gui_cmd, "_frontend_dir", return_value=frontend),
        mock.patch.object(gui_cmd, "_check_frontend_port"),
        mock.patch.object(gui_cmd, "_frontend_already_running", return_value=False),
        mock.patch.object(gui_cmd, "_ensure_frontend_dependencies"),
        mock.patch.object(
            gui_cmd.shutil, "which", side_effect=lambda name: f"/bin/{name}"
        ),
        mock.patch.object(
            gui_cmd.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run,
        mock.patch.object(gui_cmd, "_probe_api", side_effect=probe[1:] or [None]),
        mock.patch.object(
            gui_cmd,
            "_api_kind",
            return_value=(
                ("current" if probe[0] else None) if kind == "auto" else kind
            ),
        ),
        mock.patch.object(gui_cmd, "_windows_task_exists", return_value=False),
        mock.patch.object(gui_cmd.daemon_cmd, "_read_pid", return_value=pid),
        mock.patch.object(gui_cmd.daemon_cmd, "_read_state", return_value=state or {}),
        mock.patch.object(gui_cmd.time, "sleep"),
        mock.patch.object(gui_cmd.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(gui_cmd, "_wait_for_port", return_value=True),
        mock.patch.object(gui_cmd.webbrowser, "open"),
    ):
        result = CliRunner().invoke(gui_cmd.gui, [*args, "--no-browser"])
    return result, run, popen


def test_gui_reuses_api_that_is_already_running(tmp_path: Path) -> None:
    result, run, popen = _launch(tmp_path, [], probe=["ok", "ok"])

    assert result.exit_code == 0, result.output
    assert "already running on port 8000" in result.output
    run.assert_not_called()
    env = popen.call_args.kwargs["env"]
    assert env["OPENJARVIS_VITE_PROXY_TARGET"] == "http://127.0.0.1:8000"


def test_gui_follows_daemon_started_on_another_port(tmp_path: Path) -> None:
    result, run, popen = _launch(
        tmp_path,
        [],
        probe=[None, "ok"],
        pid=4242,
        state={"pid": 4242, "host": "127.0.0.1", "port": 8123},
    )

    assert result.exit_code == 0, result.output
    run.assert_not_called()
    env = popen.call_args.kwargs["env"]
    assert env["OPENJARVIS_VITE_PROXY_TARGET"] == "http://127.0.0.1:8123"


def test_gui_explicit_port_conflicting_with_daemon_explains_fix(
    tmp_path: Path,
) -> None:
    result, run, popen = _launch(
        tmp_path,
        ["--api-port", "9000"],
        probe=[None],
        pid=4242,
        state={"pid": 4242, "host": "127.0.0.1", "port": 8123},
    )

    assert result.exit_code != 0
    assert "--api-port 8123" in result.output
    run.assert_not_called()
    popen.assert_not_called()


def test_gui_reports_server_crash_with_log_tail(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("booting\nNo inference engine available.\n")
    with mock.patch.object(gui_cmd.daemon_cmd, "_LOG_FILE", log):
        result, run, popen = _launch(tmp_path, [], probe=[None, None])

    assert result.exit_code != 0
    assert "exited during startup" in result.output
    assert "No inference engine available" in result.output
    assert "ollama serve" in result.output
    run.assert_called_once()
    popen.assert_not_called()


def test_gui_opens_when_engine_is_down_and_says_why(tmp_path: Path) -> None:
    result, _run, popen = _launch(tmp_path, [], probe=["engine-down", "engine-down"])

    assert result.exit_code == 0, result.output
    assert "cannot reach its inference engine" in result.output
    popen.assert_called_once()


def test_install_uses_pinned_npm_when_local_npm_is_too_old(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"packageManager": "npm@11.19.0"}')
    with mock.patch.object(
        gui_cmd.subprocess, "run", return_value=mock.Mock(stdout="10.9.4\n")
    ):
        command = gui_cmd._install_command(tmp_path, "/bin/npm")

    assert command[:5] == ["/bin/npm", "exec", "--yes", "--", "npm@11.19.0"]
    assert command[5] == "install"


def test_install_uses_local_npm_when_new_enough(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"packageManager": "npm@11.19.0"}')
    with mock.patch.object(
        gui_cmd.subprocess, "run", return_value=mock.Mock(stdout="11.20.1\n")
    ):
        command = gui_cmd._install_command(tmp_path, "/bin/npm")

    assert command == ["/bin/npm", "install", "--no-audit", "--no-fund"]


def test_probe_api_distinguishes_engine_down_from_absent() -> None:
    error = urllib.error.HTTPError("http://x/health", 503, "down", {}, None)
    with mock.patch.object(gui_cmd.urllib.request, "urlopen", side_effect=error):
        assert gui_cmd._probe_api(8000) == "engine-down"
    with mock.patch.object(
        gui_cmd.urllib.request, "urlopen", side_effect=ConnectionRefusedError()
    ):
        assert gui_cmd._probe_api(8000) is None


def test_gui_already_open_just_opens_the_page(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    with (
        mock.patch.object(gui_cmd, "_frontend_dir", return_value=frontend),
        mock.patch.object(gui_cmd.shutil, "which", return_value="/bin/npm"),
        mock.patch.object(gui_cmd, "_frontend_already_running", return_value=True),
        mock.patch.object(gui_cmd.webbrowser, "open") as open_browser,
        mock.patch.object(gui_cmd.subprocess, "Popen") as popen,
    ):
        result = CliRunner().invoke(gui_cmd.gui, [])

    assert result.exit_code == 0, result.output
    assert "already open" in result.output
    open_browser.assert_called_once_with("http://127.0.0.1:5173")
    popen.assert_not_called()


def test_pause_on_error_keeps_the_window_open() -> None:
    with (
        mock.patch.object(gui_cmd, "_frontend_dir", return_value=None),
        mock.patch.object(gui_cmd.click, "pause") as pause,
    ):
        result = CliRunner().invoke(gui_cmd.gui, ["--pause-on-error"])

    assert result.exit_code == 1
    assert "not available in this installation" in result.output
    pause.assert_called_once()


def test_stale_server_is_restarted_with_new_code(tmp_path: Path) -> None:
    with mock.patch.object(gui_cmd, "_restart_stale_server", return_value=False) as r:
        result, run, popen = _launch(tmp_path, [], probe=[None, "ok"], kind="stale")
    assert result.exit_code == 0, result.output
    r.assert_called_once()
    run.assert_called_once()  # a fresh server was started


def test_foreign_program_on_api_port_is_reported(tmp_path: Path) -> None:
    result, run, popen = _launch(tmp_path, [], probe=[None], kind="foreign")
    assert result.exit_code != 0
    assert "another program" in result.output
    run.assert_not_called()


def test_api_kind_tells_old_servers_apart() -> None:
    def statuses(mapping):
        return lambda port, path, timeout=2.0: mapping.get(path)

    with mock.patch.object(gui_cmd, "_probe_api", return_value="ok"):
        with mock.patch.object(
            gui_cmd, "_http_status", statuses({"/v1/voice/status": 200})
        ):
            assert gui_cmd._api_kind(8000) == "current"
        with mock.patch.object(
            gui_cmd, "_http_status", statuses({"/v1/voice/status": 401})
        ):
            assert gui_cmd._api_kind(8000) == "current"
        with mock.patch.object(
            gui_cmd,
            "_http_status",
            statuses({"/v1/voice/status": 404, "/v1/info": 200}),
        ):
            assert gui_cmd._api_kind(8000) == "stale"
        with mock.patch.object(
            gui_cmd,
            "_http_status",
            statuses({"/v1/voice/status": 404, "/v1/info": 404}),
        ):
            assert gui_cmd._api_kind(8000) == "foreign"
    with mock.patch.object(gui_cmd, "_probe_api", return_value=None):
        assert gui_cmd._api_kind(8000) is None


def test_listening_pids_parses_windows_netstat(monkeypatch) -> None:
    netstat = (
        "  Proto  Local Address          Foreign Address        State           PID\n"
        "  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4321\n"
        "  TCP    127.0.0.1:18000        0.0.0.0:0              LISTENING       99\n"
        "  TCP    127.0.0.1:8000         127.0.0.1:5000         ESTABLISHED     4321\n"
    )
    monkeypatch.setattr(gui_cmd.sys, "platform", "win32")
    with mock.patch.object(
        gui_cmd.subprocess, "run", return_value=mock.Mock(stdout=netstat)
    ):
        assert gui_cmd._listening_pids(8000) == [4321]
