"""``jarvis gui`` — start and open the local graphical interface."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import click
from click.core import ParameterSource
from rich.console import Console

from openjarvis.cli import daemon_cmd

# Seconds to wait for the API's /health to answer after `jarvis start`. The
# daemon returns as soon as it has spawned the server, which still has to
# import the stack and reach the inference engine.
_API_READY_TIMEOUT = 60.0
# A cold Vite start (first dependency pre-bundle, slow disks, Windows AV
# scanning node_modules) regularly takes longer than 20s.
_FRONTEND_READY_TIMEOUT = 60.0
_LOG_TAIL_LINES = 20


def _frontend_dir() -> Path | None:
    """Find the source checkout's frontend directory."""
    configured = os.environ.get("OPENJARVIS_FRONTEND_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).resolve().parents[3] / "frontend")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "package.json").is_file():
            return candidate
    return None


def _check_frontend_port(port: int) -> None:
    """Reject a port already bound by another local application."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise click.ClickException(
            f"Frontend port {port} is unavailable. Choose another with --frontend-port."
        ) from exc


def _wait_for_port(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    timeout: float = _FRONTEND_READY_TIMEOUT,
) -> bool:
    """Wait for this launch's frontend process to listen on the requested port."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return process.poll() is None
        except OSError:
            time.sleep(0.2)
    return False


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.strip().lstrip("v").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _pinned_npm(frontend: Path) -> str | None:
    """Return the npm version the frontend pins in ``packageManager``."""
    try:
        manifest = json.loads((frontend / "package.json").read_text())
    except (OSError, ValueError):
        return None
    spec = manifest.get("packageManager") if isinstance(manifest, dict) else None
    if isinstance(spec, str) and spec.startswith("npm@"):
        return spec.split("@", 1)[1].split("+", 1)[0]
    return None


def _install_command(frontend: Path, npm: str) -> list[str]:
    """Install with the pinned npm when the local one is too old.

    The frontend sets ``engine-strict`` and requires the npm it pins, while
    Node 22 still ships npm 10, so a plain ``npm install`` fails with
    EBADENGINE on most machines. ``npm exec`` fetches the pinned npm for this
    one install; ``npm run dev`` afterwards works with the older npm.
    """
    install = ["install", "--no-audit", "--no-fund"]
    pinned = _pinned_npm(frontend)
    if pinned is None:
        return [npm, *install]
    try:
        found = subprocess.run(
            [npm, "--version"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        found = ""
    if found and _version_tuple(found) >= _version_tuple(pinned):
        return [npm, *install]
    click.echo(
        f"npm {found.strip() or '(unknown)'} is older than the npm {pinned} this "
        f"frontend requires; using npm {pinned} for the install. "
        f"Upgrade with `npm install -g npm@{pinned}` to skip this step.",
        err=True,
    )
    return [npm, "exec", "--yes", "--", f"npm@{pinned}", *install]


def _ensure_frontend_dependencies(frontend: Path, npm: str) -> None:
    """Install frontend dependencies when this checkout has not been bootstrapped."""
    vite = (
        frontend
        / "node_modules"
        / ".bin"
        / ("vite.cmd" if sys.platform == "win32" else "vite")
    )
    if vite.exists():
        return
    click.echo("Installing graphical frontend dependencies...", err=True)
    result = subprocess.run(
        _install_command(frontend, npm),
        cwd=frontend,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(
            "Could not install frontend dependencies. "
            "Run `npm install` in the frontend directory."
        )


def _probe_api(port: int, timeout: float = 1.0) -> str | None:
    """Return ``"ok"``, ``"engine-down"`` or ``None`` when nothing answers.

    ``/health`` answers 503 when the server is up but its inference engine is
    not, which is a different problem (start Ollama) from the server being
    absent, so the two are reported separately.
    """
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=timeout
        ) as response:
            return "ok" if response.status == 200 else "engine-down"
    except urllib.error.HTTPError as exc:
        return "engine-down" if exc.code == 503 else None
    except (OSError, ValueError):
        return None


def _log_tail(lines: int = _LOG_TAIL_LINES) -> str:
    """Return the end of the daemon log, for errors the user must act on."""
    try:
        text = daemon_cmd._LOG_FILE.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _startup_hint(log_tail: str) -> str:
    """Translate the most common startup failures into the next step to take."""
    if "No inference engine available" in log_tail:
        return (
            "\nNo inference engine is running. Start Ollama (`ollama serve`) or "
            "configure a cloud engine, then run `jarvis gui` again. "
            "`jarvis doctor` checks the whole setup."
        )
    if "address already in use" in log_tail.lower():
        return (
            "\nThe API port is taken by another program. Pick another with --api-port."
        )
    return "\nRun `jarvis doctor` to diagnose the setup."


def _startup_failure(message: str) -> click.ClickException:
    """Build an error carrying the daemon log tail and the likely fix."""
    tail = _log_tail()
    if tail:
        message += f"\nLast lines of {daemon_cmd._LOG_FILE}:\n{tail}"
    return click.ClickException(message + _startup_hint(tail))


def _start_command(api_port: int) -> list[str]:
    """Build the `jarvis start` invocation that has the server dependencies."""
    uv = shutil.which("uv")
    if uv is not None:
        return [
            uv,
            "run",
            "--extra",
            "desktop",
            "jarvis",
            "start",
            "--port",
            str(api_port),
        ]
    # Installs without uv (pip, pipx) can still serve when FastAPI is present.
    if importlib.util.find_spec("fastapi") is not None:
        return [
            sys.executable,
            "-m",
            "openjarvis.cli",
            "start",
            "--port",
            str(api_port),
        ]
    raise click.ClickException(
        "uv is required to start the API with desktop dependencies. "
        "Install uv or use --no-server with an already-running API."
    )


def _wait_for_api(console: Console, port: int, timeout: float) -> str | None:
    """Wait for the API to answer, failing fast when the daemon process dies."""
    deadline = time.monotonic() + timeout
    with console.status(f"Waiting for the OpenJarvis API on port {port}..."):
        while time.monotonic() < deadline:
            status = _probe_api(port)
            if status is not None:
                return status
            if daemon_cmd._read_pid() is None:
                raise _startup_failure(
                    "The OpenJarvis API server exited during startup."
                )
            time.sleep(0.5)
    return None


def _ensure_api(
    console: Console, project_root: Path, api_port: int, port_is_default: bool
) -> int:
    """Reuse a running API or start one; return the port the frontend should proxy.

    Re-running `jarvis gui` used to fail outright: the daemon from the previous
    run was still alive, so `jarvis start` exited non-zero.
    """
    if _probe_api(api_port) is not None:
        console.print(f"Using the OpenJarvis API already running on port {api_port}.")
        return api_port

    pid = daemon_cmd._read_pid()
    if pid is not None:
        state = daemon_cmd._read_state()
        running_port = state.get("port") if state.get("pid") == pid else None
        if running_port is None:
            raise click.ClickException(
                f"An OpenJarvis server is already running (PID {pid}) but its port "
                "is unknown. Stop it with `jarvis stop`, or pass its port with "
                "--api-port."
            )
        if running_port != api_port and not port_is_default:
            raise click.ClickException(
                f"An OpenJarvis server is already running on port {running_port} "
                f"(PID {pid}). Re-run with --api-port {running_port}, or stop it "
                "with `jarvis stop`."
            )
        console.print(
            f"Using the OpenJarvis server already running on port {running_port} "
            f"(PID {pid})."
        )
        return running_port

    server = subprocess.run(_start_command(api_port), cwd=project_root, check=False)
    if server.returncode != 0:
        raise _startup_failure("Could not start the OpenJarvis API server.")
    return api_port


def _report_api_status(console: Console, status: str | None, port: int) -> None:
    if status == "ok":
        console.print(f"[green]OpenJarvis API is ready[/green] on port {port}.")
    elif status == "engine-down":
        console.print(
            "[yellow]The API is up but cannot reach its inference engine.[/yellow] "
            "Start it (for Ollama: `ollama serve`) and run `jarvis doctor`; "
            "the page reconnects on its own."
        )
    else:
        console.print(
            f"[yellow]The API on port {port} is still starting.[/yellow] "
            "The page will connect once it is ready. Logs: "
            f"{daemon_cmd._LOG_FILE}"
        )


@click.command()
@click.option(
    "--frontend-port", default=5173, show_default=True, type=click.IntRange(1, 65535)
)
@click.option(
    "--api-port", default=8000, show_default=True, type=click.IntRange(1, 65535)
)
@click.option("--no-server", is_flag=True, help="Do not start the API server.")
@click.option("--no-browser", is_flag=True, help="Only start the frontend.")
@click.option(
    "--pause-on-error",
    is_flag=True,
    hidden=True,
    help="Keep the window open on failure (used by the desktop shortcut).",
)
@click.pass_context
def gui(
    ctx: click.Context,
    frontend_port: int,
    api_port: int,
    no_server: bool,
    no_browser: bool,
    pause_on_error: bool,
) -> None:
    """Start the browser-based graphical mode in the default browser.

    Re-running it reuses an OpenJarvis server that is already running, and
    when the graphical mode is already open it just opens the page. This
    command is intended for source checkouts. For an installed desktop
    application, launch OpenJarvis from the operating system menu instead.
    """
    try:
        _run_gui(ctx, frontend_port, api_port, no_server, no_browser)
    except click.ClickException as exc:
        if not pause_on_error:
            raise
        # A window opened from a desktop icon closes as soon as the process
        # exits, which would hide the reason; keep it up until a keypress.
        exc.show()
        click.pause("\nPress any key to close this window...")
        sys.exit(exc.exit_code)


def _frontend_already_running(port: int) -> bool:
    """Is our own graphical frontend already serving on *port*?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as r:
            return "OpenJarvis" in r.read(65536).decode("utf-8", "replace")
    except (OSError, ValueError):
        return False


def _run_gui(
    ctx: click.Context,
    frontend_port: int,
    api_port: int,
    no_server: bool,
    no_browser: bool,
) -> None:
    console = Console(stderr=True)
    frontend = _frontend_dir()
    if frontend is None:
        raise click.ClickException(
            "The graphical frontend is not available in this installation. "
            "Download the OpenJarvis desktop app or run this command "
            "from a source checkout."
        )
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        raise click.ClickException(
            "Node.js/npm is required for graphical mode. Install Node.js 22 or newer."
        )
    if _frontend_already_running(frontend_port):
        url = f"http://127.0.0.1:{frontend_port}"
        console.print(f"[green]OpenJarvis is already open:[/green] {url}")
        if not no_browser:
            webbrowser.open(url)
        return
    _check_frontend_port(frontend_port)
    _ensure_frontend_dependencies(frontend, npm)

    if not no_server:
        port_is_default = (
            ctx.get_parameter_source("api_port") == ParameterSource.DEFAULT
        )
        api_port = _ensure_api(console, frontend.parent, api_port, port_is_default)
        _report_api_status(
            console, _wait_for_api(console, api_port, _API_READY_TIMEOUT), api_port
        )

    env = os.environ.copy()
    # Let browser requests use Vite's same-origin proxy at any frontend port.
    # VITE_API_URL is exposed to browser code, so clear an inherited override.
    env["VITE_API_URL"] = ""
    env["OPENJARVIS_VITE_PROXY_TARGET"] = f"http://127.0.0.1:{api_port}"
    process = subprocess.Popen(
        [
            npm,
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(frontend_port),
            "--strictPort",
        ],
        cwd=frontend,
        env=env,
    )
    if not _wait_for_port(process, "127.0.0.1", frontend_port):
        process.terminate()
        process.wait()
        raise click.ClickException(
            "The graphical frontend did not start on the requested port. "
            "See the Vite output above for the cause."
        )

    url = f"http://127.0.0.1:{frontend_port}"
    console.print(f"[green]OpenJarvis graphical mode is ready:[/green] {url}")
    if not no_browser:
        webbrowser.open(url)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
    if not no_server:
        console.print(
            "Graphical frontend stopped. The API server keeps running in the "
            "background; stop it with `jarvis stop`."
        )
