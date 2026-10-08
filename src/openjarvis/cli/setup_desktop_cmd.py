"""``jarvis setup-desktop`` — one command to make OpenJarvis a desktop assistant.

Creates the desktop/Start-menu shortcut, prepares the graphical frontend so
the first double-click is quick, and starts the tray icon (with "Hey
Jarvis" and claps) at every login. Safe to run again: every step replaces
what it made before.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import click
from rich.console import Console


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001 - also no display, missing PortAudio, ...
        return False


@click.command("setup-desktop")
@click.option("--no-autostart", is_flag=True, help="Do not start Jarvis at login.")
@click.option("--no-frontend", is_flag=True, help="Skip installing frontend packages.")
def setup_desktop(no_autostart: bool, no_frontend: bool) -> None:
    """Desktop icon + tray icon + 'Hey Jarvis' at login, in one step."""
    from openjarvis.cli import desktop_integration as desk

    console = Console()
    ok = "[green]✓[/green]"
    warn = "[yellow]![/yellow]"

    # 1. Shortcut
    try:
        path = desk.create_desktop_shortcut()
        console.print(f"{ok} Desktop icon: {path}")
    except Exception as exc:  # noqa: BLE001 - report and keep going
        console.print(f"{warn} Could not create the desktop icon: {exc}")

    # 2. Frontend packages, so the first launch does not stop to install.
    if not no_frontend:
        from openjarvis.cli import gui_cmd

        frontend = gui_cmd._frontend_dir()
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if frontend is None:
            console.print(f"{warn} No frontend in this installation (desktop app?)")
        elif npm is None:
            console.print(
                f"{warn} Node.js is missing. Install Node.js 22+ "
                "(Windows: winget install OpenJS.NodeJS.LTS) and run this again."
            )
        else:
            try:
                gui_cmd._ensure_frontend_dependencies(frontend, npm)
                console.print(f"{ok} Graphical interface ready")
            except click.ClickException as exc:
                console.print(f"{warn} {exc.message}")

    # 3. Tray icon (or bare listener) at login.
    has_audio = _can_import("sounddevice")
    has_tray = _can_import("pystray") and _can_import("PIL")
    if no_autostart:
        console.print("  Autostart skipped (--no-autostart).")
    elif has_tray:
        args = ["--frontend-port", "5173"]
        path = desk.install_autostart(args, subcommand="tray")
        console.print(f"{ok} Tray icon starts with your computer: {path}")
        from openjarvis.cli.clap_cmd import _detached_kwargs, _running_listener
        from openjarvis.cli.tray_cmd import _PID_FILE as TRAY_PID
        from openjarvis.core.utils import process_alive

        try:
            tray_pid = int(TRAY_PID.read_text().strip())
        except (OSError, ValueError):
            tray_pid = None
        if tray_pid and process_alive(tray_pid):
            console.print("  The tray icon is already running.")
        else:
            if _running_listener() is not None:
                console.print(
                    "  A separate listener is running; the tray will listen "
                    "after `jarvis listen --stop`."
                )
            subprocess.Popen(
                desk.jarvis_command("tray", *args, windowless=True),
                cwd=desk.project_root(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_detached_kwargs(),
            )
            console.print(f"{ok} Tray icon started (look next to the clock)")
    elif has_audio:
        path = desk.install_autostart(["--frontend-port", "5173"], subcommand="listen")
        console.print(f"{ok} 'Hey Jarvis' listener starts with your computer: {path}")
    else:
        console.print(
            f"{warn} Audio packages missing; run: uv sync --extra desktop "
            "(Linux also: sudo apt install libportaudio2)"
        )

    py = "uv run" if shutil.which("uv") else sys.executable + " -m openjarvis.cli"
    console.print(
        "\n[bold]Next:[/bold]\n"
        "  1. Double-click [bold]OpenJarvis[/bold] on the desktop "
        "(or click the tray icon).\n"
        "  2. In the app: [bold]Hey Jarvis[/bold] (sidebar) → record → "
        "Train my voice.\n"
        "  3. Say “Hey Jarvis… sube el volumen”, or see [bold]Commands[/bold].\n"
        "\n[bold]Optional connections:[/bold]\n"
        f"  Spotify:          {py} jarvis connect spotify\n"
        f"  Google Calendar:  {py} jarvis connect gdrive\n"
        f"  Check everything: {py} jarvis doctor"
    )
