"""``jarvis clap`` and ``jarvis shortcut`` — open OpenJarvis from the desktop."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import click
from rich.console import Console

from openjarvis.cli import desktop_integration as desk
from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.utils import process_alive, terminate_process

_PID_FILE = DEFAULT_CONFIG_DIR / "clap.pid"
_GUI_LOG = DEFAULT_CONFIG_DIR / "gui.log"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _running_listener() -> int | None:
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid != os.getpid() and process_alive(pid):
        return pid
    return None


def _detached_kwargs() -> dict:
    """Let the GUI outlive the listener, without flashing a console window."""
    if sys.platform == "win32":
        return {
            "creationflags": subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
        }
    return {"start_new_session": True}


class JarvisOpener:
    """Open the GUI, or bring it up in the browser when it already runs."""

    def __init__(self, frontend_port: int, api_port: int | None) -> None:
        self.frontend_port = frontend_port
        self.api_port = api_port
        self._launch: subprocess.Popen | None = None

    def open(self) -> str:
        url = f"http://127.0.0.1:{self.frontend_port}"
        if _port_open(self.frontend_port):
            webbrowser.open(url)
            return "opened"
        if self._launch is not None and self._launch.poll() is None:
            return "starting"
        args = ["gui", "--frontend-port", str(self.frontend_port)]
        if self.api_port is not None:
            args += ["--api-port", str(self.api_port)]
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_GUI_LOG, "a") as log:
            self._launch = subprocess.Popen(
                desk.jarvis_command(*args),
                cwd=desk.project_root(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **_detached_kwargs(),
            )
        return "launched"


def _confirm_tone() -> None:
    """Short rising chirp so the user knows the claps were heard."""
    try:
        import numpy as np
        import sounddevice as sd

        rate = 22050
        parts = []
        for freq in (660, 990):
            t = np.arange(int(rate * 0.08)) / rate
            envelope = np.minimum(1, t / 0.01) * np.exp(-t * 25)
            parts.append(0.2 * np.sin(2 * np.pi * freq * t) * envelope)
        sd.play(np.concatenate(parts).astype(np.float32), rate)
    except Exception:  # noqa: BLE001 - a missing beep must never stop listening
        pass


@click.command()
@click.option(
    "--claps",
    default=2,
    show_default=True,
    type=click.IntRange(1, 5),
    help="How many claps open Jarvis.",
)
@click.option(
    "--sensitivity",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Higher hears softer claps (and more false alarms).",
)
@click.option(
    "--frontend-port", default=5173, show_default=True, type=click.IntRange(1, 65535)
)
@click.option("--api-port", default=None, type=click.IntRange(1, 65535))
@click.option(
    "--device", default=None, help="Microphone name or index (default: system mic)."
)
@click.option(
    "--test", is_flag=True, help="Show levels and detected claps; open nothing."
)
@click.option(
    "--autostart",
    "autostart",
    flag_value="on",
    default=None,
    help="Start the clap listener automatically at every login.",
)
@click.option(
    "--no-autostart",
    "autostart",
    flag_value="off",
    help="Remove the login autostart entry.",
)
@click.option("--stop", "stop_", is_flag=True, help="Stop a running clap listener.")
@click.option("--quiet", is_flag=True, help="No confirmation tone.")
def clap(
    claps: int,
    sensitivity: float,
    frontend_port: int,
    api_port: int | None,
    device: str | None,
    test: bool,
    autostart: str | None,
    stop_: bool,
    quiet: bool,
) -> None:
    """Open OpenJarvis by clapping (two claps by default).

    Listens to the microphone in the background. Audio is analysed locally,
    block by block, and never recorded or sent anywhere.
    """
    console = Console(stderr=True)

    if stop_:
        pid = _running_listener()
        if pid is None:
            console.print("[yellow]No clap listener is running.[/yellow]")
            return
        terminate_process(pid)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]Clap listener stopped[/green] (PID {pid}).")
        return

    if autostart == "off":
        removed = desk.remove_autostart()
        console.print(
            f"[green]Autostart removed:[/green] {removed}"
            if removed
            else "[yellow]No clap autostart entry was installed.[/yellow]"
        )
        return

    try:
        import sounddevice  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "Clap detection needs the sounddevice package. Run it with the "
            "desktop extra: uv run --extra desktop jarvis clap"
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            "The PortAudio audio library is missing. On Debian/Ubuntu install it "
            "with: sudo apt install libportaudio2  (Fedora: sudo dnf install "
            "portaudio). Windows and macOS include it."
        ) from exc

    if autostart == "on":
        clap_args = ["--claps", str(claps), "--sensitivity", str(sensitivity)]
        clap_args += ["--frontend-port", str(frontend_port)]
        if api_port is not None:
            clap_args += ["--api-port", str(api_port)]
        if device is not None:
            clap_args += ["--device", device]
        if quiet:
            clap_args.append("--quiet")
        path = desk.install_autostart(clap_args)
        console.print(
            f"[green]The clap listener will start at every login.[/green]\n  {path}\n"
            "Remove it with: jarvis clap --no-autostart"
        )
        if _running_listener() is None:
            console.print("Starting it now in the background...")
            subprocess.Popen(
                desk.jarvis_command("clap", *clap_args, windowless=True),
                cwd=desk.project_root(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_detached_kwargs(),
            )
        return

    from openjarvis.speech.clap import (
        ClapDetector,
        listen_for_claps,
        sensitivity_to_peak,
    )

    if not test:
        existing = _running_listener()
        if existing is not None:
            raise click.ClickException(
                f"A clap listener is already running (PID {existing}). "
                "Stop it with: jarvis clap --stop"
            )
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))

    detector = ClapDetector(
        claps_required=claps, min_peak=sensitivity_to_peak(sensitivity)
    )
    opener = JarvisOpener(frontend_port, api_port)
    mic: int | str | None = int(device) if device and device.isdigit() else device
    last_draw = 0.0
    seen = 0

    def on_pattern() -> None:
        if not quiet:
            _confirm_tone()
        if test:
            console.print(f"\n[green]{claps} claps detected[/green]")
            return
        outcome = opener.open()
        stamp = time.strftime("%H:%M:%S")
        message = {
            "opened": "opening OpenJarvis in the browser",
            "launched": f"starting OpenJarvis (log: {_GUI_LOG})",
            "starting": "OpenJarvis is still starting",
        }[outcome]
        console.print(f"[{stamp}] {claps} claps heard: {message}")

    def on_block(peak: float, rms: float, det: ClapDetector) -> None:
        nonlocal last_draw, seen
        if not test:
            return
        if det.claps_so_far > seen:
            console.print(f"\n  clap {det.claps_so_far}/{claps}")
        seen = det.claps_so_far
        now = time.monotonic()
        if now - last_draw >= 0.1:
            last_draw = now
            bar = "█" * min(40, int(peak * 40))
            mark = "|" if peak >= det.min_peak else " "
            sys.stderr.write(f"\r  level {bar:<40}{mark} peak {peak:4.2f} ")
            sys.stderr.flush()

    console.print(
        f"[bold]Listening for {claps} claps[/bold] "
        f"(sensitivity {sensitivity:.2f}). Ctrl+C to stop."
        + (
            "\nTest mode: the bar shows the mic level; | marks the clap threshold."
            if test
            else ""
        )
    )
    try:
        listen_for_claps(on_pattern, detector, on_block=on_block, device=mic)
    except RuntimeError as exc:
        raise click.ClickException(
            f"{exc}\nCheck that a microphone is connected and that this app may "
            "use it (Windows: Settings → Privacy & security → Microphone → "
            "'Let desktop apps access your microphone'; macOS: System Settings → "
            "Privacy & Security → Microphone)."
        ) from exc
    except KeyboardInterrupt:
        pass
    finally:
        if not test and _PID_FILE.exists():
            try:
                if int(_PID_FILE.read_text().strip()) == os.getpid():
                    _PID_FILE.unlink()
            except (OSError, ValueError):
                pass


@click.command()
@click.option("--remove", is_flag=True, help="Remove the desktop shortcut.")
@click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to put it (default: your Desktop).",
)
def shortcut(remove: bool, directory: Path | None) -> None:
    """Put an OpenJarvis icon on the desktop that opens the graphical mode."""
    console = Console(stderr=True)
    if remove:
        removed = desk.remove_desktop_shortcut(directory)
        if removed:
            for path in removed:
                console.print(f"[green]Removed[/green] {path}")
        else:
            console.print("[yellow]No OpenJarvis shortcut found.[/yellow]")
        return
    path = desk.create_desktop_shortcut(directory)
    console.print(
        f"[green]Shortcut created:[/green] {path}\nDouble-click it to open OpenJarvis."
    )
