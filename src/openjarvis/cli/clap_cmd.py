"""``jarvis listen`` (alias ``clap``) and ``jarvis shortcut``: talk to Jarvis."""

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


def _api_base(api_port: int | None) -> str:
    """Where the API listens: explicit port, the daemon's port, or 8000."""
    if api_port is None:
        from openjarvis.cli import daemon_cmd

        state = daemon_cmd._read_state()
        api_port = state.get("port") if isinstance(state.get("port"), int) else 8000
    return f"http://127.0.0.1:{api_port}"


def _wake_detector(sensitivity: float):
    from openjarvis.speech import wakeword

    profile = wakeword.WakeWordProfile.load()
    if profile is None or not profile.templates:
        return None
    return wakeword.WakeWordDetector(profile, sensitivity=sensitivity)


@click.command()
@click.option(
    "--claps",
    default=2,
    show_default=True,
    type=click.IntRange(1, 5),
    help="How many claps trigger Jarvis.",
)
@click.option(
    "--sensitivity",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Clap sensitivity: higher hears softer claps (and more false alarms).",
)
@click.option(
    "--wake-sensitivity",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="'Hey Jarvis' sensitivity: higher accepts looser matches.",
)
@click.option("--no-claps", is_flag=True, help="Ignore claps.")
@click.option("--no-wake", is_flag=True, help="Ignore the 'Hey Jarvis' wake word.")
@click.option(
    "--open-only",
    is_flag=True,
    help="Only open OpenJarvis; do not listen for a request afterwards.",
)
@click.option(
    "--no-speak", is_flag=True, help="Show replies in the page without speaking."
)
@click.option(
    "--frontend-port", default=5173, show_default=True, type=click.IntRange(1, 65535)
)
@click.option("--api-port", default=None, type=click.IntRange(1, 65535))
@click.option(
    "--device", default=None, help="Microphone name or index (default: system mic)."
)
@click.option(
    "--test", is_flag=True, help="Show live levels and detections; trigger nothing."
)
@click.option(
    "--autostart",
    "autostart",
    flag_value="on",
    default=None,
    help="Start the listener automatically at every login.",
)
@click.option(
    "--no-autostart",
    "autostart",
    flag_value="off",
    help="Remove the login autostart entry.",
)
@click.option("--stop", "stop_", is_flag=True, help="Stop a running listener.")
@click.option("--quiet", is_flag=True, help="No cue tones.")
def listen(
    claps: int,
    sensitivity: float,
    wake_sensitivity: float,
    no_claps: bool,
    no_wake: bool,
    open_only: bool,
    no_speak: bool,
    frontend_port: int,
    api_port: int | None,
    device: str | None,
    test: bool,
    autostart: str | None,
    stop_: bool,
    quiet: bool,
) -> None:
    """Hands-free Jarvis: say "Hey Jarvis" or clap twice, then talk.

    Opens OpenJarvis if needed, plays a chirp, records your request until you
    pause, sends it to Jarvis and speaks the answer. Train the wake word with
    your voice first (Settings -> Hey Jarvis in the app, or `jarvis wake
    train`). Audio is analysed locally; only your request is transcribed.
    """
    console = Console(stderr=True)

    if stop_:
        pid = _running_listener()
        if pid is None:
            console.print("[yellow]No listener is running.[/yellow]")
            return
        terminate_process(pid)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]Listener stopped[/green] (PID {pid}).")
        return

    if autostart == "off":
        removed = desk.remove_autostart()
        console.print(
            f"[green]Autostart removed:[/green] {removed}"
            if removed
            else "[yellow]No listener autostart entry was installed.[/yellow]"
        )
        return

    try:
        import sounddevice  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "Listening needs the sounddevice package. Run it with the "
            "desktop extra: uv run --extra desktop jarvis listen"
        ) from exc
    except OSError as exc:
        raise click.ClickException(
            "The PortAudio audio library is missing. On Debian/Ubuntu install it "
            "with: sudo apt install libportaudio2  (Fedora: sudo dnf install "
            "portaudio). Windows and macOS include it."
        ) from exc

    if autostart == "on":
        args = ["--claps", str(claps), "--sensitivity", str(sensitivity)]
        args += ["--wake-sensitivity", str(wake_sensitivity)]
        args += ["--frontend-port", str(frontend_port)]
        for flag, enabled in (
            ("--no-claps", no_claps),
            ("--no-wake", no_wake),
            ("--open-only", open_only),
            ("--no-speak", no_speak),
            ("--quiet", quiet),
        ):
            if enabled:
                args.append(flag)
        if api_port is not None:
            args += ["--api-port", str(api_port)]
        if device is not None:
            args += ["--device", device]
        path = desk.install_autostart(args)
        console.print(
            f"[green]The listener will start at every login.[/green]\n  {path}\n"
            "Remove it with: jarvis listen --no-autostart"
        )
        running = _running_listener()
        if running is not None:
            console.print(
                f"A listener is already running (PID {running}); restart it with "
                "`jarvis listen --stop` and log in again to apply new options."
            )
        else:
            console.print("Starting it now in the background...")
            subprocess.Popen(
                desk.jarvis_command("listen", *args, windowless=True),
                cwd=desk.project_root(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_detached_kwargs(),
            )
        return

    from openjarvis.speech import wakeword
    from openjarvis.speech.clap import ClapDetector, sensitivity_to_peak
    from openjarvis.speech.listener import (
        ApiClient,
        ListenerOptions,
        Speaker,
        VoiceListener,
        chirp,
        microphone_blocks,
        resolve_api_key,
    )

    wake = None if no_wake else _wake_detector(wake_sensitivity)
    if not no_wake and wake is None:
        console.print(
            "[yellow]'Hey Jarvis' is not trained yet[/yellow], so only claps work. "
            "Train it in the app (Settings -> Hey Jarvis -> Train my voice) or "
            "with `jarvis wake train`."
        )
    if no_claps and wake is None:
        raise click.ClickException(
            "Nothing to listen for: claps are off and no wake word is trained."
        )

    if not test:
        existing = _running_listener()
        if existing is not None:
            raise click.ClickException(
                f"A listener is already running (PID {existing}). "
                "Stop it with: jarvis listen --stop"
            )
        DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))

    clap_detector = (
        None
        if no_claps
        else ClapDetector(
            claps_required=claps, min_peak=sensitivity_to_peak(sensitivity)
        )
    )
    mic: int | str | None = int(device) if device and device.isdigit() else device
    triggers = []
    if clap_detector is not None:
        triggers.append(f"{claps} claps")
    if wake is not None:
        triggers.append(f"'{wake.profile.phrase}'")
    console.print(
        f"[bold]Listening for {' or '.join(triggers)}.[/bold] Ctrl+C to stop."
    )

    try:
        if test:
            _run_test(console, mic, clap_detector, wake, claps, quiet, chirp)
            return
        listener = VoiceListener(
            ApiClient(_api_base(api_port), resolve_api_key()),
            options=ListenerOptions(
                claps=clap_detector is not None,
                wake=wake is not None or not no_wake,
                converse=not open_only,
                speak_replies=not no_speak,
                sounds=not quiet,
            ),
            clap_detector=clap_detector,
            wake_detector=wake,
            open_ui=JarvisOpener(frontend_port, api_port).open,
            speaker=Speaker(),
            log=lambda message: console.print(message, highlight=False),
            profile_loader=None
            if no_wake
            else lambda: _wake_detector(wake_sensitivity),
            profile_path=None if no_wake else wakeword.wakeword_dir() / "profile.json",
        )
        try:
            for block in microphone_blocks(lambda: False, device=mic):
                listener.process(block)
        finally:
            listener.close()
    except RuntimeError as exc:
        raise click.ClickException(
            f"{exc}\nCheck that a microphone is connected and that this app may "
            "use it (Windows: Settings -> Privacy & security -> Microphone -> "
            "'Let desktop apps access your microphone'; macOS: System Settings -> "
            "Privacy & Security -> Microphone)."
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


def _run_test(console, mic, clap_detector, wake, claps, quiet, chirp) -> None:
    """Live meter: mic level, clap count and wake-word distance."""
    import numpy as np

    from openjarvis.speech.listener import Speaker, describe_distance, microphone_blocks

    speaker = Speaker()
    console.print(
        "Test mode: nothing opens. 'level' is the mic; 'wake' is how close the "
        "last sound was to your voice profile (* = match)."
    )
    last_draw = 0.0
    seen = 0
    for block in microphone_blocks(lambda: False, device=mic):
        peak = float(np.max(np.abs(block))) if block.size else 0.0
        rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
        if clap_detector is not None:
            if clap_detector.process(peak, rms):
                console.print(f"\n[green]{claps} claps detected[/green]")
                if not quiet:
                    speaker.play(chirp("listen"), 22050)
            elif clap_detector.claps_so_far > seen:
                console.print(f"\n  clap {clap_detector.claps_so_far}/{claps}")
            seen = clap_detector.claps_so_far
        if wake is not None and wake.process(block):
            console.print(f"\n[green]'{wake.profile.phrase}' detected[/green]")
            if not quiet:
                speaker.play(chirp("listen"), 22050)
        now = time.monotonic()
        if now - last_draw >= 0.1:
            last_draw = now
            bar = "#" * min(30, int(peak * 30))
            wake_text = (
                describe_distance(wake.last_distance, wake.threshold)
                if wake is not None
                else "off"
            )
            sys.stderr.write(f"\r  level {bar:<30} peak {peak:4.2f}  wake {wake_text} ")
            sys.stderr.flush()


# Backwards-compatible name from the clap-only version.
clap = listen


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
