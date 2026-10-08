"""``jarvis tray`` — OpenJarvis in the system tray, next to the clock.

The tray icon opens OpenJarvis, and by default also runs the hands-free
listener ("Hey Jarvis" / claps) in the same process. A coloured dot on the
icon shows what Jarvis is doing:

* grey: not listening
* cyan: waiting for "Hey Jarvis"
* red: listening to you
* amber: thinking
* green: speaking
"""

from __future__ import annotations

import os
import threading
import webbrowser

import click
from rich.console import Console

from openjarvis.cli import desktop_integration as desk
from openjarvis.core.config import DEFAULT_CONFIG_DIR
from openjarvis.core.utils import process_alive

_PID_FILE = DEFAULT_CONFIG_DIR / "tray.pid"

STATE_COLORS = {
    "off": (128, 128, 128),
    "idle": (34, 211, 238),
    "listening": (239, 68, 68),
    "transcribing": (245, 158, 11),
    "thinking": (245, 158, 11),
    "speaking": (34, 197, 94),
    "offline": (128, 128, 128),
}

STATE_LABELS = {
    "off": "not listening",
    "idle": "waiting for 'Hey Jarvis'",
    "listening": "listening to you",
    "transcribing": "transcribing",
    "thinking": "thinking",
    "speaking": "speaking",
    "offline": "not listening",
}


def tray_image(state: str, size: int = 64):
    """App icon with a status dot in the corner."""
    from PIL import Image, ImageDraw

    icon_file = desk.icon_path("png")
    if icon_file is not None:
        base = Image.open(icon_file).convert("RGBA").resize((size, size))
    else:
        base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(base).ellipse((4, 4, size - 4, size - 4), fill=(22, 22, 24, 255))
    draw = ImageDraw.Draw(base)
    dot = size * 0.36
    x0, y0 = size - dot - 1, size - dot - 1
    draw.ellipse((x0 - 2, y0 - 2, size, size), fill=(20, 20, 20, 255))
    draw.ellipse(
        (x0, y0, size - 2, size - 2),
        fill=(*STATE_COLORS.get(state, (128, 128, 128)), 255),
    )
    return base


class TrayApp:
    """Menu actions and the listener thread behind the tray icon."""

    def __init__(
        self,
        *,
        frontend_port: int,
        api_port: int | None,
        listen: bool,
        wake_sensitivity: float,
        claps: int,
    ) -> None:
        from openjarvis.cli.clap_cmd import JarvisOpener

        self.frontend_port = frontend_port
        self.api_port = api_port
        self.want_listen = listen
        self.wake_sensitivity = wake_sensitivity
        self.claps = claps
        self.opener = JarvisOpener(frontend_port, api_port)
        self.icon = None
        self.state = "off"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # -- icon ---------------------------------------------------------------

    def set_state(self, state: str) -> None:
        self.state = state
        if self.icon is not None:
            self.icon.icon = tray_image(state)
            self.icon.title = f"OpenJarvis: {STATE_LABELS.get(state, state)}"

    def notify(self, message: str) -> None:
        if self.icon is not None:
            try:
                self.icon.notify(message, "OpenJarvis")
            except Exception:  # noqa: BLE001 - notifications are optional
                pass

    # -- listener -------------------------------------------------------------

    @property
    def listening(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_listening(self) -> None:
        if self.listening:
            return
        from openjarvis.cli.clap_cmd import _running_listener

        other = _running_listener()
        if other is not None:
            self.notify(
                f"Another listener is already running (PID {other}). "
                "Stop it with `jarvis listen --stop` to listen from the tray."
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        self._stop.set()

    def _listen(self) -> None:
        from openjarvis.cli.clap_cmd import _api_base, _wake_detector
        from openjarvis.speech import wakeword
        from openjarvis.speech.clap import ClapDetector
        from openjarvis.speech.listener import (
            ApiClient,
            ListenerOptions,
            VoiceListener,
            microphone_blocks,
            resolve_api_key,
        )

        wake = _wake_detector(self.wake_sensitivity)
        listener = VoiceListener(
            ApiClient(_api_base(self.api_port), resolve_api_key()),
            options=ListenerOptions(),
            clap_detector=ClapDetector(claps_required=self.claps),
            wake_detector=wake,
            open_ui=self.opener.open,
            log=lambda message: None,
            profile_loader=lambda: _wake_detector(self.wake_sensitivity),
            profile_path=wakeword.wakeword_dir() / "profile.json",
            learner=wakeword.learn_from_hit,
        )
        post_state = listener.set_state

        def mirrored(state: str, detail: str | None = None) -> None:
            self.set_state(
                "idle" if state == "offline" and not self._stop.is_set() else state
            )
            post_state(state, detail)

        listener.set_state = mirrored  # type: ignore[method-assign]
        self.set_state("idle")
        if wake is None:
            self.notify(
                "Listening for claps. Train 'Hey Jarvis' from the menu to talk "
                "to Jarvis by name."
            )
        try:
            for block in microphone_blocks(self._stop.is_set):
                listener.process(block)
        except RuntimeError as exc:
            self.error = str(exc)
            self.notify(f"Microphone problem: {exc}")
        finally:
            listener.close()
            self.set_state("off")

    # -- menu actions ---------------------------------------------------------

    def open_jarvis(self, *_args) -> None:
        self.opener.open()

    def open_page(self, path: str):
        def action(*_args) -> None:
            from openjarvis.cli.clap_cmd import _port_open

            if _port_open(self.frontend_port):
                webbrowser.open(f"http://127.0.0.1:{self.frontend_port}{path}")
            else:
                self.opener.open()
                self.notify("Starting OpenJarvis… open it again in a moment.")

        return action

    def toggle_listening(self, *_args) -> None:
        if self.listening:
            self.stop_listening()
        else:
            self.start_listening()

    def toggle_autostart(self, *_args) -> None:
        if self.autostart_installed():
            desk.remove_autostart()
        else:
            desk.install_autostart(
                ["--frontend-port", str(self.frontend_port)], subcommand="tray"
            )

    @staticmethod
    def autostart_installed() -> bool:
        path = desk.autostart_path()
        try:
            return path.exists() and "tray" in path.read_text(errors="replace")
        except OSError:
            return False

    def menu(self):
        import pystray

        item = pystray.MenuItem
        return pystray.Menu(
            item("Open OpenJarvis", self.open_jarvis, default=True),
            pystray.Menu.SEPARATOR,
            item(
                "Listen for 'Hey Jarvis' and claps",
                self.toggle_listening,
                checked=lambda _: self.listening,
            ),
            item("Train my voice…", self.open_page("/wake-word")),
            item("Commands…", self.open_page("/commands")),
            pystray.Menu.SEPARATOR,
            item(
                "Start with my computer",
                self.toggle_autostart,
                checked=lambda _: self.autostart_installed(),
            ),
            item("Quit", self.quit),
        )

    def quit(self, icon, *_args) -> None:
        self.stop_listening()
        icon.stop()

    def run(self) -> None:
        import pystray

        self.icon = pystray.Icon(
            "openjarvis", tray_image("off"), "OpenJarvis", menu=self.menu()
        )

        def setup(icon) -> None:
            icon.visible = True
            if self.want_listen:
                self.start_listening()

        self.icon.run(setup=setup)


@click.command()
@click.option("--no-listen", is_flag=True, help="Only the icon; no 'Hey Jarvis'.")
@click.option(
    "--wake-sensitivity", default=0.5, show_default=True, type=click.FloatRange(0, 1)
)
@click.option("--claps", default=2, show_default=True, type=click.IntRange(1, 5))
@click.option(
    "--frontend-port", default=5173, show_default=True, type=click.IntRange(1, 65535)
)
@click.option("--api-port", default=None, type=click.IntRange(1, 65535))
@click.option(
    "--autostart",
    "autostart",
    flag_value="on",
    default=None,
    help="Show the tray icon (and listen) at every login.",
)
@click.option("--no-autostart", "autostart", flag_value="off", help="Undo --autostart.")
def tray(
    no_listen: bool,
    wake_sensitivity: float,
    claps: int,
    frontend_port: int,
    api_port: int | None,
    autostart: str | None,
) -> None:
    """Put OpenJarvis in the system tray (next to the clock).

    Click the icon to open OpenJarvis. It also listens for "Hey Jarvis" and
    claps (turn that off from its menu), and its dot shows what Jarvis is
    doing: cyan waiting, red listening, amber thinking, green speaking.
    """
    console = Console(stderr=True)
    if autostart == "off":
        removed = desk.remove_autostart()
        console.print(f"Removed {removed}" if removed else "No autostart entry found.")
        return
    if autostart == "on":
        args = ["--frontend-port", str(frontend_port), "--claps", str(claps)]
        args += ["--wake-sensitivity", str(wake_sensitivity)]
        if no_listen:
            args.append("--no-listen")
        if api_port is not None:
            args += ["--api-port", str(api_port)]
        path = desk.install_autostart(args, subcommand="tray")
        console.print(
            f"[green]The tray icon will start at every login.[/green]\n  {path}\n"
            "Starting it now..."
        )
        import subprocess

        from openjarvis.cli.clap_cmd import _detached_kwargs

        subprocess.Popen(
            desk.jarvis_command("tray", *args, windowless=True),
            cwd=desk.project_root(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detached_kwargs(),
        )
        return

    try:
        import pystray  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "The tray needs pystray and Pillow: uv run --extra desktop jarvis tray"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - e.g. no display on Linux
        raise click.ClickException(f"No desktop to show a tray icon on: {exc}") from exc

    try:
        existing = int(_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        existing = None
    if existing and existing != os.getpid() and process_alive(existing):
        raise click.ClickException(
            f"The tray icon is already running (PID {existing})."
        )
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    try:
        TrayApp(
            frontend_port=frontend_port,
            api_port=api_port,
            listen=not no_listen,
            wake_sensitivity=wake_sensitivity,
            claps=claps,
        ).run()
    finally:
        _PID_FILE.unlink(missing_ok=True)
