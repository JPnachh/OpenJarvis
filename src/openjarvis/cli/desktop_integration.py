"""Desktop shortcuts and login autostart entries for Windows, macOS and Linux.

Everything here writes plain files in the user's own folders (Desktop,
Startup folder, LaunchAgents, XDG autostart) — no administrator rights.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "OpenJarvis"
CLAP_TASK_NAME = "OpenJarvis Clap Listener"
_MAC_LABEL = "ai.openjarvis.clap"


def project_root() -> Path:
    """Source checkout root (``src/openjarvis/cli`` → three levels up)."""
    return Path(__file__).resolve().parents[3]


def icon_path(kind: str) -> Path | None:
    """Return the bundled app icon of *kind* (``ico``, ``icns`` or ``png``)."""
    icons = project_root() / "frontend" / "src-tauri" / "icons"
    candidate = icons / ("icon.png" if kind == "png" else f"icon.{kind}")
    return candidate if candidate.is_file() else None


def jarvis_command(*args: str, windowless: bool = False) -> list[str]:
    """Run this same interpreter's ``jarvis`` CLI with *args*.

    The interpreter is the virtualenv this command runs from, so shortcuts keep
    working regardless of the PATH an Explorer/Finder launch inherits.
    """
    python = Path(sys.executable)
    if windowless and sys.platform == "win32":
        pythonw = python.with_name("pythonw.exe")
        if pythonw.is_file():
            python = pythonw
    return [str(python), "-m", "openjarvis.cli", *args]


def desktop_dir() -> Path:
    """The user's Desktop folder, honouring OneDrive/XDG redirection."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[Environment]::GetFolderPath('Desktop')",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            ).stdout.strip()
            if out:
                return Path(out)
        except (OSError, subprocess.SubprocessError):
            pass
        return Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"
    if sys.platform != "darwin" and shutil.which("xdg-user-dir"):
        try:
            out = subprocess.run(
                ["xdg-user-dir", "DESKTOP"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            ).stdout.strip()
            if out:
                return Path(out)
        except (OSError, subprocess.SubprocessError):
            pass
    return Path.home() / "Desktop"


def start_menu_dir() -> Path:
    """Per-user Start menu Programs folder (Windows)."""
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _windows_shortcut(
    link: Path, command: list[str], workdir: Path, icon: Path | None
) -> None:
    """Create a .lnk through the WScript.Shell COM object."""
    script = "; ".join(
        [
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
            + _ps_quote(str(link))
            + ")",
            "$s.TargetPath = " + _ps_quote(command[0]),
            "$s.Arguments = " + _ps_quote(subprocess.list2cmdline(command[1:])),
            "$s.WorkingDirectory = " + _ps_quote(str(workdir)),
            "$s.Description = " + _ps_quote("Open OpenJarvis"),
            *(["$s.IconLocation = " + _ps_quote(str(icon))] if icon else []),
            "$s.Save()",
        ]
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not link.exists():
        raise RuntimeError(result.stderr.strip() or "PowerShell could not save it")


def create_desktop_shortcut(directory: Path | None = None) -> Path:
    """Create a desktop icon that runs ``jarvis gui``; return its path."""
    directory = directory or desktop_dir()
    directory.mkdir(parents=True, exist_ok=True)
    root = project_root()
    command = jarvis_command("gui", "--pause-on-error")

    if sys.platform == "win32":
        link = directory / f"{APP_NAME}.lnk"
        try:
            _windows_shortcut(link, command, root, icon_path("ico"))
            # Also list it in the Start menu, so Windows search finds it.
            try:
                start_menu = start_menu_dir()
                start_menu.mkdir(parents=True, exist_ok=True)
                _windows_shortcut(
                    start_menu / f"{APP_NAME}.lnk", command, root, icon_path("ico")
                )
            except (OSError, RuntimeError):
                pass
            return link
        except (OSError, RuntimeError):
            # Fall back to a batch file, which needs no COM.
            bat = directory / f"{APP_NAME}.cmd"
            bat.write_text(
                f'@echo off\r\ncd /d "{root}"\r\n{subprocess.list2cmdline(command)}\r\n'
                "if errorlevel 1 pause\r\n",
                encoding="utf-8",
            )
            return bat

    if sys.platform == "darwin":
        path = directory / f"{APP_NAME}.command"
        path.write_text(
            f"#!/bin/bash\ncd {shlex.quote(str(root))}\nexec {shlex.join(command)}\n"
        )
        path.chmod(0o755)
        return path

    icon = icon_path("png")
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=Open the OpenJarvis graphical interface\n"
        f"Exec={shlex.join(command)}\n"
        f"Path={root}\n"
        "Terminal=true\n"
        "Categories=Utility;\n" + (f"Icon={icon}\n" if icon else "")
    )
    path = directory / "openjarvis.desktop"
    path.write_text(entry)
    path.chmod(0o755)
    # Also list it in the application menu.
    menu = Path.home() / ".local" / "share" / "applications"
    try:
        menu.mkdir(parents=True, exist_ok=True)
        (menu / "openjarvis.desktop").write_text(entry)
    except OSError:
        pass
    # GNOME only launches desktop files marked trusted.
    if shutil.which("gio"):
        subprocess.run(
            ["gio", "set", str(path), "metadata::trusted", "true"],
            capture_output=True,
            check=False,
        )
    return path


def remove_desktop_shortcut(directory: Path | None = None) -> list[Path]:
    directory = directory or desktop_dir()
    names = [f"{APP_NAME}.lnk", f"{APP_NAME}.cmd", f"{APP_NAME}.command"]
    names.append("openjarvis.desktop")
    paths = [directory / n for n in names]
    paths.append(
        Path.home() / ".local" / "share" / "applications" / "openjarvis.desktop"
    )
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


# ---------------------------------------------------------------------------
# Autostart for the clap listener
# ---------------------------------------------------------------------------


def autostart_path() -> Path:
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return (
            appdata
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
            / f"{CLAP_TASK_NAME}.vbs"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{_MAC_LABEL}.plist"
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config / "autostart" / "openjarvis-clap.desktop"


def install_autostart(clap_args: list[str]) -> Path:
    """Start ``jarvis clap`` (with *clap_args*) at every login, hidden."""
    command = jarvis_command("clap", *clap_args, windowless=True)
    root = project_root()
    path = autostart_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # A .vbs in the Startup folder runs without a console window
        # (window style 0) and needs no admin rights, unlike Task Scheduler.
        line = subprocess.list2cmdline(command).replace('"', '""')
        path.write_text(
            'Set sh = CreateObject("WScript.Shell")\r\n'
            f'sh.CurrentDirectory = "{root}"\r\n'
            f'sh.Run "{line}", 0, False\r\n',
            encoding="utf-8",
        )
        return path

    if sys.platform == "darwin":
        log = Path.home() / ".openjarvis" / "clap.log"
        args = "\n".join(f"    <string>{_xml(a)}</string>" for a in command)
        path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n<dict>\n'
            f"  <key>Label</key><string>{_MAC_LABEL}</string>\n"
            f"  <key>ProgramArguments</key>\n  <array>\n{args}\n  </array>\n"
            f"  <key>WorkingDirectory</key><string>{_xml(str(root))}</string>\n"
            "  <key>RunAtLoad</key><true/>\n"
            "  <key>KeepAlive</key><false/>\n"
            f"  <key>StandardOutPath</key><string>{_xml(str(log))}</string>\n"
            f"  <key>StandardErrorPath</key><string>{_xml(str(log))}</string>\n"
            "</dict>\n</plist>\n"
        )
        return path

    path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={CLAP_TASK_NAME}\n"
        "Comment=Open OpenJarvis when you clap twice\n"
        f"Exec={shlex.join(command)}\n"
        f"Path={root}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )
    return path


def remove_autostart() -> Path | None:
    path = autostart_path()
    if path.exists():
        path.unlink()
        return path
    return None


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
