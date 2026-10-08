"""Control the local computer: volume, media keys, opening apps and sites.

No third-party packages: Windows uses virtual media keys through ``user32``
(the same keys a keyboard's volume buttons send, so every player reacts),
macOS uses ``osascript`` and Linux uses ``pactl``/``playerctl`` with an
``xdotool`` fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from typing import Optional

# Windows virtual-key codes for media keys.
_VK = {
    "mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "play_pause": 0xB3,
}
# One Windows volume key press moves the master volume by 2%.
_WIN_STEP = 2

#: Spoken app names → how to launch them (Windows target, macOS app name).
APP_ALIASES: dict[str, tuple[str, str]] = {
    "spotify": ("spotify:", "Spotify"),
    "chrome": ("chrome", "Google Chrome"),
    "google chrome": ("chrome", "Google Chrome"),
    "edge": ("microsoft-edge:", "Microsoft Edge"),
    "firefox": ("firefox", "Firefox"),
    "navegador": ("https://www.google.com", "Safari"),
    "browser": ("https://www.google.com", "Safari"),
    "calculadora": ("calc", "Calculator"),
    "calculator": ("calc", "Calculator"),
    "bloc de notas": ("notepad", "TextEdit"),
    "notepad": ("notepad", "TextEdit"),
    "notas": ("notepad", "Notes"),
    "explorador": ("explorer", "Finder"),
    "explorador de archivos": ("explorer", "Finder"),
    "archivos": ("explorer", "Finder"),
    "file explorer": ("explorer", "Finder"),
    "finder": ("explorer", "Finder"),
    "word": ("winword", "Microsoft Word"),
    "excel": ("excel", "Microsoft Excel"),
    "powerpoint": ("powerpnt", "Microsoft PowerPoint"),
    "outlook": ("outlook", "Microsoft Outlook"),
    "teams": ("msteams:", "Microsoft Teams"),
    "whatsapp": ("whatsapp:", "WhatsApp"),
    "discord": ("discord:", "Discord"),
    "telegram": ("tg:", "Telegram"),
    "steam": ("steam:", "Steam"),
    "vscode": ("code", "Visual Studio Code"),
    "visual studio code": ("code", "Visual Studio Code"),
    "terminal": ("wt", "Terminal"),
    "configuracion": ("ms-settings:", "System Settings"),
    "ajustes": ("ms-settings:", "System Settings"),
    "settings": ("ms-settings:", "System Settings"),
    "paint": ("mspaint", "Preview"),
    "camara": ("microsoft.windows.camera:", "Photo Booth"),
    "camera": ("microsoft.windows.camera:", "Photo Booth"),
}

#: Spoken site names → URL.
SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "correo": "https://mail.google.com",
    "calendario": "https://calendar.google.com",
    "google calendar": "https://calendar.google.com",
    "calendar": "https://calendar.google.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "mapas": "https://maps.google.com",
    "netflix": "https://www.netflix.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "amazon": "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
    "twitch": "https://www.twitch.tv",
    "reddit": "https://www.reddit.com",
    "spotify web": "https://open.spotify.com",
}


class ActionError(RuntimeError):
    """Raised when the computer could not carry out a request."""


def _run(args: list[str]) -> bool:
    try:
        return (
            subprocess.run(
                args, capture_output=True, check=False, timeout=10
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _win_key(name: str, times: int = 1) -> None:
    import ctypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    code = _VK[name]
    for _ in range(max(1, times)):
        user32.keybd_event(code, 0, 0, 0)
        user32.keybd_event(code, 0, 2, 0)  # KEYEVENTF_KEYUP


def _linux_key(keysym: str) -> bool:
    return bool(shutil.which("xdotool")) and _run(["xdotool", "key", keysym])


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def change_volume(delta_percent: int) -> None:
    """Raise (positive) or lower (negative) the master volume."""
    if sys.platform == "win32":
        _win_key(
            "volume_up" if delta_percent > 0 else "volume_down",
            max(1, abs(delta_percent) // _WIN_STEP),
        )
        return
    if sys.platform == "darwin":
        sign = "+" if delta_percent > 0 else "-"
        script = (
            "set volume output volume ((output volume of (get volume settings))"
            f" {sign} {abs(delta_percent)})"
        )
        if not _run(["osascript", "-e", script]):
            raise ActionError("osascript could not change the volume")
        return
    sign = "+" if delta_percent > 0 else "-"
    if shutil.which("pactl") and _run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{abs(delta_percent)}%"]
    ):
        return
    if shutil.which("amixer") and _run(
        ["amixer", "-q", "sset", "Master", f"{abs(delta_percent)}%{sign}"]
    ):
        return
    raise ActionError("Install pactl (PulseAudio/PipeWire) or amixer to change volume")


def set_volume(percent: int) -> None:
    """Set the master volume to an absolute level (0-100)."""
    percent = max(0, min(100, int(percent)))
    if sys.platform == "win32":
        # No absolute API without extra packages: go to 0, then step up.
        _win_key("volume_down", 50)
        if percent:
            _win_key("volume_up", round(percent / _WIN_STEP))
        return
    if sys.platform == "darwin":
        if not _run(["osascript", "-e", f"set volume output volume {percent}"]):
            raise ActionError("osascript could not change the volume")
        return
    if shutil.which("pactl") and _run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]
    ):
        return
    if shutil.which("amixer") and _run(
        ["amixer", "-q", "sset", "Master", f"{percent}%"]
    ):
        return
    raise ActionError("Install pactl (PulseAudio/PipeWire) or amixer to change volume")


def toggle_mute(mute: Optional[bool] = None) -> None:
    """Mute, unmute, or (``None``) toggle the master output."""
    if sys.platform == "win32":
        _win_key("mute")
        return
    if sys.platform == "darwin":
        if mute is None:
            script = (
                "set volume output muted (not (output muted of (get volume settings)))"
            )
        else:
            script = f"set volume output muted {'true' if mute else 'false'}"
        if not _run(["osascript", "-e", script]):
            raise ActionError("osascript could not change mute")
        return
    state = "toggle" if mute is None else ("1" if mute else "0")
    if shutil.which("pactl") and _run(
        ["pactl", "set-sink-mute", "@DEFAULT_SINK@", state]
    ):
        return
    raise ActionError("Install pactl (PulseAudio/PipeWire) to mute")


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

_MAC_MEDIA = {
    "play_pause": "playpause",
    "play": "play",
    "pause": "pause",
    "next": "next track",
    "previous": "previous track",
}
_PLAYERCTL = {
    "play_pause": "play-pause",
    "play": "play",
    "pause": "pause",
    "next": "next",
    "previous": "previous",
}
_XF86 = {
    "play_pause": "XF86AudioPlay",
    "play": "XF86AudioPlay",
    "pause": "XF86AudioPause",
    "next": "XF86AudioNext",
    "previous": "XF86AudioPrev",
}


def media(action: str) -> None:
    """Send a media command (play_pause, play, pause, next, previous)."""
    if action not in _MAC_MEDIA:
        raise ValueError(f"Unknown media action: {action}")
    if sys.platform == "win32":
        # Windows has a single play/pause key.
        key = "play_pause" if action in ("play", "pause") else action
        _win_key(key)
        return
    if sys.platform == "darwin":
        for app in ("Spotify", "Music"):
            script = (
                f'if application "{app}" is running then tell application "{app}" '
                f"to {_MAC_MEDIA[action]}"
            )
            if _run(["osascript", "-e", script]):
                return
        raise ActionError("No running Spotify or Music app to control")
    if shutil.which("playerctl") and _run(["playerctl", _PLAYERCTL[action]]):
        return
    if _linux_key(_XF86[action]):
        return
    raise ActionError("Install playerctl to control media players")


# ---------------------------------------------------------------------------
# Apps and sites
# ---------------------------------------------------------------------------


def _start_windows(target: str) -> bool:
    try:
        os.startfile(target)  # type: ignore[attr-defined]
        return True
    except OSError:
        return _run(["cmd", "/c", "start", "", target])


def open_url(url: str) -> None:
    if sys.platform == "win32" and _start_windows(url):
        return
    if not webbrowser.open(url):
        raise ActionError(f"Could not open {url}")


def open_app(name: str) -> str:
    """Open an app or site by spoken name; return what was opened."""
    spoken = name.strip().lower()
    if spoken in SITES:
        open_url(SITES[spoken])
        return SITES[spoken]
    win_target, mac_name = APP_ALIASES.get(spoken, (spoken, name.strip()))
    if sys.platform == "win32":
        if win_target.startswith("http"):
            open_url(win_target)
            return win_target
        if _start_windows(win_target):
            return mac_name
    elif sys.platform == "darwin":
        if mac_name.startswith("http"):
            open_url(mac_name)
            return mac_name
        if _run(["open", "-a", mac_name]):
            return mac_name
    else:
        binary = spoken.replace(" ", "-")
        if shutil.which("gtk-launch") and _run(["gtk-launch", binary]):
            return name
        if shutil.which(binary):
            subprocess.Popen(  # noqa: S603 - name comes from PATH lookup
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return name
    raise ActionError(f"I couldn't find an app called {name}")
