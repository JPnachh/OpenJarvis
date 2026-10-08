"""Speak with the operating system's own voice: no packages, no API keys.

The fallback for spoken replies when no TTS backend (Kokoro, OpenAI,
Cartesia) is installed. Windows uses the built-in SAPI voices through
PowerShell's System.Speech, macOS uses ``say``, Linux ``espeak-ng`` or
``spd-say``. A Spanish voice is chosen for Spanish text when one exists.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

_SPANISH = re.compile(
    r"[áéíóúñ¿¡]|\b(el|la|los|las|que|de|es|una?|por|para|con|hola|gracias|tienes)\b",
    re.IGNORECASE,
)

# On Windows the text travels in an environment variable (UTF-16, so accents
# survive) rather than in the script, so quotes or newlines in a reply can
# never break out of it.
_TEXT_ENV = "OPENJARVIS_SAY_TEXT"
_WINDOWS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and "
    "$_.VoiceInfo.Culture.Name.StartsWith('__LANG__') } | Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
    "$s.Speak($env:OPENJARVIS_SAY_TEXT)"
)


def speakable(text: str) -> str:
    """Strip markdown so symbols are not read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+|^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~>|#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_spanish(text: str) -> bool:
    return bool(_SPANISH.search(text))


def command_for(text: str, platform: str | None = None) -> list[str] | None:
    """The command that speaks *text*, or None if this system has no voice.

    macOS and espeak read the text from stdin; spd-say takes it as an
    argument; Windows reads it from an environment variable (see ``say``).
    """
    platform = platform or sys.platform
    spanish = is_spanish(text)
    if platform == "win32":
        script = _WINDOWS_SCRIPT.replace("__LANG__", "es" if spanish else "en")
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if platform == "darwin":
        return ["say", "-v", "Paulina", "-f", "-"] if spanish else ["say", "-f", "-"]
    if shutil.which("espeak-ng"):
        return ["espeak-ng", "--stdin", "-v", "es" if spanish else "en"]
    if shutil.which("spd-say"):
        # spd-say takes the text as an argument and -w waits for the end.
        return ["spd-say", "-w", "-l", "es" if spanish else "en"]
    return None


def say(text: str, timeout: float = 120.0) -> bool:
    """Speak *text* and wait until done; False when no system voice exists."""
    clean = speakable(text)
    if not clean:
        return True
    command = command_for(clean)
    if command is None:
        return False
    kwargs: dict = {}
    if sys.platform == "win32":
        import os

        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        kwargs["env"] = {**os.environ, _TEXT_ENV: clean}
    try:
        if command[0] == "powershell":
            result = subprocess.run(
                command, capture_output=True, timeout=timeout, **kwargs
            )
        elif command[0] == "spd-say":
            result = subprocess.run(
                [*command, clean], capture_output=True, timeout=timeout, **kwargs
            )
        else:
            result = subprocess.run(
                command,
                input=clean.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                **kwargs,
            )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0 and command[:3] == ["say", "-v", "Paulina"]:
        # No Spanish voice installed on this Mac: use the default one.
        return say_default_mac(clean, timeout)
    return result.returncode == 0


def say_default_mac(text: str, timeout: float) -> bool:
    try:
        return (
            subprocess.run(
                ["say", "-f", "-"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False
