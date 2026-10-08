"""Speaking with the operating system's own voice."""

from __future__ import annotations

from unittest import mock

from openjarvis.speech import system_tts


def test_markdown_is_not_read_aloud():
    text = "# Hola\n- **uno**\n```py\nx()\n```\nVer [docs](https://x.y) `cmd`"
    assert system_tts.speakable(text) == "Hola uno Ver docs cmd"


def test_windows_uses_builtin_voices_in_the_right_language():
    spanish = system_tts.command_for("¿Qué tal? Tienes una reunión", "win32")
    english = system_tts.command_for("You have one meeting", "win32")
    assert spanish[0] == "powershell" and "StartsWith('es')" in spanish[-1]
    assert "StartsWith('en')" in english[-1]
    # The reply never becomes part of the script.
    assert "reunión" not in spanish[-1]


def test_windows_passes_text_through_the_environment(monkeypatch):
    monkeypatch.setattr(system_tts.sys, "platform", "win32")
    monkeypatch.setattr(
        system_tts.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
    )
    with mock.patch.object(
        system_tts.subprocess, "run", return_value=mock.Mock(returncode=0)
    ) as run:
        assert system_tts.say("Hola, *soy* Jarvis")
    kwargs = run.call_args.kwargs
    assert kwargs["env"]["OPENJARVIS_SAY_TEXT"] == "Hola, soy Jarvis"
    assert "input" not in kwargs


def test_mac_and_linux(monkeypatch):
    assert system_tts.command_for("hola amigo", "darwin")[:3] == [
        "say",
        "-v",
        "Paulina",
    ]
    monkeypatch.setattr(
        system_tts.shutil,
        "which",
        lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None,
    )
    assert system_tts.command_for("hello", "linux") == [
        "espeak-ng",
        "--stdin",
        "-v",
        "en",
    ]
    monkeypatch.setattr(system_tts.shutil, "which", lambda name: None)
    assert system_tts.command_for("hello", "linux") is None
    assert system_tts.say("hello") is False
