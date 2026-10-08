"""User-taught voice commands, stored in ``~/.openjarvis/commands.json``.

Each command maps one or more phrases to an action::

    {
      "allow_shell": false,
      "commands": [
        {"phrases": ["modo trabajo", "work mode"], "action": "open",
         "target": "https://calendar.google.com", "reply": "Modo trabajo listo"},
        {"phrases": ["pon lofi"], "action": "spotify", "target": "lofi beats"}
      ]
    }

Actions: ``open`` (app, site, file or URL), ``spotify`` (search and play),
``keys`` (volume_up, volume_down, mute, play_pause, next, previous), ``say``
(just answer) and ``shell`` (run a command; disabled unless the file sets
``"allow_shell": true``, because a misheard phrase must never run code the
user did not mean to).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from openjarvis.actions.intents import normalize
from openjarvis.core.config import DEFAULT_CONFIG_DIR

ACTIONS = ("open", "spotify", "keys", "say", "shell")
KEYS = ("volume_up", "volume_down", "mute", "play_pause", "next", "previous")


def commands_path() -> Path:
    return DEFAULT_CONFIG_DIR / "commands.json"


@dataclass
class CustomCommand:
    phrases: list[str]
    action: str
    target: str = ""
    reply: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
        self.phrases = [p.strip() for p in self.phrases if p and p.strip()]
        if not self.phrases:
            raise ValueError("Add at least one phrase")
        if self.action == "keys" and self.target not in KEYS:
            raise ValueError(f"keys target must be one of {', '.join(KEYS)}")
        if self.action in ("open", "spotify", "shell") and not self.target.strip():
            raise ValueError("This action needs a target")
        if self.action == "say" and not (self.reply or self.target).strip():
            raise ValueError("Say what? Add a reply")


@dataclass
class CommandBook:
    commands: list[CustomCommand] = field(default_factory=list)
    allow_shell: bool = False

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CommandBook":
        path = path or commands_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        book = cls(allow_shell=bool(data.get("allow_shell", False)))
        for raw in data.get("commands") or []:
            try:
                command = CustomCommand(
                    phrases=list(raw.get("phrases") or []),
                    action=str(raw.get("action") or ""),
                    target=str(raw.get("target") or ""),
                    reply=str(raw.get("reply") or ""),
                    id=str(raw.get("id") or uuid.uuid4().hex[:8]),
                )
                command.validate()
            except (TypeError, ValueError):
                continue
            book.commands.append(command)
        return book

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or commands_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "allow_shell": self.allow_shell,
                    "commands": [asdict(c) for c in self.commands],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def match(self, text: str) -> Optional[CustomCommand]:
        spoken = normalize(text)
        for command in self.commands:
            if any(normalize(p) == spoken for p in command.phrases):
                return command
        return None

    def add(self, command: CustomCommand) -> CustomCommand:
        command.validate()
        taken = {normalize(p) for c in self.commands for p in c.phrases}
        clash = [p for p in command.phrases if normalize(p) in taken]
        if clash:
            raise ValueError(f"Already used: {', '.join(clash)}")
        self.commands.append(command)
        return command

    def remove(self, command_id: str) -> bool:
        before = len(self.commands)
        self.commands = [c for c in self.commands if c.id != command_id]
        return len(self.commands) != before
