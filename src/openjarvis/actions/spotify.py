"""Control Spotify playback through the Spotify Web API.

Uses the tokens of the Spotify connector (``jarvis connect spotify``), which
now requests the playback scopes. Spotify only allows playback control for
Premium accounts; without Premium (or without a connection) the play
request falls back to opening a search in the Spotify app, and play/pause/
next/previous fall back to the computer's media keys.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from openjarvis.core.config import DEFAULT_CONFIG_DIR

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
TOKEN_PATH = DEFAULT_CONFIG_DIR / "connectors" / "spotify.json"

PLAYBACK_SCOPES = (
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
)


class SpotifyError(RuntimeError):
    """Spotify could not do what was asked; the message is user-facing."""


class SpotifyNotConnected(SpotifyError):
    pass


class SpotifyClient:
    def __init__(self, token_path: Path = TOKEN_PATH, http: Any = None) -> None:
        self.token_path = Path(token_path)
        self.http = http or httpx

    # -- tokens -------------------------------------------------------------

    def _tokens(self) -> dict:
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SpotifyNotConnected(
                "Spotify is not connected. Run: jarvis connect spotify"
            ) from exc
        if not data.get("access_token"):
            raise SpotifyNotConnected(
                "Spotify is not connected. Run: jarvis connect spotify"
            )
        return data

    def connected(self) -> bool:
        try:
            self._tokens()
            return True
        except SpotifyNotConnected:
            return False

    def _refresh(self) -> str:
        data = self._tokens()
        refresh = data.get("refresh_token")
        client_id, secret = data.get("client_id"), data.get("client_secret")
        if not (refresh and client_id and secret):
            raise SpotifyNotConnected(
                "Spotify session expired. Reconnect with: jarvis connect spotify"
            )
        basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
        resp = self.http.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": refresh},
            headers={"Authorization": f"Basic {basic}"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            raise SpotifyNotConnected(
                "Spotify session expired. Reconnect with: jarvis connect spotify"
            )
        fresh = resp.json()
        data["access_token"] = fresh["access_token"]
        if fresh.get("refresh_token"):
            data["refresh_token"] = fresh["refresh_token"]
        data["refreshed_at"] = time.time()
        self.token_path.write_text(json.dumps(data), encoding="utf-8")
        return data["access_token"]

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        retried: bool = False,
        token: Optional[str] = None,
    ) -> Optional[dict]:
        token = token or self._tokens()["access_token"]
        resp = self.http.request(
            method,
            f"{API}{path}",
            params=params,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        if resp.status_code == 401 and not retried:
            return self._call(
                method,
                path,
                params=params,
                body=body,
                retried=True,
                token=self._refresh(),
            )
        if resp.status_code == 403:
            raise SpotifyError(
                "Spotify only lets Premium accounts be controlled remotely "
                "(or the connection lacks playback permission: run "
                "`jarvis connect spotify` again)."
            )
        if resp.status_code == 404:
            raise SpotifyError("NO_ACTIVE_DEVICE")
        if resp.status_code >= 400:
            raise SpotifyError(f"Spotify error {resp.status_code}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def _with_device(self, method: str, path: str, **kwargs: Any) -> None:
        """Run a player call, waking a device when none is active."""
        try:
            self._call(method, path, **kwargs)
            return
        except SpotifyError as exc:
            if str(exc) != "NO_ACTIVE_DEVICE":
                raise
        devices = (self._call("GET", "/me/player/devices") or {}).get("devices", [])
        if not devices:
            raise SpotifyError(
                "No Spotify device is open. Open Spotify on this computer or "
                "your phone and try again."
            )
        device = next((d for d in devices if d.get("type") == "Computer"), devices[0])
        params = dict(kwargs.pop("params", None) or {})
        params["device_id"] = device["id"]
        self._call(method, path, params=params, **kwargs)

    # -- player ---------------------------------------------------------------

    def play(self) -> None:
        self._with_device("PUT", "/me/player/play")

    def pause(self) -> None:
        self._with_device("PUT", "/me/player/pause")

    def next(self) -> None:
        self._with_device("POST", "/me/player/next")

    def previous(self) -> None:
        self._with_device("POST", "/me/player/previous")

    def set_volume(self, percent: int) -> None:
        self._with_device(
            "PUT",
            "/me/player/volume",
            params={"volume_percent": max(0, min(100, int(percent)))},
        )

    def now_playing(self) -> Optional[str]:
        data = self._call("GET", "/me/player/currently-playing")
        item = (data or {}).get("item")
        if not item:
            return None
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return f"{item.get('name')} — {artists}" if artists else item.get("name")

    def play_search(self, query: str, kind: str = "auto") -> str:
        """Search and play; return what is now playing.

        ``kind`` is ``track``, ``playlist``, ``artist``, ``album`` or ``auto``
        (playlist when the request says playlist/music for a mood, else track).
        """
        query = query.strip()
        if not query:
            raise SpotifyError("What should I play?")
        types = {"auto": "track,playlist,artist", "track": "track"}.get(kind, kind)
        result = (
            self._call("GET", "/search", params={"q": query, "type": types, "limit": 1})
            or {}
        )

        def first(section: str) -> Optional[dict]:
            items = (result.get(section) or {}).get("items") or []
            return next((i for i in items if i), None)

        order = {
            "playlist": ["playlists"],
            "artist": ["artists"],
            "album": ["albums"],
            "track": ["tracks"],
        }.get(kind, ["tracks", "playlists", "artists"])
        for section in order:
            item = first(section)
            if not item:
                continue
            if section == "tracks":
                self._with_device(
                    "PUT", "/me/player/play", body={"uris": [item["uri"]]}
                )
                artists = ", ".join(a["name"] for a in item.get("artists", []))
                return f"{item['name']} — {artists}" if artists else item["name"]
            self._with_device(
                "PUT", "/me/player/play", body={"context_uri": item["uri"]}
            )
            return item.get("name", query)
        raise SpotifyError(f"I couldn't find “{query}” on Spotify.")


def open_search_in_app(query: str) -> None:
    """Fallback without API access: show the search in the Spotify app."""
    from urllib.parse import quote

    from openjarvis.actions.system import open_url

    open_url(f"spotify:search:{quote(query)}")
