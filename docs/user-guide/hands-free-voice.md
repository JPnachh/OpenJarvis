# Hands-free voice: desktop icon, claps and "Hey Jarvis"

OpenJarvis can open from your desktop and work hands-free. Say **"Hey Jarvis"**
or clap twice, wait for the chirp, then ask your question. Jarvis records your
request until you pause, answers in the chat, and reads the answer aloud.

Everything runs on your computer:

- The wake word is trained on your own voice and needs no download.
- Microphone audio is analysed locally and is not stored.
- Only the request you speak after the chirp is transcribed, by your
  configured speech backend.

## 0. One-step setup

On Windows, paste this line into PowerShell. You don't need administrator
rights:

```powershell
irm https://raw.githubusercontent.com/JPnachh/OpenJarvis/main/scripts/windows/setup-jarvis.ps1 | iex
```

The script does four things:

1. It installs or updates OpenJarvis in `%LOCALAPPDATA%\OpenJarvis\src`,
   the same place the official installer uses, so an existing installation
   is switched over.
2. It installs the Python packages.
3. It runs `jarvis setup-desktop`, which:
   - creates the desktop icon;
   - prepares the graphical interface;
   - starts the **tray icon** now and at every login.
4. The tray icon also listens for "Hey Jarvis" and for claps.

You can change where it installs from with these environment variables:

- `OPENJARVIS_REPO_URL`: the repository
- `OPENJARVIS_BRANCH`: the branch
- `OPENJARVIS_HOME`: the install folder

On macOS or Linux, run these from a checkout:

```bash
uv sync --extra desktop && uv run jarvis setup-desktop
```

The individual steps are described below.

## 1. Put OpenJarvis on the desktop

```bash
uv run jarvis shortcut
```

| OS | What you get |
|---|---|
| Windows | `OpenJarvis.lnk` on the Desktop (OneDrive Desktop is supported) and in the Start menu |
| macOS | `OpenJarvis.command` on the Desktop |
| Linux | `openjarvis.desktop` on the Desktop and in the app menu |

Double-clicking the icon runs `jarvis gui`. If OpenJarvis is already open, the
icon brings up the page instead of starting it again. If something goes
wrong, the window stays open so you can read the error. Remove the icon with
`jarvis shortcut --remove`.

## 2. Train "Hey Jarvis" with your voice

In the app: **Settings → Hey Jarvis → Train my voice**, or **Hey Jarvis** in
the sidebar.

1. Record the phrase about 6 times. Each recording stops on its own when you
   pause. You can pick your own phrase, for example "Oye Jarvis".
2. Record 3 other sentences that are not the wake word, such as "what time is
   it" or "hey Travis". These teach Jarvis what to ignore.
3. Click **Train my voice**. The result rates the profile as good, fair or poor
   and gives advice.

From the terminal you can run `uv run --extra desktop jarvis wake train`
instead. It does the same steps interactively.

### Where the training files live

| Path | Contents |
|---|---|
| `~/.openjarvis/wakeword/samples/positive/*.wav` | your recordings of the phrase |
| `~/.openjarvis/wakeword/samples/negative/*.wav` | your other sentences |
| `~/.openjarvis/wakeword/profile.json` | the trained profile: voice templates and match threshold |

On Windows, `~` is `C:\Users\<you>`.

| Command | What it does |
|---|---|
| `jarvis wake status` | Shows progress and these paths |
| `jarvis wake folder` | Opens the training folder |
| `jarvis wake retrain` | Rebuilds the profile from the saved recordings |
| `jarvis wake reset` | Deletes the recordings and the profile |

You can play back or delete individual recordings on the training page.

## 3. Start the listener

```bash
uv run --extra desktop jarvis listen --test        # check levels, nothing opens
uv run --extra desktop jarvis listen --autostart   # run now and at every login
```

`--autostart` needs no administrator rights. It installs the listener as one
of these:

- Windows: a hidden Startup-folder entry
- macOS: a LaunchAgent
- Linux: an XDG autostart entry

| Option | Effect |
|---|---|
| `--wake-sensitivity 0..1` | Higher accepts looser matches of your phrase (default 0.5) |
| `--sensitivity 0..1` | Clap sensitivity |
| `--claps N` | Number of claps (default 2) |
| `--no-claps` / `--no-wake` | Use only one trigger |
| `--open-only` | Only open OpenJarvis; do not record a request |
| `--no-speak` | Show answers in the page without speaking them |
| `--quiet` | No chirps |
| `--stop` / `--no-autostart` | Stop the listener / remove the login entry |

`jarvis clap` is an alias of `jarvis listen`.

### What happens on a trigger

1. If no OpenJarvis page is open, the listener opens one, and starts the
   server and frontend if needed.
2. It plays a chirp, records until you pause (up to 15 s), and plays a
   second chirp. While it does this, the page shows **Listening** and then
   **Transcribing**.
3. The request appears in the chat and goes to the model.
4. The listener reads the answer aloud through your speakers. The listener
   speaks the reply instead of the browser because browsers block audio that
   a click did not start. The listener also ignores the microphone while
   Jarvis talks, so it does not hear itself.

The voice features need the following:

- **Speech-to-text and text-to-speech backends** (`jarvis doctor` checks
  them). Without text-to-speech, answers appear only in the page.
- **Microphone permission.** On Windows: Settings → Privacy & security →
  Microphone → "Let desktop apps access your microphone". On macOS: System
  Settings → Privacy & Security → Microphone. On Linux, also run
  `sudo apt install libportaudio2`.

## 4. The tray icon (next to the clock)

```bash
uv run --extra desktop jarvis tray              # now
uv run --extra desktop jarvis tray --autostart  # now and at every login
```

Click the icon to open OpenJarvis. The tray runs the listener itself, so
you need either the tray or `jarvis listen`, not both. The coloured dot
shows what Jarvis is doing:

| Dot | Meaning |
|---|---|
| Grey | Not listening |
| Cyan | Waiting for "Hey Jarvis" |
| Red | Listening to you |
| Amber | Thinking |
| Green | Speaking |

The menu has these items:

- Open OpenJarvis
- Listen on/off
- Train my voice
- Commands
- Start with my computer
- Quit

## 5. Direct commands: instant, no model

Some requests run immediately on your computer and get a short spoken
answer. Anything else goes to the model as usual.

| Say | Does |
|---|---|
| "sube / baja el volumen", "volumen al 40", "silencio" | System volume |
| "pausa", "dale play", "siguiente canción", "la anterior" | Music (Spotify, or the media keys) |
| "pon Bad Bunny en Spotify", "pon la playlist rock en español", "pon música de Shakira" | Searches Spotify and plays the best match |
| "abre Spotify", "abre YouTube", "abre la calculadora" | Opens an app or site |
| "¿qué tengo hoy/mañana?", "¿cuál es mi próxima reunión?" | Reads your Google Calendar |
| "agenda dentista el viernes a las 10", "recuérdame llamar a mamá en 30 minutos" | Adds a calendar event |
| "¿qué hora es?", "¿qué día es hoy?" | Time and date |

English phrasing works too, for example "next song", "turn up the volume" and
"what's on my calendar tomorrow".

These commands need some setup:

- **Spotify control:** run `jarvis connect spotify` and use a **Premium**
  account. Spotify only allows remote control for Premium. Without it,
  play, pause, next and previous use your keyboard's media keys, and a
  search opens in the Spotify app.
- **Calendar:** run `jarvis connect gdrive`. The same Google sign-in covers
  Calendar.
- **Linux volume and media:** install `pactl` and `playerctl`.

The model gets the same abilities as tools, so a normal conversation can
also use them, for example "pon algo tranquilo para trabajar" or "agenda mi
cita del jueves a las 4". The tools are `media_control`, `system_volume`,
`open_app`, `spotify_play`, `calendar_events` and `calendar_add_event`.

### Teach Jarvis your own commands

Open **Commands** in the sidebar. Type one or more phrases, then choose what
Jarvis does:

- open an app, site, file or link;
- play something on Spotify;
- press a media or volume key;
- just answer.

The commands are saved in `~/.openjarvis/commands.json`. Shell commands are
off unless you set `"allow_shell": true` in that file yourself, so that a
misheard phrase can never run a program.

## 6. Jarvis keeps learning your voice

When "Hey Jarvis" is recognised and you then ask something, that take is
saved as a training sample, marked "learned" on the training page. The
profile is rebuilt every two such takes. Only the six most recent learned
takes are kept, so the recordings you made on purpose always carry most of
the weight. To turn this off, use `jarvis listen --no-learn`.

## Troubleshooting: "it hears me but does not answer"

- **No voice, no 🔊 button.** Jarvis speaks through a TTS backend if one is
  installed (Kokoro with `uv sync --extra voice`, an OpenAI key, or a
  Cartesia key). Otherwise it uses your computer's built-in voice: the
  browser's voices in the page, and the system voice for the listener.
  **Settings → Speech → Voice** shows which one is in use and has a
  **Test voice** button.
- **The mic records but nothing happens.** Speech-to-text needs the local
  Whisper model. It downloads on first use (about 150 MB), so it needs
  internet once. **Settings → Speech → Backend status** and the mic button's
  tooltip show the exact reason. The server retries by itself, every 30
  seconds, if the model wasn't ready when it started.
- **"Not Found" everywhere.** An older server is still running. Open
  OpenJarvis from the desktop icon, which restarts it, or restart the
  computer.
- **Errors in hands-free mode.** Errors from "Hey Jarvis" or clap requests
  now appear as a notice in the page and as a tray notification.

## Tips for reliable detection

- Train in the room and at the distance where you will use it.
- Say the phrase the way you will normally say it, then make a short pause
  before the request.
- If Jarvis misses you, add samples and retrain, or raise
  `--wake-sensitivity`. If it wakes on its own, record more negative sentences
  (especially ones that sound similar) or lower the sensitivity.
