# Hands-free voice: desktop icon, claps and "Hey Jarvis"

OpenJarvis can open from your desktop and work hands-free. Say **"Hey Jarvis"**
or clap twice, wait for the chirp, then ask your question. Jarvis records your
request until you pause, answers in the chat, and reads the answer aloud.

Everything runs on your computer:

- The wake word is trained on your own voice and needs no download.
- Microphone audio is analysed locally and is not stored.
- Only the request you speak after the chirp is transcribed, by your
  configured speech backend.

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

## Tips for reliable detection

- Train in the room and at the distance where you will use it.
- Say the phrase the way you will normally say it, then make a short pause
  before the request.
- If Jarvis misses you, add samples and retrain, or raise
  `--wake-sensitivity`. If it wakes on its own, record more negative sentences
  (especially ones that sound similar) or lower the sensitivity.
