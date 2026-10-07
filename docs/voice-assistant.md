# Hands-free voice (`jarvis voice`)

A background loop that listens for **"Hey Jarvis"** (or a **double clap**), records
until you stop talking, transcribes locally, asks the Jarvis server (model
`jarvis-auto`, so it can route to Claude) and speaks the reply with your TTS
backend (ElevenLabs by default). After a reply it keeps listening for a
follow-up for a few seconds, no wake word needed.

```bash
uv sync --extra voice-wake ...        # openwakeword, sounddevice, soundfile
uv run --no-sync jarvis serve         # or the GUI; the server must be running
uv run --no-sync jarvis voice
```

Useful flags: `--list-devices`, `--device N`, `--no-clap`, `--no-wake`,
`--wake-threshold 0.5`, `--follow-up 8`, `--model jarvis-auto`.

## Triggers

- **Wake word:** openWakeWord's pre-trained English `hey_jarvis`. "Hey Jarvis" and
  "Ey Jarvis" score ~1.0 in tests. Spanish **"Oye Jarvis" is unreliable** (it
  scored 0.44 alone and 0.18 inside a sentence, below any safe threshold). A
  Spanish-specific model would have to be trained.
- **Double clap:** two sharp sounds 0.12-0.75 s apart. Speech/music are rejected
  because they do not decay quickly. Tuned on synthetic claps; adjust
  `ClapDetector` (`min_peak`, `ratio`) for your microphone and room.

Say "cancela" / "para" / "olvídalo" after the trigger to abort without asking.

## Privacy

Audio is processed locally until a trigger fires. Only the transcribed text goes
to the Jarvis server (and to Claude if the router escalates), and only the
reply text goes to the TTS provider. Audio is never written to disk.

## Testing without a microphone

```bash
uv run --no-sync jarvis voice --input-wav question.wav --no-play --save-dir out --max-turns 1
```

## Known limits

- No barge-in: the microphone is not heard while Jarvis is speaking.
- Latency is typically several seconds (local STT + model + TTS).

## Always-on mode (integrated with the server)

With `[voice_assistant] autostart = true` in `~/.openjarvis/config.toml`, the API
server starts the listener in a background thread. Anything that starts the server
(`jarvis serve`, `jarvis gui`, a Windows login task) therefore also starts voice —
no separate window. A supervisor restarts the microphone stream if it stalls
(sleep, unplugged device). Only one listener runs at a time (loopback lock).

```toml
[voice_assistant]
autostart = true
follow_up = 8.0          # seconds to keep listening after a reply
idle_reset_min = 10      # forget the spoken conversation after this idle time
speaker_verify = true    # only enforced once you enrol your voice
speaker_threshold = 0.0  # 0 = use the value computed at enrolment
chime = true             # ready / wake / end / denied sounds
```

Chimes: double rising tone = listening, single high = heard you, falling =
conversation over, low buzz = voice not recognised.

Logs: `~/.openjarvis/voice.log`. Mute switch: `jarvis voice --toggle-pause`
(creates/removes `~/.openjarvis/voice.pause`).

## Teaching Jarvis your voice

```bash
jarvis voice --enroll        # read 6 phrases; saves ~/.openjarvis/voiceprint.npz
jarvis voice --forget-voice  # delete it
```

A 26 MB WeSpeaker model (via sherpa-onnx) turns speech into a voiceprint. After
enrolment, the first utterance of each conversation (including the "Hey Jarvis"
audio) must match it, otherwise it is ignored. The voiceprint refines itself
slowly on confident matches (`speaker_adapt`). It is picked up without a restart.

**This is a convenience filter, not security.** Measured with synthetic voices:
the enrolled voice scored 0.82-0.86, a very different voice 0.42-0.55, but a
similar-sounding male voice reached 0.64-0.72. Recordings or voice clones of you
can pass, and noise or a different microphone lowers your own score. The
voiceprint is biometric data; it stays on this machine.

## Windows login

Scripts live outside the repo (they contain machine paths): a hidden launcher
`~/.openjarvis/bin/jarvis-background.{vbs,bat}` and a copy of the `.vbs` in the
user's Startup folder. Create that copy from your own (non-sandboxed) session.

## Web UI sync

The listener reports to the server (`POST /v1/voice/events`); browsers subscribe
over SSE (`GET /v1/voice/events`, snapshot at `GET /v1/voice/state`). The web app
shows a status pill (listening / heard you / thinking / speaking / voice not
recognised / paused) and mirrors every spoken exchange into a chat called
"🎙 Conversación por voz", tagged with the model that answered.

Each reply gets the real local date and time (the models have no clock); set
`location = "City, Country"` under `[voice_assistant]` for local weather.

## Reliability notes

- Ollama's CUDA runner sometimes dies loading a cold model
  (`llama-server process has terminated ... CUDA error`). The Ollama engine now
  retries up to twice, and the voice assistant falls back to Claude if the local
  model still fails (needs `ANTHROPIC_API_KEY`).
- **Do not start Jarvis with `jarvis gui`** (without `--no-server`): it runs
  `uv run --extra desktop jarvis start`, which re-syncs the virtualenv and
  removes extras such as `voice-wake`. Start the server with `jarvis serve` and
  the UI with `jarvis gui --no-server`.
