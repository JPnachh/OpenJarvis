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
