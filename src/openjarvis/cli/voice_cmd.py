"""``jarvis voice`` — hands-free assistant ("Hey Jarvis" or double clap)."""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console

from openjarvis.cli._voice_chat import VoiceSession, _terminal_safe_text


def _make_beep(play: bool):
    def beep() -> None:
        if not play:
            return
        try:
            import numpy as np
            import sounddevice as sd

            t = np.linspace(0, 0.12, int(24000 * 0.12), endpoint=False)
            tone = 0.25 * np.sin(2 * np.pi * 880 * t) * np.linspace(1, 0.2, t.size)
            sd.play(tone.astype("float32"), 24000)
            sd.wait()
        except Exception:
            pass

    return beep


@click.command("voice")
@click.option("--server", default="http://localhost:8000", show_default=True)
@click.option("--model", default="jarvis-auto", show_default=True)
@click.option("--device", type=int, default=None, help="Microphone device index.")
@click.option("--list-devices", is_flag=True, help="List audio devices and exit.")
@click.option("--clap/--no-clap", default=True, show_default=True)
@click.option("--wake/--no-wake", default=True, show_default=True)
@click.option("--wake-threshold", type=float, default=0.5, show_default=True)
@click.option("--follow-up", type=float, default=8.0, show_default=True)
@click.option(
    "--input-wav",
    type=click.Path(exists=True),
    default=None,
    help="Test mode: use this WAV instead of the microphone.",
)
@click.option("--no-play", is_flag=True, help="Do not play audio; save replies.")
@click.option("--save-dir", type=click.Path(), default=None)
@click.option("--max-turns", type=int, default=None)
def voice(
    server: str,
    model: str,
    device: Optional[int],
    list_devices: bool,
    clap: bool,
    wake: bool,
    wake_threshold: float,
    follow_up: float,
    input_wav: Optional[str],
    no_play: bool,
    save_dir: Optional[str],
    max_turns: Optional[int],
) -> None:
    """Listen for "Hey Jarvis" (or two claps), then hold a spoken conversation."""
    console = Console()

    if list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        return

    import httpx

    from openjarvis.core.config import load_config
    from openjarvis.speech.voice_daemon import (
        VOICE_SYSTEM_PROMPT,
        MicSource,
        VoiceDaemon,
        WavSource,
        strip_markdown,
    )

    config = load_config()
    session = VoiceSession(config)
    stt = session.get_stt_backend()
    if stt is None:
        raise click.ClickException("No speech-to-text backend (extra 'speech').")
    language = getattr(config.speech, "language", "") or None

    try:
        httpx.get(f"{server}/health", timeout=3).raise_for_status()
    except Exception as exc:
        raise click.ClickException(
            f"Jarvis server not reachable at {server}. Start it first "
            f"('Iniciar Jarvis.bat'). ({exc})"
        )

    history: list[dict] = []

    def transcribe(wav: bytes) -> str:
        return stt.transcribe(wav, format="wav", language=language).text.strip()

    def ask(text: str) -> Optional[str]:
        messages = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
        messages += history[-8:] + [{"role": "user", "content": text}]
        try:
            resp = httpx.post(
                f"{server}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 400,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = strip_markdown(data["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            console.print(
                f"[red]Error con el servidor: {_terminal_safe_text(exc)}[/red]"
            )
            return "No pude conectar con el servidor de Jarvis."
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        used = data.get("model", "?")
        console.print(f"[dim]({used})[/dim] Jarvis: {_terminal_safe_text(reply)}")
        return reply

    saved = {"n": 0}

    def say(text: str) -> None:
        backend = session.get_tts_backend()
        if backend is None:
            return
        voice_id, speed = session.voice_for_backend(backend, console)
        kwargs = {"output_format": "wav", "speed": speed}
        if voice_id:
            kwargs["voice_id"] = voice_id
        result = backend.synthesize(text[:1500], **kwargs)
        if not result.audio:
            return
        if no_play:
            from pathlib import Path

            out = Path(save_dir or ".")
            out.mkdir(parents=True, exist_ok=True)
            saved["n"] += 1
            (out / f"reply_{saved['n']}.wav").write_bytes(result.audio)
        else:
            from openjarvis.speech.voice_io import play_wav

            play_wav(result.audio, sample_rate=result.sample_rate)

    clap_det = None
    if clap:
        from openjarvis.speech.clap import ClapDetector

        clap_det = ClapDetector()
    wake_det = None
    if wake:
        from openjarvis.speech.wake_word import WakeWordDetector

        wake_det = WakeWordDetector(threshold=wake_threshold)
    if clap_det is None and wake_det is None:
        raise click.ClickException("Enable at least one of --clap / --wake.")

    daemon = VoiceDaemon(
        transcribe=transcribe,
        ask=ask,
        say=say,
        beep=_make_beep(not no_play and input_wav is None),
        log=lambda m: console.print(_terminal_safe_text(m)),
        wake=wake_det,
        clap=clap_det,
        follow_up_s=follow_up,
        max_turns=max_turns,
    )
    source = WavSource(input_wav) if input_wav else MicSource(device)
    try:
        daemon.run(source)
    except KeyboardInterrupt:
        console.print("\nAdiós.")
