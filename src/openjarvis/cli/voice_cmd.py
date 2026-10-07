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
@click.option(
    "--debug",
    is_flag=True,
    help="Show a live line with mic level and wake-word score.",
)
@click.option(
    "--gain",
    type=float,
    default=1.0,
    show_default=True,
    help="Digital microphone gain (use 2-6 if your mic is quiet).",
)
@click.option(
    "--find-mic",
    is_flag=True,
    help="Try every microphone while you speak and remember the one that hears you.",
)
@click.option(
    "--monitor",
    is_flag=True,
    help="Diagnostics: show mic level and wake-word score live, then exit with Ctrl+C.",
)
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
    monitor: bool,
    find_mic: bool,
    debug: bool,
    gain: float,
) -> None:
    """Listen for "Hey Jarvis" (or two claps), then hold a spoken conversation."""
    console = Console()

    if list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        return

    if find_mic:
        _find_mic(console)
        return

    if device is None:
        device = _saved_device()

    if monitor:
        _monitor(console, device, wake_threshold, gain)
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
        status=_status_line(console, wake_threshold) if debug else None,
    )
    source = WavSource(input_wav) if input_wav else MicSource(device, gain)
    if not input_wav:
        console.print(f"Micrófono: {_device_name(device)}")
    try:
        daemon.run(source)
    except KeyboardInterrupt:
        console.print("\nAdiós.")


def _device_name(device: Optional[int]) -> str:
    try:
        import sounddevice as sd

        idx = device if device is not None else sd.default.device[0]
        return f"[{idx}] {sd.query_devices(idx)['name']}"
    except Exception as exc:  # pragma: no cover - depends on audio hardware
        return f"desconocido ({exc})"


def _monitor(
    console: Console, device: Optional[int], wake_threshold: float, gain: float = 1.0
) -> None:
    """Live meter: mic level, wake-word score and clap detection."""
    import numpy as np

    from openjarvis.speech.clap import ClapDetector
    from openjarvis.speech.voice_daemon import FRAME, MicSource
    from openjarvis.speech.wake_word import WakeWordDetector

    wake = WakeWordDetector(threshold=wake_threshold)
    clap = ClapDetector()
    console.print(f"Micrófono: {_device_name(device)}")
    console.print(
        "Habla o aplaude. Nivel = volumen del micrófono; "
        f"wake = confianza de «Hey Jarvis» (se activa a {wake_threshold}). "
        "Ctrl+C para salir.\n"
    )
    peak_rms = 0.0
    best_wake = 0.0
    n = 0
    try:
        for frame in MicSource(device, gain):
            assert len(frame) == FRAME
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            fired_wake = wake.process(frame)
            fired_clap = clap.process(frame)
            peak_rms = max(peak_rms, rms)
            best_wake = max(best_wake, wake.last_score)
            n += 1
            if fired_wake or fired_clap:
                what = "HEY JARVIS" if fired_wake else "APLAUSOS"
                console.print(f"[bold green]>>> {what} detectado[/bold green]")
            if n % 6 == 0:  # about every 0.5 s
                bar = "#" * min(40, int(peak_rms / 250))
                console.print(
                    f"nivel {int(peak_rms):5d} {bar:<40} wake {best_wake:.2f}"
                )
                peak_rms = 0.0
                best_wake = 0.0
    except KeyboardInterrupt:
        console.print("\nListo.")


def _device_file():
    from pathlib import Path

    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    return Path(DEFAULT_CONFIG_DIR) / "mic_device.txt"


def _saved_device() -> Optional[int]:
    """Microphone remembered by ``--find-mic`` (None if never chosen)."""
    try:
        return int(_device_file().read_text().strip())
    except (OSError, ValueError):
        return None


def _find_mic(console: Console, seconds: float = 3.0, min_level: int = 300) -> None:
    """Record briefly from every input device and keep the loudest one."""
    import numpy as np
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    candidates = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if "MME" not in hostapis[dev["hostapi"]]["name"]:  # one entry per mic
            continue
        if "Sound Mapper" in dev["name"] or "Stereo Mix" in dev["name"]:
            continue
        candidates.append((idx, dev["name"]))

    console.print(
        f"Voy a probar {len(candidates)} micrófonos, {seconds:.0f} s cada uno.\n"
        "HABLA FUERTE todo el tiempo (por ejemplo: «Hey Jarvis, uno, dos, tres»).\n"
    )
    results = []
    for idx, name in candidates:
        try:
            audio = sd.rec(
                int(seconds * 16000),
                samplerate=16000,
                channels=1,
                dtype="int16",
                device=idx,
            )
            sd.wait()
            level = int(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        except Exception as exc:
            console.print(f"  [{idx}] {name}: error ({exc})")
            continue
        results.append((level, idx, name))
        console.print(f"  [{idx}] {name}: nivel {level}")

    if not results:
        console.print("[red]No pude abrir ningún micrófono.[/red]")
        return
    level, idx, name = max(results)
    if level < min_level:
        console.print(
            "\n[red]Ningún micrófono captó voz.[/red] Revisa: Configuración de "
            "Windows > Privacidad > Micrófono (permitir apps de escritorio) y que "
            "el micrófono no esté silenciado."
        )
        return
    file = _device_file()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(str(idx))
    console.print(f"\n[green]Usaré [{idx}] {name} (nivel {level}).[/green] Guardado.")


def _status_line(console: Console, threshold: float):
    """One in-place line: mic level bar + wake-word confidence."""

    def show(level: float, wake: float) -> None:
        bar = "#" * min(30, int(level / 250))
        mark = " <-- casi" if wake >= 0.25 else ""
        console.print(
            f"nivel {int(level):5d} {bar:<30} wake {wake:.2f}/{threshold}{mark}   ",
            end="\r",
            highlight=False,
        )

    return show
