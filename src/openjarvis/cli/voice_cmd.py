"""``jarvis voice`` — hands-free assistant ("Hey Jarvis" or double clap)."""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console

from openjarvis.cli._voice_chat import _terminal_safe_text


@click.command("voice")
@click.option(
    "--server", default=None, help="Jarvis server URL (default: from config)."
)
@click.option("--model", default=None, help="Model (default: jarvis-auto).")
@click.option("--device", type=int, default=None, help="Microphone device index.")
@click.option("--list-devices", is_flag=True, help="List audio devices and exit.")
@click.option("--clap/--no-clap", default=None)
@click.option("--wake/--no-wake", default=None)
@click.option("--wake-threshold", type=float, default=None)
@click.option("--follow-up", type=float, default=None)
@click.option(
    "--gain", type=float, default=None, help="Digital mic gain (2-6 if quiet)."
)
@click.option("--debug", is_flag=True, help="Live mic level / wake-word score line.")
@click.option(
    "--find-mic", is_flag=True, help="Try every mic while you speak; remember the best."
)
@click.option("--monitor", is_flag=True, help="Diagnostics meter, Ctrl+C to exit.")
@click.option(
    "--enroll", is_flag=True, help="Teach Jarvis your voice (reads 6 phrases)."
)
@click.option("--forget-voice", is_flag=True, help="Delete the saved voiceprint.")
@click.option("--no-speaker", is_flag=True, help="Do not verify who is speaking.")
@click.option("--toggle-pause", is_flag=True, help="Mute/unmute the listener and exit.")
@click.option(
    "--input-wav",
    type=click.Path(exists=True),
    default=None,
    help="Test: use a WAV, not the mic.",
)
@click.option("--no-play", is_flag=True, help="Do not play audio; save replies.")
@click.option("--save-dir", type=click.Path(), default=None)
@click.option("--max-turns", type=int, default=None)
def voice(
    server: Optional[str],
    model: Optional[str],
    device: Optional[int],
    list_devices: bool,
    clap: Optional[bool],
    wake: Optional[bool],
    wake_threshold: Optional[float],
    follow_up: Optional[float],
    gain: Optional[float],
    debug: bool,
    find_mic: bool,
    monitor: bool,
    enroll: bool,
    forget_voice: bool,
    no_speaker: bool,
    toggle_pause: bool,
    input_wav: Optional[str],
    no_play: bool,
    save_dir: Optional[str],
    max_turns: Optional[int],
) -> None:
    """Listen for "Hey Jarvis" (or two claps), then hold a spoken conversation."""
    from openjarvis.core.config import load_config
    from openjarvis.speech import voice_runtime as rt

    console = Console()

    if list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        return
    if find_mic:
        _find_mic(console)
        return
    if toggle_pause:
        now = not rt.is_paused()
        rt.set_paused(now)
        console.print("Escucha PAUSADA." if now else "Escucha ACTIVA.")
        return
    if forget_voice:
        from openjarvis.speech.speaker_id import SpeakerVerifier

        SpeakerVerifier().clear()
        console.print("Huella de voz borrada.")
        return

    config = load_config()
    opts = rt.options_from_config(config, server)
    if model:
        opts.model = model
    if device is not None:
        opts.device = device
    if clap is not None:
        opts.clap = clap
    if wake is not None:
        opts.wake = wake
    if wake_threshold is not None:
        opts.wake_threshold = wake_threshold
    if follow_up is not None:
        opts.follow_up = follow_up
    if gain is not None:
        opts.gain = gain
    if no_speaker:
        opts.speaker_verify = False
    opts.play = not no_play
    opts.input_wav = input_wav
    opts.save_dir = save_dir
    opts.max_turns = max_turns

    def log(message: str) -> None:
        console.print(_terminal_safe_text(message))

    if monitor:
        _monitor(console, opts.device, opts.wake_threshold, opts.gain)
        return

    if enroll:
        console.print(f"Micrófono: {_device_name(opts.device)}")
        console.print(
            "Voy a enseñarle tu voz. Habla con tu tono normal, a la distancia a la que "
            "sueles hablarle a Jarvis.\nSe pausa la escucha mientras tanto.\n"
        )
        try:
            info = rt.enroll_voice(opts, log)
        except Exception as exc:
            raise click.ClickException(f"No pude registrar tu voz: {exc}")
        console.print(
            f"\n[green]Voz registrada[/green] con {info['samples']} muestras "
            f"(consistencia {info['mean_similarity']:.2f}, "
            f"umbral {info['threshold']:.2f}). "
            "Jarvis ya la usa, no hace falta reiniciar."
        )
        return

    import httpx

    try:
        httpx.get(f"{opts.server_url}/health", timeout=3).raise_for_status()
    except Exception as exc:
        raise click.ClickException(
            f"Jarvis no responde en {opts.server_url}. Inícialo primero. ({exc})"
        )

    if not input_wav:
        lock = rt.acquire_single_instance()
        if lock is None:
            raise click.ClickException(
                "Jarvis ya está escuchando en segundo plano (no hace falta abrir este "
                "modo). Para verlo en vivo: pausa/cierra el servidor o usa --monitor."
            )

    if debug:
        opts.debug_status = _status_line(console, opts.wake_threshold)
    try:
        daemon, source = rt.build_daemon(config, opts, log)
    except Exception as exc:
        raise click.ClickException(str(exc))
    if not input_wav:
        console.print(f"Micrófono: {_device_name(opts.device)}")
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


def _find_mic(console: Console, seconds: float = 3.0, min_level: int = 300) -> None:
    """Record briefly from every input device and keep the loudest one."""
    import numpy as np
    import sounddevice as sd

    from openjarvis.speech.voice_runtime import device_file

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
    file = device_file()
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
