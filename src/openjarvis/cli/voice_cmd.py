"""``jarvis voice`` — install and try Jarvis's local voice (Kokoro-82M)."""

from __future__ import annotations

import click
from rich.console import Console


def install_voice(console: Console) -> bool:
    """Download the Kokoro model if needed; True when the voice is ready."""
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        console.print(
            "[yellow]kokoro-onnx is not installed.[/yellow] "
            "Run: uv sync --extra desktop"
        )
        return False
    from openjarvis.speech import kokoro_onnx_tts as kokoro

    if kokoro.installed():
        console.print(f"[green]✓[/green] Jarvis voice ready ({kokoro.model_dir()})")
        return True

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    console.print("Downloading Jarvis's voice (Kokoro-82M, about 200 MB, once)...")
    tasks: dict[str, int] = {}
    with Progress(
        TextColumn("  {task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:

        def report(name: str, done: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = progress.add_task(name, total=total or None)
            progress.update(tasks[name], completed=done)

        try:
            kokoro.download(report)
        except Exception as exc:  # noqa: BLE001 - network problems are user-facing
            console.print(f"[red]Download failed:[/red] {exc}")
            console.print("Check the internet connection and run: jarvis voice install")
            return False
    console.print(f"[green]✓[/green] Jarvis voice installed ({kokoro.model_dir()})")
    return True


@click.group()
def voice() -> None:
    """Jarvis's voice: a local Kokoro-82M model (Spanish and English)."""


@voice.command("install")
def voice_install() -> None:
    """Download the voice model (once, about 200 MB)."""
    if not install_voice(Console()):
        raise SystemExit(1)


@voice.command("status")
def voice_status() -> None:
    """Show which voice Jarvis uses to speak."""
    console = Console()
    from openjarvis.speech._tts_discovery import get_tts_backend

    backend = get_tts_backend("kokoro-onnx")
    if backend is None:
        console.print(
            "No voice engine installed; the app uses your computer's built-in "
            "voice. Install Jarvis's voice with: jarvis voice install"
        )
        return
    console.print(f"[green]✓[/green] Voice engine: {backend.backend_id}")


@voice.command("test")
@click.argument("text", required=False)
@click.option("--save", type=click.Path(dir_okay=False), help="Also save a WAV file.")
def voice_test(text: str | None, save: str | None) -> None:
    """Say something with Jarvis's voice (Spanish or English)."""
    console = Console()
    from openjarvis.speech._tts_discovery import get_tts_backend

    backend = get_tts_backend("kokoro-onnx")
    if backend is None:
        raise click.ClickException("No voice installed. Run: jarvis voice install")
    text = text or "Hola, soy Jarvis. Todos los sistemas están en línea."
    result = backend.synthesize(text)
    console.print(f"Voice {result.voice_id}, {result.duration_seconds:.1f} s of audio.")
    if save:
        with open(save, "wb") as fh:
            fh.write(result.audio)
        console.print(f"Saved {save}")
    try:
        import sounddevice as sd

        from openjarvis.speech.features import read_wav

        samples, rate = read_wav(result.audio)
        sd.play(samples, rate)
        sd.wait()
    except Exception as exc:  # noqa: BLE001 - no speakers / PortAudio
        if not save:
            raise click.ClickException(f"Could not play audio: {exc}") from exc
