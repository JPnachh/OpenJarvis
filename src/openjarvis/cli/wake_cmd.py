"""``jarvis wake`` — train and manage the personal "Hey Jarvis" wake word."""

from __future__ import annotations

import shutil
import subprocess
import sys
import webbrowser

import click
from rich.console import Console
from rich.table import Table

# Mirrors openjarvis.speech.wakeword, which is imported lazily: it pulls in
# numpy, and the CLI must start without it.
_DEFAULT_PHRASE = "Hey Jarvis"
_MIN_POSITIVES = 3
_RECOMMENDED_POSITIVES = 6


def _record_once(seconds_max: float) -> bytes:
    from openjarvis.speech.voice_io import record_until_silence

    return record_until_silence(
        silence_seconds=0.7,
        startup_silence_seconds=5.0,
        max_seconds=seconds_max,
    )


def _print_status(console: Console) -> dict:
    from openjarvis.speech import wakeword

    info = wakeword.status()
    table = Table(show_header=False, box=None)
    table.add_row("Phrase", info["phrase"])
    table.add_row(
        "Your samples",
        f"{info['positives']} of '{info['phrase']}' "
        f"(min {info['min_positives']}, best {info['recommended_positives']}+), "
        f"{info['negatives']} other phrases",
    )
    stats = info.get("stats") or {}
    table.add_row(
        "Trained",
        f"yes, quality: {stats.get('quality', '?')}" if info["trained"] else "no",
    )
    table.add_row("Training folder", info["directory"])
    table.add_row("Profile file", info["profile_path"])
    console.print(table)
    if stats.get("advice"):
        console.print(f"[dim]{stats['advice']}[/dim]")
    return info


@click.group()
def wake() -> None:
    """Train "Hey Jarvis" with your own voice.

    The easiest way is the app: Settings -> Hey Jarvis -> Train my voice.
    These commands do the same from the terminal.
    """


@wake.command("status")
def wake_status() -> None:
    """Show training progress and where the training files live."""

    _print_status(Console())


@wake.command("train")
@click.option("--phrase", default=_DEFAULT_PHRASE, show_default=True)
@click.option(
    "--samples",
    "count",
    default=_RECOMMENDED_POSITIVES,
    show_default=True,
    type=click.IntRange(_MIN_POSITIVES, 20),
    help="How many times to record the phrase.",
)
@click.option(
    "--negatives",
    default=3,
    show_default=True,
    type=click.IntRange(0, 10),
    help="Other sentences to record so Jarvis learns what is NOT the wake word.",
)
@click.option(
    "--keep", is_flag=True, help="Add to existing samples instead of starting over."
)
def wake_train(phrase: str, count: int, negatives: int, keep: bool) -> None:
    """Record your voice and build the wake-word profile (about 2 minutes)."""
    from openjarvis.speech import wakeword

    console = Console()
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError) as exc:
        raise click.ClickException(
            "Recording needs sounddevice/PortAudio: "
            "uv run --extra desktop jarvis wake train"
        ) from exc

    if not keep and any(True for _ in wakeword.list_samples()):
        if click.confirm("Delete your previous samples and start over?", default=True):
            for sample in wakeword.list_samples():
                wakeword.delete_sample(sample.id)

    console.print(
        f"\n[bold]Step 1/2:[/bold] say [bold cyan]{phrase}[/bold cyan] {count} "
        "times, the way you will normally say it. Vary your distance to the "
        "mic a little. Press Enter, then speak.\n"
    )
    done = 0
    while done < count:
        click.prompt(
            f"  [{done + 1}/{count}] Enter to record",
            default="",
            show_default=False,
            prompt_suffix=" ",
        )
        console.print("  [red]● recording...[/red] speak now")
        try:
            sample = wakeword.add_sample(_record_once(3.0), "positive")
        except ValueError as exc:
            console.print(f"  [yellow]{exc} Let's try that one again.[/yellow]")
            continue
        done += 1
        console.print(f"  [green]saved[/green] ({sample.seconds:.1f}s)")

    if negatives:
        console.print(
            f"\n[bold]Step 2/2:[/bold] say {negatives} ordinary sentences that "
            f"are NOT '{phrase}' (e.g. 'what time is it', 'hello there', "
            "something with similar sounds).\n"
        )
        done = 0
        while done < negatives:
            click.prompt(
                f"  [{done + 1}/{negatives}] Enter to record",
                default="",
                show_default=False,
                prompt_suffix=" ",
            )
            console.print("  [red]● recording...[/red] speak now")
            try:
                wakeword.add_sample(_record_once(8.0), "negative")
            except ValueError as exc:
                console.print(f"  [yellow]{exc} Let's try that one again.[/yellow]")
                continue
            done += 1
            console.print("  [green]saved[/green]")

    console.print("\nTraining...")
    try:
        profile = wakeword.train(phrase)
    except wakeword.TrainingError as exc:
        raise click.ClickException(str(exc)) from exc
    stats = profile.stats
    colour = {"good": "green", "fair": "yellow"}.get(stats.get("quality"), "red")
    console.print(
        f"[{colour}]Done. Quality: {stats.get('quality')}[/{colour}]  "
        f"{stats.get('advice', '')}"
    )
    console.print(
        "\nTry it: [bold]uv run --extra desktop jarvis listen --test[/bold]\n"
        "Use it:  [bold]uv run --extra desktop jarvis listen --autostart[/bold]"
    )


@wake.command("retrain")
@click.option("--phrase", default=None, help="Change the phrase name.")
def wake_retrain(phrase: str | None) -> None:
    """Rebuild the profile from the samples already saved."""
    from openjarvis.speech import wakeword

    console = Console()
    current = wakeword.WakeWordProfile.load()
    name = phrase or (current.phrase if current else wakeword.DEFAULT_PHRASE)
    try:
        profile = wakeword.train(name)
    except wakeword.TrainingError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"[green]Retrained.[/green] Quality: {profile.stats.get('quality')}. "
        f"{profile.stats.get('advice', '')}"
    )


@wake.command("reset")
@click.confirmation_option(prompt="Delete all wake-word samples and the profile?")
def wake_reset() -> None:
    """Delete every sample and the trained profile."""
    from openjarvis.speech import wakeword

    folder = wakeword.wakeword_dir()
    if folder.exists():
        shutil.rmtree(folder)
    Console().print(f"[green]Removed[/green] {folder}")


@wake.command("folder")
def wake_folder() -> None:
    """Open the training folder in your file manager."""
    from openjarvis.speech import wakeword

    folder = wakeword.wakeword_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.run(["explorer", str(folder)], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)], check=False)
    elif shutil.which("xdg-open"):
        subprocess.run(["xdg-open", str(folder)], check=False)
    Console().print(str(folder))


@wake.command("page")
@click.option("--frontend-port", default=5173, show_default=True, type=int)
def wake_page(frontend_port: int) -> None:
    """Open the training page in the app (needs `jarvis gui` running)."""
    url = f"http://127.0.0.1:{frontend_port}/wake-word"
    webbrowser.open(url)
    Console().print(url)
