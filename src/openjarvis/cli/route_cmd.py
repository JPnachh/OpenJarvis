"""``jarvis route`` — explain where a message would run (local vs Claude)."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table


@click.command("route")
@click.argument("text", nargs=-1, required=True)
def route(text: tuple[str, ...]) -> None:
    """Show which model `jarvis-auto` would pick for TEXT, and why.

    Nothing is sent to any model; this only runs the routing rules.
    """
    from openjarvis.core.config import load_config
    from openjarvis.learning.routing.complexity import score_complexity
    from openjarvis.learning.routing.hybrid_router import decide

    config = load_config()
    cfg = config.hybrid_routing
    decision = decide(
        " ".join(text),
        cfg,
        local_default=config.intelligence.default_model,
    )
    signals = score_complexity(decision.query).signals

    console = Console()
    table = Table(show_header=False)
    table.add_row("Destino", decision.target)
    table.add_row("Modelo", decision.model)
    table.add_row("Motivo", decision.reason)
    table.add_row("Complejidad", f"{decision.score} ({decision.tier})")
    table.add_row(
        "Señales",
        ", ".join(f"{k}={v}" for k, v in signals.items() if isinstance(v, float)),
    )
    table.add_row("Umbral nube", str(cfg.cloud_threshold))
    console.print(table)
    if not cfg.enabled:
        console.print("[yellow]hybrid_routing está desactivado en config.[/yellow]")
