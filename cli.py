#!/usr/bin/env python3
"""
cli.py — Command-line entry point for the Carbon-Aware Code Optimizer.

Commands
--------
  optimize   Run the full pipeline on a project directory.
  analyze    Static-analysis only (no LLM, no network required).
  check      Verify Ollama is running and list available models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Ensure project root is on path when running directly:  python cli.py
sys.path.insert(0, str(Path(__file__).parent))

console = Console()


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Carbon-Aware LLM Code Optimizer — reduce your software's carbon footprint."""


# ── optimize ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_path", default=".", metavar="REPO_PATH")
@click.option(
    "--model", "-m",
    default="qwen3", show_default=True,
    type=click.Choice(["qwen3", "llama3", "mistral"]),
    help="Local Ollama model to use for optimisation.",
)
@click.option(
    "--top-n", "-n",
    default=5, show_default=True,
    help="Maximum number of hotspot functions to optimise.",
)
@click.option(
    "--gpu", is_flag=True, default=False,
    help="Enable GPU-aware ML prompts (set when a CUDA GPU is present).",
)
def optimize(repo_path: str, model: str, top_n: int, gpu: bool) -> None:
    """
    Run the full pipeline on REPO_PATH.

    Stages: ingest → static analysis → LLM optimise → verify → report.

    \b
    Examples:
        python cli.py optimize .
        python cli.py optimize ~/projects/myapp --model llama3 --top-n 10
        python cli.py optimize . --model mistral --gpu
    """
    from pipeline.orchestrator import Orchestrator

    console.rule("[bold green]Carbon-Aware Code Optimizer[/bold green]")
    console.print(f"  Repo  : [cyan]{Path(repo_path).resolve()}[/cyan]")
    console.print(f"  Model : [cyan]{model}[/cyan]")
    console.print(f"  Top-N : [cyan]{top_n}[/cyan]")
    console.print(f"  GPU   : [cyan]{gpu}[/cyan]")
    console.rule()

    orch = Orchestrator(repo_path=repo_path, model=model, top_n=top_n, gpu=gpu)
    orch.run()


# ── analyze ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("repo_path", default=".", metavar="REPO_PATH")
@click.option(
    "--top-n", "-n",
    default=15, show_default=True,
    help="Number of top hotspots to display.",
)
def analyze(repo_path: str, top_n: int) -> None:
    """
    Run static analysis only — no LLM or network required.

    Displays a ranked table of hotspot functions.

    \b
    Examples:
        python cli.py analyze .
        python cli.py analyze ~/projects/myapp --top-n 20
    """
    from ingestor.clone import walk_repo
    from static_analyzer.analyzer import analyze_file

    files = walk_repo(repo_path)
    console.print(f"Analysing [bold]{len(files)}[/bold] source files …")

    all_funcs: list[dict] = []
    for f in files:
        result = analyze_file(f)
        for func in result.get("functions", []):
            all_funcs.append({**func, "relative_path": result["relative_path"]})

    all_funcs.sort(key=lambda x: x.get("hotspot_score", 0.0), reverse=True)

    table = Table(title=f"Top {top_n} Hotspots", expand=True)
    table.add_column("Rank",         justify="right", style="dim")
    table.add_column("Function",     style="cyan", no_wrap=True)
    table.add_column("File",         style="dim")
    table.add_column("LOC",          justify="right")
    table.add_column("Complexity",   justify="right")
    table.add_column("Hotspot Score",justify="right", style="bold red")

    for i, func in enumerate(all_funcs[:top_n], start=1):
        table.add_row(
            str(i),
            func["name"],
            func["relative_path"],
            str(func.get("loc", "?")),
            str(func.get("hotspot_score", "?")),  # reuse field (cyclomatic via parent)
            f"{func.get('hotspot_score', 0.0):.3f}",
        )

    console.print(table)


# ── check ─────────────────────────────────────────────────────────────────────

@cli.command()
def check() -> None:
    """
    Check Ollama connectivity and list locally available models.

    \b
    Examples:
        python cli.py check
    """
    from llm.client import list_local_models

    console.print("Checking Ollama …")
    models = list_local_models()

    if models and models[0].startswith("[error]"):
        console.print(f"[bold red]{models[0]}[/bold red]")
        console.print(
            "\nMake sure Ollama is running:  [bold]ollama serve[/bold]"
        )
        sys.exit(1)

    console.print(f"[bold green]Ollama is reachable.[/bold green]")
    console.print(f"Available models ({len(models)}):")
    for m in models:
        console.print(f"  • {m}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
