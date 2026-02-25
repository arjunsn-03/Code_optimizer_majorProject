"""
End-to-end pipeline orchestrator.

Pipeline stages
---------------
1. Ingest & index  — walk repo, embed functions, populate VectorStore
2. Static analysis — cyclomatic complexity, anti-patterns, hotspot scoring
3. Optimise        — RAG + LLM patch generation per hotspot
4. Verify          — apply patch in sandbox, run tests, measure energy
5. Report          — rich table to console + JSON artefact
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import DEFAULT_MODEL, MAX_PATCH_RISK, RESULTS_PATH
from estimator.energy import format_energy_report, kwh_to_co2
from indexer.embedder import embed_single, embed_texts
from indexer.store import VectorStore
from ingestor.ast_extract import extract_python_functions
from ingestor.clone import walk_repo
from llm.optimizer import optimize_function
from static_analyzer.analyzer import analyze_file
from verifier.runner import verify_patch

console = Console()


class Orchestrator:
    """
    Ties all pipeline stages together.

    Parameters
    ----------
    repo_path : str
        Absolute or relative path to the target project directory.
    model : str
        Ollama model name (``qwen3``, ``llama3`` or ``mistral``).
    top_n : int
        Maximum number of hotspot functions to optimise.
    gpu : bool
        Set True when a GPU is present (enables ML-aware prompts).
    """

    def __init__(
        self,
        repo_path: str,
        model: str = DEFAULT_MODEL,
        top_n: int = 5,
        gpu: bool = False,
    ) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.model     = model
        self.top_n     = top_n
        self.gpu       = gpu
        self.store     = VectorStore(dim=384)
        self.files: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def ingest_and_index(self) -> None:
        console.print(Panel("[bold cyan]Stage 1 — Ingesting & indexing[/bold cyan]"))

        self.files = walk_repo(self.repo_path)
        console.print(f"  Found [bold]{len(self.files)}[/bold] source files.")

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for f in self.files:
            if f["language"] != "python":
                continue
            for func in extract_python_functions(f["content"], f["path"]):
                src = func.get("source", "")
                if src and "error" not in func:
                    texts.append(src[:800])
                    metas.append({
                        "text_snippet":  src[:800],
                        "filepath":      func["filepath"],
                        "function":      func["name"],
                        "lineno":        func.get("lineno", 0),
                        "artifact_type": "code",
                    })

        if texts:
            embeddings = embed_texts(texts)
            self.store.add(embeddings, metas)
            self.store.save()
            console.print(f"  Indexed [bold]{self.store.size}[/bold] code chunks.")
        else:
            console.print("  [yellow]No Python functions found to index.[/yellow]")

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def analyze(self) -> List[Dict[str, Any]]:
        console.print(Panel("[bold cyan]Stage 2 — Static analysis[/bold cyan]"))

        hotspots: List[Dict[str, Any]] = []

        for f in self.files:
            result = analyze_file(f)
            for func in result.get("functions", []):
                if func.get("hotspot_score", 0.0) > 0.05:
                    hotspots.append({"analysis": result, "func": func})

        hotspots.sort(key=lambda x: x["func"]["hotspot_score"], reverse=True)
        console.print(
            f"  Found [bold]{len(hotspots)}[/bold] hotspot candidates. "
            f"Processing top [bold]{self.top_n}[/bold]."
        )
        return hotspots[: self.top_n]

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def optimize(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        console.print(Panel("[bold cyan]Stage 3 — LLM optimisation[/bold cyan]"))

        suggestions: List[Dict[str, Any]] = []

        for h in hotspots:
            func     = h["func"]
            analysis = h["analysis"]

            console.print(
                f"  ⟳ [bold]{func['name']}[/bold]  "
                f"({analysis['relative_path']})  "
                f"score={func['hotspot_score']}"
            )

            query_emb = embed_single(func.get("source", "")[:800])
            context   = self.store.search(query_emb, k=4)

            suggestion = optimize_function(
                func_meta=func,
                analysis=analysis,
                profiling={"cpu_secs": "n/a", "kwh": "n/a"},
                context_snippets=context,
                model=self.model,
                gpu_available=self.gpu,
            )
            suggestion["relative_path"] = analysis["relative_path"]
            suggestions.append(suggestion)

        return suggestions

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def verify(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        console.print(Panel("[bold cyan]Stage 4 — Verification[/bold cyan]"))

        verified: List[Dict[str, Any]] = []

        for s in suggestions:
            fname = s.get("function_name", "?")

            if not s.get("has_optimization") or not s.get("patch", "").strip():
                s["verification"] = {"verdict": "skipped", "reason": "no_patch"}
                console.print(f"  [dim]↷ {fname}: no patch generated[/dim]")
                verified.append(s)
                continue

            if s.get("risk_score", 1.0) > MAX_PATCH_RISK:
                s["verification"] = {
                    "verdict": "skipped",
                    "reason":  f"risk_too_high ({s.get('risk_score', '?')})",
                }
                console.print(
                    f"  [yellow]⚠ {fname}: skipped — risk too high[/yellow]"
                )
                verified.append(s)
                continue

            console.print(f"  ✓ verifying patch for [bold]{fname}[/bold] ...")
            result = verify_patch(
                patch_text=s["patch"],
                repo_path=self.repo_path,
                test_patterns=s.get("tests_to_run"),
            )
            s["verification"] = result
            verdict_str = (
                "[green]accepted[/green]"
                if result["verdict"] == "accepted"
                else "[red]rejected[/red]"
            )
            console.print(f"    → {verdict_str}  ({result.get('reason', '')})")
            verified.append(s)

        return verified

    # ── Stage 5 ───────────────────────────────────────────────────────────────

    def report(self, suggestions: List[Dict[str, Any]]) -> None:
        console.print(Panel("[bold green]Stage 5 — Results[/bold green]"))

        table = Table(title="Carbon-Aware Optimisation Report", expand=True)
        table.add_column("Function",          style="cyan",      no_wrap=True)
        table.add_column("File",              style="dim")
        table.add_column("Type",              style="magenta")
        table.add_column("Δ kWh",             justify="right")
        table.add_column("Confidence",        justify="right")
        table.add_column("Risk",              justify="right")
        table.add_column("Verdict",           style="bold")

        total_kwh_saved = 0.0

        for s in suggestions:
            verdict = s.get("verification", {}).get("verdict", "unverified")
            colour  = (
                "green"  if verdict == "accepted"  else
                "yellow" if verdict == "skipped"   else
                "red"
            )
            delta = s.get("estimated_kwh_delta", 0.0) or 0.0
            if verdict == "accepted":
                total_kwh_saved += abs(min(delta, 0.0))

            table.add_row(
                s.get("function_name", "?"),
                s.get("relative_path", "?"),
                s.get("optimization_type", "?"),
                f"{delta:+.7f}",
                f"{s.get('confidence', 0.0):.0%}",
                f"{s.get('risk_score', 0.0):.2f}",
                f"[{colour}]{verdict}[/{colour}]",
            )

        console.print(table)

        if total_kwh_saved > 0:
            console.print(
                f"\n[bold green]Estimated total kWh saved per run: "
                f"{total_kwh_saved:.9f}[/bold green]"
            )
            console.print(
                f"  ≈ {kwh_to_co2(total_kwh_saved) * 1000:.5f} g CO₂e per run"
            )

        # Persist
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(suggestions, indent=2, default=str))
        console.print(f"\n  Full results → [bold]{RESULTS_PATH}[/bold]")
        self.results = suggestions

    # ── Run all stages ────────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """Execute all pipeline stages and return the final results list."""
        self.ingest_and_index()
        hotspots    = self.analyze()
        suggestions = self.optimize(hotspots)
        verified    = self.verify(suggestions)
        self.report(verified)
        return verified
