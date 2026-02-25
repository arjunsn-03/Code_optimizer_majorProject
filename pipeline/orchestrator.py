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

from typing import Callable, Optional

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

# Log levels used by the callback
LOG_INFO    = "info"
LOG_SUCCESS = "success"
LOG_WARNING = "warning"
LOG_ERROR   = "error"
LOG_STAGE   = "stage"


class Orchestrator:
    """
    Ties all pipeline stages together.

    Parameters
    ----------
    repo_path     : Absolute or relative path to the target project directory.
    model         : Ollama model name.
    top_n         : Maximum number of hotspot functions to optimise.
    gpu           : Set True when a GPU is present (enables ML-aware prompts).
    log_callback  : Optional callable(level: str, message: str) for streaming
                    logs to a UI. Called from the same thread as run().
    """

    def __init__(
        self,
        repo_path: str,
        model: str = DEFAULT_MODEL,
        top_n: int = 5,
        gpu: bool = False,
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.repo_path    = str(Path(repo_path).resolve())
        self.model        = model
        self.top_n        = top_n
        self.gpu          = gpu
        self.log_callback = log_callback
        self.store        = VectorStore(dim=384)
        self.files: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def _log(self, level: str, message: str) -> None:
        """Emit to rich console AND to the UI callback if provided."""
        colour_map = {
            LOG_INFO:    "",
            LOG_SUCCESS: "[bold green]",
            LOG_WARNING: "[yellow]",
            LOG_ERROR:   "[bold red]",
            LOG_STAGE:   "[bold cyan]",
        }
        prefix = colour_map.get(level, "")
        suffix = "[/bold green]" if level == LOG_SUCCESS else \
                 "[/yellow]"     if level == LOG_WARNING  else \
                 "[/bold red]"   if level == LOG_ERROR    else \
                 "[/bold cyan]"  if level == LOG_STAGE    else ""
        console.print(f"  {prefix}{message}{suffix}" if prefix else f"  {message}")
        if self.log_callback:
            self.log_callback(level, message)

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def ingest_and_index(self) -> None:
        self._log(LOG_STAGE, "Stage 1 — Ingesting & indexing")

        self.files = walk_repo(self.repo_path)
        self._log(LOG_INFO, f"Found {len(self.files)} source files.")

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
            self._log(LOG_SUCCESS, f"Indexed {self.store.size} code chunks into vector store.")
        else:
            self._log(LOG_WARNING, "No Python functions found to index.")

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def analyze(self) -> List[Dict[str, Any]]:
        self._log(LOG_STAGE, "Stage 2 — Static analysis")

        hotspots: List[Dict[str, Any]] = []

        for f in self.files:
            result = analyze_file(f)
            for func in result.get("functions", []):
                if func.get("hotspot_score", 0.0) > 0.05:
                    hotspots.append({"analysis": result, "func": func})

        hotspots.sort(key=lambda x: x["func"]["hotspot_score"], reverse=True)
        self._log(LOG_INFO, f"Found {len(hotspots)} hotspot candidates — processing top {self.top_n}.")
        return hotspots[: self.top_n]

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def optimize(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._log(LOG_STAGE, f"Stage 3 — LLM optimisation  (model: {self.model})")

        suggestions: List[Dict[str, Any]] = []

        for i, h in enumerate(hotspots, 1):
            func     = h["func"]
            analysis = h["analysis"]

            self._log(LOG_INFO,
                f"[{i}/{len(hotspots)}] Optimising `{func['name']}` "
                f"in {analysis['relative_path']}  (score={func['hotspot_score']:.3f})"
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

            has = suggestion.get("has_optimization", False)
            opt_type = suggestion.get("optimization_type", "none")
            self._log(
                LOG_SUCCESS if has else LOG_WARNING,
                f"  → {'patch generated' if has else 'no optimization found'}  "
                f"type={opt_type}  "
                f"confidence={suggestion.get('confidence', 0):.0%}  "
                f"risk={suggestion.get('risk_score', 0):.2f}"
            )
            suggestions.append(suggestion)

        return suggestions

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def verify(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._log(LOG_STAGE, "Stage 4 — Verification")

        verified: List[Dict[str, Any]] = []

        for s in suggestions:
            fname = s.get("function_name", "?")

            if not s.get("has_optimization") or not s.get("patch", "").strip():
                s["verification"] = {"verdict": "skipped", "reason": "no_patch"}
                self._log(LOG_INFO, f"↷ {fname}: no patch — skipping verification")
                verified.append(s)
                continue

            if s.get("risk_score", 1.0) > MAX_PATCH_RISK:
                s["verification"] = {
                    "verdict": "skipped",
                    "reason":  f"risk_too_high ({s.get('risk_score', '?')})",
                }
                self._log(LOG_WARNING, f"⚠ {fname}: skipped — risk score too high ({s.get('risk_score', '?'):.2f} > {MAX_PATCH_RISK})")
                verified.append(s)
                continue

            self._log(LOG_INFO, f"Applying patch for `{fname}` in sandbox ...")
            result = verify_patch(
                patch_text=s["patch"],
                repo_path=self.repo_path,
                test_patterns=s.get("tests_to_run"),
            )
            s["verification"] = result
            if result["verdict"] == "accepted":
                self._log(LOG_SUCCESS, f"✓ {fname}: patch accepted — tests passed")
            else:
                self._log(LOG_ERROR, f"✗ {fname}: patch rejected — {result.get('reason', '')}")
            verified.append(s)

        return verified

    # ── Stage 5 ───────────────────────────────────────────────────────────────

    def report(self, suggestions: List[Dict[str, Any]]) -> None:
        self._log(LOG_STAGE, "Stage 5 — Report")

        table = Table(title="Carbon-Aware Optimisation Report", expand=True)
        table.add_column("Function",   style="cyan",    no_wrap=True)
        table.add_column("File",       style="dim")
        table.add_column("Type",       style="magenta")
        table.add_column("Δ kWh",      justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Risk",       justify="right")
        table.add_column("Verdict",    style="bold")

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
            self._log(LOG_SUCCESS, f"Total kWh saved per run: {total_kwh_saved:.9f}")
            self._log(LOG_SUCCESS, f"≈ {kwh_to_co2(total_kwh_saved) * 1000:.5f} g CO₂e saved per run")
        else:
            self._log(LOG_WARNING, "No accepted patches this run — no energy savings recorded.")

        # Persist
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(suggestions, indent=2, default=str))
        self._log(LOG_INFO, f"Results saved → {RESULTS_PATH}")
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
