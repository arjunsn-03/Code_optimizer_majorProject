"""
LLM Optimizer — RAG-assisted patch generation.

Detects whether code uses ML frameworks, selects the appropriate prompt
template, calls the local Ollama model, and returns a structured result dict.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from config import DEFAULT_MODEL
from llm.client import generate
from llm.prompts import ML_OPTIMIZATION_PROMPT, OPTIMIZATION_PROMPT, SYSTEM_PROMPT

# ── ML framework detection ────────────────────────────────────────────────────

_ML_FRAMEWORKS = frozenset({
    "torch", "tensorflow", "keras", "transformers", "sklearn",
    "xgboost", "lightgbm", "jax", "paddle", "mxnet", "onnx",
})


def _is_ml_code(imports: List[str]) -> bool:
    joined = " ".join(imports).lower()
    return any(fw in joined for fw in _ML_FRAMEWORKS)


def _detected_frameworks(imports: List[str]) -> List[str]:
    joined = " ".join(imports).lower()
    return [fw for fw in _ML_FRAMEWORKS if fw in joined]


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse the first JSON object found in *text*."""
    # 1. Attempt direct parse (model was well-behaved)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences and retry
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Find first {...} block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Main optimiser ────────────────────────────────────────────────────────────

def optimize_function(
    func_meta: Dict[str, Any],
    analysis: Dict[str, Any],
    profiling: Dict[str, Any],
    context_snippets: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    gpu_available: bool = False,
) -> Dict[str, Any]:
    """
    Generate an optimisation suggestion for one function / code block.

    Parameters
    ----------
    func_meta        : dict from ``ingestor.ast_extract.extract_python_functions``
    analysis         : dict from ``static_analyzer.analyzer.analyze_file``
    profiling        : dict from ``profiler.energy.measure_*``
    context_snippets : nearest-neighbour results from VectorStore
    model            : Ollama model name
    gpu_available    : enables ML-specific GPU prompts

    Returns a dict with keys: has_optimization, patch, optimization_type,
    estimated_kwh_delta, confidence, risk_score, explanation, etc.
    """
    # ── Build RAG context string ──────────────────────────────────────────────
    context_text = "\n\n".join(
        f"[Example {i + 1}]\n{s.get('text_snippet', '')[:500]}"
        for i, s in enumerate(context_snippets[:3])
    ) or "No similar optimisations found in store."

    # ── Detect ML code ────────────────────────────────────────────────────────
    from ingestor.ast_extract import extract_imports
    imports    = extract_imports(func_meta.get("source", ""))
    is_ml      = _is_ml_code(imports)
    frameworks = _detected_frameworks(imports)

    # ── Format antipatterns ───────────────────────────────────────────────────
    antipatterns_text = "; ".join(
        f"{a['type']} at line {a['lineno']}: {a['message']}"
        for a in analysis.get("antipatterns", [])
    ) or "none"

    # ── Select and fill prompt ────────────────────────────────────────────────
    if is_ml:
        prompt = ML_OPTIMIZATION_PROMPT.format(
            context=context_text,
            filepath=func_meta.get("filepath", ""),
            framework_hints=", ".join(frameworks) or "unknown",
            gpu_available=gpu_available,
            code=func_meta.get("source", "")[:3000],
        )
    else:
        prompt = OPTIMIZATION_PROMPT.format(
            context=context_text,
            filepath=func_meta.get("filepath", ""),
            language="python",
            code=func_meta.get("source", "")[:3000],
            complexity=analysis.get("overall_complexity", "unknown"),
            antipatterns=antipatterns_text,
            hotspot_score=func_meta.get("hotspot_score", 0.0),
            cpu_secs=profiling.get("cpu_secs", "n/a"),
            kwh=profiling.get("kwh", "n/a"),
        )

    # ── Call model ────────────────────────────────────────────────────────────
    raw     = generate(prompt, model=model, system=SYSTEM_PROMPT, temperature=0.2)
    parsed  = _extract_json(raw)

    if parsed is None:
        return {
            "has_optimization": False,
            "error":            "Failed to parse LLM JSON response",
            "raw_response":     raw[:800],
            "function_name":    func_meta.get("name", ""),
            "filepath":         func_meta.get("filepath", ""),
            "is_ml":            is_ml,
        }

    # ── Attach function context ───────────────────────────────────────────────
    parsed["function_name"] = func_meta.get("name", "")
    parsed["filepath"]      = func_meta.get("filepath", "")
    parsed["is_ml"]         = is_ml

    return parsed
