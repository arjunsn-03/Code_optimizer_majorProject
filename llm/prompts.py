"""
Prompt templates used by the LLM Optimizer.

All prompts instruct the model to return valid JSON only.
The schemas match the TypedDicts in optimizer.py.
"""

# ── System prompt (shared) ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert software engineer specialising in energy-efficient code and \
reducing software carbon footprint.
Your task: analyse the given code, then propose the single best \
correctness-preserving optimisation that most reduces CPU time / energy use.

Rules:
1. Preserve every public API signature.
2. All existing tests must still pass after your change.
3. Respond ONLY with a single, valid JSON object — no prose, no markdown \
   fences outside the JSON."""


# ── General-purpose optimisation prompt ───────────────────────────────────────

OPTIMIZATION_PROMPT = """\
=== SIMILAR PAST OPTIMISATIONS (for reference) ===
{context}

=== TARGET CODE ===
File:     {filepath}
Language: {language}

```{language}
{code}
```

=== STATIC ANALYSIS ===
- Overall cyclomatic complexity : {complexity}
- Anti-patterns detected        : {antipatterns}
- Hotspot risk score            : {hotspot_score}

=== PROFILING (baseline) ===
- CPU seconds  : {cpu_secs}
- Estimated kWh: {kwh}

=== CONSTRAINTS ===
- Must preserve public API signatures
- Must not break existing tests
- Prefer algorithmic improvements, vectorisation, caching, or I/O reduction

Respond with exactly this JSON schema (no extra keys):
{{
  "has_optimization"        : true | false,
  "patch"                   : "<unified-diff string or empty>",
  "optimization_type"       : "algorithmic | vectorization | caching | \
io_reduction | concurrency | memory_layout | batching | none",
  "estimated_kwh_delta"     : <float — negative = energy saving>,
  "estimated_cpu_secs_delta": <float — negative = improvement>,
  "confidence"              : <0.0–1.0>,
  "risk_score"              : <0.0–1.0>,
  "explanation"             : "<why this change reduces energy use>",
  "tests_to_run"            : ["<test file pattern>"],
  "ml_safety"               : false
}}"""


# ── ML / AI project prompt ────────────────────────────────────────────────────

ML_OPTIMIZATION_PROMPT = """\
=== SIMILAR PAST ML OPTIMISATIONS ===
{context}

=== TARGET ML CODE ===
File           : {filepath}
Framework hints: {framework_hints}
GPU available  : {gpu_available}

```python
{code}
```

=== FOCUS AREAS ===
- Mixed precision (AMP / bfloat16)
- Batch-size and dataloader worker tuning
- Data-pipeline caching (pin_memory, prefetch)
- Operator fusion / torch.compile
- Quantisation-aware training or post-training quantisation
- Pruning / distillation opportunities
- Avoiding redundant forward/backward passes

=== CONSTRAINTS ===
- Preserve model accuracy within acceptable delta (state your assumption)
- Keep validation metric loss < 1 % relative unless explicitly noted
- Set ml_safety=true for any change touching model weights / training loop

Respond with exactly this JSON schema:
{{
  "has_optimization"    : true | false,
  "patch"               : "<unified-diff string or empty>",
  "optimization_type"   : "mixed_precision | batching | data_pipeline | \
quantization | operator_fusion | pruning | caching | none",
  "estimated_kwh_delta" : <float>,
  "estimated_co2_delta_kg": <float>,
  "confidence"          : <0.0–1.0>,
  "risk_score"          : <0.0–1.0>,
  "accuracy_risk"       : <0.0–1.0>,
  "explanation"         : "<why this change reduces compute / energy use>",
  "requires_retrain"    : false | true,
  "ml_safety"           : true
}}"""
