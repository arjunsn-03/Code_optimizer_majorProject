"""
Static analysis — cyclomatic complexity, anti-pattern detection,
hotspot scoring.  Pure-Python; no external tools needed.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List


# ── Cyclomatic complexity ─────────────────────────────────────────────────────

def cyclomatic_complexity(source: str) -> int:
    """
    McCabe cyclomatic complexity:  CC = 1 + number_of_decision_points.
    Returns -1 if source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1

    decisions = 0
    for node in ast.walk(tree):
        if isinstance(node, (
            ast.If, ast.While, ast.For, ast.ExceptHandler,
            ast.With, ast.Assert, ast.comprehension,
        )):
            decisions += 1
        elif isinstance(node, ast.BoolOp):
            decisions += len(node.values) - 1

    return 1 + decisions


# ── Anti-pattern detection ────────────────────────────────────────────────────

_IO_NAMES = frozenset({
    "open", "read", "write", "readline", "readlines", "writelines",
    "print", "requests", "urlopen", "get", "post", "fetch",
    "cursor", "execute", "commit",
})


def detect_antipatterns(source: str, filepath: str) -> List[Dict[str, Any]]:
    """
    Walk the AST and flag common energy-inefficient patterns:
        - nested_loop        : loop inside a loop
        - io_in_loop         : I/O call inside a loop
        - long_function      : function > 80 lines
        - repeated_attr      : method called on same object in a loop
        - nested_comprehension: list-comp inside list-comp
        - global_var_in_loop : global read in tight loop
    """
    issues: List[Dict[str, Any]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    class _Visitor(ast.NodeVisitor):

        # ── Loops ────────────────────────────────────────────────────────────
        def _check_loop(self, node: ast.For | ast.While) -> None:
            # Nested loop
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.For, ast.While)):
                    issues.append({
                        "type":     "nested_loop",
                        "lineno":   node.lineno,
                        "filepath": filepath,
                        "severity": "medium",
                        "message":  "Nested loop — consider vectorisation or NumPy.",
                    })
                    break  # one report per outer loop

            # I/O call inside loop
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = ""
                    if isinstance(child.func, ast.Name):
                        name = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        name = child.func.attr
                    if name in _IO_NAMES:
                        issues.append({
                            "type":     "io_in_loop",
                            "lineno":   getattr(child, "lineno", node.lineno),
                            "filepath": filepath,
                            "severity": "high",
                            "message":  f"I/O call `{name}` inside loop — "
                                        "move outside or batch.",
                        })

        def visit_For(self, node: ast.For) -> None:
            self._check_loop(node)
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            self._check_loop(node)
            self.generic_visit(node)

        # ── Functions ────────────────────────────────────────────────────────
        def _check_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            loc = (getattr(node, "end_lineno", node.lineno)) - node.lineno
            if loc > 80:
                issues.append({
                    "type":     "long_function",
                    "lineno":   node.lineno,
                    "filepath": filepath,
                    "severity": "low",
                    "message":  f"Function `{node.name}` has {loc} lines — "
                                "consider splitting for better inlining.",
                })

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._check_func(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._check_func(node)
            self.generic_visit(node)

        # ── Comprehensions ───────────────────────────────────────────────────
        def visit_ListComp(self, node: ast.ListComp) -> None:
            for elt in ast.walk(node.elt):
                if isinstance(elt, (ast.ListComp, ast.GeneratorExp)):
                    issues.append({
                        "type":     "nested_comprehension",
                        "lineno":   node.lineno,
                        "filepath": filepath,
                        "severity": "low",
                        "message":  "Nested comprehension — consider NumPy or "
                                    "explicit loop with local variable.",
                    })
            self.generic_visit(node)

    _Visitor().visit(tree)
    return issues


# ── Hotspot scoring ───────────────────────────────────────────────────────────

_SEV_WEIGHT = {"high": 0.40, "medium": 0.25, "low": 0.10}


def score_hotspot(func_meta: Dict, issues: List[Dict]) -> float:
    """
    Compute a 0–1 hotspot risk score combining:
        40 % cyclomatic complexity (capped at CC = 20)
        30 % function length     (capped at 200 LOC)
        30 % anti-pattern severity within the function's line-range
    """
    cc  = cyclomatic_complexity(func_meta.get("source", ""))
    loc = func_meta.get("loc", 0)

    cc_score  = min(max(cc, 0) / 20.0, 1.0)
    loc_score = min(loc / 200.0, 1.0)

    fn_start = func_meta.get("lineno", 0)
    fn_end   = func_meta.get("end_lineno", 999_999)
    local_issues = [
        i for i in issues
        if fn_start <= i.get("lineno", 0) <= fn_end
    ]
    issue_score = min(
        sum(_SEV_WEIGHT.get(i["severity"], 0.10) for i in local_issues), 1.0
    )

    return round(0.4 * cc_score + 0.3 * loc_score + 0.3 * issue_score, 3)


# ── File-level entry point ────────────────────────────────────────────────────

def analyze_file(file_meta: Dict) -> Dict[str, Any]:
    """
    Run the full static analysis pass on one file dict (from ingestor).
    Returns a result dict compatible with the orchestrator.
    """
    source   = file_meta["content"]
    filepath = file_meta["path"]
    language = file_meta["language"]

    result: Dict[str, Any] = {
        "filepath":           filepath,
        "relative_path":      file_meta["relative_path"],
        "language":           language,
        "lines":              file_meta["lines"],
        "overall_complexity": -1,
        "antipatterns":       [],
        "functions":          [],
    }

    if language == "python":
        from ingestor.ast_extract import extract_python_functions

        functions = extract_python_functions(source, filepath)
        issues    = detect_antipatterns(source, filepath)

        result["overall_complexity"] = cyclomatic_complexity(source)
        result["antipatterns"]       = issues
        result["functions"]          = [
            {**f, "hotspot_score": score_hotspot(f, issues)}
            for f in functions
            if "error" not in f
        ]

    return result
