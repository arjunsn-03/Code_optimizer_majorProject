"""
AST extraction for Python source files.
Extracts function/method metadata and import statements using the built-in
`ast` module — no external dependencies required.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List


# ── Function / method extraction ──────────────────────────────────────────────

def extract_python_functions(source: str, filepath: str) -> List[Dict[str, Any]]:
    """
    Parse *source* and return a list of dicts, one per function / method.
    Each dict contains:
        filepath, name, class, lineno, end_lineno, source, args,
        is_async, decorators, docstring, loc
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"filepath": filepath, "error": str(exc)}]

    lines  = source.splitlines()
    result: List[Dict[str, Any]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self, cls_name: str | None = None):
            self.cls_name = cls_name

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)
            self.generic_visit(node)

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            start   = node.lineno - 1
            end     = getattr(node, "end_lineno", start + 10)
            src     = "\n".join(lines[start:end])
            decorators: List[str] = []
            if hasattr(ast, "unparse"):
                decorators = [ast.unparse(d) for d in node.decorator_list]

            result.append({
                "filepath":    filepath,
                "name":        node.name,
                "class":       self.cls_name,
                "lineno":      node.lineno,
                "end_lineno":  end,
                "source":      src,
                "args":        [a.arg for a in node.args.args],
                "is_async":    isinstance(node, ast.AsyncFunctionDef),
                "decorators":  decorators,
                "docstring":   ast.get_docstring(node) or "",
                "loc":         end - start,
            })

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            child_visitor = _Visitor(cls_name=node.name)
            for item in node.body:
                child_visitor.visit(item)

    _Visitor().visit(tree)
    return result


# ── Import extraction ─────────────────────────────────────────────────────────

def extract_imports(source: str) -> List[str]:
    """Return a flat list of top-level module names imported in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

    return imports
