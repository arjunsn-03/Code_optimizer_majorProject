"""
Jupyter Notebook (.ipynb) extractor.

Reads a .ipynb file and synthesises a single Python source string from all
code cells so the rest of the pipeline (AST, static analysis, LLM) can treat
it exactly like a .py file.

Each cell is wrapped in a named function ``cell_N_<slug>`` so the function
extractor finds real callable units and hotspot scores are meaningful.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _slug(source: str, max_len: int = 30) -> str:
    """Turn the first non-empty line of a cell into a valid Python identifier."""
    first = next((ln.strip() for ln in source.splitlines() if ln.strip()), "block")
    # strip comment markers and common notebook noise
    first = re.sub(r"^#+\s*", "", first)
    first = re.sub(r"[^a-zA-Z0-9]+", "_", first)
    return first[:max_len].strip("_").lower() or "block"


def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln for ln in text.splitlines())


def _sanitize_cell(source: str) -> str:
    """
    Convert IPython-specific syntax to valid Python so ast.parse succeeds.

    Rules:
    - ``!shell command``   → ``# SHELL: shell command``
    - ``%magic ...``       → ``# MAGIC: %magic ...``
    - ``%%cell_magic ...`` → ``# CELL_MAGIC: %%cell_magic ...``
    - ``?expr`` / ``expr?``→ ``# HELP: expr``
    """
    cleaned: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent   = line[: len(line) - len(stripped)]

        if stripped.startswith("%%"):
            cleaned.append(indent + "# CELL_MAGIC: " + stripped)
        elif stripped.startswith("!"):
            cleaned.append(indent + "# SHELL: " + stripped[1:].strip())
        elif stripped.startswith("%"):
            cleaned.append(indent + "# MAGIC: " + stripped)
        elif stripped.endswith("?") and not stripped.startswith("#"):
            cleaned.append(indent + "# HELP: " + stripped.rstrip("?"))
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


def extract_notebook_source(path: str) -> Optional[str]:
    """
    Parse *path* (.ipynb) and return a synthetic Python source string.

    Rules
    -----
    - Markdown / raw cells are included as docstrings so the LLM has context.
    - Each code cell is placed inside a generated function ``cell_N_<slug>``
      so AST extraction sees real function boundaries.
    - A bottom-level ``if __name__ == "__main__":`` block calls every cell
      function in order (mirrors notebook execution order).
    - Returns None if the file cannot be parsed.
    """
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    cells: List[dict] = nb.get("cells", [])
    parts: List[str] = [
        f'"""Synthesised from notebook: {Path(path).name}"""',
        "",
        "# ── auto-generated imports placeholder ──────────────────────────────",
        "from __future__ import annotations",
        "",
    ]

    cell_fn_names: List[str] = []
    code_cell_idx = 0

    for cell in cells:
        cell_type   = cell.get("cell_type", "")
        raw_source  = "".join(cell.get("source", []))

        if not raw_source.strip():
            continue

        if cell_type == "markdown":
            # Keep as a module-level comment block for context
            lines = raw_source.splitlines()
            parts.append("")
            parts.append("# " + " ".join(lines[:3])[:120])  # one-liner summary
            continue

        if cell_type != "code":
            continue

        code_cell_idx += 1
        fn_name = f"cell_{code_cell_idx}_{_slug(raw_source)}"
        cell_fn_names.append(fn_name)

        # Build the function wrapper
        indented = _indent(_sanitize_cell(raw_source))
        fn_lines = [
            "",
            f"def {fn_name}():",
            f'    """Cell {code_cell_idx} of notebook."""',
            indented,
            "",
        ]
        parts.extend(fn_lines)

    # Main block — sequential execution, mirrors notebook run order
    if cell_fn_names:
        parts.append("")
        parts.append('if __name__ == "__main__":')
        for fn in cell_fn_names:
            parts.append(f"    {fn}()")

    return "\n".join(parts)


def notebook_to_file_meta(path: str, repo_root: str) -> Optional[Dict]:
    """
    Convert a .ipynb file into a file-metadata dict compatible with the
    rest of the pipeline (same schema as ``walk_repo`` output).
    Returns None if extraction fails.
    """
    synthetic_source = extract_notebook_source(path)
    if synthetic_source is None:
        return None

    fpath = Path(path)
    try:
        rel = str(fpath.relative_to(repo_root))
    except ValueError:
        rel = fpath.name

    return {
        "path":          str(fpath),
        "relative_path": rel,
        "language":      "python",
        "extension":     ".ipynb",
        "content":       synthetic_source,
        "lines":         len(synthetic_source.splitlines()),
        "size_bytes":    fpath.stat().st_size,
        "is_notebook":   True,
        "notebook_name": fpath.name,
    }
