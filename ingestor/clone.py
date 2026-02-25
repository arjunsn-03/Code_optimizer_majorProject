"""
Repo walker — discovers all supported source files in a local directory.
Optionally clones a remote git repo first.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Dict

from config import SUPPORTED_EXTENSIONS

# Directories to never descend into
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".eggs", ".mypy_cache", ".tox", ".pytest_cache",
    "site-packages", ".idea", ".vscode",
}


def walk_repo(repo_path: str) -> List[Dict]:
    """
    Walk *repo_path* recursively and return a list of file-metadata dicts.
    Each dict has:
        path, relative_path, language, extension, content, lines, size_bytes
    """
    repo = Path(repo_path).resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repo path not found: {repo}")

    files: List[Dict] = []

    for root, dirs, filenames in os.walk(repo):
        # Prune unwanted directories in-place
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

        for fname in filenames:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                files.append({
                    "path":          str(fpath),
                    "relative_path": str(fpath.relative_to(repo)),
                    "language":      SUPPORTED_EXTENSIONS[ext],
                    "extension":     ext,
                    "content":       content,
                    "lines":         len(content.splitlines()),
                    "size_bytes":    fpath.stat().st_size,
                })
            except Exception as exc:
                print(f"[ingestor] Skipping {fpath}: {exc}")

    return files


def clone_repo(git_url: str, dest: str) -> str:
    """
    Shallow-clone *git_url* into *dest*.
    Returns the absolute path to the cloned directory.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth=1", git_url, str(dest_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr}")

    return str(dest_path)
