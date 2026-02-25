"""
Patch verification in an isolated sandbox directory.

Workflow
--------
1. Copy the repo into a temporary directory (sandbox).
2. Apply the unified-diff patch with the system ``patch`` utility.
3. Run pytest and measure wall-clock time + estimated energy.
4. Return a verdict dict.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from profiler.energy import measure_subprocess_energy


# ── Patch ─────────────────────────────────────────────────────────────────────

def apply_patch(patch_text: str, repo_path: str) -> bool:
    """
    Write *patch_text* to a temp file and apply it inside *repo_path*.
    Does a dry-run first; only applies if the dry-run succeeds.
    Returns ``True`` on success.
    """
    if not patch_text or not patch_text.strip():
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(patch_text)
        patch_file = fh.name

    try:
        # Dry run
        dry = subprocess.run(
            ["patch", "-p1", "--dry-run", "-i", patch_file],
            cwd=repo_path, capture_output=True, text=True,
        )
        if dry.returncode != 0:
            return False

        # Real apply
        subprocess.run(
            ["patch", "-p1", "-i", patch_file],
            cwd=repo_path, check=True, capture_output=True,
        )
        return True

    except Exception:
        return False
    finally:
        os.unlink(patch_file)


# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests(
    repo_path: str,
    test_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run pytest in *repo_path* and return an energy-measurement dict."""
    cmd = ["python", "-m", "pytest", "--tb=short", "-q"]
    if test_patterns:
        cmd.extend(test_patterns)
    return measure_subprocess_energy(cmd, cwd=repo_path)


# ── Full verification ─────────────────────────────────────────────────────────

def verify_patch(
    patch_text: str,
    repo_path: str,
    test_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify a patch by:
        1. Copying the repo into a fresh temp directory.
        2. Applying the patch.
        3. Running tests + measuring energy.

    Returns a verdict dict with keys:
        verdict, reason, tests_passed, wall_secs, cpu_secs, kwh,
        stdout, stderr
    """
    with tempfile.TemporaryDirectory(prefix="carbon_opt_sandbox_") as tmp:
        sandbox = os.path.join(tmp, "repo")
        shutil.copytree(
            repo_path, sandbox, symlinks=True,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".mypy_cache",
            ),
        )

        # Apply
        if not apply_patch(patch_text, sandbox):
            return {
                "verdict":      "rejected",
                "reason":       "patch_apply_failed",
                "tests_passed": False,
            }

        # Run tests
        test_result   = run_tests(sandbox, test_patterns)
        tests_passed  = test_result["returncode"] == 0

        return {
            "verdict":      "accepted" if tests_passed else "rejected",
            "reason":       "tests_passed" if tests_passed else "tests_failed",
            "tests_passed": tests_passed,
            "wall_secs":    test_result["wall_secs"],
            "cpu_secs":     test_result["cpu_secs"],
            "kwh":          test_result["kwh"],
            "stdout":       test_result["stdout"],
            "stderr":       test_result["stderr"],
        }
