"""
CPU-time and energy measurement.

On macOS (and Windows / non-RAPL Linux), energy is approximated via:
    energy_J = wall_time_s × (cpu_utilisation × CPU_TDP_W)

On Linux with RAPL access, pyRAPL can replace / augment this.
"""

from __future__ import annotations

import os
import subprocess
import time
import tracemalloc
from typing import Any, Callable, Dict

import psutil

from config import CPU_TDP_WATTS


def _cpu_power_w() -> float:
    """
    Estimate instantaneous CPU power in watts.
    Uses a short psutil sample to get utilisation, then scales by TDP.
    """
    try:
        util = psutil.cpu_percent(interval=0.05)
        # Idle power assumed 10 % of TDP
        idle = CPU_TDP_WATTS * 0.10
        return idle + (CPU_TDP_WATTS - idle) * (util / 100.0)
    except Exception:
        return CPU_TDP_WATTS * 0.30  # conservative fallback


# ── Subprocess measurement ────────────────────────────────────────────────────

def measure_subprocess_energy(cmd: list, cwd: str = ".") -> Dict[str, Any]:
    """
    Run *cmd* in *cwd* and measure wall-clock time, CPU seconds, and
    estimated energy.  Returns a result dict including stdout/stderr.
    """
    t0_wall = time.perf_counter()
    t0_cpu  = time.process_time()

    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    t1_wall = time.perf_counter()
    t1_cpu  = time.process_time()

    wall_s   = t1_wall - t0_wall
    cpu_s    = t1_cpu  - t0_cpu
    power_w  = _cpu_power_w()
    j        = wall_s * power_w
    kwh      = j / 3_600_000

    return {
        "wall_secs":    round(wall_s, 4),
        "cpu_secs":     round(cpu_s,  4),
        "energy_joules": round(j,     6),
        "kwh":          round(kwh,    9),
        "returncode":   proc.returncode,
        "stdout":       proc.stdout[-3000:],
        "stderr":       proc.stderr[-1000:],
    }


# ── In-process callable measurement ──────────────────────────────────────────

def measure_python_call(func: Callable, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Measure energy for an in-process callable.

    Returns ``{'result': ..., 'wall_secs': ..., 'cpu_secs': ...,
               'energy_joules': ..., 'kwh': ..., 'peak_mem_kb': ...}``
    """
    tracemalloc.start()
    t0_wall = time.perf_counter()
    t0_cpu  = time.process_time()

    result = func(*args, **kwargs)

    t1_wall = time.perf_counter()
    t1_cpu  = time.process_time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_s  = t1_wall - t0_wall
    cpu_s   = t1_cpu  - t0_cpu
    power_w = _cpu_power_w()
    j       = wall_s * power_w
    kwh     = j / 3_600_000

    return {
        "result":        result,
        "wall_secs":     round(wall_s, 4),
        "cpu_secs":      round(cpu_s,  4),
        "energy_joules": round(j,      6),
        "kwh":           round(kwh,    9),
        "peak_mem_kb":   round(peak / 1024, 2),
    }
