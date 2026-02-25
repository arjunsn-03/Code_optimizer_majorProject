"""
Smoke tests for the energy estimator and profiler.
No LLM / network required.
"""

import pytest

from estimator.energy import (
    estimate_annual_savings,
    format_energy_report,
    joules_to_kwh,
    kwh_to_co2,
)
from profiler.energy import measure_python_call


# ── joules_to_kwh ─────────────────────────────────────────────────────────────

def test_joules_to_kwh_known_value():
    assert joules_to_kwh(3_600_000) == pytest.approx(1.0, rel=1e-6)


def test_joules_to_kwh_zero():
    assert joules_to_kwh(0) == 0.0


# ── kwh_to_co2 ────────────────────────────────────────────────────────────────

def test_kwh_to_co2_proportional():
    co2_1 = kwh_to_co2(1.0)
    co2_2 = kwh_to_co2(2.0)
    assert co2_2 == pytest.approx(co2_1 * 2, rel=1e-6)


def test_kwh_to_co2_custom_intensity():
    assert kwh_to_co2(1.0, grid_intensity=0.5) == pytest.approx(0.5)


# ── estimate_annual_savings ───────────────────────────────────────────────────

def test_estimate_annual_savings_keys():
    result = estimate_annual_savings(0.001, runs_per_day=100)
    assert "annual_kwh_saved" in result
    assert "annual_co2_g_saved" in result
    assert "equivalent_km_driven" in result


def test_estimate_annual_savings_zero():
    result = estimate_annual_savings(0.0)
    assert result["annual_kwh_saved"] == 0.0
    assert result["annual_co2_g_saved"] == 0.0


# ── format_energy_report ──────────────────────────────────────────────────────

def test_format_energy_report_contains_labels():
    report = format_energy_report(0.002, 0.001)
    assert "Baseline"  in report
    assert "Optimized" in report
    assert "Delta"     in report
    assert "CO₂"       in report


def test_format_energy_report_no_saving():
    report = format_energy_report(0.001, 0.001)
    assert "+0.0%" in report or "0.0%" in report


# ── measure_python_call ───────────────────────────────────────────────────────

def test_measure_python_call_returns_result():
    metrics = measure_python_call(sum, range(1000))
    assert metrics["result"] == sum(range(1000))


def test_measure_python_call_has_energy_keys():
    metrics = measure_python_call(sum, range(100))
    assert "wall_secs"     in metrics
    assert "cpu_secs"      in metrics
    assert "kwh"           in metrics
    assert "peak_mem_kb"   in metrics


def test_measure_python_call_kwh_nonnegative():
    metrics = measure_python_call(lambda: [x**2 for x in range(500)])
    assert metrics["kwh"] >= 0.0
