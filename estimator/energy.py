"""
Energy and CO₂ accounting.

Converts raw measurements to kWh and kg CO₂e, and projects annual savings.
"""

from __future__ import annotations

from config import GRID_INTENSITY_KG_PER_KWH


def joules_to_kwh(joules: float) -> float:
    """Convert joules to kWh."""
    return joules / 3_600_000.0


def kwh_to_co2(kwh: float, grid_intensity: float = GRID_INTENSITY_KG_PER_KWH) -> float:
    """Convert kWh to kg CO₂e using the given grid intensity factor."""
    return kwh * grid_intensity


def estimate_annual_savings(
    kwh_delta_per_run: float,
    runs_per_day: int = 1_000,
) -> dict:
    """
    Project annual energy and CO₂ savings given a per-run kWh improvement.

    Returns a dict with:
        annual_kwh_saved, annual_co2_kg_saved, annual_co2_g_saved,
        equivalent_km_driven  (ICE car at ~0.21 kg CO₂/km)
    """
    annual_kwh  = kwh_delta_per_run * runs_per_day * 365
    annual_co2  = kwh_to_co2(annual_kwh)
    return {
        "annual_kwh_saved":      round(annual_kwh, 6),
        "annual_co2_kg_saved":   round(annual_co2, 6),
        "annual_co2_g_saved":    round(annual_co2 * 1_000, 3),
        "equivalent_km_driven":  round(annual_co2 / 0.21, 1),
    }


def format_energy_report(
    baseline_kwh: float,
    optimized_kwh: float,
    runs_per_day: int = 1_000,
) -> str:
    """Return a human-readable energy & CO₂ comparison string."""
    delta_kwh = baseline_kwh - optimized_kwh
    delta_pct = (delta_kwh / baseline_kwh * 100.0) if baseline_kwh > 0 else 0.0
    savings   = estimate_annual_savings(delta_kwh, runs_per_day)

    return (
        f"Baseline  : {baseline_kwh:.9f} kWh\n"
        f"Optimized : {optimized_kwh:.9f} kWh\n"
        f"Delta     : {delta_kwh:+.9f} kWh  ({delta_pct:+.1f}%)\n"
        f"CO₂ saved : {kwh_to_co2(delta_kwh) * 1_000:.4f} g CO₂e per run\n"
        f"\nAnnual savings at {runs_per_day} runs/day:\n"
        f"  Energy : {savings['annual_kwh_saved']:.4f} kWh\n"
        f"  CO₂    : {savings['annual_co2_g_saved']:.2f} g CO₂e\n"
        f"  ≈ {savings['equivalent_km_driven']:.1f} km driven (ICE car)"
    )
