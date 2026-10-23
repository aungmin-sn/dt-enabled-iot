"""Duty-cycle room pressure and per-intake fan speed for the three CO2 solvers.

Fan-on intervals use the mean measured room pressure of that 10-minute duty.
Fan-off intervals stay at 0 Pa. Each intake fan keeps its own recorded speed.
The mass balance, CONTAM, and EnergyPlus add the two datasheet operating points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulation.cases.co2_drop_validation import _run_engine, energyplus_rise_purge
from simulation.legacy_feasibility.physics import FANS, ScenarioInput, co2_generation_lps, operating_point

FAN_ON_RPM_MIN = 700.0
FIXED_FAN_ON_PRESSURE_PA = 4.0
VOLUME_M3 = 18.0
BODY_MASS_KG = 58.0
SEX = "male"
MET = 1.0
GENERATION_LPS = round(co2_generation_lps(BODY_MASS_KG, SEX, MET), 5)
# Factors selected on the 8 October fan-on intervals at a fixed 4 Pa.
# The measured-duty evaluation selects its own factors and does not reuse these values.
FROZEN_ETA = {
    "CO2_MASS_BALANCE": 0.55,
    "CONTAM": 0.60,
    "ENERGYPLUS": 0.60,
}
PRESSURE_MODES = ("fixed_4pa", "measured_duty")


def measured_mean_pressure_pa(delta_pressure) -> float:
    """Mean signed room differential pressure for one interval, in Pa."""
    values = np.asarray(delta_pressure, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Duty-cycle pressure readings are missing.")
    return float(np.mean(values))


def solver_room_pressure_pa(rpm: float, measured_mean_pa: float, mode: str) -> float:
    """Static pressure supplied to the unchanged fan curve.

    fixed_4pa repeats the existing evaluation: 4 Pa while the fans are on.
    measured_duty uses the mean measured pressure of that same fan-on interval.
    """
    if mode not in PRESSURE_MODES:
        raise ValueError(f"Unknown pressure mode '{mode}'.")
    if float(rpm) < FAN_ON_RPM_MIN:
        return 0.0
    if mode == "fixed_4pa":
        return FIXED_FAN_ON_PRESSURE_PA
    return max(0.0, float(measured_mean_pa))


def one_intake_flow_m3h(rpm: float, room_pressure_pa: float) -> float:
    """Datasheet airflow of one intake fan at its own speed and static pressure."""
    speed = float(rpm)
    pressure = float(room_pressure_pa)
    if speed <= 0.0 or pressure < 0.0:
        return 0.0
    point = operating_point(FANS["ippc3000"], 1, 1, pressure, speed, "datasheet")
    return float(point.intake_each_m3h)


def combined_intake_flow_m3h(rpm_in1: float, rpm_in2: float, room_pressure_pa: float) -> float:
    """Sum of the two intake-fan operating points."""
    return one_intake_flow_m3h(rpm_in1, room_pressure_pa) + one_intake_flow_m3h(rpm_in2, room_pressure_pa)


def build_scenario(block: dict, initial_ppm: float, room_pressure_pa: float) -> ScenarioInput:
    if GENERATION_LPS != 0.00317:
        raise RuntimeError(f"Expected generation 0.00317 L/s, got {GENERATION_LPS}")
    return ScenarioInput(
        volume_m3=VOLUME_M3,
        outdoor_co2_ppm=float(block["outdoor_ppm"]),
        initial_co2_ppm=float(initial_ppm),
        generation_lps=GENERATION_LPS,
        rise_min=0.0,
        purge_min=max(float(block["duration_min"]), 1.0 / 60.0),
        occupant_stays=True,
        n_intake=2,
        n_exhaust=1,
        fan_id="ippc3000",
        room_pressure_pa=float(room_pressure_pa),
        fan_rpm=None,
        fan_in1_rpm=float(block["rpm_in1"]),
        fan_in2_rpm=float(block["rpm_in2"]),
        fan_curve="datasheet",
        dt_s=5.0,
        window_open=False,
        leakage_m3h=0.0,
        body_mass_kg=BODY_MASS_KG,
        sex=SEX,
        met=MET,
    )


def run_block(engine: str, eta: float, block: dict, initial_ppm: float, work: Path, mode: str) -> dict:
    """Run one existing solver at the pressure selected for this interval."""
    pressure_pa = solver_room_pressure_pa(block["rpm"], block["measured_dp_pa"], mode)
    supply_m3h = combined_intake_flow_m3h(block["rpm_in1"], block["rpm_in2"], pressure_pa)
    spec = build_scenario(block, initial_ppm, pressure_pa)
    measured = [
        {"t_min": float(t), "co2_ppm": float(c)}
        for t, c in zip(block["local_t_min"], block["physical_ppm"])
    ]
    ep_rise, ep_purge = energyplus_rise_purge(spec, fit=False)
    work.mkdir(parents=True, exist_ok=True)
    result = _run_engine(
        engine,
        float(eta),
        spec,
        measured,
        work,
        eta_frozen=True,
        ep_rise=ep_rise,
        ep_purge=ep_purge,
        fit=False,
    )
    series = pd.DataFrame(result.get("series") or [])
    local = None
    ok = bool(result.get("ok")) and not series.empty
    if ok:
        series = series.sort_values("t_min")
        local = np.interp(
            block["local_t_min"],
            series["t_min"].to_numpy(dtype=float),
            series["co2_ppm"].to_numpy(dtype=float),
        )
    return {
        "ok": ok,
        "co2_ppm": local,
        "room_pressure_pa": pressure_pa,
        "intake_flow_m3h": supply_m3h,
        "notes": result.get("notes") or [],
    }
