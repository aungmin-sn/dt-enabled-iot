"""Execute the same sealed-room CO2 case in EnergyPlus when the binary is present.

The IDF prescribes the balanced fan flow from the P-Q solution. EnergyPlus then
integrates ZoneAirContaminantBalance. This module does not invent a result if
the run fails.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

from .physics import FANS, ScenarioInput, intake_supply_m3h, operating_point
from .solver_procs import run_solver
from simulation.engines.theatre_boundary import BoundaryPoint


import shutil
ENERGYPLUS_EXE = Path(os.environ.get("ENERGYPLUS_EXE", shutil.which("energyplus") or shutil.which("energyplus.exe") or "__not_configured__"))
EPW_PATH = Path(os.environ.get("ENERGYPLUS_EPW", str(Path(__file__).resolve().parents[3] / "data/rq3/weather/TMY3_NZ_AK.epw")))
RUN_ROOT = Path(__file__).resolve().parents[3] / "results" / "rq3_simulation" / "solver_runs"

# Experiment starts at 01:00 so the first hour can flush the zone to outdoor CO2
# before the sealed rise. Warmup repeats the day, and that flush keeps the
# reported start of the rise at the outdoor concentration.
EXPERIMENT_START_MIN = 60


def energyplus_available() -> dict:
    return {
        "available": ENERGYPLUS_EXE.exists() and EPW_PATH.exists(),
        "executable": str(ENERGYPLUS_EXE) if ENERGYPLUS_EXE.exists() else None,
        "weather": EPW_PATH.name if EPW_PATH.exists() else None,
    }


def _hhmm(minute_of_day: int) -> str:
    hour, minute = divmod(int(minute_of_day), 60)
    return f"{hour:02d}:{minute:02d}"


def _compact(name: str, segments: list[tuple[int, float]]) -> str:
    lines = [
        "Schedule:Compact,",
        f"  {name},",
        "  Fraction,",
        "  Through: 12/31,",
        "  For: AllDays,",
    ]
    for minute, value in segments[:-1]:
        lines.append(f"  Until: {_hhmm(minute)}, {value:.6f},")
    end_minute, end_value = segments[-1]
    if end_minute != 24 * 60:
        raise ValueError("Schedule must end at 24:00.")
    lines.append(f"  Until: 24:00, {end_value:.6f};")
    return "\n".join(lines)


def _append_until(segments: list[tuple[int, float]], minute: int, value: float) -> None:
    minute = max(1, min(24 * 60, int(minute)))
    if segments and segments[-1][0] == minute:
        segments[-1] = (minute, value)
        return
    if segments and minute <= segments[-1][0]:
        return
    segments.append((minute, value))


def _theatre_schedules(spec: ScenarioInput, start: int, rise_end: int) -> dict[str, str]:
    boundary: tuple[BoundaryPoint, ...] = tuple(spec.boundary or ())
    vent = [(start, 1.0), (rise_end, 0.0)]
    gen = [(start, 0.0)]
    _append_until(gen, rise_end, 1.0 if spec.occupant_stays else 0.0)
    window = [(start, 0.0), (rise_end, 0.0)]
    heat = [(start, 0.0), (rise_end, 0.0)]
    last_vent = 1.0
    last_occ = 1.0 if spec.occupant_stays else 0.0
    last_win = 1.0 if spec.window_open else 0.0
    last_heat = 0.0
    if boundary:
        last_vent = 0.0
        for point in boundary:
            minute = rise_end + max(1, int(round(point.t_s / 60.0)))
            last_vent = float(point.flow_frac)
            last_occ = 1.0 if point.occupied else 0.0
            last_win = 1.0 if point.window_open else 0.0
            last_heat = float(point.heater_level)
            _append_until(vent, minute, last_vent)
            _append_until(gen, minute, last_occ)
            _append_until(window, minute, last_win)
            _append_until(heat, minute, last_heat)
    else:
        occupied_end = (rise_end + int(spec.purge_min)) if spec.occupant_stays else rise_end
        _append_until(gen, occupied_end, 1.0)
        last_occ = 0.0
        last_vent = 1.0
        _append_until(vent, 24 * 60, last_vent)
    _append_until(vent, 24 * 60, last_vent)
    _append_until(gen, 24 * 60, last_occ if boundary else 0.0)
    _append_until(window, 24 * 60, last_win)
    _append_until(heat, 24 * 60, last_heat)
    return {
        "vent": _compact("Vent Fraction", vent),
        "gen": _compact("CO2 Gen Fraction", gen),
        "window": _compact("Window Open Fraction", window),
        "heat": _compact("Heater Fraction", heat),
    }


def build_idf(path: Path, spec: ScenarioInput) -> dict:
    fan = FANS[spec.fan_id]
    if spec.rise_min < 1:
        raise ValueError(
            "EnergyPlus needs a sealed rise of at least 1 minute. "
            "The fast balance still runs a purge that starts immediately."
        )
    separate = intake_supply_m3h(spec)
    if separate is None:
        point = operating_point(
            fan,
            spec.n_intake,
            spec.n_exhaust,
            spec.room_pressure_pa,
            spec.fan_rpm,
            spec.fan_curve,
        )
        supply_m3h = point.balanced_flow_m3h
        pressure_pa = point.room_pressure_pa
    else:
        supply_m3h = separate
        pressure_pa = float(spec.room_pressure_pa)
    fan_fresh_m3h = supply_m3h * (1.0 - spec.recirculation_fraction)
    leakage_m3h = max(0.0, float(spec.leakage_m3h or 0.0))
    experiment_m3h = fan_fresh_m3h + leakage_m3h
    # A zero design flow leaves the warmup day unflushed, so the next day starts
    # from accumulated CO2. Keep a nominal flush flow, and scale the experiment fraction.
    fans_off = fan_fresh_m3h <= 1e-6
    if fans_off:
        flush = operating_point(fan, spec.n_intake, spec.n_exhaust, 4.0, 2860.0, "datasheet")
        design_m3h = flush.balanced_flow_m3h
        flow_m3s = design_m3h / 3600.0
        purge_frac = 0.0 if design_m3h <= 1e-9 else min(1.0, experiment_m3h / design_m3h)
    else:
        flow_m3s = experiment_m3h / 3600.0
        purge_frac = 1.0
    fresh_m3h = experiment_m3h
    start = EXPERIMENT_START_MIN
    rise_end = start + int(spec.rise_min)
    purge_end = rise_end + int(spec.purge_min)
    if purge_end >= 24 * 60:
        raise ValueError("Rise plus purge must finish before midnight after the 01:00 start.")
    if int(spec.rise_min) != spec.rise_min or int(spec.purge_min) != spec.purge_min:
        raise ValueError("EnergyPlus schedule for this case uses whole minutes.")

    occupied_end = purge_end if spec.occupant_stays else rise_end
    if spec.boundary:
        schedules = _theatre_schedules(spec, start, rise_end)
        vent = schedules["vent"]
        gen = schedules["gen"]
        window_sched = schedules["window"]
        heat_sched = schedules["heat"]
    else:
        vent = _compact("Vent Fraction", [
            (start, 1.0),
            (rise_end, 0.0),
            (24 * 60, purge_frac),
        ])
        gen = _compact("CO2 Gen Fraction", [
            (start, 0.0),
            (occupied_end, 1.0),
            (24 * 60, 0.0),
        ])
        window_sched = _compact("Window Open Fraction", [
            (start, 0.0),
            (24 * 60, 1.0 if spec.window_open else 0.0),
        ])
        heat_sched = _compact("Heater Fraction", [
            (start, 0.0),
            (24 * 60, 0.0),
        ])
    gen_m3s = spec.generation_lps / 1000.0
    # 3 x 2 x 3 m box is the same provisional geometry as the modular-room twin.
    L, W, H = 3.0, 2.0, 3.0
    idf = f"""! Sealed 18 m3 CO2 rise then NF-F12 purge. Generated for the mini scenario test.
! Ventilation flow is the sealed-room fan balance, prescribed into EnergyPlus.
Version,26.1;

SimulationControl,
  No, No, No, No, Yes;

Building,
  Sealed Room CO2 Case, 0.0, Suburbs, 0.04, 0.4, MinimalShadowing, 25, 6;

Timestep,60;

RunPeriod,
  CO2 Case,
  1, 1, ,
  1, 1, ,
  Monday,
  Yes, Yes, No, Yes, Yes, No;

ScheduleTypeLimits,
  Any Number;
ScheduleTypeLimits,
  Fraction, 0.0, 1.0, Continuous;

Schedule:Constant, Outdoor CO2, Any Number, {spec.outdoor_co2_ppm:.3f};

{vent}

{gen}

{window_sched}

{heat_sched}

Material,
  Concrete, MediumRough, 0.10, 1.40, 2240, 900, 0.9, 0.7, 0.7;

Construction, Wall Construction, Concrete;
Construction, Roof Construction, Concrete;
Construction, Floor Construction, Concrete;

Site:GroundTemperature:BuildingSurface,
  18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18;

Zone,
  Room Zone, 0, 0, 0, 0, 1, 1, autocalculate, autocalculate;

GlobalGeometryRules,
  UpperLeftCorner, CounterClockWise, World;

BuildingSurface:Detailed,
  Floor, Floor, Floor Construction, Room Zone, , Ground, , NoSun, NoWind, 1.0, 4,
  0,0,0, 0,{W},0, {L},{W},0, {L},0,0;
BuildingSurface:Detailed,
  Roof, Roof, Roof Construction, Room Zone, , Outdoors, , SunExposed, WindExposed, 1.0, 4,
  0,0,{H}, {L},0,{H}, {L},{W},{H}, 0,{W},{H};
BuildingSurface:Detailed,
  South Wall, Wall, Wall Construction, Room Zone, , Outdoors, , SunExposed, WindExposed, 1.0, 4,
  0,0,0, {L},0,0, {L},0,{H}, 0,0,{H};
BuildingSurface:Detailed,
  North Wall, Wall, Wall Construction, Room Zone, , Outdoors, , SunExposed, WindExposed, 1.0, 4,
  0,{W},0, 0,{W},{H}, {L},{W},{H}, {L},{W},0;
BuildingSurface:Detailed,
  West Wall, Wall, Wall Construction, Room Zone, , Outdoors, , SunExposed, WindExposed, 1.0, 4,
  0,0,0, 0,0,{H}, 0,{W},{H}, 0,{W},0;
BuildingSurface:Detailed,
  East Wall, Wall, Wall Construction, Room Zone, , Outdoors, , SunExposed, WindExposed, 1.0, 4,
  {L},0,0, {L},{W},0, {L},{W},{H}, {L},0,{H};

ZoneAirContaminantBalance,
  Yes, Outdoor CO2, No;

ZoneContaminantSourceAndSink:CarbonDioxide,
  Occupant CO2, Room Zone, {gen_m3s:.8f}, CO2 Gen Fraction;

ZoneVentilation:DesignFlowRate,
  Balanced Fan Flow, Room Zone, Vent Fraction, Flow/Zone, {flow_m3s:.8f}, , , , Balanced,
  {pressure_pa:.4f}, 0.60, 1.0, 0.0, 0.0, 0.0,
  -100.0, , 100.0, , -100.0, , -100.0, , 100.0, , 40.0;

ZoneVentilation:DesignFlowRate,
  Window Assist, Room Zone, Window Open Fraction, Flow/Zone, 0.02000000, , , , Intake,
  0.0, 0.60, 1.0, 0.0, 0.0, 0.0,
  -100.0, , 100.0, , -100.0, , -100.0, , 100.0, , 40.0;

OtherEquipment,
  Room Heater, Electricity, Room Zone, Heater Fraction, EquipmentLevel, 400.0, , , , 0.0, 0.0, 0.0, General;

Output:Variable, Room Zone, Zone Air CO2 Concentration, Timestep;
Output:Variable, Room Zone, Zone Ventilation Standard Density Volume Flow Rate, Timestep;
Output:SQLite, SimpleAndTabular;
OutputControl:Table:Style, CommaAndHTML, JtoKWH;
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(idf, encoding="utf-8")
    return {
        "idf": str(path),
        "balanced_flow_m3h": fresh_m3h,
        "experiment_start_min": start,
        "rise_end_min": rise_end,
        "purge_end_min": purge_end,
    }


_CLOCK = re.compile(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})")


def _clock_minute(stamp: str) -> int | None:
    match = _CLOCK.search(stamp)
    if not match:
        return None
    return int(match.group(3)) * 60 + int(match.group(4))


def parse_energyplus_csv(csv_path: Path, spec: ScenarioInput) -> dict:
    start = EXPERIMENT_START_MIN
    end = start + int(spec.rise_min) + int(spec.purge_min)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("EnergyPlus CSV has no header.")
        fields = list(reader.fieldnames)
        dt_col = fields[0]
        co2_col = next((h for h in fields if "co2" in h.lower() and "concentration" in h.lower()), None)
        flow_col = next((h for h in fields if "ventilation" in h.lower() and "volume flow" in h.lower()), None)
        if co2_col is None:
            raise ValueError("EnergyPlus CSV has no zone CO2 concentration column. Inspect eplusout.err.")
        rows = []
        for row in reader:
            minute = _clock_minute(str(row.get(dt_col, "")))
            if minute is None or minute < start or minute > end:
                continue
            try:
                co2 = float(row.get(co2_col) or "nan")
            except ValueError:
                continue
            flow = 0.0
            if flow_col:
                try:
                    flow = float(row.get(flow_col) or 0.0) * 3600.0
                except ValueError:
                    flow = 0.0
            rows.append({
                "t_min": minute - start,
                "co2_ppm": round(co2, 3),
                "flow_m3h": round(max(0.0, flow), 3),
            })
    if not rows:
        raise ValueError("EnergyPlus CSV did not contain the experiment window.")
    at_rise = next((r for r in rows if r["t_min"] == int(spec.rise_min)), None)
    if at_rise is None:
        raise ValueError("EnergyPlus CSV is missing the timestamp at the end of the rise.")
    return {
        "rows": rows,
        "co2_at_rise_end_ppm": at_rise["co2_ppm"],
        "co2_column": co2_col,
    }


def run_energyplus(spec: ScenarioInput, work_dir: Path | None = None, timeout_s: int = 180) -> dict:
    status = energyplus_available()
    if not status["available"]:
        return {
            "ok": False,
            "source_mode": "energyplus",
            "message": "EnergyPlus executable or the Auckland EPW file is not available.",
            **status,
        }
    work = work_dir or (RUN_ROOT / "latest")
    if work.exists():
        for child in work.iterdir():
            if child.is_file():
                child.unlink()
    work.mkdir(parents=True, exist_ok=True)
    idf_path = work / "sealed_room_co2.idf"
    meta = build_idf(idf_path, spec)
    cmd = [str(ENERGYPLUS_EXE), "-w", str(EPW_PATH), "-d", str(work), "-p", "eplus", "-r", str(idf_path)]
    proc = run_solver(cmd, timeout_s=timeout_s)
    csv_path = work / "eplusout.csv"
    err_path = work / "eplusout.err"
    err_text = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
    fatal = "Fatal" in err_text or proc.returncode != 0 or not csv_path.exists()
    if fatal:
        tail = "\n".join(err_text.strip().splitlines()[-12:])
        return {
            "ok": False,
            "source_mode": "energyplus",
            "message": "EnergyPlus did not produce a CO2 trajectory.",
            "returncode": proc.returncode,
            "err_tail": tail,
            "idf": meta["idf"],
            **status,
        }
    parsed = parse_energyplus_csv(csv_path, spec)
    return {
        "ok": True,
        "source_mode": "energyplus",
        "message": "EnergyPlus 26.1 completed the sealed-room CO2 case.",
        "co2_at_rise_end_ppm": parsed["co2_at_rise_end_ppm"],
        "series": parsed["rows"],
        "co2_column": parsed["co2_column"],
        "balanced_flow_m3h": meta["balanced_flow_m3h"],
        "idf": meta["idf"],
        "csv": str(csv_path),
        **status,
    }
