"""Run the sealed-room CO2 case in NIST ContamX.

The project is one well-mixed zone. A constant-volume intake and a matching
exhaust carry the outdoor airflow from the fan curve, scaled by the
recirculation fraction. ContamX integrates the zone CO2 balance. This module
does not invent a result if ContamX fails.
"""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from dataclasses import replace

from .physics import FANS, ScenarioInput, intake_supply_m3h, operating_point
from .solver_procs import run_solver
from simulation.engines.theatre_boundary import collapse_segments


M_AIR = 28.96546
M_CO2 = 44.0095
MOLAR_VOLUME_M3 = 0.02445

CONTAMX_CANDIDATES = ()
RUN_ROOT = Path(__file__).resolve().parents[3] / "results" / "rq3_simulation" / "solver_runs"


def _contamx_path() -> Path | None:
    env = os.environ.get("CONTAMX_EXE")
    if env and Path(env).exists():
        return Path(env)
    for candidate in CONTAMX_CANDIDATES:
        if candidate.exists():
            return candidate
    found = shutil.which("contamx3.exe") or shutil.which("contamx3")
    return Path(found) if found else None


def contam_available() -> dict:
    exe = _contamx_path()
    return {
        "available": exe is not None,
        "executable": str(exe) if exe else None,
    }


def _ppm_to_cc(ppm: float) -> float:
    return ppm * 1e-6 * (M_CO2 / M_AIR)


def _cc_to_ppm(cc: float) -> float:
    return cc * (M_AIR / M_CO2) * 1_000_000.0


def _generation_kg_s(liters_per_s: float) -> float:
    return (liters_per_s / 1000.0) / MOLAR_VOLUME_M3 * (M_CO2 / 1000.0)


def _hms(minutes: float) -> str:
    seconds = int(round(minutes * 60.0))
    hour, rem = divmod(seconds, 3600)
    minute, second = divmod(rem, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _safe_point(spec: ScenarioInput, rpm: float):
    fan = FANS[spec.fan_id]
    speed = float(rpm or 0.0)
    if speed <= 0:
        return operating_point(fan, spec.n_intake, spec.n_exhaust, 0.0, 0.0, spec.fan_curve)
    try:
        return operating_point(
            fan, spec.n_intake, spec.n_exhaust, spec.room_pressure_pa, speed, spec.fan_curve
        )
    except ValueError:
        return operating_point(fan, spec.n_intake, spec.n_exhaust, 0.0, speed, spec.fan_curve)


def _fresh_m3h(spec: ScenarioInput) -> float:
    separate = intake_supply_m3h(spec)
    if separate is not None:
        return separate * (1.0 - spec.recirculation_fraction)
    fan = FANS[spec.fan_id]
    point = _safe_point(spec, spec.fan_rpm if spec.fan_rpm is not None else fan.rpm)
    return point.balanced_flow_m3h * (1.0 - spec.recirculation_fraction)


def _split_flows(spec: ScenarioInput) -> tuple[float, float]:
    leak = max(0.0, float(spec.leakage_m3h or 0.0))
    if spec.intake_flow_m3h is not None or spec.exhaust_flow_m3h is not None:
        intake = float(spec.intake_flow_m3h or 0.0) * (1.0 - spec.recirculation_fraction)
        exhaust = float(spec.exhaust_flow_m3h or 0.0) * (1.0 - spec.recirculation_fraction)
    else:
        fresh = _fresh_m3h(spec)
        intake = exhaust = fresh
    return max(0.0, intake + leak), max(0.0, exhaust + leak)


def _write_dvf(path: Path, spec: ScenarioInput) -> None:
    occupied_start = 1 if spec.rise_min > 0 or spec.occupant_stays else 0
    fan_start = 1 if spec.rise_min <= 0 else 0
    intake_start = 1 if (spec.intake_on and fan_start) else 0
    exhaust_start = 1 if (spec.exhaust_on and fan_start) else 0
    window_start = 1 if spec.window_open else 0
    lines = [
        "DiscreteValuesFile ContamW 2.1",
        "Room01 rise/purge with time-varying Occupied/Window/Intake/Exhaust",
        "01/01",
        "01/01",
        "4",
        f"Occupied\t{occupied_start}",
        f"WindowOpen\t{window_start}",
        f"IntakeOn\t{intake_start}",
        f"ExhaustOn\t{exhaust_start}",
    ]
    if spec.rise_min > 0:
        at = _hms(spec.rise_min)
        lines.append(f"01/01\t{at}\t4\t1")
        if not spec.occupant_stays:
            lines.append(f"01/01\t{at}\t1\t0")
        lines.append(f"01/01\t{at}\t3\t{1 if spec.intake_on else 0}")
        lines.append(f"01/01\t{at}\t4\t{1 if spec.exhaust_on else 0}")
    lines.append("01/01\t24:00:00\t1\t" + ("1" if spec.occupant_stays else "0"))
    lines.append("01/01\t24:00:00\t2\t" + ("1" if spec.window_open else "0"))
    lines.append("01/01\t24:00:00\t3\t" + ("1" if spec.intake_on else "0"))
    lines.append("01/01\t24:00:00\t4\t" + ("1" if spec.exhaust_on else "0"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_prj(path: Path, spec: ScenarioInput, fresh_m3h: float, end_min: int) -> None:
    intake_m3h, exhaust_m3h = _split_flows(spec)
    if intake_m3h <= 0 and exhaust_m3h <= 0:
        intake_m3h = fresh_m3h
        exhaust_m3h = fresh_m3h
    intake_m3s = max(1e-8, intake_m3h / 3600.0)
    exhaust_m3s = max(1e-8, exhaust_m3h / 3600.0)
    gen_kg_s = _generation_kg_s(spec.generation_lps)
    outdoor_cc = _ppm_to_cc(spec.outdoor_co2_ppm)
    initial_cc = _ppm_to_cc(spec.initial_co2_ppm)
    end = _hms(end_min)
    text = f"""ContamW 3.4.0.0 0
Sealed {spec.volume_m3:.2f} m3 CO2 rise then NF-F12 purge
! rows cols ud uf    T   uT     N     wH  u  Ao    a
   40   50  0  0 293.150 2    0.00 10.00 0 1.000 0.280
!  scale     us  orgRow  orgCol  invYaxis showGeom
  1.000e+00   1      38       1     0        0
! Ta       Pb      Ws    Wd    rh  day u..
293.150 101325.0  0.000 0.0 0.000 1 2 0 0 1 ! steady / fixed outdoor
293.150 101325.0  0.000 0.0 0.000 1 2 0 0 1 ! wind pressure test
null ! no weather file
null ! no contaminant file
null ! no continuous values file
scenario.dvf ! discrete values file
null ! no WPC file
null ! no EWC file
WPC description
!  Xref    Yref    Zref   angle u
   0.000   0.000   0.000   0.00 0
! epsP epsS  tShift  dStart dEnd wp mf wpctrig
  0.01 0.01 00:00:00   1/1   1/1  0  0  0
! latd  longtd   tznr  altd  Tgrnd u..
 40.00  -90.00  -6.00     0 283.15 2 0
!sim_af afcalc afmaxi afrcnvg afacnvg afrelax uac Pbldg uPb
     1      1     30  1e-005  1e-006    0.75   0 50.00   0
!   slae rs aflmaxi aflcnvg aflinit Tadj
      0   1    100  1e-006      1    0
!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc
    2             30 1.00e-004 1.00e-015 1.250         0 ! (cyclic)
          0   1  100 1.00e-006 1.00e-015 1.100 1.000   0 ! (non-trace)
          0   1  100 1.00e-006 1.00e-015 1.100 1.000   0 ! (trace)
          0   1  100  1.00e-06  1.00e-15 1.100         0 ! (cvode)
!mf_solver sim_1dz sim_1dd   celldx  sim_vjt udx
     0        1       0     1.00e-01    0     0
!cvode    rcnvg     acnvg    dtmax
   0     1.00e-06  1.00e-13   0.00
!tsdens relax tsmaxi cnvgSS densZP stackD dodMdt
     0   0.75    20     0     0     0     0
!date_st time_st  date_0 time_0   date_1 time_1    t_step   t_list   t_scrn
  Jan01 00:00:00  Jan01 00:00:00  Jan01 {end}  00:01:00 00:01:00 00:10:00
!restart  date  time
      0  Jan01 00:00:00
!list doDlg pfsave zfsave zcsave
   0     0      1      1      1
!vol ach -bw cbw exp -bw age -bw
  0   0   0   0   0   0   0   0
!rzf rzm rz1 csm srf log
  0   0   0   1   0   0
!bcx dcx pfq zfq zcq
  0   0   1   1   1
!dens   grav
 1.2041 9.8055
! 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 <- extra[]
   0   0  0  0   0   0   0   0   0   0   0   0   0   0   0   0
0 ! rvals:
!valZ valD valC
   0    0    0
!cfd   cfdcnvg  var zref maxi dtcmo solv smooth   cnvgUVW     cnvgT
   0  1.00e+00    0    0 1000     0    1      1  1.00e-03  1.00e-03
-999
1 ! contaminants:
1
1 ! species:
! # s t   molwt    mdiam       edens       decay         Dm         CCdef        Cp          Kuv     u[5]      name
  1 1 0 44.0095 0.000e+00 0.000e+00 0.000e+00 2.000e-05 {outdoor_cc:.6e} 1.000e+03 0.000e+00 1 0 0 0 0 CO2
CO2
-999
1 ! levels plus icon data:
! #  refHt   delHt  ni  u  name
  1   0.000   2.500 3 0 0 <1>
!icn col row  #
  14  10   5   0
  23  20   5   1
  15  30   5   0
-999
0 ! day-schedules:
-999
0 ! week-schedules:
-999
1 ! wind pressure profiles:
! # npts type name
  1   5    1 Wall
simple wall Cp
  0.0   0.60
 90.0  -0.50
180.0  -0.35
270.0  -0.50
360.0   0.60
-999
0 ! kinetic reactions:
-999
0 ! filter elements:
-999
0 ! filters:
-999
1 ! source/sink elements:
! #  s#  ctype    name
  1 CO2 ccf Occ_CO2
occupant CO2 generation kg/s; DVF Occupied scales it
{gen_kg_s:.6e} 0.000e+00 5 0
-999
4 ! flow elements:
! # icon dtype name
  1  23 plr_leak2 Leak
trace envelope leak so the zone remains numerically sealed
1.000e-08 1.000e-03 0.650 1.000e-03 1.000 1.000e-03 0 0 2 2 2 0
  2  27 dor_door Window
closed unless WindowOpen discrete value is 1
4.68292e-03 1.697056e-01 0.500 0.010 0.400 0.500 0.600 0 0 0
  3  23 fan_cvf IntakeFan
intake outdoor airflow m3/s
{intake_m3s:.6e} 4
  4  23 fan_cvf ExhaustFan
exhaust outdoor airflow m3/s
{exhaust_m3s:.6e} 4
-999
0 ! duct elements:
-999
0 ! control super elements:
-999
0 ! control nodes:
-999
0 ! simple AHS:
-999
1 ! zones:
! Z#  f  s#  c#  k#  l#  relHt    Vol  T0  P0  name  clr u[4]  axs cdvf <cdvf name> cfd <cfd name> <1D data:>
  1  3   0   0   0   1   0.000 {spec.volume_m3:.2f} 293.15 0 Room01 -1 0 2 0 0 0 0 0
-999
1 ! initial zone concentrations:
! Z# CO2
  1 {initial_cc:.6e}
-999
4 ! flow paths:
! P#    f  n#  m#  e#  f#  w#  a#  s#  c#  l#    X       Y      relHt  mult wPset wPmod wazm Fahs Xmax Xmin icn dir u[4] cdvf <cdvf name> cfd <cfd data[4]>
  1 1 -1 1 1 0 1 0 0 0 1 0.000 0.000 1.250 1.000 0.000 0.000 270.0 0.000 0 0 23 4 -1 0 0 0 4 0 0
  2 1 -1 1 2 0 1 0 0 0 1 0.000 0.000 1.200 1.000 0.000 0.000 270.0 0.000 0 0 23 4 -1 0 0 0 4 2 WindowOpen 0
  3 1 -1 1 3 0 0 0 0 0 1 0.000 0.000 1.200 1.000 0.000 0.000 270.0 0.000 0 0 24 4 -1 0 0 0 4 2 IntakeOn 0
  4 1 1 -1 4 0 0 0 0 0 1 0.000 0.000 1.200 1.000 0.000 0.000 270.0 0.000 0 0 24 4 -1 0 0 0 4 2 ExhaustOn 0
-999
0 ! duct junctions:
-999
0 ! initial junction concentrations:
-999
0 ! duct segments:
-999
1 ! source/sinks:
! # z e s c mult CC0 xmin ymin hmin xmax ymax hmax color u cdvf cfd
  1 1 1 0 0 1.000 0.000e+00 0 0 0 0 0 0 -1 0 2 Occupied 0
-999
0 ! occupancy schedules:
-999
0 ! exposures:
-999
0 ! annotations:
-999
* end project file.
"""
    path.write_text(text, encoding="utf-8")


def _parse_sqlite(db: Path, end_min: float) -> list[dict]:
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            """
            SELECT t.Time, z.C0
            FROM NODEZONECCDATA z
            JOIN TIME t ON t.TimeID = z.TimeID
            WHERE z.ZoneNumber = 1
            ORDER BY t.Time
            """
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise ValueError("CONTAM wrote no zone CO2 rows.")
    series = []
    for seconds, cc in rows:
        minute = float(seconds) / 60.0
        if minute > end_min + 1e-6:
            continue
        series.append({"t_min": round(minute, 4), "co2_ppm": round(_cc_to_ppm(float(cc)), 3)})
    if not series:
        raise ValueError("CONTAM CO2 rows were outside the experiment window.")
    return series


def _run_contam_once(spec: ScenarioInput, work: Path, timeout_s: int) -> dict:
    status = contam_available()
    duration = spec.rise_min + spec.purge_min
    if duration <= 0 or duration >= 24 * 60:
        return {
            "ok": False,
            "source_mode": "contam",
            "message": "CONTAM runs this case inside one day. Rise plus purge must be under 24 hours.",
            **status,
        }
    end_min = max(1, int(math.ceil(duration - 1e-9)))
    fresh = _fresh_m3h(spec)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    prj = work / "sealed_room.prj"
    _write_dvf(work / "scenario.dvf", spec)
    _write_prj(prj, spec, fresh, end_min)
    exe = _contamx_path()
    if exe is None:
        return {
            "ok": False,
            "source_mode": "contam",
            "message": "ContamX was not found. Set CONTAMX_EXE or install contamx3.exe.",
            **status,
        }
    db = work / "sealed_room.sqlite3"
    work_cwd = str(work.resolve())
    proc = None
    console = ""
    for _attempt in range(3):
        proc = run_solver(
            [str(exe), "sealed_room.prj"],
            cwd=work_cwd,
            timeout_s=timeout_s,
        )
        console = (proc.stdout or "") + (proc.stderr or "")
        (work / "contamx_console.txt").write_text(console, encoding="utf-8")
        if proc.returncode == 0 and db.exists():
            break
        time.sleep(0.35)
    if proc is None or proc.returncode != 0 or not db.exists():
        tail = "\n".join(console.strip().splitlines()[-12:])
        return {
            "ok": False,
            "source_mode": "contam",
            "message": "ContamX did not produce a CO2 trajectory.",
            "returncode": None if proc is None else proc.returncode,
            "err_tail": tail,
            "prj": str(prj),
            **status,
        }
    series = _parse_sqlite(db, duration)
    at_rise = series[0]["co2_ppm"]
    for row in series:
        if row["t_min"] + 1e-6 >= spec.rise_min:
            at_rise = row["co2_ppm"]
            break
    return {
        "ok": True,
        "source_mode": "contam",
        "message": "ContamX 3.4 completed the sealed-room CO2 case.",
        "co2_at_rise_end_ppm": at_rise,
        "series": series,
        "balanced_flow_m3h": fresh,
        "prj": str(prj),
        "sqlite": str(db),
        **status,
    }


def _segment_spec(base: ScenarioInput, point, duration_min: float, initial_co2: float) -> ScenarioInput:
    in_rpm = (point.rpm_in1 + point.rpm_in2) / 2.0
    exh_rpm = point.rpm_exh
    in_point = _safe_point(base, in_rpm)
    exh_point = _safe_point(base, exh_rpm)
    scale = (1.0 - base.recirculation_fraction) * max(0.0, min(1.0, point.eta_scale))
    return replace(
        base,
        initial_co2_ppm=initial_co2,
        rise_min=0.0,
        purge_min=max(1.0, float(duration_min)),
        occupant_stays=point.occupied,
        generation_lps=0.0045 if point.occupied else 0.0,
        fan_rpm=max(in_rpm, exh_rpm),
        window_open=point.window_open,
        intake_on=in_rpm >= 150.0,
        exhaust_on=exh_rpm >= 150.0,
        intake_flow_m3h=in_point.intake_each_m3h * base.n_intake * scale,
        exhaust_flow_m3h=exh_point.exhaust_each_m3h * base.n_exhaust * scale,
        recirculation_fraction=0.0,
        boundary=(),
        dt_s=60.0,
    )


def run_contam(spec: ScenarioInput, work_dir: Path | None = None, timeout_s: int = 180) -> dict:
    status = contam_available()
    if not status["available"]:
        return {
            "ok": False,
            "source_mode": "contam",
            "message": "ContamX was not found. Set CONTAMX_EXE or install contamx3.exe.",
            **status,
        }
    work = work_dir or (RUN_ROOT / "latest")
    boundary = tuple(spec.boundary or ())
    if not boundary:
        return _run_contam_once(spec, work, timeout_s)

    step_s = int(spec.dt_s or 60)
    segs = collapse_segments(boundary, step_s)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    series = []
    co2 = spec.initial_co2_ppm
    prj = None
    db = None
    fresh_last = 0.0
    for i, (start_s, end_s, point) in enumerate(segs):
        duration_min = max(1.0, (end_s - start_s) / 60.0)
        sub = _segment_spec(spec, point, duration_min, co2)
        result = _run_contam_once(sub, work / f"seg_{i:02d}", timeout_s)
        if not result.get("ok"):
            result["message"] = (
                f"CONTAM piecewise segment {i} (t={start_s}-{end_s}s) failed. "
                + str(result.get("message") or "")
            )
            result["segments_completed"] = i
            return result
        offset = start_s / 60.0
        for row in result.get("series") or []:
            series.append({**row, "t_min": round(row["t_min"] + offset, 4)})
        if result.get("series"):
            co2 = result["series"][-1]["co2_ppm"]
        prj = result.get("prj")
        db = result.get("sqlite")
        fresh_last = float(result.get("balanced_flow_m3h") or 0.0)
    if not series:
        return {
            "ok": False,
            "source_mode": "contam",
            "message": "CONTAM piecewise run produced no CO2 rows.",
            **status,
        }
    return {
        "ok": True,
        "source_mode": "contam",
        "message": "ContamX 3.4 integrated %d piecewise boundary segment(s) from the S01–S12 pack." % len(segs),
        "co2_at_rise_end_ppm": series[0]["co2_ppm"],
        "series": series,
        "balanced_flow_m3h": fresh_last,
        "prj": prj,
        "sqlite": db,
        "theatre_segments": len(segs),
        **status,
    }
