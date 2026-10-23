"""Expand S01–S12 controls/faults into a piecewise boundary for physics engines.

Process faults change airflow, occupancy, openings, and generation that CONTAM
and EnergyPlus integrate. Sensor spike/dropout are observation overlays applied
after the solver; they are not invented CO2 physics.
"""
from __future__ import annotations

from typing import NamedTuple


FAN_MAX_RPM = {
    "FAN-EXH": 3000.0,
    "FAN-IN-01": 3000.0,
    "FAN-IN-02": 3000.0,
}
RPM_ON = 150.0


class BoundaryPoint(NamedTuple):
    t_s: int
    occupied: bool
    window_open: bool
    door_open: bool
    heater_level: float
    pwm_exh: float
    pwm_in1: float
    pwm_in2: float
    rpm_exh: float
    rpm_in1: float
    rpm_in2: float
    eta_scale: float
    flow_frac: float
    sensor_spike: bool
    sensor_dropout: bool
    spike_severity: float


def fan_rpm_from_pwm(pwm: float) -> float:
    if pwm is None or float(pwm) <= 0:
        return 0.0
    return FAN_MAX_RPM["FAN-EXH"] * max(0.0, min(1.0, float(pwm) / 100.0)) * 0.95


def _heater_level(state: str) -> float:
    if state == "HEATER_LEVEL1":
        return 0.5
    if state == "HEATER_LEVEL2":
        return 1.0
    return 0.0


def expand_boundary(request: dict) -> tuple[BoundaryPoint, ...]:
    sim = request.get("simulation") or {}
    step_s = int(sim.get("step_s") or 60)
    horizon_s = int(sim.get("horizon_s") or 0)
    if step_s <= 0 or horizon_s <= 0:
        return ()
    init = request.get("initial_state") or {}
    occupancy = str(init.get("occupancy") or "UNKNOWN")
    door = str(init.get("door_state") or "CLOSED").upper()
    window = str(init.get("window_state") or "CLOSED").upper()
    heater = str((init.get("heater") or {}).get("state") or "HEATER_OFF")
    fans = {}
    for fan_id in FAN_MAX_RPM:
        row = (init.get("fans") or {}).get(fan_id) or {}
        fans[fan_id] = {
            "pwm": float(row.get("pwm_percent") or 0.0),
            "rpm": float(row.get("rpm") or 0.0),
        }
    eta = float((sim.get("ventilation_effectiveness_factor") or
                 ((request.get("scenario") or {}).get("expected_context") or {}).get("ventilation_effectiveness_factor")
                 or 1.0))
    eta = max(0.0, min(1.0, eta))
    controls = sorted(request.get("controls") or [], key=lambda x: int(x.get("time_s") or 0))
    faults = list(request.get("faults") or [])

    def apply_controls(t: int) -> None:
        nonlocal occupancy, door, window, heater
        for item in controls:
            if int(item.get("time_s") or 0) != t:
                continue
            target = item.get("entity_id")
            action = item.get("action")
            if action == "SET_PWM" and target in fans:
                fans[target]["pwm"] = float(item.get("value") or 0.0)
            elif action == "SET_STATE":
                value = str(item.get("value") or "")
                if target == "WINDOW-01":
                    window = value.upper()
                elif target == "DOOR-01":
                    door = value.upper()
                elif target == "HEATER-01":
                    heater = value
                elif target in {"OCC-IN-01", "ROOM01"} and value in {"PRESENT", "ABSENT"}:
                    occupancy = value

    def active_faults(t: int) -> list[dict]:
        out = []
        for fault in faults:
            start = int(fault.get("start_s") or 0)
            dur = fault.get("duration_s")
            if t >= start and (dur is None or t < start + int(dur)):
                out.append(fault)
        return out

    points = []
    for t in range(0, horizon_s + 1, step_s):
        apply_controls(t)
        for fan_id, row in fans.items():
            row["rpm"] = fan_rpm_from_pwm(row["pwm"])
        spike = False
        dropout = False
        spike_sev = 0.0
        eta_scale = 1.0
        for fault in active_faults(t):
            typ = str(fault.get("type") or "")
            target = str(fault.get("target") or "")
            sev = float(fault.get("severity") or 0.5)
            if typ == "NO_ROTATION" and target in fans:
                fans[target]["rpm"] = 0.0
            elif typ == "RPM_DEGRADATION" and target in fans:
                fans[target]["rpm"] *= max(0.0, 1.0 - sev)
            elif typ == "UNCOMMANDED_ROTATION" and target in fans:
                fans[target]["rpm"] = max(fans[target]["rpm"], FAN_MAX_RPM[target] * max(sev, 0.3))
            elif typ == "ENVIRONMENTAL_RESPONSE_DEGRADATION":
                eta_scale *= max(0.0, 1.0 - sev)
            elif typ == "SENSOR_SPIKE" and target in {"ROOM01.co2_indoor", "ROOM01"}:
                if t == int(fault.get("start_s") or 0):
                    spike = True
                    spike_sev = sev
            elif typ == "SENSOR_DROPOUT" and target in {"ROOM01.co2_indoor", "ROOM01", "SELECTED_PROPERTY_OR_NODE"}:
                dropout = True
        mean_frac = sum(fans[k]["rpm"] / FAN_MAX_RPM[k] for k in FAN_MAX_RPM) / 3.0
        flow_frac = max(0.0, min(1.0, eta * eta_scale * mean_frac))
        if window.upper() == "OPEN" or door.upper() == "OPEN":
            flow_frac = max(flow_frac, min(1.0, flow_frac + 0.12))
        points.append(BoundaryPoint(
            t_s=t,
            occupied=occupancy == "PRESENT",
            window_open=window.upper() == "OPEN",
            door_open=door.upper() == "OPEN",
            heater_level=_heater_level(heater),
            pwm_exh=fans["FAN-EXH"]["pwm"],
            pwm_in1=fans["FAN-IN-01"]["pwm"],
            pwm_in2=fans["FAN-IN-02"]["pwm"],
            rpm_exh=round(fans["FAN-EXH"]["rpm"], 1),
            rpm_in1=round(fans["FAN-IN-01"]["rpm"], 1),
            rpm_in2=round(fans["FAN-IN-02"]["rpm"], 1),
            eta_scale=round(eta_scale, 4),
            flow_frac=round(flow_frac, 4),
            sensor_spike=spike,
            sensor_dropout=dropout,
            spike_severity=spike_sev,
        ))
    return tuple(points)


def collapse_segments(boundary: tuple[BoundaryPoint, ...], step_s: int) -> list[tuple[int, int, BoundaryPoint]]:
    if not boundary:
        return []
    def key(p: BoundaryPoint):
        return (
            p.occupied,
            p.window_open,
            round(p.rpm_exh, 0),
            round(p.rpm_in1, 0),
            round(p.rpm_in2, 0),
            round(p.eta_scale, 3),
            round(p.flow_frac, 3),
            p.heater_level,
        )
    segs = []
    start = boundary[0].t_s
    cur = boundary[0]
    for point in boundary[1:]:
        if key(point) != key(cur):
            segs.append((start, point.t_s, cur))
            start = point.t_s
            cur = point
        else:
            cur = point
    end = boundary[-1].t_s + max(1, int(step_s))
    segs.append((start, end, cur))
    return segs


def nearest_point(boundary: tuple[BoundaryPoint, ...], t_s: int) -> BoundaryPoint | None:
    if not boundary:
        return None
    chosen = boundary[0]
    for point in boundary:
        if point.t_s <= t_s:
            chosen = point
        else:
            break
    return chosen


def annotate_trajectory(trajectory: list[dict], boundary: tuple[BoundaryPoint, ...]) -> list[dict]:
    """Attach prescribed RPM and observation overlays. Does not invent solver CO2."""
    out = []
    for row in trajectory or []:
        point = dict(row)
        bound = nearest_point(boundary, int(point.get("t_s") or 0))
        if bound is None:
            out.append(point)
            continue
        if point.get("fan_exh_rpm") is None:
            point["fan_exh_rpm"] = bound.rpm_exh
            point["fan_in_01_rpm"] = bound.rpm_in1
            point["fan_in_02_rpm"] = bound.rpm_in2
            point["rpm_provenance"] = "PRESCRIBED_FROM_PACK"
        co2 = point.get("co2_indoor_ppm")
        if bound.sensor_spike and isinstance(co2, (int, float)):
            point["co2_solver_ppm"] = co2
            point["co2_indoor_ppm"] = round(co2 + 400.0 * max(bound.spike_severity, 0.25), 3)
            point["observation_overlay"] = "SENSOR_SPIKE"
        if bound.sensor_dropout:
            if "co2_solver_ppm" not in point and isinstance(co2, (int, float)):
                point["co2_solver_ppm"] = co2
            point["co2_indoor_ppm"] = None
            point["observation_overlay"] = "SENSOR_DROPOUT"
        out.append(point)
    return out
