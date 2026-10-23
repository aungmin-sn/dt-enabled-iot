"""Solver adapter functions used by the measured-pressure evaluation only."""
from __future__ import annotations
import math
import time
from simulation.legacy_feasibility.physics import ScenarioInput, simulate
from simulation.legacy_feasibility.contam_case import contam_available, run_contam
from simulation.legacy_feasibility.energyplus_case import energyplus_available, run_energyplus
ENERGYPLUS_FIT_RISE_MIN = 95.0
ENERGYPLUS_FIT_PURGE_MIN = 18.0


def align_after_rise(series: list, rise_min: float) -> list:
    aligned = []
    for row in series or []:
        t = float(row["t_min"]) - float(rise_min)
        if t < -1e-9:
            continue
        aligned.append({**row, "t_min": round(t, 4)})
    return aligned

def effective_flow_m3h(nominal_flow_m3h,ventilation_effectiveness_factor):
    eta=float(ventilation_effectiveness_factor)
    if not 0.0 <= eta <= 1.0: raise ValueError("ventilation_effectiveness_factor must be between 0 and 1")
    return float(nominal_flow_m3h)*eta

def _variant(base: ScenarioInput, eta: float, rise_min=None, purge_min=None, dt_s=None) -> ScenarioInput:
    effective_flow_m3h(1.0, eta)
    return ScenarioInput(
        volume_m3=base.volume_m3,
        outdoor_co2_ppm=base.outdoor_co2_ppm,
        initial_co2_ppm=base.initial_co2_ppm,
        generation_lps=base.generation_lps,
        rise_min=base.rise_min if rise_min is None else rise_min,
        purge_min=base.purge_min if purge_min is None else purge_min,
        occupant_stays=base.occupant_stays,
        n_intake=base.n_intake,
        n_exhaust=base.n_exhaust,
        fan_id=base.fan_id,
        room_pressure_pa=base.room_pressure_pa,
        fan_rpm=base.fan_rpm,
        fan_curve=base.fan_curve,
        fan_in1_rpm=getattr(base, "fan_in1_rpm", None),
        fan_in2_rpm=getattr(base, "fan_in2_rpm", None),
        recirculation_fraction=1.0 - eta,
        body_mass_kg=base.body_mass_kg,
        sex=base.sex,
        met=base.met,
        dt_s=base.dt_s if dt_s is None else dt_s,
        window_open=getattr(base, "window_open", False),
        leakage_m3h=getattr(base, "leakage_m3h", 0.0),
    )

def _interp(series, t_min):
    if not series:
        return None
    if t_min <= series[0]["t_min"]:
        return float(series[0]["co2_ppm"])
    if t_min >= series[-1]["t_min"]:
        return float(series[-1]["co2_ppm"])
    for a, b in zip(series[:-1], series[1:]):
        if a["t_min"] <= t_min <= b["t_min"]:
            dt = b["t_min"] - a["t_min"]
            if dt <= 0:
                return float(a["co2_ppm"])
            frac = (t_min - a["t_min"]) / dt
            return float(a["co2_ppm"]) + (float(b["co2_ppm"]) - float(a["co2_ppm"])) * frac
    return None

def residual_series(measured, simulated) -> list[dict]:
    out = []
    for row in measured or []:
        t_min = float(row["t_min"])
        physical = float(row["co2_ppm"])
        predicted = _interp(simulated, t_min)
        if predicted is None:
            continue
        out.append({
            "t_min": t_min,
            "physical_co2_ppm": physical,
            "simulated_co2_ppm": round(predicted, 3),
            "residual_ppm": round(predicted - physical, 3),
            "physical_origin": "PHYSICAL",
            "simulated_origin": "SIMULATED",
        })
    return out

def _metrics(measured, simulated):
    if not simulated:
        return {"status": "NO_SERIES"}
    residuals = residual_series(measured, simulated)
    if not residuals:
        return {"status": "NO_OVERLAP"}
    errs = [row["residual_ppm"] for row in residuals]
    t1000 = None
    for point in simulated:
        if point["co2_ppm"] <= 1000:
            t1000 = float(point["t_min"])
            break
    mean_bias = sum(errs) / len(errs)
    return {
        "mae_ppm": sum(abs(e) for e in errs) / len(errs),
        "rmse_ppm": math.sqrt(sum(e * e for e in errs) / len(errs)),
        "mean_bias_ppm": mean_bias,
        "final_co2_error_ppm": residuals[-1]["residual_ppm"],
        "time_to_1000_min": t1000,
        "compared_points": len(errs),
        "residual_series": residuals,
    }

def _pack(engine, eta, series, runtime_s, ok, notes, measured, eta_frozen=False):
    slim = [{"t_min": float(p["t_min"]), "co2_ppm": float(p["co2_ppm"])} for p in series or []]
    out = {
        "engine": engine,
        "ventilation_effectiveness_factor": eta,
        "ventilation_effectiveness_factor_reused": bool(eta_frozen),
        "ok": bool(ok),
        "origin": "SIMULATED",
        "runtime_seconds": runtime_s,
        "notes": notes,
        "series": slim,
        "residual_series": [],
    }
    if ok:
        out.update(_metrics(measured, slim))
    else:
        out["status"] = "FAILED"
    return out

def energyplus_rise_purge(spec: ScenarioInput, *, fit: bool) -> tuple[float, float]:
    if fit:
        return ENERGYPLUS_FIT_RISE_MIN, ENERGYPLUS_FIT_PURGE_MIN
    volume = float(spec.volume_m3 or 18.0)
    outdoor = float(spec.outdoor_co2_ppm)
    indoor = float(spec.initial_co2_ppm)
    generation_lps = float(spec.generation_lps)
    rate_ppm_per_min = (generation_lps / 1000.0) / volume * 1_000_000.0 * 60.0
    delta = max(0.0, indoor - outdoor)
    if rate_ppm_per_min > 0 and delta > 1:
        rise_min = float(max(1, int(round(delta / rate_ppm_per_min))))
    else:
        rise_min = 1.0
    purge_min = float(max(1, int(round(spec.purge_min))))
    return rise_min, purge_min

def _run_engine(engine, eta, spec, measured, runtime_root, *, eta_frozen, ep_rise, ep_purge, fit):
    started = time.perf_counter()
    label = f"{engine.lower()}_eta_{eta:.2f}".replace(".", "p")
    if engine == "CO2_MASS_BALANCE":
        result = simulate(_variant(spec, eta))
        return _pack(
            engine,
            eta,
            result.get("series") or [],
            time.perf_counter() - started,
            True,
            [
                "Feasibility-balance lightweight surrogate, not S01–S12 fault theatre.",
                "ventilation_effectiveness_factor is empirical, not recirculation.",
            ],
            measured,
            eta_frozen=eta_frozen,
        )
    if engine == "CONTAM":
        avail = contam_available()
        if not avail.get("available"):
            return _pack(engine, eta, [], time.perf_counter() - started, False, ["ContamX unavailable."], measured, eta_frozen=eta_frozen)
        try:
            result = run_contam(_variant(spec, eta, dt_s=60), work_dir=runtime_root / label, timeout_s=180)
            ok = bool(result.get("ok"))
            notes = [result.get("message") or "ContamX finished.", "Prescribed fan-derived outdoor-air flow."]
            return _pack(engine, eta, result.get("series") if ok else [], time.perf_counter() - started, ok, notes, measured, eta_frozen=eta_frozen)
        except Exception as exc:
            return _pack(engine, eta, [], time.perf_counter() - started, False, [str(exc)], measured, eta_frozen=eta_frozen)
    if engine == "ENERGYPLUS":
        avail = energyplus_available()
        if not avail.get("available"):
            return _pack(engine, eta, [], time.perf_counter() - started, False, ["EnergyPlus unavailable."], measured, eta_frozen=eta_frozen)
        ep_spec = _variant(spec, eta, rise_min=ep_rise, purge_min=ep_purge, dt_s=60)
        try:
            result = run_energyplus(ep_spec, work_dir=runtime_root / label, timeout_s=300)
            ok = bool(result.get("ok"))
            series = align_after_rise(result.get("series") or [], ep_rise) if ok else []
            notes = [
                result.get("message") or "EnergyPlus finished.",
                (
                    "EnergyPlus cannot set initial zone CO2. Used rise_min=95 to reach ~1880 ppm, then an 18-min purge aligned to t=0."
                    if fit else
                    f"EnergyPlus cannot set initial zone CO2. Used sealed rise_min={ep_rise:.0f} then purge_min={ep_purge:.0f} aligned to t=0."
                ),
            ]
            return _pack(engine, eta, series, time.perf_counter() - started, ok, notes, measured, eta_frozen=eta_frozen)
        except Exception as exc:
            return _pack(engine, eta, [], time.perf_counter() - started, False, [str(exc)], measured, eta_frozen=eta_frozen)
    return _pack(engine, eta, [], time.perf_counter() - started, False, [f"Unsupported engine {engine}."], measured, eta_frozen=eta_frozen)
