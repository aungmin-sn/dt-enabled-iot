"""Single-zone CO2 balance for a sealed 18 m3 room and three NF-F12 fans.

The concentration equation is the well-mixed zone balance used by CONTAM and by
EnergyPlus ZoneAirContaminantBalance. It is a fast reduced-order result. It
becomes EnergyPlus evidence only when energyplus_case.py executes EnergyPlus.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


MMH2O_TO_PA = 9.80665


@dataclass(frozen=True, slots=True)
class FanSpec:
    fan_id: str
    name: str
    rpm: int
    free_air_m3h: float
    shutoff_mmh2o: float
    note: str
    pq_curve_mmh2o: tuple[tuple[float, float], ...] = ()


# Published NF-F12 industrialPPC-3000 curve at 3000 rpm. Pressure mmH2O, airflow m3/h.
IPPC3000_PQ_CURVE_MMH2O = (
    (0.00, 186.70),
    (2.49, 139.00),
    (3.76, 91.20),
    (5.48, 74.30),
    (7.63, 0.00),
)

# 30 Sep 2026 purge. Mean logged speed at 100% duty was 2860 rpm.
# The SDP810 room differential during that purge was about 4 Pa.
ACTUAL_CO2_DROP_RPM = 2860.0
ACTUAL_CO2_DROP_STATIC_PA = 4.0
ACTUAL_CO2_DROP_OUTDOOR_PPM = 462.0
ACTUAL_CO2_DROP_INITIAL_PPM = 1880.0
ACTUAL_CO2_DROP_MINUTES = 18.35


FANS: dict[str, FanSpec] = {
    "ippc3000": FanSpec(
        fan_id="ippc3000",
        name="Noctua NF-F12 industrialPPC-3000 PWM",
        rpm=3000,
        free_air_m3h=186.7,
        shutoff_mmh2o=7.63,
        note=(
            "Datasheet at 3000 rpm: 186.7 m3/h free air, 7.63 mmH2O shutoff, both ±10%. "
            "100% PWM is 3000 rpm ±10%. The published curve has five points, not only the two endpoints."
        ),
        pq_curve_mmh2o=IPPC3000_PQ_CURVE_MMH2O,
    ),
    "pwm1500": FanSpec(
        fan_id="pwm1500",
        name="Noctua NF-F12 PWM",
        rpm=1500,
        free_air_m3h=93.4,
        shutoff_mmh2o=2.61,
        note="Retail NF-F12 PWM: 1500 rpm, 93.4 m3/h free air, 2.61 mmH2O shutoff.",
    ),
}


@dataclass(frozen=True, slots=True)
class ScenarioInput:
    volume_m3: float = 18.0
    outdoor_co2_ppm: float = 420.0
    initial_co2_ppm: float = 420.0
    generation_lps: float = 0.0045
    rise_min: float = 120.0
    purge_min: float = 60.0
    occupant_stays: bool = True
    n_intake: int = 2
    n_exhaust: int = 1
    fan_id: str = "ippc3000"
    room_pressure_pa: float = 4.0
    fan_rpm: float | None = None
    fan_curve: str = "straight"
    recirculation_fraction: float = 0.0
    body_mass_kg: float | None = None
    sex: str = ""
    met: float | None = None
    dt_s: float = 30.0
    window_open: bool = False
    intake_on: bool = True
    exhaust_on: bool = True
    intake_flow_m3h: float | None = None
    exhaust_flow_m3h: float | None = None
    boundary: tuple = ()
    leakage_m3h: float = 0.0
    fan_in1_rpm: float | None = None
    fan_in2_rpm: float | None = None


@dataclass(frozen=True, slots=True)
class FanPoint:
    fan: FanSpec
    n_intake: int
    n_exhaust: int
    pressure_fraction_of_shutoff: float
    room_pressure_pa: float
    room_pressure_mmh2o: float
    intake_each_m3h: float
    exhaust_each_m3h: float
    balanced_flow_m3h: float
    exfiltration_m3h: float
    exhaust_past_free_air: bool


@dataclass(frozen=True, slots=True)
class Summary:
    co2_at_rise_end_ppm: float
    rise_min: float
    co2_steady_purge_ppm: float
    sample_after_purge_min: float
    co2_after_purge_sample_ppm: float
    minutes_to_1000_ppm: float | None
    minutes_to_near_steady: float | None
    near_steady_band_ppm: float
    balanced_flow_m3h: float
    ach: float
    room_pressure_pa: float
    room_pressure_mmh2o: float
    intake_each_m3h: float
    exhaust_each_m3h: float
    exfiltration_m3h: float
    generation_lps: float
    outdoor_co2_ppm: float
    initial_co2_ppm: float
    volume_m3: float
    occupant_stays: bool
    fan_name: str
    fan_rpm: float
    fan_curve: str = "straight"
    recirculation_fraction: float = 0.0
    fresh_air_m3h: float = 0.0
    body_mass_kg: float | None = None
    sex: str = ""
    met: float | None = None
    headline: str = ""


def fan_balance(fan: FanSpec, n_intake: int, n_exhaust: int) -> FanPoint:
    """Solve zone pressure so intake mass flow equals exhaust mass flow.

    Each fan follows Q = Q_free * (1 - dP / P_shutoff). Intake fans work against
    positive room pressure. The exhaust fan is assisted by it, which extends the
    straight line past the free-air point when room pressure is positive.
    """
    if n_intake < 1 or n_exhaust < 1:
        raise ValueError("Need at least one intake fan and one exhaust fan.")
    if fan.free_air_m3h <= 0 or fan.shutoff_mmh2o <= 0:
        raise ValueError("Fan free-air flow and shutoff pressure must be positive.")
    p_frac = (n_intake - n_exhaust) / (n_intake + n_exhaust)
    q_in = fan.free_air_m3h * (1.0 - p_frac)
    q_ex = fan.free_air_m3h * (1.0 + p_frac)
    if q_in <= 0 or q_ex <= 0:
        raise ValueError("This fan count stalls before the room can balance.")
    through = n_intake * q_in
    p_mm = p_frac * fan.shutoff_mmh2o
    return FanPoint(
        fan=fan,
        n_intake=n_intake,
        n_exhaust=n_exhaust,
        pressure_fraction_of_shutoff=p_frac,
        room_pressure_pa=p_mm * MMH2O_TO_PA,
        room_pressure_mmh2o=p_mm,
        intake_each_m3h=q_in,
        exhaust_each_m3h=q_ex,
        balanced_flow_m3h=through,
        exfiltration_m3h=0.0,
        exhaust_past_free_air=q_ex > fan.free_air_m3h + 1e-9,
    )


def fan_at_pressure(fan: FanSpec, n_intake: int, n_exhaust: int, room_pressure_pa: float) -> FanPoint:
    """Fan flows at a measured room pressure, from the straight-line P-Q curve.

    Intake fans work against that pressure. The exhaust fan is assisted by it.
    Fresh-air exchange is the outdoor air the intakes deliver. Any surplus leaves
    through openings other than the exhaust fan.
    """
    if n_intake < 1 or n_exhaust < 1:
        raise ValueError("Need at least one intake fan and one exhaust fan.")
    if fan.free_air_m3h <= 0 or fan.shutoff_mmh2o <= 0:
        raise ValueError("Fan free-air flow and shutoff pressure must be positive.")
    if room_pressure_pa < 0:
        raise ValueError("Room pressure must be zero or positive.")
    p_shutoff_pa = fan.shutoff_mmh2o * MMH2O_TO_PA
    p_frac = room_pressure_pa / p_shutoff_pa
    q_in = max(0.0, fan.free_air_m3h * (1.0 - p_frac))
    q_ex = fan.free_air_m3h * (1.0 + p_frac)
    supply = n_intake * q_in
    exhaust_total = n_exhaust * q_ex
    return FanPoint(
        fan=fan,
        n_intake=n_intake,
        n_exhaust=n_exhaust,
        pressure_fraction_of_shutoff=p_frac,
        room_pressure_pa=room_pressure_pa,
        room_pressure_mmh2o=room_pressure_pa / MMH2O_TO_PA,
        intake_each_m3h=q_in,
        exhaust_each_m3h=q_ex,
        balanced_flow_m3h=supply,
        exfiltration_m3h=supply - exhaust_total,
        exhaust_past_free_air=q_ex > fan.free_air_m3h + 1e-9,
    )


def flow_on_pq_curve(points: tuple[tuple[float, float], ...], pressure_mm: float) -> float:
    """Airflow at pressure_mm. Points are (mmH2O, m3/h), pressure increasing."""
    if len(points) < 2:
        raise ValueError("A pressure-flow curve needs at least two points.")
    if pressure_mm <= points[0][0]:
        return points[0][1]
    if pressure_mm >= points[-1][0]:
        return points[-1][1]
    for (p0, q0), (p1, q1) in zip(points, points[1:]):
        if pressure_mm <= p1:
            span = p1 - p0
            if span <= 0:
                return q1
            frac = (pressure_mm - p0) / span
            return q0 + frac * (q1 - q0)
    return points[-1][1]


def operating_point(
    fan: FanSpec,
    n_intake: int,
    n_exhaust: int,
    room_pressure_pa: float,
    rpm: float | None = None,
    curve: str = "straight",
) -> FanPoint:
    """Fan flows at a measured static pressure and an operating RPM.

    Datasheet airflow and shutoff pressure are at fan.rpm. Airflow scales with
    RPM and pressure with RPM squared. curve='straight' keeps the default
    what-if line between free air and shutoff. curve='datasheet' reads intake
    flow from the published points. The exhaust fan is still assisted by room
    pressure, extending past free air on that straight line.
    """
    if n_intake < 1 or n_exhaust < 1:
        raise ValueError("Need at least one intake fan and one exhaust fan.")
    if fan.free_air_m3h <= 0 or fan.shutoff_mmh2o <= 0 or fan.rpm <= 0:
        raise ValueError("Fan free-air flow, shutoff pressure, and rated RPM must be positive.")
    if room_pressure_pa < 0:
        raise ValueError("Fan static pressure must be zero or positive.")
    if curve not in {"straight", "datasheet"}:
        raise ValueError("Fan curve must be straight or datasheet.")
    speed = float(fan.rpm if rpm is None else rpm)
    if speed < 0:
        raise ValueError("Fan RPM cannot be negative.")
    if speed == 0:
        return FanPoint(
            fan=fan,
            n_intake=n_intake,
            n_exhaust=n_exhaust,
            pressure_fraction_of_shutoff=0.0,
            room_pressure_pa=room_pressure_pa,
            room_pressure_mmh2o=room_pressure_pa / MMH2O_TO_PA,
            intake_each_m3h=0.0,
            exhaust_each_m3h=0.0,
            balanced_flow_m3h=0.0,
            exfiltration_m3h=0.0,
            exhaust_past_free_air=False,
        )
    ratio = speed / fan.rpm
    q_free = fan.free_air_m3h * ratio
    p_shut_mm = fan.shutoff_mmh2o * ratio * ratio
    p_shut_pa = p_shut_mm * MMH2O_TO_PA
    if room_pressure_pa > p_shut_pa + 1e-6:
        raise ValueError(
            f"Fan static pressure must be between 0 and {p_shut_pa:.1f} Pa at {speed:g} rpm."
        )
    p_frac = room_pressure_pa / p_shut_pa
    p_mm = room_pressure_pa / MMH2O_TO_PA
    if curve == "datasheet":
        if not fan.pq_curve_mmh2o:
            raise ValueError(f"{fan.name} has no published pressure-flow curve.")
        scaled = tuple((p * ratio * ratio, q * ratio) for p, q in fan.pq_curve_mmh2o)
        q_in = max(0.0, flow_on_pq_curve(scaled, p_mm))
    else:
        q_in = max(0.0, q_free * (1.0 - p_frac))
    q_ex = q_free * (1.0 + p_frac)
    supply = n_intake * q_in
    exhaust_total = n_exhaust * q_ex
    return FanPoint(
        fan=fan,
        n_intake=n_intake,
        n_exhaust=n_exhaust,
        pressure_fraction_of_shutoff=p_frac,
        room_pressure_pa=room_pressure_pa,
        room_pressure_mmh2o=p_mm,
        intake_each_m3h=q_in,
        exhaust_each_m3h=q_ex,
        balanced_flow_m3h=supply,
        exfiltration_m3h=supply - exhaust_total,
        exhaust_past_free_air=q_ex > q_free + 1e-9,
    )


def intake_supply_m3h(spec: ScenarioInput) -> float | None:
    """Sum of the two intake operating points when each fan has its own speed.

    Returns None when the scenario still uses the single fan_rpm input.
    """
    if spec.fan_in1_rpm is None or spec.fan_in2_rpm is None:
        return None
    fan = FANS[spec.fan_id]
    total = 0.0
    pressure = float(spec.room_pressure_pa)
    for speed in (float(spec.fan_in1_rpm), float(spec.fan_in2_rpm)):
        if speed <= 0.0:
            continue
        point = operating_point(fan, 1, 1, pressure, speed, spec.fan_curve)
        total += float(point.intake_each_m3h)
    return total


def actual_co2_drop_scenario() -> ScenarioInput:
    """30 Sep 2026 purge: logged RPM, outdoor CO2, and the datasheet operating point."""
    return ScenarioInput(
        volume_m3=18.0,
        outdoor_co2_ppm=ACTUAL_CO2_DROP_OUTDOOR_PPM,
        initial_co2_ppm=ACTUAL_CO2_DROP_INITIAL_PPM,
        generation_lps=0.0045,
        rise_min=0.0,
        purge_min=ACTUAL_CO2_DROP_MINUTES,
        occupant_stays=True,
        fan_id="ippc3000",
        room_pressure_pa=ACTUAL_CO2_DROP_STATIC_PA,
        fan_rpm=ACTUAL_CO2_DROP_RPM,
        fan_curve="datasheet",
        dt_s=5.0,
    )


def concentration_ppm(
    c0_ppm: float,
    generation_m3s: float,
    flow_m3s: float,
    volume_m3: float,
    outdoor_ppm: float,
    t_s: float,
) -> float:
    """Well-mixed CO2. generation_m3s is pure CO2. flow_m3s is outdoor-air exchange."""
    if volume_m3 <= 0:
        raise ValueError("Room volume must be positive.")
    if t_s <= 0:
        return c0_ppm
    if flow_m3s <= 1e-12:
        return c0_ppm + (generation_m3s / volume_m3) * 1_000_000.0 * t_s
    c_ss = outdoor_ppm + (generation_m3s / flow_m3s) * 1_000_000.0
    tau = volume_m3 / flow_m3s
    return c_ss + (c0_ppm - c_ss) * math.exp(-t_s / tau)


def _minutes_to_level(
    c_start: float,
    c_target: float,
    c_ss: float,
    tau_s: float,
) -> float | None:
    """Minutes to reach c_target while moving from c_start toward c_ss."""
    if tau_s <= 0:
        return None
    span = c_start - c_ss
    remain = c_target - c_ss
    if span == 0:
        return 0.0 if c_start == c_target else None
    if span > 0:
        if c_target >= c_start or c_target <= c_ss:
            return 0.0 if c_target >= c_start else None
    else:
        if c_target <= c_start or c_target >= c_ss:
            return 0.0 if c_target <= c_start else None
    ratio = remain / span
    if ratio <= 0 or ratio >= 1:
        return None
    return (-math.log(ratio) * tau_s) / 60.0


# 70 kg male, Schofield BMR for ages 18–30, 0.7 met sleep, respiratory quotient 0.85.
RQ = 0.85


def bmr_mj_per_day(mass_kg: float, sex: str) -> float:
    """Schofield resting metabolism for ages 18–30, MJ/day."""
    if sex == "female":
        return 0.062 * mass_kg + 2.036
    if sex == "male":
        return 0.063 * mass_kg + 2.896
    raise ValueError("sex must be female or male.")


def co2_generation_lps(mass_kg: float, sex: str, met: float, rq: float = RQ) -> float:
    """Persily-style CO2 generation. met 0.7 is sleep, 1.2 is light desk work."""
    if mass_kg <= 0 or met <= 0 or rq <= 0:
        raise ValueError("Body mass, met, and respiratory quotient must be positive.")
    return 0.000569 * rq * met * bmr_mj_per_day(mass_kg, sex)


SLEEP_GENERATION_LPS = round(co2_generation_lps(70.0, "male", 0.7), 5)


def _fmt_duration(minutes: float) -> str:
    whole = int(round(minutes))
    if abs(minutes - whole) < 1e-6 and whole >= 60 and whole % 60 == 0:
        hours = whole // 60
        return "1 hour" if hours == 1 else f"{hours} hours"
    return f"{minutes:g} minutes"


def _fmt_ppm(ppm: float) -> str:
    if abs(ppm - round(ppm)) < 0.05:
        return f"{round(ppm):,}"
    return f"{ppm:,.1f}"


def _air_clause(s: Summary) -> str:
    if s.recirculation_fraction > 1e-9:
        return (
            f"The intakes move {s.balanced_flow_m3h:.0f} m³/h, and "
            f"{s.fresh_air_m3h:.0f} m³/h of that is outdoor air "
            f"at a recirculation fraction of {s.recirculation_fraction:.2f}."
        )
    return f"The intakes deliver {s.balanced_flow_m3h:.0f} m³/h of fresh air."


def _headline(s: Summary) -> str:
    stay = (
        f"The occupant stays in the room, so CO₂ falls toward {_fmt_ppm(s.co2_steady_purge_ppm)} ppm "
        f"and is {_fmt_ppm(s.co2_after_purge_sample_ppm)} ppm after {s.sample_after_purge_min:g} minutes at max RPM."
    )
    leave = (
        f"The occupant leaves as the fans start, so CO₂ falls toward {_fmt_ppm(s.outdoor_co2_ppm)} ppm "
        f"and is {_fmt_ppm(s.co2_after_purge_sample_ppm)} ppm after {s.sample_after_purge_min:g} minutes at max RPM."
    )
    tail = stay if s.occupant_stays else leave
    if s.met and s.body_mass_kg and s.sex:
        who = f"A {s.body_mass_kg:g} kg {s.sex} occupant at {s.met:g} met"
    else:
        who = "One occupant"
    if s.rise_min <= 0:
        return (
            f"{who} starts at {_fmt_ppm(s.initial_co2_ppm)} ppm. "
            f"The fans run at {s.fan_rpm:g} rpm and {s.room_pressure_pa:.1f} Pa. "
            f"{_air_clause(s)} "
            f"{tail}"
        )
    return (
        f"{who}, fans off, sealed {s.volume_m3:g} m³ room: CO₂ is "
        f"{_fmt_ppm(s.co2_at_rise_end_ppm)} ppm after {_fmt_duration(s.rise_min)}. "
        f"Both intake fans and the exhaust fan then run at {s.fan_rpm:g} rpm and {s.room_pressure_pa:.1f} Pa. "
        f"{_air_clause(s)} "
        f"{tail}"
    )


def simulate(spec: ScenarioInput) -> dict:
    if spec.rise_min < 0 or spec.purge_min <= 0:
        raise ValueError("Purge duration must be positive. Rise duration cannot be negative.")
    if spec.generation_lps < 0:
        raise ValueError("CO2 generation cannot be negative.")
    if spec.dt_s <= 0:
        raise ValueError("Time step must be positive.")
    if not 0.0 <= spec.recirculation_fraction <= 1.0:
        raise ValueError("Recirculation fraction must be between 0 and 1.")
    fan = FANS.get(spec.fan_id)
    if fan is None:
        raise ValueError(f"Unknown fan '{spec.fan_id}'.")

    speed = float(fan.rpm if spec.fan_rpm is None else spec.fan_rpm)
    separate_supply = intake_supply_m3h(spec)
    if separate_supply is None:
        point = operating_point(
            fan, spec.n_intake, spec.n_exhaust, spec.room_pressure_pa, speed, spec.fan_curve
        )
        supply_m3h = point.balanced_flow_m3h
    else:
        point = operating_point(fan, 1, 1, spec.room_pressure_pa, 0.0, spec.fan_curve)
        supply_m3h = separate_supply
        speed = (float(spec.fan_in1_rpm) + float(spec.fan_in2_rpm)) / 2.0
    g_rise = spec.generation_lps / 1000.0
    g_purge = g_rise if spec.occupant_stays else 0.0
    fresh_m3h = supply_m3h * (1.0 - spec.recirculation_fraction) + max(0.0, spec.leakage_m3h)
    q_purge = fresh_m3h / 3600.0

    if spec.rise_min <= 0:
        c_rise = spec.initial_co2_ppm
    else:
        c_rise = concentration_ppm(
            spec.initial_co2_ppm, g_rise, 0.0, spec.volume_m3, spec.outdoor_co2_ppm, spec.rise_min * 60.0
        )
    c_ss = spec.outdoor_co2_ppm if q_purge <= 0 else (
        spec.outdoor_co2_ppm + (g_purge / q_purge) * 1_000_000.0
    )
    tau_s = spec.volume_m3 / q_purge if q_purge > 0 else math.inf
    sample_min = min(10.0, spec.purge_min)
    c_sample = concentration_ppm(
        c_rise, g_purge, q_purge, spec.volume_m3, spec.outdoor_co2_ppm, sample_min * 60.0
    )
    band = 50.0
    to_1000 = None
    to_band = None
    if math.isfinite(tau_s):
        if c_rise > 1000.0 >= c_ss or c_rise < 1000.0 <= c_ss:
            to_1000 = _minutes_to_level(c_rise, 1000.0, c_ss, tau_s)
        near = c_ss + math.copysign(band, c_rise - c_ss) if abs(c_rise - c_ss) > band else c_ss
        if abs(c_rise - c_ss) > band:
            to_band = _minutes_to_level(c_rise, near, c_ss, tau_s)

    summary = Summary(
        co2_at_rise_end_ppm=c_rise,
        rise_min=spec.rise_min,
        co2_steady_purge_ppm=c_ss,
        sample_after_purge_min=sample_min,
        co2_after_purge_sample_ppm=c_sample,
        minutes_to_1000_ppm=to_1000,
        minutes_to_near_steady=to_band,
        near_steady_band_ppm=band,
        balanced_flow_m3h=supply_m3h,
        ach=fresh_m3h / spec.volume_m3,
        room_pressure_pa=point.room_pressure_pa,
        room_pressure_mmh2o=point.room_pressure_mmh2o,
        intake_each_m3h=point.intake_each_m3h,
        exhaust_each_m3h=point.exhaust_each_m3h,
        exfiltration_m3h=point.exfiltration_m3h,
        generation_lps=spec.generation_lps,
        outdoor_co2_ppm=spec.outdoor_co2_ppm,
        initial_co2_ppm=spec.initial_co2_ppm,
        volume_m3=spec.volume_m3,
        occupant_stays=spec.occupant_stays,
        fan_name=fan.name,
        fan_rpm=speed,
        fan_curve=spec.fan_curve,
        recirculation_fraction=spec.recirculation_fraction,
        fresh_air_m3h=fresh_m3h,
        body_mass_kg=spec.body_mass_kg,
        sex=spec.sex,
        met=spec.met,
        headline="",
    )
    summary = Summary(**{**asdict(summary), "headline": _headline(summary)})

    total_s = (spec.rise_min + spec.purge_min) * 60.0
    steps = int(math.floor(total_s / spec.dt_s + 1e-9))
    series: list[dict] = []
    for i in range(steps + 1):
        t_s = min(total_s, i * spec.dt_s)
        t_min = t_s / 60.0
        if spec.rise_min > 0 and t_min <= spec.rise_min + 1e-9:
            co2 = concentration_ppm(
                spec.initial_co2_ppm, g_rise, 0.0, spec.volume_m3, spec.outdoor_co2_ppm, t_s
            )
            phase = "rise"
            fan_on = False
            flow = 0.0
            occupancy = 1
            pressure = 0.0
        else:
            co2 = concentration_ppm(
                c_rise, g_purge, q_purge, spec.volume_m3, spec.outdoor_co2_ppm, t_s - spec.rise_min * 60.0
            )
            phase = "purge"
            fan_on = True
            flow = fresh_m3h
            occupancy = 1 if spec.occupant_stays else 0
            pressure = point.room_pressure_pa
        series.append({
            "t_min": round(t_min, 4),
            "co2_ppm": round(co2, 3),
            "phase": phase,
            "fan_on": fan_on,
            "flow_m3h": round(flow, 3),
            "occupancy": occupancy,
            "room_pressure_pa": round(pressure, 3),
        })

    assumptions = [
        "Envelope leakage is zero during the rise, matching the sealed-room scenario.",
        (
            "The modular-room twin mounts four NF-A12x25 G2 fans on a 400 mm inlet panel with a passive outlet. "
            "This case is the saved NF-F12 layout: two intake fans, one exhaust fan, and a sealed shell."
        ),
        (
            f"CO₂ generation is {spec.generation_lps:.5f} L/s"
            + (
                f" for one {spec.body_mass_kg:g} kg {spec.sex} at {spec.met:g} met, "
                "from the Schofield equation for ages 18–30 and a respiratory quotient of 0.85."
                if spec.met and spec.body_mass_kg and spec.sex
                else (
                    " for one person. 0.0045 L/s is the modular-room twin default. "
                    "0.005 L/s is 0.018 m³/h, the sedentary rate in the saved note."
                )
            )
        ),
        "The zone is fully mixed. Dead zones from fan placement are outside this model.",
        (
            f"Outdoor CO₂ is {spec.outdoor_co2_ppm:.0f} ppm. "
            "The ventilated steady level is that outdoor value plus the CO₂ the occupant is still adding."
        ),
    ]
    if separate_supply is None:
        ratio = speed / fan.rpm if fan.rpm else 1.0
        q_free = fan.free_air_m3h * ratio
        p_shut_mm = fan.shutoff_mmh2o * ratio * ratio
        if spec.fan_curve == "datasheet":
            assumptions.append(
                f"Intake flow uses the published five-point curve at {fan.rpm:g} rpm, "
                f"scaled to {speed:g} rpm. Airflow scales with rpm and pressure with rpm squared."
            )
        else:
            assumptions.append(
                f"Fan curve is the straight line from {q_free:.1f} m³/h at 0 Pa "
                f"to 0 m³/h at {p_shut_mm:.2f} mmH₂O"
                + (f", scaled from the {fan.rpm:g} rpm rating to {speed:g} rpm." if speed != fan.rpm else ".")
            )
        assumptions.append(
            (
                f"Fan static pressure is the entered value, {point.room_pressure_pa:.1f} Pa "
                f"({point.room_pressure_mmh2o:.2f} mmH₂O), at {speed:g} rpm. "
                f"Each intake then delivers {point.intake_each_m3h:.0f} m³/h and the exhaust delivers {point.exhaust_each_m3h:.0f} m³/h. "
                f"The intake total is {point.balanced_flow_m3h:.0f} m³/h. "
                f"Recirculation fraction is {spec.recirculation_fraction:.2f}, so outdoor air is {fresh_m3h:.0f} m³/h. "
                f"The surplus, {point.exfiltration_m3h:.0f} m³/h, leaves through the other openings."
            ),
        )
    else:
        assumptions.append(
            f"Intake 1 runs at {float(spec.fan_in1_rpm):.0f} rpm and intake 2 at {float(spec.fan_in2_rpm):.0f} rpm. "
            f"Each operating point is read from the datasheet curve at {spec.room_pressure_pa:.2f} Pa, "
            f"and the intake total is {supply_m3h:.1f} m³/h. "
            f"Recirculation fraction is {spec.recirculation_fraction:.2f}, so outdoor air is {fresh_m3h:.0f} m³/h."
        )
    if point.exhaust_past_free_air:
        assumptions.append(
            f"The exhaust operating point is {point.exhaust_each_m3h:.1f} m³/h, past the "
            f"{fan.free_air_m3h:.1f} m³/h free-air rating, because room pressure assists that fan. "
            "That point is the straight-line extension of the datasheet."
        )
    assumptions.append(
        "The charted trajectory is the fast single-zone balance. "
        "EnergyPlus and CONTAM each run this same case when their buttons are used. "
        "CONTAM is one well-mixed zone with the outdoor airflow above."
    )

    return {
        "summary": asdict(summary),
        "fan": asdict(point.fan),
        "fan_point": {
            "room_pressure_pa": point.room_pressure_pa,
            "room_pressure_mmh2o": point.room_pressure_mmh2o,
            "intake_each_m3h": point.intake_each_m3h,
            "exhaust_each_m3h": point.exhaust_each_m3h,
            "balanced_flow_m3h": supply_m3h,
            "exfiltration_m3h": point.exfiltration_m3h,
            "pressure_fraction_of_shutoff": point.pressure_fraction_of_shutoff,
            "exhaust_past_free_air": point.exhaust_past_free_air,
        },
        "series": series,
        "assumptions": assumptions,
        "source_mode": "fast_reduced_order",
    }


def parse_scenario(raw: dict) -> ScenarioInput:
    def num(key: str, default: float, lo: float, hi: float) -> float:
        value = float(raw.get(key, default))
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}.")
        return value

    stays_raw = raw.get("occupant_stays", True)
    if isinstance(stays_raw, str):
        stays = stays_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        stays = bool(stays_raw)
    fan_id = str(raw.get("fan_id", "ippc3000"))
    if fan_id not in FANS:
        raise ValueError(f"Unknown fan '{fan_id}'.")
    fan = FANS[fan_id]
    curve = str(raw.get("fan_curve", "straight")).strip().lower()
    if curve not in {"straight", "datasheet"}:
        raise ValueError("fan_curve must be straight or datasheet.")
    if curve == "datasheet" and not fan.pq_curve_mmh2o:
        raise ValueError(f"{fan.name} has no published pressure-flow curve.")
    rpm = num("fan_rpm", float(fan.rpm), 0.0, float(fan.rpm) * 1.10)
    if rpm <= 0:
        shutoff_pa = 0.0
    else:
        shutoff_pa = fan.shutoff_mmh2o * MMH2O_TO_PA * (rpm / fan.rpm) ** 2
    pressure = num("room_pressure_pa", 4.0, 0.0, shutoff_pa if shutoff_pa > 0 else 0.0)
    sex = str(raw.get("sex", "male")).strip().lower()
    if sex not in {"female", "male"}:
        raise ValueError("sex must be female or male.")
    outdoor = num("outdoor_co2_ppm", 420.0, 350.0, 2000.0)
    initial = num("initial_co2_ppm", outdoor, 350.0, 5000.0)
    met = None
    mass = None
    if str(raw.get("met", "")).strip() != "":
        met = num("met", 0.7, 0.5, 4.0)
        mass = num("body_mass_kg", 70.0, 30.0, 150.0)
        generation = co2_generation_lps(mass, sex, met)
    else:
        generation = num("generation_lps", 0.0045, 0.0, 0.05)
    return ScenarioInput(
        volume_m3=num("volume_m3", 18.0, 5.0, 100.0),
        outdoor_co2_ppm=outdoor,
        initial_co2_ppm=initial,
        generation_lps=generation,
        rise_min=num("rise_min", 120.0, 0.0, 600.0),
        purge_min=num("purge_min", 60.0, 1.0, 360.0),
        occupant_stays=stays,
        n_intake=2,
        n_exhaust=1,
        fan_id=fan_id,
        room_pressure_pa=pressure,
        fan_rpm=rpm,
        fan_curve=curve,
        recirculation_fraction=num("recirculation_fraction", 0.0, 0.0, 1.0),
        body_mass_kg=mass,
        sex=sex if met is not None else "",
        met=met,
        dt_s=30.0,
    )
