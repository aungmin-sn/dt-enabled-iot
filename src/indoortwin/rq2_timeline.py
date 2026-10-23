"""Integrated timeline for the 4 October RQ2 measurement sequence.

The figure uses the same multi-panel idea as the gateway end-to-end evidence
figure: one elapsed-time axis, logged phase bands, threshold lines, and a
categorical derived-state panel. The series are IndoorTwin observations and
FusionEngine labels from EXP_RQ2_CO2_V11_20261004_211358.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

EXPERIMENT_ID = "EXP_RQ2_CO2_V11_20261004_211358"
from indoortwin.paths import ROOT
PACKAGE_ROOT = ROOT
SOURCE_DIR = ROOT / "data/rq2" / EXPERIMENT_ID
FIGURE_DIR = ROOT / "results/rq2_timeline/figures"
TABLE_DIR = ROOT / "results/rq2_timeline/tables"
PHASE_ORDER = [
    "unoccupied_baseline",
    "occupied_accumulation",
    "ventilation_60_percent",
    "ventilation_80_percent",
    "ventilation_100_percent",
    "opening_assisted",
    "post_occupancy_recovery",
]
PHASE_META = {
    "unoccupied_baseline": ("P1", "Unoccupied baseline", "Unoccupied\nbaseline", "#dbe4f0"),
    "occupied_accumulation": ("P2", "CO₂ accumulation", "CO₂\naccumulation", "#f6e7cf"),
    "ventilation_60_percent": ("P3", "Fan duty cycle 60%", "Fan duty\ncycle 60%", "#d9ead8"),
    "ventilation_80_percent": ("P4", "Fan duty cycle 80%", "Fan duty\ncycle 80%", "#d4e4f7"),
    "ventilation_100_percent": ("P5", "Fan duty cycle 100%", "Fan duty\ncycle 100%", "#e7d8f3"),
    "opening_assisted": ("P6", "Window open", "Window\nopen", "#f8e0cf"),
    "post_occupancy_recovery": ("P7", "CO₂ recovery", "CO₂\nrecovery", "#e6e6e6"),
}
DERIVED_ROWS = [
    ("occupancy_context", "Occupancy"),
    ("ventilation_state", "Ventilation"),
    ("opening_context", "Opening"),
    ("thermal_state", "Thermal"),
    ("co2_state", "CO₂ band"),
    ("safety_context", "Rule status"),
]
STATE_COLORS = {
    "OCCUPIED": "#1b7f4e",
    "UNOCCUPIED": "#d1d5db",
    "VENTILATION_ALL_RUNNING": "#9a3412",
    "VENTILATION_OFF": "#d1d5db",
    "OPENING_PRESENT": "#d97706",
    "ALL_CLOSED": "#d1d5db",
    "TEMP_HIGH": "#b45309",
    "THERMAL_NORMAL": "#94a3b8",
    "CO2_NORMAL": "#1f4e79",
    "NO_RULE_ALERT": "#64748b",
}
STATE_LABELS = {
    "OCCUPIED": "Occupied",
    "UNOCCUPIED": "Unoccupied",
    "VENTILATION_ALL_RUNNING": "All running",
    "VENTILATION_OFF": "Ventilation off",
    "OPENING_PRESENT": "Opening present",
    "ALL_CLOSED": "All closed",
    "TEMP_HIGH": "Temperature high",
    "THERMAL_NORMAL": "Thermal normal",
    "CO2_NORMAL": "CO₂ normal",
    "NO_RULE_ALERT": "No active rule",
}
PRIMARY_DERIVED = {
    "unoccupied_baseline": "Unoccupied; ventilation off; all closed; CO2 normal; no rule alert",
    "occupied_accumulation": "Occupied; ventilation off; one 5 s occupancy reversal retained in the state record",
    "ventilation_60_percent": "All three fans running after measured RPM crossed 300",
    "ventilation_80_percent": "All-running state retained",
    "ventilation_100_percent": "All-running state retained; thermal state returns to normal",
    "opening_assisted": "Opening present; all-running state retained",
    "post_occupancy_recovery": "Unoccupied after a 25 s door change; all closed",
}
GAP_S = 15.0
SHORT_EVENT_MIN = 0.5


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_inputs(source_dir: Path = SOURCE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    trajectory = pd.read_csv(source_dir / "rq2_co2_trajectory.csv")
    trajectory["sample_time"] = pd.to_datetime(trajectory["sample_time"], utc=True)
    trajectory = trajectory.sort_values("sample_time").reset_index(drop=True)
    transitions = pd.read_csv(source_dir / "rq2_derived_transitions.csv")
    transitions["transition_time"] = pd.to_datetime(transitions["transition_time"], utc=True)
    transitions = transitions.sort_values("transition_time").reset_index(drop=True)
    events = []
    with (source_dir / "rq2_co2_phase_events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return trajectory, transitions, events


def minutes_from(times: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    delta = times - origin
    return delta.dt.total_seconds().to_numpy() / 60.0


def phase_windows(events: list[dict], origin: pd.Timestamp) -> list[dict]:
    starts = {}
    windows = []
    for event in events:
        if event.get("event") == "PHASE_START":
            phase = event["phase"]
            starts[phase["id"]] = pd.Timestamp(event["event_time"])
        elif event.get("event") == "PHASE_END":
            phase_id = event["phase_id"]
            end = pd.Timestamp(event["event_time"])
            start = starts[phase_id]
            code, short, figure_label, color = PHASE_META[phase_id]
            windows.append(
                {
                    "phase_id": phase_id,
                    "code": code,
                    "short": short,
                    "label": figure_label,
                    "color": color,
                    "x0": max(0.0, (start - origin).total_seconds() / 60.0),
                    "x1": (end - origin).total_seconds() / 60.0,
                }
            )
    order = {phase_id: index for index, phase_id in enumerate(PHASE_ORDER)}
    return sorted(windows, key=lambda item: order[item["phase_id"]])


def continuous_phase_bands(phases: list[dict], xmax: float) -> list[dict]:
    """Join adjacent display bands without altering the recorded phase windows."""
    if not phases:
        return []
    boundaries = [0.0]
    for left, right in zip(phases[:-1], phases[1:]):
        boundaries.append((left["x1"] + right["x0"]) / 2.0)
    boundaries.append(xmax)
    bands = []
    for phase, x0, x1 in zip(phases, boundaries[:-1], boundaries[1:]):
        band = dict(phase)
        band["x0"] = x0
        band["x1"] = x1
        bands.append(band)
    return bands


def line_segments(frame: pd.DataFrame, column: str) -> list[tuple[np.ndarray, np.ndarray]]:
    gaps = frame["sample_time"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    cuts = np.where(gaps > GAP_S)[0]
    bounds = [0, *cuts.tolist(), len(frame)]
    segments = []
    minutes = frame["minute"].to_numpy()
    values = frame[column].to_numpy()
    for left, right in zip(bounds[:-1], bounds[1:]):
        if right - left >= 2:
            segments.append((minutes[left:right], values[left:right]))
    return segments


def physical_intervals(frame: pd.DataFrame, column: str, active_value: str) -> list[tuple[float, float]]:
    gaps = frame["sample_time"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    active = frame[column].astype(str).eq(active_value).to_numpy()
    minutes = frame["minute"].to_numpy()
    intervals = []
    start = None
    for index, is_active in enumerate(active):
        if index > 0 and gaps[index] > GAP_S and start is not None:
            intervals.append((minutes[start], minutes[index - 1] + 5.0 / 60.0))
            start = None
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            intervals.append((minutes[start], minutes[index]))
            start = None
    if start is not None:
        intervals.append((minutes[start], minutes[-1] + 5.0 / 60.0))
    return [(x0, x1 - x0) for x0, x1 in intervals if x1 > x0]


def derived_intervals(
    transitions: pd.DataFrame,
    prop: str,
    origin: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[float, float, str]]:
    rows = transitions.loc[transitions["derived_property"] == prop].sort_values("transition_time")
    intervals = []
    times = list(rows["transition_time"])
    values = list(rows["new_value"].astype(str))
    for index, (start, value) in enumerate(zip(times, values)):
        stop = times[index + 1] if index + 1 < len(times) else end
        if stop <= origin:
            continue
        x0 = max(0.0, (start - origin).total_seconds() / 60.0)
        x1 = min((stop - origin).total_seconds() / 60.0, (end - origin).total_seconds() / 60.0)
        if x1 > x0:
            intervals.append((x0, x1 - x0, value))
    return intervals


def phase_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase_id in PHASE_ORDER:
        group = frame.loc[frame["phase_id"] == phase_id]
        elapsed_min = (group["sample_time"].iloc[-1] - group["sample_time"].iloc[0]).total_seconds() / 60.0
        start_co2 = float(group["co2_indoor_ppm"].iloc[0])
        end_co2 = float(group["co2_indoor_ppm"].iloc[-1])
        code, short, _figure_label, _color = PHASE_META[phase_id]
        rows.append(
            {
                "phase_code": code,
                "phase_name": short,
                "phase_id": phase_id,
                "rows": int(len(group)),
                "observed_min": round(elapsed_min, 1),
                "duty_percent": int(group["fan_exh_pwm_setpoint"].mode().iloc[0]),
                "mean_exh_rpm": int(round(float(group["fan_exh_rpm"].mean()))),
                "mean_in1_rpm": int(round(float(group["fan_in1_rpm"].mean()))),
                "mean_in2_rpm": int(round(float(group["fan_in2_rpm"].mean()))),
                "co2_start_ppm": int(round(start_co2)),
                "co2_end_ppm": int(round(end_co2)),
                "co2_max_ppm": int(round(float(group["co2_indoor_ppm"].max()))),
                "slope_ppm_per_min": round((end_co2 - start_co2) / elapsed_min, 2),
                "outdoor_min_ppm": int(group["co2_outdoor_ppm"].min()),
                "outdoor_max_ppm": int(group["co2_outdoor_ppm"].max()),
                "primary_derived_state": PRIMARY_DERIVED[phase_id],
            }
        )
    return pd.DataFrame(rows)


def style_time_axis(ax, xlabel: str | None = None) -> None:
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)


def shade_phases(ax, phases: list[dict], label: bool = False) -> None:
    for phase in phases:
        ax.axvspan(
            phase["x0"],
            phase["x1"],
            color=phase["color"],
            alpha=0.85,
            linewidth=0,
            antialiased=False,
            zorder=0,
        )
        if label:
            ax.text(
                (phase["x0"] + phase["x1"]) / 2.0,
                1.012,
                phase["label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#1f2937",
                clip_on=False,
            )


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.065,
        1.04,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def mark_short_events(ax, reversal_min: float, door_x0: float, door_width: float) -> None:
    ax.axvline(reversal_min, color="#7c3aed", linewidth=0.8, linestyle=":", zorder=4)
    ax.axvspan(door_x0, door_x0 + door_width, color="#c2410c", alpha=0.18, linewidth=0, zorder=1)


def plot_integrated_timeline(
    frame: pd.DataFrame,
    transitions: pd.DataFrame,
    phases: list[dict],
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    apply_style()
    origin = frame["sample_time"].iloc[0]
    end = frame["sample_time"].iloc[-1]
    xmax = float(frame["minute"].iloc[-1]) + 0.4
    display_phases = continuous_phase_bands(phases, xmax)
    reversal_min = float(minutes_from(pd.Series([pd.Timestamp("2026-10-04T08:26:05.666975+00:00")]), origin)[0])
    door_start = pd.Timestamp("2026-10-04T09:37:45.808414+00:00")
    door_end = pd.Timestamp("2026-10-04T09:38:10.820212+00:00")
    door_x0 = float(minutes_from(pd.Series([door_start]), origin)[0])
    door_width = (door_end - door_start).total_seconds() / 60.0

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.7, 9.8),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.28, 1.08, 0.78, 1.18]},
    )
    fig.get_layout_engine().set(h_pad=0.045, w_pad=0.02, hspace=0.045)

    ax = axes[0]
    shade_phases(ax, display_phases, label=True)
    for minutes, values in line_segments(frame, "co2_indoor_ppm"):
        ax.plot(minutes, values, color="#1f4e79", linewidth=1.7, zorder=3)
    for minutes, values in line_segments(frame, "co2_outdoor_ppm"):
        ax.plot(minutes, values, color="#6b7280", linewidth=1.15, linestyle="-.", zorder=3)
    ax.plot([], [], color="#1f4e79", linewidth=1.7, label="Indoor CO₂")
    ax.plot([], [], color="#6b7280", linewidth=1.15, linestyle="-.", label="Outdoor CO₂")
    ax.axhline(800, color="#c2410c", linestyle=":", linewidth=1.0, label="800 ppm ventilation trigger", zorder=2)
    ax.axhline(1000, color="#b91c1c", linestyle="--", linewidth=1.0, label="1,000 ppm normal-band limit", zorder=2)
    ax.set_ylim(350, 1120)
    ax.set_ylabel("CO₂ (ppm)")
    style_time_axis(ax)
    ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=6.6, borderpad=0.3)
    panel_letter(ax, "a")

    ax = axes[1]
    shade_phases(ax, display_phases)
    mark_short_events(ax, reversal_min, door_x0, door_width)
    rpm_specs = [
        ("fan_exh_rpm", "#9a3412", "Exhaust RPM"),
        ("fan_in1_rpm", "#1d4ed8", "Intake 1 RPM"),
        ("fan_in2_rpm", "#0f766e", "Intake 2 RPM"),
    ]
    for column, color, label in rpm_specs:
        first = True
        for minutes, values in line_segments(frame, column):
            ax.plot(minutes, values, color=color, linewidth=1.15, zorder=3, label=label if first else None)
            first = False
    ax.axhline(300, color="#9a3412", linestyle="--", linewidth=0.9, label="300 RPM running threshold", zorder=2)
    ax.set_ylim(-80, 3300)
    ax.set_ylabel("Fan speed (RPM)")
    duty = ax.twinx()
    for minutes, values in line_segments(frame, "fan_exh_pwm_setpoint"):
        duty.step(minutes, values, where="post", color="#4b5563", linewidth=1.05, zorder=2)
    duty.step([], [], where="post", color="#4b5563", linewidth=1.05, label="Fan duty cycle")
    duty.set_ylim(-4, 145)
    duty.set_ylabel("Fan duty cycle (%)")
    duty.spines["top"].set_visible(False)
    style_time_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    duty_handles, duty_labels = duty.get_legend_handles_labels()
    ax.legend(handles + duty_handles, labels + duty_labels, loc="upper left", frameon=True, ncol=2, fontsize=6.5)
    running_min = float(minutes_from(pd.Series([pd.Timestamp("2026-10-04T08:55:26.045493+00:00")]), origin)[0])
    ax.annotate(
        "All-running\nafter measured RPM",
        xy=(running_min, 1880),
        xytext=(running_min + 7.5, 2550),
        fontsize=6.3,
        color="#7f1d1d",
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "lw": 0.7},
        ha="left",
        va="bottom",
        zorder=5,
    )
    panel_letter(ax, "b")

    ax = axes[2]
    shade_phases(ax, display_phases)
    lanes = [
        (2, "Occupancy present", physical_intervals(frame, "occupancy", "PRESENT"), "#1b7f4e"),
        (1, "Door open", physical_intervals(frame, "door_state", "OPEN"), "#c2410c"),
        (0, "Window open", physical_intervals(frame, "window_state", "OPEN"), "#d97706"),
    ]
    for y_pos, _label, intervals, color in lanes:
        ax.broken_barh([(0, xmax)], (y_pos - 0.34, 0.68), facecolors="#f3f4f6", edgecolors="none", zorder=2)
        if intervals:
            ax.broken_barh(intervals, (y_pos - 0.34, 0.68), facecolors=color, edgecolors="none", zorder=3)
    ax.plot([reversal_min], [2], marker="o", color="#6d28d9", markersize=5.5, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
    ax.annotate(
        "5 s occupancy reversal",
        xy=(reversal_min, 2.05),
        xytext=(reversal_min + 4.5, 2.72),
        fontsize=6.3,
        color="#5b21b6",
        arrowprops={"arrowstyle": "->", "color": "#5b21b6", "lw": 0.7},
        annotation_clip=False,
    )
    door_mid = door_x0 + door_width / 2.0
    ax.plot([door_mid], [1], marker="o", color="#c2410c", markersize=5.5, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
    ax.annotate(
        "25 s door open",
        xy=(door_mid, 1.0),
        xytext=(door_mid - 11.5, 1.0),
        fontsize=6.3,
        color="#9a3412",
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#9a3412", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Window open", "Door open", "Occupancy present"])
    ax.set_ylim(-0.7, 3.15)
    ax.set_ylabel("Recorded input")
    style_time_axis(ax)
    panel_letter(ax, "c")

    ax = axes[3]
    shade_phases(ax, display_phases)
    legend_seen = set()
    legend_handles = []
    for row_index, (prop, label) in enumerate(DERIVED_ROWS):
        y_pos = len(DERIVED_ROWS) - 1 - row_index
        intervals = derived_intervals(transitions, prop, origin, end)
        for x0, width, value in intervals:
            color = STATE_COLORS.get(value, "#111827")
            ax.broken_barh([(x0, width)], (y_pos - 0.36, 0.72), facecolors=color, edgecolors="none", zorder=3)
            if value not in legend_seen and value not in {"UNOCCUPIED", "VENTILATION_OFF", "ALL_CLOSED"}:
                legend_seen.add(value)
                legend_handles.append(Patch(facecolor=color, edgecolor="none", label=STATE_LABELS.get(value, value)))
            if width >= 8.0:
                ax.text(
                    x0 + width / 2.0,
                    y_pos,
                    STATE_LABELS.get(value, value),
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if value not in {"UNOCCUPIED", "VENTILATION_OFF", "ALL_CLOSED", "THERMAL_NORMAL"} else "#1f2937",
                    zorder=4,
                )
            elif width < SHORT_EVENT_MIN:
                ax.plot(
                    x0 + width / 2.0,
                    y_pos,
                    marker="o",
                    color=color,
                    markeredgecolor="#111827",
                    markeredgewidth=0.45,
                    markersize=4.6,
                    zorder=5,
                )
    ax.set_yticks(list(range(len(DERIVED_ROWS))))
    ax.set_yticklabels([label for _prop, label in reversed(DERIVED_ROWS)])
    ax.set_ylim(-0.62, len(DERIVED_ROWS) - 0.38)
    ax.set_ylabel("Derived state")
    style_time_axis(ax, "Elapsed time from the first retained sample (min)")
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=4,
        frameon=False,
        fontsize=6.4,
    )
    panel_letter(ax, "d")
    for axis in axes:
        axis.set_xlim(0, xmax)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    for extension in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(extension), dpi=300 if extension == ".png" else None)
    plt.close(fig)
    summary.to_csv(TABLE_DIR / "rq2_figure54_phase_summary.csv", index=False)
















TABLE53_ROWS = [
    ("ROOM01.occupancy_context", "Occupancy"),
    ("ROOM01.ventilation_state", "Ventilation"),
    ("ROOM01.opening_context", "Opening"),
    ("ROOM01.heater_context", "Heater"),
    ("FAN-EXH.health_context", "Exhaust health"),
    ("ROOM01.safety_context", "Rule status"),
    ("ROOM01.co2_state", "CO₂ band"),
    ("ROOM01.thermal_state", "Thermal"),
]
TABLE53_LABELS = {
    "OCCUPIED": "Occupied",
    "UNOCCUPIED": "Unoccupied",
    "VENTILATION_OFF": "Off",
    "VENTILATION_ALL_RUNNING": "All running",
    "VENTILATION_PARTIAL": "Partial",
    "VENTILATION_COMMANDED_NO_RESPONSE": "Commanded,\nno response",
    "ALL_CLOSED": "All closed",
    "OPENING_PRESENT": "Opening present",
    "HEATER_OFF": "Off",
    "HEATER_LEVEL1": "Level 1",
    "FAN_RESPONSE_NORMAL": "Response\nnormal",
    "COMMAND_NO_ROTATION": "No rotation",
    "NO_RULE_ALERT": "No active rule",
    "OCCUPIED_HEATING_CLOSED_ROOM": "Heating,\nclosed",
    "CO2_NORMAL": "CO₂ normal",
    "TEMP_HIGH+RH_HIGH": "T+RH high",
}
TABLE53_COLORS = {
    "OCCUPIED": "#1b7f4e",
    "UNOCCUPIED": "#d1d5db",
    "VENTILATION_OFF": "#d1d5db",
    "VENTILATION_ALL_RUNNING": "#9a3412",
    "VENTILATION_PARTIAL": "#d97706",
    "VENTILATION_COMMANDED_NO_RESPONSE": "#7f1d1d",
    "ALL_CLOSED": "#e5e7eb",
    "OPENING_PRESENT": "#d97706",
    "HEATER_OFF": "#e5e7eb",
    "HEATER_LEVEL1": "#c2410c",
    "FAN_RESPONSE_NORMAL": "#0f766e",
    "COMMAND_NO_ROTATION": "#b91c1c",
    "NO_RULE_ALERT": "#64748b",
    "OCCUPIED_HEATING_CLOSED_ROOM": "#9f1239",
    "CO2_NORMAL": "#1f4e79",
    "TEMP_HIGH+RH_HIGH": "#9a3412",
}
LIGHT_LABELS = {
    "UNOCCUPIED",
    "VENTILATION_OFF",
    "ALL_CLOSED",
    "HEATER_OFF",
    "VENTILATION_PARTIAL",
    "OPENING_PRESENT",
}


def _snapshot_value(snapshot: dict, group: str, key: str):
    return snapshot[group].get(key, {}).get("value")


def load_table53() -> pd.DataFrame:
    evidence_root = SOURCE_DIR.parent
    pilot = evidence_root / "EXP_RQ2_PHYS_V11_20261001_190449"
    p03_record = evidence_root / "EXP_RQ2_P03_RETEST_20261001_212343"
    checks = pd.read_csv(ROOT / "results/rq2_states/tables/rq2_expected_state_checks.csv")
    scored = {}
    latency = {}
    for row in checks.itertuples(index=False):
        step = str(row.step_id).replace("RQ2-", "")
        scored.setdefault(step, {})[row.entity_property] = str(row.observed)
        latency[step] = int(row.latency_ms)
    p03_summary = json.loads((p03_record / "retest_summary.json").read_text(encoding="utf-8"))
    rows = []
    with (pilot / "rq2_physical_evidence.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            step = event["step_id"].replace("RQ2-", "")
            physical = event["snapshot"]["physical"]
            derived = event["snapshot"]["derived"]
            record = {
                "step": step,
                "latency_ms": latency[step],
                "co2_ppm": float(_snapshot_value(event["snapshot"], "physical", "ROOM01.co2_indoor")),
                "occupancy": _snapshot_value(event["snapshot"], "physical", "ROOM01.occupancy"),
                "window": _snapshot_value(event["snapshot"], "physical", "WINDOW-01.state"),
                "heater": _snapshot_value(event["snapshot"], "physical", "HEATER-01.state"),
                "pwm_exh": float(_snapshot_value(event["snapshot"], "physical", "FAN-EXH.pwm_desired")),
                "rpm_exh": float(_snapshot_value(event["snapshot"], "physical", "FAN-EXH.rpm")),
                "rpm_in1": float(_snapshot_value(event["snapshot"], "physical", "FAN-IN-01.rpm")),
                "rpm_in2": float(_snapshot_value(event["snapshot"], "physical", "FAN-IN-02.rpm")),
            }
            for prop, _label in TABLE53_ROWS:
                record[prop] = scored.get(step, {}).get(prop, str(derived[prop]["value"]))
                record[f"{prop}__scored"] = prop in scored.get(step, {})
            if step == "P03":
                record["pwm_exh"] = float(p03_summary["fans"]["FAN-EXH"]["pwm"])
                record["rpm_exh"] = float(p03_summary["fans"]["FAN-EXH"]["rpm"])
                record["rpm_in1"] = float(p03_summary["fans"]["FAN-IN-01"]["rpm"])
                record["rpm_in2"] = float(p03_summary["fans"]["FAN-IN-02"]["rpm"])
            rows.append(record)
    frame = pd.DataFrame(rows)
    if list(frame["step"]) != [f"P{i:02d}" for i in range(1, 11)]:
        raise RuntimeError("Controlled-action evidence is not in P01–P10 order.")
    if int(frame.loc[frame["step"] == "P03", "latency_ms"].iloc[0]) != 2018:
        raise RuntimeError("P03 latency is not 2,018 ms.")
    return frame


def plot_table53(frame: pd.DataFrame, destination: Path) -> None:
    apply_style()
    steps = frame["step"].tolist()
    x = np.arange(len(steps))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.7, 9.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [0.70, 1.05, 0.88, 1.52]},
    )
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.02, hspace=0.05)
    band_colors = ["#f3f4f6", "#eef2ff"]

    def shade(ax) -> None:
        for index in range(len(steps)):
            ax.axvspan(index - 0.5, index + 0.5, color=band_colors[index % 2], linewidth=0, zorder=0)

    ax = axes[0]
    shade(ax)
    ax.bar(x, frame["latency_ms"], color="#1f4e79", width=0.72, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.45, 9000)
    for index, latency in enumerate(frame["latency_ms"]):
        label = f"{int(latency):,} ms" if int(latency) >= 1000 else f"{int(latency)} ms"
        ax.text(index, latency * 1.35, label, ha="center", va="bottom", fontsize=6.2, color="#1f2937")
    ax.set_ylabel("Transition latency (ms)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    panel_letter(ax, "a")

    ax = axes[1]
    shade(ax)
    width = 0.22
    rpm_specs = [
        (-width, "rpm_exh", "#9a3412", "Exhaust"),
        (0.0, "rpm_in1", "#1d4ed8", "Intake 1"),
        (width, "rpm_in2", "#0f766e", "Intake 2"),
    ]
    for offset, column, color, label in rpm_specs:
        ax.bar(x + offset, frame[column], width=width, color=color, label=label, zorder=3)
    ax.axhline(300, color="#9a3412", linestyle="--", linewidth=0.9, label="300 RPM running threshold", zorder=2)
    ax.set_ylabel("Fan speed (RPM)")
    ax.set_ylim(-40, 1450)
    duty = ax.twinx()
    duty.plot(x, frame["pwm_exh"], color="#111827", marker="o", markersize=3.5, linewidth=1.0, label="Commanded duty", zorder=4)
    duty.set_ylim(-5, 120)
    duty.set_ylabel("Commanded duty (%)")
    duty.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    duty_handles, duty_labels = duty.get_legend_handles_labels()
    ax.legend(handles + duty_handles, labels + duty_labels, loc="upper left", frameon=True, ncol=2, fontsize=6.2)
    panel_letter(ax, "b")

    ax = axes[2]
    input_rows = [
        (
            2,
            "Occupancy",
            frame["occupancy"].astype(str),
            {"PRESENT": "Present", "ABSENT": "Absent"},
            "PRESENT",
            "#1b7f4e",
        ),
        (
            1,
            "Window",
            frame["window"].astype(str),
            {"OPEN": "Open", "CLOSED": "Closed"},
            "OPEN",
            "#d97706",
        ),
        (
            0,
            "Heater",
            frame["heater"].astype(str),
            {"HEATER_LEVEL1": "Level 1", "HEATER_OFF": "Off"},
            "HEATER_LEVEL1",
            "#c2410c",
        ),
    ]
    for y_pos, _row_label, values, value_labels, active_value, active_color in input_rows:
        for index, value in enumerate(values):
            active = value == active_value
            facecolor = active_color if active else "#f3f4f6"
            text_color = "white" if active else "#374151"
            ax.bar(
                index,
                0.70,
                bottom=y_pos - 0.35,
                color=facecolor,
                edgecolor="#cbd5e1",
                linewidth=0.45,
                width=0.84,
                zorder=3,
            )
            ax.text(
                index,
                y_pos,
                value_labels.get(value, value.replace("_", " ").title()),
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
                zorder=4,
            )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Heater", "Window", "Occupancy"])
    ax.set_ylim(-0.55, 2.55)
    ax.set_ylabel("Input condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_letter(ax, "c")

    ax = axes[3]
    shade(ax)
    for row_index, (prop, label) in enumerate(TABLE53_ROWS):
        y_pos = len(TABLE53_ROWS) - 1 - row_index
        for index, value in enumerate(frame[prop].astype(str)):
            color = TABLE53_COLORS.get(value, "#111827")
            ax.bar(index, 0.72, bottom=y_pos - 0.36, color=color, width=0.86, zorder=3)
            ax.text(
                index,
                y_pos,
                TABLE53_LABELS.get(value, value),
                ha="center",
                va="center",
                fontsize=5.5,
                color="#1f2937" if value in LIGHT_LABELS else "white",
                zorder=4,
            )
            if bool(frame.iloc[index][f"{prop}__scored"]):
                ax.add_patch(
                    plt.Rectangle(
                        (index - 0.43, y_pos - 0.36),
                        0.86,
                        0.72,
                        fill=False,
                        edgecolor="#111827",
                        linewidth=1.05,
                        zorder=5,
                    )
                )
    ax.set_yticks(list(range(len(TABLE53_ROWS))))
    ax.set_yticklabels([label for _prop, label in reversed(TABLE53_ROWS)])
    ax.set_ylim(-0.55, len(TABLE53_ROWS) - 0.45)
    ax.set_ylabel("Derived state")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x, steps)
    ax.set_xlabel("Controlled physical step")
    ax.set_xlim(-0.55, len(steps) - 0.45)
    panel_letter(ax, "d")
    for axis in axes[:-1]:
        axis.set_xlim(-0.55, len(steps) - 0.45)
        axis.tick_params(axis="x", labelbottom=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.with_suffix("")
    for extension in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(extension), dpi=300 if extension == ".png" else None)
    plt.close(fig)
