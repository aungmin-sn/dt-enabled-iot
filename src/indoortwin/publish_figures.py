"""Build dissertation figures from computed tables and evaluated plot panels."""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def combine(inputs, destination, width=2200, crop_first=False):
    panels = []
    for index, path in enumerate(inputs):
        with Image.open(path) as opened:
            panel = opened.convert("RGB")
        if crop_first and index < len(inputs) - 1:
            panel = panel.crop((0, 0, panel.width, int(panel.height * 0.88)))
        panel = panel.resize((width, round(panel.height * width / panel.width)), Image.Resampling.LANCZOS)
        panels.append(panel)
    gap = 18
    canvas = Image.new("RGB", (width, sum(p.height for p in panels) + gap * (len(panels)-1)), "white")
    top = 0
    for panel in panels:
        canvas.paste(panel, (0, top))
        top += panel.height + gap
    canvas.save(destination, dpi=(300, 300), optimize=True)


def simulation_composite(figures):
    combine([figures / f"figure_{stamp}_whole_record.png" for stamp in ("8_Oct", "9_Oct", "10_Oct")],
            figures / "figure_measured_pressure_three_records.png", crop_first=True)


def anomaly_figures(tables: Path, figures: Path):
    """Compute plotted values from evaluated intervals; no hand-entered metrics."""
    paired = pd.read_csv(tables / "run15_rule_detector_comparison.csv")
    s07 = pd.read_csv(tables / "run15_s07_interval_results.csv")
    extra = pd.read_csv(tables / "run15_s05_s09_s12_interval_results.csv")
    values = []
    specifications = [
        ("S05", "expanded_no_response", "S05 no-response check"),
        ("S05", "fan_residual", "S05 fan residual"),
        ("S06", "expanded_degraded_response", "S06 degraded-response check"),
        ("S06", "fan_residual", "S06 fan residual"),
        ("S08", "expanded_uncommanded_rotation", "S08 rotation check"),
        ("S08", "fan_residual", "S08 fan residual"),
        ("S11", "expanded_co2_spike", "S11 spike check"),
    ]
    for scenario, method, label in specifications:
        rows = paired.loc[paired.scenario.eq(scenario) & paired.method.eq(method)]
        if rows.empty:
            raise ValueError(f"Missing evaluated comparison: {scenario}/{method}")
        detected, events = int(rows.detected.sum()), int(rows.events.sum())
        # The paired table is split by severity; use the actual pooled intervals below
        # to avoid averaging precision or F1 across unequal denominators.
        intervals = pd.read_csv(tables / "run15_expanded_s05_s06_s08_s11_intervals.csv")
        cases = intervals.loc[intervals.scenario.eq(scenario) & intervals.detector.eq(method)]
        if cases.empty:
            raise ValueError(f"Missing evaluated intervals: {scenario}/{method}")
        truth, pred = cases.truth_anomaly.eq(1), cases.predicted_anomaly.eq(1)
        tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
        if tp != detected or tp + fn != events:
            raise ValueError(f"Interval counts differ from the evaluated summary: {scenario}/{method}")
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and precision + recall > 0 else np.nan
        values.append(dict(label=label, scenario=scenario, method=method, events=events, detected=detected,
                           precision=precision, recall=recall, f1=f1))

    def from_intervals(frame, scenario, method, label):
        rows = frame.loc[frame.scenario.eq(scenario) & frame.detector.eq(method)]
        if rows.empty:
            raise ValueError(f"Missing interval data: {scenario}/{method}")
        truth, pred = rows.truth_anomaly.eq(1), rows.predicted_anomaly.eq(1)
        tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and precision + recall > 0 else np.nan
        return dict(label=label, scenario=scenario, method=method, events=tp+fn, detected=tp,
                    precision=precision, recall=recall, f1=f1)

    values.append(from_intervals(extra, "S12", "quality_flag", "S12 missing-data check"))
    values.append(from_intervals(s07, "S07", "mass_balance_co2_residual", "S07 CO₂ residual"))
    values.append(from_intervals(extra, "S09", "mass_balance_co2_residual", "S09 CO₂ residual"))
    metrics = pd.DataFrame(values)
    metrics.to_csv(tables / "dissertation_anomaly_metrics.csv", index=False)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for offset, column, color, label in [(-1, "precision", "#315A7D", "Precision"), (0, "recall", "#D9782D", "Recall"), (1, "f1", "#4F7C61", "F₁")]:
        for index, value in enumerate(metrics[column]):
            if pd.isna(value):
                ax.text(index + offset*.24, .04, "undef.", rotation=90, ha="center", fontsize=7)
        ax.bar(x + offset*.24, metrics[column], width=.23, color=color, label=label)
    ax.set_xticks(x, metrics.label, rotation=35, ha="right")
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "figure_anomaly_scores.png", dpi=200)
    fig.savefig(figures / "figure_anomaly_scores.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].barh(x, metrics.recall, color="#315A7D")
    axes[0].set_yticks(x, metrics.label)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.22)
    axes[0].set_xlabel("Labelled-interval recall")
    for index, row in metrics.iterrows():
        axes[0].text(row.recall + .015, index, f"{row.detected}/{row.events}", va="center", fontsize=9)
    physical = pd.read_csv(tables / "run15_method_physical_false_alarm_comparison.csv")
    methods = ["expanded_no_response", "fan_residual_settled", "mass_balance_co2_residual"]
    rates = [physical.loc[physical.method.eq(m), "episodes"].sum() / physical.loc[physical.method.eq(m), "exposure_h"].sum() for m in methods]
    axes[1].bar(["Direct\ncheck", "Fan\nresidual", "CO₂\nresidual"], rates, color=["#315A7D", "#4F7C61", "#D9782D"])
    axes[1].set_ylabel("False-alert episodes per hour")
    axes[1].set_title("Unchanged command windows")
    fig.tight_layout()
    fig.savefig(figures / "dissertation_anomaly_summary.png", dpi=200)
    plt.close(fig)
    combine([figures / "dissertation_anomaly_summary.png", figures / "figure_s06_decision_trace.png"],
            figures / "figure_summary_and_s06_trace.png")
    combine([figures / "figure_s07_s09_flag_ranges.png", figures / "figure_s07_overlap_recall.png"],
            figures / "figure_ranges_and_overlap_recall.png")
