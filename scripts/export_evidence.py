"""Export only the figures and result tables directly reported in the dissertation."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
from indoortwin.evidence import PLOTS

def main():
    destination = ROOT / "evidence"
    for folder in (destination / "plots", destination / "tables"):
        folder.mkdir(parents=True, exist_ok=True)
    for slug, notebook, relative, caption in PLOTS:
        src = RESULTS / relative
        if not src.is_file():
            raise FileNotFoundError(f"Missing {caption}: {src}. Run notebook {notebook} first.")
        shutil.copyfile(src, destination / "plots" / f"{slug}.png")
    direct_tables = {
        "physical_power_timing.csv": "rq1_power/tables/node_cycle_metrics.csv",
        "state_transition_timing.csv": "rq2_states/tables/table_5_2_state_transition_timing.csv",
        "room_phase_summary.csv": "rq2_timeline/tables/table_5_3_phase_summary.csv",
        "command_response_timing.csv": "rq4/tables/table_5_7_command_response.csv",
    }
    direct_tables.update({
        'power_property_transition_metrics.csv': 'rq1_power/tables/property_transition_metrics.csv',
        'power_threshold_class_summary.csv': 'rq1_power/tables/threshold_class_summary.csv',
        'telemetry_cycle_metrics.csv': 'rq1_repeated/tables/rq1_cycle_metrics.csv',
        'telemetry_property_transition_metrics.csv': 'rq1_repeated/tables/rq1_property_transition_metrics.csv',
        'telemetry_representative_phase_response.csv': 'rq1_repeated/tables/rq1_representative_phase_response.csv',
        'controlled_action_states.csv': 'rq2_timeline/tables/rq2_step_states.csv',
        'anomaly_physical_false_alarm_comparison.csv': 'rq3_anomaly/tables/run15_method_physical_false_alarm_comparison.csv',
        'anomaly_overlap_summary.csv': 'rq3_anomaly/tables/run15_overlap_summary.csv',
        'anomaly_rule_detector_comparison.csv': 'rq3_anomaly/tables/run15_rule_detector_comparison.csv',
        'anomaly_co2_reporting_checks.csv': 'rq3_anomaly/tables/run15_s07_reporting_checks.csv',
        'simulation_duty_cycle_metrics.csv': 'rq3_simulation/tables/duty_cycle_metrics.csv',
        'simulation_duty_pressure_and_airflow.csv': 'rq3_simulation/tables/duty_pressure_and_airflow.csv',
        'simulation_ventilation_factor_search.csv': 'rq3_simulation/tables/eta_search_measured_duty_8oct.csv',
        'simulation_record_metrics.csv': 'rq3_simulation/tables/record_scope_metrics.csv',
        'simulation_three_record_comparison.csv': 'rq3_simulation/tables/three_record_scope_comparison.csv',
        'command_response_latency_summary.csv': 'rq4/tables/rq4_latency_summary.csv',
        'command_response_repeatability.csv': 'rq4/tables/rq4_repeatability_summary.csv',
        'command_response_trials.csv': 'rq4/tables/rq4_trial_results.csv',
    })
    for filename, relative in direct_tables.items():
        if filename == "simulation_duty_pressure_and_airflow.csv":
            # Export only the measured-pressure fields evaluated in the evidence table.
            pd.read_csv(RESULTS / relative).drop(
                columns=["solver_pressure_4pa", "intake_flow_4pa_m3h", "intake_flow_change_m3h"],
                errors="ignore",
            ).to_csv(destination / "tables" / filename, index=False)
        else:
            shutil.copyfile(RESULTS / relative, destination / "tables" / filename)
    sim = RESULTS / "rq3_simulation"
    factors = json.loads((sim / "frozen_ventilation_effectiveness_measured_duty.json").read_text(encoding="utf-8"))
    pd.DataFrame([{"engine": engine, **values} for engine, values in factors["engines"].items()]).to_csv(
        destination / "tables/ventilation_factor_selection.csv", index=False)
    record = pd.read_csv(sim / "tables/record_scope_metrics.csv")
    duty = pd.read_csv(sim / "tables/duty_cycle_metrics.csv")
    fit_record = factors["fit_record"]
    rows = []
    for scope, frame in [("Continuous time series", record.loc[record.scope.eq("full record")]),
                         ("Fan-on intervals reinitialised from measured CO2", duty)]:
        held_out = frame.loc[frame.record.ne(fit_record) & frame.pressure_mode.eq("measured_duty")]
        for engine, group in held_out.groupby("engine", sort=False):
            n = group.n.to_numpy(dtype=float)
            rows.append({"scope": scope, "engine": engine, "n": int(n.sum()),
                         "mae_ppm": float(np.average(group.mae_ppm, weights=n)),
                         "rmse_ppm": float(np.sqrt(np.average(group.rmse_ppm**2, weights=n))),
                         "bias_ppm": float(np.average(group.bias_ppm, weights=n)),
                         "largest_record_or_interval_mae_ppm": float(group.mae_ppm.max())})
    pd.DataFrame(rows).to_csv(destination / "tables/simulation_evaluation.csv", index=False)
    print(f"Exported {len(PLOTS)} evaluated plots and {len(direct_tables) + 2} tables to evidence/.")


if __name__ == "__main__":
    main()
