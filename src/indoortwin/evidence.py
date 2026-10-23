"""Evaluated evidence plots, with descriptive names and notebook display helpers."""
from pathlib import Path
from indoortwin.paths import ROOT

PLOTS = [('fan_curve_airflow', '05', 'rq3_simulation/figures/figure_fan_curve_measured_input.png', 'Fan-curve airflow calculation'), ('power_interruption_freshness', '01', 'rq1_power/figures/rq1_191233_fresh_property_count_timeline.png', 'Fresh properties during six node interruptions'), ('power_interruption_measurements', '01', 'rq1_figures/figure_rq1_measurement_continuity_physical.png', 'Measurements during six power interruptions'), ('telemetry_interruption_conformance', '02', 'rq1_repeated/figures/figure_rq1_threshold_and_recovery.png', 'Freshness conformance and restoration'), ('telemetry_interruption_restoration', '02', 'rq1_repeated/figures/figure_rq1_representative_phase_response.png', 'Freshness loss and restoration'), ('telemetry_interruption_measurements', '02', 'rq1_figures/figure_rq1_measurement_continuity_repeated.png', 'Measurements during repeated telemetry interruptions'), ('controlled_action_states', '04', 'rq2_timeline/figures/figure_rq2_calculated_states.png', 'Calculated states across controlled actions'), ('integrated_room_timeline', '04', 'rq2_timeline/figures/figure_rq2_integrated_timeline.png', '4 October measurements and calculated states'), ('ventilation_factor_selection', '05', 'rq3_simulation/figures/figure_eta_search_215609.png', 'Ventilation-effectiveness factor selection'), ('measured_and_predicted_co2', '05', 'rq3_simulation/figures/figure_measured_pressure_three_records.png', 'Measured and predicted CO2'), ('simulation_error_comparison', '05', 'rq3_simulation/figures/figure_three_record_mae_comparison.png', 'Simulation error by record and ventilation state'), ('anomaly_summary_and_degraded_response', '06', 'rq3_anomaly/figures/figure_summary_and_s06_trace.png', 'Anomaly results and S06 trace'), ('anomaly_detection_scores', '06', 'rq3_anomaly/figures/figure_anomaly_scores.png', 'Anomaly precision recall and F1'), ('co2_residual_trace', '06', 'rq3_anomaly/figures/figure_s07_residual_trace.png', 'S07 residual trace'), ('anomaly_ranges_and_overlap_recall', '06', 'rq3_anomaly/figures/figure_ranges_and_overlap_recall.png', 'CO2 coverage and overlap recall'), ('fan_command_repeatability', '07', 'rq4/figures/figure_rq4_pwm_rpm_repeatability.png', 'Fan speed across command orders'), ('command_response_timing', '07', 'rq4/figures/figure_rq4_latency_distributions.png', 'Acknowledgement and fan-response timing'), ('acknowledgement_without_rotation', '07', 'rq4/figures/figure_rq4_ack_without_rotation.png', 'Acknowledgement without exhaust-fan rotation')]

def display_plots(notebook_prefix):
    """Embed the evaluated PNGs in notebook outputs so review needs no rerun."""
    from IPython.display import Image, Markdown, display
    for slug, prefix, relative, title in PLOTS:
        if prefix != notebook_prefix:
            continue
        path = ROOT / "results" / relative
        if not path.is_file():
            raise FileNotFoundError(f"Generate the evaluated plot before displaying it: {relative}")
        display(Markdown("### " + title))
        display(Image(filename=str(path), width=1100))
