# Data and evaluation notes

The seven executed notebooks, `data/` and `evidence/` contain the selected evidence used in the dissertation. Saved tables and plots can be inspected without running Python. Original collected-record filenames and experiment IDs are retained; exported result tables have descriptive names.

## Included records

| Evaluation | Records and scope |
| --- | --- |
| Physical power interruptions | `data/rq1_power/`: six node interventions, accepted measurements, property transitions and the evaluated registry/freshness configuration. |
| Telemetry interruptions | `data/rq1_repeated/`: eighteen bootloader-hold cycles, measurements, integrity records and restoration transitions. |
| State interpretation | `data/rq2/`: synthetic checks, the original physical pilot, P03 retest, consolidated checks and the seven-phase room timeline. |
| CO₂ simulation and anomaly detection | `data/rq3/`: the three 8–10 October campaigns, physical traces, command windows, recorded labels and required weather input. The two analyses share these records. |
| Command and response | `data/rq4/`: ascending, descending and permuted command sequences, qualification records and exception checks. |

Source manifests and configurations inside the experiment folders identify the recorded protocols. Earlier campaigns, live-system code and unrelated datasets are excluded.

## Reading the data

CSV headers retain the original field names. `entity_id` and `property` identify a measured quantity. CO₂ is in ppm, temperature in °C, relative humidity and PWM in %, speed in RPM, current in A, differential pressure in Pa and vibration RMS in g. Occupancy, opening and equipment states are categorical. `_s` and `_ms` suffixes denote seconds and milliseconds.

Timestamps retain their recorded offsets; local-time plots use Pacific/Auckland. Missing numeric values remain missing. Commands and acknowledgements are distinct from measured physical responses. Experiment, session, sequence and message identifiers describe provenance rather than measured quantities. Exact filtering, thresholds and pooling are shown in each notebook.

## Interpretation

- RQ1 reports host-observed freshness and restoration intervals. The bootloader-hold intervention suspends telemetry and is separate from the physical power interruptions; neither measures isolated network or dashboard-render latency.
- RQ2 evaluates recorded synthetic/physical outcomes and state snapshots. It does not rerun the hardware or original live FusionEngine. P03 combines successful retest fan/ventilation results and the 2,018 ms interval with other contextual states from the earlier pilot. Those contextual cells are not simultaneous retest observations. The original unsuccessful attempt remains in the pilot data. Contextual ventilation/health states do not determine the outcome of steps assessing other properties.
- The timeline's 800 ppm line is a protocol reference; accumulation ended by duration. Sequential fan-duty phases start at different CO₂ concentrations and do not establish a causal ranking. Display shading bridges setup gaps; recorded phase times are retained.
- The 8–9 October record selects simulation factors; 9 and 10 October provide evaluation. Anomaly scenarios alter copies of measured records and are not additional physically observed faults. Detector parameters remain inherited and frozen; physical false alarms and altered-replay detection results are separate evaluations.
- RQ4 uses one apparatus across three command orders. Controller acknowledgement is assessed separately from tachometer-confirmed rotation.

## Optional regeneration

Use Python 3.12 and install `requirements.txt`. Run `python scripts/run_notebooks.py`, then `python scripts/export_evidence.py`, from the repository root. Successful runs retain outputs in `notebooks/`; the export helper refreshes `evidence/plots/` and `evidence/tables/`. Runtime work is written to ignored `results/` and `.validation/` folders. No live hardware connection is required.

Notebook 05 additionally requires NIST ContamX and EnergyPlus 26.1.0. Set `CONTAMX_EXE` and `ENERGYPLUS_EXE` to the installed executables. The exact evaluated weather input is `data/rq3/weather/TMY3_NZ_AK.epw`: NIWA TMY3 New Zealand Auckland, source identifier C74082, NIWA Client report 2023341WN (July 2024). Its original header attribution is retained. Executables are not bundled.
