# Quantum Low-Thrust Robust Initialization Benchmark

Research-code scaffold for the controlled CR3BP initialization benchmark:
"Quantum-Ready Robust Maneuver Initialization for Low-Thrust Cislunar
Transfers Under Missed-Thrust Events".

This is not a flight-ready trajectory design tool. It is a reproducible
normalized Earth-Moon CR3BP benchmark for comparing binary thrust-window
initialization methods under missed-thrust outages.

## Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If `py -3.11` is unavailable, use any Python 3.11+ interpreter and keep the
virtual environment project-local. The verified artifact runs recorded in
`data/results/**/metadata.json` used the global Python 3.11.9 interpreter on
Windows. A project-local `.venv` is recommended for reproduction, but local venv
creation on the NAS workspace was reported to stall during packaging; use a
local disk clone or an already working Python 3.11 environment if that happens.

For the exact direct dependencies used by the verified runs:

```powershell
python -m pip install -r requirements-lock.txt
```

## Short Verification

```powershell
py -3.11 -m pytest tests -q
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This is the intended clean-clone verification path after installing the pinned
dependencies. The long experiment commands below are expensive; use the recorded
artifacts unless intentionally regenerating evidence.

## Reproducibility Manifest

See `REPRODUCIBILITY.md` for the artifact map, expected outputs, known expensive
runs, and commands used for the paper evidence. A machine-readable SHA-256
manifest is written to `data/results/artifact_manifest.json`.

Historical experiment metadata may contain `git_head: null` because those runs
predated the first commit. The repository now has commits; the current manifest
records the HEAD available before the integration commit and file-level hashes
for the working tree at manifest generation time. The manifest intentionally
does not include an entry for itself.

## Main Commands

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --config configs\default.yaml
```

For the calibrated branch-recovery replicated run used to address the robust
refinement blocker:

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --config configs\q1_candidate.yaml
```

The `q1_candidate` configuration uses `refinement.mode: branch_recovery`, adds
an `all_windows_continuous` direct continuous-control baseline, and enables
equal-total-budget accounting so classical schedule search receives a true
evaluation budget comparable to QUBO/QAOA after shared QUBO training is charged.

For the teacher-generated feasible benchmark and the oracle-only continuous
initialization diagnostic:

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --config configs\q1_teacher_feasible.yaml
```

The `teacher_controls_oracle_diagnostic` row uses the hidden teacher controls as
a continuous initial guess for the teacher binary schedule. It is a diagnostic
baseline, not a normal competing schedule initializer.

For the catalog-derived halo phase-shift benchmark:

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --config configs\q1_phase_shift.yaml
```

For the cardinality-prior phase-shift benchmark:

```powershell
.\.venv\Scripts\python scripts\run_experiment.py --config configs\q1_phase_shift_cardinality.yaml
```

For the QAOA depth ablation:

```powershell
.\.venv\Scripts\python scripts\run_qaoa_depth_ablation.py --config configs\qaoa_depth_ablation_30seed.yaml --angle-restarts 1 --maxiter 10
```

This 30-seed package writes `data/results/qaoa_depth_ablation_30seed/*`,
`tables/qaoa_depth_ablation_30seed/*`, and
`figures/qaoa_depth_ablation_30seed/*`. It is the main QAOA statistical result:
optimized `p=2` QAOA improves over random-angle QAOA and is competitive with
surrogate-QUBO simulated annealing, but paired tests do not support a superiority
or quantum-advantage claim. The older `qaoa_depth_ablation` artifacts are kept
as legacy 10-seed outputs.

For the bounded non-teacher phase suite:

```powershell
.\.venv\Scripts\python scripts\run_bounded_phase_suite.py --resume
```

For the robust-margin suite:

```powershell
.\.venv\Scripts\python scripts\run_robust_margin_suite.py --resume
```

The robust-margin suite writes `data/results/robust_margin_suite/*`,
`tables/robust_margin_suite/*`, and `figures/robust_margin_suite/*`.

For the selected-outage hard-catalog feasibility-envelope diagnostics:

```powershell
py -3.11 scripts\run_catalog_feasibility_envelope.py --config configs\hard_catalog_selected_outage_envelope.yaml --resume
```

The selected-outage hard-catalog package writes
`data/results/hard_catalog_selected_outage_envelope/*`,
`tables/hard_catalog_selected_outage_envelope/*`, and
`figures/hard_catalog_selected_outage_envelope/*`. It is negative robustness
evidence: selected recovery branch errors are small for the chosen masks, but
the nominal trajectory fails, optimizer/backend success is false at the
evaluation cap, and all-mask diagnostics remain high.

For the continuation-extension continuous-backend baseline:

```powershell
py -3.11 scripts\run_continuation_margin_suite.py --config configs\continuation_extension_suite.yaml --resume
```

The continuation-extension suite writes
`data/results/continuation_extension_suite/*`, nominal-control sidecars under
`data/results/continuation_extension_suite/controls/`,
`tables/continuation_extension_suite/*`, and
`figures/continuation_extension_suite/*`. This is a direct multiple-shooting
continuation baseline, not a QUBO, QAOA, quantum, or discrete-sampler result.

For the cardinality ablation:

```powershell
.\.venv\Scripts\python scripts\run_cardinality_ablation.py
```

For the direct-collocation baseline:

```powershell
.\.venv\Scripts\python scripts\run_direct_collocation_baseline.py --config configs\direct_collocation_baseline.yaml
```

For feasibility/collocation diagnostics, see `REPRODUCIBILITY.md`; some are
expensive and should not be rerun casually.

The `q1_phase_shift` configuration writes to `data/results/phase_shift`,
`figures/phase_shift`, and `tables/phase_shift`. Its target is generated by
ballistic CR3BP phase propagation of the JPL `initial_nrho_like_l2_southern_halo`
state for `benchmark.phase_time`; it is not a teacher-controlled low-thrust
target and does not enable teacher-seeded or teacher-oracle baselines.

The command writes:

- `data/results/raw_results.csv`
- `data/results/summary.csv`
- `data/results/summary.json`
- `data/results/qubo_diagnostics.csv`
- `data/results/run_metadata.json`
- `data/results/qubo_coefficients_seed*.json`
- `data/results/qubo_fit_seed*.csv`
- `figures/trajectory_example.{png,pdf}`
- `figures/method_comparison.{png,pdf}`
- `figures/qubo_fit.{png,pdf}`
- `figures/refinement_success.{png,pdf}`
- `figures/recovery_fuel.{png,pdf}`
- `tables/results_table.tex`
- `tables/ablation_table.tex`
- `tables/recovery_table.tex`

The benchmark source states and NASA/JPL API query metadata are stored in
`data/source_states.json`.
