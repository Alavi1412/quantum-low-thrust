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
predated the first commit. The refreshed manifest records the current scoped file
set and working-tree hashes at generation time. The manifest intentionally does
not include an entry for itself.

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

For the 30-seed main-method cardinality-prior package:

```powershell
py -3.11 scripts\run_experiment.py --config configs\q1_phase_shift_cardinality_30seed.yaml
py -3.11 scripts\run_main_method_statistics.py --config configs\q1_phase_shift_cardinality_30seed.yaml
```

This package writes `data/results/phase_shift_cardinality_30seed/*`,
`tables/phase_shift_cardinality_30seed/*`, and
`figures/phase_shift_cardinality_30seed/*`. It contains 210 rows
(30 seeds x 7 methods). The high-duty classical methods and surrogate-QUBO
simulated annealing succeed in all 30 seeds, but paired selected-worst-error
comparisons favor the all-windows continuous baseline. It does not support a
quantum-advantage or QAOA-superiority claim.

For the QAOA depth ablation:

```powershell
.\.venv\Scripts\python scripts\run_qaoa_depth_ablation.py --config configs\qaoa_depth_ablation_30seed.yaml --angle-restarts 1 --maxiter 10
```

This 30-seed package writes `data/results/qaoa_depth_ablation_30seed/*`,
`tables/qaoa_depth_ablation_30seed/*`, and
`figures/qaoa_depth_ablation_30seed/*`. It is the QAOA-depth statistical result:
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

For the locked-nominal hard-catalog branch-recovery diagnostic:

```powershell
py -3.11 scripts\run_locked_nominal_recovery.py --config configs\hard_catalog_locked_nominal_recovery.yaml --resume
```

The locked-nominal package writes
`data/results/hard_catalog_locked_nominal_recovery/*`,
`tables/hard_catalog_locked_nominal_recovery/*`, and
`figures/hard_catalog_locked_nominal_recovery/*`. It is a continuous-backend
diagnostic only, not QUBO, QAOA, or quantum evidence. It freezes a feasible
nominal control (nominal error `0.0131`) and optimizes each selected
missed-thrust branch independently after the outage. The bounded one-segment
selected subset with at least six recovery segments
(`locked_hard_single_min6_selected8`) meets the thresholds, while the selected
one/two-segment (`locked_hard_selected1`, `locked_hard_selected3`) and
all-single-outage (`locked_hard_all_single`) scopes fail. The passing
`locked_hard_single_min6_selected8` row carries an `optimizer_success` caveat:
it is threshold-feasible but not optimizer-converged because some selected
branches hit the evaluation cap. The all-mask column is a diagnostic over every
configured mask, not a robustness claim.

For the delayed-arrival locked-nominal hard-catalog recovery diagnostic:

```powershell
py -3.11 scripts\run_delayed_locked_recovery.py --config configs\hard_catalog_delayed_recovery.yaml --resume
```

The delayed-arrival package writes
`data/results/hard_catalog_delayed_recovery/*`,
`tables/hard_catalog_delayed_recovery/*`, and
`figures/hard_catalog_delayed_recovery/*`. It is continuous-backend
delayed-arrival horizon evidence only, not fixed-final-time robustness, not
QUBO, QAOA, or quantum evidence, and not a fuel-optimality result. The
nominal-only row records provenance with nominal error `0.013133`, a delayed
coast selected metric of `0.025525`, and no branch optimizer. The h4
regularized all-single row is a negative case with selected/all delayed worst
error `1.398973`. The h6 portfolio row optimizes all 14 one-segment outage
masks against the delayed target with variants `regularized_001` and
`terminal_only`; it charges all variant evaluations and reaches selected/all
delayed worst error `0.004040`, max control norm `1.0`, zero reported bound
violation, and `branch_optimizer_all_success=true`. This delayed-arrival
package does not by itself establish fixed-final-time recovery and does not
cover two-segment outage families.

For the fixed-final-time tail-coast hard-catalog recovery diagnostic:

```powershell
py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --resume
```

The tail-coast package writes
`data/results/hard_catalog_tail_coast_recovery/*`,
`tables/hard_catalog_tail_coast_recovery/*`, and
`figures/hard_catalog_tail_coast_recovery/*`. It is continuous-backend
fixed-final-time evidence only, not QUBO, QAOA, quantum evidence, or a
fuel-optimality result. The nominal is solved and then re-refined with the final
five nominal controls fixed exactly to zero; the tail-coast nominal error is
`0.02299233817855882`. The nominal-only row runs no branch optimizer, so its
all-mask masked-nominal diagnostic remains `27.835075017369785`. The
`tail_coast_all_single_t5_portfolio` row selects all 14 one-segment masks,
keeps the original target and original final time, and reaches selected/all
fixed-final-time worst error `0.02299233817855882`, so the all-single
one-segment scope meets the configured thresholds in this diagnostic. Fallback
starts are declared and charged; only mask 7 evaluates fallbacks
(`fallback_evals=[0,0,0,0,0,0,0,2,0,0,0,0,0,0]`) and its accepted fallback is
`constant_y_minus_0p5`. The late-tail mask 13 has no recovery variables and is
labeled `no_recovery_variables`, so `branch_optimizer_all_success=false` even
though the row is threshold-feasible. Two-segment outage families remain
unresolved.

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

For the constant-control Hermite-Simpson continuation baseline/probe:

```powershell
py -3.11 scripts\run_hermite_simpson_continuation.py --resume
```

The Hermite-Simpson package writes
`data/results/hermite_simpson_continuation_baseline/*`, nominal-control
sidecars under `data/results/hermite_simpson_continuation_baseline/controls/`,
`tables/hermite_simpson_continuation_baseline/*`, and
`figures/hermite_simpson_continuation_baseline/*`. It is a continuous-backend
diagnostic with persisted nominal-control warm starts and trajectory-stacking
semantics, not a QUBO, QAOA, quantum, or discrete-sampler result. The
`hs_hard_p04_amax02_warm_from_p03` row is a lower-thrust catalog halo phase-shift
stress probe, not the unresolved hard NRHO-like catalog benchmark.

For the independent-midpoint-control Hermite-Simpson continuation evidence:

```powershell
py -3.11 scripts\run_independent_hs_continuation.py --config configs\independent_hs_continuation_baseline.yaml --source-states data\source_states.json --resume
```

The independent-HS package writes
`data/results/independent_hs_continuation_baseline/*`, endpoint-plus-midpoint
nominal-control sidecars under
`data/results/independent_hs_continuation_baseline/controls/`,
`tables/independent_hs_continuation_baseline/*`, and
`figures/independent_hs_continuation_baseline/*`. Its sidecar schema persists
endpoint and midpoint nominal controls and records endpoint, midpoint, and
combined sidecar hashes so midpoint-control trajectory, fuel, and terminal-error
diagnostics are reproducible. This is continuous-backend evidence only; the
phase-time `0.2` diagnostic and bounded catalog-DRO selected-outage row remain
unresolved.

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
