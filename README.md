# Robust Low-Thrust Cislunar Initialization Benchmark

Research-code scaffold for the controlled CR3BP initialization benchmark:
"A Reproducible Binary-Schedule Benchmark for Robust Low-Thrust Cislunar
Initialization Under Missed-Thrust Events".

This is not a flight-ready trajectory design tool. It is a reproducible
normalized Earth-Moon CR3BP benchmark for comparing binary thrust-window
initialization methods under missed-thrust outages. The manuscript is framed
for `Astrodynamics` as a benchmark, negative-evidence, and reproducibility
paper rather than as a quantum-advantage claim.

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

Reviewer-facing checklist for the current artifact snapshot:

```powershell
git status --short
py -3.11 -m pytest tests/test_smoke.py -q -p no:cacheprovider
py -3.11 -m pytest tests/test_claim_evidence_ledger.py -q -p no:cacheprovider
py -3.11 -m pytest tests/test_evidence_synthesis.py -q -p no:cacheprovider
py -3.11 -m pytest tests/test_replay_stress_validation.py -q -p no:cacheprovider
py -3.11 scripts\run_threshold_sensitivity.py
py -3.11 scripts\run_claim_evidence_ledger.py
py -3.11 scripts\run_evidence_synthesis.py
py -3.11 scripts\run_replay_stress_validation.py
py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --regenerate-artifacts-only --allow-artifact-refresh-fingerprint-mismatch
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
cd ..
py -3.11 scripts\write_artifact_manifest.py --check
git diff --check
```

For a broader local check, run `py -3.11 -m pytest tests -q` after installing
the pinned dependencies. The long experiment commands below are expensive; use
the recorded artifacts unless intentionally regenerating evidence. The primary
review artifacts are `paper/main.pdf`, `paper/supplement.pdf`,
`data/results/claim_evidence_ledger/*`,
`data/results/evidence_synthesis/*`,
`data/results/replay_stress_validation/*`,
`data/results/phase_shift_cardinality_30seed/*`,
`data/results/qaoa_depth_ablation_30seed/*`,
`data/results/hard_catalog_tail_coast_recovery/*`, and
`data/results/artifact_manifest.json`.

## Reproducibility Manifest

See `REPRODUCIBILITY.md` for the artifact map, expected outputs, known expensive
runs, and commands used for the paper evidence. A machine-readable SHA-256
manifest is written to `data/results/artifact_manifest.json` with:

```powershell
py -3.11 scripts\write_artifact_manifest.py
py -3.11 scripts\write_artifact_manifest.py --check
```

Historical experiment metadata may contain `git_head: null` because those runs
predated the first commit. The refreshed manifest uses
`git_head_at_generation`, `git_head_semantics`, and
`working_tree_status_at_generation` to make commit provenance explicit. A
committed manifest necessarily records the HEAD before the final manifest commit;
the file hashes and byte counts are the authoritative artifact identities. The
manifest intentionally does not include an entry for itself. Historical long-run
metadata may also contain dirty-state records; the current submission snapshot
should be checked by clean git status after final commit, manifest `--check`,
pytest, and local LaTeX builds rather than by treating old run metadata as final
repository provenance.

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
py -3.11 scripts\run_threshold_sensitivity.py
py -3.11 scripts\run_claim_evidence_ledger.py
py -3.11 scripts\run_evidence_synthesis.py
```

This package writes `data/results/phase_shift_cardinality_30seed/*`,
`tables/phase_shift_cardinality_30seed/*`, and
`figures/phase_shift_cardinality_30seed/*`. It contains 210 rows
(30 seeds x 7 methods). The high-duty classical methods and surrogate-QUBO
simulated annealing succeed in all 30 seeds, but paired selected-worst-error
comparisons favor the all-windows continuous baseline. The threshold-sensitivity
postprocessor derives `threshold_sensitivity.csv`,
`threshold_sensitivity_metadata.json`, and
`tables/phase_shift_cardinality_30seed/threshold_sensitivity_table.tex` from the
recorded raw CSV only; it does not rerun trajectory optimization. At the tight
`(0.05, 0.09)` threshold check, all sampled methods are `0/30` while
all-windows continuous remains `30/30`. It does not support a quantum-advantage
or QAOA-superiority claim.

The claim evidence ledger postprocessor writes
`data/results/claim_evidence_ledger/claim_evidence_ledger.csv`,
`data/results/claim_evidence_ledger/claim_evidence_ledger_metadata.json`,
`data/results/claim_evidence_ledger/tail_coast_threshold_audit.csv`,
`data/results/claim_evidence_ledger/tail_coast_branch_audit.csv`, and matching
LaTeX tables under `tables/claim_evidence_ledger/`:

```powershell
py -3.11 scripts\run_claim_evidence_ledger.py
```

It reads recorded artifacts only and does not rerun trajectory optimization.
The ledger separates selected-branch evidence, all-mask diagnostics, and
all-configured-mask evidence. The tail-coast audit confirms the combined row
passes recorded-error thresholds through `(0.025, 0.095)` and fails the tighter
`0.09` robust threshold and the `0.02` nominal threshold. The branch audit is a
JSON summary only; accepted branch controls are not persisted or replayed.

The evidence synthesis postprocessor writes
`data/results/evidence_synthesis/evidence_synthesis.csv`,
`data/results/evidence_synthesis/evidence_synthesis_metadata.json`,
`tables/evidence_synthesis/evidence_synthesis_table.tex`, and
`tables/evidence_synthesis/practitioner_lessons_table.tex`. It reads recorded
CSV/JSON artifacts only and does not rerun trajectory optimization. The table
cross-indexes tight 30-seed threshold sensitivity, continuation-extension
multiple-shooting rows, compact direct-collocation and independent-midpoint
Hermite-Simpson diagnostics, and the scoped hard-catalog tail-coast row used in
the main manuscript claim path.

The replay/stress validation postprocessor writes
`data/results/replay_stress_validation/replay_stress_validation.csv`,
`data/results/replay_stress_validation/replay_stress_validation_metadata.json`,
and `tables/replay_stress_validation/replay_stress_validation_table.tex`:

```powershell
py -3.11 scripts\run_replay_stress_validation.py
```

It repropagates persisted nominal-control sidecars for representative
continuation-extension and independent-midpoint Hermite-Simpson phase-shift
rows. The source-substep baselines reproduce recorded nominal errors to within
`1e-12`; refined substeps and direct +/-1% acceleration scaling are stress
diagnostics only. It does not run least-squares optimization, replay branch
recovery controls, or claim high-fidelity validation.

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

To refresh only the table, figure, and metadata from the recorded CSV after
reporting-code changes, without launching optimization or rewriting the raw CSV:

```powershell
py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --regenerate-artifacts-only --allow-artifact-refresh-fingerprint-mismatch
```

The tail-coast package writes
`data/results/hard_catalog_tail_coast_recovery/*`,
`tables/hard_catalog_tail_coast_recovery/*`, and
`figures/hard_catalog_tail_coast_recovery/*`. It is continuous-backend
fixed-final-time evidence only, not QUBO, QAOA, quantum evidence, or a
fuel-optimality result. The nominal is solved and then re-refined with the final
five nominal controls fixed exactly to zero; the tail-coast nominal error is
`0.02299233817855882`. The nominal-only row runs no branch optimizer, so its
all-mask masked-nominal diagnostic remains `27.835075017369785`.

The main row is now `tail_coast_all_one_two_segment_t5_portfolio`. It uses
`outage_lengths=[1,2]`, `selected_outage_policy=all_configured`,
`outage_count=27`, and `selected_outage_count=27`, so the configured one- and
two-segment masks are selected/evaluated together in one fixed-final-time case
row. It reaches selected/all fixed-final-time worst error
`0.0936063931709301` with all five final nominal controls fixed to zero, so it
meets the configured thresholds with nominal tail-coast error
`0.02299233817855882`. Fallback starts are declared and charged;
`branch_fallback_initialization_evaluated_branch_count=4` and accepted count is
also `4`. The branch recovery segments include the one-segment `[13..0]` and
two-segment `[12..0]` sequences. Branches with recovery variables are
optimizer-converged in `25/25` cases; the two `no_recovery_variables` late-tail
branches are threshold-feasible direct evaluations rather than
optimizer-converged branch solves, so `branch_optimizer_all_success=false`.
The current CSV records `nfev=5929` and runtime about 1833.6 s.

The `tail_coast_all_single_t5_portfolio` and
`tail_coast_all_two_segment_t5_portfolio` rows remain provenance/scope rows for
the separate one- and two-segment cases. This package does not establish fuel
optimality, certified flight-design recovery, high-fidelity validation, broader
outage-family robustness beyond the configured one/two masks, or quantum
advantage.

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
