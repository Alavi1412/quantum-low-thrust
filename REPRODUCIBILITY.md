# Reproducibility and Artifact Manifest

This package captures the current paper artifacts for the normalized Earth-Moon
CR3BP low-thrust initialization benchmark. It is a research benchmark package,
not a flight-ready trajectory design tool.

## Provenance Snapshot

- Verified runs recorded Python `3.11.9 (MSC v.1938 64 bit AMD64)` on Windows.
- The verified package versions are pinned in `requirements-lock.txt`.
- A project-local `.venv` is recommended. During this packaging pass, local venv
  creation on the NAS workspace was reported to stall, while the recorded
  artifact runs used the global Python 3.11.9 interpreter.
- The repository now has commits. Some historical experiment metadata record
  `git_head: null` because those runs predated the first commit. For the current
  snapshot, use git history together with `data/results/artifact_manifest.json`.
  The refreshed manifest records the current scoped file set and working-tree
  hashes at generation time. The manifest intentionally has no self-entry.

## Quick Verification

```powershell
py -3.11 -m pip install -r requirements-lock.txt
py -3.11 -m pytest tests -q
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The pytest suite and LaTeX build are the intended short clean-clone verification
path. Do not rerun the long experiments unless the paper artifacts need to be
regenerated; use the recorded artifacts for normal verification.

## Artifact Map

| Evidence family | Command | Key inputs | Key outputs | Runtime note |
| --- | --- | --- | --- | --- |
| Paper PDF | `latexmk -pdf paper/main.tex` or equivalent local LaTeX build | `paper/main.tex`, `paper/references.bib`, generated `tables/`, `figures/` | `paper/main.pdf` | Build time depends on local TeX install; not an experiment. |
| Smoke tests | `python -m pytest tests` | `tests/test_smoke.py`, `src/qlt/*`, `configs/smoke.yaml` | pytest pass/fail output | Short. |
| Phase-shift benchmark | `python scripts/run_experiment.py --config configs/q1_phase_shift.yaml` | `configs/q1_phase_shift.yaml`, `data/source_states.json` | `data/results/phase_shift/*`, `figures/phase_shift/*`, `tables/phase_shift/*` | Moderate; metadata records package versions but no total runtime field. |
| Phase-shift cardinality benchmark | `python scripts/run_experiment.py --config configs/q1_phase_shift_cardinality.yaml` | `configs/q1_phase_shift_cardinality.yaml`, `data/source_states.json` | `data/results/phase_shift_cardinality/*`, `figures/phase_shift_cardinality/*`, `tables/phase_shift_cardinality/*` | Moderate to expensive; 10 seeds with branch recovery. |
| Bounded phase suite | `python scripts/run_bounded_phase_suite.py --resume` | `configs/bounded_phase_suite.yaml`, `data/source_states.json` | `data/results/bounded_phase_suite/bounded_phase_suite.csv`, `figures/bounded_phase_suite/*`, `tables/bounded_phase_suite/*` | Expensive; configured runtime budget is 600 s, with recorded cases from about 13 s to 367 s. |
| Robust-margin suite | `python scripts/run_robust_margin_suite.py --resume` | `configs/robust_margin_suite.yaml`, `data/source_states.json` | `data/results/robust_margin_suite/*`, `figures/robust_margin_suite/*`, `tables/robust_margin_suite/*` | Expensive if regenerated; recorded rows include selected-branch thrust-margin and all one-segment outage branch diagnostics. |
| Continuation-extension suite | `py -3.11 scripts\run_continuation_margin_suite.py --config configs/continuation_extension_suite.yaml --resume` | `configs/continuation_extension_suite.yaml`, `data/source_states.json`, persisted nominal-control sidecars for warm rows | `data/results/continuation_extension_suite/*`, `data/results/continuation_extension_suite/controls/*`, `figures/continuation_extension_suite/*`, `tables/continuation_extension_suite/*` | Expensive if regenerated; continuous-backend direct multiple-shooting continuation baseline, not a quantum or discrete-sampler run. |
| Direct-collocation baseline | `python scripts/run_direct_collocation_baseline.py --config configs/direct_collocation_baseline.yaml` | `configs/direct_collocation_baseline.yaml`, `src/qlt/direct_collocation.py`, `data/source_states.json` | `data/results/direct_collocation_baseline/*`, `figures/direct_collocation_baseline/*`, `tables/direct_collocation_baseline/*` | Expensive if regenerated; use recorded artifacts for short verification. |
| Constant-control Hermite-Simpson continuation baseline/probe | `py -3.11 scripts\run_hermite_simpson_continuation.py --resume` | `configs/hermite_simpson_continuation_baseline.yaml`, `data/source_states.json`, persisted nominal-control sidecars for warm rows | `data/results/hermite_simpson_continuation_baseline/*`, `data/results/hermite_simpson_continuation_baseline/controls/*`, `figures/hermite_simpson_continuation_baseline/*`, `tables/hermite_simpson_continuation_baseline/*` | Expensive if regenerated; continuous-backend diagnostic with constant segment controls and no independent midpoint controls. Not quantum/discrete evidence. |
| Independent-midpoint-control Hermite-Simpson continuation evidence | `py -3.11 scripts\run_independent_hs_continuation.py --config configs\independent_hs_continuation_baseline.yaml --source-states data\source_states.json --resume` | `configs/independent_hs_continuation_baseline.yaml`, `data/source_states.json`, persisted endpoint-plus-midpoint nominal-control sidecars for warm rows | `data/results/independent_hs_continuation_baseline/*`, `data/results/independent_hs_continuation_baseline/controls/*`, `figures/independent_hs_continuation_baseline/*`, `tables/independent_hs_continuation_baseline/*` | Expensive if regenerated; continuous-backend diagnostic with independent midpoint controls and endpoint/midpoint sidecar hashes. Not quantum/discrete evidence. The phase-time `0.2` and bounded catalog-DRO diagnostics remain negative. |
| Phase-shift cardinality main-method 30-seed package | `py -3.11 scripts\run_experiment.py --config configs\q1_phase_shift_cardinality_30seed.yaml` and `py -3.11 scripts\run_main_method_statistics.py --config configs\q1_phase_shift_cardinality_30seed.yaml` | `configs/q1_phase_shift_cardinality_30seed.yaml`, `data/source_states.json` | `data/results/phase_shift_cardinality_30seed/*`, `figures/phase_shift_cardinality_30seed/*`, `tables/phase_shift_cardinality_30seed/*` | Expensive to regenerate the raw run; statistics generation is short. 210 rows = 30 seeds x 7 methods; no quantum-advantage or QAOA-superiority claim. |
| QAOA depth ablation, 30-seed statistics | `python scripts/run_qaoa_depth_ablation.py --config configs/qaoa_depth_ablation_30seed.yaml --angle-restarts 1 --maxiter 10` | `configs/qaoa_depth_ablation_30seed.yaml`, `configs/q1_phase_shift_cardinality.yaml` | `data/results/qaoa_depth_ablation_30seed/*`, `figures/qaoa_depth_ablation_30seed/*`, `tables/qaoa_depth_ablation_30seed/*` | Expensive; recorded runtime is 3301.5 s. QAOA-depth statistical package; no superiority or quantum-advantage claim. |
| Cardinality ablation | `python scripts/run_cardinality_ablation.py` | `configs/q1_phase_shift_cardinality.yaml` | `data/results/phase_shift_cardinality_ablation/*`, `figures/phase_shift_cardinality_ablation/*`, `tables/phase_shift_cardinality_ablation/*` | Expensive; recorded runtime is 2069.8 s. |
| Teacher feasible benchmark | `python scripts/run_experiment.py --config configs/q1_teacher_feasible.yaml` | `configs/q1_teacher_feasible.yaml`, teacher target metadata in run output | `data/results/teacher_feasible/*`, `figures/teacher_feasible/*`, `tables/teacher_feasible/*` | Moderate; teacher controls are diagnostic and disclosed in metadata. |
| Feasibility sweep | `python scripts/run_feasibility_sweep.py --config configs/q1_candidate.yaml --resume --max-cases 0` | `configs/q1_candidate.yaml` | `data/results/feasibility_sweep.csv`, `data/results/feasibility_metadata.json`, `tables/feasibility_table.tex` | Resume-only command is short; full sweep can be expensive. |
| Catalog targeted feasibility | `python scripts/run_feasibility_sweep.py --config configs/catalog_targeted_feasibility.yaml --transfer-times 4.0 --amax 0.3 --segments 14 --max-nfev 250 --multistart --random-starts 3 --include-bang-bang --min-recovery-segments 4 --state-residual-weight 1.25 --robust-residual-weight 1.15 --fuel-residual-weight 0.01 --smooth-residual-weight 0.006 --control-regularization 0.006 --max-cases 1` | `configs/catalog_targeted_feasibility.yaml`, `data/source_states.json` | `data/results/catalog_targeted_feasibility/*`, `figures/catalog_targeted_feasibility/*`, `tables/catalog_targeted_feasibility/*` | Expensive if expanded beyond the single recorded case. |
| Selected-outage hard-catalog envelope | `python scripts/run_catalog_feasibility_envelope.py --config configs/hard_catalog_selected_outage_envelope.yaml --resume` | `configs/hard_catalog_selected_outage_envelope.yaml`, `data/source_states.json` | `data/results/hard_catalog_selected_outage_envelope/*`, `figures/hard_catalog_selected_outage_envelope/*`, `tables/hard_catalog_selected_outage_envelope/*` | Expensive if regenerated; negative robustness probe. Selected recovery errors are small for chosen masks, but nominal thresholds fail, optimizer/backend success is false, and all-mask diagnostics remain high. |
| Locked-nominal hard-catalog branch recovery | `py -3.11 scripts\run_locked_nominal_recovery.py --config configs\hard_catalog_locked_nominal_recovery.yaml --resume` | `configs/hard_catalog_locked_nominal_recovery.yaml`, `src/qlt/locked_recovery.py`, `data/source_states.json` | `data/results/hard_catalog_locked_nominal_recovery/*`, `figures/hard_catalog_locked_nominal_recovery/*`, `tables/hard_catalog_locked_nominal_recovery/*` | Expensive if regenerated; continuous-backend diagnostic only, not quantum evidence. Freezes a feasible nominal control and optimizes each selected branch independently. The bounded one-segment, six-recovery-segment subset (`locked_hard_single_min6_selected8`) meets thresholds but is not optimizer-converged; selected one/two-segment and all-single-outage scopes fail. The all-mask column is a diagnostic, not a robustness claim. |
| Delayed-arrival locked-nominal hard-catalog portfolio recovery | `py -3.11 scripts\run_delayed_locked_recovery.py --config configs\hard_catalog_delayed_recovery.yaml --resume` | `configs/hard_catalog_delayed_recovery.yaml`, `src/qlt/delayed_recovery.py`, `scripts/run_delayed_locked_recovery.py`, `data/source_states.json` | `data/results/hard_catalog_delayed_recovery/*`, `figures/hard_catalog_delayed_recovery/*`, `tables/hard_catalog_delayed_recovery/*` | Expensive if regenerated; the recorded h6 portfolio row took about 2639.7 s. Continuous-backend delayed-arrival horizon evidence only, not fixed-final-time, fuel-optimal, quantum, QUBO, or QAOA evidence. The h6 portfolio recovers all one-segment masks against the delayed target; the h4 regularized all-single row fails; two-segment outage families remain unresolved. |
| Tail-coast fixed-final-time hard-catalog recovery | `py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --resume` | `configs/hard_catalog_tail_coast_recovery.yaml`, `src/qlt/tail_coast_recovery.py`, `scripts/run_tail_coast_recovery.py`, `data/source_states.json` | `data/results/hard_catalog_tail_coast_recovery/*`, `figures/hard_catalog_tail_coast_recovery/*`, `tables/hard_catalog_tail_coast_recovery/*` | Expensive if regenerated; recorded all-single row took about 1145.7 s. Continuous-backend fixed-final-time evidence only, not fuel-optimal, quantum, QUBO, or QAOA evidence. The all-single portfolio row recovers all 14 one-segment masks at the original target/final time with selected/all fixed-time worst error `0.02299233817855882`; two-segment outage families remain unresolved. |
| Multiple-shooting feasibility | `python scripts/run_multiple_shooting_feasibility.py --config configs/q1_candidate.yaml --resume --max-cases 0` | `configs/q1_candidate.yaml` | `data/results/multiple_shooting_feasibility.csv`, `data/results/multiple_shooting_feasibility_metadata.json`, `figures/multiple_shooting_feasibility.*`, `tables/multiple_shooting_feasibility_table.tex` | Resume-only command is short; full case recorded 294.7 s. |
| Catalog collocation feasibility | `python scripts/run_catalog_collocation_feasibility.py --resume --max-cases 0` | catalog collocation settings encoded by the script and metadata | `data/results/catalog_collocation_feasibility/*`, `figures/catalog_collocation_feasibility/*`, `tables/catalog_collocation_feasibility/*` | Resume-only command is short; full collocation search is expensive and currently has no feasible case. |

## Claim-to-Artifact Trace

- Controlled benchmark framing and limitations: `paper/main.tex`, `README.md`,
  `data/results/*/run_metadata.json`, and this file.
- Non-teacher catalog phase-shift results: `data/results/phase_shift/*` and
  `data/results/phase_shift_cardinality/*`.
- Bounded projected multiple-shooting feasibility claims:
  `data/results/bounded_phase_suite/bounded_phase_suite_metadata.json` and
  `tables/bounded_phase_suite/bounded_phase_suite_table.tex`.
- Robust-margin selected-branch and all one-segment outage claims:
  `data/results/robust_margin_suite/robust_margin_suite_metadata.json`,
  `data/results/robust_margin_suite/robust_margin_suite.csv`, and
  `tables/robust_margin_suite/robust_margin_suite_table.tex`.
- Continuation-extension continuous-backend claims, warm-start provenance, and
  control-sidecar hashes:
  `data/results/continuation_extension_suite/continuation_margin_suite_metadata.json`,
  `data/results/continuation_extension_suite/continuation_margin_suite.csv`,
  `data/results/continuation_extension_suite/controls/*`,
  `tables/continuation_extension_suite/continuation_margin_suite_table.tex`, and
  `figures/continuation_extension_suite/continuation_margin_suite.*`.
- Direct-collocation baseline comparison:
  `data/results/direct_collocation_baseline/*`,
  `tables/direct_collocation_baseline/*`, and
  `figures/direct_collocation_baseline/*`.
- Constant-control Hermite-Simpson continuation baseline/probe:
  `data/results/hermite_simpson_continuation_baseline/hermite_simpson_continuation_baseline_metadata.json`,
  `data/results/hermite_simpson_continuation_baseline/hermite_simpson_continuation_baseline.csv`,
  `data/results/hermite_simpson_continuation_baseline/controls/*`,
  `tables/hermite_simpson_continuation_baseline/hermite_simpson_continuation_baseline_table.tex`,
  and `figures/hermite_simpson_continuation_baseline/hermite_simpson_continuation_baseline.*`.
- Independent-midpoint-control Hermite-Simpson continuation evidence and
  endpoint-plus-midpoint sidecar provenance:
  `data/results/independent_hs_continuation_baseline/independent_hs_continuation_baseline_metadata.json`,
  `data/results/independent_hs_continuation_baseline/independent_hs_continuation_baseline.csv`,
  `data/results/independent_hs_continuation_baseline/controls/*`,
  `tables/independent_hs_continuation_baseline/independent_hs_continuation_baseline_table.tex`,
  and `figures/independent_hs_continuation_baseline/independent_hs_continuation_baseline.*`.
- Duty-cycle-prior 30-seed main-method statistics:
  `data/results/phase_shift_cardinality_30seed/raw_results.csv`,
  `data/results/phase_shift_cardinality_30seed/success_intervals.csv`,
  `data/results/phase_shift_cardinality_30seed/paired_comparisons.csv`,
  `data/results/phase_shift_cardinality_30seed/main_method_statistics_metadata.json`,
  `tables/phase_shift_cardinality_30seed/results_table.tex`,
  `tables/phase_shift_cardinality_30seed/main_method_statistics_table.tex`,
  and `figures/phase_shift_cardinality_30seed/main_method_statistics_summary.*`.
- QAOA depth 30-seed statistical interpretation limits:
  `data/results/qaoa_depth_ablation_30seed/metadata.json`,
  `data/results/qaoa_depth_ablation_30seed/success_intervals.csv`,
  `data/results/qaoa_depth_ablation_30seed/paired_comparisons.csv`,
  `data/results/qaoa_depth_ablation_30seed/raw_results.csv`,
  `tables/qaoa_depth_ablation_30seed/qaoa_depth_ablation_table.tex`,
  `tables/qaoa_depth_ablation_30seed/qaoa_depth_ablation_statistics_table.tex`,
  and `figures/qaoa_depth_ablation_30seed/qaoa_depth_ablation_summary.*`.
- Cardinality and high-duty availability analysis:
  `data/results/phase_shift_cardinality_ablation/metadata.json` and
  `tables/phase_shift_cardinality_ablation/*`.
- Teacher feasibility and oracle diagnostic disclosure:
  `data/results/teacher_feasible/run_metadata.json` and
  `tables/teacher_feasible/*`.
- Feasibility and catalog collocation caveats:
  `data/results/feasibility_metadata.json`,
  `data/results/catalog_targeted_feasibility/feasibility_metadata.json`, and
  `data/results/catalog_collocation_feasibility/multiple_shooting_feasibility_metadata.json`.
- Selected-outage hard-catalog negative robustness probe:
  `data/results/hard_catalog_selected_outage_envelope/catalog_feasibility_envelope_metadata.json`,
  `data/results/hard_catalog_selected_outage_envelope/catalog_feasibility_envelope.csv`,
  `tables/hard_catalog_selected_outage_envelope/catalog_feasibility_envelope_table.tex`,
  and `figures/hard_catalog_selected_outage_envelope/catalog_feasibility_envelope.*`.
- Locked-nominal hard-catalog branch-recovery diagnostic (continuous-backend
  only; bounded one-segment/six-recovery-segment subset passes, broader scopes
  fail, passing row not optimizer-converged):
  `data/results/hard_catalog_locked_nominal_recovery/locked_nominal_recovery_metadata.json`,
  `data/results/hard_catalog_locked_nominal_recovery/locked_nominal_recovery.csv`,
  `tables/hard_catalog_locked_nominal_recovery/locked_nominal_recovery_table.tex`,
  and `figures/hard_catalog_locked_nominal_recovery/locked_nominal_recovery.*`.
- Delayed-arrival hard-catalog h6 portfolio recovery diagnostic
  (continuous-backend delayed-arrival horizon evidence only; all 14
  one-segment masks pass against the delayed target, h4 regularized all-single
  fails, and two-segment recovery remains unresolved):
  `data/results/hard_catalog_delayed_recovery/delayed_locked_recovery_metadata.json`,
  `data/results/hard_catalog_delayed_recovery/delayed_locked_recovery.csv`,
  `tables/hard_catalog_delayed_recovery/delayed_locked_recovery_table.tex`,
  and `figures/hard_catalog_delayed_recovery/delayed_locked_recovery.*`.
- Tail-coast fixed-final-time hard-catalog recovery diagnostic
  (continuous-backend evidence only; final five nominal controls fixed exactly
  zero; all 14 one-segment masks pass at the original target/final time; mask 7
  uses the `constant_y_minus_0p5` fallback; mask 13 is
  `no_recovery_variables`; two-segment recovery remains unresolved):
  `data/results/hard_catalog_tail_coast_recovery/tail_coast_recovery_metadata.json`,
  `data/results/hard_catalog_tail_coast_recovery/tail_coast_recovery.csv`,
  `tables/hard_catalog_tail_coast_recovery/tail_coast_recovery_table.tex`,
  and `figures/hard_catalog_tail_coast_recovery/tail_coast_recovery.*`.

Legacy 10-seed QAOA-depth artifacts remain under `data/results/qaoa_depth_ablation/`,
`tables/qaoa_depth_ablation/`, and `figures/qaoa_depth_ablation/`, but they are
not the current 30-seed QAOA-depth evidence for the manuscript.

## Integrity Manifest

`data/results/artifact_manifest.json` records scoped SHA-256 hashes for key
source, configuration, manuscript, result, table, and figure files. It excludes
`.venv`, caches, logs, exhaustive generated intermediates, and the manifest
file itself.
