# Reproducibility and Artifact Manifest

This package captures the current paper artifacts for the normalized Earth-Moon
CR3BP low-thrust missed-thrust initialization benchmark resource. It is a
research benchmark package, not a flight-ready trajectory design tool. The
manuscript is currently framed for `Astrodynamics` as a focused astrodynamics
benchmark-resource paper with explicit negative evidence and provenance.

## Provenance Snapshot

- Verified runs recorded Python `3.11.9 (MSC v.1938 64 bit AMD64)` on Windows.
- The verified package versions are pinned in `requirements-lock.txt`.
- A project-local `.venv` is recommended. During this packaging pass, local venv
  creation on the NAS workspace was reported to stall, while the recorded
  artifact runs used the global Python 3.11.9 interpreter.
- The repository now has commits. Some historical experiment metadata record
  `git_head: null` because those runs predated the first commit. For the current
  snapshot, use git history together with `data/results/artifact_manifest.json`.
  The refreshed manifest records `git_head_at_generation`,
  `git_head_semantics`, `working_tree_status_at_generation`, scoped file hashes,
  and byte counts. A committed manifest necessarily records the HEAD before the
  final manifest commit; the file hashes are authoritative for artifact identity.
  The manifest intentionally has no self-entry. Historical long-run metadata may
  also contain dirty-state records; final submission provenance should be
  checked from clean git status after final commit, manifest `--check`, tests,
  and local LaTeX builds.

## Quick Verification

```powershell
py -3.11 -m pip install -r requirements-lock.txt
py -3.11 scripts\verify_submission_snapshot.py
```

The verifier is read-only by default. It checks key primary artifact paths, runs
`scripts\write_artifact_manifest.py --check`, runs a focused pytest subset for
paper/reproducibility artifacts, and runs `git diff --check`; it does not
regenerate artifacts, rerun trajectory optimization, rebuild PDFs, clean LaTeX
auxiliary files, or create an archive DOI. A broader local check can run
`py -3.11 scripts\verify_submission_snapshot.py --full-tests`. Do not rerun the
long experiments unless the paper artifacts need to be regenerated; use the
recorded artifacts for normal verification. The primary review artifacts are
`paper/main.pdf`,
`paper/supplement.pdf`, `data/results/claim_evidence_ledger/*`,
`data/results/horizons_ephemeris_force_model_contrast/*`,
`data/cache/horizons/*`,
`data/results/bicircular_solar_tidal_stress/*`,
`data/results/bicircular_tail_coast_recovery/*`,
`data/results/independent_hs_bicircular_phase_stress/*`,
`data/results/independent_hs_horizons_solar_tidal_replay/*`,
`data/results/evidence_synthesis/*`,
`data/results/replay_stress_validation/*`,
`data/results/independent_hs_all_configured_headroom/*`,
`data/results/independent_hs_branch_control_replay/*`,
`data/results/phase_shift_cardinality_30seed/*`,
`data/results/qaoa_depth_ablation_30seed/*`,
`data/results/hard_catalog_tail_coast_recovery/*`,
`data/results/hard_catalog_tail_coast_branch_control_replay/*`, and
`data/results/artifact_manifest.json`.

## Artifact Map

| Evidence family | Command | Key inputs | Key outputs | Runtime note |
| --- | --- | --- | --- | --- |
| Paper PDFs | `latexmk -pdf paper/main.tex` and `latexmk -pdf paper/supplement.tex` or equivalent local LaTeX build | `paper/main.tex`, `paper/supplement.tex`, `paper/references.bib`, generated `tables/`, `figures/` | `paper/main.pdf`, `paper/supplement.pdf` | Build time depends on local TeX install; not an experiment. |
| Smoke tests | `python -m pytest tests` | `tests/test_smoke.py`, `src/qlt/*`, `configs/smoke.yaml` | pytest pass/fail output | Short. |
| Claim evidence ledger | `py -3.11 scripts\run_claim_evidence_ledger.py` | Recorded summary/statistical CSV/JSON artifacts from the 30-seed main-method package, QAOA/QUBO ablation, continuation extension, direct collocation, independent-midpoint Hermite-Simpson baseline, independent-HS all-configured headroom, independent-HS branch-control replay, independent-HS bicircular phase-sweep stress, independent-HS cached-Horizons solar-tidal replay, tail-coast, delayed-recovery, focused tail-coast branch-control replay, Horizons ephemeris contrast, bicircular solar-tidal stress, and bicircular retuned recovery packages | `data/results/claim_evidence_ledger/claim_evidence_ledger.csv`, `data/results/claim_evidence_ledger/claim_evidence_ledger_metadata.json`, `data/results/claim_evidence_ledger/tail_coast_threshold_audit.csv`, `data/results/claim_evidence_ledger/tail_coast_branch_audit.csv`, `tables/claim_evidence_ledger/*` | Short deterministic postprocessor; no trajectory optimization or high-fidelity validation. The current snapshot emits the independent-HS all-configured headroom row, the independent-HS normalized CR3BP branch-control replay row, the positive independent-HS simple bicircular phase-sweep stress row, the positive independent-HS cached-Horizons-derived solar-tidal replay row, the tail-coast normalized CR3BP accepted-control replay row, the Horizons force-model contrast row, the negative bicircular solar-tidal stress row, and the completed negative bicircular retuned recovery row because those packages exist. |
| Evidence synthesis replay | `py -3.11 scripts\run_evidence_synthesis.py` | Recorded CSV/JSON artifacts from threshold sensitivity, continuation extension, direct collocation, independent-midpoint Hermite-Simpson baseline, independent-HS all-configured headroom, independent-HS branch-control replay, independent-HS bicircular phase-sweep stress, independent-HS cached-Horizons solar-tidal replay, and tail-coast packages | `data/results/evidence_synthesis/evidence_synthesis.csv`, `data/results/evidence_synthesis/evidence_synthesis_metadata.json`, `tables/evidence_synthesis/evidence_synthesis_table.tex`, `tables/evidence_synthesis/practitioner_lessons_table.tex` | Short deterministic postprocessor; no trajectory optimization is rerun. |
| Recorded-control replay/stress validation | `py -3.11 scripts\run_replay_stress_validation.py` | Recorded nominal-control sidecars and source rows from continuation extension, independent-midpoint Hermite-Simpson baseline, and independent-HS all-configured headroom packages; `data/source_states.json` | `data/results/replay_stress_validation/replay_stress_validation.csv`, `data/results/replay_stress_validation/replay_stress_validation_metadata.json`, `tables/replay_stress_validation/replay_stress_validation_table.tex` | Short deterministic postprocessor; repropagates nominal endpoint controls and, for IHS rows, midpoint controls. No least-squares optimization, branch recovery replay, high-fidelity force model, or operational validation claim. |
| Independent-HS branch-control replay | `py -3.11 scripts\run_independent_hs_branch_control_replay.py --config configs\independent_hs_all_configured_headroom.yaml --source-states data\source_states.json` | `configs/independent_hs_all_configured_headroom.yaml`, `data/source_states.json`, independent-HS all-configured CSV, replay-ready branch manifests, nominal sidecars, and full-length endpoint-plus-midpoint branch-control sidecars | `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay.csv`, `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay_metadata.json`, `tables/independent_hs_branch_control_replay/independent_hs_branch_control_replay_table.tex` | Short deterministic postprocessor; validates manifest/sidecar SHA-256 hashes and repropagates persisted controls under normalized CR3BP only. Current package has 2 nominal rows, 16 branch rows, zero replay deltas at tolerance `1e-10`, and `passes_tolerance=True`. No optimizer rerun, high-fidelity validation, production solver parity, fuel optimality, or quantum claim. |
| Independent-HS bicircular phase-sweep stress replay | `py -3.11 scripts\run_independent_hs_bicircular_phase_stress.py` | `configs/independent_hs_all_configured_headroom.yaml`, `data/source_states.json`, independent-HS all-configured CSV, replay-ready branch manifests, nominal sidecars, and full-length endpoint-plus-midpoint branch-control sidecars | `data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress.csv`, `data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_metadata.json`, `tables/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_table.tex` | Short deterministic postprocessor; validates manifest/sidecar SHA-256 hashes, repropagates persisted controls under normalized CR3BP, and sweeps Sun phases 0/45/.../315 deg under the simple circular solar-tidal bicircular model. For the converged polish row, all 8 nominal phases and all 64 branch-phase checks pass; max nominal `0.022138676654057693`, max branch `0.08557051343145317`. Not SPICE/high-fidelity validation, production solver parity, fuel optimality, or quantum evidence. |
| Independent-HS cached-Horizons solar-tidal replay | `py -3.11 scripts\run_independent_hs_horizons_solar_tidal_replay.py` | Committed cache `data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`, `configs/independent_hs_all_configured_headroom.yaml`, `data/source_states.json`, independent-HS all-configured CSV, replay-ready branch manifests, nominal sidecars, and full-length endpoint-plus-midpoint branch-control sidecars | `data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay.csv`, `data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_metadata.json`, `tables/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_table.tex` | Short deterministic postprocessor; default path is offline. It validates cache compatibility for the representative 2026-Jan-01 epoch, transfer time `0.5`, 8-segment grid, canonical time unit `375190.259 s`, and `384400 km/LU`, then repropagates persisted controls with a simplified solar tide from interpolated cached JPL Horizons Sun geometry. For the converged polish row: nominal `0.018363195236986728`, branch worst `0.07422350563850917`, branch pass `8/8`, CR3BP replay delta `0.0`, Sun LU range `382.6857920288508--382.693044178952`, cache SHA-256 `13fe699371ad67bf1616d38b7afd316bbff72811bbc0f8337cff51d6333897b2`. Not SPICE/high-fidelity/flight validation, production solver parity, fuel optimality, or quantum evidence. Use `--refresh-cache` only when intentionally regenerating the JPL Horizons cache. |
| Horizons ephemeris force-model contrast | `py -3.11 scripts\run_horizons_ephemeris_force_model_contrast.py` | Committed cache `data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json`, focused accepted-control sidecars, `configs/hard_catalog_tail_coast_branch_control_replay.yaml`, `src/qlt/ephemeris_contrast.py`, `data/source_states.json` | `data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast.csv`, `data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_metadata.json`, `tables/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_table.tex` | Short deterministic postprocessor; default path is offline. It validates cache metadata against the configured epoch, transfer time, segment grid, canonical time unit, and fixed reference distance before comparing cached Earth/Moon/Sun geometry and solar-tidal acceleration assumptions. Not SPICE validation, high-fidelity propagation, accepted-control retuning, or a threshold-feasibility result. Use `--refresh-cache` only when intentionally regenerating the JPL Horizons cache. |
| Tail-coast accepted branch-control replay | `py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml --resume`, then `py -3.11 scripts\run_tail_coast_branch_control_replay.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml` | `configs/hard_catalog_tail_coast_branch_control_replay.yaml`, `src/qlt/tail_coast_recovery.py`, `scripts/run_tail_coast_recovery.py`, incremental accepted-control sidecars, progress CSV, manifest, `data/source_states.json` | `data/results/hard_catalog_tail_coast_branch_control_replay/*`, `tables/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay_table.tex` | Included current evidence package, optional to regenerate. The recovery run is checkpointed/resumable and expensive because it reruns the combined tail-coast case; the replay postprocessor is deterministic and should run only after the completed recovery package exists. Replay is normalized CR3BP accepted-control replay only; no optimization rerun, high-fidelity validation, production solver parity, fuel optimality, or quantum advantage claim. |
| Bicircular solar-tidal stress replay | `py -3.11 scripts\run_bicircular_solar_tidal_stress.py` | Focused accepted-control replay package under `data/results/hard_catalog_tail_coast_branch_control_replay/`, `configs/hard_catalog_tail_coast_branch_control_replay.yaml`, `src/qlt/bicircular.py`, `data/source_states.json` | `data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress.csv`, `data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_metadata.json`, `tables/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_table.tex` | Short deterministic postprocessor; no optimizer rerun. Replays nominal plus 27 accepted branch controls for Sun phases 0/90/180/270 deg. Negative external-validity stress result: CR3BP replay delta is 0.0, but nominal solar-tidal rows fail and only 22/108 branch-phase rows pass. Not SPICE, high-fidelity validation, production solver parity, or fuel optimality. |
| Bicircular tail-coast retuned recovery | `py -3.11 scripts\run_bicircular_tail_coast_recovery.py --resume` | Focused accepted-control replay package under `data/results/hard_catalog_tail_coast_branch_control_replay/`, `configs/hard_catalog_tail_coast_branch_control_replay.yaml`, `src/qlt/bicircular.py`, `src/qlt/bicircular_tail_coast_recovery.py`, `data/source_states.json` | `data/results/bicircular_tail_coast_recovery/*`, `tables/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_table.tex` | Expensive completed retuning batch, recorded runtime about 761.5 s. Retunes nominal plus all 27 one/two-segment masks under the simple bicircular solar-tidal model at fixed Sun phase 0 deg, original fixed target, and original final time. Negative result: nominal `0.316772`, configured branch pass `19/27`, max branch `6.0299`, strict branch pass `16/27`, and configured thresholds fail. Not SPICE/high-fidelity/flight validation, production solver parity, fuel optimality, quantum, QUBO, or QAOA evidence. |
| Phase-shift benchmark | `python scripts/run_experiment.py --config configs/q1_phase_shift.yaml` | `configs/q1_phase_shift.yaml`, `data/source_states.json` | `data/results/phase_shift/*`, `figures/phase_shift/*`, `tables/phase_shift/*` | Moderate; metadata records package versions but no total runtime field. |
| Phase-shift cardinality benchmark | `python scripts/run_experiment.py --config configs/q1_phase_shift_cardinality.yaml` | `configs/q1_phase_shift_cardinality.yaml`, `data/source_states.json` | `data/results/phase_shift_cardinality/*`, `figures/phase_shift_cardinality/*`, `tables/phase_shift_cardinality/*` | Moderate to expensive; 10 seeds with branch recovery. |
| Bounded phase suite | `python scripts/run_bounded_phase_suite.py --resume` | `configs/bounded_phase_suite.yaml`, `data/source_states.json` | `data/results/bounded_phase_suite/bounded_phase_suite.csv`, `figures/bounded_phase_suite/*`, `tables/bounded_phase_suite/*` | Expensive; configured runtime budget is 600 s, with recorded cases from about 13 s to 367 s. |
| Robust-margin suite | `python scripts/run_robust_margin_suite.py --resume` | `configs/robust_margin_suite.yaml`, `data/source_states.json` | `data/results/robust_margin_suite/*`, `figures/robust_margin_suite/*`, `tables/robust_margin_suite/*` | Expensive if regenerated; recorded rows include selected-branch thrust-margin and all one-segment outage branch diagnostics. |
| Continuation-extension suite | `py -3.11 scripts\run_continuation_margin_suite.py --config configs/continuation_extension_suite.yaml --resume` | `configs/continuation_extension_suite.yaml`, `data/source_states.json`, persisted nominal-control sidecars for warm rows | `data/results/continuation_extension_suite/*`, `data/results/continuation_extension_suite/controls/*`, `figures/continuation_extension_suite/*`, `tables/continuation_extension_suite/*` | Expensive if regenerated; continuous-backend direct multiple-shooting continuation baseline, not a quantum or discrete-sampler run. |
| Direct-collocation baseline | `python scripts/run_direct_collocation_baseline.py --config configs/direct_collocation_baseline.yaml` | `configs/direct_collocation_baseline.yaml`, `src/qlt/direct_collocation.py`, `data/source_states.json` | `data/results/direct_collocation_baseline/*`, `figures/direct_collocation_baseline/*`, `tables/direct_collocation_baseline/*` | Expensive if regenerated; use recorded artifacts for short verification. |
| Constant-control Hermite-Simpson continuation baseline/probe | `py -3.11 scripts\run_hermite_simpson_continuation.py --resume` | `configs/hermite_simpson_continuation_baseline.yaml`, `data/source_states.json`, persisted nominal-control sidecars for warm rows | `data/results/hermite_simpson_continuation_baseline/*`, `data/results/hermite_simpson_continuation_baseline/controls/*`, `figures/hermite_simpson_continuation_baseline/*`, `tables/hermite_simpson_continuation_baseline/*` | Expensive if regenerated; continuous-backend diagnostic with constant segment controls and no independent midpoint controls. Not quantum/discrete evidence. |
| Independent-midpoint-control Hermite-Simpson continuation evidence | `py -3.11 scripts\run_independent_hs_continuation.py --config configs\independent_hs_continuation_baseline.yaml --source-states data\source_states.json --resume` | `configs/independent_hs_continuation_baseline.yaml`, `data/source_states.json`, persisted endpoint-plus-midpoint nominal-control sidecars for warm rows | `data/results/independent_hs_continuation_baseline/*`, `data/results/independent_hs_continuation_baseline/controls/*`, `figures/independent_hs_continuation_baseline/*`, `tables/independent_hs_continuation_baseline/*` | Expensive if regenerated; continuous-backend diagnostic with independent midpoint controls and endpoint/midpoint sidecar hashes. Not quantum/discrete evidence. The phase-time `0.2` and bounded catalog-DRO diagnostics remain negative. |
| Independent-midpoint-control Hermite-Simpson all-configured headroom | `py -3.11 scripts\run_independent_hs_continuation.py --config configs\independent_hs_all_configured_headroom.yaml --source-states data\source_states.json --resume` | `configs/independent_hs_all_configured_headroom.yaml`, `data/source_states.json`, persisted endpoint-plus-midpoint nominal-control sidecars from the p=0.3 source row and replay-ready branch-control sidecars for the p=0.4 rows | `data/results/independent_hs_all_configured_headroom/*`, `data/results/independent_hs_all_configured_headroom/controls/*`, `figures/independent_hs_all_configured_headroom/*`, `tables/independent_hs_all_configured_headroom/*` | Expensive if regenerated; recorded continuation run took about 3433.7 s after the p=0.3 source row. Normalized-CR3BP continuous-backend evidence only. The original p=0.4, `amax=0.2` row selects/evaluates all 8 configured one-segment masks, records nominal `0.011115187774142957`, selected/all worst `0.07741645121655767`, and reaches `max_nfev=120` with `optimizer_success=False`. The polished p=0.4 row warm-starts from that row and its branch controls, records nominal `0.011333095366088189`, selected/all worst `0.07792080291839382`, and converges with `optimizer_success=True` at `nfev=25` under `max_nfev=240`. Both p=0.4 rows have 8 replay-ready branch sidecars. Not high-fidelity validation, production solver parity, fuel optimality, broader outage-family robustness, QUBO/QAOA, or quantum evidence. |
| Phase-shift cardinality main-method 30-seed package | `py -3.11 scripts\run_experiment.py --config configs\q1_phase_shift_cardinality_30seed.yaml`, `py -3.11 scripts\run_main_method_statistics.py --config configs\q1_phase_shift_cardinality_30seed.yaml`, and `py -3.11 scripts\run_threshold_sensitivity.py` | `configs/q1_phase_shift_cardinality_30seed.yaml`, `data/source_states.json`, recorded `data/results/phase_shift_cardinality_30seed/raw_results.csv` | `data/results/phase_shift_cardinality_30seed/*`, `figures/phase_shift_cardinality_30seed/*`, `tables/phase_shift_cardinality_30seed/*`, including `threshold_sensitivity.csv`, `threshold_sensitivity_metadata.json`, and `threshold_sensitivity_table.tex` | Expensive to regenerate the raw run; statistics and threshold-sensitivity generation are short. 210 rows = 30 seeds x 7 methods; threshold sensitivity is derived from recorded raw results only and does not rerun optimization. No quantum-advantage or QAOA-superiority claim. |
| QAOA depth ablation, 30-seed statistics | `python scripts/run_qaoa_depth_ablation.py --config configs/qaoa_depth_ablation_30seed.yaml --angle-restarts 1 --maxiter 10` | `configs/qaoa_depth_ablation_30seed.yaml`, `configs/q1_phase_shift_cardinality.yaml` | `data/results/qaoa_depth_ablation_30seed/*`, `figures/qaoa_depth_ablation_30seed/*`, `tables/qaoa_depth_ablation_30seed/*` | Expensive; recorded runtime is 3301.5 s. QAOA-depth statistical package; no superiority or quantum-advantage claim. |
| Cardinality ablation | `python scripts/run_cardinality_ablation.py` | `configs/q1_phase_shift_cardinality.yaml` | `data/results/phase_shift_cardinality_ablation/*`, `figures/phase_shift_cardinality_ablation/*`, `tables/phase_shift_cardinality_ablation/*` | Expensive; recorded runtime is 2069.8 s. |
| Teacher feasible benchmark | `python scripts/run_experiment.py --config configs/q1_teacher_feasible.yaml` | `configs/q1_teacher_feasible.yaml`, teacher target metadata in run output | `data/results/teacher_feasible/*`, `figures/teacher_feasible/*`, `tables/teacher_feasible/*` | Moderate; teacher controls are diagnostic and disclosed in metadata. |
| Feasibility sweep | `python scripts/run_feasibility_sweep.py --config configs/q1_candidate.yaml --resume --max-cases 0` | `configs/q1_candidate.yaml` | `data/results/feasibility_sweep.csv`, `data/results/feasibility_metadata.json`, `tables/feasibility_table.tex` | Resume-only command is short; full sweep can be expensive. |
| Catalog targeted feasibility | `python scripts/run_feasibility_sweep.py --config configs/catalog_targeted_feasibility.yaml --transfer-times 4.0 --amax 0.3 --segments 14 --max-nfev 250 --multistart --random-starts 3 --include-bang-bang --min-recovery-segments 4 --state-residual-weight 1.25 --robust-residual-weight 1.15 --fuel-residual-weight 0.01 --smooth-residual-weight 0.006 --control-regularization 0.006 --max-cases 1` | `configs/catalog_targeted_feasibility.yaml`, `data/source_states.json` | `data/results/catalog_targeted_feasibility/*`, `figures/catalog_targeted_feasibility/*`, `tables/catalog_targeted_feasibility/*` | Expensive if expanded beyond the single recorded case. |
| Selected-outage hard-catalog envelope | `python scripts/run_catalog_feasibility_envelope.py --config configs/hard_catalog_selected_outage_envelope.yaml --resume` | `configs/hard_catalog_selected_outage_envelope.yaml`, `data/source_states.json` | `data/results/hard_catalog_selected_outage_envelope/*`, `figures/hard_catalog_selected_outage_envelope/*`, `tables/hard_catalog_selected_outage_envelope/*` | Expensive if regenerated; negative robustness probe. Selected recovery errors are small for chosen masks, but nominal thresholds fail, optimizer/backend success is false, and all-mask diagnostics remain high. |
| Locked-nominal hard-catalog branch recovery | `py -3.11 scripts\run_locked_nominal_recovery.py --config configs\hard_catalog_locked_nominal_recovery.yaml --resume` | `configs/hard_catalog_locked_nominal_recovery.yaml`, `src/qlt/locked_recovery.py`, `data/source_states.json` | `data/results/hard_catalog_locked_nominal_recovery/*`, `figures/hard_catalog_locked_nominal_recovery/*`, `tables/hard_catalog_locked_nominal_recovery/*` | Expensive if regenerated; continuous-backend diagnostic only, not quantum evidence. Freezes a feasible nominal control and optimizes each selected branch independently. The bounded one-segment, six-recovery-segment subset (`locked_hard_single_min6_selected8`) meets thresholds but is not optimizer-converged; selected one/two-segment and all-single-outage scopes fail. The all-mask column is a diagnostic, not a robustness claim. |
| Delayed-arrival locked-nominal hard-catalog portfolio recovery | `py -3.11 scripts\run_delayed_locked_recovery.py --config configs\hard_catalog_delayed_recovery.yaml --resume` | `configs/hard_catalog_delayed_recovery.yaml`, `src/qlt/delayed_recovery.py`, `scripts/run_delayed_locked_recovery.py`, `data/source_states.json` | `data/results/hard_catalog_delayed_recovery/*`, `figures/hard_catalog_delayed_recovery/*`, `tables/hard_catalog_delayed_recovery/*` | Expensive if regenerated; the recorded h6 portfolio row took about 2639.7 s. Continuous-backend delayed-arrival horizon evidence only, not fixed-final-time, fuel-optimal, quantum, QUBO, or QAOA evidence. The h6 portfolio recovers all one-segment masks against the delayed target; the h4 regularized all-single row fails; no delayed-arrival two-segment row is included. |
| Tail-coast fixed-final-time hard-catalog recovery | Evidence replay: `py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --resume`; artifact-only refresh: `py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_recovery.yaml --regenerate-artifacts-only --allow-artifact-refresh-fingerprint-mismatch` | `configs/hard_catalog_tail_coast_recovery.yaml`, `src/qlt/tail_coast_recovery.py`, `scripts/run_tail_coast_recovery.py`, `data/source_states.json`, recorded `tail_coast_recovery.csv` for artifact-only refresh | `data/results/hard_catalog_tail_coast_recovery/*`, `figures/hard_catalog_tail_coast_recovery/*`, `tables/hard_catalog_tail_coast_recovery/*` | Expensive if regenerated; the recorded combined one/two row took about 1833.6 s. Continuous-backend fixed-final-time evidence only, not fuel-optimal, quantum, QUBO, or QAOA evidence. The combined `tail_coast_all_one_two_segment_t5_portfolio` row uses `outage_lengths=[1,2]`, selects/evaluates all 27 configured one- and two-segment masks, has nominal tail-coast error `0.02299233817855882`, selected/all fixed-time worst error `0.0936063931709301`, `meets_thresholds=True`, `nfev=5929`, `25/25` eligible optimizer-converged branches, and `branch_optimizer_all_success=False` because two no-recovery-variable late-tail branches are threshold-feasible direct evaluations. Separate all-single and all-two-segment rows remain provenance/scope rows. |
| Multiple-shooting feasibility | `python scripts/run_multiple_shooting_feasibility.py --config configs/q1_candidate.yaml --resume --max-cases 0` | `configs/q1_candidate.yaml` | `data/results/multiple_shooting_feasibility.csv`, `data/results/multiple_shooting_feasibility_metadata.json`, `figures/multiple_shooting_feasibility.*`, `tables/multiple_shooting_feasibility_table.tex` | Resume-only command is short; full case recorded 294.7 s. |
| Catalog collocation feasibility | `python scripts/run_catalog_collocation_feasibility.py --resume --max-cases 0` | catalog collocation settings encoded by the script and metadata | `data/results/catalog_collocation_feasibility/*`, `figures/catalog_collocation_feasibility/*`, `tables/catalog_collocation_feasibility/*` | Resume-only command is short; full collocation search is expensive and currently has no feasible case. |

## Claim-to-Artifact Trace

- Controlled benchmark framing and limitations: `paper/main.tex`, `README.md`,
  `data/results/*/run_metadata.json`, and this file.
- Reviewer-facing claim evidence ledger:
  `data/results/claim_evidence_ledger/claim_evidence_ledger.csv`,
  `data/results/claim_evidence_ledger/claim_evidence_ledger_metadata.json`,
  `data/results/claim_evidence_ledger/tail_coast_threshold_audit.csv`,
  `data/results/claim_evidence_ledger/tail_coast_branch_audit.csv`, and
  `tables/claim_evidence_ledger/*`. The ledger separates selected-branch
  evidence, all-mask diagnostics, and all-configured-mask evidence; it is a
  deterministic replay over recorded artifacts and does not rerun trajectory
  optimization or claim high-fidelity validation. The current snapshot includes
  the independent-HS all-configured headroom row, the independent-HS
  branch-control replay row, the positive independent-HS simple bicircular
  phase-sweep stress row, the positive independent-HS cached-Horizons-derived
  solar-tidal replay row, the focused tail-coast branch-control replay row, the
  Horizons force-model contrast row, the negative bicircular solar-tidal stress
  row, and the completed negative bicircular retuned recovery row because their
  real artifacts exist.
- Cross-backend evidence synthesis and practitioner lessons:
  `data/results/evidence_synthesis/evidence_synthesis.csv`,
  `data/results/evidence_synthesis/evidence_synthesis_metadata.json`,
  `tables/evidence_synthesis/evidence_synthesis_table.tex`, and
  `tables/evidence_synthesis/practitioner_lessons_table.tex`. The synthesis is
  a deterministic replay over recorded CSV/JSON artifacts and does not rerun
  trajectory optimization. The current synthesis includes the independent-HS
  branch-control replay row and the independent-HS simple bicircular phase-sweep
  stress row plus the independent-HS cached-Horizons-derived solar-tidal replay
  row when their real replay CSV and metadata exist.
- Independent-HS all-configured CR3BP headroom:
  `configs/independent_hs_all_configured_headroom.yaml`,
  `data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom.csv`,
  `data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom_metadata.json`,
  `data/results/independent_hs_all_configured_headroom/controls/*`,
  `tables/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom_table.tex`,
  and `figures/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom.*`.
  This package optimizes all eight configured one-segment masks for
  `ihs_all_single_p04_amax02_warm_from_p03` and records nominal
  `0.011115187774142957` and selected/all worst `0.07741645121655767`;
  that original row remains threshold-feasible but retains the
  `max_nfev=120`/`optimizer_success=False` caveat. The polished p=0.4 row
  `ihs_all_single_p04_amax02_polish_from_p04` warm-starts from that row and
  its persisted branch controls, records nominal `0.011333095366088189` and
  selected/all worst `0.07792080291839382`, and converges with
  `optimizer_success=True` at `nfev=25`. Both p=0.4 rows have replay-ready
  manifests with 8 branch-control sidecars. It is normalized-CR3BP
  continuous-backend evidence only.
- Recorded-control replay/stress validation:
  `data/results/replay_stress_validation/replay_stress_validation.csv`,
  `data/results/replay_stress_validation/replay_stress_validation_metadata.json`,
  and `tables/replay_stress_validation/replay_stress_validation_table.tex`.
  This repropagates persisted nominal-control sidecars only, now including the
  all-configured independent-HS headroom row with endpoint and midpoint nominal
  controls; it does not run optimization, replay branch recovery controls, or
  claim high-fidelity validation.
- Independent-HS branch-control replay:
  `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay.csv`,
  `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay_metadata.json`,
  and `tables/independent_hs_branch_control_replay/independent_hs_branch_control_replay_table.tex`.
  This deterministic postprocessor validates the p=0.4 independent-HS
  branch-control manifests and sidecar hashes, then repropagates full-length
  endpoint and midpoint branch schedules under normalized CR3BP only. It covers
  2 p=0.4 cases, 16 branch rows, zero replay deltas at tolerance `1e-10`, and
  `passes_tolerance=True`. It does not rerun optimization or claim
  high-fidelity validation, production solver parity, fuel optimality, or
  quantum advantage.
- Independent-HS simple bicircular phase-sweep stress replay:
  `data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress.csv`,
  `data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_metadata.json`,
  and
  `tables/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_table.tex`.
  This deterministic postprocessor validates the replay-ready p=0.4
  independent-HS sidecar manifests and SHA-256 hashes, repropagates persisted
  endpoint-plus-midpoint controls under normalized CR3BP, and then sweeps Sun
  phases 0/45/.../315 deg under the simple circular solar-tidal bicircular
  model. For the converged polish row, all 8 nominal phases and all 64
  branch-phase checks pass the configured `0.09/0.17` thresholds, with maximum
  nominal error `0.022138676654057693` and maximum branch-phase error
  `0.08557051343145317`. It is not SPICE/high-fidelity validation, production
  solver parity, fuel optimality, or quantum evidence.
- Independent-HS cached-Horizons-derived solar-tidal replay:
  `data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`,
  `data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay.csv`,
  `data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_metadata.json`,
  and
  `tables/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_table.tex`.
  The default path is offline from the committed representative 2026-Jan-01
  JPL Horizons Moon/Sun vector cache. It validates cache compatibility with
  `tf=0.5`, 8 segments, canonical time unit `375190.259 s`, and `384400 km/LU`,
  then repropagates persisted endpoint-plus-midpoint controls with a simplified
  solar-tidal term from linearly interpolated cached Horizons Sun geometry. For
  the converged polish row, nominal error is `0.018363195236986728`, branch
  worst is `0.07422350563850917`, branch pass count is `8/8`, CR3BP replay
  delta is `0.0`, Sun distance range is
  `382.6857920288508--382.693044178952` LU, and cache SHA-256 is
  `13fe699371ad67bf1616d38b7afd316bbff72811bbc0f8337cff51d6333897b2`. It is
  stronger than the simple bicircular stress row because it uses cached JPL
  Horizons geometry, but it is not SPICE/high-fidelity/flight validation,
  production solver parity, flight-ready evidence, fuel optimality, or quantum
  evidence.
- Horizons ephemeris force-model contrast:
  `data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json`,
  `data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast.csv`,
  `data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_metadata.json`,
  and
  `tables/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_table.tex`.
  The default path is offline from the committed cache, which records exact JPL
  Horizons API URLs and query parameters. This package compares Earth-Moon
  distance/rate variation, fixed-reference Sun distance/phase variation, and solar-tidal
  acceleration differences only; it is not SPICE validation, high-fidelity
  propagation, accepted-control retuning, or a threshold-feasibility result.
- Focused tail-coast accepted branch-control replay:
  `data/results/hard_catalog_tail_coast_branch_control_replay/tail_coast_recovery.csv`,
  `data/results/hard_catalog_tail_coast_branch_control_replay/controls/*`,
  `data/results/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay.csv`,
  `data/results/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay_metadata.json`,
  and `tables/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay_table.tex`.
  These completed recovery and replay artifacts are part of the current reported
  artifact snapshot. The replay repropagates persisted accepted full-control
  schedules under normalized CR3BP only; it does not rerun branch optimization,
  high-fidelity validation, fuel-optimal analysis, production solver parity
  checks, or any quantum/QUBO/QAOA workflow.
- Bicircular solar-tidal stress replay:
  `data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress.csv`,
  `data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_metadata.json`,
  and `tables/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_table.tex`.
  This deterministic beyond-CR3BP stress probe reuses persisted accepted
  controls and a simple circular solar third-body tidal term. It is not SPICE
  ephemeris validation, production solver parity, fuel optimality, or
  high-fidelity flight validation. The current run has zero CR3BP replay delta
  but fails the configured solar-tidal threshold sweep: nominal rows fail and
  only 22/108 branch-phase rows pass.
- Bicircular tail-coast retuned recovery:
  `data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery.csv`,
  `data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_summary.csv`,
  `data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_metadata.json`,
  `data/results/bicircular_tail_coast_recovery/controls/*`, and
  `tables/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_table.tex`.
  This completed retuning batch uses a simple bicircular solar-tidal model at
  fixed phase 0 deg, the original fixed target, the original final time, and
  all 27 configured one/two-segment masks. It retunes but remains negative:
  nominal error `0.316772`, configured branch pass `19/27`, max retuned branch
  error `6.0299`, strict branch pass `16/27`, and configured thresholds fail.
  It is not SPICE/high-fidelity/flight validation, production solver parity,
  fuel optimality, quantum, QUBO, or QAOA evidence.
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
- Independent-midpoint-control Hermite-Simpson all-configured headroom:
  `data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom_metadata.json`,
  `data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom.csv`,
  `data/results/independent_hs_all_configured_headroom/controls/*`,
  `tables/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom_table.tex`,
  and `figures/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom.*`.
- Independent-midpoint-control Hermite-Simpson branch-control replay:
  `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay.csv`,
  `data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay_metadata.json`,
  and `tables/independent_hs_branch_control_replay/independent_hs_branch_control_replay_table.tex`.
- Duty-cycle-prior 30-seed main-method statistics:
  `data/results/phase_shift_cardinality_30seed/raw_results.csv`,
  `data/results/phase_shift_cardinality_30seed/success_intervals.csv`,
  `data/results/phase_shift_cardinality_30seed/paired_comparisons.csv`,
  `data/results/phase_shift_cardinality_30seed/threshold_sensitivity.csv`,
  `data/results/phase_shift_cardinality_30seed/threshold_sensitivity_metadata.json`,
  `data/results/phase_shift_cardinality_30seed/main_method_statistics_metadata.json`,
  `tables/phase_shift_cardinality_30seed/results_table.tex`,
  `tables/phase_shift_cardinality_30seed/main_method_statistics_table.tex`,
  `tables/phase_shift_cardinality_30seed/threshold_sensitivity_table.tex`,
  and `figures/phase_shift_cardinality_30seed/main_method_statistics_summary.*`.
  The threshold-sensitivity table is a raw-CSV-only sanity analysis: at
  `(0.05, 0.09)`, all sampled methods are `0/30` and all-windows continuous is
  `30/30`.
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
  fails, and no delayed-arrival two-segment row is included):
  `data/results/hard_catalog_delayed_recovery/delayed_locked_recovery_metadata.json`,
  `data/results/hard_catalog_delayed_recovery/delayed_locked_recovery.csv`,
  `tables/hard_catalog_delayed_recovery/delayed_locked_recovery_table.tex`,
  and `figures/hard_catalog_delayed_recovery/delayed_locked_recovery.*`.
- Tail-coast fixed-final-time hard-catalog recovery diagnostic
  (continuous-backend evidence only; final five nominal controls fixed exactly
  zero; the combined `tail_coast_all_one_two_segment_t5_portfolio` row
  selects/evaluates all 27 configured one- and two-segment masks in one case
  row; nominal tail-coast error is `0.02299233817855882`, selected/all fixed-time
  worst error is `0.0936063931709301`, fallback starts are evaluated and
  accepted for 4 branches, eligible branches with recovery variables converge
  `25/25`, and the two no-recovery-variable late-tail branches are
  threshold-feasible direct evaluations rather than optimizer convergence;
  separate all-single and all-two-segment rows are retained as provenance/scope
  rows):
  `data/results/hard_catalog_tail_coast_recovery/tail_coast_recovery_metadata.json`,
  `data/results/hard_catalog_tail_coast_recovery/tail_coast_recovery.csv`,
  `tables/hard_catalog_tail_coast_recovery/tail_coast_recovery_table.tex`,
  and `figures/hard_catalog_tail_coast_recovery/tail_coast_recovery.*`.

Legacy 10-seed QAOA-depth artifacts remain under `data/results/qaoa_depth_ablation/`,
`tables/qaoa_depth_ablation/`, and `figures/qaoa_depth_ablation/`, but they are
not the current 30-seed QAOA-depth evidence for the manuscript.

## Integrity Manifest

`data/results/artifact_manifest.json` records scoped SHA-256 hashes for key
source, configuration, manuscript, result, table, and figure files. It is
generated and checked with:

```powershell
py -3.11 scripts\write_artifact_manifest.py
py -3.11 scripts\write_artifact_manifest.py --check
```

It excludes `.venv`, runtime caches, logs, LaTeX auxiliary files, and the
manifest file itself. The committed Horizons cache under `data/cache/horizons/`
is included as a data artifact. The manifest hashes and byte counts are
authoritative for artifact identity. The `git_head_at_generation` field is an audit snapshot, not the final
commit identifier after the manifest is committed; a committed manifest records
the parent/pre-final state by construction. Historical long-run metadata may
predate the repository or contain dirty-state records, so final submission
provenance should be established from clean git status after final commit,
manifest `--check`, tests, and local LaTeX builds.
