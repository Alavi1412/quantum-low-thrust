# Robust Low-Thrust Cislunar Initialization Benchmark

Research-code scaffold for the controlled CR3BP initialization benchmark
resource: "A Reproducible CR3BP Benchmark Resource for Low-Thrust Cislunar
Missed-Thrust Initialization".

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

Reviewer-facing read-only check for the current artifact snapshot:

```powershell
py -3.11 -m pip install -r requirements-lock.txt
py -3.11 scripts\verify_submission_snapshot.py
```

The verifier checks key primary artifact paths, runs
`scripts\write_artifact_manifest.py --check`, runs the focused reproducibility
pytest subset, and runs `git diff --check`. It does not regenerate artifacts,
rerun trajectory optimization, rebuild PDFs, clean LaTeX auxiliary files, or
create an archive DOI. Use `--full-tests` to replace the focused pytest subset
with the full suite; `--skip-tests` and `--skip-git-diff-check` are available for
constrained environments. The long experiment commands below are expensive; use
the recorded artifacts unless intentionally regenerating evidence. The primary
review artifacts are `paper/main.pdf`, `paper/supplement.pdf`,
`data/results/claim_evidence_ledger/*`,
`data/results/independent_hs_all_configured_headroom/*`,
`data/results/horizons_ephemeris_force_model_contrast/*`,
`data/cache/horizons/*`,
`data/results/bicircular_solar_tidal_stress/*`,
`data/results/bicircular_tail_coast_recovery/*`,
`data/results/independent_hs_bicircular_phase_stress/*`,
`data/results/independent_hs_horizons_solar_tidal_replay/*`,
`data/results/independent_hs_horizons_point_mass_retuning/*`,
`data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/*`,
`data/results/independent_hs_casadi_ipopt_bridge/*`,
`data/results/evidence_synthesis/*`,
`data/results/replay_stress_validation/*`,
`data/results/independent_hs_branch_control_replay/*`,
`data/results/phase_shift_cardinality_30seed/*`,
`data/results/qaoa_depth_ablation_30seed/*`,
`data/results/hard_catalog_tail_coast_recovery/*`,
`data/results/hard_catalog_tail_coast_branch_control_replay/*`, and
`data/results/artifact_manifest.json`.

## Reproducibility Manifest

See `REPRODUCIBILITY.md` for the artifact map, expected outputs, known expensive
runs, and commands used for the paper evidence. `ARCHIVAL_RELEASE.md` lists the
files to include in an external repository deposit and does not claim a DOI. A
machine-readable SHA-256
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
The ledger separates selected-branch evidence, all-mask diagnostics,
all-configured-mask evidence, and deterministic branch-control replay rows. The
current snapshot includes the new independent-HS all-configured headroom row,
the independent-HS branch-control replay CSV/metadata, the independent-HS
simple bicircular phase-sweep stress CSV/metadata, the independent-HS
cached-Horizons-derived solar-tidal replay CSV/metadata, the independent-HS
cached-Horizons Earth/Moon/Sun point-mass retuning CSV/metadata, the independent-HS
multi-epoch cached-Horizons point-mass retuning CSV/metadata, the independent-HS
SPICE-derived ephemeris replay CSV/metadata, the independent-HS CasADi/IPOPT
bridge CSV/metadata, the focused tail-coast replay
CSV/metadata, focused source recovery CSV, bicircular solar-tidal stress
CSV/metadata, and Horizons ephemeris force-model contrast CSV/metadata, plus
the bicircular retuned recovery CSV, summary, and metadata, so the ledger has 20
claim rows. The independent-HS bicircular row is a positive simple stress-probe
replay for the converged all-configured row; the independent-HS cached-Horizons
row is a stronger representative-epoch stress replay using cached JPL Horizons
geometry; the new point-mass row reports that persisted controls fail direct
Earth/Moon/Sun point-mass replay but independent retuning restores representative
epoch feasibility, and the multi-epoch point-mass row repeats that stress/retuning
check over four fixed 2026 cached-Horizons epochs; the new SPICE replay row
replays those already-retuned controls under compact SPICE-derived Moon/Sun
vectors with branch pass count `32/32`; the new CasADi/IPOPT row locally
refines the accepted polish nominal plus eight branch rows under the same
normalized CR3BP target and branch recovery semantics with IPOPT success `9/9`;
the
hard-catalog solar-tidal row is a negative stress-probe
row, the retuned recovery row is a completed negative simple-bicircular retuning
row, and the hard-catalog Horizons row is a force-model contrast row. The SPICE
row is a point-mass ephemeris-source replay, not full high-fidelity/flight
validation, broad production-solver validation/parity, or quantum evidence. The CasADi/IPOPT row
is a scoped CasADi/IPOPT mature NLP backend bridge check, not production mission design,
high-fidelity validation, global/fuel optimality, DOI evidence, or quantum
evidence.
The tail-coast audit confirms the combined row
passes recorded-error thresholds through `(0.025, 0.095)` and fails the tighter
`0.09` robust threshold and the `0.02` nominal threshold. The branch audit is a
JSON summary of the historical four-row package only.

The evidence synthesis postprocessor writes
`data/results/evidence_synthesis/evidence_synthesis.csv`,
`data/results/evidence_synthesis/evidence_synthesis_metadata.json`,
`tables/evidence_synthesis/evidence_synthesis_table.tex`, and
`tables/evidence_synthesis/practitioner_lessons_table.tex`. It reads recorded
CSV/JSON artifacts only and does not rerun trajectory optimization. The table
cross-indexes tight 30-seed threshold sensitivity, continuation-extension
multiple-shooting rows, compact direct-collocation and independent-midpoint
Hermite-Simpson diagnostics, the new all-configured independent-HS headroom
row, the independent-HS branch-control replay row, the positive independent-HS
simple bicircular phase-sweep stress row, the independent-HS
cached-Horizons-derived solar-tidal replay row, the independent-HS
cached-Horizons point-mass retuning row, the independent-HS multi-epoch
cached-Horizons point-mass retuning row, the independent-HS SPICE-derived
ephemeris replay row, the independent-HS CasADi/IPOPT bridge row, and the
scoped hard-catalog tail-coast row used in the main manuscript claim path.

The replay/stress validation postprocessor writes
`data/results/replay_stress_validation/replay_stress_validation.csv`,
`data/results/replay_stress_validation/replay_stress_validation_metadata.json`,
and `tables/replay_stress_validation/replay_stress_validation_table.tex`:

```powershell
py -3.11 scripts\run_replay_stress_validation.py
```

It repropagates persisted nominal-control sidecars for representative
continuation-extension and independent-midpoint Hermite-Simpson phase-shift
rows, including the all-configured independent-HS headroom row
`ihs_all_single_p04_amax02_warm_from_p03`. The source-substep baselines
reproduce recorded nominal errors to within `1e-12`; refined substeps and direct
+/-1% acceleration scaling are stress diagnostics only. It does not run
least-squares optimization, replay branch recovery controls, or claim
high-fidelity validation.

The independent-HS branch-control replay postprocessor writes
`data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay.csv`,
`data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay_metadata.json`,
and
`tables/independent_hs_branch_control_replay/independent_hs_branch_control_replay_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_branch_control_replay.py --config configs\independent_hs_all_configured_headroom.yaml --source-states data\source_states.json
```

It validates the all-configured independent-HS branch-control manifests and
sidecar SHA-256 hashes, then repropagates the persisted full endpoint and
midpoint schedules under the normalized CR3BP model. The current package covers
the original p=0.4 row and the polished p=0.4 row, with 8 branch rows each, zero
nominal/branch/all-mask replay deltas at tolerance `1e-10`, and
`passes_tolerance=True`. This is deterministic recorded-control replay only; it
does not rerun optimization, certify optimizer convergence for the original
row, add high-fidelity validation, establish broad production-solver validation/parity, or claim
fuel optimality or quantum advantage.

The independent-HS bicircular phase-sweep stress postprocessor writes
`data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress.csv`,
`data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_metadata.json`,
and
`tables/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_bicircular_phase_stress.py
```

It validates the replay-ready p=0.4 independent-HS sidecar manifests and SHA-256
hashes, repropagates persisted endpoint-plus-midpoint nominal and branch
controls under normalized CR3BP, and then sweeps Sun phases
`0, 45, 90, 135, 180, 225, 270, 315` degrees under the simple circular
solar-tidal bicircular model. For the converged polish row
`ihs_all_single_p04_amax02_polish_from_p04`, all 8 nominal phases and all 64
branch-phase checks pass the configured `0.09/0.17` thresholds. The maximum
simple-bicircular nominal error is `0.022138676654057693`, and the maximum
branch-phase error is `0.08557051343145317`. This is a deterministic
beyond-CR3BP stress probe only; it is not SPICE validation, high-fidelity
validation, broad production-solver validation/parity, fuel optimality, or quantum evidence.

The independent-HS cached-Horizons-derived solar-tidal replay postprocessor
writes
`data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay.csv`,
`data/results/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_metadata.json`,
and
`tables/independent_hs_horizons_solar_tidal_replay/independent_hs_horizons_solar_tidal_replay_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_horizons_solar_tidal_replay.py
```

The default path is offline and reads
`data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`. Use
`--refresh-cache` only when intentionally regenerating the representative
2026-Jan-01 JPL Horizons Moon/Sun vector cache. The script validates cache
compatibility with the p=0.4 independent-HS transfer grid (`tf=0.5`, 8
segments, canonical time unit `375190.259 s`, `384400 km/LU`), validates
manifest and sidecar SHA-256 hashes, and repropagates persisted endpoint-plus-
midpoint controls against the original CR3BP target. For
`ihs_all_single_p04_amax02_polish_from_p04`, the cached-Horizons-derived
solar-tidal replay gives nominal error `0.018363195236986728`, branch worst
`0.07422350563850917`, branch pass count `8/8`, CR3BP replay delta `0.0`, Sun
distance range `382.6857920288508--382.693044178952` LU, and cache SHA-256
`13fe699371ad67bf1616d38b7afd316bbff72811bbc0f8337cff51d6333897b2`. This is
stronger than the simple bicircular phase sweep because it uses cached JPL
Horizons geometry, but it remains a simplified stress probe, not SPICE,
high-fidelity, flight-ready, or broad production-solver validation.

The independent-HS cached-Horizons Earth/Moon/Sun point-mass retuning
postprocessor writes
`data/results/independent_hs_horizons_point_mass_retuning/independent_hs_horizons_point_mass_retuning.csv`,
`data/results/independent_hs_horizons_point_mass_retuning/independent_hs_horizons_point_mass_retuning_metadata.json`,
retuned endpoint-plus-midpoint control sidecars under
`data/results/independent_hs_horizons_point_mass_retuning/controls/`, and
`tables/independent_hs_horizons_point_mass_retuning/independent_hs_horizons_point_mass_retuning_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_horizons_point_mass_retuning.py
```

The default path is offline and uses the same committed representative
2026-Jan-01 JPL Horizons Moon/Sun vector cache. It converts the CR3BP rotating
barycentric state to a geocentric inertial frame, propagates Earth central
gravity plus indirect Moon/Sun point-mass terms, and retunes the nominal and
each branch independently with endpoint-plus-midpoint controls bounded at
`amax=0.2`; branch outage-masked segments remain inactive. Persisted controls
honestly fail direct replay: nominal error is `0.3812580376880591`, branch
worst is `0.3797450961017463`, and persisted branch pass count is `0/8`.
After retuning, nominal error is `0.02143944130524006`, branch worst is
`0.02473065115224942`, branch pass count is `8/8`, SciPy success count is
`9/9`, and total `nfev` is `71`. This is a cached-Horizons point-mass retuning
stress package only; it is not SPICE, full high-fidelity or flight validation,
broad production-solver validation/parity, fuel optimality, DOI evidence, or quantum evidence.

The independent-HS multi-epoch cached-Horizons point-mass retuning wrapper writes
`data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/independent_hs_horizons_multi_epoch_point_mass_retuning.csv`,
`data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/independent_hs_horizons_multi_epoch_point_mass_retuning_metadata.json`,
per-epoch retuned controls under
`data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/epochs/`,
and
`tables/independent_hs_horizons_multi_epoch_point_mass_retuning/independent_hs_horizons_multi_epoch_point_mass_retuning_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_horizons_multi_epoch_point_mass_retuning.py
```

The default path is offline when the committed 2026-Jan-01, 2026-Apr-01,
2026-Jul-01, and 2026-Oct-01 caches exist. It aggregates 36 rows (4 nominal and
32 branch rows). Nominal direct replay fails in all 4 epochs; direct branch pass
count is `18/32` overall, including `8/8` in July. Worst direct nominal/branch
errors are `0.3812580376880591`/`0.3797450961017463`.
After independent retuning, all rows pass: worst retuned nominal/branch errors
are `0.02143944130524006`/`0.02473065115224942`, branch pass count is `32/32`,
SciPy success is `36/36`, and total `nfev` is `197`. This is a stronger
representative-epoch set for the point-mass stress concern, not SPICE/full
high-fidelity/flight validation, broad production-solver validation/parity, fuel optimality, DOI
evidence, or quantum evidence.

The independent-HS SPICE-derived ephemeris replay postprocessor writes
`data/results/independent_hs_spice_ephemeris_replay/independent_hs_spice_ephemeris_replay.csv`,
`data/results/independent_hs_spice_ephemeris_replay/independent_hs_spice_ephemeris_replay_metadata.json`,
compact vector caches under `data/cache/spice/`, and
`tables/independent_hs_spice_ephemeris_replay/independent_hs_spice_ephemeris_replay_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_spice_ephemeris_replay.py
```

The default path is offline from four committed compact SPICE-derived vector
caches. Use `--refresh-spice-cache` only when intentionally downloading NAIF
kernels and regenerating those caches; raw `*.bsp`, `*.tls`, and `*.tpc`
kernels are ignored and removed by the default refresh path. The cache metadata
records `naif0012.tls`, `de442s.bsp`, `gm_de440.tpc`, kernel SHA-256 values,
SpiceyPy `8.1.2`, CSPICE `CSPICE_N0067`, J2000 frame, Earth observer, Moon/Sun
targets, `NONE` aberration correction, and the canonical node JD_TDB grid. The
replay reads the 36 controls already retuned by the multi-epoch
cached-Horizons point-mass package and does not rerun optimization or retuning.
Across 4 nominal and 32 branch rows, all SPICE-derived point-mass replays pass:
worst nominal/branch errors are `0.021439441253166033`/`0.024730650824609506`,
branch pass is `32/32`, and the maximum absolute delta from the Horizons-retuned
replay is `3.2763991519857427e-10`. This is SPICE-derived ephemeris replay under
the same Earth/Moon/Sun point-mass stress model, not full high-fidelity/flight
validation, broad production-solver validation/parity, fuel optimality, DOI evidence, or quantum
evidence.

The independent-HS CasADi/IPOPT bridge check writes
`data/results/independent_hs_casadi_ipopt_bridge/independent_hs_casadi_ipopt_bridge.csv`,
`data/results/independent_hs_casadi_ipopt_bridge/independent_hs_casadi_ipopt_bridge_metadata.json`,
refined endpoint-plus-midpoint control sidecars under
`data/results/independent_hs_casadi_ipopt_bridge/controls/`, and
`tables/independent_hs_casadi_ipopt_bridge/independent_hs_casadi_ipopt_bridge_table.tex`:

```powershell
py -3.11 scripts\run_independent_hs_casadi_ipopt_bridge.py
```

The bridge reads the accepted
`ihs_all_single_p04_amax02_polish_from_p04` nominal and eight branch sidecars,
exports the nominal controls and post-recovery active branch endpoint-plus-midpoint
controls to a CasADi/IPOPT direct-shooting NLP, keeps outage-masked branch
segments fixed inactive, and fixes every branch pre-recovery prefix to the
refined nominal controls with the outage mask applied. It uses the same
normalized CR3BP target, scales, configured thresholds, and branch-recovery
semantics as the source independent-HS artifacts. The current run reports
source replay nominal/branch-worst errors
`0.011333095366088189`/`0.07792080291839382` and CasADi/IPOPT refined
nominal/branch-worst errors `0.009138565365046585`/`0.015534969964216154`,
with IPOPT success `9/9`, bridge pass `9/9`, total IPOPT iterations `66`,
max prefix delta `0.0`, and max control-bound violation `0.0`. This is a scoped mature NLP backend bridge
check only; it is not production mission design, high-fidelity or flight
validation, global/fuel optimality, DOI evidence, or quantum evidence.

The Horizons ephemeris force-model contrast postprocessor writes
`data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast.csv`,
`data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_metadata.json`,
and `tables/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast_table.tex`:

```powershell
py -3.11 scripts\run_horizons_ephemeris_force_model_contrast.py
```

The default path is offline and reads
`data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json`, which
stores exact JPL Horizons API URLs, query parameters, raw responses, and parsed
Moon/Sun vectors for the 15 hard-catalog segment nodes. The current contrast
records Earth-Moon distance ratio range `0.931638--1.04763`, angular-rate ratio
range `0.903116--1.1483`, Sun distance range `382.686--382.896` fixed CR3BP LU
(`384400 km/LU`), and maximum nominal-node solar-tidal acceleration delta
`2.19803e-3` against the aligned bicircular model. The Earth-Moon distance ratio
is a diagnostic ratio to the window mean; Sun vectors and tidal accelerations
use the fixed reference distance recorded in the cache metadata. This is a
force-model contrast only; it is not SPICE validation, high-fidelity propagation,
accepted-control retuning, or a threshold-feasibility result. Use
`--refresh-cache` only when intentionally
regenerating the cached Horizons data.

The focused hard-catalog tail-coast accepted branch-control replay package is
included in the current artifact snapshot. Regenerating the recovery sidecars is
an optional long-run action; use the two-step path only when intentionally
refreshing that evidence:

```powershell
py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml --resume
py -3.11 scripts\run_tail_coast_branch_control_replay.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml
```

The first command reruns only the combined
`tail_coast_all_one_two_segment_t5_portfolio` case into
`data/results/hard_catalog_tail_coast_branch_control_replay/` and writes a
nominal-control sidecar, incremental accepted full-control sidecars, a progress
CSV, and a manifest with SHA-256 hashes. `--resume` reuses compatible completed
branch sidecars by mask index and skips those branch optimizations. The second
command is the short deterministic replay postprocessor: it repropagates the
completed persisted controls under the configured normalized CR3BP model and writes
`tail_coast_branch_control_replay.csv`,
`tail_coast_branch_control_replay_metadata.json`, and
`tables/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay_table.tex`.
The current replay package records one nominal row plus 27 branch rows, maximum
branch terminal-error delta `0.0`, and `passes_tolerance=True`. It does not rerun
optimization, branch portfolio selection, fallback search, high-fidelity
validation, broad production-solver validation/parity checks, fuel-optimal analysis, or any
QUBO/QAOA/quantum workflow.

The bicircular solar-tidal stress postprocessor writes
`data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress.csv`,
`data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_metadata.json`,
and `tables/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_table.tex`:

```powershell
py -3.11 scripts\run_bicircular_solar_tidal_stress.py
```

It reuses the focused accepted-control sidecars and sweeps Sun phases
`0, 90, 180, 270` degrees under a circular solar third-body tidal term with
`sun_distance_lu=389.17`, `sun_mu_ratio=328900.56`, and rotating-frame phase
rate `-0.9252`. The CR3BP replay delta is `0.0`, but the stress probe is
negative: nominal solar-tidal rows fail the `0.09` threshold and only `22/108`
branch-phase rows pass the `0.17` branch threshold. This is beyond-CR3BP stress
evidence only, not SPICE ephemeris validation, broad production-solver validation/parity,
fuel optimality, or high-fidelity flight validation.

The bicircular tail-coast retuned recovery package is a completed expensive
negative retuning batch:

```powershell
py -3.11 scripts\run_bicircular_tail_coast_recovery.py --resume
```

It uses the same focused tail-coast accepted-control sidecars as seeds, retunes
under the simple circular solar-tidal model at fixed Sun phase `0` degrees, and
keeps the original fixed target and final time. The current package covers all
27 configured one- and two-segment masks and writes
`data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery.csv`,
`data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_summary.csv`,
`data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_metadata.json`,
and `tables/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_table.tex`.
It retunes but still fails: nominal error is `0.316772` against the `0.09`
configured threshold, configured branch pass count is `19/27`, maximum retuned
branch error is `6.0299`, and the strict `(0.05,0.09)` branch pass count is
`16/27`. This is not SPICE/high-fidelity/flight validation, production solver
parity, fuel optimality, quantum, QUBO, or QAOA evidence.

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

For accepted branch-control persistence and replay without rewriting the
historical four-row package, use the focused config:

```powershell
py -3.11 scripts\run_tail_coast_recovery.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml --resume
py -3.11 scripts\run_tail_coast_branch_control_replay.py --config configs\hard_catalog_tail_coast_branch_control_replay.yaml
```

This focused package keeps the same scientific setup and thresholds as the
combined row, but writes only that case under
`data/results/hard_catalog_tail_coast_branch_control_replay/`. In the current
snapshot, the completed recovery CSV, replay CSV, metadata, and table exist and
support the accepted-control replay row in the claim ledger. The replay is a
normalized CR3BP accepted-control replay only; it is not an optimization rerun,
high-fidelity validation, broad production-solver validation/parity, fuel optimality, or quantum
advantage evidence.

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

For the independent-midpoint-control Hermite-Simpson all-configured headroom
package:

```powershell
py -3.11 scripts\run_independent_hs_continuation.py --config configs\independent_hs_all_configured_headroom.yaml --source-states data\source_states.json --resume
```

The package writes `data/results/independent_hs_all_configured_headroom/*`,
endpoint-plus-midpoint sidecars under
`data/results/independent_hs_all_configured_headroom/controls/`,
`tables/independent_hs_all_configured_headroom/*`, and
`figures/independent_hs_all_configured_headroom/*`. For rows with
`persist_branch_controls: true`, it also writes branch-control manifests and
full-length endpoint-plus-midpoint branch-control sidecars under the same
controls directory. The original key row
`ihs_all_single_p04_amax02_warm_from_p03` selects and evaluates all 8 configured
one-segment outage masks at phase time `0.4`, transfer time `0.5`, and
`amax=0.2`, with nominal error `0.011115187774142957` and selected/all worst
error `0.07741645121655767`; it remains threshold-feasible but reaches
`max_nfev=120` with `optimizer_success=False`. The polish row
`ihs_all_single_p04_amax02_polish_from_p04` warm-starts from that row and its
branch sidecars, selects/evaluates the same 8 masks, records nominal error
`0.011333095366088189` and selected/all worst error `0.07792080291839382`, and
converges with `optimizer_success=True` at `nfev=25` under `max_nfev=240`.
Both p=0.4 rows have replay-ready manifests with 8 branch-control sidecars. This
is normalized-CR3BP all-configured continuous backend evidence only; it does not
claim high-fidelity validation, broad production-solver validation/parity, fuel optimality,
broader outage-family robustness, QUBO/QAOA evidence, or quantum advantage.

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
