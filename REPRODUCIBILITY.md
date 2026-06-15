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
  snapshot, use git history together with `data/results/artifact_manifest.json`,
  which records the HEAD available before the integration commit and file-level
  hashes for the working tree at manifest generation time. The manifest
  intentionally has no self-entry, so it cannot record the commit that contains
  itself.

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
| Continuation-margin suite | `py -3.11 scripts\run_continuation_margin_suite.py --resume` | `configs/continuation_margin_suite.yaml`, `data/source_states.json`, persisted nominal-control sidecars for warm rows | `data/results/continuation_margin_suite/*`, `data/results/continuation_margin_suite/controls/*`, `figures/continuation_margin_suite/*`, `tables/continuation_margin_suite/*` | Expensive if regenerated; continuous-backend direct multiple-shooting continuation baseline, not a quantum or discrete-sampler run. |
| Direct-collocation baseline | `python scripts/run_direct_collocation_baseline.py --config configs/direct_collocation_baseline.yaml` | `configs/direct_collocation_baseline.yaml`, `src/qlt/direct_collocation.py`, `data/source_states.json` | `data/results/direct_collocation_baseline/*`, `figures/direct_collocation_baseline/*`, `tables/direct_collocation_baseline/*` | Expensive if regenerated; use recorded artifacts for short verification. |
| QAOA depth ablation | `python scripts/run_qaoa_depth_ablation.py --angle-restarts 1 --maxiter 10` | `configs/qaoa_depth_ablation.yaml`, `configs/q1_phase_shift_cardinality.yaml` | `data/results/qaoa_depth_ablation/*`, `figures/qaoa_depth_ablation/*`, `tables/qaoa_depth_ablation/*` | Expensive; recorded runtime is 1291.5 s. |
| Cardinality ablation | `python scripts/run_cardinality_ablation.py` | `configs/q1_phase_shift_cardinality.yaml` | `data/results/phase_shift_cardinality_ablation/*`, `figures/phase_shift_cardinality_ablation/*`, `tables/phase_shift_cardinality_ablation/*` | Expensive; recorded runtime is 2069.8 s. |
| Teacher feasible benchmark | `python scripts/run_experiment.py --config configs/q1_teacher_feasible.yaml` | `configs/q1_teacher_feasible.yaml`, teacher target metadata in run output | `data/results/teacher_feasible/*`, `figures/teacher_feasible/*`, `tables/teacher_feasible/*` | Moderate; teacher controls are diagnostic and disclosed in metadata. |
| Feasibility sweep | `python scripts/run_feasibility_sweep.py --config configs/q1_candidate.yaml --resume --max-cases 0` | `configs/q1_candidate.yaml` | `data/results/feasibility_sweep.csv`, `data/results/feasibility_metadata.json`, `tables/feasibility_table.tex` | Resume-only command is short; full sweep can be expensive. |
| Catalog targeted feasibility | `python scripts/run_feasibility_sweep.py --config configs/catalog_targeted_feasibility.yaml --transfer-times 4.0 --amax 0.3 --segments 14 --max-nfev 250 --multistart --random-starts 3 --include-bang-bang --min-recovery-segments 4 --state-residual-weight 1.25 --robust-residual-weight 1.15 --fuel-residual-weight 0.01 --smooth-residual-weight 0.006 --control-regularization 0.006 --max-cases 1` | `configs/catalog_targeted_feasibility.yaml`, `data/source_states.json` | `data/results/catalog_targeted_feasibility/*`, `figures/catalog_targeted_feasibility/*`, `tables/catalog_targeted_feasibility/*` | Expensive if expanded beyond the single recorded case. |
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
- Continuation-margin continuous-backend claims, warm-start provenance, and
  control-sidecar hashes:
  `data/results/continuation_margin_suite/continuation_margin_suite_metadata.json`,
  `data/results/continuation_margin_suite/continuation_margin_suite.csv`,
  `data/results/continuation_margin_suite/controls/*`,
  `tables/continuation_margin_suite/continuation_margin_suite_table.tex`, and
  `figures/continuation_margin_suite/continuation_margin_suite.*`.
- Direct-collocation baseline comparison:
  `data/results/direct_collocation_baseline/*`,
  `tables/direct_collocation_baseline/*`, and
  `figures/direct_collocation_baseline/*`.
- QAOA depth interpretation limits:
  `data/results/qaoa_depth_ablation/metadata.json`,
  `tables/qaoa_depth_ablation/qaoa_depth_ablation_table.tex`, and
  `figures/qaoa_depth_ablation/qaoa_depth_ablation_summary.*`.
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

## Integrity Manifest

`data/results/artifact_manifest.json` records scoped SHA-256 hashes for key
source, configuration, manuscript, result, table, and figure files. It excludes
`.venv`, caches, logs, exhaustive generated intermediates, and the manifest
file itself.
