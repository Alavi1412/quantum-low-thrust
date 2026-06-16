# Archival Release Checklist

This repository snapshot does not claim a DOI. A DOI should be added only after
an external archival deposit, such as Zenodo or an institutional repository,
has been completed and the assigned identifier is known.

Archive these files and directories for the current paper package:

- `paper/main.tex`, `paper/main.pdf`, `paper/supplement.tex`,
  `paper/supplement.pdf`, and `paper/references.bib`
- `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`, and
  `requirements-lock.txt`
- `README.md`, `REPRODUCIBILITY.md`, and `ARCHIVAL_RELEASE.md`
- `data/source_states.json`
- `data/results/artifact_manifest.json`
- `data/results/claim_evidence_ledger/`
- `data/results/bicircular_solar_tidal_stress/`
- `data/results/evidence_synthesis/`
- `data/results/replay_stress_validation/`
- `data/results/phase_shift_cardinality_30seed/`
- `data/results/qaoa_depth_ablation_30seed/`
- `data/results/hard_catalog_tail_coast_recovery/`
- `data/results/hard_catalog_tail_coast_branch_control_replay/`
- `tables/` and `figures/`

Before deposit, run the short verification path in `REPRODUCIBILITY.md`, refresh
`data/results/artifact_manifest.json`, and record the clean repository commit or
archive hash used for the deposit. After a DOI is assigned, update the manuscript
and documentation with that real DOI only.
