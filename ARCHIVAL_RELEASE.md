# Archival Release Checklist

This repository snapshot does not claim a DOI. Add one only after an external
archival deposit, such as Zenodo or an institutional repository, has completed
and assigned a real identifier.

## Release Identifiers

- Release tag: `TBD`
- Archive DOI: `TBD`
- Archive URL: `TBD`
- Clean commit or archive hash: `TBD`
- Release date: `TBD`

## Files To Archive

- `paper/main.tex`, `paper/main.pdf`, `paper/supplement.tex`,
  `paper/supplement.pdf`, and `paper/references.bib`
- `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`,
  `requirements-lock.txt`, and `CITATION.cff`
- `README.md`, `REPRODUCIBILITY.md`, and `ARCHIVAL_RELEASE.md`
- `data/source_states.json`
- `data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json`
- `data/results/artifact_manifest.json`
- `data/results/claim_evidence_ledger/`
- `data/results/horizons_ephemeris_force_model_contrast/`
- `data/results/bicircular_solar_tidal_stress/`
- `data/results/evidence_synthesis/`
- `data/results/replay_stress_validation/`
- `data/results/phase_shift_cardinality_30seed/`
- `data/results/qaoa_depth_ablation_30seed/`
- `data/results/hard_catalog_tail_coast_recovery/`
- `data/results/hard_catalog_tail_coast_branch_control_replay/`
- `tables/` and `figures/`

## Pre-Deposit Checks

Run the short verification path from `REPRODUCIBILITY.md`, including:

```powershell
py -3.11 scripts\run_horizons_ephemeris_force_model_contrast.py
py -3.11 scripts\run_bicircular_solar_tidal_stress.py
py -3.11 scripts\run_claim_evidence_ledger.py
py -3.11 scripts\write_artifact_manifest.py
py -3.11 scripts\write_artifact_manifest.py --check
git diff --check
```

Build `paper/main.pdf` and `paper/supplement.pdf` locally with `latexmk` before
deposit. Do not rerun expensive long experiments unless intentionally refreshing
the evidence package.

## Post-Deposit Updates

After the archive assigns a DOI:

- Replace `Archive DOI: TBD` above with the real DOI.
- Add the real DOI to `CITATION.cff` if desired by the release workflow.
- Update the manuscript Data and Code Availability statement with the real DOI.
- Regenerate `data/results/artifact_manifest.json`.
- Rebuild both PDFs.

Do not write a placeholder DOI into the manuscript or claim archival release
readiness before those steps are complete.
