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
- `data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`
- `data/results/artifact_manifest.json`
- `data/results/claim_evidence_ledger/`
- `data/results/horizons_ephemeris_force_model_contrast/`
- `data/results/bicircular_solar_tidal_stress/`
- `data/results/bicircular_tail_coast_recovery/`
- `data/results/independent_hs_bicircular_phase_stress/`
- `data/results/independent_hs_horizons_solar_tidal_replay/`
- `data/results/independent_hs_horizons_point_mass_retuning/`
- `data/results/evidence_synthesis/`
- `data/results/replay_stress_validation/`
- `data/results/independent_hs_all_configured_headroom/`
- `data/results/independent_hs_branch_control_replay/`
- `data/results/phase_shift_cardinality_30seed/`
- `data/results/qaoa_depth_ablation_30seed/`
- `data/results/hard_catalog_tail_coast_recovery/`
- `data/results/hard_catalog_tail_coast_branch_control_replay/`
- `tables/` and `figures/`

## Pre-Deposit Checks

Run the short verification path from `REPRODUCIBILITY.md`, including:

```powershell
py -3.11 scripts\verify_submission_snapshot.py
```

The verifier is read-only and checks the manifest, focused tests, primary
artifact paths, and whitespace diffs; it does not mint or imply a DOI. Build
`paper/main.pdf` and `paper/supplement.pdf` locally with `latexmk` before
deposit, then refresh `data/results/artifact_manifest.json` if any archived file
changed and rerun the verifier. Do not rerun expensive long experiments unless
intentionally refreshing the evidence package. The completed bicircular retuned
recovery package is archived under `data/results/bicircular_tail_coast_recovery/`;
refresh it only when intentionally rerunning the expensive negative retuning batch with
`py -3.11 scripts\run_bicircular_tail_coast_recovery.py --resume`.
The independent-HS bicircular phase-sweep stress package is a short deterministic
postprocessor artifact archived under
`data/results/independent_hs_bicircular_phase_stress/`; it is not a
SPICE/high-fidelity validation package.
The independent-HS cached-Horizons-derived solar-tidal replay package is another
short deterministic postprocessor artifact archived under
`data/results/independent_hs_horizons_solar_tidal_replay/` and backed by
`data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`; it is a
representative 2026-Jan-01 stress probe, not SPICE/high-fidelity/flight
validation or production solver parity.
The independent-HS cached-Horizons Earth/Moon/Sun point-mass retuning package is
archived under `data/results/independent_hs_horizons_point_mass_retuning/` and
uses the same cache. It reports failed direct persisted-control replay followed
by independent retuning feasibility at the representative epoch; it is not
SPICE/full high-fidelity/flight validation, production solver parity, fuel
optimality, DOI evidence, or quantum evidence.

## Post-Deposit Updates

After the archive assigns a DOI:

- Replace `Archive DOI: TBD` above with the real DOI.
- Add the real DOI to `CITATION.cff` if desired by the release workflow.
- Update the manuscript Data and Code Availability statement with the real DOI.
- Regenerate `data/results/artifact_manifest.json`.
- Rebuild both PDFs.

Do not write a placeholder DOI into the manuscript or claim archival release
readiness before those steps are complete.
