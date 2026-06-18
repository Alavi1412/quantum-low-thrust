# Archival Release Metadata

Zenodo has assigned the external archive DOI below. This document records the
release identifiers associated with that archive record.

## Release Identifiers

- Release tag: `v1.0.1`
- Tagged commit: `af8443871bae7e0adbbe906c117ddbfe011bf207`
- Archive DOI: `10.5281/zenodo.20746480`
- DOI URL: <https://doi.org/10.5281/zenodo.20746480>
- Zenodo record URL: <https://zenodo.org/records/20746480>

Note: this file revision is a post-deposit metadata update. The Zenodo
`v1.0.1` source archive was created from the tagged commit above, so the
DOI-bearing updates to this file and other metadata files are not inside that
source archive. A later release is needed if those DOI-bearing metadata files
themselves must be archived.

## Archive Scope

- `paper/main.tex`, `paper/main.pdf`, `paper/supplement.tex`,
  `paper/supplement.pdf`, and `paper/references.bib`
- `src/`, `scripts/`, `configs/`, `tests/`, `requirements.txt`,
  `requirements-lock.txt`, and `CITATION.cff`
- `README.md`, `REPRODUCIBILITY.md`, and `ARCHIVAL_RELEASE.md`
- `data/source_states.json`
- `data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json`
- `data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json`
- `data/cache/horizons/independent_hs_phase_shift_2026apr01_vectors.json`
- `data/cache/horizons/independent_hs_phase_shift_2026jul01_vectors.json`
- `data/cache/horizons/independent_hs_phase_shift_2026oct01_vectors.json`
- `data/results/artifact_manifest.json`
- `data/results/claim_evidence_ledger/`
- `data/results/horizons_ephemeris_force_model_contrast/`
- `data/results/bicircular_solar_tidal_stress/`
- `data/results/bicircular_tail_coast_recovery/`
- `data/results/independent_hs_bicircular_phase_stress/`
- `data/results/independent_hs_horizons_solar_tidal_replay/`
- `data/results/independent_hs_horizons_point_mass_retuning/`
- `data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/`
- `data/results/independent_hs_casadi_ipopt_bridge/`
- `data/results/evidence_synthesis/`
- `data/results/replay_stress_validation/`
- `data/results/independent_hs_all_configured_headroom/`
- `data/results/independent_hs_branch_control_replay/`
- `data/results/phase_shift_cardinality_30seed/`
- `data/results/qaoa_depth_ablation_30seed/`
- `data/results/hard_catalog_tail_coast_recovery/`
- `data/results/hard_catalog_tail_coast_branch_control_replay/`
- `tables/` and `figures/`

## Archive Verification

Run the short verification path from `REPRODUCIBILITY.md`, including:

```powershell
py -3.11 scripts\verify_submission_snapshot.py
```

The verifier is read-only and checks the manifest, focused tests, primary
artifact paths, and whitespace diffs; it does not mint or imply a DOI. Rebuild
`paper/main.pdf` and `paper/supplement.pdf` locally with `latexmk` after
manuscript changes, then refresh `data/results/artifact_manifest.json` if any
archived file changed and rerun the verifier. Do not rerun expensive long
experiments unless intentionally refreshing the evidence package. The completed
bicircular retuned recovery package is archived under
`data/results/bicircular_tail_coast_recovery/`; refresh it only when
intentionally rerunning the expensive negative retuning batch with
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
validation or broad production-solver validation/parity.
The independent-HS cached-Horizons Earth/Moon/Sun point-mass retuning package is
archived under `data/results/independent_hs_horizons_point_mass_retuning/` and
uses the same cache. It reports failed direct persisted-control replay followed
by independent retuning feasibility at the representative epoch; it is not
SPICE/full high-fidelity/flight validation, broad production-solver validation/parity, fuel
optimality, DOI evidence, or quantum evidence.
The independent-HS multi-epoch cached-Horizons point-mass retuning package is
archived under
`data/results/independent_hs_horizons_multi_epoch_point_mass_retuning/` and uses
the committed 2026-Jan-01, 2026-Apr-01, 2026-Jul-01, and 2026-Oct-01 caches. It
shows nominal direct replay fails in all four epochs, direct branch replay pass
count `18/32` overall with July at `8/8`, and retuned feasibility for all
nominal and branch rows, but it remains a point-mass stress/retuning package,
not SPICE/full high-fidelity/flight validation, broad production-solver validation/parity, fuel
optimality, DOI evidence, or quantum evidence.
The independent-HS SPICE-derived ephemeris replay package is archived under
`data/results/independent_hs_spice_ephemeris_replay/` and backed by compact
vector caches in `data/cache/spice/`. It replays the 36 already-retuned
multi-epoch controls under SPICE-derived Moon/Sun J2000 geometric vectors with
branch pass `32/32`, worst nominal/branch errors
`0.021439441253166033`/`0.024730650824609506`, and max delta
`3.2763991519857427e-10` from the Horizons-retuned replay. It is a no-retune
point-mass ephemeris-source replay, not full high-fidelity/flight validation,
broad production-solver validation/parity, fuel optimality, DOI evidence, or quantum evidence.
Raw NAIF kernel binaries are not archive artifacts; only compact JSON vector
caches are committed.
The independent-HS CasADi/IPOPT bridge package is archived under
`data/results/independent_hs_casadi_ipopt_bridge/`. It refines the accepted
polish nominal row and eight branch rows under the same normalized CR3BP target,
scales, thresholds, and branch recovery semantics with pre-recovery prefixes
fixed to refined nominal masked controls, post-recovery active branch controls
as IPOPT variables, IPOPT success `9/9`, and max prefix delta `0.0`; it is a scoped
mature NLP backend bridge check, not production mission design, high-fidelity
or flight validation, global/fuel optimality, DOI evidence, or quantum evidence.
