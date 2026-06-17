"""Read-only reviewer verification for the current submission snapshot."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PRIMARY_ARTIFACT_PATHS = [
    "paper/main.pdf",
    "paper/supplement.pdf",
    "README.md",
    "REPRODUCIBILITY.md",
    "ARCHIVAL_RELEASE.md",
    "data/source_states.json",
    "data/cache/horizons/hard_catalog_tail_coast_2026jan01_vectors.json",
    "data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json",
    "data/results/artifact_manifest.json",
    "data/results/claim_evidence_ledger/claim_evidence_ledger.csv",
    "data/results/claim_evidence_ledger/claim_evidence_ledger_metadata.json",
    "data/results/evidence_synthesis/evidence_synthesis.csv",
    "data/results/evidence_synthesis/evidence_synthesis_metadata.json",
    "data/results/replay_stress_validation/replay_stress_validation.csv",
    "data/results/replay_stress_validation/replay_stress_validation_metadata.json",
    "data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom.csv",
    "data/results/independent_hs_all_configured_headroom/independent_hs_all_configured_headroom_metadata.json",
    (
        "data/results/independent_hs_all_configured_headroom/controls/"
        "ihs_all_single_p04_amax02_warm_from_p03_nominal_controls.json"
    ),
    (
        "data/results/independent_hs_all_configured_headroom/controls/"
        "ihs_all_single_p04_amax02_warm_from_p03_branch_control_manifest.json"
    ),
    (
        "data/results/independent_hs_all_configured_headroom/controls/"
        "ihs_all_single_p04_amax02_polish_from_p04_branch_control_manifest.json"
    ),
    "data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay.csv",
    "data/results/independent_hs_branch_control_replay/independent_hs_branch_control_replay_metadata.json",
    "data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress.csv",
    "data/results/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_metadata.json",
    (
        "data/results/independent_hs_horizons_solar_tidal_replay/"
        "independent_hs_horizons_solar_tidal_replay.csv"
    ),
    (
        "data/results/independent_hs_horizons_solar_tidal_replay/"
        "independent_hs_horizons_solar_tidal_replay_metadata.json"
    ),
    "data/results/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay.csv",
    "data/results/hard_catalog_tail_coast_branch_control_replay/tail_coast_branch_control_replay_metadata.json",
    "data/results/horizons_ephemeris_force_model_contrast/horizons_ephemeris_force_model_contrast.csv",
    (
        "data/results/horizons_ephemeris_force_model_contrast/"
        "horizons_ephemeris_force_model_contrast_metadata.json"
    ),
    "data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress.csv",
    "data/results/bicircular_solar_tidal_stress/bicircular_solar_tidal_stress_metadata.json",
    "data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery.csv",
    "data/results/bicircular_tail_coast_recovery/bicircular_tail_coast_recovery_metadata.json",
    "tables/replay_stress_validation/replay_stress_validation_table.tex",
    "tables/independent_hs_branch_control_replay/independent_hs_branch_control_replay_table.tex",
    "tables/independent_hs_bicircular_phase_stress/independent_hs_bicircular_phase_stress_table.tex",
    (
        "tables/independent_hs_horizons_solar_tidal_replay/"
        "independent_hs_horizons_solar_tidal_replay_table.tex"
    ),
    "tables/evidence_synthesis/evidence_synthesis_table.tex",
    "tables/claim_evidence_ledger/claim_evidence_ledger_table.tex",
]

FOCUSED_TEST_TARGETS = [
    "tests/test_replay_stress_validation.py",
    "tests/test_independent_hs_branch_control_replay.py",
    "tests/test_independent_hs_bicircular_phase_stress.py",
    "tests/test_independent_hs_horizons_solar_tidal_replay.py",
    "tests/test_independent_hs_direct_collocation.py",
    "tests/test_claim_evidence_ledger.py",
    "tests/test_evidence_synthesis.py",
    "tests/test_horizons_ephemeris_force_model_contrast.py",
    "tests/test_bicircular_solar_tidal_stress.py",
    "tests/test_bicircular_tail_coast_recovery.py",
    "tests/test_tail_coast_recovery.py",
]


def missing_primary_artifacts(root: Path) -> list[str]:
    return [relative for relative in PRIMARY_ARTIFACT_PATHS if not (root / relative).is_file()]


def manifest_check_command(root: Path) -> list[str]:
    return [sys.executable, str(root / "scripts" / "write_artifact_manifest.py"), "--check"]


def pytest_command(*, full_tests: bool) -> list[str]:
    targets = ["tests"] if full_tests else FOCUSED_TEST_TARGETS
    return [sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"]


def git_diff_check_command() -> list[str]:
    return ["git", "diff", "--check"]


def run_command(command: list[str], *, root: Path) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=root)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only checks for the current submission snapshot. This script does not regenerate "
            "artifacts, rebuild PDFs, or remove LaTeX auxiliary files."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to verify.")
    parser.add_argument("--full-tests", action="store_true", help="Run the full pytest suite instead of the focused subset.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest execution.")
    parser.add_argument("--skip-git-diff-check", action="store_true", help="Skip git diff --check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    failures: list[str] = []

    missing = missing_primary_artifacts(root)
    if missing:
        print("Missing required primary artifacts:", file=sys.stderr)
        for relative in missing:
            print(f"  {relative}", file=sys.stderr)
        failures.append("primary artifact path check")
    else:
        print(f"primary artifact path check passed: {len(PRIMARY_ARTIFACT_PATHS)} files")

    for label, command in [("manifest check", manifest_check_command(root))]:
        if run_command(command, root=root) != 0:
            failures.append(label)

    if not args.skip_tests:
        label = "full pytest suite" if args.full_tests else "focused pytest subset"
        if run_command(pytest_command(full_tests=bool(args.full_tests)), root=root) != 0:
            failures.append(label)

    if not args.skip_git_diff_check:
        if run_command(git_diff_check_command(), root=root) != 0:
            failures.append("git diff --check")

    if failures:
        print("submission snapshot verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("submission snapshot verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
