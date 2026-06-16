"""Replay persisted tail-coast controls with a deterministic solar-tidal model.

This postprocessor reads the focused hard-catalog tail-coast accepted-control
package and repropagates the nominal plus accepted branch full-control
schedules under both the configured normalized CR3BP model and a simple
bicircular solar third-body tidal perturbation. It does not rerun any
optimizer, branch selection, or high-fidelity ephemeris workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tail_coast_recovery as tail_runner
from qlt.bicircular import (
    DEFAULT_SUN_DISTANCE_LU,
    DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO,
    DEFAULT_SUN_MU_RATIO,
    SolarTidalParameters,
    propagate_controls_batch_bicircular,
)
from qlt.cr3bp import propagate_controls_batch
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "hard_catalog_tail_coast_branch_control_replay.yaml"
DEFAULT_RESULTS_DIR = Path("data/results/bicircular_solar_tidal_stress")
DEFAULT_TABLES_DIR = Path("tables/bicircular_solar_tidal_stress")

STRESS_CSV_NAME = "bicircular_solar_tidal_stress.csv"
STRESS_METADATA_NAME = "bicircular_solar_tidal_stress_metadata.json"
STRESS_TABLE_NAME = "bicircular_solar_tidal_stress_table.tex"

STRESS_COLUMNS = [
    "suite_case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "phase_degrees",
    "phase_radians",
    "controls_path",
    "controls_sha256",
    "source_cr3bp_terminal_error",
    "cr3bp_terminal_error",
    "cr3bp_delta_from_source",
    "solar_tidal_terminal_error",
    "solar_tidal_delta_from_cr3bp",
    "configured_threshold",
    "cr3bp_passes_configured_threshold",
    "solar_tidal_passes_configured_threshold",
    "threshold_semantics",
    "recovery_start",
    "recovery_segments",
    "optimizer_ran",
    "optimizer_success",
    "accepted_candidate",
    "accepted_weight_variant",
    "accepted_initialization",
    "control_count",
    "substeps_per_segment",
    "transfer_time",
    "solar_tidal_semantics",
]

_CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
        break
    except OverflowError:
        _CSV_FIELD_LIMIT //= 10


def _json_bytes(data: object) -> bytes:
    text = json.dumps(sanitize_json(data), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(data))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd(), ROOT):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return str(path)


def _resolve_existing_path(value: object) -> Path:
    text = str(value).strip()
    if not text:
        raise RuntimeError("expected artifact path, got blank")
    path = Path(text)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _read_json_verified(path_text: object, expected_sha256: object | None = None) -> tuple[dict, Path, str]:
    path = _resolve_existing_path(path_text)
    if not path.is_file():
        raise RuntimeError(f"sidecar not found: {path}")
    actual = _sha256(path)
    if expected_sha256 not in (None, "") and str(expected_sha256).strip() != actual:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return json.loads(path.read_text(encoding="utf-8")), path, actual


def _read_input_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"tail-coast recovery CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _case_by_id(config: dict) -> dict[str, dict]:
    return {str(case["suite_case_id"]): case for case in tail_runner._suite_cases(config)}


def _append_artifact(items: list[dict[str, object]], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    items.append({"path": _relative_or_absolute(path), "sha256": _sha256(path), "bytes": path.stat().st_size})


def _cr3bp_terminal_error(state0: np.ndarray, target: np.ndarray, cfg, controls: np.ndarray) -> float:
    controls = np.asarray(controls, dtype=float).reshape((int(cfg.n_segments), 3))
    final, _ = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)
    return float(state_error(final[0], target, cfg.position_scale, cfg.velocity_scale))


def _solar_tidal_terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    controls: np.ndarray,
    phase_rad: float,
    parameters: SolarTidalParameters,
) -> float:
    controls = np.asarray(controls, dtype=float).reshape((int(cfg.n_segments), 3))
    final, _ = propagate_controls_batch_bicircular(
        state0,
        controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        phase_rad=float(phase_rad),
        parameters=parameters,
    )
    return float(state_error(final[0], target, cfg.position_scale, cfg.velocity_scale))


def _record_rows_for_phases(
    *,
    suite_case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    controls_path: Path,
    controls_sha256: str,
    source_cr3bp_terminal_error: float,
    cr3bp_terminal_error: float,
    threshold: float,
    threshold_semantics: str,
    recovery_start: int | None,
    recovery_segments: int | None,
    optimizer_ran: bool | None,
    optimizer_success: bool | None,
    accepted_candidate: str,
    accepted_weight_variant: str,
    accepted_initialization: str,
    control_count: int,
    substeps_per_segment: int,
    transfer_time: float,
    solar_errors_by_phase: list[tuple[float, float, float]],
) -> list[dict[str, object]]:
    rows = []
    cr3bp_delta = abs(float(cr3bp_terminal_error) - float(source_cr3bp_terminal_error))
    for phase_degrees, phase_radians, solar_error in solar_errors_by_phase:
        rows.append(
            {
                "suite_case_id": suite_case_id,
                "record_type": record_type,
                "branch_order": "" if branch_order is None else int(branch_order),
                "mask_index": "" if mask_index is None else int(mask_index),
                "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
                "phase_degrees": float(phase_degrees),
                "phase_radians": float(phase_radians),
                "controls_path": _relative_or_absolute(controls_path),
                "controls_sha256": controls_sha256,
                "source_cr3bp_terminal_error": float(source_cr3bp_terminal_error),
                "cr3bp_terminal_error": float(cr3bp_terminal_error),
                "cr3bp_delta_from_source": float(cr3bp_delta),
                "solar_tidal_terminal_error": float(solar_error),
                "solar_tidal_delta_from_cr3bp": abs(float(solar_error) - float(cr3bp_terminal_error)),
                "configured_threshold": float(threshold),
                "cr3bp_passes_configured_threshold": bool(cr3bp_terminal_error <= float(threshold)),
                "solar_tidal_passes_configured_threshold": bool(solar_error <= float(threshold)),
                "threshold_semantics": threshold_semantics,
                "recovery_start": "" if recovery_start is None else int(recovery_start),
                "recovery_segments": "" if recovery_segments is None else int(recovery_segments),
                "optimizer_ran": "" if optimizer_ran is None else bool(optimizer_ran),
                "optimizer_success": "" if optimizer_success is None else bool(optimizer_success),
                "accepted_candidate": accepted_candidate,
                "accepted_weight_variant": accepted_weight_variant,
                "accepted_initialization": accepted_initialization,
                "control_count": int(control_count),
                "substeps_per_segment": int(substeps_per_segment),
                "transfer_time": float(transfer_time),
                "solar_tidal_semantics": (
                    "persisted accepted controls repropagated against the original CR3BP target "
                    "with a simple circular solar third-body tidal acceleration; no retuning, "
                    "no SPICE ephemeris, no high-fidelity validation"
                ),
            }
        )
    return rows


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / STRESS_TABLE_NAME
    if not summary_rows:
        path.write_text("% No bicircular solar-tidal stress rows.\n", encoding="utf-8")
        return path
    display = pd.DataFrame(summary_rows)[
        [
            "phase_degrees",
            "nominal_solar_tidal_terminal_error",
            "max_branch_solar_tidal_terminal_error",
            "max_branch_solar_tidal_delta_from_cr3bp",
            "branch_solar_tidal_pass_count",
            "branch_row_count",
            "all_branch_solar_tidal_pass",
        ]
    ].rename(
        columns={
            "phase_degrees": "Sun phase (deg)",
            "nominal_solar_tidal_terminal_error": "Nominal solar-tidal error",
            "max_branch_solar_tidal_terminal_error": "Max branch solar-tidal error",
            "max_branch_solar_tidal_delta_from_cr3bp": "Max branch delta",
            "branch_solar_tidal_pass_count": "Branch passes",
            "branch_row_count": "Branch rows",
            "all_branch_solar_tidal_pass": "All branch pass",
        }
    )
    display.to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


def _summary_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for phase, group in df.groupby("phase_degrees", sort=True):
        nominal = group[group["record_type"] == "nominal"]
        branch = group[group["record_type"] == "branch"]
        rows.append(
            {
                "phase_degrees": float(phase),
                "record_count": int(len(group)),
                "branch_row_count": int(len(branch)),
                "nominal_solar_tidal_terminal_error": float(nominal["solar_tidal_terminal_error"].max())
                if len(nominal)
                else None,
                "nominal_solar_tidal_delta_from_cr3bp": float(nominal["solar_tidal_delta_from_cr3bp"].max())
                if len(nominal)
                else None,
                "max_branch_solar_tidal_terminal_error": float(branch["solar_tidal_terminal_error"].max())
                if len(branch)
                else None,
                "max_branch_solar_tidal_delta_from_cr3bp": float(branch["solar_tidal_delta_from_cr3bp"].max())
                if len(branch)
                else None,
                "branch_solar_tidal_pass_count": int(branch["solar_tidal_passes_configured_threshold"].map(bool).sum())
                if len(branch)
                else 0,
                "all_branch_solar_tidal_pass": bool(branch["solar_tidal_passes_configured_threshold"].map(bool).all())
                if len(branch)
                else False,
                "max_cr3bp_delta_from_source": float(group["cr3bp_delta_from_source"].max()) if len(group) else 0.0,
            }
        )
    return rows


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.is_file() and not args.config.is_absolute():
        config_path = ROOT / args.config
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    if not source_states.is_file() and not args.source_states.is_absolute():
        source_states = ROOT / args.source_states
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_results_dir, _, _ = output_directories(Path.cwd(), config)
    tail_csv = source_results_dir / "tail_coast_recovery.csv"
    input_rows = _read_input_rows(tail_csv)
    cases = _case_by_id(config)

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    phase_degrees = [float(value) for value in args.phases_degrees]
    phases = [(phase, float(np.deg2rad(phase))) for phase in phase_degrees]
    parameters = SolarTidalParameters(
        sun_distance_lu=float(args.sun_distance_lu),
        sun_mu_ratio=float(args.sun_mu_ratio),
        sun_inertial_angular_rate_ratio=float(args.sun_inertial_angular_rate_ratio),
    )

    input_artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in (config_path, source_states, tail_csv):
        _append_artifact(input_artifacts, seen_artifacts, path)

    stress_rows: list[dict[str, object]] = []
    for row in input_rows:
        case_id = str(row.get("suite_case_id", ""))
        if case_id not in cases:
            continue
        if not _bool_value(row.get("branch_control_replay_ready", "")):
            continue
        case = cases[case_id]
        case_config = tail_runner._case_config(config, case)
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        target_state = np.asarray(states.target, dtype=float)
        thresholds = case_config["objective"]["thresholds"]
        nominal_threshold = float(thresholds["nominal_success"])
        robust_threshold = float(thresholds["robust_success"])

        manifest, manifest_path, manifest_sha = _read_json_verified(
            row["branch_control_manifest_path"],
            row.get("branch_control_manifest_sha256"),
        )
        nominal, nominal_path, nominal_sha = _read_json_verified(
            manifest["nominal_control_path"],
            manifest.get("nominal_control_sha256"),
        )
        _append_artifact(input_artifacts, seen_artifacts, manifest_path)
        _append_artifact(input_artifacts, seen_artifacts, nominal_path)

        recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
        if recorded_target.shape != target_state.shape:
            raise RuntimeError(f"manifest target_state shape mismatch for {case_id}")
        target_state_max_abs_delta = float(np.max(np.abs(target_state - recorded_target)))
        if target_state_max_abs_delta > float(args.baseline_tolerance):
            raise RuntimeError(
                f"target_state mismatch for {case_id}: max abs delta {target_state_max_abs_delta}"
            )

        nominal_controls = np.asarray(nominal["controls"], dtype=float)
        nominal_cr3bp = _cr3bp_terminal_error(states.initial, target_state, cfg, nominal_controls)
        nominal_source = float(nominal.get("nominal_error", row.get("nominal_error")))
        nominal_solar = [
            (
                phase,
                radians,
                _solar_tidal_terminal_error(
                    state0=states.initial,
                    target=target_state,
                    cfg=cfg,
                    controls=nominal_controls,
                    phase_rad=radians,
                    parameters=parameters,
                ),
            )
            for phase, radians in phases
        ]
        stress_rows.extend(
            _record_rows_for_phases(
                suite_case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                controls_path=nominal_path,
                controls_sha256=nominal_sha,
                source_cr3bp_terminal_error=nominal_source,
                cr3bp_terminal_error=nominal_cr3bp,
                threshold=nominal_threshold,
                threshold_semantics="nominal row uses configured objective.thresholds.nominal_success",
                recovery_start=None,
                recovery_segments=None,
                optimizer_ran=None,
                optimizer_success=None,
                accepted_candidate=str(nominal.get("nominal_accepted_candidate", "")),
                accepted_weight_variant="",
                accepted_initialization="",
                control_count=int(nominal_controls.shape[0]),
                substeps_per_segment=int(cfg.substeps),
                transfer_time=float(cfg.tf),
                solar_errors_by_phase=nominal_solar,
            )
        )

        branch_entries = list(manifest.get("branch_control_sidecars", []))
        expected_branch_count = int(row.get("branch_control_sidecar_count", len(branch_entries)))
        if len(branch_entries) != expected_branch_count:
            raise RuntimeError(
                f"manifest branch sidecar count {len(branch_entries)} does not match row {expected_branch_count}"
            )
        for entry in branch_entries:
            branch, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            _append_artifact(input_artifacts, seen_artifacts, branch_path)
            branch_controls = np.asarray(branch["branch_controls"], dtype=float)
            branch_cr3bp = _cr3bp_terminal_error(states.initial, target_state, cfg, branch_controls)
            branch_source = float(branch["terminal_error"])
            branch_solar = [
                (
                    phase,
                    radians,
                    _solar_tidal_terminal_error(
                        state0=states.initial,
                        target=target_state,
                        cfg=cfg,
                        controls=branch_controls,
                        phase_rad=radians,
                        parameters=parameters,
                    ),
                )
                for phase, radians in phases
            ]
            stress_rows.extend(
                _record_rows_for_phases(
                    suite_case_id=case_id,
                    record_type="branch",
                    branch_order=int(branch["branch_order"]),
                    mask_index=int(branch["mask_index"]),
                    outage_mask=[int(value) for value in branch["outage_mask"]],
                    controls_path=branch_path,
                    controls_sha256=branch_sha,
                    source_cr3bp_terminal_error=branch_source,
                    cr3bp_terminal_error=branch_cr3bp,
                    threshold=robust_threshold,
                    threshold_semantics="branch rows use configured objective.thresholds.robust_success",
                    recovery_start=int(branch["recovery_start"]),
                    recovery_segments=int(branch["recovery_segments"]),
                    optimizer_ran=bool(branch.get("optimizer_ran", False)),
                    optimizer_success=bool(branch.get("optimizer_success", False)),
                    accepted_candidate=str(branch.get("accepted_candidate", "")),
                    accepted_weight_variant=str(branch.get("accepted_branch_weight_variant_label", "")),
                    accepted_initialization=str(branch.get("accepted_branch_initialization_label", "")),
                    control_count=int(branch_controls.shape[0]),
                    substeps_per_segment=int(cfg.substeps),
                    transfer_time=float(cfg.tf),
                    solar_errors_by_phase=branch_solar,
                )
            )

    if not stress_rows:
        raise RuntimeError("no branch-control replay-ready rows found for bicircular solar-tidal stress probe")

    df = pd.DataFrame(stress_rows, columns=STRESS_COLUMNS)
    csv_path = results_dir / STRESS_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")
    summary_rows = _summary_rows(df)
    table_path = _write_table(summary_rows, tables_dir)

    branch_rows = df[df["record_type"] == "branch"]
    max_cr3bp_delta = float(df["cr3bp_delta_from_source"].max()) if len(df) else 0.0
    max_solar_error = float(df["solar_tidal_terminal_error"].max()) if len(df) else 0.0
    max_solar_delta = float(df["solar_tidal_delta_from_cr3bp"].max()) if len(df) else 0.0
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "source_record_count": int(len(df) // max(1, len(phases))),
        "phase_degrees": phase_degrees,
        "phase_count": int(len(phases)),
        "nominal_source_record_count": int((df["record_type"] == "nominal").sum() // max(1, len(phases))),
        "branch_source_record_count": int(len(branch_rows) // max(1, len(phases))),
        "optimization_rerun": False,
        "uses_recorded_artifacts_only": True,
        "branch_control_replay": True,
        "bicircular_solar_tidal_stress_probe": True,
        "high_fidelity_validation": False,
        "spice_ephemeris_validation": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "solar_tidal_parameters": parameters.as_dict(),
        "baseline_tolerance": float(args.baseline_tolerance),
        "baseline_reproduction": {
            "max_cr3bp_delta_from_source": max_cr3bp_delta,
            "passes_baseline_tolerance": bool(max_cr3bp_delta <= float(args.baseline_tolerance)),
        },
        "solar_tidal_summary": {
            "max_solar_tidal_terminal_error": max_solar_error,
            "max_solar_tidal_delta_from_cr3bp": max_solar_delta,
            "solar_tidal_pass_count": int(df["solar_tidal_passes_configured_threshold"].map(bool).sum()),
            "solar_tidal_row_count": int(len(df)),
            "branch_solar_tidal_pass_count": int(branch_rows["solar_tidal_passes_configured_threshold"].map(bool).sum()),
            "branch_solar_tidal_row_count": int(len(branch_rows)),
            "all_solar_tidal_rows_pass": bool(df["solar_tidal_passes_configured_threshold"].map(bool).all()),
            "all_branch_solar_tidal_rows_pass": bool(branch_rows["solar_tidal_passes_configured_threshold"].map(bool).all()),
        },
        "summary": summary_rows,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "bicircular_solar_tidal_stress_csv": _relative_or_absolute(csv_path),
            "bicircular_solar_tidal_stress_metadata_json": _relative_or_absolute(results_dir / STRESS_METADATA_NAME),
            "bicircular_solar_tidal_stress_table_tex": _relative_or_absolute(table_path),
        },
        "interpretation_limits": [
            "This is a deterministic beyond-CR3BP solar-tidal stress replay of persisted accepted controls.",
            "It is not SPICE ephemeris validation, high-fidelity flight validation, production solver parity, fuel optimality, or quantum evidence.",
            "Controls are not retuned under the solar-tidal perturbation; negative threshold outcomes are external-validity limitations.",
            "Terminal errors are measured against the original CR3BP target at the original fixed final time.",
        ],
    }
    _write_json(results_dir / STRESS_METADATA_NAME, metadata)
    print(
        "Completed bicircular solar-tidal stress probe "
        f"with {int(len(df))} rows, max_solar_error={max_solar_error:.3e}, "
        f"all_branch_solar_tidal_rows_pass={metadata['solar_tidal_summary']['all_branch_solar_tidal_rows_pass']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay persisted hard-catalog tail-coast controls under CR3BP and a solar-tidal stress model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--phases-degrees", type=float, nargs="+", default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--sun-distance-lu", type=float, default=DEFAULT_SUN_DISTANCE_LU)
    parser.add_argument("--sun-mu-ratio", type=float, default=DEFAULT_SUN_MU_RATIO)
    parser.add_argument(
        "--sun-inertial-angular-rate-ratio",
        type=float,
        default=DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
