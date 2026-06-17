"""Phase-sweep bicircular stress replay for persisted independent-HS controls.

This postprocessor reads replay-ready independent-midpoint Hermite-Simpson
nominal and branch sidecars, validates their SHA-256 hashes, and repropagates
the endpoint-plus-midpoint controls under normalized CR3BP and a simple
circular solar-tidal bicircular stress model. It does not rerun optimization,
use SPICE ephemerides, perform high-fidelity validation, or establish
production-solver parity.
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

import run_independent_hs_continuation as ihs_runner
from qlt.bicircular import (
    DEFAULT_SUN_DISTANCE_LU,
    DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO,
    DEFAULT_SUN_MU_RATIO,
    SolarTidalParameters,
    propagate_piecewise_controls_bicircular,
)
from qlt.direct_collocation import propagate_piecewise_controls
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "independent_hs_all_configured_headroom.yaml"
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_bicircular_phase_stress")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_bicircular_phase_stress")
STRESS_CSV_NAME = "independent_hs_bicircular_phase_stress.csv"
STRESS_METADATA_NAME = "independent_hs_bicircular_phase_stress_metadata.json"
STRESS_TABLE_NAME = "independent_hs_bicircular_phase_stress_table.tex"
DEFAULT_PHASES_DEGREES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

STRESS_COLUMNS = [
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "phase_degrees",
    "phase_radians",
    "controls_path",
    "controls_sha256",
    "midpoint_controls_replayed",
    "recorded_cr3bp_terminal_error",
    "cr3bp_terminal_error",
    "cr3bp_delta_from_recorded",
    "bicircular_terminal_error",
    "bicircular_delta_from_cr3bp",
    "configured_threshold",
    "cr3bp_passes_configured_threshold",
    "bicircular_passes_configured_threshold",
    "threshold_semantics",
    "recovery_start",
    "recovery_segments",
    "optimizer_success",
    "nfev",
    "phase_time",
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
    text = json.dumps(
        sanitize_json(data),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
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
    return resolved.as_posix()


def _resolve_existing_path(value: object) -> Path:
    text = str(value or "").strip()
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
        raise RuntimeError(f"independent-HS continuation CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _case_by_id(config: dict) -> dict[str, dict]:
    return {str(case["case_id"]): case for case in ihs_runner._suite_cases(config)}


def _case_config(config: dict, case: dict) -> dict:
    ihs_runner._configure_base(config)
    return ihs_runner._base._case_config(config, case)


def _append_artifact(items: list[dict[str, object]], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    items.append({"path": _relative_or_absolute(path), "sha256": _sha256(path), "bytes": path.stat().st_size})


def _cr3bp_terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
) -> float:
    final, _ = propagate_piecewise_controls(
        state0,
        endpoint_controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=midpoint_controls,
    )
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def _bicircular_terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
    phase_rad: float,
    parameters: SolarTidalParameters,
) -> float:
    final, _ = propagate_piecewise_controls_bicircular(
        state0,
        endpoint_controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=midpoint_controls,
        phase_rad=float(phase_rad),
        parameters=parameters,
    )
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def _rows_for_phases(
    *,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    controls_path: Path,
    controls_sha256: str,
    midpoint_controls_replayed: bool,
    recorded_cr3bp_terminal_error: float,
    cr3bp_terminal_error: float,
    threshold: float,
    threshold_semantics: str,
    recovery_start: int | None,
    recovery_segments: int | None,
    optimizer_success: bool,
    nfev: int,
    phase_time: float,
    control_count: int,
    substeps_per_segment: int,
    transfer_time: float,
    bicircular_errors_by_phase: list[tuple[float, float, float]],
) -> list[dict[str, object]]:
    rows = []
    cr3bp_delta = abs(float(cr3bp_terminal_error) - float(recorded_cr3bp_terminal_error))
    for phase_degrees, phase_radians, bicircular_error in bicircular_errors_by_phase:
        rows.append(
            {
                "case_id": case_id,
                "record_type": record_type,
                "branch_order": "" if branch_order is None else int(branch_order),
                "mask_index": "" if mask_index is None else int(mask_index),
                "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
                "phase_degrees": float(phase_degrees),
                "phase_radians": float(phase_radians),
                "controls_path": _relative_or_absolute(controls_path),
                "controls_sha256": controls_sha256,
                "midpoint_controls_replayed": bool(midpoint_controls_replayed),
                "recorded_cr3bp_terminal_error": float(recorded_cr3bp_terminal_error),
                "cr3bp_terminal_error": float(cr3bp_terminal_error),
                "cr3bp_delta_from_recorded": float(cr3bp_delta),
                "bicircular_terminal_error": float(bicircular_error),
                "bicircular_delta_from_cr3bp": abs(float(bicircular_error) - float(cr3bp_terminal_error)),
                "configured_threshold": float(threshold),
                "cr3bp_passes_configured_threshold": bool(cr3bp_terminal_error <= float(threshold)),
                "bicircular_passes_configured_threshold": bool(bicircular_error <= float(threshold)),
                "threshold_semantics": threshold_semantics,
                "recovery_start": "" if recovery_start is None else int(recovery_start),
                "recovery_segments": "" if recovery_segments is None else int(recovery_segments),
                "optimizer_success": bool(optimizer_success),
                "nfev": int(nfev),
                "phase_time": float(phase_time),
                "control_count": int(control_count),
                "substeps_per_segment": int(substeps_per_segment),
                "transfer_time": float(transfer_time),
                "solar_tidal_semantics": (
                    "persisted independent-HS endpoint-plus-midpoint controls repropagated "
                    "against the original CR3BP target with a simple circular solar third-body "
                    "tidal acceleration; no retuning, no SPICE ephemeris, no high-fidelity "
                    "validation, no production solver parity"
                ),
            }
        )
    return rows


def _summary_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for case_id, group in df.groupby("case_id", sort=True):
        nominal = group[group["record_type"] == "nominal"]
        branch = group[group["record_type"] == "branch"]
        branch_phase_count = int(len(branch))
        rows.append(
            {
                "case_id": str(case_id),
                "phase_count": int(group["phase_degrees"].nunique()),
                "nominal_phase_row_count": int(len(nominal)),
                "branch_phase_row_count": branch_phase_count,
                "nominal_bicircular_terminal_error_max": float(nominal["bicircular_terminal_error"].max())
                if len(nominal)
                else None,
                "branch_bicircular_terminal_error_max": float(branch["bicircular_terminal_error"].max())
                if len(branch)
                else None,
                "overall_bicircular_terminal_error_max": float(group["bicircular_terminal_error"].max())
                if len(group)
                else None,
                "nominal_bicircular_pass_count": int(nominal["bicircular_passes_configured_threshold"].map(bool).sum())
                if len(nominal)
                else 0,
                "branch_bicircular_pass_count": int(branch["bicircular_passes_configured_threshold"].map(bool).sum())
                if len(branch)
                else 0,
                "all_nominal_bicircular_pass": bool(nominal["bicircular_passes_configured_threshold"].map(bool).all())
                if len(nominal)
                else False,
                "all_branch_bicircular_pass": bool(branch["bicircular_passes_configured_threshold"].map(bool).all())
                if len(branch)
                else False,
                "all_bicircular_rows_pass": bool(group["bicircular_passes_configured_threshold"].map(bool).all())
                if len(group)
                else False,
                "max_cr3bp_delta_from_recorded": float(group["cr3bp_delta_from_recorded"].max()) if len(group) else 0.0,
                "nominal_threshold": float(nominal["configured_threshold"].max()) if len(nominal) else None,
                "branch_threshold": float(branch["configured_threshold"].max()) if len(branch) else None,
                "optimizer_success": bool(group["optimizer_success"].map(bool).all()) if len(group) else False,
            }
        )
    return rows


def _branch_maxima(df: pd.DataFrame, case_id: str) -> list[dict[str, object]]:
    branch = df[(df["case_id"] == case_id) & (df["record_type"] == "branch")]
    rows = []
    for mask_index, group in branch.groupby("mask_index", sort=True):
        rows.append(
            {
                "mask_index": int(mask_index),
                "max_bicircular_terminal_error": float(group["bicircular_terminal_error"].max()),
                "pass_count": int(group["bicircular_passes_configured_threshold"].map(bool).sum()),
                "phase_count": int(group["phase_degrees"].nunique()),
            }
        )
    return rows


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / STRESS_TABLE_NAME
    if not summary_rows:
        path.write_text("% No independent-HS bicircular phase-stress rows.\n", encoding="utf-8")
        return path
    display = pd.DataFrame(summary_rows)[
        [
            "case_id",
            "phase_count",
            "nominal_bicircular_terminal_error_max",
            "branch_bicircular_terminal_error_max",
            "nominal_bicircular_pass_count",
            "nominal_phase_row_count",
            "branch_bicircular_pass_count",
            "branch_phase_row_count",
            "all_bicircular_rows_pass",
        ]
    ].rename(
        columns={
            "case_id": "Case",
            "phase_count": "Phases",
            "nominal_bicircular_terminal_error_max": "Max nominal",
            "branch_bicircular_terminal_error_max": "Max branch",
            "nominal_bicircular_pass_count": "Nominal pass",
            "nominal_phase_row_count": "Nominal rows",
            "branch_bicircular_pass_count": "Branch pass",
            "branch_phase_row_count": "Branch rows",
            "all_bicircular_rows_pass": "All pass",
        }
    )
    display.to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


def _selected_rows(input_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    phase_times = {float(value) for value in (args.phase_time or [0.4])}
    case_ids = set(args.case_id or [])
    selected = []
    for row in input_rows:
        if case_ids and str(row.get("case_id", "")) not in case_ids:
            continue
        if not _bool_value(row.get("branch_control_replay_ready", "")):
            continue
        if phase_times and float(row.get("phase_time", "nan")) not in phase_times:
            continue
        selected.append(row)
    return selected


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.is_file() and not args.config.is_absolute():
        config_path = ROOT / args.config
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    if not source_states.is_file() and not args.source_states.is_absolute():
        source_states = ROOT / args.source_states
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_results_dir, _, _ = output_directories(Path.cwd(), config)
    input_csv = source_results_dir / "independent_hs_all_configured_headroom.csv"
    if not input_csv.is_file():
        artifact_stem = str((config.get("run", {}) or {}).get("artifact_stem", "independent_hs_all_configured_headroom"))
        input_csv = source_results_dir / f"{artifact_stem}.csv"
    input_rows = _selected_rows(_read_input_rows(input_csv), args)
    if not input_rows:
        raise RuntimeError("no replay-ready independent-HS p=0.4 rows found for bicircular phase stress")

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
    for path in (config_path, source_states, input_csv):
        _append_artifact(input_artifacts, seen_artifacts, path)

    stress_rows: list[dict[str, object]] = []
    for row in input_rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in cases:
            continue
        case = cases[case_id]
        case_config = _case_config(config, case)
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        target_state = np.asarray(states.target, dtype=float)
        thresholds = case_config["objective"]["thresholds"]
        nominal_threshold = float(thresholds["nominal_success"])
        robust_threshold = float(thresholds["robust_success"])

        manifest, manifest_path, manifest_sha = _read_json_verified(
            row["branch_control_manifest_path"],
            row.get("branch_control_manifest_hash"),
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

        nominal_endpoint = np.asarray(nominal.get("nominal_endpoint_controls", nominal.get("controls")), dtype=float)
        nominal_midpoint_raw = nominal.get("nominal_midpoint_controls")
        nominal_midpoint = None if nominal_midpoint_raw is None else np.asarray(nominal_midpoint_raw, dtype=float)
        nominal_cr3bp = _cr3bp_terminal_error(
            state0=states.initial,
            target=target_state,
            cfg=cfg,
            endpoint_controls=nominal_endpoint,
            midpoint_controls=nominal_midpoint,
        )
        nominal_recorded = float(manifest.get("nominal_error", row.get("nominal_error")))
        nominal_bicircular = [
            (
                phase,
                radians,
                _bicircular_terminal_error(
                    state0=states.initial,
                    target=target_state,
                    cfg=cfg,
                    endpoint_controls=nominal_endpoint,
                    midpoint_controls=nominal_midpoint,
                    phase_rad=radians,
                    parameters=parameters,
                ),
            )
            for phase, radians in phases
        ]
        stress_rows.extend(
            _rows_for_phases(
                case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                controls_path=nominal_path,
                controls_sha256=nominal_sha,
                midpoint_controls_replayed=nominal_midpoint is not None,
                recorded_cr3bp_terminal_error=nominal_recorded,
                cr3bp_terminal_error=nominal_cr3bp,
                threshold=nominal_threshold,
                threshold_semantics="nominal rows use configured objective.thresholds.nominal_success",
                recovery_start=None,
                recovery_segments=None,
                optimizer_success=_bool_value(row.get("optimizer_success", "")),
                nfev=int(float(row.get("nfev", 0) or 0)),
                phase_time=float(row.get("phase_time", case_config["benchmark"].get("phase_time", 0.0))),
                control_count=int(nominal_endpoint.shape[0]),
                substeps_per_segment=int(cfg.substeps),
                transfer_time=float(cfg.tf),
                bicircular_errors_by_phase=nominal_bicircular,
            )
        )

        branch_entries = list(manifest.get("branch_control_sidecars", []))
        expected_branch_count = int(manifest.get("expected_branch_count", row.get("branch_control_sidecar_count", 0)))
        if len(branch_entries) != expected_branch_count:
            raise RuntimeError(
                f"manifest branch sidecar count {len(branch_entries)} does not match expected {expected_branch_count}"
            )
        for entry in branch_entries:
            branch, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            _append_artifact(input_artifacts, seen_artifacts, branch_path)
            endpoint_controls = np.asarray(branch.get("branch_endpoint_controls", branch.get("branch_controls")), dtype=float)
            midpoint_raw = branch.get("branch_midpoint_controls")
            midpoint_controls = None if midpoint_raw is None else np.asarray(midpoint_raw, dtype=float)
            branch_cr3bp = _cr3bp_terminal_error(
                state0=states.initial,
                target=target_state,
                cfg=cfg,
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
            )
            branch_recorded = float(branch["recorded_branch_terminal_error"])
            branch_bicircular = [
                (
                    phase,
                    radians,
                    _bicircular_terminal_error(
                        state0=states.initial,
                        target=target_state,
                        cfg=cfg,
                        endpoint_controls=endpoint_controls,
                        midpoint_controls=midpoint_controls,
                        phase_rad=radians,
                        parameters=parameters,
                    ),
                )
                for phase, radians in phases
            ]
            stress_rows.extend(
                _rows_for_phases(
                    case_id=case_id,
                    record_type="branch",
                    branch_order=int(branch["branch_order"]),
                    mask_index=int(branch["mask_index"]),
                    outage_mask=[int(value) for value in branch["outage_mask"]],
                    controls_path=branch_path,
                    controls_sha256=branch_sha,
                    midpoint_controls_replayed=midpoint_controls is not None,
                    recorded_cr3bp_terminal_error=branch_recorded,
                    cr3bp_terminal_error=branch_cr3bp,
                    threshold=robust_threshold,
                    threshold_semantics="branch rows use configured objective.thresholds.robust_success",
                    recovery_start=int(branch["recovery_start"]),
                    recovery_segments=int(branch["recovery_segments"]),
                    optimizer_success=_bool_value(row.get("optimizer_success", "")),
                    nfev=int(float(row.get("nfev", 0) or 0)),
                    phase_time=float(row.get("phase_time", case_config["benchmark"].get("phase_time", 0.0))),
                    control_count=int(endpoint_controls.shape[0]),
                    substeps_per_segment=int(cfg.substeps),
                    transfer_time=float(cfg.tf),
                    bicircular_errors_by_phase=branch_bicircular,
                )
            )

    if not stress_rows:
        raise RuntimeError("no independent-HS bicircular phase-stress rows were produced")

    df = pd.DataFrame(stress_rows, columns=STRESS_COLUMNS)
    csv_path = results_dir / STRESS_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")
    summary_rows = _summary_rows(df)
    table_path = _write_table(summary_rows, tables_dir)

    branch_rows = df[df["record_type"] == "branch"]
    nominal_rows = df[df["record_type"] == "nominal"]
    max_cr3bp_delta = float(df["cr3bp_delta_from_recorded"].max()) if len(df) else 0.0
    max_bicircular = float(df["bicircular_terminal_error"].max()) if len(df) else 0.0
    max_branch_bicircular = float(branch_rows["bicircular_terminal_error"].max()) if len(branch_rows) else 0.0
    max_nominal_bicircular = float(nominal_rows["bicircular_terminal_error"].max()) if len(nominal_rows) else 0.0
    polish_case_id = "ihs_all_single_p04_amax02_polish_from_p04"
    polish_rows = df[df["case_id"] == polish_case_id]
    polish_branch_rows = polish_rows[polish_rows["record_type"] == "branch"]
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "case_count": int(df["case_id"].nunique()),
        "source_record_count": int(len(df) // max(1, len(phases))),
        "phase_degrees": phase_degrees,
        "phase_count": int(len(phases)),
        "nominal_phase_row_count": int(len(nominal_rows)),
        "branch_phase_row_count": int(len(branch_rows)),
        "nominal_source_record_count": int(len(nominal_rows) // max(1, len(phases))),
        "branch_source_record_count": int(len(branch_rows) // max(1, len(phases))),
        "optimization_rerun": False,
        "uses_recorded_artifacts_only": True,
        "branch_control_replay": True,
        "independent_hs_bicircular_phase_stress_probe": True,
        "bicircular_solar_tidal_stress_probe": True,
        "high_fidelity_validation": False,
        "spice_ephemeris_validation": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "solar_tidal_parameters": parameters.as_dict(),
        "baseline_tolerance": float(args.baseline_tolerance),
        "baseline_reproduction": {
            "max_cr3bp_delta_from_recorded": max_cr3bp_delta,
            "passes_baseline_tolerance": bool(max_cr3bp_delta <= float(args.baseline_tolerance)),
        },
        "phase_stress_summary": {
            "max_nominal_bicircular_terminal_error": max_nominal_bicircular,
            "max_branch_bicircular_terminal_error": max_branch_bicircular,
            "max_bicircular_terminal_error": max_bicircular,
            "nominal_bicircular_pass_count": int(nominal_rows["bicircular_passes_configured_threshold"].map(bool).sum()),
            "nominal_bicircular_row_count": int(len(nominal_rows)),
            "branch_bicircular_pass_count": int(branch_rows["bicircular_passes_configured_threshold"].map(bool).sum()),
            "branch_bicircular_row_count": int(len(branch_rows)),
            "all_nominal_bicircular_rows_pass": bool(nominal_rows["bicircular_passes_configured_threshold"].map(bool).all()),
            "all_branch_bicircular_rows_pass": bool(branch_rows["bicircular_passes_configured_threshold"].map(bool).all()),
            "all_bicircular_rows_pass": bool(df["bicircular_passes_configured_threshold"].map(bool).all()),
        },
        "polish_case_summary": {
            "case_id": polish_case_id,
            "row_count": int(len(polish_rows)),
            "nominal_phase_count": int((polish_rows["record_type"] == "nominal").sum()),
            "branch_phase_count": int(len(polish_branch_rows)),
            "max_nominal_bicircular_terminal_error": float(
                polish_rows[polish_rows["record_type"] == "nominal"]["bicircular_terminal_error"].max()
            )
            if len(polish_rows)
            else None,
            "max_branch_bicircular_terminal_error": float(polish_branch_rows["bicircular_terminal_error"].max())
            if len(polish_branch_rows)
            else None,
            "nominal_bicircular_pass_count": int(
                polish_rows[polish_rows["record_type"] == "nominal"]["bicircular_passes_configured_threshold"]
                .map(bool)
                .sum()
            )
            if len(polish_rows)
            else 0,
            "branch_bicircular_pass_count": int(
                polish_branch_rows["bicircular_passes_configured_threshold"].map(bool).sum()
            )
            if len(polish_branch_rows)
            else 0,
            "branch_maxima_by_mask": _branch_maxima(df, polish_case_id),
        },
        "summary": summary_rows,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_bicircular_phase_stress_csv": _relative_or_absolute(csv_path),
            "independent_hs_bicircular_phase_stress_metadata_json": _relative_or_absolute(
                results_dir / STRESS_METADATA_NAME
            ),
            "independent_hs_bicircular_phase_stress_table_tex": _relative_or_absolute(table_path),
        },
        "interpretation_limits": [
            "This is a deterministic beyond-CR3BP stress replay of persisted independent-HS controls.",
            "The bicircular model is a simple circular solar third-body tidal perturbation in canonical units.",
            "It is not SPICE ephemeris validation, high-fidelity flight validation, production solver parity, fuel optimality, or quantum evidence.",
            "Controls are not retuned under the solar-tidal perturbation; terminal errors are measured against the original CR3BP target at the original fixed final time.",
        ],
    }
    _write_json(results_dir / STRESS_METADATA_NAME, metadata)
    print(
        "Completed independent-HS bicircular phase stress "
        f"with {int(len(df))} rows, max_nominal={max_nominal_bicircular:.3e}, "
        f"max_branch={max_branch_bicircular:.3e}, "
        f"all_branch_pass={metadata['phase_stress_summary']['all_branch_bicircular_rows_pass']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay persisted independent-HS endpoint-plus-midpoint controls under normalized CR3BP "
            "and a simple bicircular solar-tidal phase sweep."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--phase-time", type=float, action="append", default=None)
    parser.add_argument("--phases-degrees", type=float, nargs="+", default=DEFAULT_PHASES_DEGREES)
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
