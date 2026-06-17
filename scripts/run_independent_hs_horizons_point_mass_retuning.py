"""Retune independent-HS controls under a cached-Horizons point-mass model.

This postprocessor reads the committed independent-HS polish row and the
committed JPL Horizons Moon/Sun vector cache, then evaluates and independently
retunes the persisted nominal and branch endpoint-plus-midpoint controls under
a geocentric Earth/Moon/Sun point-mass stress model. It is intentionally scoped
as a conservative retuning evidence package: it does not use SPICE, perform
full high-fidelity or flight validation, claim production-solver parity, claim
fuel optimality, assign a DOI, or support quantum advantage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_independent_hs_continuation as ihs_runner
from qlt.ephemeris_contrast import (
    DEFAULT_CACHE_START_JD_TDB,
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    canonical_point_mass_scales,
    horizons_point_mass_profile_from_cache,
    horizons_point_mass_terminal_error,
    load_horizons_cache,
    propagate_piecewise_controls_horizons_point_mass,
    validate_horizons_cache_compatibility,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.refinement import project_controls_to_ball
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "independent_hs_all_configured_headroom.yaml"
DEFAULT_CACHE = ROOT / "data" / "cache" / "horizons" / "independent_hs_phase_shift_2026jan01_vectors.json"
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_horizons_point_mass_retuning")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_horizons_point_mass_retuning")
EXPECTED_CACHE_SHA256 = "13fe699371ad67bf1616d38b7afd316bbff72811bbc0f8337cff51d6333897b2"

POLISH_CASE_ID = "ihs_all_single_p04_amax02_polish_from_p04"
RETUNING_CSV_NAME = "independent_hs_horizons_point_mass_retuning.csv"
RETUNING_METADATA_NAME = "independent_hs_horizons_point_mass_retuning_metadata.json"
RETUNING_TABLE_NAME = "independent_hs_horizons_point_mass_retuning_table.tex"
RETUNED_CONTROL_MANIFEST_NAME = "independent_hs_horizons_point_mass_retuned_control_manifest.json"

RETUNING_COLUMNS = [
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "source_controls_path",
    "source_controls_sha256",
    "retuned_controls_path",
    "retuned_controls_sha256",
    "midpoint_controls_retuned",
    "active_control_mask",
    "recorded_cr3bp_terminal_error",
    "point_mass_replay_terminal_error",
    "point_mass_retuned_terminal_error",
    "point_mass_replay_delta_from_recorded_cr3bp",
    "configured_threshold",
    "point_mass_replay_passes_configured_threshold",
    "point_mass_retuned_passes_configured_threshold",
    "retune_success",
    "scipy_success",
    "scipy_message",
    "nfev",
    "cost",
    "optimality",
    "runtime_seconds",
    "control_bound",
    "retuned_endpoint_norm_max",
    "retuned_midpoint_norm_max",
    "retuned_endpoint_bound_violation",
    "retuned_midpoint_bound_violation",
    "substeps_per_segment",
    "transfer_time",
    "start_jd_tdb",
    "cache_path",
    "cache_sha256",
    "retuning_semantics",
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


def _resolve_cache_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def _cache_epoch_label(cache: dict[str, object], *, start_jd_tdb: float) -> str:
    metadata = cache.get("metadata", {})
    if isinstance(metadata, dict):
        label = str(metadata.get("start_calendar_tdb", "")).strip()
        if label:
            return label
    return f"JD {float(start_jd_tdb):.9f} TDB"


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


def _selected_rows(input_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    phase_times = {float(value) for value in (args.phase_time or [0.4])}
    case_ids = set(args.case_id or [POLISH_CASE_ID])
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


def _control_norm_diagnostics(
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray,
    *,
    amax: float,
) -> dict[str, float]:
    endpoint_norms = np.linalg.norm(np.asarray(endpoint_controls, dtype=float), axis=1)
    midpoint_norms = np.linalg.norm(np.asarray(midpoint_controls, dtype=float), axis=1)
    return {
        "endpoint_norm_max": float(endpoint_norms.max()) if endpoint_norms.size else 0.0,
        "midpoint_norm_max": float(midpoint_norms.max()) if midpoint_norms.size else 0.0,
        "endpoint_bound_violation": float(max(0.0, endpoint_norms.max() - float(amax))) if endpoint_norms.size else 0.0,
        "midpoint_bound_violation": float(max(0.0, midpoint_norms.max() - float(amax))) if midpoint_norms.size else 0.0,
    }


def _scaled_residual(final_state: np.ndarray, target: np.ndarray, *, position_scale: float, velocity_scale: float) -> np.ndarray:
    scale = np.asarray(
        [position_scale, position_scale, position_scale, velocity_scale, velocity_scale, velocity_scale],
        dtype=float,
    )
    return (np.asarray(final_state, dtype=float) - np.asarray(target, dtype=float)) / np.maximum(scale, 1.0e-12)


def _retune_endpoint_midpoint_controls(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    profile,
    endpoint_seed: np.ndarray,
    midpoint_seed: np.ndarray,
    active_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    amax = float(cfg.amax)
    endpoint_seed = project_controls_to_ball(np.asarray(endpoint_seed, dtype=float), amax)
    midpoint_seed = project_controls_to_ball(np.asarray(midpoint_seed, dtype=float), amax)
    active = np.asarray(active_mask, dtype=float) > 0.5
    if active.shape != (endpoint_seed.shape[0],):
        raise ValueError(f"active_mask shape {active.shape} does not match controls {endpoint_seed.shape}")
    if not np.any(active):
        raise ValueError("retuning requires at least one active control segment")
    endpoint_seed = endpoint_seed * active[:, None]
    midpoint_seed = midpoint_seed * active[:, None]
    active_count = int(np.count_nonzero(active))
    endpoint_var_count = active_count * 3
    x0 = np.concatenate((endpoint_seed[active].reshape(-1), midpoint_seed[active].reshape(-1)))

    def unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        endpoint = np.zeros_like(endpoint_seed)
        midpoint = np.zeros_like(midpoint_seed)
        endpoint[active] = vec[:endpoint_var_count].reshape((-1, 3))
        midpoint[active] = vec[endpoint_var_count:].reshape((-1, 3))
        endpoint = project_controls_to_ball(endpoint, amax) * active[:, None]
        midpoint = project_controls_to_ball(midpoint, amax) * active[:, None]
        return endpoint, midpoint

    def residual(vec: np.ndarray) -> np.ndarray:
        endpoint, midpoint = unpack(vec)
        final, _ = propagate_piecewise_controls_horizons_point_mass(
            state0,
            endpoint,
            cfg.mu,
            cfg.tf,
            cfg.substeps,
            profile=profile,
            midpoint_controls=midpoint,
            reference_distance_km=float(args.reference_distance_km),
            canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
        )
        parts = [
            float(args.terminal_residual_weight)
            * _scaled_residual(
                final,
                target,
                position_scale=float(cfg.position_scale),
                velocity_scale=float(cfg.velocity_scale),
            )
        ]
        control_weight = float(args.control_regularization_weight)
        smooth_weight = float(args.smooth_regularization_weight)
        if control_weight:
            parts.append(control_weight * endpoint[active].reshape(-1) / max(amax, 1.0e-12))
            parts.append(control_weight * midpoint[active].reshape(-1) / max(amax, 1.0e-12))
        if smooth_weight and active_count > 1:
            parts.append(smooth_weight * np.diff(endpoint[active], axis=0).reshape(-1) / max(amax, 1.0e-12))
            parts.append(smooth_weight * np.diff(midpoint[active], axis=0).reshape(-1) / max(amax, 1.0e-12))
        return np.concatenate(parts)

    start = time.perf_counter()
    result = least_squares(
        residual,
        x0,
        bounds=(-amax, amax),
        max_nfev=int(args.max_nfev),
        xtol=float(args.xtol),
        ftol=float(args.ftol),
        gtol=float(args.gtol),
        verbose=0,
    )
    runtime = float(time.perf_counter() - start)
    endpoint, midpoint = unpack(result.x)
    final, _ = propagate_piecewise_controls_horizons_point_mass(
        state0,
        endpoint,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        profile=profile,
        midpoint_controls=midpoint,
        reference_distance_km=float(args.reference_distance_km),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
    )
    terminal_error = float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))
    return {
        "endpoint_controls": endpoint,
        "midpoint_controls": midpoint,
        "final_state": final,
        "terminal_error": terminal_error,
        "scipy_success": bool(result.success),
        "scipy_message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "runtime_seconds": runtime,
    }


def _write_retuned_sidecar(
    *,
    controls_dir: Path,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    active_mask: np.ndarray,
    source_controls_path: Path,
    source_controls_sha256: str,
    recorded_cr3bp_terminal_error: float,
    point_mass_replay_terminal_error: float,
    configured_threshold: float,
    retune_result: dict[str, object],
    amax: float,
    cache_path: Path,
    cache_sha256: str,
    start_jd_tdb: float,
    representative_epoch_label: str,
) -> tuple[Path, str]:
    if record_type == "nominal":
        filename = f"{case_id}_nominal_point_mass_retuned_controls.json"
    else:
        filename = f"{case_id}_branch_{int(branch_order):03d}_mask_{int(mask_index):03d}_point_mass_retuned_controls.json"
    path = controls_dir / filename
    endpoint = np.asarray(retune_result["endpoint_controls"], dtype=float)
    midpoint = np.asarray(retune_result["midpoint_controls"], dtype=float)
    diagnostics = _control_norm_diagnostics(endpoint, midpoint, amax=float(amax))
    limitations = [
        f"Cached JPL Horizons Moon/Sun vectors provide representative {representative_epoch_label} geometry only.",
        "Dynamics are Earth central gravity plus indirect Moon/Sun point-mass terms in a geocentric inertial frame.",
        "Controls were retuned independently for this stress model and should not be read as production solver parity.",
        "This is not SPICE propagation, full high-fidelity or flight validation, fuel optimality, DOI evidence, or quantum advantage evidence.",
    ]
    payload = {
        "schema_version": 1,
        "sidecar_type": "independent_hs_horizons_point_mass_retuned_controls",
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else [int(value) for value in outage_mask],
        "active_control_mask": [int(value) for value in np.asarray(active_mask, dtype=int).tolist()],
        "amax": float(amax),
        "source_controls_path": _relative_or_absolute(source_controls_path),
        "source_controls_sha256": source_controls_sha256,
        "cache_path": _relative_or_absolute(cache_path),
        "cache_sha256": cache_sha256,
        "start_jd_tdb": float(start_jd_tdb),
        "representative_epoch_label": representative_epoch_label,
        "recorded_cr3bp_terminal_error": float(recorded_cr3bp_terminal_error),
        "point_mass_replay_terminal_error": float(point_mass_replay_terminal_error),
        "point_mass_retuned_terminal_error": float(retune_result["terminal_error"]),
        "configured_threshold": float(configured_threshold),
        "retuned_passes_configured_threshold": bool(float(retune_result["terminal_error"]) <= float(configured_threshold)),
        "scipy_success": bool(retune_result["scipy_success"]),
        "scipy_message": str(retune_result["scipy_message"]),
        "nfev": int(retune_result["nfev"]),
        "cost": float(retune_result["cost"]),
        "optimality": float(retune_result["optimality"]),
        "runtime_seconds": float(retune_result["runtime_seconds"]),
        "retuned_endpoint_controls": endpoint.tolist(),
        "retuned_midpoint_controls": midpoint.tolist(),
        "control_norm_diagnostics": diagnostics,
        "retuning_semantics": (
            "Independent least-squares retuning of endpoint-plus-midpoint controls under cached-Horizons "
            "Earth/Moon/Sun point-mass dynamics; branch outage-masked segments remain inactive."
        ),
        "limitations": limitations,
    }
    _write_json(path, payload)
    return path, _sha256(path)


def _row_for_result(
    *,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    source_controls_path: Path,
    source_controls_sha256: str,
    retuned_controls_path: Path,
    retuned_controls_sha256: str,
    active_mask: np.ndarray,
    recorded_cr3bp_terminal_error: float,
    point_mass_replay_terminal_error: float,
    configured_threshold: float,
    retune_result: dict[str, object],
    cfg,
    cache_path: Path,
    cache_sha256: str,
    start_jd_tdb: float,
) -> dict[str, object]:
    endpoint = np.asarray(retune_result["endpoint_controls"], dtype=float)
    midpoint = np.asarray(retune_result["midpoint_controls"], dtype=float)
    diagnostics = _control_norm_diagnostics(endpoint, midpoint, amax=float(cfg.amax))
    retuned_error = float(retune_result["terminal_error"])
    return {
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
        "source_controls_path": _relative_or_absolute(source_controls_path),
        "source_controls_sha256": source_controls_sha256,
        "retuned_controls_path": _relative_or_absolute(retuned_controls_path),
        "retuned_controls_sha256": retuned_controls_sha256,
        "midpoint_controls_retuned": True,
        "active_control_mask": json.dumps([int(value) for value in np.asarray(active_mask, dtype=int).tolist()]),
        "recorded_cr3bp_terminal_error": float(recorded_cr3bp_terminal_error),
        "point_mass_replay_terminal_error": float(point_mass_replay_terminal_error),
        "point_mass_retuned_terminal_error": retuned_error,
        "point_mass_replay_delta_from_recorded_cr3bp": abs(
            float(point_mass_replay_terminal_error) - float(recorded_cr3bp_terminal_error)
        ),
        "configured_threshold": float(configured_threshold),
        "point_mass_replay_passes_configured_threshold": bool(
            float(point_mass_replay_terminal_error) <= float(configured_threshold)
        ),
        "point_mass_retuned_passes_configured_threshold": bool(retuned_error <= float(configured_threshold)),
        "retune_success": bool(retuned_error <= float(configured_threshold)),
        "scipy_success": bool(retune_result["scipy_success"]),
        "scipy_message": str(retune_result["scipy_message"]),
        "nfev": int(retune_result["nfev"]),
        "cost": float(retune_result["cost"]),
        "optimality": float(retune_result["optimality"]),
        "runtime_seconds": float(retune_result["runtime_seconds"]),
        "control_bound": float(cfg.amax),
        "retuned_endpoint_norm_max": diagnostics["endpoint_norm_max"],
        "retuned_midpoint_norm_max": diagnostics["midpoint_norm_max"],
        "retuned_endpoint_bound_violation": diagnostics["endpoint_bound_violation"],
        "retuned_midpoint_bound_violation": diagnostics["midpoint_bound_violation"],
        "substeps_per_segment": int(cfg.substeps),
        "transfer_time": float(cfg.tf),
        "start_jd_tdb": float(start_jd_tdb),
        "cache_path": _relative_or_absolute(cache_path),
        "cache_sha256": cache_sha256,
        "retuning_semantics": (
            "persisted controls fail direct cached-Horizons Earth/Moon/Sun point-mass replay, "
            "then endpoint-plus-midpoint controls are independently retuned under the same stress model; "
            "not SPICE/full high-fidelity/flight validation and not production solver parity"
        ),
    }


def _branch_rows_by_mask(df: pd.DataFrame) -> list[dict[str, object]]:
    branch = df[df["record_type"] == "branch"]
    rows = []
    ordered = branch.assign(_mask_index=branch["mask_index"].astype(int)).sort_values("_mask_index")
    for _, row in ordered.iterrows():
        rows.append(
            {
                "mask_index": int(row["mask_index"]),
                "point_mass_replay_terminal_error": float(row["point_mass_replay_terminal_error"]),
                "point_mass_retuned_terminal_error": float(row["point_mass_retuned_terminal_error"]),
                "configured_threshold": float(row["configured_threshold"]),
                "retuned_passes_configured_threshold": bool(row["point_mass_retuned_passes_configured_threshold"]),
                "nfev": int(row["nfev"]),
                "runtime_seconds": float(row["runtime_seconds"]),
            }
        )
    return rows


def _write_table(summary: dict[str, object], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / RETUNING_TABLE_NAME
    rows = [
        {
            "Case": POLISH_CASE_ID,
            "Persisted nominal replay": float(summary["persisted_nominal_point_mass_replay_error"]),
            "Retuned nominal": float(summary["retuned_nominal_point_mass_error"]),
            "Persisted branch worst": float(summary["persisted_branch_point_mass_replay_worst_error"]),
            "Retuned branch worst": float(summary["retuned_branch_point_mass_worst_error"]),
            "Branch pass": f"{int(summary['retuned_branch_pass_count'])}/{int(summary['branch_row_count'])}",
            "Cache SHA-256": str(summary["cache_sha256"])[:12],
        }
    ]
    pd.DataFrame(rows).to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


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
        raise RuntimeError("no replay-ready independent-HS polish row found for point-mass retuning")

    cases = _case_by_id(config)
    cache_path = _resolve_cache_path(args.cache)
    if not cache_path.is_file():
        raise RuntimeError(f"Horizons cache not found: {cache_path}")
    cache_sha256 = _sha256(cache_path)
    expected_cache_sha = str(args.expected_cache_sha256 or "").strip()
    if expected_cache_sha and cache_sha256 != expected_cache_sha:
        raise RuntimeError(f"Horizons cache SHA mismatch: expected {expected_cache_sha}, got {cache_sha256}")
    cache = load_horizons_cache(cache_path)
    profile = horizons_point_mass_profile_from_cache(cache)
    representative_epoch_label = _cache_epoch_label(cache, start_jd_tdb=float(args.start_jd_tdb))

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    controls_dir = results_dir / "controls"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)

    input_artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in (config_path, source_states, input_csv, cache_path):
        _append_artifact(input_artifacts, seen_artifacts, path)

    retuning_rows: list[dict[str, object]] = []
    retuned_sidecars: list[dict[str, object]] = []
    for row in input_rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in cases:
            raise RuntimeError(f"configured suite is missing selected case {case_id}")
        case_config = _case_config(config, cases[case_id])
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        validate_horizons_cache_compatibility(
            cache,
            start_jd_tdb=float(args.start_jd_tdb),
            tf=float(cfg.tf),
            n_segments=int(cfg.n_segments),
            canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
            reference_distance_km=float(args.reference_distance_km),
        )
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
        target_delta = float(np.max(np.abs(target_state - recorded_target)))
        if target_delta > float(args.baseline_tolerance):
            raise RuntimeError(f"target_state mismatch for {case_id}: max abs delta {target_delta}")

        nominal_endpoint = np.asarray(nominal.get("nominal_endpoint_controls", nominal.get("controls")), dtype=float)
        nominal_midpoint = np.asarray(nominal.get("nominal_midpoint_controls"), dtype=float)
        nominal_replay = horizons_point_mass_terminal_error(
            np.asarray(states.initial, dtype=float),
            target_state,
            nominal_endpoint,
            cfg.mu,
            cfg.tf,
            cfg.substeps,
            profile=profile,
            midpoint_controls=nominal_midpoint,
            position_scale=cfg.position_scale,
            velocity_scale=cfg.velocity_scale,
            reference_distance_km=float(args.reference_distance_km),
            canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
        )
        nominal_retuned = _retune_endpoint_midpoint_controls(
            state0=np.asarray(states.initial, dtype=float),
            target=target_state,
            cfg=cfg,
            profile=profile,
            endpoint_seed=nominal_endpoint,
            midpoint_seed=nominal_midpoint,
            active_mask=np.ones(int(cfg.n_segments), dtype=int),
            args=args,
        )
        nominal_sidecar_path, nominal_sidecar_sha = _write_retuned_sidecar(
            controls_dir=controls_dir,
            case_id=case_id,
            record_type="nominal",
            branch_order=None,
            mask_index=None,
            outage_mask=None,
            active_mask=np.ones(int(cfg.n_segments), dtype=int),
            source_controls_path=nominal_path,
            source_controls_sha256=nominal_sha,
            recorded_cr3bp_terminal_error=float(manifest.get("nominal_error", row.get("nominal_error"))),
            point_mass_replay_terminal_error=nominal_replay,
            configured_threshold=nominal_threshold,
            retune_result=nominal_retuned,
            amax=float(cfg.amax),
            cache_path=cache_path,
            cache_sha256=cache_sha256,
            start_jd_tdb=float(args.start_jd_tdb),
            representative_epoch_label=representative_epoch_label,
        )
        retuned_sidecars.append(
            {
                "record_type": "nominal",
                "path": _relative_or_absolute(nominal_sidecar_path),
                "sha256": nominal_sidecar_sha,
                "point_mass_retuned_terminal_error": float(nominal_retuned["terminal_error"]),
            }
        )
        retuning_rows.append(
            _row_for_result(
                case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                source_controls_path=nominal_path,
                source_controls_sha256=nominal_sha,
                retuned_controls_path=nominal_sidecar_path,
                retuned_controls_sha256=nominal_sidecar_sha,
                active_mask=np.ones(int(cfg.n_segments), dtype=int),
                recorded_cr3bp_terminal_error=float(manifest.get("nominal_error", row.get("nominal_error"))),
                point_mass_replay_terminal_error=nominal_replay,
                configured_threshold=nominal_threshold,
                retune_result=nominal_retuned,
                cfg=cfg,
                cache_path=cache_path,
                cache_sha256=cache_sha256,
                start_jd_tdb=float(args.start_jd_tdb),
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
            midpoint_controls = np.asarray(branch.get("branch_midpoint_controls"), dtype=float)
            outage_mask = [int(value) for value in branch["outage_mask"]]
            active_mask = np.asarray(outage_mask, dtype=int)
            branch_replay = horizons_point_mass_terminal_error(
                np.asarray(states.initial, dtype=float),
                target_state,
                endpoint_controls,
                cfg.mu,
                cfg.tf,
                cfg.substeps,
                profile=profile,
                midpoint_controls=midpoint_controls,
                position_scale=cfg.position_scale,
                velocity_scale=cfg.velocity_scale,
                reference_distance_km=float(args.reference_distance_km),
                canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
            )
            branch_retuned = _retune_endpoint_midpoint_controls(
                state0=np.asarray(states.initial, dtype=float),
                target=target_state,
                cfg=cfg,
                profile=profile,
                endpoint_seed=endpoint_controls,
                midpoint_seed=midpoint_controls,
                active_mask=active_mask,
                args=args,
            )
            sidecar_path, sidecar_sha = _write_retuned_sidecar(
                controls_dir=controls_dir,
                case_id=case_id,
                record_type="branch",
                branch_order=int(branch["branch_order"]),
                mask_index=int(branch["mask_index"]),
                outage_mask=outage_mask,
                active_mask=active_mask,
                source_controls_path=branch_path,
                source_controls_sha256=branch_sha,
                recorded_cr3bp_terminal_error=float(branch["recorded_branch_terminal_error"]),
                point_mass_replay_terminal_error=branch_replay,
                configured_threshold=robust_threshold,
                retune_result=branch_retuned,
                amax=float(cfg.amax),
                cache_path=cache_path,
                cache_sha256=cache_sha256,
                start_jd_tdb=float(args.start_jd_tdb),
                representative_epoch_label=representative_epoch_label,
            )
            retuned_sidecars.append(
                {
                    "record_type": "branch",
                    "branch_order": int(branch["branch_order"]),
                    "mask_index": int(branch["mask_index"]),
                    "path": _relative_or_absolute(sidecar_path),
                    "sha256": sidecar_sha,
                    "point_mass_retuned_terminal_error": float(branch_retuned["terminal_error"]),
                }
            )
            retuning_rows.append(
                _row_for_result(
                    case_id=case_id,
                    record_type="branch",
                    branch_order=int(branch["branch_order"]),
                    mask_index=int(branch["mask_index"]),
                    outage_mask=outage_mask,
                    source_controls_path=branch_path,
                    source_controls_sha256=branch_sha,
                    retuned_controls_path=sidecar_path,
                    retuned_controls_sha256=sidecar_sha,
                    active_mask=active_mask,
                    recorded_cr3bp_terminal_error=float(branch["recorded_branch_terminal_error"]),
                    point_mass_replay_terminal_error=branch_replay,
                    configured_threshold=robust_threshold,
                    retune_result=branch_retuned,
                    cfg=cfg,
                    cache_path=cache_path,
                    cache_sha256=cache_sha256,
                    start_jd_tdb=float(args.start_jd_tdb),
                )
            )

    if not retuning_rows:
        raise RuntimeError("no independent-HS cached-Horizons point-mass retuning rows were produced")

    df = pd.DataFrame(retuning_rows, columns=RETUNING_COLUMNS)
    csv_path = results_dir / RETUNING_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")

    nominal_rows = df[df["record_type"] == "nominal"]
    branch_rows = df[df["record_type"] == "branch"]
    summary = {
        "case_id": POLISH_CASE_ID,
        "persisted_nominal_point_mass_replay_error": float(nominal_rows["point_mass_replay_terminal_error"].max()),
        "retuned_nominal_point_mass_error": float(nominal_rows["point_mass_retuned_terminal_error"].max()),
        "persisted_branch_point_mass_replay_worst_error": float(branch_rows["point_mass_replay_terminal_error"].max()),
        "retuned_branch_point_mass_worst_error": float(branch_rows["point_mass_retuned_terminal_error"].max()),
        "persisted_nominal_replay_passes_configured_threshold": bool(
            nominal_rows["point_mass_replay_passes_configured_threshold"].map(bool).all()
        ),
        "persisted_branch_replay_pass_count": int(branch_rows["point_mass_replay_passes_configured_threshold"].map(bool).sum()),
        "retuned_nominal_pass_count": int(nominal_rows["point_mass_retuned_passes_configured_threshold"].map(bool).sum()),
        "retuned_nominal_row_count": int(len(nominal_rows)),
        "retuned_branch_pass_count": int(branch_rows["point_mass_retuned_passes_configured_threshold"].map(bool).sum()),
        "branch_row_count": int(len(branch_rows)),
        "retuned_all_rows_pass": bool(df["point_mass_retuned_passes_configured_threshold"].map(bool).all()),
        "cache_sha256": cache_sha256,
    }
    table_path = _write_table(summary, tables_dir)

    control_manifest_path = controls_dir / RETUNED_CONTROL_MANIFEST_NAME
    control_manifest_payload = {
        "schema_version": 1,
        "manifest_type": "independent_hs_horizons_point_mass_retuned_control_manifest",
        "case_id": POLISH_CASE_ID,
        "retuned_sidecar_count": int(len(retuned_sidecars)),
        "retuned_control_sidecars": retuned_sidecars,
        "source_manifest_path": _relative_or_absolute(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "cache_path": _relative_or_absolute(cache_path),
        "cache_sha256": cache_sha256,
        "summary": summary,
        "limitations": [
            "Retuned controls are stress-model artifacts for cached-Horizons Earth/Moon/Sun point-mass dynamics.",
            "This package does not claim SPICE, full high-fidelity or flight validation, production solver parity, fuel optimality, DOI status, or quantum advantage.",
        ],
    }
    _write_json(control_manifest_path, control_manifest_payload)
    control_manifest_sha = _sha256(control_manifest_path)

    scales = canonical_point_mass_scales(
        mu=float(cfg.mu),
        reference_distance_km=float(args.reference_distance_km),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
    )
    limitations = [
        f"Cached JPL Horizons Moon/Sun vectors provide a representative {representative_epoch_label} epoch only.",
        "The force model is Earth central gravity plus indirect Moon/Sun point masses in a geocentric inertial frame.",
        "Persisted controls fail direct point-mass replay; feasibility is restored only after independent retuning.",
        "The retuning optimizer runs for this package, so this is not a recorded-controls-only replay.",
        "This is not SPICE propagation, full high-fidelity or flight validation, production solver parity, fuel optimality, DOI evidence, or quantum advantage evidence.",
    ]
    metadata_path = results_dir / RETUNING_METADATA_NAME
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "nominal_row_count": int(len(nominal_rows)),
        "branch_row_count": int(len(branch_rows)),
        "optimization_rerun": True,
        "retuning": True,
        "uses_cached_horizons_vectors": True,
        "runtime_network_dependency": False,
        "cached_horizons_earth_moon_sun_point_mass_retuning": True,
        "spice_ephemeris_validation": False,
        "high_fidelity_validation": False,
        "high_fidelity_flight_validation": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "doi_claim": False,
        "quantum_advantage_claim": False,
        "cache_sha256": cache_sha256,
        "cache": {
            "path": _relative_or_absolute(cache_path),
            "sha256": cache_sha256,
            "cache_type": cache.get("cache_type"),
            "metadata": cache.get("metadata", {}),
        },
        "canonical_point_mass_constants": {
            "reference_distance_km": scales.reference_distance_km,
            "canonical_time_unit_seconds": scales.canonical_time_unit_seconds,
            "gm_total_km3_s2": scales.gm_total_km3_s2,
            "gm_earth_km3_s2": scales.gm_earth_km3_s2,
            "gm_moon_km3_s2": scales.gm_moon_km3_s2,
            "gm_sun_km3_s2": scales.gm_sun_km3_s2,
            "velocity_scale_km_s": scales.velocity_scale_km_s,
            "control_acceleration_scale_km_s2": scales.acceleration_scale_km_s2,
        },
        "retuning_settings": {
            "max_nfev": int(args.max_nfev),
            "xtol": float(args.xtol),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "terminal_residual_weight": float(args.terminal_residual_weight),
            "control_regularization_weight": float(args.control_regularization_weight),
            "smooth_regularization_weight": float(args.smooth_regularization_weight),
            "branch_outage_segments_fixed_inactive": True,
        },
        "polish_case_summary": {
            **summary,
            "branch_errors_by_mask": _branch_rows_by_mask(df),
        },
        "point_mass_replay_summary": {
            "persisted_nominal_point_mass_replay_error": summary["persisted_nominal_point_mass_replay_error"],
            "persisted_branch_point_mass_replay_worst_error": summary["persisted_branch_point_mass_replay_worst_error"],
            "persisted_nominal_replay_passes_configured_threshold": summary[
                "persisted_nominal_replay_passes_configured_threshold"
            ],
            "persisted_branch_replay_pass_count": summary["persisted_branch_replay_pass_count"],
            "branch_row_count": summary["branch_row_count"],
        },
        "point_mass_retuned_summary": {
            "retuned_nominal_point_mass_error": summary["retuned_nominal_point_mass_error"],
            "retuned_branch_point_mass_worst_error": summary["retuned_branch_point_mass_worst_error"],
            "retuned_nominal_pass_count": summary["retuned_nominal_pass_count"],
            "retuned_nominal_row_count": summary["retuned_nominal_row_count"],
            "retuned_branch_pass_count": summary["retuned_branch_pass_count"],
            "branch_row_count": summary["branch_row_count"],
            "retuned_all_rows_pass": summary["retuned_all_rows_pass"],
            "scipy_success_count": int(df["scipy_success"].map(bool).sum()),
            "total_nfev": int(df["nfev"].sum()),
            "total_runtime_seconds": float(df["runtime_seconds"].sum()),
            "max_endpoint_bound_violation": float(df["retuned_endpoint_bound_violation"].max()),
            "max_midpoint_bound_violation": float(df["retuned_midpoint_bound_violation"].max()),
        },
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_horizons_point_mass_retuning_csv": _relative_or_absolute(csv_path),
            "independent_hs_horizons_point_mass_retuning_metadata_json": _relative_or_absolute(metadata_path),
            "independent_hs_horizons_point_mass_retuning_table_tex": _relative_or_absolute(table_path),
            "retuned_control_manifest_json": _relative_or_absolute(control_manifest_path),
            "retuned_control_manifest_sha256": control_manifest_sha,
            "retuned_control_sidecars": retuned_sidecars,
        },
        "limitations": limitations,
        "interpretation_limits": limitations,
    }
    _write_json(metadata_path, metadata)
    print(
        "Completed independent-HS cached-Horizons point-mass retuning "
        f"with nominal replay={summary['persisted_nominal_point_mass_replay_error']:.6g}, "
        f"retuned nominal={summary['retuned_nominal_point_mass_error']:.6g}, "
        f"retuned branch worst={summary['retuned_branch_point_mass_worst_error']:.6g}, "
        f"branch pass={summary['retuned_branch_pass_count']}/{summary['branch_row_count']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and retune independent-HS polish controls under cached-Horizons "
            "Earth/Moon/Sun point-mass dynamics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--expected-cache-sha256", default=EXPECTED_CACHE_SHA256)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--phase-time", type=float, action="append", default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--start-jd-tdb", type=float, default=DEFAULT_CACHE_START_JD_TDB)
    parser.add_argument("--canonical-time-unit-seconds", type=float, default=DEFAULT_CANONICAL_TIME_UNIT_SECONDS)
    parser.add_argument("--reference-distance-km", type=float, default=DEFAULT_REFERENCE_DISTANCE_KM)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--xtol", type=float, default=1.0e-5)
    parser.add_argument("--ftol", type=float, default=1.0e-5)
    parser.add_argument("--gtol", type=float, default=1.0e-4)
    parser.add_argument("--terminal-residual-weight", type=float, default=1.0)
    parser.add_argument("--control-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--smooth-regularization-weight", type=float, default=5.0e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
