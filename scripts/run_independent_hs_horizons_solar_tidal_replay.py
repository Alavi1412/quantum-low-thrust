"""Cached-Horizons-derived solar-tidal replay for persisted independent-HS controls.

This postprocessor validates replay-ready independent-midpoint Hermite-Simpson
nominal and branch sidecars, validates a committed JPL Horizons Moon/Sun vector
cache, and repropagates persisted endpoint-plus-midpoint controls under CR3BP
plus a simplified solar-tidal acceleration from linearly interpolated cached
Horizons Sun geometry. It does not rerun optimization, use SPICE propagation,
perform high-fidelity or flight validation, establish production solver parity,
or retune controls under the perturbation.
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
from qlt.bicircular import SolarTidalParameters
from qlt.direct_collocation import propagate_piecewise_controls
from qlt.ephemeris_contrast import (
    DEFAULT_CACHE_START_JD_TDB,
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    canonical_node_jds,
    canonical_node_times,
    fetch_horizons_vectors,
    horizons_query_url,
    horizons_sun_vector_profile_from_cache,
    horizons_vectors_query_params,
    load_horizons_cache,
    propagate_piecewise_controls_horizons_solar_tidal,
    validate_horizons_cache_compatibility,
    write_horizons_cache,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "independent_hs_all_configured_headroom.yaml"
DEFAULT_CACHE = ROOT / "data" / "cache" / "horizons" / "independent_hs_phase_shift_2026jan01_vectors.json"
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_horizons_solar_tidal_replay")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_horizons_solar_tidal_replay")
REPLAY_CSV_NAME = "independent_hs_horizons_solar_tidal_replay.csv"
REPLAY_METADATA_NAME = "independent_hs_horizons_solar_tidal_replay_metadata.json"
REPLAY_TABLE_NAME = "independent_hs_horizons_solar_tidal_replay_table.tex"
POLISH_CASE_ID = "ihs_all_single_p04_amax02_polish_from_p04"

REPLAY_COLUMNS = [
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "controls_path",
    "controls_sha256",
    "midpoint_controls_replayed",
    "recorded_cr3bp_terminal_error",
    "cr3bp_terminal_error",
    "cr3bp_delta_from_recorded",
    "horizons_solar_tidal_terminal_error",
    "horizons_solar_tidal_delta_from_cr3bp",
    "configured_threshold",
    "cr3bp_passes_configured_threshold",
    "horizons_solar_tidal_passes_configured_threshold",
    "threshold_semantics",
    "recovery_start",
    "recovery_segments",
    "optimizer_success",
    "nfev",
    "phase_time",
    "control_count",
    "substeps_per_segment",
    "transfer_time",
    "start_jd_tdb",
    "cache_path",
    "cache_sha256",
    "sun_distance_lu_min",
    "sun_distance_lu_max",
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


def _resolve_cache_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


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


def _horizons_terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    sun_mu_ratio: float,
) -> float:
    final, _ = propagate_piecewise_controls_horizons_solar_tidal(
        state0,
        endpoint_controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=float(sun_mu_ratio),
        midpoint_controls=midpoint_controls,
    )
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def _row_for_record(
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
    horizons_solar_tidal_terminal_error: float,
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
    start_jd_tdb: float,
    cache_path: Path,
    cache_sha256: str,
    sun_distance_lu_min: float,
    sun_distance_lu_max: float,
) -> dict[str, object]:
    cr3bp_delta = abs(float(cr3bp_terminal_error) - float(recorded_cr3bp_terminal_error))
    return {
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
        "controls_path": _relative_or_absolute(controls_path),
        "controls_sha256": controls_sha256,
        "midpoint_controls_replayed": bool(midpoint_controls_replayed),
        "recorded_cr3bp_terminal_error": float(recorded_cr3bp_terminal_error),
        "cr3bp_terminal_error": float(cr3bp_terminal_error),
        "cr3bp_delta_from_recorded": float(cr3bp_delta),
        "horizons_solar_tidal_terminal_error": float(horizons_solar_tidal_terminal_error),
        "horizons_solar_tidal_delta_from_cr3bp": abs(
            float(horizons_solar_tidal_terminal_error) - float(cr3bp_terminal_error)
        ),
        "configured_threshold": float(threshold),
        "cr3bp_passes_configured_threshold": bool(cr3bp_terminal_error <= float(threshold)),
        "horizons_solar_tidal_passes_configured_threshold": bool(
            horizons_solar_tidal_terminal_error <= float(threshold)
        ),
        "threshold_semantics": threshold_semantics,
        "recovery_start": "" if recovery_start is None else int(recovery_start),
        "recovery_segments": "" if recovery_segments is None else int(recovery_segments),
        "optimizer_success": bool(optimizer_success),
        "nfev": int(nfev),
        "phase_time": float(phase_time),
        "control_count": int(control_count),
        "substeps_per_segment": int(substeps_per_segment),
        "transfer_time": float(transfer_time),
        "start_jd_tdb": float(start_jd_tdb),
        "cache_path": _relative_or_absolute(cache_path),
        "cache_sha256": cache_sha256,
        "sun_distance_lu_min": float(sun_distance_lu_min),
        "sun_distance_lu_max": float(sun_distance_lu_max),
        "solar_tidal_semantics": (
            "persisted independent-HS endpoint-plus-midpoint controls repropagated against the "
            "original CR3BP target with a simplified solar-tidal acceleration from linearly "
            "interpolated cached JPL Horizons Sun geometry; no retuning, no SPICE propagation, "
            "no high-fidelity or flight validation, no production solver parity"
        ),
    }


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


def _refresh_cache(args: argparse.Namespace, *, cfg) -> Path:
    cache_path = _resolve_cache_path(args.cache)
    jds = canonical_node_jds(
        start_jd_tdb=float(args.start_jd_tdb),
        tf=float(cfg.tf),
        n_segments=int(cfg.n_segments),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
    )
    moon_params = horizons_vectors_query_params("301", "500@399", jds)
    sun_params = horizons_vectors_query_params("10", "500@399", jds)
    moon_response = fetch_horizons_vectors(moon_params, timeout_seconds=float(args.fetch_timeout_seconds))
    sun_response = fetch_horizons_vectors(sun_params, timeout_seconds=float(args.fetch_timeout_seconds))
    metadata = {
        "representative_epoch_note": (
            "Fixed representative 2026-Jan-01 TDB epoch for an independent-HS cached-Horizons-derived "
            "solar-tidal replay/stress probe only; it is not a mission epoch."
        ),
        "start_jd_tdb": float(args.start_jd_tdb),
        "start_calendar_tdb": "A.D. 2026-Jan-01 00:00:00.0000",
        "canonical_transfer_time": float(cfg.tf),
        "segments": int(cfg.n_segments),
        "canonical_node_times": [float(value) for value in canonical_node_times(cfg.tf, cfg.n_segments)],
        "node_jd_tdb": [float(value) for value in jds],
        "canonical_time_unit_seconds": float(args.canonical_time_unit_seconds),
        "reference_distance_km": float(args.reference_distance_km),
        "query_docs": {
            "horizons_api": "https://ssd-api.jpl.nasa.gov/doc/horizons.html",
            "horizons_manual": "https://ssd.jpl.nasa.gov/horizons/manual.html",
            "de440_de441": "https://ssd.jpl.nasa.gov/doc/de440_de441.html",
        },
        "query_urls": {
            "moon_geocentric": horizons_query_url(moon_params),
            "sun_geocentric": horizons_query_url(sun_params),
        },
        "body_queries": {
            "moon_geocentric": {
                "COMMAND": "301",
                "CENTER": "500@399",
                "description": "Moon geometric vectors relative to Earth center",
            },
            "sun_geocentric": {
                "COMMAND": "10",
                "CENTER": "500@399",
                "description": "Sun geometric vectors relative to Earth center",
            },
        },
        "ephemeris_note": (
            "Horizons result headers for this cache report the active Horizons vector source; "
            "the replay remains a simplified solar-tidal stress probe."
        ),
    }
    write_horizons_cache(path=cache_path, moon_response=moon_response, sun_response=sun_response, metadata=metadata)
    return cache_path


def _summary_rows(df: pd.DataFrame, *, cache_sha256: str, sun_distance_range: tuple[float, float]) -> list[dict[str, object]]:
    rows = []
    for case_id, group in df.groupby("case_id", sort=True):
        nominal = group[group["record_type"] == "nominal"]
        branch = group[group["record_type"] == "branch"]
        rows.append(
            {
                "case_id": str(case_id),
                "nominal_horizons_solar_tidal_terminal_error": float(
                    nominal["horizons_solar_tidal_terminal_error"].max()
                )
                if len(nominal)
                else None,
                "branch_horizons_solar_tidal_worst_error": float(
                    branch["horizons_solar_tidal_terminal_error"].max()
                )
                if len(branch)
                else None,
                "branch_horizons_solar_tidal_pass_count": int(
                    branch["horizons_solar_tidal_passes_configured_threshold"].map(bool).sum()
                ),
                "branch_row_count": int(len(branch)),
                "nominal_horizons_solar_tidal_pass_count": int(
                    nominal["horizons_solar_tidal_passes_configured_threshold"].map(bool).sum()
                ),
                "nominal_row_count": int(len(nominal)),
                "max_cr3bp_delta_from_recorded": float(group["cr3bp_delta_from_recorded"].max()) if len(group) else 0.0,
                "all_horizons_solar_tidal_rows_pass": bool(
                    group["horizons_solar_tidal_passes_configured_threshold"].map(bool).all()
                )
                if len(group)
                else False,
                "optimizer_success": bool(group["optimizer_success"].map(bool).all()) if len(group) else False,
                "sun_distance_lu_min": float(sun_distance_range[0]),
                "sun_distance_lu_max": float(sun_distance_range[1]),
                "cache_sha256": cache_sha256,
            }
        )
    return rows


def _branch_rows_by_mask(df: pd.DataFrame, case_id: str) -> list[dict[str, object]]:
    branch = df[(df["case_id"] == case_id) & (df["record_type"] == "branch")]
    rows = []
    ordered = branch.assign(_mask_index=branch["mask_index"].astype(int)).sort_values("_mask_index")
    for _, row in ordered.iterrows():
        rows.append(
            {
                "mask_index": int(row["mask_index"]),
                "horizons_solar_tidal_terminal_error": float(row["horizons_solar_tidal_terminal_error"]),
                "configured_threshold": float(row["configured_threshold"]),
                "passes_configured_threshold": bool(row["horizons_solar_tidal_passes_configured_threshold"]),
            }
        )
    return rows


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / REPLAY_TABLE_NAME
    if not summary_rows:
        path.write_text("% No independent-HS cached-Horizons replay rows.\n", encoding="utf-8")
        return path
    display = pd.DataFrame(summary_rows)[
        [
            "case_id",
            "nominal_horizons_solar_tidal_terminal_error",
            "branch_horizons_solar_tidal_worst_error",
            "branch_horizons_solar_tidal_pass_count",
            "branch_row_count",
            "max_cr3bp_delta_from_recorded",
            "sun_distance_lu_min",
            "sun_distance_lu_max",
            "cache_sha256",
        ]
    ].rename(
        columns={
            "case_id": "Case",
            "nominal_horizons_solar_tidal_terminal_error": "Horizons nominal",
            "branch_horizons_solar_tidal_worst_error": "Horizons branch worst",
            "branch_horizons_solar_tidal_pass_count": "Branch pass",
            "branch_row_count": "Branch rows",
            "max_cr3bp_delta_from_recorded": "CR3BP replay delta",
            "sun_distance_lu_min": "Sun LU min",
            "sun_distance_lu_max": "Sun LU max",
            "cache_sha256": "Cache SHA-256",
        }
    )
    display["Cache SHA-256"] = display["Cache SHA-256"].astype(str).str.slice(0, 12)
    display.to_latex(path, index=False, float_format="%.6g", escape=True)
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
        raise RuntimeError("no replay-ready independent-HS p=0.4 rows found for cached-Horizons replay")

    cases = _case_by_id(config)
    first_case_id = str(input_rows[0].get("case_id", ""))
    if first_case_id not in cases:
        raise RuntimeError(f"configured suite is missing selected case {first_case_id}")
    first_case_config = _case_config(config, cases[first_case_id])
    first_states = load_configured_states(Path.cwd(), first_case_config, source_states)
    first_cfg = make_objective_config(first_case_config, first_states.mu)

    if bool(args.refresh_cache):
        cache_path = _refresh_cache(args, cfg=first_cfg)
    else:
        cache_path = _resolve_cache_path(args.cache)
    if not cache_path.is_file():
        raise RuntimeError(f"Horizons cache not found: {cache_path}. Re-run with --refresh-cache to regenerate it.")
    cache = load_horizons_cache(cache_path)
    cache_sha256 = _sha256(cache_path)

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in (config_path, source_states, input_csv, cache_path):
        _append_artifact(input_artifacts, seen_artifacts, path)

    replay_rows: list[dict[str, object]] = []
    sun_distance_range = (0.0, 0.0)
    for row in input_rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in cases:
            continue
        case = cases[case_id]
        case_config = _case_config(config, case)
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
        profile = horizons_sun_vector_profile_from_cache(
            cache,
            mu=float(cfg.mu),
            reference_distance_km=float(args.reference_distance_km),
        )
        canonical_times = np.asarray(profile["canonical_time"], dtype=float)
        sun_vectors = np.asarray(profile["sun_barycenter_rotating_lu"], dtype=float)
        sun_distance = np.asarray(profile["sun_distance_lu"], dtype=float)
        sun_distance_range = (float(sun_distance.min()), float(sun_distance.max()))

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
            raise RuntimeError(f"target_state mismatch for {case_id}: max abs delta {target_state_max_abs_delta}")

        nominal_endpoint = np.asarray(nominal.get("nominal_endpoint_controls", nominal.get("controls")), dtype=float)
        nominal_midpoint_raw = nominal.get("nominal_midpoint_controls")
        nominal_midpoint = None if nominal_midpoint_raw is None else np.asarray(nominal_midpoint_raw, dtype=float)
        nominal_cr3bp = _cr3bp_terminal_error(
            state0=np.asarray(states.initial, dtype=float),
            target=target_state,
            cfg=cfg,
            endpoint_controls=nominal_endpoint,
            midpoint_controls=nominal_midpoint,
        )
        nominal_horizons = _horizons_terminal_error(
            state0=np.asarray(states.initial, dtype=float),
            target=target_state,
            cfg=cfg,
            endpoint_controls=nominal_endpoint,
            midpoint_controls=nominal_midpoint,
            canonical_times=canonical_times,
            sun_vectors_rotating_lu=sun_vectors,
            sun_mu_ratio=float(args.sun_mu_ratio),
        )
        nominal_recorded = float(manifest.get("nominal_error", row.get("nominal_error")))
        replay_rows.append(
            _row_for_record(
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
                horizons_solar_tidal_terminal_error=nominal_horizons,
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
                start_jd_tdb=float(args.start_jd_tdb),
                cache_path=cache_path,
                cache_sha256=cache_sha256,
                sun_distance_lu_min=sun_distance_range[0],
                sun_distance_lu_max=sun_distance_range[1],
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
                state0=np.asarray(states.initial, dtype=float),
                target=target_state,
                cfg=cfg,
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
            )
            branch_horizons = _horizons_terminal_error(
                state0=np.asarray(states.initial, dtype=float),
                target=target_state,
                cfg=cfg,
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
                canonical_times=canonical_times,
                sun_vectors_rotating_lu=sun_vectors,
                sun_mu_ratio=float(args.sun_mu_ratio),
            )
            replay_rows.append(
                _row_for_record(
                    case_id=case_id,
                    record_type="branch",
                    branch_order=int(branch["branch_order"]),
                    mask_index=int(branch["mask_index"]),
                    outage_mask=[int(value) for value in branch["outage_mask"]],
                    controls_path=branch_path,
                    controls_sha256=branch_sha,
                    midpoint_controls_replayed=midpoint_controls is not None,
                    recorded_cr3bp_terminal_error=float(branch["recorded_branch_terminal_error"]),
                    cr3bp_terminal_error=branch_cr3bp,
                    horizons_solar_tidal_terminal_error=branch_horizons,
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
                    start_jd_tdb=float(args.start_jd_tdb),
                    cache_path=cache_path,
                    cache_sha256=cache_sha256,
                    sun_distance_lu_min=sun_distance_range[0],
                    sun_distance_lu_max=sun_distance_range[1],
                )
            )

    if not replay_rows:
        raise RuntimeError("no independent-HS cached-Horizons solar-tidal replay rows were produced")

    df = pd.DataFrame(replay_rows, columns=REPLAY_COLUMNS)
    csv_path = results_dir / REPLAY_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")
    summary_rows = _summary_rows(df, cache_sha256=cache_sha256, sun_distance_range=sun_distance_range)
    table_path = _write_table(summary_rows, tables_dir)

    branch_rows = df[df["record_type"] == "branch"]
    nominal_rows = df[df["record_type"] == "nominal"]
    max_cr3bp_delta = float(df["cr3bp_delta_from_recorded"].max()) if len(df) else 0.0
    max_nominal_horizons = (
        float(nominal_rows["horizons_solar_tidal_terminal_error"].max()) if len(nominal_rows) else 0.0
    )
    max_branch_horizons = (
        float(branch_rows["horizons_solar_tidal_terminal_error"].max()) if len(branch_rows) else 0.0
    )
    polish_rows = df[df["case_id"] == POLISH_CASE_ID]
    polish_nominal_rows = polish_rows[polish_rows["record_type"] == "nominal"]
    polish_branch_rows = polish_rows[polish_rows["record_type"] == "branch"]
    polish_nominal_error = (
        float(polish_nominal_rows["horizons_solar_tidal_terminal_error"].max()) if len(polish_nominal_rows) else None
    )
    polish_branch_worst = (
        float(polish_branch_rows["horizons_solar_tidal_terminal_error"].max()) if len(polish_branch_rows) else None
    )
    polish_branch_pass_count = int(polish_branch_rows["horizons_solar_tidal_passes_configured_threshold"].map(bool).sum())
    polish_branch_row_count = int(len(polish_branch_rows))
    limitations = [
        "Cached JPL Horizons Moon/Sun vectors provide representative 2026-Jan-01 geometry only.",
        "Propagation is normalized CR3BP plus a simplified differential solar-tidal acceleration from interpolated Sun vectors.",
        "Controls are persisted independent-HS endpoint-plus-midpoint controls and are not retuned under the perturbation.",
        "This is not SPICE propagation, high-fidelity or flight validation, production solver parity, fuel optimality, or quantum evidence.",
        "Terminal errors are measured against the original CR3BP target at the original fixed final time.",
    ]
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "case_count": int(df["case_id"].nunique()),
        "nominal_row_count": int(len(nominal_rows)),
        "branch_row_count": int(len(branch_rows)),
        "optimization_rerun": False,
        "uses_recorded_artifacts_only": True,
        "uses_cached_horizons_vectors": True,
        "runtime_network_dependency": bool(args.refresh_cache),
        "independent_hs_horizons_solar_tidal_replay": True,
        "cached_horizons_derived_solar_tidal_replay": True,
        "cached_horizons_derived_solar_tidal_stress_probe": True,
        "high_fidelity_validation": False,
        "spice_ephemeris_validation": False,
        "accepted_control_high_fidelity_replay": False,
        "production_solver_parity_claim": False,
        "flight_ready_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "sun_mu_ratio": float(args.sun_mu_ratio),
        "baseline_tolerance": float(args.baseline_tolerance),
        "cr3bp_max_replay_delta": max_cr3bp_delta,
        "polish_nominal_horizons_solar_tidal_terminal_error": polish_nominal_error,
        "polish_branch_horizons_solar_tidal_worst_error": polish_branch_worst,
        "polish_branch_horizons_solar_tidal_pass_count": polish_branch_pass_count,
        "polish_branch_horizons_solar_tidal_row_count": polish_branch_row_count,
        "sun_distance_lu_range": [float(sun_distance_range[0]), float(sun_distance_range[1])],
        "cache_sha256": cache_sha256,
        "cache": {
            "path": _relative_or_absolute(cache_path),
            "sha256": cache_sha256,
            "cache_type": cache.get("cache_type"),
            "metadata": cache.get("metadata", {}),
        },
        "baseline_reproduction": {
            "max_cr3bp_delta_from_recorded": max_cr3bp_delta,
            "passes_baseline_tolerance": bool(max_cr3bp_delta <= float(args.baseline_tolerance)),
        },
        "horizons_solar_tidal_summary": {
            "max_nominal_horizons_solar_tidal_terminal_error": max_nominal_horizons,
            "max_branch_horizons_solar_tidal_terminal_error": max_branch_horizons,
            "nominal_horizons_solar_tidal_pass_count": int(
                nominal_rows["horizons_solar_tidal_passes_configured_threshold"].map(bool).sum()
            ),
            "nominal_horizons_solar_tidal_row_count": int(len(nominal_rows)),
            "branch_horizons_solar_tidal_pass_count": int(
                branch_rows["horizons_solar_tidal_passes_configured_threshold"].map(bool).sum()
            ),
            "branch_horizons_solar_tidal_row_count": int(len(branch_rows)),
            "all_nominal_horizons_solar_tidal_rows_pass": bool(
                nominal_rows["horizons_solar_tidal_passes_configured_threshold"].map(bool).all()
            )
            if len(nominal_rows)
            else False,
            "all_branch_horizons_solar_tidal_rows_pass": bool(
                branch_rows["horizons_solar_tidal_passes_configured_threshold"].map(bool).all()
            )
            if len(branch_rows)
            else False,
            "all_horizons_solar_tidal_rows_pass": bool(
                df["horizons_solar_tidal_passes_configured_threshold"].map(bool).all()
            ),
        },
        "polish_case_summary": {
            "case_id": POLISH_CASE_ID,
            "row_count": int(len(polish_rows)),
            "nominal_row_count": int(len(polish_nominal_rows)),
            "branch_row_count": polish_branch_row_count,
            "nominal_horizons_solar_tidal_terminal_error": polish_nominal_error,
            "branch_horizons_solar_tidal_worst_error": polish_branch_worst,
            "branch_horizons_solar_tidal_pass_count": polish_branch_pass_count,
            "branch_horizons_solar_tidal_row_count": polish_branch_row_count,
            "branch_errors_by_mask": _branch_rows_by_mask(df, POLISH_CASE_ID),
        },
        "summary": summary_rows,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_horizons_solar_tidal_replay_csv": _relative_or_absolute(csv_path),
            "independent_hs_horizons_solar_tidal_replay_metadata_json": _relative_or_absolute(
                results_dir / REPLAY_METADATA_NAME
            ),
            "independent_hs_horizons_solar_tidal_replay_table_tex": _relative_or_absolute(table_path),
        },
        "limitations": limitations,
        "interpretation_limits": limitations,
    }
    _write_json(results_dir / REPLAY_METADATA_NAME, metadata)
    print(
        "Completed independent-HS cached-Horizons solar-tidal replay "
        f"with {int(len(df))} rows, polish_nominal={polish_nominal_error:.6g}, "
        f"polish_branch_worst={polish_branch_worst:.6g}, "
        f"polish_branch_pass={polish_branch_pass_count}/{polish_branch_row_count}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay persisted independent-HS endpoint-plus-midpoint controls under normalized CR3BP "
            "plus a cached-Horizons-derived simplified solar-tidal acceleration."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-cache", action="store_true", help="Fetch and rewrite the Horizons cache before running.")
    parser.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--phase-time", type=float, action="append", default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--start-jd-tdb", type=float, default=DEFAULT_CACHE_START_JD_TDB)
    parser.add_argument("--canonical-time-unit-seconds", type=float, default=DEFAULT_CANONICAL_TIME_UNIT_SECONDS)
    parser.add_argument("--reference-distance-km", type=float, default=DEFAULT_REFERENCE_DISTANCE_KM)
    parser.add_argument("--sun-mu-ratio", type=float, default=SolarTidalParameters().sun_mu_ratio)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
