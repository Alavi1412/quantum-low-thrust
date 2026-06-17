"""Build an offline Horizons-derived force-model contrast for the tail-coast package.

The default path reads a committed JPL Horizons vector cache and compares the
simple bicircular solar-tidal assumptions against cached Earth/Moon/Sun
geometry over the configured hard-catalog transfer window. It does not rerun
optimization, retune controls, perform SPICE validation, or propagate an
accepted-control trajectory under a high-fidelity model.
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
from qlt.bicircular import SolarTidalParameters, solar_tidal_acceleration
from qlt.cr3bp import propagate_controls_batch
from qlt.ephemeris_contrast import (
    DEFAULT_CACHE_START_JD_TDB,
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    canonical_node_jds,
    canonical_node_times,
    fetch_horizons_vectors,
    horizons_query_url,
    horizons_rotating_geometry,
    horizons_vectors_query_params,
    load_horizons_cache,
    samples_from_cache,
    solar_tidal_acceleration_from_sun_vector,
    validate_horizons_cache_compatibility,
    wrapped_angle_delta,
    write_horizons_cache,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "hard_catalog_tail_coast_branch_control_replay.yaml"
DEFAULT_CACHE = ROOT / "data" / "cache" / "horizons" / "hard_catalog_tail_coast_2026jan01_vectors.json"
DEFAULT_RESULTS_DIR = Path("data/results/horizons_ephemeris_force_model_contrast")
DEFAULT_TABLES_DIR = Path("tables/horizons_ephemeris_force_model_contrast")

CONTRAST_CSV_NAME = "horizons_ephemeris_force_model_contrast.csv"
CONTRAST_METADATA_NAME = "horizons_ephemeris_force_model_contrast_metadata.json"
CONTRAST_TABLE_NAME = "horizons_ephemeris_force_model_contrast_table.tex"
TAIL_COAST_COMBINED_CASE = "tail_coast_all_one_two_segment_t5_portfolio"

CONTRAST_COLUMNS = [
    "node_index",
    "canonical_time",
    "jd_tdb",
    "calendar_tdb",
    "em_distance_km",
    "em_distance_ratio_to_mean",
    "em_distance_delta_percent_vs_mean",
    "em_angular_rate_ratio_to_mean",
    "em_angular_rate_delta_percent_vs_mean",
    "sun_distance_lu",
    "sun_distance_delta_percent_vs_bicircular",
    "sun_phase_rotating_rad",
    "bicircular_aligned_phase_rad",
    "sun_phase_delta_degrees_wrapped",
    "nominal_horizons_tidal_accel_norm",
    "nominal_bicircular_tidal_accel_norm",
    "nominal_tidal_accel_delta_norm",
    "nominal_tidal_accel_delta_fraction",
    "representative_branch_mask_index",
    "representative_branch_horizons_tidal_accel_norm",
    "representative_branch_bicircular_tidal_accel_norm",
    "representative_branch_tidal_accel_delta_norm",
    "representative_branch_tidal_accel_delta_fraction",
    "contrast_semantics",
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


def _refresh_cache(args: argparse.Namespace, *, cfg) -> Path:
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
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
            "Fixed representative epoch for force-model contrast only; it is not a mission epoch."
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
            "Horizons result headers for this cache report DE441 sources for the Moon and Sun vectors."
        ),
    }
    write_horizons_cache(path=cache_path, moon_response=moon_response, sun_response=sun_response, metadata=metadata)
    return cache_path


def _load_case_context(args: argparse.Namespace) -> tuple[dict, dict, object, object, Path, Path, Path, list[dict[str, str]]]:
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
    if TAIL_COAST_COMBINED_CASE not in cases:
        raise RuntimeError(f"configured suite is missing {TAIL_COAST_COMBINED_CASE}")
    case_config = tail_runner._case_config(config, cases[TAIL_COAST_COMBINED_CASE])
    states = load_configured_states(Path.cwd(), case_config, source_states)
    cfg = make_objective_config(case_config, states.mu)
    return config, case_config, states, cfg, config_path, source_states, tail_csv, input_rows


def _tail_replay_row(input_rows: list[dict[str, str]]) -> dict[str, str]:
    for row in input_rows:
        if str(row.get("suite_case_id", "")) == TAIL_COAST_COMBINED_CASE and _bool_value(
            row.get("branch_control_replay_ready", "")
        ):
            return row
    raise RuntimeError(f"no branch-control replay-ready row found for {TAIL_COAST_COMBINED_CASE}")


def _segment_node_history(state0: np.ndarray, controls: np.ndarray, cfg) -> np.ndarray:
    _, history = propagate_controls_batch(
        state0,
        np.asarray(controls, dtype=float).reshape((int(cfg.n_segments), 3)),
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        return_history=True,
    )
    if history is None:
        raise RuntimeError("expected propagation history")
    node_indices = [i * int(cfg.substeps) for i in range(int(cfg.n_segments) + 1)]
    return history[node_indices, 0, :]


def _selected_branch_sidecar(manifest: dict) -> dict[str, object]:
    entries = list(manifest.get("branch_control_sidecars", []))
    if not entries:
        raise RuntimeError("branch-control manifest has no branch sidecars")
    return max(entries, key=lambda item: float(item.get("terminal_error", 0.0)))


def _phase_delta_degrees(horizons_phase: np.ndarray, bicircular_phase: np.ndarray) -> np.ndarray:
    return np.rad2deg(wrapped_angle_delta(horizons_phase - bicircular_phase))


def _safe_fraction(delta: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return delta / np.maximum(reference, 1.0e-18)


def _summary_range(values: pd.Series, *, precision: int = 6) -> str:
    return f"{float(values.min()):.{precision}g}--{float(values.max()):.{precision}g}"


def _write_table(
    df: pd.DataFrame,
    tables_dir: Path,
    *,
    parameters: SolarTidalParameters,
    reference_distance_km: float,
) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / CONTRAST_TABLE_NAME
    rows = [
        {
            "Metric": "Earth--Moon distance",
            "Horizons-derived range": _summary_range(df["em_distance_ratio_to_mean"]),
            "Simple-model reference": "1.0 fixed",
            "Interpretation": "diagnostic ratio to window mean; not the Sun-vector LU scale",
        },
        {
            "Metric": "Earth--Moon angular rate",
            "Horizons-derived range": _summary_range(df["em_angular_rate_ratio_to_mean"]),
            "Simple-model reference": "1.0 fixed",
            "Interpretation": "angular-rate variation absent from CR3BP normalization",
        },
        {
            "Metric": "Sun distance",
            "Horizons-derived range": _summary_range(df["sun_distance_lu"]),
            "Simple-model reference": f"{parameters.sun_distance_lu:.6g} LU; {float(reference_distance_km):.6g} km/LU",
            "Interpretation": "cached Sun vector divided by the fixed CR3BP length unit",
        },
        {
            "Metric": "Sun phase offset",
            "Horizons-derived range": _summary_range(df["sun_phase_delta_degrees_wrapped"]),
            "Simple-model reference": f"{parameters.rotating_frame_phase_rate:.6g} rad/TU",
            "Interpretation": "aligned at first node, then compared to circular phase drift",
        },
        {
            "Metric": "Nominal tidal acceleration delta",
            "Horizons-derived range": f"max {float(df['nominal_tidal_accel_delta_norm'].max()):.6g}",
            "Simple-model reference": "bicircular solar tide",
            "Interpretation": "force contrast at persisted nominal trajectory nodes",
        },
        {
            "Metric": "Representative branch tidal acceleration delta",
            "Horizons-derived range": f"max {float(df['representative_branch_tidal_accel_delta_norm'].max()):.6g}",
            "Simple-model reference": "bicircular solar tide",
            "Interpretation": "force contrast at the accepted branch with largest recorded CR3BP terminal error",
        },
    ]
    pd.DataFrame(rows).to_latex(path, index=False, escape=True)
    return path


def _build_contrast_rows(
    *,
    cache: dict[str, object],
    states,
    cfg,
    nominal_controls: np.ndarray,
    branch_controls: np.ndarray,
    branch_mask_index: int,
    parameters: SolarTidalParameters,
    reference_distance_km: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    moon_samples = samples_from_cache(cache, "moon_geocentric")
    sun_samples = samples_from_cache(cache, "sun_geocentric")
    expected_nodes = int(cfg.n_segments) + 1
    if len(moon_samples) != expected_nodes:
        raise RuntimeError(f"Horizons cache has {len(moon_samples)} nodes, expected {expected_nodes}")
    geometry = horizons_rotating_geometry(
        moon_samples=moon_samples,
        sun_samples=sun_samples,
        mu=float(cfg.mu),
        reference_distance_km=float(reference_distance_km),
    )
    node_times = canonical_node_times(cfg.tf, cfg.n_segments)
    cached_jd = np.asarray([sample.jd_tdb for sample in moon_samples], dtype=float)
    if not np.allclose(np.asarray(geometry["jd_tdb"], dtype=float), cached_jd, rtol=0.0, atol=5.0e-10):
        raise RuntimeError("internal JD mismatch in Horizons geometry")

    nominal_nodes = _segment_node_history(states.initial, nominal_controls, cfg)
    branch_nodes = _segment_node_history(states.initial, branch_controls, cfg)
    sun_vectors = np.asarray(geometry["sun_barycenter_rotating_lu"], dtype=float)
    horizons_nominal_accel = solar_tidal_acceleration_from_sun_vector(
        nominal_nodes[:, :3],
        sun_vectors,
        sun_mu_ratio=parameters.sun_mu_ratio,
    )
    horizons_branch_accel = solar_tidal_acceleration_from_sun_vector(
        branch_nodes[:, :3],
        sun_vectors,
        sun_mu_ratio=parameters.sun_mu_ratio,
    )

    phase0 = float(np.asarray(geometry["sun_phase_rotating_rad"], dtype=float)[0])
    bicircular_phase = phase0 + float(parameters.rotating_frame_phase_rate) * node_times
    bicircular_nominal_accel = np.asarray(
        [
            solar_tidal_acceleration(
                nominal_nodes[i, :3],
                float(node_times[i]),
                phase_rad=phase0,
                parameters=parameters,
            )
            for i in range(len(node_times))
        ],
        dtype=float,
    )
    bicircular_branch_accel = np.asarray(
        [
            solar_tidal_acceleration(
                branch_nodes[i, :3],
                float(node_times[i]),
                phase_rad=phase0,
                parameters=parameters,
            )
            for i in range(len(node_times))
        ],
        dtype=float,
    )

    nominal_h_norm = np.linalg.norm(horizons_nominal_accel, axis=1)
    nominal_b_norm = np.linalg.norm(bicircular_nominal_accel, axis=1)
    nominal_delta = np.linalg.norm(horizons_nominal_accel - bicircular_nominal_accel, axis=1)
    branch_h_norm = np.linalg.norm(horizons_branch_accel, axis=1)
    branch_b_norm = np.linalg.norm(bicircular_branch_accel, axis=1)
    branch_delta = np.linalg.norm(horizons_branch_accel - bicircular_branch_accel, axis=1)

    sun_distance = np.asarray(geometry["sun_distance_lu"], dtype=float)
    em_distance_ratio = np.asarray(geometry["em_distance_ratio_to_mean"], dtype=float)
    em_distance_ratio_to_reference = np.asarray(geometry["em_distance_ratio_to_reference"], dtype=float)
    em_rate_ratio = np.asarray(geometry["em_angular_rate_ratio_to_mean"], dtype=float)
    sun_phase = np.asarray(geometry["sun_phase_rotating_rad"], dtype=float)
    phase_delta = _phase_delta_degrees(sun_phase, bicircular_phase)
    rows: list[dict[str, object]] = []
    for i, moon_sample in enumerate(moon_samples):
        rows.append(
            {
                "node_index": int(i),
                "canonical_time": float(node_times[i]),
                "jd_tdb": float(moon_sample.jd_tdb),
                "calendar_tdb": moon_sample.calendar_tdb,
                "em_distance_km": float(np.asarray(geometry["em_distance_km"], dtype=float)[i]),
                "em_distance_ratio_to_mean": float(em_distance_ratio[i]),
                "em_distance_delta_percent_vs_mean": float((em_distance_ratio[i] - 1.0) * 100.0),
                "em_angular_rate_ratio_to_mean": float(em_rate_ratio[i]),
                "em_angular_rate_delta_percent_vs_mean": float((em_rate_ratio[i] - 1.0) * 100.0),
                "sun_distance_lu": float(sun_distance[i]),
                "sun_distance_delta_percent_vs_bicircular": float(
                    (sun_distance[i] / float(parameters.sun_distance_lu) - 1.0) * 100.0
                ),
                "sun_phase_rotating_rad": float(sun_phase[i]),
                "bicircular_aligned_phase_rad": float(bicircular_phase[i]),
                "sun_phase_delta_degrees_wrapped": float(phase_delta[i]),
                "nominal_horizons_tidal_accel_norm": float(nominal_h_norm[i]),
                "nominal_bicircular_tidal_accel_norm": float(nominal_b_norm[i]),
                "nominal_tidal_accel_delta_norm": float(nominal_delta[i]),
                "nominal_tidal_accel_delta_fraction": float(_safe_fraction(nominal_delta, nominal_h_norm)[i]),
                "representative_branch_mask_index": int(branch_mask_index),
                "representative_branch_horizons_tidal_accel_norm": float(branch_h_norm[i]),
                "representative_branch_bicircular_tidal_accel_norm": float(branch_b_norm[i]),
                "representative_branch_tidal_accel_delta_norm": float(branch_delta[i]),
                "representative_branch_tidal_accel_delta_fraction": float(_safe_fraction(branch_delta, branch_h_norm)[i]),
                "contrast_semantics": (
                    "cached Horizons-derived Earth/Moon/Sun geometry force-model contrast only; "
                    "no high-fidelity replay, no SPICE validation, no retuning"
                ),
            }
        )
    summary = {
        "node_count": int(len(rows)),
        "mean_em_distance_km": float(np.asarray(geometry["mean_em_distance_km"])),
        "reference_distance_km": float(np.asarray(geometry["reference_distance_km"])),
        "mean_em_distance_to_reference_ratio": float(
            float(np.asarray(geometry["mean_em_distance_km"])) / float(reference_distance_km)
        ),
        "em_distance_ratio_to_reference_min": float(em_distance_ratio_to_reference.min()),
        "em_distance_ratio_to_reference_max": float(em_distance_ratio_to_reference.max()),
        "mean_em_angular_rate_rad_s": float(np.asarray(geometry["mean_em_angular_rate_rad_s"])),
        "em_distance_ratio_min": float(em_distance_ratio.min()),
        "em_distance_ratio_max": float(em_distance_ratio.max()),
        "em_distance_delta_percent_max_abs": float(np.max(np.abs((em_distance_ratio - 1.0) * 100.0))),
        "em_angular_rate_ratio_min": float(em_rate_ratio.min()),
        "em_angular_rate_ratio_max": float(em_rate_ratio.max()),
        "em_angular_rate_delta_percent_max_abs": float(np.max(np.abs((em_rate_ratio - 1.0) * 100.0))),
        "sun_distance_lu_min": float(sun_distance.min()),
        "sun_distance_lu_max": float(sun_distance.max()),
        "sun_distance_delta_percent_vs_bicircular_max_abs": float(
            np.max(np.abs((sun_distance / float(parameters.sun_distance_lu) - 1.0) * 100.0))
        ),
        "sun_phase_delta_degrees_wrapped_max_abs": float(np.max(np.abs(phase_delta))),
        "nominal_tidal_accel_delta_norm_max": float(nominal_delta.max()),
        "nominal_tidal_accel_delta_fraction_max": float(_safe_fraction(nominal_delta, nominal_h_norm).max()),
        "representative_branch_mask_index": int(branch_mask_index),
        "representative_branch_tidal_accel_delta_norm_max": float(branch_delta.max()),
        "representative_branch_tidal_accel_delta_fraction_max": float(
            _safe_fraction(branch_delta, branch_h_norm).max()
        ),
    }
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS), summary


def run(args: argparse.Namespace) -> pd.DataFrame:
    config, case_config, states, cfg, config_path, source_states, tail_csv, input_rows = _load_case_context(args)
    del config, case_config
    if bool(args.refresh_cache):
        cache_path = _refresh_cache(args, cfg=cfg)
    else:
        cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    if not cache_path.is_file():
        raise RuntimeError(
            f"Horizons cache not found: {cache_path}. Re-run with --refresh-cache to regenerate it."
        )
    cache = load_horizons_cache(cache_path)
    validate_horizons_cache_compatibility(
        cache,
        start_jd_tdb=float(args.start_jd_tdb),
        tf=float(cfg.tf),
        n_segments=int(cfg.n_segments),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
        reference_distance_km=float(args.reference_distance_km),
    )

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    row = _tail_replay_row(input_rows)
    manifest, manifest_path, manifest_sha = _read_json_verified(
        row["branch_control_manifest_path"],
        row.get("branch_control_manifest_sha256"),
    )
    nominal, nominal_path, nominal_sha = _read_json_verified(
        manifest["nominal_control_path"],
        manifest.get("nominal_control_sha256"),
    )
    branch_entry = _selected_branch_sidecar(manifest)
    branch, branch_path, branch_sha = _read_json_verified(branch_entry["path"], branch_entry.get("sha256"))

    recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
    if recorded_target.shape != np.asarray(states.target, dtype=float).shape:
        raise RuntimeError("manifest target_state shape mismatch")
    target_state_max_abs_delta = float(np.max(np.abs(np.asarray(states.target, dtype=float) - recorded_target)))
    if target_state_max_abs_delta > float(args.baseline_tolerance):
        raise RuntimeError(f"target_state mismatch: max abs delta {target_state_max_abs_delta}")

    parameters = SolarTidalParameters(
        sun_distance_lu=float(args.bicircular_sun_distance_lu),
        sun_mu_ratio=float(args.bicircular_sun_mu_ratio),
        sun_inertial_angular_rate_ratio=float(args.bicircular_sun_inertial_angular_rate_ratio),
    )
    df, contrast_summary = _build_contrast_rows(
        cache=cache,
        states=states,
        cfg=cfg,
        nominal_controls=np.asarray(nominal["controls"], dtype=float),
        branch_controls=np.asarray(branch["branch_controls"], dtype=float),
        branch_mask_index=int(branch["mask_index"]),
        parameters=parameters,
        reference_distance_km=float(args.reference_distance_km),
    )
    csv_path = results_dir / CONTRAST_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")
    table_path = _write_table(
        df,
        tables_dir,
        parameters=parameters,
        reference_distance_km=float(args.reference_distance_km),
    )

    input_artifacts: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path in (config_path, source_states, tail_csv, cache_path, manifest_path, nominal_path, branch_path):
        _append_artifact(input_artifacts, seen, path)

    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "optimization_rerun": False,
        "uses_recorded_artifacts_only": True,
        "uses_cached_horizons_vectors": True,
        "runtime_network_dependency": bool(args.refresh_cache),
        "force_model_contrast_only": True,
        "high_fidelity_validation": False,
        "spice_ephemeris_validation": False,
        "accepted_control_high_fidelity_replay": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "cache": {
            "path": _relative_or_absolute(cache_path),
            "sha256": _sha256(cache_path),
            "cache_type": cache.get("cache_type"),
            "metadata": cache.get("metadata", {}),
        },
        "bicircular_reference_parameters": parameters.as_dict(),
        "length_unit_scaling": {
            "reference_distance_km": float(args.reference_distance_km),
            "sun_vector_lu_scaling": (
                "Cached Sun barycentric rotating vectors are divided by reference_distance_km."
            ),
            "trajectory_lu_scaling": (
                "Persisted CR3BP trajectory states are interpreted in the same fixed length unit."
            ),
            "earth_moon_distance_ratio_to_mean": (
                "The reported Earth-Moon distance ratio to the window mean is diagnostic only and is not used "
                "to scale Sun vectors or tidal accelerations."
            ),
        },
        "case": {
            "suite_case_id": TAIL_COAST_COMBINED_CASE,
            "transfer_time": float(cfg.tf),
            "segments": int(cfg.n_segments),
            "substeps_per_segment": int(cfg.substeps),
            "target_mode": "catalog_dro_phase",
            "target_state_max_abs_delta_from_manifest": target_state_max_abs_delta,
        },
        "representative_states": {
            "nominal_controls_path": _relative_or_absolute(nominal_path),
            "nominal_controls_sha256": nominal_sha,
            "branch_selection_policy": "accepted branch sidecar with largest recorded CR3BP terminal_error",
            "branch_mask_index": int(branch["mask_index"]),
            "branch_order": int(branch["branch_order"]),
            "branch_terminal_error": float(branch["terminal_error"]),
            "branch_controls_path": _relative_or_absolute(branch_path),
            "branch_controls_sha256": branch_sha,
            "manifest_path": _relative_or_absolute(manifest_path),
            "manifest_sha256": manifest_sha,
        },
        "geometry_summary": {
            key: value
            for key, value in contrast_summary.items()
            if key
            in {
                "node_count",
                "mean_em_distance_km",
                "reference_distance_km",
                "mean_em_distance_to_reference_ratio",
                "em_distance_ratio_to_reference_min",
                "em_distance_ratio_to_reference_max",
                "mean_em_angular_rate_rad_s",
                "em_distance_ratio_min",
                "em_distance_ratio_max",
                "em_distance_delta_percent_max_abs",
                "em_angular_rate_ratio_min",
                "em_angular_rate_ratio_max",
                "em_angular_rate_delta_percent_max_abs",
                "sun_distance_lu_min",
                "sun_distance_lu_max",
                "sun_distance_delta_percent_vs_bicircular_max_abs",
                "sun_phase_delta_degrees_wrapped_max_abs",
            }
        },
        "solar_tidal_acceleration_summary": {
            key: value
            for key, value in contrast_summary.items()
            if key
            in {
                "nominal_tidal_accel_delta_norm_max",
                "nominal_tidal_accel_delta_fraction_max",
                "representative_branch_mask_index",
                "representative_branch_tidal_accel_delta_norm_max",
                "representative_branch_tidal_accel_delta_fraction_max",
            }
        },
        "input_artifacts": input_artifacts,
        "artifacts": {
            "horizons_ephemeris_force_model_contrast_csv": _relative_or_absolute(csv_path),
            "horizons_ephemeris_force_model_contrast_metadata_json": _relative_or_absolute(
                results_dir / CONTRAST_METADATA_NAME
            ),
            "horizons_ephemeris_force_model_contrast_table_tex": _relative_or_absolute(table_path),
        },
        "interpretation_limits": [
            "This package is an offline force-model contrast between simple bicircular assumptions and cached Horizons-derived geometry.",
            "It is not SPICE validation, high-fidelity flight validation, accepted-control high-fidelity replay, production solver parity, or fuel optimality evidence.",
            "Accepted controls are sampled only to evaluate representative solar-tidal acceleration differences at CR3BP trajectory nodes; controls are not retuned or repropagated under Horizons dynamics.",
            "The fixed 2026-Jan-01 TDB cache epoch is representative for stress contrast only and is not a mission epoch.",
        ],
    }
    _write_json(results_dir / CONTRAST_METADATA_NAME, metadata)
    print(
        "Completed Horizons ephemeris force-model contrast "
        f"with {int(len(df))} nodes, "
        f"EM distance ratio range={contrast_summary['em_distance_ratio_min']:.6g}--"
        f"{contrast_summary['em_distance_ratio_max']:.6g}, "
        f"max nominal tidal delta={contrast_summary['nominal_tidal_accel_delta_norm_max']:.3e}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare simple bicircular assumptions against cached JPL Horizons Earth/Moon/Sun geometry."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-cache", action="store_true", help="Fetch and rewrite the Horizons cache before running.")
    parser.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--start-jd-tdb", type=float, default=DEFAULT_CACHE_START_JD_TDB)
    parser.add_argument("--canonical-time-unit-seconds", type=float, default=DEFAULT_CANONICAL_TIME_UNIT_SECONDS)
    parser.add_argument("--reference-distance-km", type=float, default=DEFAULT_REFERENCE_DISTANCE_KM)
    parser.add_argument("--bicircular-sun-distance-lu", type=float, default=SolarTidalParameters().sun_distance_lu)
    parser.add_argument("--bicircular-sun-mu-ratio", type=float, default=SolarTidalParameters().sun_mu_ratio)
    parser.add_argument(
        "--bicircular-sun-inertial-angular-rate-ratio",
        type=float,
        default=SolarTidalParameters().sun_inertial_angular_rate_ratio,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
