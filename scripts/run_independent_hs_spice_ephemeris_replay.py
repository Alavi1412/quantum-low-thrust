"""Replay independent-HS retuned controls with SPICE-derived ephemeris vectors.

This postprocessor reads the existing four-epoch cached-Horizons point-mass
retuning package and replays the already-retuned endpoint-plus-midpoint controls
under compact SPICE-derived Moon/Sun geocentric vector caches. It does not
retune controls or rerun trajectory optimization. The force model remains the
same Earth/Moon/Sun point-mass stress model used by the Horizons retuning
package; only the Moon/Sun vector source changes to SPICE-derived geometric
J2000 states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_independent_hs_horizons_multi_epoch_point_mass_retuning as multi_epoch
import run_independent_hs_horizons_point_mass_retuning as single_epoch
from qlt.ephemeris_contrast import (
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    HorizonsVectorSample,
    canonical_node_jds,
    canonical_node_times,
    load_spice_cache,
    sample_to_cache_row,
    spice_point_mass_profile_from_cache,
    validate_spice_cache_compatibility,
    horizons_point_mass_terminal_error,
)
from qlt.reporting import sanitize_json


DEFAULT_SOURCE_RESULTS_DIR = Path("data/results/independent_hs_horizons_multi_epoch_point_mass_retuning")
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_spice_ephemeris_replay")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_spice_ephemeris_replay")
DEFAULT_SPICE_CACHE_DIR = Path("data/cache/spice")
DEFAULT_KERNEL_DIR = Path("data/cache/spice/kernels")

REPLAY_CSV_NAME = "independent_hs_spice_ephemeris_replay.csv"
REPLAY_METADATA_NAME = "independent_hs_spice_ephemeris_replay_metadata.json"
REPLAY_TABLE_NAME = "independent_hs_spice_ephemeris_replay_table.tex"

SPICE_KERNELS = {
    "lsk": {
        "filename": "naif0012.tls",
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    },
    "spk": {
        "filename": "de442s.bsp",
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp",
    },
    "gm": {
        "filename": "gm_de440.tpc",
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
    },
}

REPLAY_COLUMNS = [
    "epoch_id",
    "epoch_label",
    "epoch_start_calendar_tdb",
    "epoch_start_jd_tdb",
    "epoch_order",
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "source_horizons_retuned_controls_path",
    "source_horizons_retuned_controls_sha256",
    "source_horizons_retuned_terminal_error",
    "spice_replay_terminal_error",
    "spice_replay_delta_from_horizons_retuned",
    "configured_threshold",
    "spice_replay_passes_configured_threshold",
    "source_horizons_retuned_passes_configured_threshold",
    "recorded_cr3bp_terminal_error",
    "source_horizons_replay_terminal_error",
    "source_horizons_replay_delta_from_recorded_cr3bp",
    "active_control_mask",
    "control_bound",
    "retuned_endpoint_norm_max",
    "retuned_midpoint_norm_max",
    "substeps_per_segment",
    "transfer_time",
    "spice_cache_path",
    "spice_cache_sha256",
    "spice_kernel_lsk_sha256",
    "spice_kernel_spk_sha256",
    "spice_kernel_gm_sha256",
    "no_retune_semantics",
]

LIMITATION_FLAGS = {
    "spice_ephemeris_validation": True,
    "retuning": False,
    "optimization_rerun": False,
    "high_fidelity_validation": False,
    "high_fidelity_flight_validation": False,
    "production_solver_parity_claim": False,
    "fuel_optimality_claim": False,
    "doi_claim": False,
    "quantum_advantage_claim": False,
}

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


def _resolve_existing_file(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.is_file():
        return cwd_path
    return ROOT / path


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
    return path if path.is_absolute() else ROOT / path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"source CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json_verified(path_text: object, expected_sha256: object | None = None) -> tuple[dict, Path, str]:
    path = _resolve_existing_path(path_text)
    if not path.is_file():
        raise RuntimeError(f"sidecar not found: {path}")
    actual = _sha256(path)
    expected = str(expected_sha256 or "").strip()
    if expected and expected != actual:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")
    return json.loads(path.read_text(encoding="utf-8")), path, actual


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blank_or_int(value: object) -> str | int:
    text = str(value or "").strip()
    if text == "":
        return ""
    return int(float(text))


def _float_from_row(row: dict[str, str], key: str) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        raise RuntimeError(f"source row missing numeric field {key}")
    return float(text)


def _cache_path_for_epoch(cache_dir: Path, epoch_id: str) -> Path:
    return cache_dir / f"independent_hs_phase_shift_{epoch_id}_spice_vectors.json"


def _download_kernel(url: str, path: Path, *, timeout_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=float(timeout_seconds)) as response:
        data = response.read()
    path.write_bytes(data)


def _load_spiceypy():
    try:
        import spiceypy as spice  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SpiceyPy is required only for --refresh-spice-cache. "
            "Install spiceypy or run without refresh to use committed compact caches."
        ) from exc
    return spice


def _kernel_paths(kernel_dir: Path) -> dict[str, Path]:
    return {
        key: kernel_dir / str(info["filename"])
        for key, info in SPICE_KERNELS.items()
    }


def _refresh_kernels(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    kernel_dir = _resolve_cache_path(args.kernel_dir)
    paths = _kernel_paths(kernel_dir)
    for key, path in paths.items():
        _download_kernel(
            str(SPICE_KERNELS[key]["url"]),
            path,
            timeout_seconds=float(args.download_timeout_seconds),
        )
    return {
        key: {
            "path": path,
            "filename": path.name,
            "url": str(SPICE_KERNELS[key]["url"]),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for key, path in paths.items()
    }


def _cleanup_kernel_dir(kernel_dir: Path) -> None:
    if not kernel_dir.exists():
        return
    for path in _kernel_paths(kernel_dir).values():
        path.unlink(missing_ok=True)
    try:
        if not any(kernel_dir.iterdir()):
            kernel_dir.rmdir()
    except OSError:
        pass


def _spice_calendar_label(spice, et: float, jd_tdb: float) -> str:
    try:
        return str(spice.timout(float(et), "YYYY-MON-DD HR:MN:SC.### TDB ::TDB")).strip()
    except Exception:
        return f"JD TDB {float(jd_tdb):.9f}"


def _state_sample_from_spice(spice, *, target: str, jd_tdb: float) -> HorizonsVectorSample:
    et = (float(jd_tdb) - 2451545.0) * 86400.0
    state, light_time = spice.spkezr(str(target), float(et), "J2000", "NONE", "EARTH")
    state_arr = np.asarray(state, dtype=float)
    position = state_arr[:3]
    velocity = state_arr[3:]
    distance = float(np.linalg.norm(position))
    range_rate = float(np.dot(position, velocity) / max(distance, 1.0e-15))
    return HorizonsVectorSample(
        jd_tdb=float(jd_tdb),
        calendar_tdb=_spice_calendar_label(spice, et, float(jd_tdb)),
        position_km=position,
        velocity_km_s=velocity,
        light_time_s=float(light_time),
        range_km=distance,
        range_rate_km_s=range_rate,
    )


def _spice_cache_payload(
    args: argparse.Namespace,
    *,
    spec: multi_epoch.EpochSpec,
    cfg,
    kernel_info: dict[str, dict[str, object]],
) -> dict[str, object]:
    spice = _load_spiceypy()
    spice.kclear()
    try:
        for key in ("lsk", "spk", "gm"):
            spice.furnsh(str(kernel_info[key]["path"]))
        node_times = canonical_node_times(float(cfg.tf), int(cfg.n_segments))
        node_jds = canonical_node_jds(
            start_jd_tdb=float(spec.start_jd_tdb),
            tf=float(cfg.tf),
            n_segments=int(cfg.n_segments),
            canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
        )
        moon_samples = [
            _state_sample_from_spice(spice, target="MOON", jd_tdb=float(jd))
            for jd in node_jds
        ]
        sun_samples = [
            _state_sample_from_spice(spice, target="SUN", jd_tdb=float(jd))
            for jd in node_jds
        ]
        kernel_public = {
            key: {
                "filename": str(info["filename"]),
                "url": str(info["url"]),
                "sha256": str(info["sha256"]),
                "bytes": int(info["bytes"]),
            }
            for key, info in kernel_info.items()
        }
        limitations = [
            "SPICE-derived vectors are compact cached geometric states from spkezr target states.",
            "States are Moon and Sun relative to Earth in J2000 with aberration correction NONE.",
            "The replay still uses the paper's Earth/Moon/Sun point-mass stress model and normalized CR3BP target/scales.",
            "This cache is not a full high-fidelity force model, flight-validation product, or production solver parity check.",
        ]
        return {
            "schema_version": 1,
            "cache_type": "spice_geocentric_moon_sun_vectors",
            "metadata": {
                "representative_epoch_note": (
                    f"Fixed representative {spec.label} TDB epoch for independent-HS SPICE-derived "
                    "Moon/Sun vector replay under the paper point-mass stress model; it is not a mission epoch."
                ),
                "epoch_id": spec.epoch_id,
                "epoch_label": spec.label,
                "start_jd_tdb": float(spec.start_jd_tdb),
                "start_calendar_tdb": spec.start_calendar_tdb,
                "canonical_transfer_time": float(cfg.tf),
                "segments": int(cfg.n_segments),
                "canonical_node_times": [float(value) for value in node_times],
                "node_jd_tdb": [float(value) for value in node_jds],
                "canonical_time_unit_seconds": float(args.canonical_time_unit_seconds),
                "reference_distance_km": float(args.reference_distance_km),
                "frame": "J2000",
                "observer": "EARTH",
                "targets": ["MOON", "SUN"],
                "aberration_correction": "NONE",
                "spkezr_call": "spkezr(target, et, 'J2000', 'NONE', 'EARTH')",
                "kernel_urls": {key: str(info["url"]) for key, info in kernel_public.items()},
                "kernel_sha256": {key: str(info["sha256"]) for key, info in kernel_public.items()},
                "source_limitations": limitations,
            },
            "spice": {
                "spiceypy_version": str(getattr(spice, "__version__", "unknown")),
                "cspice_toolkit_version": str(spice.tkvrsn("TOOLKIT")),
            },
            "kernels": kernel_public,
            "bodies": {
                "moon_geocentric": {
                    "target": "MOON",
                    "observer": "EARTH",
                    "frame": "J2000",
                    "aberration_correction": "NONE",
                    "samples": [sample_to_cache_row(sample) for sample in moon_samples],
                },
                "sun_geocentric": {
                    "target": "SUN",
                    "observer": "EARTH",
                    "frame": "J2000",
                    "aberration_correction": "NONE",
                    "samples": [sample_to_cache_row(sample) for sample in sun_samples],
                },
            },
            "interpretation_limits": limitations,
        }
    finally:
        spice.kclear()


def _refresh_spice_caches(args: argparse.Namespace, *, cfg) -> dict[str, dict[str, object]]:
    kernel_info = _refresh_kernels(args)
    cache_dir = _resolve_cache_path(args.spice_cache_dir)
    cache_info: dict[str, dict[str, object]] = {}
    try:
        for spec in multi_epoch.DEFAULT_EPOCHS:
            cache_path = _cache_path_for_epoch(cache_dir, spec.epoch_id)
            cache = _spice_cache_payload(args, spec=spec, cfg=cfg, kernel_info=kernel_info)
            _write_json(cache_path, cache)
            cache_info[spec.epoch_id] = {
                "path": cache_path,
                "sha256": _sha256(cache_path),
                "refreshed_this_run": True,
            }
    finally:
        if not bool(args.keep_kernels):
            _cleanup_kernel_dir(_resolve_cache_path(args.kernel_dir))
    return cache_info


def _ensure_spice_caches(args: argparse.Namespace, *, cfg) -> dict[str, dict[str, object]]:
    if bool(args.refresh_spice_cache):
        _refresh_spice_caches(args, cfg=cfg)

    cache_dir = _resolve_cache_path(args.spice_cache_dir)
    cache_info: dict[str, dict[str, object]] = {}
    for spec in multi_epoch.DEFAULT_EPOCHS:
        cache_path = _cache_path_for_epoch(cache_dir, spec.epoch_id)
        if not cache_path.is_file():
            raise RuntimeError(
                f"SPICE vector cache not found for {spec.label}: {cache_path}. "
                "Re-run with --refresh-spice-cache to derive it from NAIF kernels."
            )
        cache_info[spec.epoch_id] = {
            "path": cache_path,
            "sha256": _sha256(cache_path),
            "refreshed_this_run": bool(args.refresh_spice_cache),
        }
    return cache_info


def _first_objective_config(args: argparse.Namespace):
    return multi_epoch._first_objective_config(args)


def _case_context(args: argparse.Namespace, *, case_id: str):
    config_path = _resolve_existing_file(args.config)
    source_states = _resolve_existing_file(args.source_states)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = single_epoch._case_by_id(config)
    if case_id not in cases:
        raise RuntimeError(f"configured suite is missing selected case {case_id}")
    case_config = single_epoch._case_config(config, cases[case_id])
    states = single_epoch.load_configured_states(Path.cwd(), case_config, source_states)
    cfg = single_epoch.make_objective_config(case_config, states.mu)
    return config_path, source_states, case_config, states, cfg


def _kernel_hashes(cache: dict[str, object]) -> dict[str, str]:
    kernels = cache.get("kernels", {})
    if not isinstance(kernels, dict):
        raise RuntimeError("SPICE cache missing kernels")
    return {
        key: str(kernels.get(key, {}).get("sha256", "")) if isinstance(kernels.get(key), dict) else ""
        for key in ("lsk", "spk", "gm")
    }


def _row_for_replay(
    *,
    source_row: dict[str, str],
    spice_error: float,
    cache_path: Path,
    cache_sha256: str,
    kernel_hashes: dict[str, str],
) -> dict[str, object]:
    source_error = _float_from_row(source_row, "point_mass_retuned_terminal_error")
    threshold = _float_from_row(source_row, "configured_threshold")
    return {
        "epoch_id": source_row["epoch_id"],
        "epoch_label": source_row["epoch_label"],
        "epoch_start_calendar_tdb": source_row["epoch_start_calendar_tdb"],
        "epoch_start_jd_tdb": _float_from_row(source_row, "epoch_start_jd_tdb"),
        "epoch_order": int(float(source_row["epoch_order"])),
        "case_id": source_row["case_id"],
        "record_type": source_row["record_type"],
        "branch_order": _blank_or_int(source_row.get("branch_order")),
        "mask_index": _blank_or_int(source_row.get("mask_index")),
        "outage_mask": source_row.get("outage_mask", ""),
        "source_horizons_retuned_controls_path": source_row["retuned_controls_path"],
        "source_horizons_retuned_controls_sha256": source_row["retuned_controls_sha256"],
        "source_horizons_retuned_terminal_error": source_error,
        "spice_replay_terminal_error": float(spice_error),
        "spice_replay_delta_from_horizons_retuned": abs(float(spice_error) - source_error),
        "configured_threshold": threshold,
        "spice_replay_passes_configured_threshold": bool(float(spice_error) <= threshold),
        "source_horizons_retuned_passes_configured_threshold": _bool_value(
            source_row.get("point_mass_retuned_passes_configured_threshold", "")
        ),
        "recorded_cr3bp_terminal_error": _float_from_row(source_row, "recorded_cr3bp_terminal_error"),
        "source_horizons_replay_terminal_error": _float_from_row(source_row, "point_mass_replay_terminal_error"),
        "source_horizons_replay_delta_from_recorded_cr3bp": _float_from_row(
            source_row,
            "point_mass_replay_delta_from_recorded_cr3bp",
        ),
        "active_control_mask": source_row.get("active_control_mask", ""),
        "control_bound": _float_from_row(source_row, "control_bound"),
        "retuned_endpoint_norm_max": _float_from_row(source_row, "retuned_endpoint_norm_max"),
        "retuned_midpoint_norm_max": _float_from_row(source_row, "retuned_midpoint_norm_max"),
        "substeps_per_segment": int(float(source_row["substeps_per_segment"])),
        "transfer_time": _float_from_row(source_row, "transfer_time"),
        "spice_cache_path": _relative_or_absolute(cache_path),
        "spice_cache_sha256": cache_sha256,
        "spice_kernel_lsk_sha256": kernel_hashes["lsk"],
        "spice_kernel_spk_sha256": kernel_hashes["spk"],
        "spice_kernel_gm_sha256": kernel_hashes["gm"],
        "no_retune_semantics": (
            "No optimization or retuning is rerun: this row replays the already-retuned "
            "cached-Horizons endpoint-plus-midpoint controls under SPICE-derived Moon/Sun vectors."
        ),
    }


def _epoch_summary(epoch_id: str, epoch_rows: list[dict[str, object]]) -> dict[str, object]:
    nominal = [row for row in epoch_rows if row["record_type"] == "nominal"]
    branch = [row for row in epoch_rows if row["record_type"] == "branch"]
    if len(nominal) != 1:
        raise RuntimeError(f"expected one nominal SPICE replay row for {epoch_id}, got {len(nominal)}")
    if len(branch) != 8:
        raise RuntimeError(f"expected eight branch SPICE replay rows for {epoch_id}, got {len(branch)}")
    return {
        "epoch_id": epoch_id,
        "epoch_label": str(nominal[0]["epoch_label"]),
        "start_calendar_tdb": str(nominal[0]["epoch_start_calendar_tdb"]),
        "start_jd_tdb": float(nominal[0]["epoch_start_jd_tdb"]),
        "row_count": len(epoch_rows),
        "nominal_row_count": len(nominal),
        "branch_row_count": len(branch),
        "nominal_spice_replay_error": float(max(row["spice_replay_terminal_error"] for row in nominal)),
        "branch_spice_replay_worst_error": float(max(row["spice_replay_terminal_error"] for row in branch)),
        "nominal_spice_replay_pass": bool(all(row["spice_replay_passes_configured_threshold"] for row in nominal)),
        "branch_spice_replay_pass_count": int(sum(bool(row["spice_replay_passes_configured_threshold"]) for row in branch)),
        "source_horizons_retuned_nominal_error": float(
            max(row["source_horizons_retuned_terminal_error"] for row in nominal)
        ),
        "source_horizons_retuned_branch_worst_error": float(
            max(row["source_horizons_retuned_terminal_error"] for row in branch)
        ),
        "max_abs_delta_from_horizons_retuned": float(
            max(row["spice_replay_delta_from_horizons_retuned"] for row in epoch_rows)
        ),
        "cache_path": str(nominal[0]["spice_cache_path"]),
        "cache_sha256": str(nominal[0]["spice_cache_sha256"]),
        "kernel_sha256": {
            "lsk": str(nominal[0]["spice_kernel_lsk_sha256"]),
            "spk": str(nominal[0]["spice_kernel_spk_sha256"]),
            "gm": str(nominal[0]["spice_kernel_gm_sha256"]),
        },
    }


def _overall_summary(epoch_summaries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "epoch_count": len(epoch_summaries),
        "row_count": int(sum(int(row["row_count"]) for row in epoch_summaries)),
        "nominal_row_count": int(sum(int(row["nominal_row_count"]) for row in epoch_summaries)),
        "branch_row_count": int(sum(int(row["branch_row_count"]) for row in epoch_summaries)),
        "nominal_spice_replay_worst_over_epochs": float(
            max(float(row["nominal_spice_replay_error"]) for row in epoch_summaries)
        ),
        "branch_spice_replay_worst_over_epochs": float(
            max(float(row["branch_spice_replay_worst_error"]) for row in epoch_summaries)
        ),
        "branch_spice_replay_pass_count_total": int(
            sum(int(row["branch_spice_replay_pass_count"]) for row in epoch_summaries)
        ),
        "branch_row_count_total": int(sum(int(row["branch_row_count"]) for row in epoch_summaries)),
        "nominal_spice_replay_pass_count_total": int(
            sum(1 for row in epoch_summaries if bool(row["nominal_spice_replay_pass"]))
        ),
        "max_abs_delta_from_horizons_retuned": float(
            max(float(row["max_abs_delta_from_horizons_retuned"]) for row in epoch_summaries)
        ),
        "source_horizons_retuned_nominal_worst_over_epochs": float(
            max(float(row["source_horizons_retuned_nominal_error"]) for row in epoch_summaries)
        ),
        "source_horizons_retuned_branch_worst_over_epochs": float(
            max(float(row["source_horizons_retuned_branch_worst_error"]) for row in epoch_summaries)
        ),
    }


def _write_table(epoch_summaries: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / REPLAY_TABLE_NAME
    rows = []
    for row in epoch_summaries:
        rows.append(
            {
                "Epoch": row["epoch_label"],
                "SPICE nominal": float(row["nominal_spice_replay_error"]),
                "SPICE branch worst": float(row["branch_spice_replay_worst_error"]),
                "SPICE branch pass": f"{int(row['branch_spice_replay_pass_count'])}/{int(row['branch_row_count'])}",
                "Horizons-retuned nominal": float(row["source_horizons_retuned_nominal_error"]),
                "Horizons-retuned branch": float(row["source_horizons_retuned_branch_worst_error"]),
                "Max delta": float(row["max_abs_delta_from_horizons_retuned"]),
                "Cache SHA-256": str(row["cache_sha256"])[:12],
            }
        )
    pd.DataFrame(rows).to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


def run(args: argparse.Namespace) -> pd.DataFrame:
    cfg_for_cache = _first_objective_config(args)
    cache_info = _ensure_spice_caches(args, cfg=cfg_for_cache)
    source_results_dir = _resolve_existing_path(args.source_results_dir)
    source_csv = source_results_dir / multi_epoch.MULTI_EPOCH_CSV_NAME
    source_metadata_path = source_results_dir / multi_epoch.MULTI_EPOCH_METADATA_NAME
    source_rows = _read_csv_rows(source_csv)
    if len(source_rows) != 36:
        raise RuntimeError(f"expected 36 source retuned rows, got {len(source_rows)}")

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    replay_rows: list[dict[str, object]] = []
    input_artifacts: list[dict[str, object]] = [
        {
            "path": _relative_or_absolute(source_csv),
            "sha256": _sha256(source_csv),
            "bytes": source_csv.stat().st_size,
        }
    ]
    if source_metadata_path.is_file():
        input_artifacts.append(
            {
                "path": _relative_or_absolute(source_metadata_path),
                "sha256": _sha256(source_metadata_path),
                "bytes": source_metadata_path.stat().st_size,
            }
        )

    context_cache: dict[str, tuple[object, object, object]] = {}
    for spec in multi_epoch.DEFAULT_EPOCHS:
        info = cache_info[spec.epoch_id]
        cache_path = Path(str(info["path"]))
        cache_sha = str(info["sha256"])
        cache = load_spice_cache(cache_path)
        profile = spice_point_mass_profile_from_cache(cache)
        kernel_hashes = _kernel_hashes(cache)
        epoch_source_rows = [row for row in source_rows if row.get("epoch_id") == spec.epoch_id]
        if len(epoch_source_rows) != 9:
            raise RuntimeError(f"expected 9 source rows for {spec.epoch_id}, got {len(epoch_source_rows)}")

        input_artifacts.append(
            {
                "path": _relative_or_absolute(cache_path),
                "sha256": cache_sha,
                "bytes": cache_path.stat().st_size,
            }
        )

        for source_row in epoch_source_rows:
            case_id = str(source_row["case_id"])
            if case_id not in context_cache:
                config_path, source_states, _case_config, states, cfg = _case_context(args, case_id=case_id)
                context_cache[case_id] = (states, cfg, source_states)
                for path in (config_path, source_states):
                    artifact = {
                        "path": _relative_or_absolute(path),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    if artifact not in input_artifacts:
                        input_artifacts.append(artifact)
            states, cfg, _source_states = context_cache[case_id]
            validate_spice_cache_compatibility(
                cache,
                start_jd_tdb=float(spec.start_jd_tdb),
                tf=float(cfg.tf),
                n_segments=int(cfg.n_segments),
                canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
                reference_distance_km=float(args.reference_distance_km),
            )
            sidecar, sidecar_path, sidecar_sha = _read_json_verified(
                source_row["retuned_controls_path"],
                source_row.get("retuned_controls_sha256"),
            )
            endpoint = np.asarray(sidecar["retuned_endpoint_controls"], dtype=float)
            midpoint = np.asarray(sidecar["retuned_midpoint_controls"], dtype=float)
            spice_error = horizons_point_mass_terminal_error(
                np.asarray(states.initial, dtype=float),
                np.asarray(states.target, dtype=float),
                endpoint,
                cfg.mu,
                cfg.tf,
                cfg.substeps,
                profile=profile,
                midpoint_controls=midpoint,
                position_scale=cfg.position_scale,
                velocity_scale=cfg.velocity_scale,
                reference_distance_km=float(args.reference_distance_km),
                canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
            )
            replay_rows.append(
                _row_for_replay(
                    source_row=source_row,
                    spice_error=spice_error,
                    cache_path=cache_path,
                    cache_sha256=cache_sha,
                    kernel_hashes=kernel_hashes,
                )
            )
            artifact = {
                "path": _relative_or_absolute(sidecar_path),
                "sha256": sidecar_sha,
                "bytes": sidecar_path.stat().st_size,
            }
            if artifact not in input_artifacts:
                input_artifacts.append(artifact)

    df = pd.DataFrame(replay_rows, columns=REPLAY_COLUMNS)
    csv_path = results_dir / REPLAY_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")

    epoch_summaries = [
        _epoch_summary(spec.epoch_id, [row for row in replay_rows if row["epoch_id"] == spec.epoch_id])
        for spec in multi_epoch.DEFAULT_EPOCHS
    ]
    overall = _overall_summary(epoch_summaries)
    table_path = _write_table(epoch_summaries, tables_dir)
    metadata_path = results_dir / REPLAY_METADATA_NAME
    limitations = [
        "This package replays controls already retuned under cached-Horizons point-mass dynamics; it does not retune.",
        "SPICE is used only to derive compact Moon/Sun geocentric vector caches on the canonical node grid.",
        "Propagation remains an Earth/Moon/Sun point-mass stress model with normalized CR3BP target/scales/thresholds.",
        "The package is not full high-fidelity propagation, flight validation, production solver parity, fuel optimality, DOI evidence, or quantum evidence.",
    ]
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "nominal_row_count": int((df["record_type"] == "nominal").sum()),
        "branch_row_count": int((df["record_type"] == "branch").sum()),
        "runtime_network_dependency": bool(args.refresh_spice_cache),
        "uses_committed_spice_vector_caches": not bool(args.refresh_spice_cache),
        "spice_derived_ephemeris_replay": True,
        "spice_point_mass_replay": True,
        "source_horizons_retuned_controls_replayed": True,
        "no_retune_semantics": True,
        **LIMITATION_FLAGS,
        "epochs": epoch_summaries,
        "overall_summary": overall,
        "cache_sha256_by_epoch": {
            row["epoch_id"]: row["cache_sha256"]
            for row in epoch_summaries
        },
        "kernel_sha256_by_epoch": {
            row["epoch_id"]: row["kernel_sha256"]
            for row in epoch_summaries
        },
        "spice_kernel_urls": {
            key: str(info["url"])
            for key, info in SPICE_KERNELS.items()
        },
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_spice_ephemeris_replay_csv": _relative_or_absolute(csv_path),
            "independent_hs_spice_ephemeris_replay_metadata_json": _relative_or_absolute(metadata_path),
            "independent_hs_spice_ephemeris_replay_table_tex": _relative_or_absolute(table_path),
        },
        "limitations": limitations,
        "interpretation_limits": limitations,
    }
    _write_json(metadata_path, metadata)
    print(
        "Completed independent-HS SPICE-derived ephemeris replay "
        f"for {overall['epoch_count']} epochs, "
        f"SPICE branch worst={overall['branch_spice_replay_worst_over_epochs']:.6g}, "
        f"branch pass={overall['branch_spice_replay_pass_count_total']}/"
        f"{overall['branch_row_count_total']}, "
        f"max delta={overall['max_abs_delta_from_horizons_retuned']:.6g}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing independent-HS multi-epoch Horizons-retuned controls using "
            "SPICE-derived Moon/Sun geocentric vector caches."
        )
    )
    parser.add_argument("--config", type=Path, default=single_epoch.DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--source-results-dir", type=Path, default=DEFAULT_SOURCE_RESULTS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--spice-cache-dir", type=Path, default=DEFAULT_SPICE_CACHE_DIR)
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    parser.add_argument(
        "--refresh-spice-cache",
        action="store_true",
        help="Download NAIF kernels and regenerate compact SPICE vector caches before replay.",
    )
    parser.add_argument(
        "--keep-kernels",
        action="store_true",
        help="Keep downloaded kernel binaries under --kernel-dir after refresh. By default they are removed.",
    )
    parser.add_argument("--download-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--phase-time", type=float, action="append", default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--canonical-time-unit-seconds", type=float, default=DEFAULT_CANONICAL_TIME_UNIT_SECONDS)
    parser.add_argument("--reference-distance-km", type=float, default=DEFAULT_REFERENCE_DISTANCE_KM)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
