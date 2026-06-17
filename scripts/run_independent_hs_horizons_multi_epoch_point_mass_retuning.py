"""Multi-epoch cached-Horizons point-mass retuning for independent-HS controls.

This wrapper evaluates the strongest independent-HS polish case across fixed
representative 2026 epochs. Each epoch is run through the single-epoch
Earth/Moon/Sun point-mass retuning script in an epoch-specific output folder,
then a compact row-level aggregate CSV, metadata JSON, and LaTeX summary table
are written. The package is a cached-Horizons point-mass stress/retuning
artifact only. It is not SPICE propagation, high-fidelity or flight validation,
production-solver parity, fuel-optimality evidence, DOI evidence, or quantum
evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_independent_hs_horizons_point_mass_retuning as single_epoch
from qlt.ephemeris_contrast import (
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    canonical_node_jds,
    canonical_node_times,
    fetch_horizons_vectors,
    horizons_query_url,
    horizons_vectors_query_params,
    write_horizons_cache,
)


DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_horizons_multi_epoch_point_mass_retuning")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_horizons_multi_epoch_point_mass_retuning")
MULTI_EPOCH_CSV_NAME = "independent_hs_horizons_multi_epoch_point_mass_retuning.csv"
MULTI_EPOCH_METADATA_NAME = "independent_hs_horizons_multi_epoch_point_mass_retuning_metadata.json"
MULTI_EPOCH_TABLE_NAME = "independent_hs_horizons_multi_epoch_point_mass_retuning_table.tex"

LIMITATION_FLAGS = [
    "spice_ephemeris_validation",
    "high_fidelity_validation",
    "high_fidelity_flight_validation",
    "production_solver_parity_claim",
    "fuel_optimality_claim",
    "doi_claim",
    "quantum_advantage_claim",
]

CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(CSV_FIELD_LIMIT)
        break
    except OverflowError:
        CSV_FIELD_LIMIT //= 10


@dataclass(frozen=True)
class EpochSpec:
    epoch_id: str
    label: str
    start_jd_tdb: float
    start_calendar_tdb: str
    cache_path: Path


DEFAULT_EPOCHS = (
    EpochSpec(
        "2026jan01",
        "2026-Jan-01",
        2461041.5,
        "A.D. 2026-Jan-01 00:00:00.0000",
        Path("data/cache/horizons/independent_hs_phase_shift_2026jan01_vectors.json"),
    ),
    EpochSpec(
        "2026apr01",
        "2026-Apr-01",
        2461131.5,
        "A.D. 2026-Apr-01 00:00:00.0000",
        Path("data/cache/horizons/independent_hs_phase_shift_2026apr01_vectors.json"),
    ),
    EpochSpec(
        "2026jul01",
        "2026-Jul-01",
        2461222.5,
        "A.D. 2026-Jul-01 00:00:00.0000",
        Path("data/cache/horizons/independent_hs_phase_shift_2026jul01_vectors.json"),
    ),
    EpochSpec(
        "2026oct01",
        "2026-Oct-01",
        2461314.5,
        "A.D. 2026-Oct-01 00:00:00.0000",
        Path("data/cache/horizons/independent_hs_phase_shift_2026oct01_vectors.json"),
    ),
)


MULTI_EPOCH_COLUMNS = [
    "epoch_id",
    "epoch_label",
    "epoch_start_calendar_tdb",
    "epoch_start_jd_tdb",
    "epoch_order",
    *single_epoch.RETUNING_COLUMNS,
    "single_epoch_results_dir",
    "single_epoch_metadata_json",
]


def _json_bytes(data: object) -> bytes:
    text = json.dumps(
        single_epoch.sanitize_json(data),
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


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _resolve_cache_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def _input_csv_for_config(config: dict) -> Path:
    source_results_dir, _, _ = single_epoch.output_directories(Path.cwd(), config)
    input_csv = source_results_dir / "independent_hs_all_configured_headroom.csv"
    if input_csv.is_file():
        return input_csv
    artifact_stem = str((config.get("run", {}) or {}).get("artifact_stem", "independent_hs_all_configured_headroom"))
    return source_results_dir / f"{artifact_stem}.csv"


def _first_objective_config(args: argparse.Namespace):
    config_path = _resolve_existing_file(args.config)
    source_states = _resolve_existing_file(args.source_states)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_csv = _input_csv_for_config(config)
    input_rows = single_epoch._selected_rows(single_epoch._read_input_rows(input_csv), args)
    if not input_rows:
        raise RuntimeError("no replay-ready independent-HS polish row found for multi-epoch point-mass retuning")
    cases = single_epoch._case_by_id(config)
    case_id = str(input_rows[0].get("case_id", ""))
    if case_id not in cases:
        raise RuntimeError(f"configured suite is missing selected case {case_id}")
    case_config = single_epoch._case_config(config, cases[case_id])
    states = single_epoch.load_configured_states(Path.cwd(), case_config, source_states)
    cfg = single_epoch.make_objective_config(case_config, states.mu)
    return cfg


def _cache_metadata(args: argparse.Namespace, *, spec: EpochSpec, cfg) -> dict[str, object]:
    node_times = canonical_node_times(float(cfg.tf), int(cfg.n_segments))
    jds = canonical_node_jds(
        start_jd_tdb=float(spec.start_jd_tdb),
        tf=float(cfg.tf),
        n_segments=int(cfg.n_segments),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
    )
    moon_params = horizons_vectors_query_params("301", "500@399", jds)
    sun_params = horizons_vectors_query_params("10", "500@399", jds)
    return {
        "representative_epoch_note": (
            f"Fixed representative {spec.label} TDB epoch for independent-HS cached-Horizons "
            "Earth/Moon/Sun point-mass retuning/stress only; it is not a mission epoch."
        ),
        "epoch_id": spec.epoch_id,
        "epoch_label": spec.label,
        "start_jd_tdb": float(spec.start_jd_tdb),
        "start_calendar_tdb": spec.start_calendar_tdb,
        "canonical_transfer_time": float(cfg.tf),
        "segments": int(cfg.n_segments),
        "canonical_node_times": [float(value) for value in node_times],
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
            "the retuning package remains a cached point-mass stress model."
        ),
    }


def _refresh_cache(args: argparse.Namespace, *, spec: EpochSpec, cfg) -> Path:
    cache_path = _resolve_cache_path(spec.cache_path)
    jds = canonical_node_jds(
        start_jd_tdb=float(spec.start_jd_tdb),
        tf=float(cfg.tf),
        n_segments=int(cfg.n_segments),
        canonical_time_unit_seconds=float(args.canonical_time_unit_seconds),
    )
    moon_params = horizons_vectors_query_params("301", "500@399", jds)
    sun_params = horizons_vectors_query_params("10", "500@399", jds)
    moon_response = fetch_horizons_vectors(moon_params, timeout_seconds=float(args.fetch_timeout_seconds))
    sun_response = fetch_horizons_vectors(sun_params, timeout_seconds=float(args.fetch_timeout_seconds))
    write_horizons_cache(
        path=cache_path,
        moon_response=moon_response,
        sun_response=sun_response,
        metadata=_cache_metadata(args, spec=spec, cfg=cfg),
    )
    return cache_path


def _ensure_epoch_caches(args: argparse.Namespace, *, epochs: tuple[EpochSpec, ...], cfg) -> dict[str, dict[str, object]]:
    cache_info: dict[str, dict[str, object]] = {}
    created_paths: list[Path] = []
    try:
        for spec in epochs:
            cache_path = _resolve_cache_path(spec.cache_path)
            fetched = False
            if not cache_path.is_file():
                if not bool(args.refresh_caches):
                    raise RuntimeError(
                        f"Horizons cache not found for {spec.label}: {cache_path}. "
                        "Re-run with --refresh-caches to fetch missing caches."
                    )
                cache_path = _refresh_cache(args, spec=spec, cfg=cfg)
                created_paths.append(cache_path)
                fetched = True
            cache_info[spec.epoch_id] = {
                "path": cache_path,
                "sha256": _sha256(cache_path),
                "fetched_this_run": fetched,
            }
    except Exception:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return cache_info


def _single_epoch_args(
    args: argparse.Namespace,
    *,
    spec: EpochSpec,
    cache_path: Path,
    cache_sha256: str,
    results_dir: Path,
    tables_dir: Path,
) -> argparse.Namespace:
    single_args = single_epoch.build_parser().parse_args([])
    single_args.config = args.config
    single_args.source_states = args.source_states
    single_args.cache = cache_path
    single_args.expected_cache_sha256 = cache_sha256
    single_args.results_dir = results_dir
    single_args.tables_dir = tables_dir
    single_args.case_id = args.case_id
    single_args.phase_time = args.phase_time
    single_args.baseline_tolerance = args.baseline_tolerance
    single_args.start_jd_tdb = float(spec.start_jd_tdb)
    single_args.canonical_time_unit_seconds = float(args.canonical_time_unit_seconds)
    single_args.reference_distance_km = float(args.reference_distance_km)
    single_args.max_nfev = int(args.max_nfev)
    single_args.xtol = float(args.xtol)
    single_args.ftol = float(args.ftol)
    single_args.gtol = float(args.gtol)
    single_args.terminal_residual_weight = float(args.terminal_residual_weight)
    single_args.control_regularization_weight = float(args.control_regularization_weight)
    single_args.smooth_regularization_weight = float(args.smooth_regularization_weight)
    return single_args


def _summarize_epoch_rows(
    *,
    spec: EpochSpec,
    df: pd.DataFrame,
    single_metadata: dict[str, object],
    cache_path: Path,
    cache_sha256: str,
    fetched_this_run: bool,
) -> dict[str, object]:
    nominal_rows = df[df["record_type"] == "nominal"]
    branch_rows = df[df["record_type"] == "branch"]
    if len(nominal_rows) != 1:
        raise RuntimeError(f"expected one nominal row for {spec.label}, got {len(nominal_rows)}")
    if len(branch_rows) != 8:
        raise RuntimeError(f"expected eight branch rows for {spec.label}, got {len(branch_rows)}")

    nominal_replay_pass = _bool_series(nominal_rows["point_mass_replay_passes_configured_threshold"])
    branch_replay_pass = _bool_series(branch_rows["point_mass_replay_passes_configured_threshold"])
    nominal_retuned_pass = _bool_series(nominal_rows["point_mass_retuned_passes_configured_threshold"])
    branch_retuned_pass = _bool_series(branch_rows["point_mass_retuned_passes_configured_threshold"])
    scipy_success = _bool_series(df["scipy_success"])
    retuned_all_rows_pass = _bool_series(df["point_mass_retuned_passes_configured_threshold"]).all()

    replay_failure_recorded = bool((not nominal_replay_pass.all()) or int(branch_replay_pass.sum()) < len(branch_rows))
    return {
        "epoch_id": spec.epoch_id,
        "epoch_label": spec.label,
        "start_calendar_tdb": spec.start_calendar_tdb,
        "start_jd_tdb": float(spec.start_jd_tdb),
        "cache_path": _relative_or_absolute(cache_path),
        "cache_sha256": cache_sha256,
        "cache_fetched_this_run": bool(fetched_this_run),
        "nominal_row_count": int(len(nominal_rows)),
        "branch_row_count": int(len(branch_rows)),
        "row_count": int(len(df)),
        "persisted_nominal_point_mass_replay_error": float(nominal_rows["point_mass_replay_terminal_error"].max()),
        "persisted_branch_point_mass_replay_worst_error": float(branch_rows["point_mass_replay_terminal_error"].max()),
        "persisted_nominal_replay_passes_configured_threshold": bool(nominal_replay_pass.all()),
        "persisted_branch_replay_pass_count": int(branch_replay_pass.sum()),
        "direct_replay_failure_recorded": replay_failure_recorded,
        "retuned_nominal_point_mass_error": float(nominal_rows["point_mass_retuned_terminal_error"].max()),
        "retuned_branch_point_mass_worst_error": float(branch_rows["point_mass_retuned_terminal_error"].max()),
        "retuned_nominal_pass_count": int(nominal_retuned_pass.sum()),
        "retuned_branch_pass_count": int(branch_retuned_pass.sum()),
        "retuned_all_rows_pass": bool(retuned_all_rows_pass),
        "scipy_success_count": int(scipy_success.sum()),
        "total_nfev": int(pd.to_numeric(df["nfev"]).sum()),
        "total_runtime_seconds": float(pd.to_numeric(df["runtime_seconds"]).sum()),
        "single_epoch_metadata_json": str(
            single_metadata.get("artifacts", {}).get("independent_hs_horizons_point_mass_retuning_metadata_json", "")
            if isinstance(single_metadata.get("artifacts"), dict)
            else ""
        ),
        "single_epoch_results_dir": _relative_or_absolute(
            Path(str(single_metadata.get("artifacts", {}).get("independent_hs_horizons_point_mass_retuning_csv", ""))).parent
            if isinstance(single_metadata.get("artifacts"), dict)
            else Path("")
        ),
    }


def _aggregate_overall_summary(epoch_summaries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "epoch_count": int(len(epoch_summaries)),
        "row_count": int(sum(int(row["row_count"]) for row in epoch_summaries)),
        "nominal_row_count": int(sum(int(row["nominal_row_count"]) for row in epoch_summaries)),
        "branch_row_count": int(sum(int(row["branch_row_count"]) for row in epoch_summaries)),
        "branch_rows_per_epoch": [int(row["branch_row_count"]) for row in epoch_summaries],
        "direct_replay_failure_epoch_count": int(
            sum(1 for row in epoch_summaries if bool(row["direct_replay_failure_recorded"]))
        ),
        "all_epochs_record_direct_replay_failure": bool(
            all(bool(row["direct_replay_failure_recorded"]) for row in epoch_summaries)
        ),
        "retuned_all_epochs_all_rows_pass": bool(all(bool(row["retuned_all_rows_pass"]) for row in epoch_summaries)),
        "retuned_nominal_worst_over_epochs": float(
            max(float(row["retuned_nominal_point_mass_error"]) for row in epoch_summaries)
        ),
        "retuned_branch_worst_over_epochs": float(
            max(float(row["retuned_branch_point_mass_worst_error"]) for row in epoch_summaries)
        ),
        "persisted_nominal_replay_worst_over_epochs": float(
            max(float(row["persisted_nominal_point_mass_replay_error"]) for row in epoch_summaries)
        ),
        "persisted_branch_replay_worst_over_epochs": float(
            max(float(row["persisted_branch_point_mass_replay_worst_error"]) for row in epoch_summaries)
        ),
        "retuned_branch_pass_count_total": int(sum(int(row["retuned_branch_pass_count"]) for row in epoch_summaries)),
        "branch_row_count_total": int(sum(int(row["branch_row_count"]) for row in epoch_summaries)),
        "scipy_success_count_total": int(sum(int(row["scipy_success_count"]) for row in epoch_summaries)),
        "total_nfev": int(sum(int(row["total_nfev"]) for row in epoch_summaries)),
        "total_runtime_seconds": float(sum(float(row["total_runtime_seconds"]) for row in epoch_summaries)),
    }


def _write_table(epoch_summaries: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / MULTI_EPOCH_TABLE_NAME
    rows = []
    for row in epoch_summaries:
        rows.append(
            {
                "Epoch": row["epoch_label"],
                "Start JD TDB": float(row["start_jd_tdb"]),
                "Direct nominal": float(row["persisted_nominal_point_mass_replay_error"]),
                "Direct branch worst": float(row["persisted_branch_point_mass_replay_worst_error"]),
                "Direct branch pass": f"{int(row['persisted_branch_replay_pass_count'])}/{int(row['branch_row_count'])}",
                "Retuned nominal": float(row["retuned_nominal_point_mass_error"]),
                "Retuned branch worst": float(row["retuned_branch_point_mass_worst_error"]),
                "Retuned branch pass": f"{int(row['retuned_branch_pass_count'])}/{int(row['branch_row_count'])}",
                "SciPy success": f"{int(row['scipy_success_count'])}/{int(row['row_count'])}",
                "nfev": int(row["total_nfev"]),
                "Cache SHA-256": str(row["cache_sha256"])[:12],
            }
        )
    pd.DataFrame(rows).to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


def run(args: argparse.Namespace) -> pd.DataFrame:
    epochs = DEFAULT_EPOCHS
    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    per_epoch_results_root = results_dir / "epochs"
    per_epoch_tables_root = tables_dir / "epochs"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cfg = _first_objective_config(args)
    cache_info = _ensure_epoch_caches(args, epochs=epochs, cfg=cfg)

    aggregate_frames: list[pd.DataFrame] = []
    epoch_summaries: list[dict[str, object]] = []
    per_epoch_artifacts: list[dict[str, object]] = []

    for order, spec in enumerate(epochs):
        info = cache_info[spec.epoch_id]
        cache_path = Path(str(info["path"]))
        cache_sha256 = str(info["sha256"])
        single_results_dir = per_epoch_results_root / spec.epoch_id
        single_tables_dir = per_epoch_tables_root / spec.epoch_id
        single_args = _single_epoch_args(
            args,
            spec=spec,
            cache_path=cache_path,
            cache_sha256=cache_sha256,
            results_dir=single_results_dir,
            tables_dir=single_tables_dir,
        )
        df = single_epoch.run(single_args)
        metadata_path = single_results_dir / single_epoch.RETUNING_METADATA_NAME
        single_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = _summarize_epoch_rows(
            spec=spec,
            df=df,
            single_metadata=single_metadata,
            cache_path=cache_path,
            cache_sha256=cache_sha256,
            fetched_this_run=bool(info["fetched_this_run"]),
        )
        epoch_summaries.append(summary)

        frame = df.copy()
        frame.insert(0, "epoch_order", int(order))
        frame.insert(0, "epoch_start_jd_tdb", float(spec.start_jd_tdb))
        frame.insert(0, "epoch_start_calendar_tdb", spec.start_calendar_tdb)
        frame.insert(0, "epoch_label", spec.label)
        frame.insert(0, "epoch_id", spec.epoch_id)
        frame["single_epoch_results_dir"] = _relative_or_absolute(single_results_dir)
        frame["single_epoch_metadata_json"] = _relative_or_absolute(metadata_path)
        aggregate_frames.append(frame)

        per_epoch_artifacts.append(
            {
                "epoch_id": spec.epoch_id,
                "epoch_label": spec.label,
                "results_dir": _relative_or_absolute(single_results_dir),
                "tables_dir": _relative_or_absolute(single_tables_dir),
                "metadata_json": _relative_or_absolute(metadata_path),
                "csv": _relative_or_absolute(single_results_dir / single_epoch.RETUNING_CSV_NAME),
                "table_tex": _relative_or_absolute(single_tables_dir / single_epoch.RETUNING_TABLE_NAME),
            }
        )

    aggregate_df = pd.concat(aggregate_frames, ignore_index=True)
    aggregate_df = aggregate_df[MULTI_EPOCH_COLUMNS]
    csv_path = results_dir / MULTI_EPOCH_CSV_NAME
    aggregate_df.to_csv(csv_path, index=False, float_format="%.17g")

    table_path = _write_table(epoch_summaries, tables_dir)
    metadata_path = results_dir / MULTI_EPOCH_METADATA_NAME
    overall_summary = _aggregate_overall_summary(epoch_summaries)
    cache_sha256_by_epoch = {row["epoch_id"]: row["cache_sha256"] for row in epoch_summaries}
    limitations = [
        "Representative epochs are fixed 2026 cached-Horizons point-mass stress cases, not mission epochs.",
        "Each epoch uses Earth central gravity plus indirect Moon/Sun point-mass terms from cached Horizons vectors.",
        "Persisted controls are directly replayed first; independent retuning then runs nominal and each branch separately.",
        "Retuned controls are stress-model artifacts and should not be read as production solver parity.",
        "This is not SPICE propagation, full high-fidelity or flight validation, fuel optimality, DOI evidence, or quantum advantage evidence.",
    ]
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(aggregate_df)),
        "epoch_count": int(len(epoch_summaries)),
        "nominal_row_count": int((aggregate_df["record_type"] == "nominal").sum()),
        "branch_row_count": int((aggregate_df["record_type"] == "branch").sum()),
        "optimization_rerun": True,
        "retuning": True,
        "uses_cached_horizons_vectors": True,
        "runtime_network_dependency": bool(any(bool(row["cache_fetched_this_run"]) for row in epoch_summaries)),
        "cached_horizons_earth_moon_sun_multi_epoch_point_mass_retuning": True,
        "multi_epoch_cached_horizons_point_mass_stress_retuning": True,
        **{flag: False for flag in LIMITATION_FLAGS},
        "default_epoch_count": len(DEFAULT_EPOCHS),
        "epochs": epoch_summaries,
        "cache_sha256_by_epoch": cache_sha256_by_epoch,
        "overall_summary": overall_summary,
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
        "input_artifacts": [
            {
                "path": _relative_or_absolute(_resolve_existing_file(args.config)),
                "sha256": _sha256(_resolve_existing_file(args.config)),
                "bytes": _resolve_existing_file(args.config).stat().st_size,
            },
            {
                "path": _relative_or_absolute(_resolve_existing_file(args.source_states)),
                "sha256": _sha256(_resolve_existing_file(args.source_states)),
                "bytes": _resolve_existing_file(args.source_states).stat().st_size,
            },
            *[
                {
                    "path": row["cache_path"],
                    "sha256": row["cache_sha256"],
                    "bytes": (_resolve_cache_path(DEFAULT_EPOCHS[index].cache_path)).stat().st_size,
                }
                for index, row in enumerate(epoch_summaries)
            ],
        ],
        "artifacts": {
            "independent_hs_horizons_multi_epoch_point_mass_retuning_csv": _relative_or_absolute(csv_path),
            "independent_hs_horizons_multi_epoch_point_mass_retuning_metadata_json": _relative_or_absolute(
                metadata_path
            ),
            "independent_hs_horizons_multi_epoch_point_mass_retuning_table_tex": _relative_or_absolute(table_path),
            "per_epoch_artifacts": per_epoch_artifacts,
        },
        "limitations": limitations,
        "interpretation_limits": limitations,
    }
    _write_json(metadata_path, metadata)
    print(
        "Completed independent-HS multi-epoch cached-Horizons point-mass retuning "
        f"for {overall_summary['epoch_count']} epochs, "
        f"retuned branch worst={overall_summary['retuned_branch_worst_over_epochs']:.6g}, "
        f"branch pass={overall_summary['retuned_branch_pass_count_total']}/"
        f"{overall_summary['branch_row_count_total']}.",
        flush=True,
    )
    return aggregate_df


def build_parser() -> argparse.ArgumentParser:
    single_defaults = single_epoch.build_parser().parse_args([])
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and independently retune the independent-HS polish row across fixed "
            "representative cached-Horizons Earth/Moon/Sun point-mass epochs."
        )
    )
    parser.add_argument("--config", type=Path, default=single_epoch.DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument(
        "--refresh-caches",
        action="store_true",
        help="Fetch any missing Horizons caches before running. Existing caches are left untouched.",
    )
    parser.add_argument("--fetch-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--phase-time", type=float, action="append", default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=single_defaults.baseline_tolerance)
    parser.add_argument(
        "--canonical-time-unit-seconds",
        type=float,
        default=DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    )
    parser.add_argument("--reference-distance-km", type=float, default=DEFAULT_REFERENCE_DISTANCE_KM)
    parser.add_argument("--max-nfev", type=int, default=single_defaults.max_nfev)
    parser.add_argument("--xtol", type=float, default=single_defaults.xtol)
    parser.add_argument("--ftol", type=float, default=single_defaults.ftol)
    parser.add_argument("--gtol", type=float, default=single_defaults.gtol)
    parser.add_argument("--terminal-residual-weight", type=float, default=single_defaults.terminal_residual_weight)
    parser.add_argument("--control-regularization-weight", type=float, default=single_defaults.control_regularization_weight)
    parser.add_argument("--smooth-regularization-weight", type=float, default=single_defaults.smooth_regularization_weight)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
