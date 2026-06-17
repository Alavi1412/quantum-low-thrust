"""Retune focused hard-catalog tail-coast controls under simple bicircular dynamics.

This experiment reads the completed accepted-control sidecars from the focused
CR3BP tail-coast package, uses those controls as deterministic seeds, and
retunes the nominal and selected branch controls against the original target at
the original fixed final time under the simple circular solar-tidal model. It is
not a SPICE, high-fidelity, production-parity, fuel-optimality, or quantum
validation workflow.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_tail_coast_recovery as tail_runner
from qlt.bicircular import (
    DEFAULT_SUN_DISTANCE_LU,
    DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO,
    DEFAULT_SUN_MU_RATIO,
    SolarTidalParameters,
)
from qlt.bicircular_tail_coast_recovery import (
    MODE,
    branch_recovery_weights_from_mapping,
    optimize_bicircular_tail_coast_branch,
    optimize_bicircular_tail_coast_nominal,
    terminal_error_bicircular,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.locked_recovery import BranchRecoveryWeights
from qlt.objective import outage_masks
from qlt.refinement import control_fuel, project_controls_to_ball
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "hard_catalog_tail_coast_branch_control_replay.yaml"
DEFAULT_RESULTS_DIR = Path("data/results/bicircular_tail_coast_recovery")
DEFAULT_TABLES_DIR = Path("tables/bicircular_tail_coast_recovery")
DEFAULT_CASE_ID = "tail_coast_all_one_two_segment_t5_portfolio"

RESULT_CSV_NAME = "bicircular_tail_coast_recovery.csv"
SUMMARY_CSV_NAME = "bicircular_tail_coast_recovery_summary.csv"
METADATA_NAME = "bicircular_tail_coast_recovery_metadata.json"
CHECKPOINT_NAME = "bicircular_tail_coast_recovery_checkpoint.json"
CONTROL_MANIFEST_NAME = "bicircular_tail_coast_recovery_control_manifest.json"
PROGRESS_CSV_NAME = "bicircular_tail_coast_recovery_progress.csv"
TABLE_NAME = "bicircular_tail_coast_recovery_table.tex"

STRICT_NOMINAL_THRESHOLD = 0.05
STRICT_ROBUST_THRESHOLD = 0.09

RESULT_COLUMNS = [
    "suite_case_id",
    "record_type",
    "phase_degrees",
    "phase_radians",
    "branch_order",
    "mask_index",
    "outage_mask",
    "source_controls_path",
    "source_controls_sha256",
    "retuned_controls_path",
    "retuned_controls_sha256",
    "source_cr3bp_terminal_error",
    "initial_bicircular_terminal_error",
    "retuned_bicircular_terminal_error",
    "configured_threshold",
    "passes_configured_threshold",
    "strict_threshold",
    "passes_strict_threshold",
    "recovery_start",
    "recovery_segments",
    "optimizer_ran",
    "optimizer_success",
    "accepted_candidate",
    "weight_source",
    "weights",
    "nfev",
    "runtime_seconds",
    "cost",
    "optimality",
    "fuel",
    "control_max_norm",
    "control_bound_violation",
    "retuning_semantics",
]

SUMMARY_COLUMNS = [
    "suite_case_id",
    "phase_degrees",
    "phase_radians",
    "package_complete",
    "nominal_error",
    "nominal_pass",
    "branch_row_count",
    "expected_branch_count",
    "branch_pass_count",
    "all_branch_pass",
    "max_branch_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "meets_thresholds",
    "strict_nominal_threshold",
    "strict_robust_threshold",
    "strict_nominal_pass",
    "strict_branch_pass_count",
    "strict_all_branch_pass",
    "strict_meets_thresholds",
    "total_nfev",
    "total_runtime_seconds",
    "nominal_nfev",
    "total_branch_nfev",
    "optimizer_success_count",
    "optimizer_ran_count",
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


def _write_json_with_sha256(path: Path, data: object) -> str:
    payload = _json_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


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
    for base in (Path.cwd(), ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _safe_stem(value: object) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def _phase_label(phase_degrees: float) -> str:
    text = f"{float(phase_degrees):.6g}".replace("-", "m").replace(".", "p")
    return f"phase_{text}"


def _json_field(value: object) -> str:
    return json.dumps(sanitize_json(value), sort_keys=True, ensure_ascii=True, allow_nan=False)


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


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
    if expected_sha256 not in (None, "") and str(expected_sha256).strip() != actual:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return json.loads(path.read_text(encoding="utf-8")), path, actual


def _append_artifact(items: list[dict[str, object]], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    items.append({"path": _relative_or_absolute(path), "sha256": _sha256(path), "bytes": path.stat().st_size})


def _case_by_id(config: dict) -> dict[str, dict]:
    return {str(case["suite_case_id"]): case for case in tail_runner._suite_cases(config)}


def _first_source_row(rows: list[dict[str, str]], case_id: str, source_csv: Path) -> dict[str, str]:
    for row in rows:
        if str(row.get("suite_case_id", "")) == str(case_id):
            return row
    raise RuntimeError(f"no source row in {_relative_or_absolute(source_csv)} for suite_case_id={case_id}")


def _settings_fingerprint(settings: dict) -> str:
    payload = json.dumps(sanitize_json(settings), sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _result_key(row: dict[str, object]) -> tuple[str, float, str, int]:
    record_type = str(row["record_type"])
    branch_order = -1 if record_type == "nominal" else int(row["branch_order"])
    return str(row["suite_case_id"]), float(row["phase_degrees"]), record_type, branch_order


def _checkpoint_context(row: dict[str, object]) -> str:
    try:
        case_id, phase, record_type, branch_order = _result_key(row)
    except (KeyError, TypeError, ValueError):
        return f"row={row!r}"
    if record_type == "branch":
        return f"suite_case_id={case_id}, phase_degrees={phase:g}, record_type=branch, branch_order={branch_order}"
    return f"suite_case_id={case_id}, phase_degrees={phase:g}, record_type=nominal"


def _checkpoint_validation_error(row: dict[str, object], message: str) -> RuntimeError:
    return RuntimeError(f"invalid resume checkpoint row ({_checkpoint_context(row)}): {message}")


def _require_checkpoint_value(row: dict[str, object], key: str) -> object:
    value = row.get(key)
    if value in (None, ""):
        raise _checkpoint_validation_error(row, f"missing {key!r}")
    return value


def _require_equal(row: dict[str, object], sidecar: dict, key: str, expected: object) -> None:
    actual = sidecar.get(key)
    if str(actual) != str(expected):
        raise _checkpoint_validation_error(row, f"sidecar {key!r} mismatch: expected {expected!r}, got {actual!r}")


def _require_optional_record_type(row: dict[str, object], sidecar: dict, expected: str) -> None:
    actual = sidecar.get("record_type")
    if actual in (None, ""):
        # Legacy retuned sidecars predate record_type. In that case, the
        # required sidecar_type, row identity, and controls-payload checks below
        # provide the unambiguous nominal/branch validation.
        return
    if str(actual) != str(expected):
        raise _checkpoint_validation_error(
            row,
            f"sidecar 'record_type' mismatch: expected {expected!r}, got {actual!r}",
        )


def _require_float_equal(row: dict[str, object], sidecar: dict, key: str, expected: object) -> None:
    try:
        actual_float = float(sidecar.get(key))
        expected_float = float(expected)
    except (TypeError, ValueError) as exc:
        raise _checkpoint_validation_error(row, f"sidecar {key!r} is not numeric") from exc
    if not np.isclose(actual_float, expected_float, rtol=0.0, atol=1.0e-12):
        raise _checkpoint_validation_error(
            row,
            f"sidecar {key!r} mismatch: expected {expected_float!r}, got {actual_float!r}",
        )


def _require_int_equal(row: dict[str, object], sidecar: dict, key: str, expected: object) -> None:
    try:
        actual_int = int(sidecar.get(key))
        expected_int = int(expected)
    except (TypeError, ValueError) as exc:
        raise _checkpoint_validation_error(row, f"sidecar {key!r} is not an integer") from exc
    if actual_int != expected_int:
        raise _checkpoint_validation_error(row, f"sidecar {key!r} mismatch: expected {expected_int}, got {actual_int}")


def _validated_controls_payload(row: dict[str, object], sidecar: dict, key: str, expected_segments: int) -> np.ndarray:
    if key not in sidecar:
        raise _checkpoint_validation_error(row, f"sidecar missing controls payload {key!r}")
    try:
        controls = np.asarray(sidecar[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise _checkpoint_validation_error(row, f"sidecar controls payload {key!r} is not numeric") from exc
    expected_shape = (int(expected_segments), 3)
    if expected_shape[0] == 0 and controls.size == 0:
        controls = np.empty(expected_shape, dtype=float)
    if controls.shape != expected_shape:
        raise _checkpoint_validation_error(
            row,
            f"sidecar controls payload {key!r} has shape {controls.shape}, expected {expected_shape}",
        )
    if not np.all(np.isfinite(controls)):
        raise _checkpoint_validation_error(row, f"sidecar controls payload {key!r} contains non-finite values")
    return controls


def _validate_checkpoint_row_sidecar(
    row: dict[str, object],
    *,
    settings_fingerprint: str,
    settings: dict[str, object],
) -> None:
    record_type = str(_require_checkpoint_value(row, "record_type"))
    if record_type not in {"nominal", "branch"}:
        raise _checkpoint_validation_error(row, f"unsupported record_type {record_type!r}")
    try:
        sidecar, sidecar_path, _ = _read_json_verified(
            _require_checkpoint_value(row, "retuned_controls_path"),
            _require_checkpoint_value(row, "retuned_controls_sha256"),
        )
    except RuntimeError as exc:
        raise _checkpoint_validation_error(row, str(exc)) from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise _checkpoint_validation_error(row, f"could not read retuned sidecar: {exc}") from exc

    expected_sidecar_type = (
        "bicircular_tail_coast_nominal_controls"
        if record_type == "nominal"
        else "bicircular_tail_coast_branch_controls"
    )
    _require_equal(row, sidecar, "settings_fingerprint", settings_fingerprint)
    _require_equal(row, sidecar, "sidecar_type", expected_sidecar_type)
    _require_optional_record_type(row, sidecar, record_type)
    _require_equal(row, sidecar, "suite_case_id", _require_checkpoint_value(row, "suite_case_id"))
    _require_float_equal(row, sidecar, "phase_degrees", _require_checkpoint_value(row, "phase_degrees"))
    controls_key = "nominal_controls" if record_type == "nominal" else "branch_controls"
    _validated_controls_payload(row, sidecar, controls_key, int(settings["segments"]))

    if record_type == "branch":
        _require_int_equal(row, sidecar, "branch_order", _require_checkpoint_value(row, "branch_order"))
        _require_int_equal(row, sidecar, "mask_index", _require_checkpoint_value(row, "mask_index"))
        if "recovery_controls" in sidecar:
            _validated_controls_payload(
                row,
                sidecar,
                "recovery_controls",
                int(_require_checkpoint_value(row, "recovery_segments")),
            )
    if not sidecar_path.is_file():
        raise _checkpoint_validation_error(row, f"retuned sidecar disappeared during validation: {sidecar_path}")


def _load_checkpoint(path: Path, settings_fingerprint: str, settings: dict[str, object]) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("settings_fingerprint", "")) != str(settings_fingerprint):
        return []
    rows = [dict(row) for row in data.get("rows", [])]
    keyed: dict[tuple[str, float, str, int], dict[str, object]] = {}
    for row in rows:
        _validate_checkpoint_row_sidecar(row, settings_fingerprint=settings_fingerprint, settings=settings)
        keyed[_result_key(row)] = row
    return list(keyed.values())


def _controls_from_row(row: dict[str, object]) -> np.ndarray:
    data, _, _ = _read_json_verified(row["retuned_controls_path"], row.get("retuned_controls_sha256"))
    key = "nominal_controls" if str(row["record_type"]) == "nominal" else "branch_controls"
    return np.asarray(data[key], dtype=float)


def _write_control_manifest(
    *,
    rows: list[dict[str, object]],
    results_dir: Path,
    controls_dir: Path,
    settings_fingerprint: str,
    settings: dict,
    complete: bool,
    expected_branch_count_per_phase: int,
) -> tuple[Path, str | None, Path, str | None]:
    progress_rows: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: (float(item["phase_degrees"]), str(item["record_type"]), int(item["branch_order"] or -1))):
        entry = {
            "suite_case_id": str(row["suite_case_id"]),
            "record_type": str(row["record_type"]),
            "phase_degrees": float(row["phase_degrees"]),
            "branch_order": row["branch_order"],
            "mask_index": row["mask_index"],
            "path": str(row["retuned_controls_path"]),
            "sha256": str(row["retuned_controls_sha256"]),
            "terminal_error": float(row["retuned_bicircular_terminal_error"]),
            "passes_configured_threshold": bool(row["passes_configured_threshold"]),
            "passes_strict_threshold": bool(row["passes_strict_threshold"]),
        }
        manifest_entries.append(entry)
        progress_rows.append({**entry, "status": "complete"})
    progress_path = controls_dir / PROGRESS_CSV_NAME
    if progress_rows:
        pd.DataFrame(progress_rows).to_csv(progress_path, index=False)
        progress_sha = _sha256(progress_path)
    else:
        progress_sha = None
    manifest_path = controls_dir / CONTROL_MANIFEST_NAME
    if not rows:
        return manifest_path, None, progress_path, progress_sha
    branch_entries = [row for row in rows if str(row["record_type"]) == "branch"]
    manifest = {
        "schema_version": 1,
        "sidecar_type": "bicircular_tail_coast_recovery_control_manifest",
        "settings_fingerprint": settings_fingerprint,
        "settings": settings,
        "progress_state": "complete" if complete else "in_progress",
        "expected_branch_count_per_phase": int(expected_branch_count_per_phase),
        "record_count": int(len(rows)),
        "branch_record_count": int(len(branch_entries)),
        "progress_csv_path": _relative_or_absolute(progress_path) if progress_sha else None,
        "progress_csv_sha256": progress_sha,
        "control_sidecars": manifest_entries,
        "retuning_semantics": (
            "Retuned controls under the simple bicircular solar-tidal model against the original fixed target "
            "and original transfer time; not SPICE/high-fidelity validation or fuel-optimality evidence."
        ),
    }
    manifest_sha = _write_json_with_sha256(manifest_path, manifest)
    return manifest_path, manifest_sha, progress_path, progress_sha


def _summary_rows(rows: list[dict[str, object]], *, expected_branch_count: int) -> list[dict[str, object]]:
    if not rows:
        return []
    summaries: list[dict[str, object]] = []
    by_phase = sorted({float(row["phase_degrees"]) for row in rows})
    for phase in by_phase:
        group = [row for row in rows if float(row["phase_degrees"]) == phase]
        nominal_rows = [row for row in group if str(row["record_type"]) == "nominal"]
        branch_rows = [row for row in group if str(row["record_type"]) == "branch"]
        if nominal_rows:
            nominal = nominal_rows[0]
            nominal_error = float(nominal["retuned_bicircular_terminal_error"])
            nominal_pass = bool(nominal["passes_configured_threshold"])
            nominal_nfev = int(nominal["nfev"])
        else:
            nominal_error = float("nan")
            nominal_pass = False
            nominal_nfev = 0
        branch_errors = [float(row["retuned_bicircular_terminal_error"]) for row in branch_rows]
        branch_passes = [bool(row["passes_configured_threshold"]) for row in branch_rows]
        strict_branch_passes = [bool(row["passes_strict_threshold"]) for row in branch_rows]
        complete = bool(len(nominal_rows) == 1 and len(branch_rows) == int(expected_branch_count))
        max_branch = float(np.max(branch_errors)) if branch_errors else float("nan")
        strict_nominal_pass = bool(nominal_error <= STRICT_NOMINAL_THRESHOLD) if np.isfinite(nominal_error) else False
        total_branch_nfev = int(sum(int(row["nfev"]) for row in branch_rows))
        summaries.append(
            {
                "suite_case_id": str(group[0]["suite_case_id"]),
                "phase_degrees": float(phase),
                "phase_radians": float(group[0]["phase_radians"]),
                "package_complete": complete,
                "nominal_error": nominal_error,
                "nominal_pass": nominal_pass,
                "branch_row_count": int(len(branch_rows)),
                "expected_branch_count": int(expected_branch_count),
                "branch_pass_count": int(sum(branch_passes)),
                "all_branch_pass": bool(len(branch_rows) == int(expected_branch_count) and all(branch_passes)),
                "max_branch_error": max_branch,
                "selected_worst_error": max_branch,
                "all_mask_worst_error": max_branch,
                "meets_thresholds": bool(complete and nominal_pass and all(branch_passes)),
                "strict_nominal_threshold": STRICT_NOMINAL_THRESHOLD,
                "strict_robust_threshold": STRICT_ROBUST_THRESHOLD,
                "strict_nominal_pass": strict_nominal_pass,
                "strict_branch_pass_count": int(sum(strict_branch_passes)),
                "strict_all_branch_pass": bool(len(branch_rows) == int(expected_branch_count) and all(strict_branch_passes)),
                "strict_meets_thresholds": bool(complete and strict_nominal_pass and all(strict_branch_passes)),
                "total_nfev": int(nominal_nfev + total_branch_nfev),
                "total_runtime_seconds": float(sum(float(row["runtime_seconds"]) for row in group)),
                "nominal_nfev": nominal_nfev,
                "total_branch_nfev": total_branch_nfev,
                "optimizer_success_count": int(sum(bool(row["optimizer_success"]) for row in group)),
                "optimizer_ran_count": int(sum(bool(row["optimizer_ran"]) for row in group)),
            }
        )
    return summaries


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / TABLE_NAME
    if not summary_rows:
        path.write_text("% No bicircular tail-coast retuning rows.\n", encoding="utf-8")
        return path
    display = pd.DataFrame(summary_rows)[
        [
            "phase_degrees",
            "nominal_error",
            "branch_pass_count",
            "expected_branch_count",
            "max_branch_error",
            "meets_thresholds",
            "strict_meets_thresholds",
            "total_nfev",
        ]
    ].rename(
        columns={
            "phase_degrees": "Sun phase (deg)",
            "nominal_error": "Retuned nominal error",
            "branch_pass_count": "Branch passes",
            "expected_branch_count": "Branch rows",
            "max_branch_error": "Max retuned branch error",
            "meets_thresholds": "Pass 0.09/0.17",
            "strict_meets_thresholds": "Pass 0.05/0.09",
            "total_nfev": "nfev",
        }
    )
    display.to_latex(path, index=False, float_format="%.6g", escape=True)
    return path


def _write_outputs(
    *,
    rows: list[dict[str, object]],
    results_dir: Path,
    tables_dir: Path,
    controls_dir: Path,
    settings_fingerprint: str,
    settings: dict,
    complete: bool,
    expected_branch_count: int,
    command: str,
    config: dict,
    input_artifacts: list[dict[str, object]],
    source_control_sidecar_hashes: dict[str, object],
) -> dict[str, object]:
    rows = sorted(rows, key=lambda item: (float(item["phase_degrees"]), str(item["record_type"]), int(item["branch_order"] or -1)))
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / RESULT_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")
    summary_rows = _summary_rows(rows, expected_branch_count=expected_branch_count)
    summary_path = results_dir / SUMMARY_CSV_NAME
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False, float_format="%.17g")
    table_path = _write_table(summary_rows, tables_dir)
    manifest_path, manifest_sha, progress_path, progress_sha = _write_control_manifest(
        rows=rows,
        results_dir=results_dir,
        controls_dir=controls_dir,
        settings_fingerprint=settings_fingerprint,
        settings=settings,
        complete=complete,
        expected_branch_count_per_phase=expected_branch_count,
    )
    checkpoint = {
        "schema_version": 1,
        "settings_fingerprint": settings_fingerprint,
        "settings": settings,
        "complete": bool(complete),
        "row_count": int(len(rows)),
        "rows": rows,
    }
    checkpoint_path = results_dir / CHECKPOINT_NAME
    _write_json(checkpoint_path, checkpoint)
    final_summary = summary_rows[-1] if summary_rows else {}
    metadata = {
        "command": command,
        "row_count": int(len(rows)),
        "summary_row_count": int(len(summary_rows)),
        "settings_fingerprint": settings_fingerprint,
        "settings": settings,
        "config": config,
        "retuning_optimization_rerun": True,
        "uses_recorded_cr3bp_controls_as_seeds": True,
        "bicircular_tail_coast_retuned_recovery": True,
        "simple_bicircular_solar_tidal_model": True,
        "branch_control_replay": False,
        "spice_ephemeris_validation": False,
        "high_fidelity_validation": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "scope": {
            "case": DEFAULT_CASE_ID,
            "target_family": "catalog-DRO",
            "target_mode": "catalog_dro_phase",
            "fixed_final_time": True,
            "retuned_under_bicircular_dynamics": True,
            "phase_degrees": settings["phase_degrees"],
            "expected_branch_count_per_phase": int(expected_branch_count),
            "outage_lengths": settings["outage_lengths"],
            "tail_coast_segments": int(settings["tail_coast_segments"]),
        },
        "thresholds": {
            "configured_nominal_success": float(settings["thresholds"]["nominal_success"]),
            "configured_robust_success": float(settings["thresholds"]["robust_success"]),
            "strict_nominal_success": STRICT_NOMINAL_THRESHOLD,
            "strict_robust_success": STRICT_ROBUST_THRESHOLD,
        },
        "summary": summary_rows,
        "final_summary": final_summary,
        "source_control_sidecar_hashes": source_control_sidecar_hashes,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "bicircular_tail_coast_recovery_csv": _relative_or_absolute(csv_path),
            "bicircular_tail_coast_recovery_summary_csv": _relative_or_absolute(summary_path),
            "bicircular_tail_coast_recovery_metadata_json": _relative_or_absolute(results_dir / METADATA_NAME),
            "bicircular_tail_coast_recovery_checkpoint_json": _relative_or_absolute(checkpoint_path),
            "bicircular_tail_coast_recovery_control_manifest_json": _relative_or_absolute(manifest_path) if manifest_sha else None,
            "bicircular_tail_coast_recovery_progress_csv": _relative_or_absolute(progress_path) if progress_sha else None,
            "bicircular_tail_coast_recovery_table_tex": _relative_or_absolute(table_path),
        },
        "interpretation_limits": [
            "This is a deterministic retuning stress experiment under the existing simple bicircular solar-tidal dynamics.",
            "It is not SPICE ephemeris validation, high-fidelity flight validation, production solver parity, fuel optimality, or quantum evidence.",
            "Terminal errors are measured against the original CR3BP target at the original fixed final time.",
            "Configured pass/fail uses the recorded normalized thresholds unless explicitly reported otherwise.",
            "The strict (0.05, 0.09) audit is a recorded-error threshold audit over this generated retuning package.",
        ],
    }
    _write_json(results_dir / METADATA_NAME, metadata)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "metadata": metadata,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "table_path": table_path,
        "checkpoint_path": checkpoint_path,
    }


def _make_sidecar_common(
    *,
    settings: dict,
    source_path: Path,
    source_sha: str,
    phase_degrees: float,
    phase_radians: float,
    result: dict,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "settings_fingerprint": settings["settings_fingerprint"],
        "suite_case_id": settings["suite_case_id"],
        "target_mode": settings["target_mode"],
        "target_state": settings["target_state"],
        "thresholds": settings["thresholds"],
        "strict_thresholds": {
            "nominal_success": STRICT_NOMINAL_THRESHOLD,
            "robust_success": STRICT_ROBUST_THRESHOLD,
        },
        "transfer_time": float(settings["transfer_time"]),
        "segments": int(settings["segments"]),
        "substeps_per_segment": int(settings["substeps_per_segment"]),
        "amax": float(settings["amax"]),
        "tail_coast_segments": int(settings["tail_coast_segments"]),
        "phase_degrees": float(phase_degrees),
        "phase_radians": float(phase_radians),
        "solar_tidal_parameters": settings["solar_tidal_parameters"],
        "source_cr3bp_controls_path": _relative_or_absolute(source_path),
        "source_cr3bp_controls_sha256": source_sha,
        "accepted_candidate": str(result.get("accepted_candidate", "")),
        "accepted_nominal_weight_variant_label": result.get("accepted_nominal_weight_variant_label"),
        "accepted_nominal_weight_variant_index": result.get("accepted_nominal_weight_variant_index"),
        "accepted_nominal_weights": result.get("accepted_nominal_weights"),
        "nominal_weight_portfolio_candidates": result.get("nominal_weight_portfolio_candidates"),
        "optimizer_ran": bool(result.get("optimizer_ran", False)),
        "optimizer_success": bool(result.get("optimizer_success", False)),
        "message": str(result.get("message", "")),
        "nfev": int(result.get("nfev", 0)),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        "cost": float(result.get("cost", 0.0)),
        "optimality": float(result.get("optimality", 0.0)) if result.get("optimality") is not None else None,
        "retuning_semantics": (
            "Controls were retuned under the simple bicircular solar-tidal dynamics against the original fixed target; "
            "this sidecar is not high-fidelity validation or fuel-optimality evidence."
        ),
    }


def _row_from_nominal(
    *,
    settings: dict,
    phase_degrees: float,
    phase_radians: float,
    source_path: Path,
    source_sha: str,
    source_cr3bp_error: float,
    result: dict,
    sidecar_path: Path,
    sidecar_sha: str,
) -> dict[str, object]:
    error = float(result["nominal_error"])
    threshold = float(settings["thresholds"]["nominal_success"])
    return {
        "suite_case_id": settings["suite_case_id"],
        "record_type": "nominal",
        "phase_degrees": float(phase_degrees),
        "phase_radians": float(phase_radians),
        "branch_order": "",
        "mask_index": "",
        "outage_mask": "",
        "source_controls_path": _relative_or_absolute(source_path),
        "source_controls_sha256": source_sha,
        "retuned_controls_path": _relative_or_absolute(sidecar_path),
        "retuned_controls_sha256": sidecar_sha,
        "source_cr3bp_terminal_error": float(source_cr3bp_error),
        "initial_bicircular_terminal_error": float(result.get("initial_terminal_error", error)),
        "retuned_bicircular_terminal_error": error,
        "configured_threshold": threshold,
        "passes_configured_threshold": bool(error <= threshold),
        "strict_threshold": STRICT_NOMINAL_THRESHOLD,
        "passes_strict_threshold": bool(error <= STRICT_NOMINAL_THRESHOLD),
        "recovery_start": "",
        "recovery_segments": "",
        "optimizer_ran": bool(result.get("optimizer_ran", False)),
        "optimizer_success": bool(result.get("optimizer_success", False)),
        "accepted_candidate": str(result.get("accepted_candidate", "")),
        "weight_source": "nominal_weight_portfolio",
        "weights": _json_field(result.get("accepted_nominal_weights", result.get("weights", {}))),
        "nfev": int(result.get("nfev", 0)),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        "cost": float(result.get("cost", 0.0)),
        "optimality": result.get("optimality"),
        "fuel": control_fuel(np.asarray(result["nominal_controls"], dtype=float), float(settings["transfer_time"])),
        "control_max_norm": float(np.max(np.linalg.norm(np.asarray(result["nominal_controls"], dtype=float), axis=1))),
        "control_bound_violation": float(max(0.0, np.max(np.linalg.norm(np.asarray(result["nominal_controls"], dtype=float), axis=1)) - float(settings["amax"]))),
        "retuning_semantics": "nominal tail-coast controls retuned under simple bicircular solar-tidal dynamics",
    }


def _row_from_branch(
    *,
    settings: dict,
    phase_degrees: float,
    phase_radians: float,
    branch_order: int,
    branch_json: dict,
    source_path: Path,
    source_sha: str,
    result: dict,
    sidecar_path: Path,
    sidecar_sha: str,
    weight_source: str,
) -> dict[str, object]:
    error = float(result["terminal_error"])
    threshold = float(settings["thresholds"]["robust_success"])
    return {
        "suite_case_id": settings["suite_case_id"],
        "record_type": "branch",
        "phase_degrees": float(phase_degrees),
        "phase_radians": float(phase_radians),
        "branch_order": int(branch_order),
        "mask_index": int(branch_json["mask_index"]),
        "outage_mask": _json_field(branch_json["outage_mask"]),
        "source_controls_path": _relative_or_absolute(source_path),
        "source_controls_sha256": source_sha,
        "retuned_controls_path": _relative_or_absolute(sidecar_path),
        "retuned_controls_sha256": sidecar_sha,
        "source_cr3bp_terminal_error": float(branch_json["terminal_error"]),
        "initial_bicircular_terminal_error": float(result.get("initial_terminal_error", error)),
        "retuned_bicircular_terminal_error": error,
        "configured_threshold": threshold,
        "passes_configured_threshold": bool(error <= threshold),
        "strict_threshold": STRICT_ROBUST_THRESHOLD,
        "passes_strict_threshold": bool(error <= STRICT_ROBUST_THRESHOLD),
        "recovery_start": int(result["recovery_start"]),
        "recovery_segments": int(result["recovery_segments"]),
        "optimizer_ran": bool(result.get("optimizer_ran", False)),
        "optimizer_success": bool(result.get("optimizer_success", False)),
        "accepted_candidate": str(result.get("accepted_candidate", "")),
        "weight_source": weight_source,
        "weights": _json_field(result.get("branch_weights", {})),
        "nfev": int(result.get("nfev", 0)),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
        "cost": float(result.get("cost", 0.0)),
        "optimality": result.get("optimality"),
        "fuel": float(result.get("branch_fuel", 0.0)),
        "control_max_norm": float(result.get("control_max_norm", 0.0)),
        "control_bound_violation": float(result.get("control_bound_violation", 0.0)),
        "retuning_semantics": "branch recovery controls retuned under simple bicircular solar-tidal dynamics from persisted CR3BP branch-control seeds",
    }


def _write_nominal_sidecar(
    *,
    controls_dir: Path,
    settings: dict,
    phase_degrees: float,
    phase_radians: float,
    source_path: Path,
    source_sha: str,
    source_cr3bp_error: float,
    result: dict,
) -> tuple[Path, str]:
    path = controls_dir / f"{_safe_stem(settings['suite_case_id'])}_{_phase_label(phase_degrees)}_nominal_controls.json"
    controls = project_controls_to_ball(np.asarray(result["nominal_controls"], dtype=float), float(settings["amax"]))
    data = {
        "sidecar_type": "bicircular_tail_coast_nominal_controls",
        "record_type": "nominal",
        **_make_sidecar_common(
            settings=settings,
            source_path=source_path,
            source_sha=source_sha,
            phase_degrees=phase_degrees,
            phase_radians=phase_radians,
            result=result,
        ),
        "source_cr3bp_terminal_error": float(source_cr3bp_error),
        "initial_bicircular_terminal_error": float(result.get("initial_terminal_error", result["nominal_error"])),
        "retuned_bicircular_terminal_error": float(result["nominal_error"]),
        "accepted_nominal_weight_variant_label": str(result.get("accepted_nominal_weight_variant_label", "")),
        "accepted_nominal_weight_variant_index": int(result.get("accepted_nominal_weight_variant_index", 0)),
        "accepted_nominal_weights": dict(result.get("accepted_nominal_weights", result.get("weights", {}))),
        "nominal_weight_portfolio_candidates": list(result.get("nominal_weight_portfolio_candidates", [])),
        "configured_threshold": float(settings["thresholds"]["nominal_success"]),
        "passes_configured_threshold": bool(float(result["nominal_error"]) <= float(settings["thresholds"]["nominal_success"])),
        "strict_threshold": STRICT_NOMINAL_THRESHOLD,
        "passes_strict_threshold": bool(float(result["nominal_error"]) <= STRICT_NOMINAL_THRESHOLD),
        "nominal_controls": controls.tolist(),
        "control_norms": np.linalg.norm(controls, axis=1).astype(float).tolist(),
        "fuel": control_fuel(controls, float(settings["transfer_time"])),
    }
    return path, _write_json_with_sha256(path, data)


def _write_branch_sidecar(
    *,
    controls_dir: Path,
    settings: dict,
    phase_degrees: float,
    phase_radians: float,
    branch_order: int,
    branch_json: dict,
    source_path: Path,
    source_sha: str,
    result: dict,
    weight_source: str,
) -> tuple[Path, str]:
    mask_index = int(branch_json["mask_index"])
    path = controls_dir / (
        f"{_safe_stem(settings['suite_case_id'])}_{_phase_label(phase_degrees)}_"
        f"branch_{int(branch_order):03d}_mask_{mask_index:03d}_controls.json"
    )
    controls = project_controls_to_ball(np.asarray(result["branch_controls"], dtype=float), float(settings["amax"]))
    recovery_controls = project_controls_to_ball(np.asarray(result["recovery_controls"], dtype=float), float(settings["amax"]))
    data = {
        "sidecar_type": "bicircular_tail_coast_branch_controls",
        "record_type": "branch",
        **_make_sidecar_common(
            settings=settings,
            source_path=source_path,
            source_sha=source_sha,
            phase_degrees=phase_degrees,
            phase_radians=phase_radians,
            result=result,
        ),
        "branch_order": int(branch_order),
        "mask_index": mask_index,
        "outage_mask": [int(value) for value in branch_json["outage_mask"]],
        "source_cr3bp_terminal_error": float(branch_json["terminal_error"]),
        "initial_bicircular_terminal_error": float(result.get("initial_terminal_error", result["terminal_error"])),
        "retuned_bicircular_terminal_error": float(result["terminal_error"]),
        "configured_threshold": float(settings["thresholds"]["robust_success"]),
        "passes_configured_threshold": bool(float(result["terminal_error"]) <= float(settings["thresholds"]["robust_success"])),
        "strict_threshold": STRICT_ROBUST_THRESHOLD,
        "passes_strict_threshold": bool(float(result["terminal_error"]) <= STRICT_ROBUST_THRESHOLD),
        "recovery_start": int(result["recovery_start"]),
        "recovery_segments": int(result["recovery_segments"]),
        "weight_source": weight_source,
        "branch_weights": dict(result.get("branch_weights", {})),
        "branch_fuel": float(result.get("branch_fuel", 0.0)),
        "control_max_norm": float(result.get("control_max_norm", 0.0)),
        "control_bound_violation": float(result.get("control_bound_violation", 0.0)),
        "branch_controls": controls.tolist(),
        "recovery_controls": recovery_controls.tolist(),
        "control_norms": np.linalg.norm(controls, axis=1).astype(float).tolist(),
        "recovery_control_norms": np.linalg.norm(recovery_controls, axis=1).astype(float).tolist()
        if recovery_controls.size
        else [],
    }
    return path, _write_json_with_sha256(path, data)


def _branch_weight_for_result(args, config: dict, case: dict, branch_json: dict) -> tuple[BranchRecoveryWeights, str]:
    if args.branch_weight_source == "accepted" and branch_json.get("accepted_branch_weights"):
        return branch_recovery_weights_from_mapping(branch_json["accepted_branch_weights"]), "accepted_cr3bp_branch_weights"
    return tail_runner._branch_weights(config, case["case_raw"]), "configured_branch_weights"


def _distinct_weight_variants(variants: list[tuple[str, BranchRecoveryWeights]]) -> list[tuple[str, BranchRecoveryWeights]]:
    out: list[tuple[str, BranchRecoveryWeights]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for label, weights in variants:
        key = (float(weights.terminal), float(weights.control), float(weights.smooth), float(weights.continuity))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, weights))
    return out


def _nominal_weight_variants(config: dict, case: dict) -> list[tuple[str, BranchRecoveryWeights]]:
    return _distinct_weight_variants(
        [
            ("configured_tail_nominal", tail_runner._tail_nominal_weights(config, case["case_raw"])),
            ("regularized_001", BranchRecoveryWeights(terminal=4.0, control=0.01, smooth=0.01, continuity=0.0)),
            ("terminal_only", BranchRecoveryWeights(terminal=4.0, control=0.0, smooth=0.0, continuity=0.0)),
        ]
    )


def _optimize_nominal_portfolio(
    *,
    config: dict,
    case: dict,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    seed_controls: np.ndarray,
    tail_coast_segments: int,
    max_nfev: int,
    phase_rad: float,
    parameters: SolarTidalParameters,
    tolerances: dict[str, float],
) -> dict:
    candidates: list[dict[str, object]] = []
    for index, (label, weights) in enumerate(_nominal_weight_variants(config, case)):
        result = optimize_bicircular_tail_coast_nominal(
            state0=state0,
            target=target,
            cfg=cfg,
            seed_controls=seed_controls,
            tail_coast_segments=tail_coast_segments,
            max_nfev=max_nfev,
            phase_rad=phase_rad,
            parameters=parameters,
            weights=weights,
            **tolerances,
        )
        result = dict(result)
        result["nominal_weight_variant_label"] = label
        result["nominal_weight_variant_index"] = int(index)
        result["weights"] = weights.as_dict()
        candidates.append(result)
    accepted = min(
        candidates,
        key=lambda item: (
            float(item["nominal_error"]),
            float(control_fuel(np.asarray(item["nominal_controls"], dtype=float), float(cfg.tf))),
        ),
    )
    accepted = dict(accepted)
    accepted["accepted_nominal_weight_variant_label"] = str(accepted["nominal_weight_variant_label"])
    accepted["accepted_nominal_weight_variant_index"] = int(accepted["nominal_weight_variant_index"])
    accepted["accepted_nominal_weights"] = dict(accepted["weights"])
    accepted["accepted_variant_nfev"] = int(accepted["nfev"])
    accepted["accepted_variant_runtime_seconds"] = float(accepted["runtime_seconds"])
    accepted["nfev"] = int(sum(int(candidate["nfev"]) for candidate in candidates))
    accepted["runtime_seconds"] = float(sum(float(candidate["runtime_seconds"]) for candidate in candidates))
    accepted["nominal_weight_portfolio_enabled"] = True
    accepted["nominal_weight_portfolio_candidate_count"] = int(len(candidates))
    accepted["nominal_weight_portfolio_candidates"] = [
        {
            "label": str(candidate["nominal_weight_variant_label"]),
            "index": int(candidate["nominal_weight_variant_index"]),
            "weights": dict(candidate["weights"]),
            "accepted_candidate": str(candidate["accepted_candidate"]),
            "initial_terminal_error": float(candidate.get("initial_terminal_error", float("nan"))),
            "retuned_terminal_error": float(candidate["nominal_error"]),
            "optimizer_ran": bool(candidate.get("optimizer_ran", False)),
            "optimizer_success": bool(candidate.get("optimizer_success", False)),
            "nfev": int(candidate["nfev"]),
            "runtime_seconds": float(candidate["runtime_seconds"]),
            "message": str(candidate.get("message", "")),
        }
        for candidate in candidates
    ]
    return accepted


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.is_file() and not args.config.is_absolute():
        config_path = ROOT / args.config
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    if not source_states.is_file() and not args.source_states.is_absolute():
        source_states = ROOT / args.source_states
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = _case_by_id(config)
    if args.case_id not in cases:
        raise RuntimeError(f"case_id {args.case_id!r} is not present in {config_path}")
    case = cases[args.case_id]
    case_config = tail_runner._case_config(config, case)
    states = load_configured_states(Path.cwd(), case_config, source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    source_results_dir, _, _ = output_directories(Path.cwd(), config)
    source_csv = source_results_dir / "tail_coast_recovery.csv"
    source_row = _first_source_row(_read_csv_rows(source_csv), args.case_id, source_csv)
    if not _bool_value(source_row.get("branch_control_replay_ready", "")):
        raise RuntimeError(f"source row {args.case_id} is not branch-control replay ready")

    manifest, manifest_path, manifest_sha = _read_json_verified(
        source_row["branch_control_manifest_path"],
        source_row.get("branch_control_manifest_sha256"),
    )
    nominal_json, nominal_path, nominal_sha = _read_json_verified(
        manifest["nominal_control_path"],
        manifest.get("nominal_control_sha256"),
    )
    target_state = np.asarray(states.target, dtype=float)
    recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
    if recorded_target.shape != target_state.shape:
        raise RuntimeError("source manifest target_state shape does not match configured target")
    target_delta = float(np.max(np.abs(target_state - recorded_target)))
    if target_delta > float(args.baseline_tolerance):
        raise RuntimeError(f"source manifest target_state mismatch: max abs delta {target_delta}")

    branch_entries = sorted(list(manifest.get("branch_control_sidecars", [])), key=lambda item: int(item["branch_order"]))
    expected_branch_count = int(manifest.get("branch_control_sidecar_count", len(branch_entries)))
    if expected_branch_count != len(branch_entries):
        raise RuntimeError("source manifest branch_control_sidecar_count does not match branch sidecar list")

    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    controls_dir = results_dir / "controls"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        sidecar_prefix = f"{_safe_stem(args.case_id)}_phase_"
        for stale in controls_dir.glob(f"{sidecar_prefix}*_controls.json"):
            stale.unlink()

    phase_degrees = [float(value) for value in args.phase_degrees]
    phases = [(phase, float(np.deg2rad(phase))) for phase in phase_degrees]
    parameters = SolarTidalParameters(
        sun_distance_lu=float(args.sun_distance_lu),
        sun_mu_ratio=float(args.sun_mu_ratio),
        sun_inertial_angular_rate_ratio=float(args.sun_inertial_angular_rate_ratio),
    )
    tolerances = tail_runner._tolerances(config, case["case_raw"])
    nominal_max_nfev = int(args.nominal_max_nfev if args.nominal_max_nfev is not None else case["tail_nominal_max_nfev"])
    branch_max_nfev = int(args.branch_max_nfev if args.branch_max_nfev is not None else case["branch_max_nfev"])
    settings = {
        "suite_case_id": args.case_id,
        "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
        "target_state": target_state.astype(float).tolist(),
        "phase_degrees": phase_degrees,
        "solar_tidal_parameters": parameters.as_dict(),
        "transfer_time": float(cfg.tf),
        "segments": int(cfg.n_segments),
        "substeps_per_segment": int(cfg.substeps),
        "amax": float(cfg.amax),
        "tail_coast_segments": int(case["tail_coast_segments"]),
        "outage_lengths": [int(value) for value in case["outage_lengths"]],
        "thresholds": dict(case_config["objective"]["thresholds"]),
        "strict_thresholds": {
            "nominal_success": STRICT_NOMINAL_THRESHOLD,
            "robust_success": STRICT_ROBUST_THRESHOLD,
        },
        "nominal_max_nfev": nominal_max_nfev,
        "nominal_weight_variants": [
            {"label": label, "weights": weights.as_dict()}
            for label, weights in _nominal_weight_variants(config, case)
        ],
        "branch_max_nfev": branch_max_nfev,
        "branch_weight_source": str(args.branch_weight_source),
        "tolerances": tolerances,
        "source_config_path": _relative_or_absolute(config_path),
        "source_states_path": _relative_or_absolute(source_states),
        "source_tail_coast_csv": _relative_or_absolute(source_csv),
        "source_manifest_path": _relative_or_absolute(manifest_path),
        "source_manifest_sha256": manifest_sha,
    }
    settings_fingerprint = _settings_fingerprint(settings)
    settings["settings_fingerprint"] = settings_fingerprint

    input_artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in (config_path, source_states, source_csv, manifest_path, nominal_path):
        _append_artifact(input_artifacts, seen_artifacts, path)
    source_control_sidecar_hashes: dict[str, object] = {
        "nominal": {"path": _relative_or_absolute(nominal_path), "sha256": nominal_sha},
        "branches": [],
    }
    for entry in branch_entries:
        branch_path = _resolve_existing_path(entry["path"])
        _append_artifact(input_artifacts, seen_artifacts, branch_path)
        source_control_sidecar_hashes["branches"].append(
            {
                "branch_order": int(entry["branch_order"]),
                "mask_index": int(entry["mask_index"]),
                "path": _relative_or_absolute(branch_path),
                "sha256": _sha256(branch_path),
            }
        )

    rows: list[dict[str, object]] = []
    if args.resume:
        rows = _load_checkpoint(results_dir / CHECKPOINT_NAME, settings_fingerprint, settings)
    row_keys = {_result_key(row) for row in rows}
    started = time.perf_counter()

    def persist(complete: bool = False) -> None:
        _write_outputs(
            rows=rows,
            results_dir=results_dir,
            tables_dir=tables_dir,
            controls_dir=controls_dir,
            settings_fingerprint=settings_fingerprint,
            settings=settings,
            complete=complete,
            expected_branch_count=expected_branch_count,
            command=" ".join(sys.argv),
            config=config,
            input_artifacts=input_artifacts,
            source_control_sidecar_hashes=source_control_sidecar_hashes,
        )

    nominal_seed = np.asarray(nominal_json["controls"], dtype=float)
    nominal_source_error = float(nominal_json.get("nominal_error", source_row.get("nominal_error", float("nan"))))
    for phase_degrees_value, phase_rad in phases:
        nominal_key = (args.case_id, float(phase_degrees_value), "nominal", -1)
        if nominal_key not in row_keys:
            nominal_result = _optimize_nominal_portfolio(
                config=config,
                case=case,
                state0=states.initial,
                target=target_state,
                cfg=cfg,
                seed_controls=nominal_seed,
                tail_coast_segments=int(case["tail_coast_segments"]),
                max_nfev=nominal_max_nfev,
                phase_rad=phase_rad,
                parameters=parameters,
                tolerances=tolerances,
            )
            nominal_sidecar_path, nominal_sidecar_sha = _write_nominal_sidecar(
                controls_dir=controls_dir,
                settings=settings,
                phase_degrees=phase_degrees_value,
                phase_radians=phase_rad,
                source_path=nominal_path,
                source_sha=nominal_sha,
                source_cr3bp_error=nominal_source_error,
                result=nominal_result,
            )
            nominal_row = _row_from_nominal(
                settings=settings,
                phase_degrees=phase_degrees_value,
                phase_radians=phase_rad,
                source_path=nominal_path,
                source_sha=nominal_sha,
                source_cr3bp_error=nominal_source_error,
                result=nominal_result,
                sidecar_path=nominal_sidecar_path,
                sidecar_sha=nominal_sidecar_sha,
            )
            rows.append(nominal_row)
            row_keys.add(nominal_key)
            persist(complete=False)
            print(
                f"phase={phase_degrees_value:g} nominal initial={nominal_row['initial_bicircular_terminal_error']:.6g} "
                f"retuned={nominal_row['retuned_bicircular_terminal_error']:.6g} "
                f"pass={nominal_row['passes_configured_threshold']} nfev={nominal_row['nfev']} "
                f"runtime={nominal_row['runtime_seconds']:.1f}s",
                flush=True,
            )

        nominal_row = next(
            row
            for row in rows
            if str(row["record_type"]) == "nominal" and float(row["phase_degrees"]) == float(phase_degrees_value)
        )
        retuned_nominal_controls = _controls_from_row(nominal_row)
        phase_branch_entries = branch_entries[: int(args.max_branches)] if args.max_branches is not None else branch_entries
        for entry in phase_branch_entries:
            branch_order = int(entry["branch_order"])
            branch_key = (args.case_id, float(phase_degrees_value), "branch", branch_order)
            if branch_key in row_keys:
                continue
            if args.runtime_budget_seconds is not None and time.perf_counter() - started >= float(args.runtime_budget_seconds):
                persist(complete=False)
                print("runtime budget reached before launching next branch; checkpoint written", flush=True)
                return pd.DataFrame(rows, columns=RESULT_COLUMNS)
            branch_json, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            seed_branch_controls = np.asarray(branch_json["branch_controls"], dtype=float)
            weight, weight_source = _branch_weight_for_result(args, config, case, branch_json)
            branch_result = optimize_bicircular_tail_coast_branch(
                state0=states.initial,
                target=target_state,
                cfg=cfg,
                nominal_controls=retuned_nominal_controls,
                mask=masks[int(branch_json["mask_index"])],
                mask_index=int(branch_json["mask_index"]),
                seed_branch_controls=seed_branch_controls,
                tail_coast_segments=int(case["tail_coast_segments"]),
                max_nfev=branch_max_nfev,
                robust_threshold=float(case_config["objective"]["thresholds"]["robust_success"]),
                phase_rad=phase_rad,
                parameters=parameters,
                weights=weight,
                initialization_label="accepted_cr3bp_branch_controls",
                **tolerances,
            )
            branch_sidecar_path, branch_sidecar_sha = _write_branch_sidecar(
                controls_dir=controls_dir,
                settings=settings,
                phase_degrees=phase_degrees_value,
                phase_radians=phase_rad,
                branch_order=branch_order,
                branch_json=branch_json,
                source_path=branch_path,
                source_sha=branch_sha,
                result=branch_result,
                weight_source=weight_source,
            )
            branch_row = _row_from_branch(
                settings=settings,
                phase_degrees=phase_degrees_value,
                phase_radians=phase_rad,
                branch_order=branch_order,
                branch_json=branch_json,
                source_path=branch_path,
                source_sha=branch_sha,
                result=branch_result,
                sidecar_path=branch_sidecar_path,
                sidecar_sha=branch_sidecar_sha,
                weight_source=weight_source,
            )
            rows.append(branch_row)
            row_keys.add(branch_key)
            persist(complete=False)
            print(
                f"phase={phase_degrees_value:g} branch={branch_order:03d} mask={branch_row['mask_index']} "
                f"initial={branch_row['initial_bicircular_terminal_error']:.6g} "
                f"retuned={branch_row['retuned_bicircular_terminal_error']:.6g} "
                f"pass={branch_row['passes_configured_threshold']} nfev={branch_row['nfev']} "
                f"runtime={branch_row['runtime_seconds']:.1f}s",
                flush=True,
            )

    expected_total = len(phases) * (1 + expected_branch_count)
    complete = bool(args.max_branches is None and len(rows) == expected_total)
    persist(complete=complete)
    summary = _summary_rows(rows, expected_branch_count=expected_branch_count)
    if summary:
        last = summary[-1]
        print(
            "Completed bicircular tail-coast retuning "
            f"rows={len(rows)}, complete={complete}, nominal={last['nominal_error']:.6g}, "
            f"branch_pass={last['branch_pass_count']}/{last['expected_branch_count']}, "
            f"max_branch={last['max_branch_error']:.6g}, meets={last['meets_thresholds']}, "
            f"strict_meets={last['strict_meets_thresholds']}, nfev={last['total_nfev']}.",
            flush=True,
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retune focused hard-catalog tail-coast controls under the simple bicircular solar-tidal model."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--phase-degrees", type=float, nargs="+", default=[0.0])
    parser.add_argument("--nominal-max-nfev", type=int, default=None)
    parser.add_argument("--branch-max-nfev", type=int, default=None)
    parser.add_argument("--max-branches", type=int, default=None, help="Optional smoke-test limit; default retunes all branches.")
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--branch-weight-source", choices=["accepted", "configured"], default="accepted")
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
