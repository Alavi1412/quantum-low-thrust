from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import load_configured_states, make_objective_config, output_directories
import qlt.multiple_shooting as multiple_shooting
from qlt.objective import outage_masks
from qlt.reporting import write_json


CONTINUATION_COLUMNS = [
    "case_id",
    "case_order",
    "case_group",
    "group_purpose",
    "target_mode",
    "target_generation",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "outage_lengths",
    "selected_outages",
    "outage_count",
    "selected_all_outages",
    "warm_start_from_case_id",
    "warm_start_from_phase_time",
    "warm_start_kind",
    "warm_start_source_settings_fingerprint",
    "warm_start_source_control_hash",
    "max_nfev",
    "min_recovery_segments",
    "node_initialization",
    "node_initialization_blend",
    "initial_weight",
    "defect_weight",
    "terminal_weight",
    "branch_terminal_weight",
    "branch_start_weight",
    "control_weight",
    "smooth_weight",
    "solver_mode",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "thresholds",
    "nominal_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "optimizer_success",
    "multiple_shooting_success",
    "accepted_candidate",
    "nfev",
    "runtime_seconds",
    "control_max_norm",
    "control_bound_violation",
    "nominal_fuel",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "nominal_control_path",
    "nominal_control_hash",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "message",
]

TARGET_GENERATION = (
    "non-teacher catalog_halo_phase_shift target from the JPL initial_nrho_like_l2_southern_halo "
    "source state propagated ballistically in normalized CR3BP for the configured phase_time"
)


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_hash(config: dict) -> str:
    return _json_hash(config)


def _file_identity(path: Path | str | None) -> str:
    if path is None:
        return "default"
    raw = Path(path)
    try:
        resolved = raw.resolve(strict=False)
    except OSError:
        resolved = raw
    digest = "missing"
    try:
        if raw.is_file():
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    except OSError:
        digest = "unreadable"
    return f"{resolved.as_posix()}|sha256={digest}"


def _control_hash(controls: np.ndarray) -> str:
    arr = np.asarray(controls, dtype=float)
    payload = {
        "shape": list(arr.shape),
        "controls": arr.astype(float).tolist(),
    }
    return _json_hash(payload)


def _as_list(value, cast) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [cast(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _outage_count(segments: int, block_lengths: list[int]) -> int:
    count = 0
    for length in block_lengths:
        if int(length) <= int(segments):
            count += int(segments) - int(length) + 1
    return count


def _selected_outages(raw_value, segments: int, block_lengths: list[int]) -> int:
    if isinstance(raw_value, str) and raw_value.strip().lower() in {
        "all",
        "all_masks",
        "all_configured_outage_masks",
        "all_configured_outages",
        "all_single_outage_masks",
        "all_single_outages",
    }:
        if raw_value.strip().lower() in {"all_single_outage_masks", "all_single_outages"}:
            if sorted(int(length) for length in block_lengths) != [1]:
                raise ValueError("all_single_outage_masks requires outage_lengths/block_lengths: [1]")
        return _outage_count(segments, block_lengths)
    return int(raw_value)


def _node_initialization(config: dict) -> tuple[str, float]:
    refinement = config.get("refinement", {}) or {}
    mode = str(refinement.get("node_initialization", "linear"))
    blend = float(refinement.get("node_initialization_blend", 0.5))
    return mode, blend


def _residual_weights(config: dict) -> dict[str, float]:
    configured = ((config.get("suite", {}) or {}).get("residual_weights", {}) or {})
    return {
        key: float(configured.get(key, default))
        for key, default in multiple_shooting.RESIDUAL_WEIGHT_DEFAULTS.items()
    }


def _backend_residual_weights(config: dict) -> dict[str, float]:
    row_weights = _residual_weights(config)
    return {
        "initial": row_weights["initial_weight"],
        "defect": row_weights["defect_weight"],
        "terminal": row_weights["terminal_weight"],
        "branch_terminal": row_weights["branch_terminal_weight"],
        "branch_start": row_weights["branch_start_weight"],
        "control": row_weights["control_weight"],
        "smooth": row_weights["smooth_weight"],
    }


def _suite_cases(config: dict) -> list[dict]:
    suite = config.get("suite", {}) or {}
    groups = suite.get("groups", {}) or {}
    default_benchmark = config.get("benchmark", {}) or {}
    default_outages = [int(value) for value in config.get("outages", {}).get("block_lengths", [1])]
    refinement = config.get("refinement", {}) or {}

    cases: list[dict] = []
    order = 0
    for group_name, group in groups.items():
        group_outages = [int(value) for value in group.get("outage_lengths", group.get("block_lengths", default_outages))]
        purpose = str(group.get("purpose", ""))
        for raw_case in group.get("cases", []):
            case = dict(raw_case)
            segments = int(case.get("segments", default_benchmark.get("segments")))
            outage_total = _outage_count(segments, group_outages)
            selected = _selected_outages(
                case.get("selected_outages", refinement.get("selected_outages", 1)),
                segments,
                group_outages,
            )
            if selected > outage_total:
                raise ValueError(f"selected_outages={selected} exceeds outage_count={outage_total} for {case.get('case_id')}")
            warm_from = case.get("warm_start_from_case_id")
            warm_kind = str(case.get("warm_start_kind", "nominal_controls" if warm_from else "cold"))
            cases.append(
                {
                    "case_id": str(case.get("case_id") or f"{group_name}_{order}"),
                    "case_group": str(group_name),
                    "group_purpose": purpose,
                    "phase_time": float(case.get("phase_time", default_benchmark.get("phase_time"))),
                    "transfer_time": float(case.get("transfer_time", default_benchmark.get("transfer_time"))),
                    "amax": float(case.get("amax", default_benchmark.get("amax"))),
                    "segments": segments,
                    "outage_lengths": group_outages,
                    "selected_outages": int(selected),
                    "outage_count": int(outage_total),
                    "selected_all_outages": int(selected) == int(outage_total),
                    "warm_start_from_case_id": None if warm_from in (None, "") else str(warm_from),
                    "warm_start_kind": warm_kind,
                    "max_nfev": int(case.get("max_nfev", refinement.get("max_nfev", 80))),
                    "min_recovery_segments": int(
                        case.get(
                            "min_recovery_segments",
                            refinement.get("outage_selection_min_recovery_segments", 1),
                        )
                    ),
                    "case_order": order,
                }
            )
            order += 1
    return cases


def _case_config(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["target_mode"] = "catalog_halo_phase_shift"
    benchmark["phase_time"] = float(case["phase_time"])
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    config.setdefault("outages", {})["block_lengths"] = [int(value) for value in case["outage_lengths"]]
    refinement = config.setdefault("refinement", {})
    refinement["mode"] = "multiple_shooting_branch_recovery"
    refinement["solver_mode"] = "bounded_projected_multiple_shooting"
    refinement["max_nfev"] = int(case["max_nfev"])
    refinement["selected_outages"] = int(case["selected_outages"])
    refinement["outage_selection_min_recovery_segments"] = int(case["min_recovery_segments"])
    return config


def _source_case(cases_by_id: dict[str, dict], case: dict) -> dict | None:
    source_id = case.get("warm_start_from_case_id")
    if not source_id:
        return None
    if source_id not in cases_by_id:
        raise ValueError(f"{case['case_id']} warm-start source {source_id!r} is not in this suite")
    return cases_by_id[source_id]


def _base_settings_payload(base_config: dict, source_states: Path, case: dict) -> dict:
    node_mode, node_blend = _node_initialization(base_config)
    thresholds = base_config["objective"]["thresholds"]
    return {
        "case_id": str(case["case_id"]),
        "case_order": int(case["case_order"]),
        "case_group": str(case["case_group"]),
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": [int(value) for value in case["outage_lengths"]],
        "selected_outages": int(case["selected_outages"]),
        "outage_count": int(case["outage_count"]),
        "selected_all_outages": bool(case["selected_all_outages"]),
        "warm_start_from_case_id": case.get("warm_start_from_case_id"),
        "warm_start_kind": str(case["warm_start_kind"]),
        "max_nfev": int(case["max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "node_initialization": node_mode,
        "node_initialization_blend": float(node_blend),
        "residual_weights": _residual_weights(base_config),
        "solver_mode": "bounded_projected_multiple_shooting",
        "target_mode": "catalog_halo_phase_shift",
        "thresholds": {
            "nominal_success": float(thresholds["nominal_success"]),
            "robust_success": float(thresholds["robust_success"]),
        },
        "config_hash": _config_hash(base_config),
        "source_states_id": _file_identity(source_states),
    }


def _settings_payload(
    base_config: dict,
    source_states: Path,
    case: dict,
    *,
    source_settings_fingerprint: str | None = None,
    source_control_hash: str | None = None,
) -> dict:
    payload = _base_settings_payload(base_config, source_states, case)
    payload["continuation_dependency"] = {
        "required": bool(case.get("warm_start_from_case_id")),
        "source_case_id": case.get("warm_start_from_case_id"),
        "source_settings_fingerprint": source_settings_fingerprint,
        "source_control_hash": source_control_hash,
    }
    return payload


def _settings_fingerprint(
    base_config: dict,
    source_states: Path,
    case: dict,
    *,
    source_settings_fingerprint: str | None = None,
    source_control_hash: str | None = None,
) -> str:
    return _json_hash(
        _settings_payload(
            base_config,
            source_states,
            case,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
        )
    )


def _control_sidecar_path(controls_dir: Path, case_id: str) -> Path:
    return controls_dir / f"{case_id}_nominal_controls.json"


def _write_controls(
    controls_dir: Path,
    *,
    case_id: str,
    settings_fingerprint: str,
    controls: np.ndarray,
    row: dict,
) -> dict:
    controls_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(controls, dtype=float)
    digest = _control_hash(arr)
    path = _control_sidecar_path(controls_dir, case_id)
    payload = {
        "case_id": str(case_id),
        "settings_fingerprint": str(settings_fingerprint),
        "control_hash": digest,
        "shape": list(arr.shape),
        "controls": arr.astype(float).tolist(),
        "row_summary": {
            "phase_time": float(row["phase_time"]),
            "transfer_time": float(row["transfer_time"]),
            "amax": float(row["amax"]),
            "segments": int(row["segments"]),
            "outage_lengths": json.loads(row["outage_lengths"]),
            "warm_start_kind": str(row["warm_start_kind"]),
            "warm_start_from_case_id": row.get("warm_start_from_case_id") or "",
        },
    }
    write_json(path, payload)
    return {"path": path, "control_hash": digest, "controls": arr}


def _load_controls(controls_dir: Path, case_id: str, expected_settings_fingerprint: str) -> tuple[np.ndarray, str, Path] | None:
    path = _control_sidecar_path(controls_dir, case_id)
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if str(payload.get("case_id")) != str(case_id):
            return None
        if str(payload.get("settings_fingerprint")) != str(expected_settings_fingerprint):
            return None
        controls = np.asarray(payload.get("controls"), dtype=float)
        found_hash = str(payload.get("control_hash", ""))
        if _control_hash(controls) != found_hash:
            return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return controls, found_hash, path


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=CONTINUATION_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in CONTINUATION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[CONTINUATION_COLUMNS]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _case_by_id(cases: list[dict]) -> dict[str, dict]:
    indexed = {}
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in indexed:
            raise ValueError(f"duplicate continuation case_id: {case_id}")
        indexed[case_id] = case
    return indexed


def _compatible_existing_rows(
    df: pd.DataFrame,
    *,
    base_config: dict,
    source_states: Path,
    cases: list[dict],
    controls_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict], list[dict]]:
    if df.empty:
        return df, {}, []

    cases_by_id = _case_by_id(cases)
    kept_rows: list[dict] = []
    controls_by_case_id: dict[str, dict] = {}
    rejected: list[dict] = []
    seen_case_ids: set[str] = set()

    for case in cases:
        case_id = str(case["case_id"])
        matches = [row for row in df.to_dict(orient="records") if str(row.get("case_id", "")) == case_id]
        if not matches:
            continue
        row = matches[-1]
        if case_id in seen_case_ids:
            rejected.append({"case_id": case_id, "reason": "duplicate case_id in compatible scan"})
            continue
        source = _source_case(cases_by_id, case)
        source_fingerprint = None
        source_control_hash = None
        if source is not None:
            source_info = controls_by_case_id.get(str(source["case_id"]))
            if source_info is None:
                rejected.append(
                    {
                        "case_id": case_id,
                        "reason": "warm-start source controls are unavailable or stale",
                        "source_case_id": str(source["case_id"]),
                    }
                )
                continue
            source_fingerprint = str(source_info["settings_fingerprint"])
            source_control_hash = str(source_info["control_hash"])

        expected = _settings_fingerprint(
            base_config,
            source_states,
            case,
            source_settings_fingerprint=source_fingerprint,
            source_control_hash=source_control_hash,
        )
        found = row.get("settings_fingerprint")
        if _is_missing(found) or str(found) != expected:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected,
                    "found_settings_fingerprint": None if _is_missing(found) else str(found),
                }
            )
            continue

        mismatched = []
        for field, expected_value in (
            ("config_hash", _config_hash(base_config)),
            ("source_states_id", _file_identity(source_states)),
        ):
            if _is_missing(row.get(field)) or str(row.get(field)) != str(expected_value):
                mismatched.append(field)
        if mismatched:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "provenance field mismatch",
                    "mismatched_fields": mismatched,
                    "expected_settings_fingerprint": expected,
                    "found_settings_fingerprint": str(found),
                }
            )
            continue

        loaded = _load_controls(controls_dir, case_id, expected)
        if loaded is None:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "nominal-control sidecar missing, stale, or hash mismatched",
                    "expected_settings_fingerprint": expected,
                }
            )
            continue
        controls, control_hash, control_path = loaded
        if controls.shape != (int(case["segments"]), 3):
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "nominal-control sidecar shape does not match case segments",
                    "found_shape": list(controls.shape),
                    "expected_shape": [int(case["segments"]), 3],
                }
            )
            continue

        normalized = {column: row.get(column) for column in CONTINUATION_COLUMNS}
        normalized["case_order"] = int(case["case_order"])
        normalized["nominal_control_path"] = control_path.as_posix()
        normalized["nominal_control_hash"] = control_hash
        kept_rows.append(normalized)
        controls_by_case_id[case_id] = {
            "controls": controls,
            "control_hash": control_hash,
            "path": control_path,
            "settings_fingerprint": expected,
            "row": normalized,
        }
        seen_case_ids.add(case_id)

    existing_case_ids = set(str(row.get("case_id", "")) for row in df.to_dict(orient="records"))
    for case_id in sorted(existing_case_ids - set(cases_by_id)):
        rejected.append({"case_id": case_id, "reason": "case is not in the current requested suite"})

    return pd.DataFrame(kept_rows, columns=CONTINUATION_COLUMNS), controls_by_case_id, rejected


def _row_from_result(
    *,
    base_config: dict,
    source_states: Path,
    case: dict,
    result: dict,
    settings_fingerprint: str,
    source_settings_fingerprint: str | None,
    source_control_hash: str | None,
) -> dict:
    thresholds = base_config["objective"]["thresholds"]
    nominal_threshold = float(thresholds["nominal_success"])
    robust_threshold = float(thresholds["robust_success"])
    row_weights = _residual_weights(base_config)
    node_mode, node_blend = _node_initialization(base_config)
    source_phase = np.nan
    if case.get("warm_start_from_case_id"):
        source_phase = float(case.get("warm_start_from_phase_time", np.nan))
    nominal_error = float(result["nominal_error"])
    selected_worst = float(result["selected_worst_error"])
    row = {
        "case_id": str(case["case_id"]),
        "case_order": int(case["case_order"]),
        "case_group": str(case["case_group"]),
        "group_purpose": str(case["group_purpose"]),
        "target_mode": "catalog_halo_phase_shift",
        "target_generation": TARGET_GENERATION,
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": json.dumps([int(value) for value in case["outage_lengths"]]),
        "selected_outages": int(case["selected_outages"]),
        "outage_count": int(case["outage_count"]),
        "selected_all_outages": bool(case["selected_all_outages"]),
        "warm_start_from_case_id": case.get("warm_start_from_case_id") or "",
        "warm_start_from_phase_time": source_phase,
        "warm_start_kind": str(case["warm_start_kind"]),
        "warm_start_source_settings_fingerprint": source_settings_fingerprint or "",
        "warm_start_source_control_hash": source_control_hash or "",
        "max_nfev": int(case["max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "node_initialization": str(result.get("node_initialization", node_mode)),
        "node_initialization_blend": float(result.get("node_initialization_blend", node_blend)),
        **row_weights,
        "solver_mode": str(result.get("solver_mode", "bounded_projected_multiple_shooting")),
        "nominal_error": nominal_error,
        "selected_worst_error": selected_worst,
        "all_mask_worst_error": float(result["all_mask_worst_error"]),
        "thresholds": json.dumps(
            {"nominal_success": nominal_threshold, "robust_success": robust_threshold},
            sort_keys=True,
        ),
        "nominal_threshold": nominal_threshold,
        "selected_worst_threshold": robust_threshold,
        "meets_nominal_threshold": bool(nominal_error <= nominal_threshold),
        "meets_selected_worst_threshold": bool(selected_worst <= robust_threshold),
        "meets_thresholds": bool(nominal_error <= nominal_threshold and selected_worst <= robust_threshold),
        "optimizer_success": bool(result["optimizer_success"]),
        "multiple_shooting_success": bool(result["success"]),
        "accepted_candidate": str(result.get("accepted_candidate", "optimizer")),
        "nfev": int(result["nfev"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "control_max_norm": float(result["control_max_norm"]),
        "control_bound_violation": float(result["control_bound_violation"]),
        "nominal_fuel": float(result["nominal_fuel"]),
        "selected_outage_indices": json.dumps(result["selected_outage_indices"]),
        "selected_outage_errors": json.dumps(result["selected_outage_errors"]),
        "all_outage_errors": json.dumps(result["all_outage_errors"]),
        "nominal_control_path": "",
        "nominal_control_hash": "",
        "settings_fingerprint": settings_fingerprint,
        "config_hash": _config_hash(base_config),
        "source_states_id": _file_identity(source_states),
        "message": str(result["message"]),
    }
    return {column: row.get(column) for column in CONTINUATION_COLUMNS}


def _run_case(
    *,
    base_config: dict,
    source_states: Path,
    case: dict,
    source_info: dict | None,
) -> tuple[dict, np.ndarray]:
    case_config = _case_config(base_config, case)
    states = load_configured_states(Path.cwd(), case_config, source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = case_config["objective"]["thresholds"]
    node_mode, node_blend = _node_initialization(base_config)
    warm_start_info = {
        "enabled": source_info is not None,
        "kind": str(case["warm_start_kind"]),
        "source_case_id": case.get("warm_start_from_case_id"),
        "source_settings_fingerprint": None if source_info is None else str(source_info["settings_fingerprint"]),
        "source_control_hash": None if source_info is None else str(source_info["control_hash"]),
        "continuous_backend_baseline": True,
    }
    nominal_guess = None if source_info is None else np.asarray(source_info["controls"], dtype=float)
    if nominal_guess is not None and nominal_guess.shape != (cfg.n_segments, 3):
        raise ValueError(
            f"{case['case_id']} warm-start controls shape {nominal_guess.shape} does not match {(cfg.n_segments, 3)}"
        )
    result = multiple_shooting.run_multiple_shooting_baseline(
        state0=states.initial,
        target=states.target,
        cfg=cfg,
        masks=masks,
        thresholds=thresholds,
        selected_outages=int(case["selected_outages"]),
        max_nfev=int(case["max_nfev"]),
        min_recovery_segments=int(case["min_recovery_segments"]),
        residual_weights=_backend_residual_weights(base_config),
        nominal_control_guess=nominal_guess,
        selected_branch_control_guesses=None,
        node_initialization=node_mode,
        node_initialization_blend=node_blend,
        warm_start_info=warm_start_info,
    )
    return result, np.asarray(result["controls"], dtype=float)


def _table_group_label(value: str) -> str:
    labels = {
        "phase_continuation_all_single": "All single outages",
        "two_segment_all_mask_diagnostic": "All one/two-segment outages",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values("case_order")
    table = ordered[
        [
            "case_group",
            "phase_time",
            "warm_start_kind",
            "warm_start_from_phase_time",
            "segments",
            "selected_outages",
            "outage_count",
            "max_nfev",
            "nominal_error",
            "selected_worst_error",
            "all_mask_worst_error",
            "optimizer_success",
            "multiple_shooting_success",
            "nfev",
            "runtime_seconds",
            "meets_thresholds",
        ]
    ].copy()
    table["case_group"] = table["case_group"].map(_table_group_label)
    table["warm_start_kind"] = table.apply(
        lambda row: "cold"
        if str(row["warm_start_kind"]) == "cold"
        else f"from {float(row['warm_start_from_phase_time']):.1f}",
        axis=1,
    )
    table["selected_outages"] = table["selected_outages"].astype(int).astype(str) + "/" + table["outage_count"].astype(int).astype(str)
    table = table.drop(columns=["outage_count", "warm_start_from_phase_time"])
    table.columns = [
        "Group",
        "Phase time",
        "Warm start",
        "Segments",
        "Outage masks",
        "Max nfev",
        "Nominal error",
        "Selected worst error",
        "All-mask worst error",
        "Optimizer success",
        "MS success",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
    ]
    (tables_dir / "continuation_margin_suite_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def _write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    groups = [
        ("phase_continuation_all_single", "All single outages"),
        ("two_segment_all_mask_diagnostic", "All one/two-segment outages"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False)
    for ax, (group_name, title) in zip(axes, groups):
        group = df[df["case_group"] == group_name].sort_values("case_order")
        if group.empty:
            ax.set_axis_off()
            ax.set_title(title)
            continue
        x = np.arange(len(group))
        colors = ["tab:blue" if str(kind) == "cold" else "tab:orange" for kind in group["warm_start_kind"]]
        ax.bar(x, group["selected_worst_error"].astype(float), color=colors, alpha=0.72, label="selected/all-mask worst")
        ax.plot(x, group["nominal_error"].astype(float), color="black", marker="o", linewidth=1.4, label="nominal")
        for index, (_, row) in enumerate(group.iterrows()):
            marker = "pass" if bool(row["meets_thresholds"]) else "fail"
            ax.text(index, float(row["selected_worst_error"]) + 0.012, marker, ha="center", fontsize=8)
        ax.axhline(float(group["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(float(group["selected_worst_threshold"].iloc[0]), color="0.25", linestyle=":", linewidth=1.0)
        labels = []
        for _, row in group.iterrows():
            if str(row["warm_start_kind"]) == "cold":
                labels.append(f"p={float(row['phase_time']):.1f}\ncold")
            else:
                labels.append(f"p={float(row['phase_time']):.1f}\nfrom {float(row['warm_start_from_phase_time']):.1f}")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel("Normalized terminal error")
        ax.grid(axis="y", alpha=0.25)
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", linewidth=1.4, label="nominal"),
            Patch(facecolor="tab:blue", alpha=0.72, label="cold selected/all-mask worst"),
            Patch(facecolor="tab:orange", alpha=0.72, label="warm-start selected/all-mask worst"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"continuation_margin_suite{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def _metadata(
    df: pd.DataFrame,
    config: dict,
    command: str,
    cases: list[dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> dict:
    feasible_count = int(df["meets_thresholds"].astype(bool).sum()) if not df.empty else 0
    return {
        "command": command,
        "row_count": int(len(df)),
        "feasible_row_count": feasible_count,
        "config": config,
        "expected_case_count": int(len(cases)),
        "completed_case_count": int(len(df)),
        "skipped_cases": skipped_cases,
        "resume_rejected_rows": resume_rejected_rows,
        "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_worst_error <= robust_success",
        "semantics": {
            "continuation": (
                "Continuation rows pass the previous row's persisted nominal controls to "
                "qlt.multiple_shooting.run_multiple_shooting_baseline as nominal_control_guess."
            ),
            "backend": (
                "This is a continuous-backend direct multiple-shooting implementation baseline; "
                "it is not a quantum, QUBO, QAOA, or discrete schedule-search result."
            ),
            "all_mask": (
                "all_mask_worst_error and all_outage_errors evaluate all configured outage masks for that row. "
                "Because selected_all_outages is true in this suite, selected_worst_error uses the same configured mask set."
            ),
            "two_segment_diagnostic": (
                "The two_segment_all_mask_diagnostic group uses N=6, which is intentionally smaller and coarser "
                "than the N=8 one-segment continuation group."
            ),
        },
        "limitations": [
            "Normalized Earth-Moon CR3BP only.",
            "No flight-ready trajectory optimization or mission-design claim is made.",
            "Rows are optimizer outcomes for reproducible evidence; skipped or missing cases are not extrapolated.",
            "Warm starts use only persisted nominal controls from the named source row, not teacher or oracle controls.",
        ],
        "target_generation": TARGET_GENERATION,
        "expected_cases": cases,
        "rows": df.to_dict(orient="records"),
    }


def _regenerate(
    df: pd.DataFrame,
    *,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    cases: list[dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> None:
    df = pd.DataFrame(df, columns=CONTINUATION_COLUMNS)
    df.to_csv(results_dir / "continuation_margin_suite.csv", index=False)
    _write_table(df, tables_dir)
    _write_plot(df, figures_dir)
    write_json(
        results_dir / "continuation_margin_suite_metadata.json",
        _metadata(df, config, command, cases, resume_rejected_rows, skipped_cases),
    )


def _runtime_budget(args, config: dict) -> float | None:
    if args.runtime_budget_seconds is not None:
        return float(args.runtime_budget_seconds)
    suite_value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    if suite_value is None:
        return None
    return float(suite_value)


def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    controls_dir = results_dir / "controls"
    for directory in (results_dir, figures_dir, tables_dir, controls_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "continuation_margin_suite.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    cases_by_id = _case_by_id(cases)

    if args.resume:
        loaded = _load_existing(csv_path)
        df, controls_by_case_id, resume_rejected_rows = _compatible_existing_rows(
            loaded,
            base_config=config,
            source_states=args.source_states,
            cases=cases,
            controls_dir=controls_dir,
        )
    else:
        df = pd.DataFrame(columns=CONTINUATION_COLUMNS)
        controls_by_case_id = {}
        resume_rejected_rows = []

    completed_fingerprints = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())
    command = " ".join(sys.argv)
    skipped_cases: list[dict] = []
    budget = _runtime_budget(args, config)
    start = time.perf_counter()
    active_stack: set[str] = set()

    def run_or_skip(case: dict) -> None:
        nonlocal df
        case_id = str(case["case_id"])
        if case_id in active_stack:
            raise RuntimeError(f"cycle in continuation dependencies at {case_id}")
        active_stack.add(case_id)
        source_info = None
        source_settings_fingerprint = None
        source_control_hash = None
        source_case = _source_case(cases_by_id, case)
        if source_case is not None:
            run_or_skip(source_case)
            source_info = controls_by_case_id.get(str(source_case["case_id"]))
            if source_info is None:
                skipped_cases.append(
                    {
                        "case_id": case_id,
                        "case_group": str(case["case_group"]),
                        "phase_time": float(case["phase_time"]),
                        "reason": "warm-start source controls are unavailable",
                        "source_case_id": str(source_case["case_id"]),
                    }
                )
                active_stack.remove(case_id)
                return
            source_settings_fingerprint = str(source_info["settings_fingerprint"])
            source_control_hash = str(source_info["control_hash"])
            case["warm_start_from_phase_time"] = float(source_case["phase_time"])

        settings_fingerprint = _settings_fingerprint(
            config,
            args.source_states,
            case,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
        )
        if settings_fingerprint in completed_fingerprints and case_id in controls_by_case_id:
            active_stack.remove(case_id)
            return

        elapsed = time.perf_counter() - start
        if budget is not None and elapsed >= budget:
            skipped_cases.append(
                {
                    "case_id": case_id,
                    "case_group": str(case["case_group"]),
                    "phase_time": float(case["phase_time"]),
                    "reason": "runtime budget reached before launching case",
                    "elapsed_seconds": elapsed,
                    "runtime_budget_seconds": budget,
                }
            )
            active_stack.remove(case_id)
            return

        result, controls = _run_case(
            base_config=config,
            source_states=args.source_states,
            case=case,
            source_info=source_info,
        )
        row = _row_from_result(
            base_config=config,
            source_states=args.source_states,
            case=case,
            result=result,
            settings_fingerprint=settings_fingerprint,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
        )
        sidecar = _write_controls(
            controls_dir,
            case_id=case_id,
            settings_fingerprint=settings_fingerprint,
            controls=controls,
            row=row,
        )
        row["nominal_control_path"] = sidecar["path"].as_posix()
        row["nominal_control_hash"] = sidecar["control_hash"]
        controls_by_case_id[case_id] = {
            "controls": sidecar["controls"],
            "control_hash": sidecar["control_hash"],
            "path": sidecar["path"],
            "settings_fingerprint": settings_fingerprint,
            "row": row,
        }
        if case_id in set(str(value) for value in df["case_id"].dropna().tolist()):
            df = df[df["case_id"].astype(str) != case_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed_fingerprints.add(settings_fingerprint)
        _regenerate(
            df,
            results_dir=results_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            config=config,
            command=command,
            cases=cases,
            resume_rejected_rows=resume_rejected_rows,
            skipped_cases=skipped_cases,
        )
        print(
            f"{case['case_group']} {case_id} phase={case['phase_time']} "
            f"warm={row['warm_start_kind']} selected={row['selected_outages']}/{row['outage_count']}: "
            f"nominal={row['nominal_error']:.6f}, selected={row['selected_worst_error']:.6f}, "
            f"all={row['all_mask_worst_error']:.6f}, met={row['meets_thresholds']}, "
            f"opt={row['optimizer_success']}, nfev={row['nfev']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )
        active_stack.remove(case_id)

    for case in cases:
        run_or_skip(case)

    df = pd.DataFrame(df, columns=CONTINUATION_COLUMNS)
    if not df.empty:
        order = {str(case["case_id"]): int(case["case_order"]) for case in cases}
        df["_case_order"] = df["case_id"].map(order)
        df = df.sort_values("_case_order").drop(columns=["_case_order"]).reset_index(drop=True)
    _regenerate(
        df,
        results_dir=results_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        config=config,
        command=command,
        cases=cases,
        resume_rejected_rows=resume_rejected_rows,
        skipped_cases=skipped_cases,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuation warm-start margin suite for direct multiple shooting.")
    parser.add_argument("--config", type=Path, default=Path("configs/continuation_margin_suite.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
