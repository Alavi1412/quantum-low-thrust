"""Hermite-Simpson direct-collocation continuation baseline.

Uses ``qlt.direct_collocation.run_direct_collocation_baseline`` with
``method: hermite_simpson``.  Each warm-start case receives the persisted
nominal controls from its named source case as ``nominal_control_guess``.

This is a continuous-backend Hermite-Simpson direct-collocation baseline/probe
with persisted nominal-control warm starts and trajectory stacking.
It is NOT a quantum, QUBO, QAOA, or discrete schedule-search result.
"""
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.direct_collocation import (
    collocation_method_uses_midpoint_controls,
    config_hash as _dc_config_hash,
    file_identity as _dc_file_identity,
    run_direct_collocation_baseline,
    settings_fingerprint as _dc_settings_fingerprint,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import outage_masks
from qlt.reporting import sanitize_json, write_json


SUITE_NAME = "hermite_simpson_continuation_baseline"
ARTIFACT_STEM = "hermite_simpson_continuation_baseline"
DEFAULT_CONFIG_PATH = Path("configs/hermite_simpson_continuation_baseline.yaml")
DEFAULT_COLLOCATION_METHOD = "hermite_simpson"
FORCE_COLLOCATION_METHOD: str | None = None
SCRIPT_DESCRIPTION = (
    "Hermite-Simpson direct-collocation continuation baseline. "
    "Continuous-backend probe; not a quantum/discrete result."
)
METADATA_BACKEND_SEMANTICS = (
    "This is a continuous-backend Hermite-Simpson direct-collocation implementation baseline/probe "
    "with persisted nominal-control warm starts and trajectory stacking; "
    "it is NOT a quantum, QUBO, QAOA, or discrete schedule-search result."
)
METADATA_METHOD_KEY = "hermite_simpson"
METADATA_METHOD_SEMANTICS = (
    "Hermite-Simpson constant-control direct transcription; "
    "controls are held constant over each segment and no independent midpoint control "
    "variables are optimized."
)
METADATA_LIMITATIONS = [
    "Normalized Earth-Moon CR3BP only; not a flight-ready trajectory optimization.",
    "Constant-control Hermite-Simpson scheme; no independent midpoint control variables.",
    "Warm starts use only persisted nominal controls from the named source row.",
    "Failed/negative diagnostics are preserved honestly and not extrapolated.",
    "The optimizer may stop at max_nfev; such rows are retained with optimizer_success=False.",
]


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------
HS_CONTINUATION_COLUMNS = [
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
    "method_type",
    "collocation_method",
    "collocation_scheme_semantics",
    "fuel_quadrature",
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
    "direct_collocation_success",
    "cost",
    "optimality",
    "nfev",
    "runtime_seconds",
    "control_max_norm",
    "control_bound_violation",
    "nominal_fuel",
    "recovery_fuel_mean",
    "recovery_fuel_max",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "nominal_control_path",
    "nominal_control_hash",
    "nominal_control_sidecar_hash",
    "nominal_endpoint_control_hash",
    "nominal_midpoint_control_present",
    "nominal_midpoint_control_hash",
    "nominal_midpoint_warm_start_loaded",
    "branch_control_manifest_path",
    "branch_control_manifest_hash",
    "branch_control_sidecar_count",
    "branch_control_replay_ready",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "message",
]

TARGET_GENERATION = (
    "non-teacher catalog_halo_phase_shift target from the JPL initial_nrho_like_l2_southern_halo "
    "source state propagated ballistically in normalized CR3BP for the configured phase_time"
)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd(), ROOT):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _resolve_artifact_path(value: object) -> Path:
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


def _write_deterministic_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = sanitize_json(payload)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return _sha256(path)


def _control_norm_diagnostics(controls: np.ndarray, amax: float) -> dict[str, object]:
    arr = np.asarray(controls, dtype=float)
    if arr.size == 0:
        norms = np.zeros(0, dtype=float)
    else:
        norms = np.linalg.norm(arr.reshape((-1, 3)), axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    mean_norm = float(np.mean(norms)) if norms.size else 0.0
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    violation = max(0.0, max_norm - float(amax))
    if violation <= 1.0e-12:
        violation = 0.0
        max_norm = min(max_norm, float(amax))
    return {
        "shape": list(arr.shape),
        "control_norm_max": max_norm,
        "control_norm_mean": mean_norm,
        "control_max_abs_component": max_abs,
        "amax": float(amax),
        "control_bound_violation": violation,
    }


def _combined_control_sidecar_hash(controls: np.ndarray, midpoint_controls: np.ndarray | None) -> str:
    endpoint_hash = _control_hash(controls)
    if midpoint_controls is None:
        return endpoint_hash
    midpoint_arr = np.asarray(midpoint_controls, dtype=float)
    payload = {
        "schema_version": 2,
        "endpoint_control_hash": endpoint_hash,
        "midpoint_control_hash": _control_hash(midpoint_arr),
        "endpoint_shape": list(np.asarray(controls, dtype=float).shape),
        "midpoint_shape": list(midpoint_arr.shape),
    }
    return _json_hash(payload)


class ControlSidecar:
    def __init__(
        self,
        *,
        controls: np.ndarray,
        control_hash: str,
        path: Path,
        endpoint_control_hash: str,
        midpoint_controls: np.ndarray | None = None,
        midpoint_control_hash: str = "",
        midpoint_control_present: bool = False,
    ):
        self.controls = controls
        self.control_hash = control_hash
        self.path = path
        self.endpoint_control_hash = endpoint_control_hash
        self.midpoint_controls = midpoint_controls
        self.midpoint_control_hash = midpoint_control_hash
        self.midpoint_control_present = midpoint_control_present

    def __iter__(self):
        yield self.controls
        yield self.control_hash
        yield self.path


# ---------------------------------------------------------------------------
# Case loading helpers
# ---------------------------------------------------------------------------

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


def _suite_cases(config: dict) -> list[dict]:
    suite = config.get("suite", {}) or {}
    groups = suite.get("groups", {}) or {}
    default_benchmark = config.get("benchmark", {}) or {}
    default_outages = [int(v) for v in config.get("outages", {}).get("block_lengths", [1])]
    direct = config.get("direct_collocation", {}) or {}
    branch_persistence_default = bool(
        direct.get("persist_branch_controls", suite.get("persist_branch_controls", False))
    )

    cases: list[dict] = []
    order = 0
    for group_name, group in groups.items():
        group_outages = [int(v) for v in group.get("outage_lengths", group.get("block_lengths", default_outages))]
        purpose = str(group.get("purpose", ""))
        group_persist_branch_controls = bool(group.get("persist_branch_controls", branch_persistence_default))
        for raw_case in group.get("cases", []):
            case = dict(raw_case)
            target_mode = str(
                case.get(
                    "target_mode",
                    group.get("target_mode", default_benchmark.get("target_mode", "catalog_halo_phase_shift")),
                )
            )
            segments = int(case.get("segments", default_benchmark.get("segments")))
            outage_total = _outage_count(segments, group_outages)
            selected = _selected_outages(
                case.get("selected_outages", direct.get("selected_outages", 1)),
                segments,
                group_outages,
            )
            if selected > outage_total:
                raise ValueError(
                    f"selected_outages={selected} exceeds outage_count={outage_total} for {case.get('case_id')}"
                )
            warm_from = case.get("warm_start_from_case_id")
            warm_kind = str(case.get("warm_start_kind", "nominal_controls" if warm_from else "cold"))
            branch_guess_from = case.get("branch_control_guess_from_case_id")
            cases.append(
                {
                    "case_id": str(case.get("case_id") or f"{group_name}_{order}"),
                    "case_group": str(group_name),
                    "group_purpose": purpose,
                    "target_mode": target_mode,
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
                    "branch_control_guess_from_case_id": (
                        None if branch_guess_from in (None, "") else str(branch_guess_from)
                    ),
                    "persist_branch_controls": bool(
                        case.get("persist_branch_controls", group_persist_branch_controls)
                    ),
                    "max_nfev": int(case.get("max_nfev", direct.get("max_nfev", 50))),
                    "min_recovery_segments": int(
                        case.get("min_recovery_segments", direct.get("min_recovery_segments", 1))
                    ),
                    "case_order": order,
                }
            )
            order += 1
    return cases


def _case_by_id(cases: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in indexed:
            raise ValueError(f"duplicate HS continuation case_id: {case_id}")
        indexed[case_id] = case
    return indexed


def _source_case(cases_by_id: dict[str, dict], case: dict) -> dict | None:
    source_id = case.get("warm_start_from_case_id")
    if not source_id:
        return None
    if source_id not in cases_by_id:
        raise ValueError(f"{case['case_id']} warm-start source {source_id!r} is not in this suite")
    return cases_by_id[source_id]


def _branch_control_guess_source_case(cases_by_id: dict[str, dict], case: dict) -> dict | None:
    source_id = case.get("branch_control_guess_from_case_id")
    if not source_id:
        return None
    if source_id not in cases_by_id:
        raise ValueError(f"{case['case_id']} branch-control guess source {source_id!r} is not in this suite")
    return cases_by_id[source_id]


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _case_config(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["target_mode"] = str(case.get("target_mode", "catalog_halo_phase_shift"))
    benchmark["phase_time"] = float(case["phase_time"])
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    config.setdefault("outages", {})["block_lengths"] = [int(v) for v in case["outage_lengths"]]
    return config


def _dc_params(base_config: dict) -> dict:
    """Extract the direct-collocation parameters used in the settings fingerprint."""
    direct = base_config.get("direct_collocation", {}) or {}
    return {
        "method": str(direct.get("method", DEFAULT_COLLOCATION_METHOD)),
        "node_initialization": str(direct.get("node_initialization", "blend")),
        "node_initialization_blend": float(direct.get("node_initialization_blend", 0.35)),
        "xtol": float(direct.get("xtol", 1e-5)),
        "ftol": float(direct.get("ftol", 1e-5)),
        "gtol": float(direct.get("gtol", 1e-5)),
        "weights": {
            k: float(v)
            for k, v in (direct.get("weights", {}) or {}).items()
        },
    }


def _requires_midpoint_controls(base_config: dict) -> bool:
    return collocation_method_uses_midpoint_controls(_dc_params(base_config)["method"])


def _base_settings_payload(base_config: dict, source_states: Path, case: dict) -> dict:
    thresholds = base_config["objective"]["thresholds"]
    dc_params = _dc_params(base_config)
    return {
        "suite": SUITE_NAME,
        "case_id": str(case["case_id"]),
        "case_order": int(case["case_order"]),
        "case_group": str(case["case_group"]),
        "target_mode": str(case.get("target_mode", "catalog_halo_phase_shift")),
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": [int(v) for v in case["outage_lengths"]],
        "selected_outages": int(case["selected_outages"]),
        "outage_count": int(case["outage_count"]),
        "selected_all_outages": bool(case["selected_all_outages"]),
        "warm_start_from_case_id": case.get("warm_start_from_case_id"),
        "warm_start_kind": str(case["warm_start_kind"]),
        "max_nfev": int(case["max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "direct_collocation": dc_params,
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
    source_branch_control_manifest_hash: str | None = None,
) -> dict:
    payload = _base_settings_payload(base_config, source_states, case)
    payload["continuation_dependency"] = {
        "required": bool(case.get("warm_start_from_case_id")),
        "source_case_id": case.get("warm_start_from_case_id"),
        "source_settings_fingerprint": source_settings_fingerprint,
        "source_control_hash": source_control_hash,
    }
    payload["branch_control_guess_dependency"] = {
        "required": bool(case.get("branch_control_guess_from_case_id")),
        "source_case_id": case.get("branch_control_guess_from_case_id"),
        "source_branch_control_manifest_hash": source_branch_control_manifest_hash,
    }
    return payload


def compute_settings_fingerprint(
    base_config: dict,
    source_states: Path,
    case: dict,
    *,
    source_settings_fingerprint: str | None = None,
    source_control_hash: str | None = None,
    source_branch_control_manifest_hash: str | None = None,
) -> str:
    return _json_hash(
        _settings_payload(
            base_config,
            source_states,
            case,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
            source_branch_control_manifest_hash=source_branch_control_manifest_hash,
        )
    )


# ---------------------------------------------------------------------------
# Control sidecar persistence
# ---------------------------------------------------------------------------

def _control_sidecar_path(controls_dir: Path, case_id: str) -> Path:
    return controls_dir / f"{case_id}_nominal_controls.json"


def write_control_sidecar(
    controls_dir: Path,
    *,
    case_id: str,
    settings_fingerprint: str,
    controls: np.ndarray,
    midpoint_controls: np.ndarray | None = None,
    row: dict,
) -> dict:
    controls_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(controls, dtype=float)
    endpoint_digest = _control_hash(arr)
    midpoint_arr = None if midpoint_controls is None else np.asarray(midpoint_controls, dtype=float)
    if midpoint_arr is not None and midpoint_arr.shape != arr.shape:
        raise ValueError(
            f"midpoint_controls shape {midpoint_arr.shape} does not match endpoint controls shape {arr.shape}"
        )
    midpoint_digest = "" if midpoint_arr is None else _control_hash(midpoint_arr)
    sidecar_digest = _combined_control_sidecar_hash(arr, midpoint_arr)
    path = _control_sidecar_path(controls_dir, case_id)
    payload = {
        "schema_version": 2,
        "case_id": str(case_id),
        "settings_fingerprint": str(settings_fingerprint),
        "control_hash": endpoint_digest,
        "endpoint_control_hash": endpoint_digest,
        "midpoint_control_present": midpoint_arr is not None,
        "midpoint_control_hash": midpoint_digest,
        "sidecar_hash": sidecar_digest,
        "shape": list(arr.shape),
        "controls": arr.astype(float).tolist(),
        "nominal_endpoint_controls": arr.astype(float).tolist(),
        "nominal_midpoint_controls": None if midpoint_arr is None else midpoint_arr.astype(float).tolist(),
        "row_summary": {
            "phase_time": float(row.get("phase_time", 0.0)),
            "transfer_time": float(row.get("transfer_time", 0.0)),
            "amax": float(row.get("amax", 0.0)),
            "segments": int(row.get("segments", 0)),
            "outage_lengths": json.loads(str(row.get("outage_lengths", "[1]"))),
            "warm_start_kind": str(row.get("warm_start_kind", "cold")),
            "warm_start_from_case_id": row.get("warm_start_from_case_id") or "",
            "collocation_method": str(row.get("collocation_method", DEFAULT_COLLOCATION_METHOD)),
        },
    }
    write_json(path, payload)
    return {
        "path": path,
        "control_hash": sidecar_digest,
        "sidecar_hash": sidecar_digest,
        "endpoint_control_hash": endpoint_digest,
        "midpoint_control_hash": midpoint_digest,
        "midpoint_control_present": midpoint_arr is not None,
        "controls": arr,
        "midpoint_controls": midpoint_arr,
    }


def load_control_sidecar(
    controls_dir: Path,
    case_id: str,
    expected_settings_fingerprint: str,
    *,
    require_midpoint_controls: bool = False,
) -> ControlSidecar | None:
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
        raw_controls = payload.get("nominal_endpoint_controls", payload.get("controls"))
        controls = np.asarray(raw_controls, dtype=float)
        endpoint_hash = str(payload.get("endpoint_control_hash", payload.get("control_hash", "")))
        legacy_control_hash = str(payload.get("control_hash", endpoint_hash))
        if legacy_control_hash != endpoint_hash:
            return None
        if _control_hash(controls) != endpoint_hash:
            return None
        midpoint_present = bool(payload.get("midpoint_control_present", False))
        midpoint_controls: np.ndarray | None = None
        midpoint_hash = ""
        if midpoint_present:
            if payload.get("nominal_midpoint_controls") is None:
                return None
            midpoint_controls = np.asarray(payload.get("nominal_midpoint_controls"), dtype=float)
            if midpoint_controls.shape != controls.shape:
                return None
            midpoint_hash = str(payload.get("midpoint_control_hash", ""))
            if _control_hash(midpoint_controls) != midpoint_hash:
                return None
        if require_midpoint_controls and midpoint_controls is None:
            return None
        sidecar_hash = str(payload.get("sidecar_hash", ""))
        expected_sidecar_hash = _combined_control_sidecar_hash(controls, midpoint_controls)
        if not sidecar_hash:
            sidecar_hash = endpoint_hash
        if sidecar_hash != expected_sidecar_hash:
            return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return ControlSidecar(
        controls=controls,
        control_hash=sidecar_hash,
        path=path,
        endpoint_control_hash=endpoint_hash,
        midpoint_controls=midpoint_controls,
        midpoint_control_hash=midpoint_hash,
        midpoint_control_present=midpoint_controls is not None,
    )


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    if _is_missing(value):
        return False
    return bool(value)


def _row_from_result(
    *,
    base_config: dict,
    source_states: Path,
    case: dict,
    result: dict,
    settings_fp: str,
    source_settings_fingerprint: str | None,
    source_control_hash: str | None,
) -> dict:
    thresholds = base_config["objective"]["thresholds"]
    nominal_threshold = float(thresholds["nominal_success"])
    robust_threshold = float(thresholds["robust_success"])
    dc_params = _dc_params(base_config)
    source_phase = float("nan")
    if case.get("warm_start_from_case_id"):
        source_phase = float(case.get("warm_start_from_phase_time", float("nan")))
    nominal_error = float(result["nominal_error"])
    selected_worst = float(result["selected_worst_error"])
    row = {
        "case_id": str(case["case_id"]),
        "case_order": int(case["case_order"]),
        "case_group": str(case["case_group"]),
        "group_purpose": str(case["group_purpose"]),
        "target_mode": str(case.get("target_mode", "catalog_halo_phase_shift")),
        "target_generation": str(result.get("target_generation", TARGET_GENERATION)),
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": json.dumps([int(v) for v in case["outage_lengths"]]),
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
        "node_initialization": str(dc_params["node_initialization"]),
        "node_initialization_blend": float(dc_params["node_initialization_blend"]),
        "method_type": str(result.get("method_type", "")),
        "collocation_method": str(result.get("collocation_method", DEFAULT_COLLOCATION_METHOD)),
        "collocation_scheme_semantics": str(result.get("collocation_scheme_semantics", "")),
        "fuel_quadrature": str(result.get("fuel_quadrature", "")),
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
        "meets_thresholds": bool(
            nominal_error <= nominal_threshold and selected_worst <= robust_threshold
        ),
        "optimizer_success": bool(result["optimizer_success"]),
        "direct_collocation_success": bool(result["success"]),
        "cost": float(result.get("cost", float("nan"))),
        "optimality": float(result.get("optimality", float("nan"))),
        "nfev": int(result["nfev"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "control_max_norm": float(result["control_max_norm"]),
        "control_bound_violation": float(result["control_bound_violation"]),
        "nominal_fuel": float(result["nominal_fuel"]),
        "recovery_fuel_mean": float(result.get("recovery_fuel_mean", float("nan"))),
        "recovery_fuel_max": float(result.get("recovery_fuel_max", float("nan"))),
        "selected_outage_indices": json.dumps(result["selected_outage_indices"]),
        "selected_outage_errors": json.dumps(result["selected_outage_errors"]),
        "all_outage_errors": json.dumps(result["all_outage_errors"]),
        "selected_branch_semantics": str(result.get("selected_branch_semantics", "")),
        "all_mask_diagnostic_semantics": str(result.get("all_mask_diagnostic_semantics", "")),
        "control_bound_semantics": str(result.get("control_bound_semantics", "")),
        "nominal_control_path": "",
        "nominal_control_hash": "",
        "nominal_control_sidecar_hash": "",
        "nominal_endpoint_control_hash": "",
        "nominal_midpoint_control_present": bool(result.get("nominal_midpoint_controls") is not None),
        "nominal_midpoint_control_hash": "",
        "nominal_midpoint_warm_start_loaded": bool(
            (result.get("warm_start_info") or {}).get("source_midpoint_control_present", False)
        ),
        "branch_control_manifest_path": "",
        "branch_control_manifest_hash": "",
        "branch_control_sidecar_count": 0,
        "branch_control_replay_ready": False,
        "settings_fingerprint": settings_fp,
        "config_hash": _config_hash(base_config),
        "source_states_id": _file_identity(source_states),
        "message": str(result.get("message", "")),
    }
    return {column: row.get(column) for column in HS_CONTINUATION_COLUMNS}


# ---------------------------------------------------------------------------
# Branch-control sidecar persistence
# ---------------------------------------------------------------------------

def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _outage_end(mask: np.ndarray) -> int:
    missed = np.flatnonzero(np.asarray(mask, dtype=float) < 0.5)
    if missed.size == 0:
        return 0
    return int(missed[-1] + 1)


def _branch_control_replay_limitations() -> list[str]:
    return [
        "Branch-control replay repropagates persisted controls in the normalized CR3BP model only.",
        "Replay is deterministic postprocessing and does not rerun least-squares optimization.",
        "The sidecars do not establish high-fidelity validation, SPICE/ephemeris propagation, production solver parity, fuel optimality, or quantum advantage.",
        "All-configured evidence is limited to the outage family, target, transfer time, segment grid, thrust bound, and thresholds recorded in the manifest.",
    ]


def _empty_branch_sidecar_columns() -> dict[str, object]:
    return {
        "branch_control_manifest_path": "",
        "branch_control_manifest_hash": "",
        "branch_control_sidecar_count": 0,
        "branch_control_replay_ready": False,
    }


class BranchControlSidecarStore:
    def __init__(
        self,
        *,
        results_dir: Path,
        case: dict,
        case_config: dict,
        states,
        cfg,
        masks: np.ndarray,
        row: dict,
        settings_fingerprint: str,
        nominal_sidecar: dict,
        source_states: Path,
    ) -> None:
        self.case = case
        self.case_config = case_config
        self.states = states
        self.cfg = cfg
        self.masks = np.asarray(masks, dtype=float)
        self.row = row
        self.settings_fingerprint = str(settings_fingerprint)
        self.nominal_sidecar = nominal_sidecar
        self.source_states = source_states
        self.controls_dir = results_dir / "controls"
        self.case_id = str(case["case_id"])
        self.safe_case_id = _safe_file_stem(self.case_id)
        self.manifest_path = self.controls_dir / f"{self.safe_case_id}_branch_control_manifest.json"
        self.branch_entries: list[dict[str, object]] = []

    def _common_metadata(self) -> dict[str, object]:
        thresholds = self.case_config["objective"]["thresholds"]
        return {
            "suite_name": SUITE_NAME,
            "artifact_stem": ARTIFACT_STEM,
            "case_id": self.case_id,
            "case_order": int(self.case["case_order"]),
            "case_group": str(self.case["case_group"]),
            "settings_fingerprint": self.settings_fingerprint,
            "config_hash": _config_hash(self.case_config),
            "runner_base_config_hash": str(self.row.get("config_hash", "")),
            "source_states_id": _file_identity(self.source_states),
            "target_mode": str(self.case.get("target_mode", "catalog_halo_phase_shift")),
            "target_generation": str(self.row.get("target_generation", TARGET_GENERATION)),
            "target_state": np.asarray(self.states.target, dtype=float).tolist(),
            "thresholds": {
                "nominal_success": float(thresholds["nominal_success"]),
                "robust_success": float(thresholds["robust_success"]),
            },
            "phase_time": float(self.case["phase_time"]),
            "transfer_time": float(self.cfg.tf),
            "segments": int(self.cfg.n_segments),
            "substeps_per_segment": int(self.cfg.substeps),
            "amax": float(self.cfg.amax),
            "outage_lengths": [int(value) for value in self.case["outage_lengths"]],
            "collocation_method": str(self.row.get("collocation_method", DEFAULT_COLLOCATION_METHOD)),
            "fuel_quadrature": str(self.row.get("fuel_quadrature", "")),
            "implementation_identities": {
                "runner": _file_identity(Path(__file__)),
                "direct_collocation_module": _dc_file_identity(ROOT / "src" / "qlt" / "direct_collocation.py"),
                "objective_module": _file_identity(ROOT / "src" / "qlt" / "objective.py"),
                "cr3bp_module": _file_identity(ROOT / "src" / "qlt" / "cr3bp.py"),
            },
            "replay_semantics": (
                "Normalized CR3BP replay of persisted nominal and selected branch controls only; "
                "no optimization rerun, high-fidelity validation, production solver parity, "
                "fuel-optimality claim, or quantum-advantage claim."
            ),
            "limitations": _branch_control_replay_limitations(),
        }

    def _nominal_sidecar_path(self) -> Path:
        path = self.nominal_sidecar["path"]
        return path if isinstance(path, Path) else Path(path)

    def _branch_payload(
        self,
        *,
        result: dict,
        branch_order: int,
        mask_index: int,
        endpoint_controls: np.ndarray,
        midpoint_controls: np.ndarray | None,
    ) -> tuple[dict[str, object], Path]:
        if mask_index < 0 or mask_index >= int(self.masks.shape[0]):
            raise ValueError(f"branch sidecar mask_index {mask_index} outside configured mask array")
        outage_mask = np.asarray(self.masks[mask_index], dtype=int)
        recovery_start = _outage_end(outage_mask)
        nominal_endpoint = np.asarray(self.nominal_sidecar["controls"], dtype=float).reshape(
            (int(self.cfg.n_segments), 3)
        )
        endpoint_raw = np.asarray(endpoint_controls, dtype=float)
        if endpoint_raw.shape == (int(self.cfg.n_segments), 3):
            endpoint_controls = endpoint_raw
        elif endpoint_raw.size == (int(self.cfg.n_segments) - int(recovery_start)) * 3:
            endpoint_controls = nominal_endpoint.copy() * outage_mask[:, None]
            if int(recovery_start) < int(self.cfg.n_segments):
                endpoint_controls[int(recovery_start) :] = endpoint_raw.reshape((-1, 3))
        else:
            raise ValueError(
                f"{self.case_id} branch {branch_order} endpoint controls shape {endpoint_raw.shape} "
                f"cannot form full {(int(self.cfg.n_segments), 3)} schedule"
            )
        midpoint_arr = None
        if midpoint_controls is not None:
            midpoint_raw = np.asarray(midpoint_controls, dtype=float)
            if midpoint_raw.shape == (int(self.cfg.n_segments), 3):
                midpoint_arr = midpoint_raw
            elif midpoint_raw.size == (int(self.cfg.n_segments) - int(recovery_start)) * 3:
                nominal_midpoint = self.nominal_sidecar.get("midpoint_controls")
                if nominal_midpoint is None:
                    nominal_midpoint = nominal_endpoint
                midpoint_arr = np.asarray(nominal_midpoint, dtype=float).reshape(
                    (int(self.cfg.n_segments), 3)
                )
                midpoint_arr = midpoint_arr.copy() * outage_mask[:, None]
                if int(recovery_start) < int(self.cfg.n_segments):
                    midpoint_arr[int(recovery_start) :] = midpoint_raw.reshape((-1, 3))
            else:
                raise ValueError(
                    f"{self.case_id} branch {branch_order} midpoint controls shape {midpoint_raw.shape} "
                    f"cannot form full {(int(self.cfg.n_segments), 3)} schedule"
                )
        all_errors = list(result.get("all_outage_errors", []))
        if mask_index >= len(all_errors):
            raise ValueError(f"all_outage_errors does not contain mask_index {mask_index}")
        endpoint_hash = _control_hash(endpoint_controls)
        midpoint_hash = "" if midpoint_arr is None else _control_hash(midpoint_arr)
        combined_hash = _combined_control_sidecar_hash(endpoint_controls, midpoint_arr)
        branch_path = (
            self.controls_dir
            / f"{self.safe_case_id}_branch_{int(branch_order):03d}_mask_{int(mask_index):03d}_controls.json"
        )
        common = self._common_metadata()
        payload = {
            "schema_version": 1,
            "sidecar_type": "hs_direct_collocation_branch_controls",
            **common,
            "branch_order": int(branch_order),
            "mask_index": int(mask_index),
            "outage_mask": outage_mask.astype(int).tolist(),
            "recovery_start": int(recovery_start),
            "recovery_segments": int(self.cfg.n_segments) - int(recovery_start),
            "recorded_branch_terminal_error": float(all_errors[mask_index]),
            "recorded_selected_worst_error": float(result.get("selected_worst_error", float("nan"))),
            "recorded_all_mask_worst_error": float(result.get("all_mask_worst_error", float("nan"))),
            "selected_outage_indices": [int(value) for value in result.get("selected_outage_indices", [])],
            "selected_outage_errors": [float(value) for value in result.get("selected_outage_errors", [])],
            "all_outage_errors": [float(value) for value in all_errors],
            "nominal_control_path": _artifact_path(self._nominal_sidecar_path()),
            "nominal_control_sha256": _sha256(self._nominal_sidecar_path()),
            "nominal_control_sidecar_hash": str(self.nominal_sidecar.get("sidecar_hash", "")),
            "nominal_endpoint_control_hash": str(self.nominal_sidecar.get("endpoint_control_hash", "")),
            "nominal_midpoint_control_hash": str(self.nominal_sidecar.get("midpoint_control_hash", "")),
            "branch_endpoint_control_hash": endpoint_hash,
            "branch_midpoint_control_present": midpoint_arr is not None,
            "branch_midpoint_control_hash": midpoint_hash,
            "branch_control_hash": combined_hash,
            "branch_endpoint_controls": endpoint_controls.tolist(),
            "branch_controls": endpoint_controls.tolist(),
            "branch_midpoint_controls": None if midpoint_arr is None else midpoint_arr.tolist(),
            "branch_endpoint_control_norm_diagnostics": _control_norm_diagnostics(
                endpoint_controls,
                float(self.cfg.amax),
            ),
            "branch_midpoint_control_norm_diagnostics": (
                None if midpoint_arr is None else _control_norm_diagnostics(midpoint_arr, float(self.cfg.amax))
            ),
        }
        return payload, branch_path

    def finalize(self, result: dict) -> dict[str, object]:
        if not bool(self.case.get("persist_branch_controls", False)):
            return _empty_branch_sidecar_columns()
        selected_indices = [int(value) for value in result.get("selected_outage_indices", [])]
        branch_controls = [np.asarray(value, dtype=float) for value in result.get("selected_branch_controls", [])]
        branch_midpoint_controls_raw = list(result.get("selected_branch_midpoint_controls", []))
        if not selected_indices or not branch_controls:
            return _empty_branch_sidecar_columns()
        if len(selected_indices) != len(branch_controls):
            raise ValueError(
                f"{self.case_id} selected_outage_indices count {len(selected_indices)} "
                f"does not match selected_branch_controls count {len(branch_controls)}"
            )
        self.branch_entries = []
        for branch_order, (mask_index, endpoint_controls) in enumerate(zip(selected_indices, branch_controls)):
            midpoint_controls = None
            if branch_order < len(branch_midpoint_controls_raw):
                raw_midpoint = branch_midpoint_controls_raw[branch_order]
                if raw_midpoint is not None:
                    midpoint_controls = np.asarray(raw_midpoint, dtype=float)
            payload, branch_path = self._branch_payload(
                result=result,
                branch_order=int(branch_order),
                mask_index=int(mask_index),
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
            )
            branch_sha = _write_deterministic_json(branch_path, payload)
            self.branch_entries.append(
                {
                    "branch_order": int(branch_order),
                    "mask_index": int(mask_index),
                    "outage_mask": payload["outage_mask"],
                    "recovery_start": int(payload["recovery_start"]),
                    "recovery_segments": int(payload["recovery_segments"]),
                    "recorded_branch_terminal_error": float(payload["recorded_branch_terminal_error"]),
                    "branch_control_hash": str(payload["branch_control_hash"]),
                    "branch_endpoint_control_hash": str(payload["branch_endpoint_control_hash"]),
                    "branch_midpoint_control_present": bool(payload["branch_midpoint_control_present"]),
                    "branch_midpoint_control_hash": str(payload["branch_midpoint_control_hash"]),
                    "path": _artifact_path(branch_path),
                    "sha256": branch_sha,
                }
            )

        nominal_path = self._nominal_sidecar_path()
        expected_branch_count = len(selected_indices)
        replay_ready = bool(len(self.branch_entries) == expected_branch_count)
        manifest = {
            "schema_version": 1,
            "sidecar_type": "hs_direct_collocation_branch_control_manifest",
            **self._common_metadata(),
            "nominal_control_path": _artifact_path(nominal_path),
            "nominal_control_sha256": _sha256(nominal_path),
            "nominal_control_sidecar_hash": str(self.nominal_sidecar.get("sidecar_hash", "")),
            "nominal_endpoint_control_hash": str(self.nominal_sidecar.get("endpoint_control_hash", "")),
            "nominal_midpoint_control_present": bool(self.nominal_sidecar.get("midpoint_control_present", False)),
            "nominal_midpoint_control_hash": str(self.nominal_sidecar.get("midpoint_control_hash", "")),
            "nominal_error": float(result.get("nominal_error", float("nan"))),
            "selected_worst_error": float(result.get("selected_worst_error", float("nan"))),
            "all_mask_worst_error": float(result.get("all_mask_worst_error", float("nan"))),
            "meets_thresholds": bool(self.row.get("meets_thresholds", False)),
            "optimizer_success": bool(result.get("optimizer_success", False)),
            "direct_collocation_success": bool(result.get("success", False)),
            "cost": float(result.get("cost", float("nan"))),
            "optimality": float(result.get("optimality", float("nan"))),
            "nfev": int(result.get("nfev", 0)),
            "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
            "expected_branch_count": expected_branch_count,
            "branch_count": int(len(self.branch_entries)),
            "branch_control_sidecar_count": int(len(self.branch_entries)),
            "branch_control_replay_ready": replay_ready,
            "selected_outage_count": expected_branch_count,
            "selected_outage_indices": selected_indices,
            "selected_outage_errors": [float(value) for value in result.get("selected_outage_errors", [])],
            "all_outage_errors": [float(value) for value in result.get("all_outage_errors", [])],
            "branch_control_sidecars": self.branch_entries,
            "branch_control_guess_source": self.case.get("branch_control_guess_from_case_id") or "",
            "resume_semantics": (
                "HS branch sidecars are not mandatory for historical rows. They are written only for "
                "cases with persist_branch_controls=true and can be replayed without optimizer reruns."
            ),
        }
        manifest_sha = _write_deterministic_json(self.manifest_path, manifest)
        return {
            "branch_control_manifest_path": _artifact_path(self.manifest_path),
            "branch_control_manifest_hash": manifest_sha,
            "branch_control_sidecar_count": int(len(self.branch_entries)),
            "branch_control_replay_ready": replay_ready,
        }


def _load_branch_manifest_info_from_row(
    *,
    row: dict,
    case: dict,
    expected_settings_fingerprint: str,
    require_ready: bool,
) -> tuple[dict[str, object] | None, str | None]:
    ready = _as_bool(row.get("branch_control_replay_ready"))
    path_value = row.get("branch_control_manifest_path")
    hash_value = row.get("branch_control_manifest_hash")
    if not ready or _is_missing(path_value) or not str(path_value).strip():
        if require_ready:
            return None, "branch-control manifest missing or not replay-ready"
        return None, None
    try:
        path = _resolve_artifact_path(path_value)
        if not path.is_file():
            return None, f"branch-control manifest not found: {path}"
        actual_hash = _sha256(path)
        if not _is_missing(hash_value) and str(hash_value).strip() and str(hash_value).strip() != actual_hash:
            return None, "branch-control manifest hash mismatched"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(manifest.get("case_id", "")) != str(case["case_id"]):
            return None, "branch-control manifest case_id mismatched"
        if str(manifest.get("settings_fingerprint", "")) != str(expected_settings_fingerprint):
            return None, "branch-control manifest settings_fingerprint mismatched"
        if not bool(manifest.get("branch_control_replay_ready", False)):
            return None, "branch-control manifest is not replay-ready"
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"branch-control manifest could not be loaded: {exc}"
    return {
        "branch_control_manifest_path": path,
        "branch_control_manifest_hash": actual_hash,
        "branch_control_sidecar_count": int(manifest.get("branch_control_sidecar_count", 0)),
        "branch_control_replay_ready": True,
        "branch_control_manifest": manifest,
    }, None


def load_branch_control_guesses(branch_info: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    manifest = branch_info.get("branch_control_manifest")
    if not isinstance(manifest, dict):
        manifest_path = branch_info.get("branch_control_manifest_path")
        manifest_hash = branch_info.get("branch_control_manifest_hash")
        path = _resolve_artifact_path(manifest_path)
        actual_hash = _sha256(path)
        if manifest_hash and str(manifest_hash) != actual_hash:
            raise RuntimeError(f"branch-control manifest hash mismatch for {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    if not bool(manifest.get("branch_control_replay_ready", False)):
        raise RuntimeError("branch-control manifest is not replay-ready")
    endpoint_guesses: list[np.ndarray] = []
    midpoint_guesses: list[np.ndarray | None] = []
    for entry in sorted(manifest.get("branch_control_sidecars", []), key=lambda item: int(item["branch_order"])):
        path = _resolve_artifact_path(entry["path"])
        actual_hash = _sha256(path)
        if str(entry.get("sha256", "")) != actual_hash:
            raise RuntimeError(f"branch-control sidecar hash mismatch for {path}")
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        endpoint = np.asarray(sidecar.get("branch_endpoint_controls", sidecar.get("branch_controls")), dtype=float)
        midpoint_raw = sidecar.get("branch_midpoint_controls")
        midpoint = None if midpoint_raw is None else np.asarray(midpoint_raw, dtype=float)
        endpoint_guesses.append(endpoint)
        midpoint_guesses.append(midpoint)
    return endpoint_guesses, midpoint_guesses


# ---------------------------------------------------------------------------
# Running a single case
# ---------------------------------------------------------------------------

def _run_case(
    *,
    base_config: dict,
    source_states: Path,
    case: dict,
    source_info: dict | None,
    branch_guess_info: dict[str, object] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray | None, dict[str, object]]:
    case_config = _case_config(base_config, case)
    states = load_configured_states(Path.cwd(), case_config, source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = case_config["objective"]["thresholds"]
    direct_cfg = copy.deepcopy(case_config.get("direct_collocation", {}) or {})
    direct_cfg["max_nfev"] = int(case["max_nfev"])
    direct_cfg["selected_outages"] = int(case["selected_outages"])
    direct_cfg["min_recovery_segments"] = int(case["min_recovery_segments"])
    collocation_method = str(direct_cfg.get("method", DEFAULT_COLLOCATION_METHOD))

    warm_start_info = {
        "enabled": source_info is not None,
        "kind": str(case["warm_start_kind"]),
        "source_case_id": case.get("warm_start_from_case_id"),
        "source_settings_fingerprint": None if source_info is None else str(source_info["settings_fingerprint"]),
        "source_control_hash": None if source_info is None else str(source_info["control_hash"]),
        "source_endpoint_control_hash": None if source_info is None else str(source_info.get("endpoint_control_hash", "")),
        "source_midpoint_control_hash": None if source_info is None else str(source_info.get("midpoint_control_hash", "")),
        "source_midpoint_control_present": bool(
            source_info is not None and source_info.get("midpoint_controls") is not None
        ),
        "branch_control_guess_enabled": branch_guess_info is not None,
        "branch_control_guess_source_case_id": case.get("branch_control_guess_from_case_id"),
        "branch_control_guess_manifest_hash": (
            None if branch_guess_info is None else str(branch_guess_info.get("branch_control_manifest_hash", ""))
        ),
        "continuous_backend_baseline": True,
        "collocation_method": collocation_method,
    }

    nominal_guess: np.ndarray | None = None
    nominal_midpoint_guess: np.ndarray | None = None
    branch_control_guesses: list[np.ndarray] | None = None
    branch_midpoint_control_guesses: list[np.ndarray | None] | None = None
    if source_info is not None:
        nominal_guess = np.asarray(source_info["controls"], dtype=float)
        if nominal_guess.shape != (cfg.n_segments, 3):
            raise ValueError(
                f"{case['case_id']} warm-start controls shape {nominal_guess.shape} "
                f"does not match {(cfg.n_segments, 3)}"
            )
        if source_info.get("midpoint_controls") is not None:
            nominal_midpoint_guess = np.asarray(source_info["midpoint_controls"], dtype=float)
            if nominal_midpoint_guess.shape != (cfg.n_segments, 3):
                raise ValueError(
                    f"{case['case_id']} warm-start midpoint controls shape {nominal_midpoint_guess.shape} "
                    f"does not match {(cfg.n_segments, 3)}"
                )

    if branch_guess_info is not None:
        branch_control_guesses, branch_midpoint_control_guesses = load_branch_control_guesses(branch_guess_info)
        if len(branch_control_guesses) != int(case["selected_outages"]):
            raise ValueError(
                f"{case['case_id']} loaded {len(branch_control_guesses)} branch-control guesses, "
                f"expected selected_outages={int(case['selected_outages'])}"
            )

    backend_kwargs = {
        "state0": states.initial,
        "target": states.target,
        "cfg": cfg,
        "masks": masks,
        "thresholds": thresholds,
        "selected_outages": int(case["selected_outages"]),
        "max_nfev": int(case["max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "collocation_config": direct_cfg,
        "nominal_control_guess": nominal_guess,
        "selected_branch_control_guesses": branch_control_guesses,
        "warm_start_info": warm_start_info,
    }
    if nominal_midpoint_guess is not None:
        backend_kwargs["nominal_midpoint_control_guess"] = nominal_midpoint_guess
    if branch_midpoint_control_guesses is not None and any(value is not None for value in branch_midpoint_control_guesses):
        backend_kwargs["selected_branch_midpoint_control_guesses"] = branch_midpoint_control_guesses
    result = run_direct_collocation_baseline(**backend_kwargs)
    result["target_generation"] = str(
        (getattr(states, "target_metadata", {}) or {}).get("target_state_generation", TARGET_GENERATION)
    )
    nominal_controls = np.asarray(result["nominal_controls"], dtype=float)
    raw_midpoint_controls = result.get("nominal_midpoint_controls")
    nominal_midpoint_controls = (
        None if raw_midpoint_controls is None else np.asarray(raw_midpoint_controls, dtype=float)
    )
    context = {
        "case_config": case_config,
        "states": states,
        "cfg": cfg,
        "masks": masks,
    }
    return result, nominal_controls, nominal_midpoint_controls, context


# ---------------------------------------------------------------------------
# Resume / compatibility
# ---------------------------------------------------------------------------

def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=HS_CONTINUATION_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in HS_CONTINUATION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[HS_CONTINUATION_COLUMNS]


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
    require_midpoint_controls = _requires_midpoint_controls(base_config)

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
        source_fingerprint: str | None = None
        source_control_hash: str | None = None
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

        branch_source = _branch_control_guess_source_case(cases_by_id, case)
        source_branch_manifest_hash: str | None = None
        if branch_source is not None:
            branch_source_info = controls_by_case_id.get(str(branch_source["case_id"]))
            if branch_source_info is None:
                rejected.append(
                    {
                        "case_id": case_id,
                        "reason": "branch-control guess source controls are unavailable or stale",
                        "source_case_id": str(branch_source["case_id"]),
                    }
                )
                continue
            if not _as_bool(branch_source_info.get("branch_control_replay_ready", False)):
                rejected.append(
                    {
                        "case_id": case_id,
                        "reason": "branch-control guess source manifest is unavailable or not replay-ready",
                        "source_case_id": str(branch_source["case_id"]),
                    }
                )
                continue
            source_branch_manifest_hash = str(branch_source_info.get("branch_control_manifest_hash", ""))

        expected_fp = compute_settings_fingerprint(
            base_config,
            source_states,
            case,
            source_settings_fingerprint=source_fingerprint,
            source_control_hash=source_control_hash,
            source_branch_control_manifest_hash=source_branch_manifest_hash,
        )
        found_fp = row.get("settings_fingerprint")
        if _is_missing(found_fp) or str(found_fp) != expected_fp:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_fp,
                    "found_settings_fingerprint": None if _is_missing(found_fp) else str(found_fp),
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
                }
            )
            continue

        loaded = load_control_sidecar(
            controls_dir,
            case_id,
            expected_fp,
            require_midpoint_controls=require_midpoint_controls,
        )
        if loaded is None:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "nominal-control sidecar missing, stale, or hash mismatched",
                    "expected_settings_fingerprint": expected_fp,
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

        branch_manifest_info, branch_manifest_reason = _load_branch_manifest_info_from_row(
            row=row,
            case=case,
            expected_settings_fingerprint=expected_fp,
            require_ready=bool(case.get("persist_branch_controls", False)),
        )
        if branch_manifest_reason is not None:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": branch_manifest_reason,
                    "expected_settings_fingerprint": expected_fp,
                }
            )
            continue

        normalized = {column: row.get(column) for column in HS_CONTINUATION_COLUMNS}
        normalized["case_order"] = int(case["case_order"])
        normalized["nominal_control_path"] = control_path.as_posix()
        normalized["nominal_control_hash"] = control_hash
        normalized["nominal_control_sidecar_hash"] = control_hash
        normalized["nominal_endpoint_control_hash"] = loaded.endpoint_control_hash
        normalized["nominal_midpoint_control_present"] = bool(loaded.midpoint_control_present)
        normalized["nominal_midpoint_control_hash"] = loaded.midpoint_control_hash
        if branch_manifest_info is not None:
            normalized["branch_control_manifest_path"] = _artifact_path(
                branch_manifest_info["branch_control_manifest_path"]  # type: ignore[arg-type]
            )
            normalized["branch_control_manifest_hash"] = str(branch_manifest_info["branch_control_manifest_hash"])
            normalized["branch_control_sidecar_count"] = int(branch_manifest_info["branch_control_sidecar_count"])
            normalized["branch_control_replay_ready"] = True
        else:
            normalized.update(_empty_branch_sidecar_columns())
        kept_rows.append(normalized)
        case_info = {
            "controls": controls,
            "midpoint_controls": loaded.midpoint_controls,
            "control_hash": control_hash,
            "endpoint_control_hash": loaded.endpoint_control_hash,
            "midpoint_control_hash": loaded.midpoint_control_hash,
            "midpoint_control_present": loaded.midpoint_control_present,
            "path": control_path,
            "settings_fingerprint": expected_fp,
            "row": normalized,
        }
        if branch_manifest_info is not None:
            case_info.update(branch_manifest_info)
        controls_by_case_id[case_id] = case_info
        seen_case_ids.add(case_id)

    existing_case_ids = set(str(r.get("case_id", "")) for r in df.to_dict(orient="records"))
    for case_id in sorted(existing_case_ids - set(cases_by_id)):
        rejected.append({"case_id": case_id, "reason": "case is not in the current requested suite"})

    return pd.DataFrame(kept_rows, columns=HS_CONTINUATION_COLUMNS), controls_by_case_id, rejected


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _table_group_label(value: str) -> str:
    labels = {
        "phase_continuation_single": "Phase continuation (single outage)",
        "hard_catalog_probe": "Hard catalog stress probe",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values("case_order")
    table = ordered[
        [
            "case_id",
            "case_group",
            "phase_time",
            "warm_start_kind",
            "warm_start_from_phase_time",
            "segments",
            "selected_outages",
            "outage_count",
            "max_nfev",
            "collocation_method",
            "nominal_error",
            "selected_worst_error",
            "all_mask_worst_error",
            "optimizer_success",
            "direct_collocation_success",
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
    table["selected_outages"] = (
        table["selected_outages"].astype(int).astype(str)
        + "/"
        + table["outage_count"].astype(int).astype(str)
    )
    table = table.drop(columns=["outage_count", "warm_start_from_phase_time"])
    table.columns = [
        "Case ID",
        "Group",
        "Phase time",
        "Warm start",
        "Segments",
        "Outage masks",
        "Max nfev",
        "Method",
        "Nominal error",
        "Selected worst error",
        "All-mask worst error",
        "Optimizer success",
        "DC success",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
    ]
    (tables_dir / f"{ARTIFACT_STEM}_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def _write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values("case_order")
    groups = ordered["case_group"].dropna().astype(str).drop_duplicates().tolist()
    if not groups:
        return
    ncols = min(2, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows), sharey=False)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax, group_name in zip(axes, groups):
        title = _table_group_label(group_name)
        group = df[df["case_group"] == group_name].sort_values("case_order")
        x = np.arange(len(group))
        colors = ["tab:blue" if str(kind) == "cold" else "tab:orange" for kind in group["warm_start_kind"]]
        selected_worst = group["selected_worst_error"].astype(float)
        nominal = group["nominal_error"].astype(float)
        nominal_threshold = group["nominal_threshold"].astype(float)
        selected_threshold = group["selected_worst_threshold"].astype(float)
        ax.bar(x, selected_worst, color=colors, alpha=0.72, label="selected worst error")
        ax.plot(x, nominal, color="black", marker="o", linewidth=1.4, label="nominal error")
        y_top = float(
            max(
                selected_worst.max(),
                nominal.max(),
                nominal_threshold.max(),
                selected_threshold.max(),
            )
        )
        y_pad = max(0.012, 0.06 * y_top)
        for index, (_, row) in enumerate(group.iterrows()):
            marker = "pass" if _as_bool(row["meets_thresholds"]) else "fail"
            marker_base = max(float(row["selected_worst_error"]), float(row["nominal_error"]))
            ax.text(index, marker_base + y_pad, marker, ha="center", fontsize=8)
        if nominal_threshold.nunique(dropna=True) == 1:
            ax.axhline(float(nominal_threshold.iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        else:
            ax.plot(x, nominal_threshold, color="0.45", linestyle="--", linewidth=1.0)
        if selected_threshold.nunique(dropna=True) == 1:
            ax.axhline(float(selected_threshold.iloc[0]), color="0.25", linestyle=":", linewidth=1.0)
        else:
            ax.plot(x, selected_threshold, color="0.25", linestyle=":", linewidth=1.0)
        labels = []
        for _, row in group.iterrows():
            if str(row["warm_start_kind"]) == "cold":
                labels.append(f"p={float(row['phase_time']):.1f}\ncold")
            else:
                labels.append(f"p={float(row['phase_time']):.1f}\nfrom {float(row['warm_start_from_phase_time']):.1f}")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel("Normalized terminal error")
        ax.set_ylim(top=y_top + 3 * y_pad)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(groups):]:
        fig.delaxes(ax)
    fig.legend(
        handles=[
            Line2D([0], [0], color="black", marker="o", linewidth=1.4, label="nominal error"),
            Patch(facecolor="tab:blue", alpha=0.72, label="cold start selected worst"),
            Patch(facecolor="tab:orange", alpha=0.72, label="warm-start selected worst"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in (".png", ".pdf"):
        fig.savefig(
            figures_dir / f"{ARTIFACT_STEM}{suffix}",
            dpi=220 if suffix == ".png" else None,
        )
    plt.close(fig)


def _metadata(
    df: pd.DataFrame,
    config: dict,
    command: str,
    cases: list[dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> dict:
    feasible_count = int(df["meets_thresholds"].map(_as_bool).sum()) if not df.empty else 0
    return {
        "command": command,
        "row_count": int(len(df)),
        "feasible_row_count": feasible_count,
        "config": config,
        "expected_case_count": int(len(cases)),
        "completed_case_count": int(len(df)),
        "skipped_cases": skipped_cases,
        "resume_rejected_rows": resume_rejected_rows,
        "threshold_rule": (
            "meets_thresholds requires nominal_error <= nominal_success "
            "and selected_worst_error <= robust_success"
        ),
        "semantics": {
            "backend": METADATA_BACKEND_SEMANTICS,
            "continuation": (
                "Continuation rows pass the previous row's persisted nominal endpoint controls to "
                "qlt.direct_collocation.run_direct_collocation_baseline as nominal_control_guess. "
                "For independent-midpoint HS rows, persisted nominal midpoint controls are also passed "
                "as nominal_midpoint_control_guess when present. Controls are Euclidean projected to "
                "the amax ball before use."
            ),
            "sidecar_schema": (
                "Nominal-control sidecars are backward compatible with legacy endpoint-only files. "
                "Rows use nominal_control_hash/nominal_control_sidecar_hash for the full sidecar hash; "
                "nominal_endpoint_control_hash records endpoint controls, and "
                "nominal_midpoint_control_present/nominal_midpoint_control_hash record optional "
                "independent midpoint controls."
            ),
            METADATA_METHOD_KEY: METADATA_METHOD_SEMANTICS,
            "all_mask": (
                "all_mask_worst_error and all_outage_errors evaluate all configured outage masks. "
                "Selected masks use optimized branch recovery controls; "
                "unselected masks use masked nominal controls only."
            ),
            "target": (
                "Each row records target_mode and target_generation; phase-shift rows use zero-thrust "
                "CR3BP phase propagation of the JPL halo source state, while catalog-DRO rows use the "
                "configured catalog-DRO target loader."
            ),
        },
        "limitations": list(METADATA_LIMITATIONS),
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
    df = pd.DataFrame(df, columns=HS_CONTINUATION_COLUMNS)
    df.to_csv(results_dir / f"{ARTIFACT_STEM}.csv", index=False)
    _write_table(df, tables_dir)
    _write_plot(df, figures_dir)
    write_json(
        results_dir / f"{ARTIFACT_STEM}_metadata.json",
        _metadata(df, config, command, cases, resume_rejected_rows, skipped_cases),
    )


def _runtime_budget(args, config: dict) -> float | None:
    if getattr(args, "runtime_budget_seconds", None) is not None:
        return float(args.runtime_budget_seconds)
    suite_value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    if suite_value is None:
        return None
    return float(suite_value)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if FORCE_COLLOCATION_METHOD is not None:
        config.setdefault("direct_collocation", {})["method"] = str(FORCE_COLLOCATION_METHOD)
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    controls_dir = results_dir / "controls"
    for directory in (results_dir, figures_dir, tables_dir, controls_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{ARTIFACT_STEM}.csv"

    cases = _suite_cases(config)
    if getattr(args, "max_cases", None) is not None:
        cases = cases[: int(args.max_cases)]
    cases_by_id = _case_by_id(cases)

    if getattr(args, "resume", False):
        loaded = _load_existing(csv_path)
        df, controls_by_case_id, resume_rejected_rows = _compatible_existing_rows(
            loaded,
            base_config=config,
            source_states=args.source_states,
            cases=cases,
            controls_dir=controls_dir,
        )
    else:
        df = pd.DataFrame(columns=HS_CONTINUATION_COLUMNS)
        controls_by_case_id = {}
        resume_rejected_rows = []

    completed_fingerprints: set[str] = set(
        str(v) for v in df["settings_fingerprint"].dropna().tolist()
    )
    command = " ".join(sys.argv)
    skipped_cases: list[dict] = []
    budget = _runtime_budget(args, config)
    start = time.perf_counter()
    active_stack: set[str] = set()

    def run_or_skip(case: dict) -> None:
        nonlocal df
        case_id = str(case["case_id"])
        if case_id in active_stack:
            raise RuntimeError(f"cycle in HS continuation dependencies at {case_id}")
        active_stack.add(case_id)
        source_info: dict | None = None
        source_settings_fingerprint: str | None = None
        source_control_hash: str | None = None
        branch_guess_info: dict[str, object] | None = None
        source_branch_manifest_hash: str | None = None
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

        branch_guess_source_case = _branch_control_guess_source_case(cases_by_id, case)
        if branch_guess_source_case is not None:
            run_or_skip(branch_guess_source_case)
            branch_guess_info = controls_by_case_id.get(str(branch_guess_source_case["case_id"]))
            if branch_guess_info is None or not _as_bool(branch_guess_info.get("branch_control_replay_ready", False)):
                skipped_cases.append(
                    {
                        "case_id": case_id,
                        "case_group": str(case["case_group"]),
                        "phase_time": float(case["phase_time"]),
                        "reason": "branch-control guess source manifest is unavailable",
                        "source_case_id": str(branch_guess_source_case["case_id"]),
                    }
                )
                active_stack.remove(case_id)
                return
            source_branch_manifest_hash = str(branch_guess_info.get("branch_control_manifest_hash", ""))

        settings_fp = compute_settings_fingerprint(
            config,
            args.source_states,
            case,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
            source_branch_control_manifest_hash=source_branch_manifest_hash,
        )
        if settings_fp in completed_fingerprints and case_id in controls_by_case_id:
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

        result, nominal_controls, nominal_midpoint_controls, case_context = _run_case(
            base_config=config,
            source_states=args.source_states,
            case=case,
            source_info=source_info,
            branch_guess_info=branch_guess_info,
        )
        row = _row_from_result(
            base_config=config,
            source_states=args.source_states,
            case=case,
            result=result,
            settings_fp=settings_fp,
            source_settings_fingerprint=source_settings_fingerprint,
            source_control_hash=source_control_hash,
        )
        sidecar = write_control_sidecar(
            controls_dir,
            case_id=case_id,
            settings_fingerprint=settings_fp,
            controls=nominal_controls,
            midpoint_controls=nominal_midpoint_controls,
            row=row,
        )
        row["nominal_control_path"] = sidecar["path"].as_posix()
        row["nominal_control_hash"] = sidecar["control_hash"]
        row["nominal_control_sidecar_hash"] = sidecar["sidecar_hash"]
        row["nominal_endpoint_control_hash"] = sidecar["endpoint_control_hash"]
        row["nominal_midpoint_control_present"] = bool(sidecar["midpoint_control_present"])
        row["nominal_midpoint_control_hash"] = sidecar["midpoint_control_hash"]
        row["nominal_midpoint_warm_start_loaded"] = bool(
            (result.get("warm_start_info") or {}).get("source_midpoint_control_present", False)
        )
        branch_sidecars = BranchControlSidecarStore(
            results_dir=results_dir,
            case=case,
            case_config=case_context["case_config"],  # type: ignore[arg-type]
            states=case_context["states"],
            cfg=case_context["cfg"],
            masks=case_context["masks"],  # type: ignore[arg-type]
            row=row,
            settings_fingerprint=settings_fp,
            nominal_sidecar=sidecar,
            source_states=args.source_states,
        ).finalize(result)
        row.update(branch_sidecars)
        controls_by_case_id[case_id] = {
            "controls": sidecar["controls"],
            "midpoint_controls": sidecar["midpoint_controls"],
            "control_hash": sidecar["control_hash"],
            "endpoint_control_hash": sidecar["endpoint_control_hash"],
            "midpoint_control_hash": sidecar["midpoint_control_hash"],
            "midpoint_control_present": sidecar["midpoint_control_present"],
            "path": sidecar["path"],
            "settings_fingerprint": settings_fp,
            "row": row,
            **branch_sidecars,
        }
        if case_id in set(str(v) for v in df["case_id"].dropna().tolist()):
            df = df[df["case_id"].astype(str) != case_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed_fingerprints.add(settings_fp)
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

    df = pd.DataFrame(df, columns=HS_CONTINUATION_COLUMNS)
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
    parser = argparse.ArgumentParser(
        description=SCRIPT_DESCRIPTION
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--source-states",
        type=Path,
        default=Path("data/source_states.json"),
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
