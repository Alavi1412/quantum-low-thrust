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
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .cr3bp import rk4_step
from .experiment import load_configured_states, make_objective_config, output_directories
from .objective import Evaluator, ObjectiveConfig, outage_masks, schedule_to_string, state_error
from .refinement import (
    control_fuel,
    feedback_controls_for_schedule,
    project_controls_to_ball,
    refine_schedule,
    selected_outage_indices_for_schedule,
)
from .reporting import write_json, write_metadata


MULTIPLE_SHOOTING_COLUMNS = [
    "transfer_time",
    "amax",
    "segments",
    "max_nfev",
    "substeps_per_segment",
    "selected_outages",
    "node_initialization",
    "node_initialization_blend",
    "nominal_first_homotopy",
    "nominal_first_max_nfev",
    "min_recovery_segments",
    "initial_weight",
    "defect_weight",
    "terminal_weight",
    "branch_terminal_weight",
    "branch_start_weight",
    "control_weight",
    "smooth_weight",
    "solver_mode",
    "bounded_controls_in_residual",
    "residual_control_max_norm",
    "evaluation_control_max_norm",
    "control_bound_violation",
    "warm_start_refinement",
    "warm_start_max_nfev",
    "config_hash",
    "source_states_id",
    "settings_fingerprint",
    "schedule",
    "base_nominal_error",
    "base_worst_error",
    "nominal_error",
    "selected_recovery_worst_error",
    "selected_worst_error",
    "all_outage_worst_error",
    "all_mask_worst_error",
    "nominal_threshold",
    "selected_recovery_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_recovery_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "optimizer_success",
    "multiple_shooting_success",
    "cost",
    "optimality",
    "nfev",
    "runtime_seconds",
    "control_max_norm",
    "accepted_candidate",
    "warm_start_refinement_nfev",
    "warm_start_nominal_error",
    "warm_start_selected_worst_error",
    "nominal_first_nfev",
    "nominal_first_nominal_error",
    "nominal_first_selected_worst_error",
    "nominal_first_all_mask_worst_error",
    "nominal_fuel",
    "recovery_fuel_mean",
    "recovery_fuel_max",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "message",
]


def _has_value(row: dict, key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _coalesce(row: dict, *keys: str, default=np.nan):
    for key in keys:
        if _has_value(row, key):
            return row[key]
    return default


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


RESIDUAL_WEIGHT_DEFAULTS = {
    "initial_weight": 10.0,
    "defect_weight": 1.0,
    "terminal_weight": 4.0,
    "branch_terminal_weight": 4.0,
    "branch_start_weight": 5.0,
    "control_weight": 0.01,
    "smooth_weight": 0.01,
}


def _solver_mode_setting(config: dict | None = None, args=None) -> str:
    refinement = (config or {}).get("refinement", {}) or {}
    requested = getattr(args, "solver_mode", None) if args is not None else None
    if requested is None:
        requested = refinement.get("solver_mode", refinement.get("multiple_shooting_solver_mode"))
    normalized = str(requested or "bounded_projected_multiple_shooting").strip().lower().replace("-", "_")
    aliases = {
        "bounded_multiple_shooting": "bounded_projected_multiple_shooting",
        "bounded_projected_direct_shooting": "bounded_projected_multiple_shooting",
        "bounded_direct_collocation": "bounded_projected_multiple_shooting",
        "bounded_projected_direct_collocation": "bounded_projected_multiple_shooting",
        "multiple_shooting_branch_recovery": "bounded_projected_multiple_shooting",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized != "bounded_projected_multiple_shooting":
        raise ValueError("solver_mode must be bounded_projected_multiple_shooting")
    return normalized


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _residual_weights_from_args(args) -> dict[str, float]:
    return {
        key: float(getattr(args, key, default))
        for key, default in RESIDUAL_WEIGHT_DEFAULTS.items()
    }


def _min_recovery_segments_setting(config: dict, args) -> int:
    refinement = config.get("refinement", {}) or {}
    value = getattr(args, "min_recovery_segments", None)
    if value is None:
        value = refinement.get("outage_selection_min_recovery_segments", 4)
    return int(value)


def _warm_start_settings(args) -> tuple[bool, int]:
    return bool(getattr(args, "warm_start_refinement", False)), int(getattr(args, "warm_start_nfev", 90))


def _settings_payload(values: dict) -> dict:
    nominal_first_homotopy = _as_bool(values["nominal_first_homotopy"])
    return {
        "transfer_time": float(values["transfer_time"]),
        "amax": float(values["amax"]),
        "segments": int(values["segments"]),
        "max_nfev": int(values["max_nfev"]),
        "selected_outages": int(values["selected_outages"]),
        "min_recovery_segments": int(values["min_recovery_segments"]),
        "node_initialization": _normalize_node_initialization(values["node_initialization"]),
        "node_initialization_blend": _node_initialization_blend(values["node_initialization_blend"]),
        "nominal_first_homotopy": nominal_first_homotopy,
        "nominal_first_max_nfev": int(values["nominal_first_max_nfev"]) if nominal_first_homotopy else 0,
        "residual_weights": {
            key: float(values[key])
            for key in RESIDUAL_WEIGHT_DEFAULTS
        },
        "solver_mode": _solver_mode_setting({"refinement": {"solver_mode": values.get("solver_mode")}}),
        "warm_start_refinement": _as_bool(values["warm_start_refinement"]),
        "warm_start_max_nfev": int(values["warm_start_max_nfev"]),
        "config_hash": str(values["config_hash"]),
        "source_states_id": str(values["source_states_id"]),
    }


def _settings_fingerprint(values: dict) -> str:
    encoded = json.dumps(_settings_payload(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_multiple_shooting_row(row: dict) -> dict:
    normalized = dict(row)
    selected_worst = _coalesce(
        normalized,
        "selected_recovery_worst_error",
        "selected_worst_error",
        "worst_error",
    )
    normalized["selected_recovery_worst_error"] = selected_worst
    normalized["selected_worst_error"] = _coalesce(
        normalized,
        "selected_worst_error",
        "selected_recovery_worst_error",
        "worst_error",
        default=selected_worst,
    )
    all_outage_worst = _coalesce(
        normalized,
        "all_outage_worst_error",
        "all_mask_worst_error",
        default=normalized["selected_worst_error"],
    )
    normalized["all_outage_worst_error"] = all_outage_worst
    normalized["all_mask_worst_error"] = _coalesce(
        normalized,
        "all_mask_worst_error",
        "all_outage_worst_error",
        default=all_outage_worst,
    )

    robust_threshold = _coalesce(
        normalized,
        "selected_recovery_threshold",
        "selected_worst_threshold",
    )
    normalized["selected_recovery_threshold"] = robust_threshold
    normalized["selected_worst_threshold"] = _coalesce(
        normalized,
        "selected_worst_threshold",
        "selected_recovery_threshold",
        default=robust_threshold,
    )

    if not _has_value(normalized, "meets_nominal_threshold"):
        normalized["meets_nominal_threshold"] = bool(
            float(normalized["nominal_error"]) <= float(normalized["nominal_threshold"])
        )
    if not _has_value(normalized, "meets_selected_recovery_threshold"):
        normalized["meets_selected_recovery_threshold"] = bool(
            float(normalized["selected_recovery_worst_error"]) <= float(normalized["selected_recovery_threshold"])
        )
    normalized["meets_selected_worst_threshold"] = _coalesce(
        normalized,
        "meets_selected_worst_threshold",
        "meets_selected_recovery_threshold",
        default=normalized["meets_selected_recovery_threshold"],
    )
    if not _has_value(normalized, "meets_thresholds"):
        normalized["meets_thresholds"] = bool(
            _as_bool(normalized["meets_nominal_threshold"])
            and _as_bool(normalized["meets_selected_recovery_threshold"])
        )

    defaults = {
        "accepted_candidate": "unknown",
        "node_initialization": "rollout",
        "node_initialization_blend": 0.5,
        "nominal_first_homotopy": False,
        "nominal_first_max_nfev": _coalesce(normalized, "nominal_first_nfev", default=0),
        "min_recovery_segments": 4,
        **RESIDUAL_WEIGHT_DEFAULTS,
        "solver_mode": "bounded_projected_multiple_shooting",
        "bounded_controls_in_residual": True,
        "residual_control_max_norm": _coalesce(normalized, "control_max_norm", default=np.nan),
        "evaluation_control_max_norm": _coalesce(normalized, "control_max_norm", default=np.nan),
        "control_bound_violation": 0.0,
        "warm_start_refinement": False,
        "warm_start_max_nfev": 90,
        "config_hash": "legacy_unknown",
        "source_states_id": "legacy_unknown",
        "warm_start_refinement_nfev": 0,
        "warm_start_nominal_error": np.nan,
        "warm_start_selected_worst_error": np.nan,
        "nominal_first_nfev": 0,
        "nominal_first_nominal_error": np.nan,
        "nominal_first_selected_worst_error": np.nan,
        "nominal_first_all_mask_worst_error": np.nan,
        "selected_outage_indices": "[]",
        "selected_outage_errors": "[]",
        "all_outage_errors": "[]",
        "message": "",
    }
    for key, value in defaults.items():
        if not _has_value(normalized, key):
            normalized[key] = value
    if not _has_value(normalized, "settings_fingerprint"):
        normalized["settings_fingerprint"] = _settings_fingerprint(normalized)
    return {column: normalized.get(column, np.nan) for column in MULTIPLE_SHOOTING_COLUMNS}


def normalize_multiple_shooting_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MULTIPLE_SHOOTING_COLUMNS)
    rows = [normalize_multiple_shooting_row(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=MULTIPLE_SHOOTING_COLUMNS)


def _scale_vector(cfg: ObjectiveConfig) -> np.ndarray:
    return np.array(
        [
            cfg.position_scale,
            cfg.position_scale,
            cfg.position_scale,
            cfg.velocity_scale,
            cfg.velocity_scale,
            cfg.velocity_scale,
        ],
        dtype=float,
    )


def _scaled_state_residual(state: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    return (np.asarray(state, dtype=float) - np.asarray(target, dtype=float)) / np.maximum(_scale_vector(cfg), 1e-12)


def _propagate_segment(state: np.ndarray, control: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    out = np.asarray(state, dtype=float).copy()
    dt = cfg.tf / float(cfg.n_segments * cfg.substeps)
    for _ in range(cfg.substeps):
        out = rk4_step(out, cfg.mu, dt, control)
    return out


def _rollout_nodes(state0: np.ndarray, controls: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    nodes = np.zeros((cfg.n_segments + 1, 6), dtype=float)
    nodes[0] = np.asarray(state0, dtype=float)
    for i in range(cfg.n_segments):
        nodes[i + 1] = _propagate_segment(nodes[i], controls[i], cfg)
    return nodes


def _linear_nodes(start: np.ndarray, target: np.ndarray, node_count: int) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.linspace(0.0, 1.0, int(node_count), dtype=float)[:, None]
    return (1.0 - weights) * start[None, :] + weights * target[None, :]


def _normalize_node_initialization(mode: str | None) -> str:
    normalized = str(mode or "rollout").strip().lower().replace("_", "-")
    aliases = {
        "feedback": "rollout",
        "feedback-rollout": "rollout",
        "rollout-feedback": "rollout",
        "linear-interpolation": "linear",
        "linearly-interpolated": "linear",
        "blended": "blend",
        "rollout-linear-blend": "blend",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"rollout", "linear", "blend"}:
        raise ValueError("node_initialization must be one of: rollout, linear, blend")
    return normalized


def _node_initialization_blend(value: float | None) -> float:
    if value is None:
        return 0.5
    blend = float(value)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("node_initialization_blend must be between 0 and 1")
    return blend


def _initial_nodes_for_mode(
    *,
    start: np.ndarray,
    target: np.ndarray,
    rollout_nodes: np.ndarray,
    mode: str | None,
    blend: float | None,
) -> np.ndarray:
    mode = _normalize_node_initialization(mode)
    rollout_nodes = np.asarray(rollout_nodes, dtype=float)
    if mode == "rollout":
        return rollout_nodes.copy()

    linear_nodes = _linear_nodes(start, target, rollout_nodes.shape[0])
    if mode == "linear":
        return linear_nodes

    alpha = _node_initialization_blend(blend)
    return (1.0 - alpha) * rollout_nodes + alpha * linear_nodes


def _outage_end(mask: np.ndarray) -> int:
    missed = np.flatnonzero(np.asarray(mask, dtype=float) < 0.5)
    if missed.size == 0:
        return 0
    return int(missed[-1] + 1)


def _branch_controls_from_recovery(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    recovery_start: int,
    recovery_controls: np.ndarray,
    amax: float,
) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    if recovery_start < controls.shape[0]:
        controls[recovery_start:] = recovery_controls
    return project_controls_to_ball(controls, amax)


class _DecisionLayout:
    def __init__(self, cfg: ObjectiveConfig, selected_masks: np.ndarray):
        self.cfg = cfg
        self.selected_masks = np.asarray(selected_masks, dtype=float)
        self.starts = [_outage_end(mask) for mask in self.selected_masks]
        cursor = 0
        self.nominal_nodes = slice(cursor, cursor + (cfg.n_segments + 1) * 6)
        cursor = self.nominal_nodes.stop
        self.nominal_controls = slice(cursor, cursor + cfg.n_segments * 3)
        cursor = self.nominal_controls.stop
        self.branch_nodes: list[slice] = []
        self.branch_controls: list[slice] = []
        for start in self.starts:
            node_count = cfg.n_segments - start + 1
            control_count = cfg.n_segments - start
            self.branch_nodes.append(slice(cursor, cursor + node_count * 6))
            cursor = self.branch_nodes[-1].stop
            self.branch_controls.append(slice(cursor, cursor + control_count * 3))
            cursor = self.branch_controls[-1].stop
        self.size = cursor

    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        cfg = self.cfg
        nominal_nodes = vec[self.nominal_nodes].reshape((cfg.n_segments + 1, 6))
        nominal_controls = project_controls_to_ball(vec[self.nominal_controls].reshape((cfg.n_segments, 3)), cfg.amax)
        branch_nodes = []
        branch_controls = []
        for start, node_slice, control_slice in zip(self.starts, self.branch_nodes, self.branch_controls):
            branch_nodes.append(vec[node_slice].reshape((cfg.n_segments - start + 1, 6)))
            control_count = cfg.n_segments - start
            if control_count:
                controls = vec[control_slice].reshape((control_count, 3))
            else:
                controls = np.zeros((0, 3), dtype=float)
            branch_controls.append(project_controls_to_ball(controls, cfg.amax))
        return nominal_nodes, nominal_controls, branch_nodes, branch_controls


def _mark_block(matrix: lil_matrix, rows: slice, cols: slice) -> None:
    matrix[rows, cols] = True


def _jacobian_sparsity(layout: _DecisionLayout, include_smooth: bool = True) -> lil_matrix:
    cfg = layout.cfg
    row = 0
    # Keep the row accounting in lockstep with residual().
    residual_len = 6
    residual_len += cfg.n_segments * 6
    residual_len += 6
    residual_len += cfg.n_segments * 3
    if include_smooth and cfg.n_segments > 1:
        residual_len += (cfg.n_segments - 1) * 3
    for start in layout.starts:
        recovery_segments = cfg.n_segments - start
        residual_len += 6
        residual_len += recovery_segments * 6
        residual_len += 6
        residual_len += recovery_segments * 3
        if include_smooth and recovery_segments > 1:
            residual_len += (recovery_segments - 1) * 3

    sparsity = lil_matrix((residual_len, layout.size), dtype=bool)

    def nominal_node_cols(index: int) -> slice:
        begin = layout.nominal_nodes.start + index * 6
        return slice(begin, begin + 6)

    def nominal_control_cols(index: int) -> slice:
        begin = layout.nominal_controls.start + index * 3
        return slice(begin, begin + 3)

    # Nominal initial condition.
    _mark_block(sparsity, slice(row, row + 6), nominal_node_cols(0))
    row += 6

    # Nominal defects.
    for i in range(cfg.n_segments):
        rows = slice(row, row + 6)
        _mark_block(sparsity, rows, nominal_node_cols(i))
        _mark_block(sparsity, rows, nominal_node_cols(i + 1))
        _mark_block(sparsity, rows, nominal_control_cols(i))
        row += 6

    # Nominal terminal condition.
    _mark_block(sparsity, slice(row, row + 6), nominal_node_cols(cfg.n_segments))
    row += 6

    # Nominal control regularization and smoothness.
    for i in range(cfg.n_segments):
        _mark_block(sparsity, slice(row, row + 3), nominal_control_cols(i))
        row += 3
    if include_smooth and cfg.n_segments > 1:
        for i in range(cfg.n_segments - 1):
            rows = slice(row, row + 3)
            _mark_block(sparsity, rows, nominal_control_cols(i))
            _mark_block(sparsity, rows, nominal_control_cols(i + 1))
            row += 3

    for branch_slice, control_slice, start in zip(layout.branch_nodes, layout.branch_controls, layout.starts):
        def branch_node_cols(local_index: int) -> slice:
            begin = branch_slice.start + local_index * 6
            return slice(begin, begin + 6)

        def branch_control_cols(local_index: int) -> slice:
            begin = control_slice.start + local_index * 3
            return slice(begin, begin + 3)

        recovery_segments = cfg.n_segments - start
        rows = slice(row, row + 6)
        _mark_block(sparsity, rows, branch_node_cols(0))
        for i in range(start):
            _mark_block(sparsity, rows, nominal_control_cols(i))
        row += 6

        for local_i in range(recovery_segments):
            rows = slice(row, row + 6)
            _mark_block(sparsity, rows, branch_node_cols(local_i))
            _mark_block(sparsity, rows, branch_node_cols(local_i + 1))
            _mark_block(sparsity, rows, branch_control_cols(local_i))
            row += 6

        _mark_block(sparsity, slice(row, row + 6), branch_node_cols(recovery_segments))
        row += 6

        for local_i in range(recovery_segments):
            _mark_block(sparsity, slice(row, row + 3), branch_control_cols(local_i))
            row += 3

        if include_smooth and recovery_segments > 1:
            for local_i in range(recovery_segments - 1):
                rows = slice(row, row + 3)
                _mark_block(sparsity, rows, branch_control_cols(local_i))
                _mark_block(sparsity, rows, branch_control_cols(local_i + 1))
                row += 3

    if row != residual_len:
        raise RuntimeError(f"internal residual sparsity size mismatch: {row} != {residual_len}")
    return sparsity


def _control_norm_diagnostics(control_sets: list[np.ndarray], amax: float) -> dict[str, float]:
    max_norm = 0.0
    for controls in control_sets:
        arr = np.asarray(controls, dtype=float)
        if arr.size == 0:
            continue
        max_norm = max(max_norm, float(np.max(np.linalg.norm(arr, axis=1))))
    return {
        "control_max_norm": max_norm,
        "control_bound_violation": max(0.0, max_norm - float(amax)),
    }


def _initial_guess(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    selected_masks: np.ndarray,
    nominal_control_guess: np.ndarray | None = None,
    selected_branch_control_guesses: list[np.ndarray] | None = None,
    node_initialization: str | None = "rollout",
    node_initialization_blend: float | None = 0.5,
) -> tuple[_DecisionLayout, np.ndarray]:
    node_initialization = _normalize_node_initialization(node_initialization)
    node_initialization_blend = _node_initialization_blend(node_initialization_blend)
    schedule = np.ones(cfg.n_segments, dtype=int)
    if nominal_control_guess is None:
        nominal_controls = feedback_controls_for_schedule(schedule, state0, target, cfg)
    else:
        nominal_controls = np.asarray(nominal_control_guess, dtype=float)
    nominal_controls = project_controls_to_ball(nominal_controls, cfg.amax)
    nominal_rollout_nodes = _rollout_nodes(state0, nominal_controls, cfg)
    nominal_nodes = _initial_nodes_for_mode(
        start=state0,
        target=target,
        rollout_nodes=nominal_rollout_nodes,
        mode=node_initialization,
        blend=node_initialization_blend,
    )
    layout = _DecisionLayout(cfg, selected_masks)
    parts: list[np.ndarray] = [nominal_nodes.reshape(-1), nominal_controls.reshape(-1)]
    selected_branch_control_guesses = selected_branch_control_guesses or []
    for branch_index, (mask, start) in enumerate(zip(selected_masks, layout.starts)):
        if branch_index < len(selected_branch_control_guesses):
            branch_controls = project_controls_to_ball(np.asarray(selected_branch_control_guesses[branch_index], dtype=float), cfg.amax)
        else:
            branch_controls = _branch_controls_from_recovery(
                nominal_controls,
                mask,
                start,
                nominal_controls[start:].copy(),
                cfg.amax,
            )
        branch_rollout_nodes = _rollout_nodes(state0, branch_controls, cfg)[start:]
        branch_nodes = _initial_nodes_for_mode(
            start=branch_rollout_nodes[0],
            target=target,
            rollout_nodes=branch_rollout_nodes,
            mode=node_initialization,
            blend=node_initialization_blend,
        )
        parts.append(branch_nodes.reshape(-1))
        parts.append(branch_controls[start:].reshape(-1))
    return layout, np.concatenate(parts)


class _BoundedMultipleShootingProblem:
    """Direct multiple-shooting residual with projected acceleration controls."""

    mode = "bounded_projected_multiple_shooting"
    bounded_controls_in_residual = True

    def __init__(
        self,
        *,
        state0: np.ndarray,
        target: np.ndarray,
        cfg: ObjectiveConfig,
        masks: np.ndarray,
        selected: np.ndarray,
        selected_masks: np.ndarray,
        layout: _DecisionLayout,
        weights: dict[str, float],
    ):
        self.state0 = np.asarray(state0, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.cfg = cfg
        self.masks = np.asarray(masks, dtype=float)
        self.selected = np.asarray(selected, dtype=int)
        self.selected_masks = np.asarray(selected_masks, dtype=float)
        self.layout = layout
        self.weights = weights
        self.scale = np.maximum(_scale_vector(cfg), 1e-12)

    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        return self.layout.unpack(vec)

    def residual_control_diagnostics(self, vec: np.ndarray) -> dict[str, float]:
        _, nominal_controls, _, branch_recovery_controls = self.unpack(vec)
        diagnostics = _control_norm_diagnostics([nominal_controls, *branch_recovery_controls], self.cfg.amax)
        return {
            "residual_control_max_norm": diagnostics["control_max_norm"],
            "residual_control_bound_violation": diagnostics["control_bound_violation"],
        }

    def residual(self, vec: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        weights = self.weights
        nominal_nodes, nominal_controls, branch_nodes, branch_recovery_controls = self.unpack(vec)
        res: list[np.ndarray] = [
            weights["initial"] * (nominal_nodes[0] - self.state0) / self.scale,
        ]
        for i in range(cfg.n_segments):
            predicted = _propagate_segment(nominal_nodes[i], nominal_controls[i], cfg)
            res.append(weights["defect"] * (nominal_nodes[i + 1] - predicted) / self.scale)
        res.append(weights["terminal"] * _scaled_state_residual(nominal_nodes[-1], self.target, cfg))
        res.append(weights["control"] * nominal_controls.reshape(-1) / max(cfg.amax, 1e-12))
        if cfg.n_segments > 1:
            res.append(weights["smooth"] * np.diff(nominal_controls, axis=0).reshape(-1) / max(cfg.amax, 1e-12))

        for mask, recovery_start, nodes, recovery_controls in zip(
            self.selected_masks,
            self.layout.starts,
            branch_nodes,
            branch_recovery_controls,
        ):
            pre_controls = nominal_controls * np.asarray(mask, dtype=float)[:, None]
            start_state = self.state0.copy()
            for i in range(recovery_start):
                start_state = _propagate_segment(start_state, pre_controls[i], cfg)
            res.append(weights["branch_start"] * (nodes[0] - start_state) / self.scale)
            for local_i in range(cfg.n_segments - recovery_start):
                predicted = _propagate_segment(nodes[local_i], recovery_controls[local_i], cfg)
                res.append(weights["defect"] * (nodes[local_i + 1] - predicted) / self.scale)
            res.append(weights["branch_terminal"] * _scaled_state_residual(nodes[-1], self.target, cfg))
            if recovery_controls.size:
                res.append(weights["control"] * recovery_controls.reshape(-1) / max(cfg.amax, 1e-12))
            if recovery_controls.shape[0] > 1:
                res.append(weights["smooth"] * np.diff(recovery_controls, axis=0).reshape(-1) / max(cfg.amax, 1e-12))
        return np.concatenate(res)

    def evaluate_vector(self, vec: np.ndarray) -> dict:
        cfg = self.cfg
        nominal_nodes, nominal_controls, _, branch_recovery_controls = self.unpack(vec)
        del nominal_nodes
        nominal_final = _rollout_nodes(self.state0, nominal_controls, cfg)[-1]
        nominal_error = float(state_error(nominal_final, self.target, cfg.position_scale, cfg.velocity_scale))
        selected_lookup = {
            int(mask_index): _branch_controls_from_recovery(
                nominal_controls,
                mask,
                recovery_start,
                recovery_controls,
                cfg.amax,
            )
            for mask_index, mask, recovery_start, recovery_controls in zip(
                self.selected,
                self.selected_masks,
                self.layout.starts,
                branch_recovery_controls,
            )
        }
        selected_set = set(int(i) for i in self.selected)
        all_errors: list[float] = []
        selected_errors: list[float] = []
        branch_fuels: list[float] = []
        for mask_index, mask in enumerate(self.masks):
            if mask_index in selected_lookup:
                controls = selected_lookup[mask_index]
            else:
                controls = nominal_controls * np.asarray(mask, dtype=float)[:, None]
            final = _rollout_nodes(self.state0, controls, cfg)[-1]
            error = float(state_error(final, self.target, cfg.position_scale, cfg.velocity_scale))
            all_errors.append(error)
            if mask_index in selected_set:
                selected_errors.append(error)
                branch_fuels.append(control_fuel(controls, cfg.tf))
        selected_worst = float(np.max(selected_errors)) if selected_errors else nominal_error
        all_worst = float(np.max(all_errors)) if all_errors else selected_worst
        diagnostics = _control_norm_diagnostics([nominal_controls, *selected_lookup.values()], cfg.amax)
        return {
            "nominal_controls": nominal_controls,
            "nominal_error": nominal_error,
            "selected_worst_error": selected_worst,
            "all_mask_worst_error": all_worst,
            "selected_outage_errors": selected_errors,
            "all_outage_errors": all_errors,
            "nominal_fuel": control_fuel(nominal_controls, cfg.tf),
            "recovery_fuel_mean": float(np.mean(branch_fuels)) if branch_fuels else control_fuel(nominal_controls, cfg.tf),
            "recovery_fuel_max": float(np.max(branch_fuels)) if branch_fuels else control_fuel(nominal_controls, cfg.tf),
            "control_max_norm": diagnostics["control_max_norm"],
            "control_bound_violation": diagnostics["control_bound_violation"],
        }


def run_multiple_shooting_baseline(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray,
    thresholds: dict,
    selected_outages: int = 3,
    max_nfev: int = 120,
    min_recovery_segments: int = 4,
    residual_weights: dict | None = None,
    nominal_control_guess: np.ndarray | None = None,
    selected_branch_control_guesses: list[np.ndarray] | None = None,
    node_initialization: str | None = "rollout",
    node_initialization_blend: float | None = 0.5,
    warm_start_info: dict | None = None,
) -> dict:
    """Solve the all-windows branch-recovery case with direct multiple shooting."""
    start_time = time.perf_counter()
    schedule = np.ones(cfg.n_segments, dtype=int)
    node_initialization = _normalize_node_initialization(node_initialization)
    node_initialization_blend = _node_initialization_blend(node_initialization_blend)
    if int(selected_outages) <= 0:
        selected = np.zeros(0, dtype=int)
    else:
        selected = selected_outage_indices_for_schedule(
            schedule,
            state0,
            target,
            cfg,
            masks,
            selected_outages,
            min_recovery_segments,
        )
    selected_masks = masks[selected]
    layout, x0 = _initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        nominal_control_guess=nominal_control_guess,
        selected_branch_control_guesses=selected_branch_control_guesses,
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
    )
    weights = {
        "initial": 10.0,
        "defect": 1.0,
        "terminal": 4.0,
        "branch_terminal": 4.0,
        "branch_start": 5.0,
        "control": 0.01,
        "smooth": 0.01,
    }
    if residual_weights:
        weights.update({key: float(value) for key, value in residual_weights.items() if value is not None})

    problem = _BoundedMultipleShootingProblem(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        selected=selected,
        selected_masks=selected_masks,
        layout=layout,
        weights=weights,
    )

    result = least_squares(
        problem.residual,
        x0,
        jac_sparsity=_jacobian_sparsity(layout, include_smooth=cfg.n_segments > 1),
        max_nfev=int(max_nfev),
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        verbose=0,
    )
    nominal_threshold = float(thresholds["nominal_success"])
    robust_threshold = float(thresholds["robust_success"])
    candidates = {
        "initial": problem.evaluate_vector(x0),
        "optimizer": problem.evaluate_vector(result.x),
    }
    accepted_label, accepted = min(
        candidates.items(),
        key=lambda item: (
            max(0.0, item[1]["nominal_error"] - nominal_threshold)
            + max(0.0, item[1]["selected_worst_error"] - robust_threshold),
            item[1]["selected_worst_error"],
            item[1]["nominal_error"],
        ),
    )
    nominal_error = float(accepted["nominal_error"])
    selected_worst = float(accepted["selected_worst_error"])
    all_worst = float(accepted["all_mask_worst_error"])
    converged = bool(nominal_error <= nominal_threshold and selected_worst <= robust_threshold)
    residual_diagnostics = problem.residual_control_diagnostics(x0 if accepted_label == "initial" else result.x)
    evaluation_control_max_norm = float(accepted["control_max_norm"])
    control_bound_violation = max(
        float(accepted["control_bound_violation"]),
        float(residual_diagnostics["residual_control_bound_violation"]),
    )

    return {
        "success": converged,
        "mode": "multiple_shooting_branch_recovery",
        "solver_mode": problem.mode,
        "bounded_controls_in_residual": problem.bounded_controls_in_residual,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "nominal_error": nominal_error,
        "worst_error_semantics": "selected_recovery_worst_error",
        "worst_error": selected_worst,
        "selected_recovery_worst_error": selected_worst,
        "selected_worst_error": selected_worst,
        "all_outage_worst_error": all_worst,
        "all_mask_worst_error": all_worst,
        "nominal_fuel": float(accepted["nominal_fuel"]),
        "recovery_fuel_mean": float(accepted["recovery_fuel_mean"]),
        "recovery_fuel_max": float(accepted["recovery_fuel_max"]),
        "control_max_norm": evaluation_control_max_norm,
        "residual_control_max_norm": float(residual_diagnostics["residual_control_max_norm"]),
        "evaluation_control_max_norm": evaluation_control_max_norm,
        "control_bound_violation": control_bound_violation,
        "controls": accepted["nominal_controls"],
        "accepted_candidate": accepted_label,
        "node_initialization": node_initialization,
        "node_initialization_blend": node_initialization_blend,
        "warm_start_info": warm_start_info or {},
        "selected_outage_indices": selected.astype(int).tolist(),
        "selected_outage_errors": accepted["selected_outage_errors"],
        "all_outage_errors": accepted["all_outage_errors"],
        "recovery_starts": layout.starts,
        "residual_weights": weights,
    }


def _apply_effective_case_config(
    config: dict,
    *,
    transfer_time: float,
    amax: float,
    segments: int,
    max_nfev: int,
    selected_outages: int,
    node_initialization: str | None = None,
    node_initialization_blend: float | None = None,
    nominal_first_homotopy: bool | None = None,
    nominal_first_nfev: int | None = None,
    min_recovery_segments: int | None = None,
    residual_weights: dict[str, float] | None = None,
    solver_mode: str | None = None,
    warm_start_refinement: bool | None = None,
    warm_start_nfev: int | None = None,
) -> None:
    config.setdefault("run", {}).setdefault("label", "q1_multiple_shooting_feasibility")
    benchmark = config.setdefault("benchmark", {})
    benchmark["transfer_time"] = float(transfer_time)
    benchmark["amax"] = float(amax)
    benchmark["segments"] = int(segments)
    refinement = config.setdefault("refinement", {})
    refinement["mode"] = "multiple_shooting_branch_recovery"
    refinement["max_nfev"] = int(max_nfev)
    refinement["selected_outages"] = int(selected_outages)
    if node_initialization is not None:
        refinement["node_initialization"] = _normalize_node_initialization(node_initialization)
    if node_initialization_blend is not None:
        refinement["node_initialization_blend"] = _node_initialization_blend(node_initialization_blend)
    if nominal_first_homotopy is not None:
        refinement["nominal_first_homotopy"] = bool(nominal_first_homotopy)
    if nominal_first_nfev is not None:
        refinement["nominal_first_nfev"] = int(nominal_first_nfev)
    if min_recovery_segments is not None:
        refinement["outage_selection_min_recovery_segments"] = int(min_recovery_segments)
    if residual_weights is not None:
        refinement["multiple_shooting_residual_weights"] = {
            key: float(value)
            for key, value in residual_weights.items()
        }
    if solver_mode is not None:
        refinement["solver_mode"] = _solver_mode_setting({"refinement": {"solver_mode": solver_mode}})
    if warm_start_refinement is not None:
        refinement["warm_start_refinement"] = bool(warm_start_refinement)
    if warm_start_nfev is not None:
        refinement["warm_start_nfev"] = int(warm_start_nfev)


def _node_initialization_settings(config: dict, args) -> tuple[str, float]:
    refinement = config.get("refinement", {}) or {}
    requested_mode = getattr(args, "node_initialization", None)
    if requested_mode is None:
        requested_mode = refinement.get("node_initialization", "rollout")
    requested_blend = getattr(args, "node_initialization_blend", None)
    if requested_blend is None:
        requested_blend = getattr(args, "node_blend", None)
    if requested_blend is None:
        requested_blend = refinement.get("node_initialization_blend", 0.5)
    return _normalize_node_initialization(requested_mode), _node_initialization_blend(requested_blend)


def _nominal_first_settings(config: dict, args) -> tuple[bool, int]:
    refinement = config.get("refinement", {}) or {}
    requested = getattr(args, "nominal_first_homotopy", None)
    if requested is None:
        requested = getattr(args, "nominal_first", None)
    enabled = bool(refinement.get("nominal_first_homotopy", False) if requested is None else requested)
    nfev = getattr(args, "nominal_first_nfev", None)
    if nfev is None:
        nfev = refinement.get("nominal_first_nfev", refinement.get("max_nfev", 120))
    return enabled, int(nfev)


def config_for_case(base_config: dict, transfer_time: float, amax: float, segments: int, max_nfev: int, args) -> dict:
    config = copy.deepcopy(base_config)
    node_initialization, node_initialization_blend = _node_initialization_settings(config, args)
    nominal_first_homotopy, nominal_first_nfev = _nominal_first_settings(config, args)
    warm_start_refinement, warm_start_nfev = _warm_start_settings(args)
    _apply_effective_case_config(
        config,
        transfer_time=transfer_time,
        amax=amax,
        segments=segments,
        max_nfev=max_nfev,
        selected_outages=int(args.selected_outages),
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
        nominal_first_homotopy=nominal_first_homotopy,
        nominal_first_nfev=nominal_first_nfev,
        min_recovery_segments=_min_recovery_segments_setting(config, args),
        residual_weights=_residual_weights_from_args(args),
        solver_mode=_solver_mode_setting(config, args),
        warm_start_refinement=warm_start_refinement,
        warm_start_nfev=warm_start_nfev,
    )
    return config


def effective_config_for_row(base_config: dict, row: dict, args) -> dict:
    selected_outages = row["selected_outages"] if _has_value(row, "selected_outages") else getattr(args, "selected_outages", 3)
    config = copy.deepcopy(base_config)
    fallback_node_initialization, fallback_node_initialization_blend = _node_initialization_settings(config, args)
    fallback_nominal_first_homotopy, fallback_nominal_first_nfev = _nominal_first_settings(config, args)
    fallback_warm_start_refinement, fallback_warm_start_nfev = _warm_start_settings(args)
    node_initialization = (
        row["node_initialization"] if _has_value(row, "node_initialization") else fallback_node_initialization
    )
    node_initialization_blend = (
        row["node_initialization_blend"]
        if _has_value(row, "node_initialization_blend")
        else fallback_node_initialization_blend
    )
    nominal_first_homotopy = (
        _as_bool(row["nominal_first_homotopy"])
        if _has_value(row, "nominal_first_homotopy")
        else fallback_nominal_first_homotopy
    )
    nominal_first_nfev = (
        row["nominal_first_max_nfev"]
        if _has_value(row, "nominal_first_max_nfev")
        else (
            row["nominal_first_nfev"]
            if _has_value(row, "nominal_first_nfev")
            else fallback_nominal_first_nfev
        )
    )
    min_recovery_segments = (
        row["min_recovery_segments"]
        if _has_value(row, "min_recovery_segments")
        else _min_recovery_segments_setting(config, args)
    )
    residual_weights = {
        key: float(row[key]) if _has_value(row, key) else _residual_weights_from_args(args)[key]
        for key in RESIDUAL_WEIGHT_DEFAULTS
    }
    solver_mode = row["solver_mode"] if _has_value(row, "solver_mode") else _solver_mode_setting(config, args)
    warm_start_refinement = (
        _as_bool(row["warm_start_refinement"])
        if _has_value(row, "warm_start_refinement")
        else fallback_warm_start_refinement
    )
    warm_start_nfev = (
        int(row["warm_start_max_nfev"])
        if _has_value(row, "warm_start_max_nfev")
        else fallback_warm_start_nfev
    )
    _apply_effective_case_config(
        config,
        transfer_time=float(row["transfer_time"]),
        amax=float(row["amax"]),
        segments=int(row["segments"]),
        max_nfev=int(row["max_nfev"]),
        selected_outages=int(selected_outages),
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
        nominal_first_homotopy=nominal_first_homotopy,
        nominal_first_nfev=nominal_first_nfev,
        min_recovery_segments=min_recovery_segments,
        residual_weights=residual_weights,
        solver_mode=solver_mode,
        warm_start_refinement=warm_start_refinement,
        warm_start_nfev=warm_start_nfev,
    )
    return config


def effective_metadata_config(base_config: dict, df: pd.DataFrame, args) -> tuple[dict, list[dict]]:
    effective_cases = []
    for index, row in enumerate(df.to_dict(orient="records")):
        effective_cases.append(
            {
                "case_index": int(index),
                "case": {
                    "transfer_time": float(row["transfer_time"]),
                    "amax": float(row["amax"]),
                    "segments": int(row["segments"]),
                    "max_nfev": int(row["max_nfev"]),
                },
                "config": effective_config_for_row(base_config, row, args),
            }
        )
    if len(effective_cases) == 1:
        return effective_cases[0]["config"], effective_cases
    return {"base": base_config, "effective_cases": effective_cases}, effective_cases


def run_case(base_config: dict, transfer_time: float, amax: float, segments: int, max_nfev: int, args) -> dict:
    config = config_for_case(base_config, transfer_time, amax, segments, max_nfev, args)
    states = load_configured_states(Path.cwd(), config, args.source_states)
    cfg = make_objective_config(config, states.mu)
    schedule = np.ones(cfg.n_segments, dtype=int)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = config["objective"]["thresholds"]
    evaluator = Evaluator(states.initial, states.target, cfg)
    base = evaluator.evaluate(schedule)
    nominal_control_guess = None
    selected_branch_control_guesses = None
    warm_start_info: dict = {}
    node_initialization, node_initialization_blend = _node_initialization_settings(config, args)
    nominal_first_homotopy, nominal_first_nfev = _nominal_first_settings(config, args)
    min_recovery_segments = _min_recovery_segments_setting(config, args)
    warm_start_refinement, warm_start_nfev = _warm_start_settings(args)
    row_weights = _residual_weights_from_args(args)
    solver_mode = _solver_mode_setting(config, args)
    weights = {
        "initial": row_weights["initial_weight"],
        "defect": row_weights["defect_weight"],
        "terminal": row_weights["terminal_weight"],
        "branch_terminal": row_weights["branch_terminal_weight"],
        "branch_start": row_weights["branch_start_weight"],
        "control": row_weights["control_weight"],
        "smooth": row_weights["smooth_weight"],
    }
    if bool(getattr(args, "warm_start_refinement", False)):
        refine_cfg = copy.deepcopy(config["refinement"])
        refine_cfg["mode"] = "branch_recovery"
        refine_cfg["max_nfev"] = warm_start_nfev
        refine_cfg["selected_outages"] = int(args.selected_outages)
        refine_cfg["outage_selection_min_recovery_segments"] = min_recovery_segments
        refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, refine_cfg, thresholds)
        nominal_control_guess = np.asarray(refined.get("nominal_controls", refined.get("controls")), dtype=float)
        branch_by_index = {
            int(index): np.asarray(branch, dtype=float)
            for index, branch in zip(refined.get("selected_outage_indices", []), refined.get("recovery_controls", []))
        }
        selected_indices = selected_outage_indices_for_schedule(
            schedule,
            states.initial,
            states.target,
            cfg,
            masks,
            int(args.selected_outages),
            min_recovery_segments,
        )
        selected_branch_control_guesses = [branch_by_index[index] for index in selected_indices if int(index) in branch_by_index]
        if len(selected_branch_control_guesses) != len(selected_indices):
            selected_branch_control_guesses = None
        warm_start_info = {
            "enabled": True,
            "nfev": int(refined.get("nfev", 0) or 0),
            "nominal_error": float(refined.get("nominal_error", np.nan)),
            "selected_worst_error": float(refined.get("selected_worst_error", refined.get("worst_error", np.nan))),
            "all_mask_worst_error": float(refined.get("all_mask_worst_error", np.nan)),
            "success": bool(refined.get("success", False)),
        }
    nominal_first_info: dict = {}
    if nominal_first_homotopy:
        nominal_first = run_multiple_shooting_baseline(
            state0=states.initial,
            target=states.target,
            cfg=cfg,
            masks=masks,
            thresholds=thresholds,
            selected_outages=0,
            max_nfev=int(nominal_first_nfev),
            min_recovery_segments=min_recovery_segments,
            residual_weights=weights,
            nominal_control_guess=nominal_control_guess,
            selected_branch_control_guesses=None,
            node_initialization=node_initialization,
            node_initialization_blend=node_initialization_blend,
            warm_start_info={"stage": "nominal_first_homotopy"},
        )
        nominal_control_guess = np.asarray(nominal_first["controls"], dtype=float)
        selected_branch_control_guesses = None
        nominal_first_info = {
            "enabled": True,
            "nfev": int(nominal_first.get("nfev", 0) or 0),
            "nominal_error": float(nominal_first.get("nominal_error", np.nan)),
            "selected_worst_error": float(nominal_first.get("selected_worst_error", np.nan)),
            "all_mask_worst_error": float(nominal_first.get("all_mask_worst_error", np.nan)),
            "success": bool(nominal_first.get("success", False)),
            "node_initialization": str(nominal_first.get("node_initialization", node_initialization)),
        }
        warm_start_info["nominal_first_homotopy"] = nominal_first_info
    result = run_multiple_shooting_baseline(
        state0=states.initial,
        target=states.target,
        cfg=cfg,
        masks=masks,
        thresholds=thresholds,
        selected_outages=int(args.selected_outages),
        max_nfev=int(max_nfev),
        min_recovery_segments=min_recovery_segments,
        residual_weights=weights,
        nominal_control_guess=nominal_control_guess,
        selected_branch_control_guesses=selected_branch_control_guesses,
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
        warm_start_info=warm_start_info,
    )
    nominal_threshold = float(thresholds["nominal_success"])
    robust_threshold = float(thresholds["robust_success"])
    nominal_error = float(result["nominal_error"])
    selected_worst = float(result["selected_worst_error"])
    row = {
        "transfer_time": float(transfer_time),
        "amax": float(amax),
        "segments": int(segments),
        "max_nfev": int(max_nfev),
        "substeps_per_segment": int(cfg.substeps),
        "selected_outages": int(args.selected_outages),
        "node_initialization": str(result.get("node_initialization", node_initialization)),
        "node_initialization_blend": float(result.get("node_initialization_blend", node_initialization_blend)),
        "nominal_first_homotopy": bool(nominal_first_homotopy),
        "nominal_first_max_nfev": int(nominal_first_nfev) if nominal_first_homotopy else 0,
        "min_recovery_segments": int(min_recovery_segments),
        **row_weights,
        "solver_mode": str(result.get("solver_mode", solver_mode)),
        "bounded_controls_in_residual": bool(result.get("bounded_controls_in_residual", True)),
        "residual_control_max_norm": float(result.get("residual_control_max_norm", result["control_max_norm"])),
        "evaluation_control_max_norm": float(result.get("evaluation_control_max_norm", result["control_max_norm"])),
        "control_bound_violation": float(result.get("control_bound_violation", max(0.0, result["control_max_norm"] - amax))),
        "warm_start_refinement": bool(warm_start_refinement),
        "warm_start_max_nfev": int(warm_start_nfev),
        "config_hash": _config_hash(base_config),
        "source_states_id": _file_identity(args.source_states),
        "schedule": schedule_to_string(schedule),
        "base_nominal_error": float(base["nominal_error"]),
        "base_worst_error": float(base["worst_error"]),
        "nominal_error": nominal_error,
        "selected_recovery_worst_error": selected_worst,
        "selected_worst_error": selected_worst,
        "all_outage_worst_error": float(result["all_mask_worst_error"]),
        "all_mask_worst_error": float(result["all_mask_worst_error"]),
        "nominal_threshold": nominal_threshold,
        "selected_recovery_threshold": robust_threshold,
        "selected_worst_threshold": robust_threshold,
        "meets_nominal_threshold": bool(nominal_error <= nominal_threshold),
        "meets_selected_recovery_threshold": bool(selected_worst <= robust_threshold),
        "meets_selected_worst_threshold": bool(selected_worst <= robust_threshold),
        "meets_thresholds": bool(nominal_error <= nominal_threshold and selected_worst <= robust_threshold),
        "optimizer_success": bool(result["optimizer_success"]),
        "multiple_shooting_success": bool(result["success"]),
        "cost": float(result["cost"]),
        "optimality": float(result["optimality"]),
        "nfev": int(result["nfev"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "control_max_norm": float(result["control_max_norm"]),
        "accepted_candidate": str(result.get("accepted_candidate", "optimizer")),
        "warm_start_refinement_nfev": int(warm_start_info.get("nfev", 0) or 0),
        "warm_start_nominal_error": float(warm_start_info.get("nominal_error", np.nan)),
        "warm_start_selected_worst_error": float(warm_start_info.get("selected_worst_error", np.nan)),
        "nominal_first_nfev": int(nominal_first_info.get("nfev", 0) or 0),
        "nominal_first_nominal_error": float(nominal_first_info.get("nominal_error", np.nan)),
        "nominal_first_selected_worst_error": float(nominal_first_info.get("selected_worst_error", np.nan)),
        "nominal_first_all_mask_worst_error": float(nominal_first_info.get("all_mask_worst_error", np.nan)),
        "nominal_fuel": float(result["nominal_fuel"]),
        "recovery_fuel_mean": float(result["recovery_fuel_mean"]),
        "recovery_fuel_max": float(result["recovery_fuel_max"]),
        "selected_outage_indices": json.dumps(result["selected_outage_indices"]),
        "selected_outage_errors": json.dumps(result["selected_outage_errors"]),
        "all_outage_errors": json.dumps(result["all_outage_errors"]),
        "message": str(result["message"]),
    }
    row["settings_fingerprint"] = _settings_fingerprint(row)
    return row


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def settings_values_for_case(
    base_config: dict,
    transfer_time: float,
    amax: float,
    segments: int,
    max_nfev: int,
    args,
) -> dict:
    config = copy.deepcopy(base_config)
    node_initialization, node_initialization_blend = _node_initialization_settings(config, args)
    nominal_first_homotopy, nominal_first_nfev = _nominal_first_settings(config, args)
    warm_start_refinement, warm_start_nfev = _warm_start_settings(args)
    return {
        "transfer_time": float(transfer_time),
        "amax": float(amax),
        "segments": int(segments),
        "max_nfev": int(max_nfev),
        "selected_outages": int(args.selected_outages),
        "min_recovery_segments": _min_recovery_segments_setting(config, args),
        "node_initialization": node_initialization,
        "node_initialization_blend": node_initialization_blend,
        "nominal_first_homotopy": bool(nominal_first_homotopy),
        "nominal_first_max_nfev": int(nominal_first_nfev) if nominal_first_homotopy else 0,
        **_residual_weights_from_args(args),
        "solver_mode": _solver_mode_setting(config, args),
        "warm_start_refinement": bool(warm_start_refinement),
        "warm_start_max_nfev": int(warm_start_nfev),
        "config_hash": _config_hash(base_config),
        "source_states_id": _file_identity(args.source_states),
    }


def write_multiple_shooting_table(df: pd.DataFrame, tables_dir: Path) -> None:
    ordered = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True])
    table = ordered[
        [
            "transfer_time",
            "amax",
            "segments",
            "selected_outages",
            "node_initialization",
            "nominal_error",
            "selected_recovery_worst_error",
            "all_outage_worst_error",
            "nominal_fuel",
            "control_max_norm",
            "control_bound_violation",
            "nfev",
            "runtime_seconds",
            "meets_thresholds",
        ]
    ]
    table.columns = [
        "Transfer time",
        "amax",
        "Segments",
        "Selected outages",
        "Initialization",
        "Nominal error",
        "Selected recovery worst error",
        "All-outage diagnostic worst error",
        "Fuel",
        "Max ||u||",
        "Bound violation",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
    ]
    tables_dir.joinpath("multiple_shooting_feasibility_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def plot_multiple_shooting(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = np.where(df["meets_thresholds"].to_numpy(dtype=bool), "tab:green", "tab:red")
    ax.scatter(df["nominal_error"], df["selected_worst_error"], c=colors, s=90, edgecolors="black", linewidths=0.6)
    for _, row in df.iterrows():
        ax.annotate(
            f"T={row['transfer_time']:.1f}, a={row['amax']:.2f}",
            (row["nominal_error"], row["selected_worst_error"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    nominal_threshold = float(df["nominal_threshold"].iloc[0])
    robust_threshold = float(df["selected_worst_threshold"].iloc[0])
    ax.axvline(nominal_threshold, color="0.45", linestyle="--", linewidth=1.0)
    ax.axhline(robust_threshold, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Nominal error")
    ax.set_ylabel("Selected-worst recovery error")
    ax.set_title("Multiple-shooting all-windows feasibility")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"multiple_shooting_feasibility{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def _observed_values(df: pd.DataFrame, column: str, cast):
    if column not in df.columns or df.empty:
        return []
    values = []
    for value in df[column].dropna().tolist():
        if isinstance(value, str) and not value.strip():
            continue
        values.append(cast(value))
    return sorted(set(values))


def observed_case_grid(df: pd.DataFrame) -> dict:
    return {
        "transfer_time": _observed_values(df, "transfer_time", float),
        "amax": _observed_values(df, "amax", float),
        "segments": _observed_values(df, "segments", int),
        "max_nfev": _observed_values(df, "max_nfev", int),
        "selected_outages": _observed_values(df, "selected_outages", int),
        "min_recovery_segments": _observed_values(df, "min_recovery_segments", int),
        "node_initialization": _observed_values(df, "node_initialization", str),
        "node_initialization_blend": _observed_values(df, "node_initialization_blend", float),
        "nominal_first_homotopy": _observed_values(df, "nominal_first_homotopy", _as_bool),
        "nominal_first_max_nfev": _observed_values(df, "nominal_first_max_nfev", int),
        "solver_mode": _observed_values(df, "solver_mode", str),
    }


def write_summary(df: pd.DataFrame, path: Path, args) -> None:
    feasible = df[df["meets_thresholds"]].copy()
    best = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True]).head(5)
    summary = {
        "command": " ".join(sys.argv),
        "case_count": int(len(df)),
        "feasible_case_count": int(len(feasible)),
        "thresholds": {
            "nominal_success": float(df["nominal_threshold"].iloc[0]) if not df.empty else None,
            "selected_recovery_success": float(df["selected_worst_threshold"].iloc[0]) if not df.empty else None,
        },
        "targeted_cases": observed_case_grid(df),
        "command_case_grid": {
            "transfer_time": parse_float_grid(args.transfer_times),
            "amax": parse_float_grid(args.amax),
            "segments": parse_int_grid(args.segments),
            "max_nfev": parse_int_grid(args.max_nfev),
            "selected_outages": [int(getattr(args, "selected_outages", 3))],
            "node_initialization": getattr(args, "node_initialization", None),
            "node_initialization_blend": getattr(args, "node_initialization_blend", getattr(args, "node_blend", None)),
            "nominal_first_homotopy": bool(
                getattr(args, "nominal_first_homotopy", False) or getattr(args, "nominal_first", False)
            ),
            "solver_mode": _solver_mode_setting(args=args),
        },
        "best_cases": best.to_dict(orient="records"),
    }
    write_json(path, summary)


def regenerate_artifacts(df: pd.DataFrame, results_dir: Path, figures_dir: Path, tables_dir: Path, args, config: dict) -> None:
    if df.empty:
        return
    write_summary(df, results_dir / "multiple_shooting_feasibility_metadata.json", args)
    write_multiple_shooting_table(df, tables_dir)
    plot_multiple_shooting(df, figures_dir)
    metadata_config, effective_cases = effective_metadata_config(config, df, args)
    extra = {
        "multiple_shooting_rows": int(len(df)),
        "multiple_shooting_output": str(results_dir / "multiple_shooting_feasibility.csv"),
        "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_recovery_worst_error <= robust_success",
        "mask_semantics": "selected_recovery_worst_error/selected_worst_error evaluate only the selected outage masks optimized with branch recovery; all_outage_worst_error/all_mask_worst_error evaluate every outage mask as a diagnostic. worst_error and selected_worst_error are retained as backwards-compatible aliases for selected_recovery_worst_error in the multiple-shooting baseline result.",
        "control_bound": "bounded_projected_multiple_shooting projects every nominal and branch-recovery acceleration vector to ||u_i|| <= amax inside the residual before RK4 propagation, fuel/smoothness residuals, evaluation propagation, and reporting",
        "bounded_controls_in_residual": bool(df["bounded_controls_in_residual"].astype(bool).all()),
        "max_control_bound_violation": float(df["control_bound_violation"].max()),
        "node_initialization": "rollout preserves the historical feedback-rollout node seed; linear seeds state nodes by interpolation from the branch start to the hard target; blend linearly combines rollout and linear nodes before optimization.",
    }
    if len(effective_cases) > 1:
        extra["effective_cases"] = effective_cases
    write_metadata(
        results_dir / "multiple_shooting_metadata.json",
        " ".join(sys.argv),
        metadata_config,
        extra,
    )


def run(args) -> pd.DataFrame:
    root = Path.cwd()
    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir, figures_dir, tables_dir = output_directories(root, base_config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "multiple_shooting_feasibility.csv"

    rows: list[dict] = []
    completed: set[str] = set()
    if args.resume and csv_path.exists():
        existing = normalize_multiple_shooting_dataframe(pd.read_csv(csv_path))
        rows.extend(existing.to_dict(orient="records"))
        completed = {str(row["settings_fingerprint"]) for row in rows if _has_value(row, "settings_fingerprint")}

    cases = [
        (tf, amax, segments, max_nfev)
        for tf in parse_float_grid(args.transfer_times)
        for amax in parse_float_grid(args.amax)
        for segments in parse_int_grid(args.segments)
        for max_nfev in parse_int_grid(args.max_nfev)
    ]
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    for transfer_time, amax, segments, max_nfev in cases:
        case_settings = settings_values_for_case(base_config, transfer_time, amax, segments, max_nfev, args)
        key = _settings_fingerprint(case_settings)
        if key in completed:
            continue
        row = run_case(base_config, transfer_time, amax, segments, max_nfev, args)
        rows.append(row)
        completed.add(str(row["settings_fingerprint"]))
        df = normalize_multiple_shooting_dataframe(pd.DataFrame(rows))
        df.to_csv(csv_path, index=False)
        regenerate_artifacts(df, results_dir, figures_dir, tables_dir, args, base_config)
        print(
            "case "
            f"tf={transfer_time} amax={amax} N={segments} max_nfev={max_nfev}: "
            f"init={row['node_initialization']}, nominal={row['nominal_error']:.4f}, "
            f"selected_worst={row['selected_worst_error']:.4f}, "
            f"all_worst={row['all_mask_worst_error']:.4f}, met={row['meets_thresholds']}, "
            f"runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    df = normalize_multiple_shooting_dataframe(pd.DataFrame(rows))
    df.to_csv(csv_path, index=False)
    regenerate_artifacts(df, results_dir, figures_dir, tables_dir, args, base_config)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Targeted multiple-shooting feasibility runs for all-windows branch recovery.")
    parser.add_argument("--config", type=Path, default=Path("configs/q1_candidate.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--transfer-times", default="4.0")
    parser.add_argument("--amax", default="0.2")
    parser.add_argument("--segments", default="14")
    parser.add_argument("--max-nfev", default="120")
    parser.add_argument("--selected-outages", type=int, default=3)
    parser.add_argument("--min-recovery-segments", type=int, default=4)
    parser.add_argument("--node-initialization", choices=["rollout", "linear", "blend"], default=None)
    parser.add_argument("--node-initialization-blend", "--node-blend", dest="node_initialization_blend", type=float, default=None)
    parser.add_argument("--nominal-first-homotopy", "--nominal-first", dest="nominal_first_homotopy", action="store_true", default=None)
    parser.add_argument("--nominal-first-nfev", type=int, default=None)
    parser.add_argument("--initial-weight", type=float, default=10.0)
    parser.add_argument("--defect-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-start-weight", type=float, default=5.0)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--solver-mode", default=None, help="bounded projected control mode; legacy aliases route to bounded_projected_multiple_shooting")
    parser.add_argument("--warm-start-refinement", action="store_true")
    parser.add_argument("--warm-start-nfev", type=int, default=90)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
