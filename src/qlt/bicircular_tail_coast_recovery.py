from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

from .bicircular import SolarTidalParameters, propagate_controls_batch_bicircular
from .locked_recovery import BranchRecoveryWeights, control_norm_diagnostics, outage_end, scaled_state_residual
from .objective import ObjectiveConfig, state_error
from .refinement import control_fuel, project_controls_to_ball
from .tail_coast_recovery import (
    NO_RECOVERY_VARIABLES_LABEL,
    NOMINAL_BRANCH_INITIALIZATION_LABEL,
    _tail_zero_notes,
    nominal_segment_duration,
    normalize_tail_coast_segments,
    tail_coast_branch_full_controls,
    tail_coast_full_controls,
)


MODE = "simple_bicircular_fixed_final_time_tail_coast_retuned_recovery"


def branch_recovery_weights_from_mapping(value: BranchRecoveryWeights | Mapping | None) -> BranchRecoveryWeights:
    if isinstance(value, BranchRecoveryWeights):
        return value
    raw = dict(value or {})
    if any(key in raw for key in ("terminal", "control", "smooth", "continuity")):
        return BranchRecoveryWeights(
            terminal=float(raw.get("terminal", BranchRecoveryWeights.terminal)),
            control=float(raw.get("control", BranchRecoveryWeights.control)),
            smooth=float(raw.get("smooth", BranchRecoveryWeights.smooth)),
            continuity=float(raw.get("continuity", BranchRecoveryWeights.continuity)),
        )
    return BranchRecoveryWeights.from_config(raw)


def propagate_tail_controls_bicircular(
    state0: np.ndarray,
    controls: np.ndarray,
    cfg: ObjectiveConfig,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    return_history: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    controls = project_controls_to_ball(np.asarray(controls, dtype=float), cfg.amax)
    finals, history = propagate_controls_batch_bicircular(
        state0,
        controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        phase_rad=float(phase_rad),
        parameters=parameters,
        return_history=bool(return_history),
    )
    if history is None:
        return finals[0], None
    return finals[0], history[:, 0, :]


def terminal_error_bicircular(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    controls: np.ndarray,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> float:
    final, _ = propagate_tail_controls_bicircular(
        state0,
        controls,
        cfg,
        phase_rad=phase_rad,
        parameters=parameters,
        return_history=False,
    )
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def bicircular_terminal_residual(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    controls: np.ndarray,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    terminal_weight: float = 1.0,
) -> np.ndarray:
    final, _ = propagate_tail_controls_bicircular(
        state0,
        controls,
        cfg,
        phase_rad=phase_rad,
        parameters=parameters,
        return_history=False,
    )
    return float(terminal_weight) * scaled_state_residual(final, target, cfg)


def optimize_bicircular_tail_coast_nominal(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    seed_controls: np.ndarray,
    tail_coast_segments: int,
    max_nfev: int,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    weights: BranchRecoveryWeights | Mapping | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    weights = branch_recovery_weights_from_mapping(weights)
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    prefix_count = int(cfg.n_segments) - int(tail)
    seed = project_controls_to_ball(np.asarray(seed_controls, dtype=float), cfg.amax)
    if seed.shape != (int(cfg.n_segments), 3):
        raise ValueError(f"seed_controls has shape {seed.shape}, expected {(int(cfg.n_segments), 3)}")
    scale = max(float(cfg.amax), 1e-12)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return tail_coast_full_controls(np.asarray(vec, dtype=float).reshape((prefix_count, 3)), tail, cfg)

    def residual(vec: np.ndarray) -> np.ndarray:
        controls = unpack(vec)
        chunks = [
            bicircular_terminal_residual(
                state0=state0,
                target=target,
                cfg=cfg,
                controls=controls,
                phase_rad=phase_rad,
                parameters=parameters,
                terminal_weight=weights.terminal,
            )
        ]
        prefix = controls[:prefix_count]
        if weights.control and prefix.size:
            chunks.append(weights.control * prefix.reshape(-1) / scale)
        if weights.smooth and prefix.shape[0] > 1:
            chunks.append(weights.smooth * np.diff(prefix, axis=0).reshape(-1) / scale)
        return np.concatenate(chunks)

    x0 = seed[:prefix_count].reshape(-1)
    initial_controls = unpack(x0)
    initial_residual = residual(x0)
    initial_error = terminal_error_bicircular(
        state0,
        target,
        cfg,
        initial_controls,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    initial_eval = {
        "accepted_candidate": "initial",
        "controls": initial_controls,
        "nominal_controls": initial_controls,
        "nominal_error": initial_error,
        "terminal_error": initial_error,
        "error": initial_error,
        "initial_terminal_error": initial_error,
        "optimizer_terminal_error": float("nan"),
        "cost": float(0.5 * np.dot(initial_residual, initial_residual)),
        "accepted_cost": float(0.5 * np.dot(initial_residual, initial_residual)),
        "optimality": float("nan"),
        "accepted_optimality": float("nan"),
        "optimizer_ran": False,
        "optimizer_success": False,
        "message": "bicircular tail-coast nominal optimization skipped because max_nfev <= 0",
        "nfev": 0,
    }
    if int(max_nfev) <= 0 or prefix_count <= 0:
        if prefix_count <= 0:
            initial_eval["message"] = "all nominal segments are exact tail-coast zero controls"
        initial_eval["runtime_seconds"] = float(time.perf_counter() - start_time)
        return initial_eval

    result = least_squares(
        residual,
        x0,
        bounds=(-float(cfg.amax), float(cfg.amax)),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(ftol),
        gtol=float(gtol),
        verbose=0,
    )
    optimizer_controls = unpack(result.x)
    optimizer_error = terminal_error_bicircular(
        state0,
        target,
        cfg,
        optimizer_controls,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    optimizer_eval = {
        "accepted_candidate": "optimizer",
        "controls": optimizer_controls,
        "nominal_controls": optimizer_controls,
        "nominal_error": optimizer_error,
        "terminal_error": optimizer_error,
        "error": optimizer_error,
        "initial_terminal_error": initial_error,
        "optimizer_terminal_error": optimizer_error,
        "cost": float(result.cost),
        "accepted_cost": float(result.cost),
        "optimality": float(result.optimality),
        "accepted_optimality": float(result.optimality),
        "optimizer_ran": True,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
    }
    accepted = min(
        [initial_eval, optimizer_eval],
        key=lambda item: (float(item["nominal_error"]), float(control_fuel(item["controls"], cfg.tf))),
    )
    accepted = dict(accepted)
    if accepted["accepted_candidate"] == "initial":
        accepted.update(
            {
                "optimizer_ran": True,
                "optimizer_success": bool(result.success),
                "message": f"accepted initial bicircular tail-coast controls; optimizer message: {result.message}",
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "optimizer_terminal_error": optimizer_error,
                "nfev": int(result.nfev),
            }
        )
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted


def optimize_bicircular_tail_coast_branch(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    seed_branch_controls: np.ndarray,
    tail_coast_segments: int,
    max_nfev: int,
    robust_threshold: float,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    weights: BranchRecoveryWeights | Mapping | None = None,
    initialization_label: str = NOMINAL_BRANCH_INITIALIZATION_LABEL,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    weights = branch_recovery_weights_from_mapping(weights)
    nominal_locked = project_controls_to_ball(np.asarray(nominal_controls, dtype=float).copy(), cfg.amax)
    if nominal_locked.shape != (int(cfg.n_segments), 3):
        raise ValueError(f"nominal_controls has shape {nominal_locked.shape}, expected {(int(cfg.n_segments), 3)}")
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    if tail:
        nominal_locked[-tail:] = 0.0
    mask = np.asarray(mask, dtype=float)
    start = outage_end(mask)
    recovery_segments = int(cfg.n_segments) - int(start)
    seed_branch = project_controls_to_ball(np.asarray(seed_branch_controls, dtype=float), cfg.amax)
    if seed_branch.shape != (int(cfg.n_segments), 3):
        raise ValueError(f"seed_branch_controls has shape {seed_branch.shape}, expected {(int(cfg.n_segments), 3)}")
    scale = max(float(cfg.amax), 1e-12)
    notes = _tail_zero_notes(nominal_locked, mask, tail)

    def evaluate(recovery: np.ndarray, label: str, optimizer_result=None) -> dict:
        full = tail_coast_branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        final, history = propagate_tail_controls_bicircular(
            state0,
            full,
            cfg,
            phase_rad=phase_rad,
            parameters=parameters,
            return_history=True,
        )
        error = float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        out = {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": int(recovery_segments),
            "tail_coast_segments": int(tail),
            "branch_control_count": int(full.shape[0]),
            "nominal_segments": int(cfg.n_segments),
            "nominal_dt": float(nominal_segment_duration(cfg)),
            "original_transfer_time": float(cfg.tf),
            "branch_total_duration": float(cfg.tf),
            "accepted_candidate": label,
            "branch_initialization_label": str(initialization_label),
            "branch_initialization_index": 0,
            "branch_initialization_is_fallback": False,
            "branch_initialization_kind": str(initialization_label),
            "terminal_error": error,
            "error": error,
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target at the original fixed final time under simple bicircular solar-tidal dynamics",
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": project_controls_to_ball(recovery, cfg.amax),
            "history": history,
            "meets_robust_threshold": bool(error <= float(robust_threshold)),
            **notes,
            **diagnostics,
        }
        if optimizer_result is not None:
            out.update(
                {
                    "optimizer_ran": True,
                    "optimizer_success": bool(optimizer_result.success),
                    "message": str(optimizer_result.message),
                    "cost": float(optimizer_result.cost),
                    "optimality": float(optimizer_result.optimality),
                    "nfev": int(optimizer_result.nfev),
                }
            )
        return out

    if recovery_segments <= 0:
        full = tail_coast_branch_full_controls(nominal_locked, mask, np.zeros((0, 3), dtype=float), cfg.amax)
        error = terminal_error_bicircular(
            state0,
            target,
            cfg,
            full,
            phase_rad=phase_rad,
            parameters=parameters,
        )
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        return {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": 0,
            "tail_coast_segments": int(tail),
            "branch_control_count": int(full.shape[0]),
            "nominal_segments": int(cfg.n_segments),
            "nominal_dt": float(nominal_segment_duration(cfg)),
            "original_transfer_time": float(cfg.tf),
            "branch_total_duration": float(cfg.tf),
            "accepted_candidate": NO_RECOVERY_VARIABLES_LABEL,
            "branch_initialization_label": NO_RECOVERY_VARIABLES_LABEL,
            "branch_initialization_index": 0,
            "branch_initialization_is_fallback": False,
            "branch_initialization_kind": NO_RECOVERY_VARIABLES_LABEL,
            "optimizer_ran": False,
            "optimizer_success": False,
            "no_recovery_variable_threshold_feasible": bool(error <= float(robust_threshold)),
            "message": "no post-outage fixed-final-time recovery controls are available; bicircular threshold feasibility is evaluated directly",
            "cost": 0.0,
            "accepted_cost": 0.0,
            "optimality": float("nan"),
            "accepted_optimality": float("nan"),
            "nfev": 0,
            "runtime_seconds": float(time.perf_counter() - start_time),
            "terminal_error": error,
            "error": error,
            "initial_terminal_error": error,
            "optimizer_terminal_error": float("nan"),
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target at the original fixed final time under simple bicircular solar-tidal dynamics",
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": np.zeros((0, 3), dtype=float),
            "history": None,
            "meets_robust_threshold": bool(error <= float(robust_threshold)),
            **notes,
            **diagnostics,
        }

    initial_controls = seed_branch[start:]
    initial_controls = project_controls_to_ball(initial_controls, cfg.amax)
    x0 = initial_controls.reshape(-1)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return project_controls_to_ball(np.asarray(vec, dtype=float).reshape((recovery_segments, 3)), cfg.amax)

    def residual(vec: np.ndarray) -> np.ndarray:
        recovery = unpack(vec)
        full = tail_coast_branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        chunks = [
            bicircular_terminal_residual(
                state0=state0,
                target=target,
                cfg=cfg,
                controls=full,
                phase_rad=phase_rad,
                parameters=parameters,
                terminal_weight=weights.terminal,
            )
        ]
        if weights.control:
            chunks.append(weights.control * recovery.reshape(-1) / scale)
        if weights.smooth and recovery.shape[0] > 1:
            chunks.append(weights.smooth * np.diff(recovery, axis=0).reshape(-1) / scale)
        if weights.continuity and recovery.shape[0] > 0:
            previous = nominal_locked[start - 1] * mask[start - 1] if start > 0 else nominal_locked[0]
            chunks.append(weights.continuity * (recovery[0] - previous) / scale)
        return np.concatenate(chunks)

    initial_residual = residual(x0)
    initial_eval = evaluate(unpack(x0), "initial")
    initial_eval["initial_terminal_error"] = float(initial_eval["terminal_error"])
    initial_eval["optimizer_terminal_error"] = float("nan")
    initial_eval["cost"] = float(0.5 * np.dot(initial_residual, initial_residual))
    initial_eval["accepted_cost"] = float(initial_eval["cost"])
    initial_eval["optimality"] = float("nan")
    initial_eval["accepted_optimality"] = float("nan")
    if int(max_nfev) <= 0:
        initial_eval.update(
            {
                "optimizer_ran": False,
                "optimizer_success": False,
                "message": "bicircular branch optimization skipped because max_nfev <= 0",
                "nfev": 0,
                "runtime_seconds": float(time.perf_counter() - start_time),
            }
        )
        return initial_eval

    result = least_squares(
        residual,
        x0,
        bounds=(-float(cfg.amax), float(cfg.amax)),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(ftol),
        gtol=float(gtol),
        verbose=0,
    )
    optimizer_eval = evaluate(unpack(result.x), "optimizer", result)
    optimizer_eval["initial_terminal_error"] = float(initial_eval["terminal_error"])
    optimizer_eval["optimizer_terminal_error"] = float(optimizer_eval["terminal_error"])
    optimizer_eval["accepted_cost"] = float(optimizer_eval["cost"])
    optimizer_eval["accepted_optimality"] = float(optimizer_eval["optimality"])
    accepted = min([initial_eval, optimizer_eval], key=lambda item: (float(item["terminal_error"]), float(item["branch_fuel"])))
    accepted = dict(accepted)
    if accepted["accepted_candidate"] == "initial":
        accepted.update(
            {
                "optimizer_ran": True,
                "optimizer_success": bool(result.success),
                "message": f"accepted initial bicircular recovery controls from {initialization_label}; optimizer message: {result.message}",
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "optimizer_terminal_error": float(optimizer_eval["terminal_error"]),
                "nfev": int(result.nfev),
            }
        )
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted
