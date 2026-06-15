from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .cr3bp import propagate_controls_batch
from .multiple_shooting import run_multiple_shooting_baseline
from .objective import ObjectiveConfig, state_error
from .refinement import control_fuel, project_controls_to_ball


MODE = "locked_nominal_independent_branch_recovery"


@dataclass(frozen=True)
class BranchRecoveryWeights:
    terminal: float = 4.0
    control: float = 0.01
    smooth: float = 0.01
    continuity: float = 0.0

    @classmethod
    def from_config(cls, config: dict | None) -> "BranchRecoveryWeights":
        raw = dict(config or {})
        weights = dict(raw.get("weights", {}) or {})
        return cls(
            terminal=float(weights.get("terminal", raw.get("terminal_weight", cls.terminal))),
            control=float(weights.get("control", raw.get("control_weight", cls.control))),
            smooth=float(weights.get("smooth", raw.get("smooth_weight", cls.smooth))),
            continuity=float(weights.get("continuity", raw.get("continuity_weight", cls.continuity))),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "terminal": self.terminal,
            "control": self.control,
            "smooth": self.smooth,
            "continuity": self.continuity,
        }


@dataclass(frozen=True)
class SelectedOutagePolicy:
    kind: str
    count: int | None = None
    raw: str | int = 0

    @property
    def label(self) -> str:
        if self.kind == "hardest":
            return str(int(self.count or 0))
        return self.kind


def scale_vector(cfg: ObjectiveConfig) -> np.ndarray:
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


def scaled_state_residual(state: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    return (np.asarray(state, dtype=float) - np.asarray(target, dtype=float)) / np.maximum(scale_vector(cfg), 1e-12)


def outage_end(mask: np.ndarray) -> int:
    missed = np.flatnonzero(np.asarray(mask, dtype=float) < 0.5)
    if missed.size == 0:
        return 0
    return int(missed[-1] + 1)


def normalize_selected_outage_policy(value) -> SelectedOutagePolicy:
    if isinstance(value, SelectedOutagePolicy):
        return value
    if isinstance(value, (int, np.integer)):
        count = int(value)
        if count < 0:
            raise ValueError("selected_outages must be non-negative")
        return SelectedOutagePolicy(kind="hardest", count=count, raw=count)
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "all": "all_configured",
        "all_masks": "all_configured",
        "all_configured_outages": "all_configured",
        "all_configured_outage_masks": "all_configured",
        "all_single_outages": "all_single",
        "all_single_outage_masks": "all_single",
        "single": "all_single",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"all_single", "all_configured"}:
        return SelectedOutagePolicy(kind=normalized, raw=value)
    try:
        count = int(normalized)
    except ValueError as exc:
        raise ValueError("selected_outages must be an integer, all_single, all_configured, or all") from exc
    if count < 0:
        raise ValueError("selected_outages must be non-negative")
    return SelectedOutagePolicy(kind="hardest", count=count, raw=value)


def selected_outage_count_for_policy(policy, masks: np.ndarray) -> int:
    parsed = normalize_selected_outage_policy(policy)
    masks = np.asarray(masks, dtype=float)
    if parsed.kind == "hardest":
        return min(int(parsed.count or 0), int(masks.shape[0]))
    if parsed.kind == "all_configured":
        return int(masks.shape[0])
    if parsed.kind == "all_single":
        if masks.size == 0:
            return 0
        return int(np.count_nonzero(np.sum(masks < 0.5, axis=1) == 1))
    raise ValueError(f"unknown selected outage policy: {parsed.kind}")


def propagate_projected_controls(
    state0: np.ndarray,
    controls: np.ndarray,
    cfg: ObjectiveConfig,
) -> tuple[np.ndarray, np.ndarray]:
    controls = project_controls_to_ball(np.asarray(controls, dtype=float), cfg.amax)
    finals, history = propagate_controls_batch(
        state0,
        controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        return_history=True,
    )
    return finals[0], history[:, 0, :]


def terminal_error(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    controls: np.ndarray,
) -> float:
    final, _ = propagate_projected_controls(state0, controls, cfg)
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def masked_nominal_controls(nominal_controls: np.ndarray, mask: np.ndarray, amax: float) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    return project_controls_to_ball(controls, amax)


def branch_full_controls(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    recovery_controls: np.ndarray,
    amax: float,
) -> np.ndarray:
    nominal = project_controls_to_ball(np.asarray(nominal_controls, dtype=float), amax)
    mask = np.asarray(mask, dtype=float)
    start = outage_end(mask)
    controls = nominal.copy()
    if start > 0:
        controls[:start] *= mask[:start, None]
    if start < controls.shape[0]:
        recovery = np.asarray(recovery_controls, dtype=float).reshape((controls.shape[0] - start, 3))
        controls[start:] = recovery
    return project_controls_to_ball(controls, amax)


def control_norm_diagnostics(control_sets: list[np.ndarray], amax: float) -> dict[str, float]:
    max_norm = 0.0
    for controls in control_sets:
        arr = np.asarray(controls, dtype=float)
        if arr.size == 0:
            continue
        max_norm = max(max_norm, float(np.max(np.linalg.norm(arr, axis=-1))))
    return {
        "control_max_norm": max_norm,
        "control_bound_violation": max(0.0, max_norm - float(amax)),
    }


def outage_hardness_errors(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    masks = np.asarray(masks, dtype=float)
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    nominal = project_controls_to_ball(np.asarray(nominal_controls, dtype=float), cfg.amax)
    controls = project_controls_to_ball(nominal[None, :, :] * masks[:, :, None], cfg.amax)
    finals, _ = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)
    return state_error(finals, target, cfg.position_scale, cfg.velocity_scale).astype(float)


def select_outage_indices(
    *,
    policy,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    masks: np.ndarray,
    min_recovery_segments: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    parsed = normalize_selected_outage_policy(policy)
    masks = np.asarray(masks, dtype=float)
    hardness = outage_hardness_errors(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
    )
    if parsed.kind == "all_configured":
        selected = np.arange(masks.shape[0], dtype=int)
        semantics = "all configured outage masks are selected for independent locked-nominal recovery"
        return selected, hardness, semantics
    if parsed.kind == "all_single":
        selected = np.flatnonzero(np.sum(masks < 0.5, axis=1) == 1).astype(int)
        semantics = "all one-segment outage masks are selected for independent locked-nominal recovery"
        return selected, hardness, semantics

    count = min(int(parsed.count or 0), int(masks.shape[0]))
    if count <= 0:
        semantics = "selected_outages=0; no branch recovery optimizer is run"
        return np.zeros(0, dtype=int), hardness, semantics

    eligible = np.arange(masks.shape[0], dtype=int)
    if int(min_recovery_segments) > 0:
        filtered = np.asarray(
            [
                int(index)
                for index, mask in enumerate(masks)
                if cfg.n_segments - outage_end(mask) >= int(min_recovery_segments)
            ],
            dtype=int,
        )
        if filtered.size >= count:
            eligible = filtered
    ranked = eligible[np.argsort(hardness[eligible])[::-1]]
    selected = ranked[:count].astype(int)
    semantics = (
        f"the {count} hardest eligible outage masks are selected by terminal error under fixed "
        "locked nominal controls with the missed-thrust mask applied"
    )
    return selected, hardness, semantics


def optimize_locked_recovery_branch(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    max_nfev: int,
    weights: BranchRecoveryWeights | dict | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    if not isinstance(weights, BranchRecoveryWeights):
        weights = BranchRecoveryWeights.from_config(weights)
    nominal_locked = project_controls_to_ball(np.asarray(nominal_controls, dtype=float).copy(), cfg.amax)
    mask = np.asarray(mask, dtype=float)
    start = outage_end(mask)
    recovery_segments = cfg.n_segments - start
    scale = max(float(cfg.amax), 1e-12)

    def evaluate(recovery: np.ndarray, label: str, optimizer_result=None) -> dict:
        full = branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        final, history = propagate_projected_controls(state0, full, cfg)
        error = float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        out = {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": int(recovery_segments),
            "accepted_candidate": label,
            "terminal_error": error,
            "error": error,
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": project_controls_to_ball(recovery, cfg.amax),
            "history": history,
            **diagnostics,
        }
        if optimizer_result is not None:
            out.update(
                {
                    "optimizer_success": bool(optimizer_result.success),
                    "message": str(optimizer_result.message),
                    "cost": float(optimizer_result.cost),
                    "optimality": float(optimizer_result.optimality),
                    "nfev": int(optimizer_result.nfev),
                }
            )
        return out

    if recovery_segments <= 0:
        full = masked_nominal_controls(nominal_locked, mask, cfg.amax)
        error = terminal_error(state0, target, cfg, full)
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        return {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": 0,
            "accepted_candidate": "no_recovery_variables",
            "optimizer_success": True,
            "message": "no post-outage recovery controls are available",
            "cost": 0.0,
            "optimality": float("nan"),
            "nfev": 0,
            "runtime_seconds": float(time.perf_counter() - start_time),
            "terminal_error": error,
            "error": error,
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": np.zeros((0, 3), dtype=float),
            "history": None,
            **diagnostics,
        }

    x0 = nominal_locked[start:].reshape(-1)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return project_controls_to_ball(np.asarray(vec, dtype=float).reshape((recovery_segments, 3)), cfg.amax)

    def residual(vec: np.ndarray) -> np.ndarray:
        recovery = unpack(vec)
        full = branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        final, _ = propagate_projected_controls(state0, full, cfg)
        chunks = [weights.terminal * scaled_state_residual(final, target, cfg)]
        if weights.control:
            chunks.append(weights.control * recovery.reshape(-1) / scale)
        if weights.smooth and recovery.shape[0] > 1:
            chunks.append(weights.smooth * np.diff(recovery, axis=0).reshape(-1) / scale)
        if weights.continuity and recovery.shape[0] > 0:
            previous = nominal_locked[start - 1] * mask[start - 1] if start > 0 else nominal_locked[0]
            chunks.append(weights.continuity * (recovery[0] - previous) / scale)
        return np.concatenate(chunks)

    initial_residual = residual(x0)
    initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
    initial_eval = evaluate(unpack(x0), "initial")
    initial_eval["cost"] = initial_cost
    initial_eval["accepted_cost"] = initial_cost
    initial_eval["optimality"] = float("nan")
    initial_eval["accepted_optimality"] = float("nan")
    if int(max_nfev) <= 0:
        initial_eval.update(
            {
                "optimizer_success": True,
                "message": "branch optimization skipped because max_nfev <= 0",
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
    optimizer_eval["accepted_cost"] = float(optimizer_eval["cost"])
    optimizer_eval["accepted_optimality"] = float(optimizer_eval["optimality"])
    accepted = min(
        [initial_eval, optimizer_eval],
        key=lambda item: (
            float(item["terminal_error"]),
            float(item["branch_fuel"]),
        ),
    )
    accepted = dict(accepted)
    if accepted["accepted_candidate"] == "initial":
        accepted.update(
            {
                "optimizer_success": bool(result.success),
                "message": f"accepted initial recovery controls; optimizer message: {result.message}",
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted


def branch_summary_for_json(branch: dict) -> dict:
    return {
        "mask_index": int(branch["mask_index"]),
        "recovery_start": int(branch["recovery_start"]),
        "recovery_segments": int(branch["recovery_segments"]),
        "terminal_error": float(branch["terminal_error"]),
        "branch_fuel": float(branch["branch_fuel"]),
        "nfev": int(branch["nfev"]),
        "runtime_seconds": float(branch["runtime_seconds"]),
        "optimizer_success": bool(branch["optimizer_success"]),
        "accepted_candidate": str(branch["accepted_candidate"]),
        "message": str(branch["message"]),
    }


def run_locked_nominal_recovery_baseline(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray,
    thresholds: dict,
    selected_outages=1,
    nominal_max_nfev: int = 140,
    branch_max_nfev: int = 120,
    min_recovery_segments: int = 0,
    nominal_residual_weights: dict | None = None,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    node_initialization: str | None = "linear",
    node_initialization_blend: float | None = 0.5,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    if not isinstance(branch_weights, BranchRecoveryWeights):
        branch_weights = BranchRecoveryWeights.from_config(branch_weights)
    masks = np.asarray(masks, dtype=float)
    nominal_result = run_multiple_shooting_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds=thresholds,
        selected_outages=0,
        max_nfev=int(nominal_max_nfev),
        min_recovery_segments=int(min_recovery_segments),
        residual_weights=nominal_residual_weights,
        nominal_control_guess=None,
        selected_branch_control_guesses=None,
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
        warm_start_info={"stage": "locked_nominal_solve"},
    )
    nominal_controls = project_controls_to_ball(np.asarray(nominal_result["controls"], dtype=float), cfg.amax)
    nominal_error = terminal_error(state0, target, cfg, nominal_controls)
    nominal_reported_error = float(nominal_result.get("nominal_error", nominal_error))
    nominal_lock_error_delta = abs(nominal_error - nominal_reported_error)

    selected, nominal_mask_errors, selection_semantics = select_outage_indices(
        policy=selected_outages,
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        min_recovery_segments=int(min_recovery_segments),
    )

    branches: list[dict] = []
    for index in selected.astype(int).tolist():
        branches.append(
            optimize_locked_recovery_branch(
                state0=state0,
                target=target,
                cfg=cfg,
                nominal_controls=nominal_controls,
                mask=masks[int(index)],
                mask_index=int(index),
                max_nfev=int(branch_max_nfev),
                weights=branch_weights,
                xtol=float(xtol),
                ftol=float(ftol),
                gtol=float(gtol),
            )
        )

    selected_errors = [float(branch["terminal_error"]) for branch in branches]
    selected_worst = float(np.max(selected_errors)) if selected_errors else nominal_error
    all_errors = nominal_mask_errors.astype(float).tolist()
    for branch in branches:
        all_errors[int(branch["mask_index"])] = float(branch["terminal_error"])
    all_worst = float(np.max(all_errors)) if all_errors else selected_worst
    branch_controls = [np.asarray(branch["branch_controls"], dtype=float) for branch in branches]
    branch_fuels = [float(branch["branch_fuel"]) for branch in branches]
    nominal_fuel = control_fuel(nominal_controls, cfg.tf)
    diagnostics = control_norm_diagnostics([nominal_controls, *branch_controls], cfg.amax)
    nominal_threshold = float(thresholds["nominal_success"])
    selected_threshold = float(thresholds["robust_success"])
    meets_nominal = bool(nominal_error <= nominal_threshold)
    meets_selected = bool(selected_worst <= selected_threshold)
    backend_success = bool(meets_nominal and meets_selected)
    branch_optimizer_success = bool(all(bool(branch["optimizer_success"]) for branch in branches))
    total_branch_nfev = int(sum(int(branch["nfev"]) for branch in branches))
    total_branch_runtime = float(sum(float(branch["runtime_seconds"]) for branch in branches))
    nominal_nfev = int(nominal_result.get("nfev", 0) or 0)
    nominal_runtime = float(nominal_result.get("runtime_seconds", 0.0) or 0.0)

    branch_summaries = [branch_summary_for_json(branch) for branch in branches]
    return {
        "success": backend_success,
        "backend_success": backend_success,
        "mode": MODE,
        "method_type": MODE,
        "optimizer_success": branch_optimizer_success,
        "nominal_optimizer_success": bool(nominal_result.get("optimizer_success", False)),
        "nominal_backend_success": bool(nominal_result.get("success", False)),
        "message": "locked nominal branch recovery complete",
        "nominal_message": str(nominal_result.get("message", "")),
        "nominal_accepted_candidate": str(nominal_result.get("accepted_candidate", "optimizer")),
        "nominal_error": nominal_error,
        "nominal_baseline_error": nominal_reported_error,
        "nominal_lock_error_delta": float(nominal_lock_error_delta),
        "worst_error_semantics": "selected_worst_error is the worst terminal error among independently optimized selected locked-nominal branches",
        "worst_error": selected_worst,
        "selected_recovery_worst_error": selected_worst,
        "selected_worst_error": selected_worst,
        "all_outage_worst_error": all_worst,
        "all_mask_worst_error": all_worst,
        "nominal_threshold": nominal_threshold,
        "selected_worst_threshold": selected_threshold,
        "selected_recovery_threshold": selected_threshold,
        "meets_nominal_threshold": meets_nominal,
        "meets_selected_worst_threshold": meets_selected,
        "meets_selected_recovery_threshold": meets_selected,
        "meets_thresholds": backend_success,
        "nominal_fuel": float(nominal_fuel),
        "recovery_fuel_mean": float(np.mean(branch_fuels)) if branch_fuels else float(nominal_fuel),
        "recovery_fuel_max": float(np.max(branch_fuels)) if branch_fuels else float(nominal_fuel),
        "control_max_norm": float(diagnostics["control_max_norm"]),
        "control_bound_violation": float(diagnostics["control_bound_violation"]),
        "controls": nominal_controls,
        "nominal_controls": nominal_controls,
        "selected_outage_indices": selected.astype(int).tolist(),
        "selected_outage_errors": selected_errors,
        "all_outage_errors": all_errors,
        "nominal_masked_outage_errors": nominal_mask_errors.astype(float).tolist(),
        "branch_results": branch_summaries,
        "branch_nfev": [int(branch["nfev"]) for branch in branches],
        "branch_runtime_seconds": [float(branch["runtime_seconds"]) for branch in branches],
        "branch_optimizer_success": [bool(branch["optimizer_success"]) for branch in branches],
        "branch_recovery_starts": [int(branch["recovery_start"]) for branch in branches],
        "branch_recovery_segments": [int(branch["recovery_segments"]) for branch in branches],
        "total_branch_nfev": total_branch_nfev,
        "total_branch_runtime_seconds": total_branch_runtime,
        "nominal_nfev": nominal_nfev,
        "nominal_runtime_seconds": nominal_runtime,
        "nfev": int(nominal_nfev + total_branch_nfev),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "cost": float(sum(float(branch.get("cost", 0.0)) for branch in branches)),
        "optimality": float(max([float(branch.get("optimality", 0.0)) for branch in branches] or [0.0])),
        "nominal_cost": float(nominal_result.get("cost", 0.0)),
        "nominal_optimality": float(nominal_result.get("optimality", 0.0)),
        "branch_weights": branch_weights.as_dict(),
        "selection_semantics": selection_semantics,
        "backend_semantics": (
            "Locked-nominal continuous baseline: nominal all-windows controls are solved once, kept fixed, "
            "and each selected missed-thrust recovery branch is optimized independently after the outage."
        ),
        "selected_branch_semantics": (
            "For selected masks, controls before outage_end(mask) are exactly locked nominal controls with missed "
            "segments zeroed; only controls from outage_end(mask) onward are optimization variables."
        ),
        "all_mask_diagnostic_semantics": (
            "all_outage_errors evaluates every configured outage mask; selected masks use their optimized recovery "
            "branches, while unselected masks use masked locked nominal controls only."
        ),
        "control_bound_semantics": (
            "Every nominal and recovery acceleration vector is projected to the Euclidean ball ||u_i|| <= amax "
            "before propagation, residual evaluation, fuel computation, and reporting."
        ),
        "nominal_lock_semantics": (
            "Branch optimization never changes nominal_controls; nominal_error is the locked nominal baseline "
            "terminal error and nominal_lock_error_delta reports agreement with the nominal solve output."
        ),
    }
