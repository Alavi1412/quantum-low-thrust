from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .cr3bp import propagate_ballistic, propagate_controls_batch
from .locked_recovery import (
    BranchRecoveryWeights,
    control_norm_diagnostics,
    normalize_selected_outage_policy,
    outage_end,
    scaled_state_residual,
)
from .multiple_shooting import run_multiple_shooting_baseline
from .objective import ObjectiveConfig, state_error
from .refinement import control_fuel, project_controls_to_ball


MODE = "delayed_arrival_locked_nominal_independent_branch_recovery"


def normalize_branch_weight_variants(
    *,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    branch_weight_variants: list[dict] | tuple[dict, ...] | None = None,
) -> list[dict]:
    if not branch_weight_variants:
        if not isinstance(branch_weights, BranchRecoveryWeights):
            branch_weights = BranchRecoveryWeights.from_config(branch_weights)
        return [{"label": "configured", "index": 0, "weights": branch_weights}]

    variants: list[dict] = []
    for index, raw in enumerate(branch_weight_variants):
        if isinstance(raw, BranchRecoveryWeights):
            label = f"variant_{index}"
            weights = raw
        else:
            entry = dict(raw or {})
            label = str(entry.get("label", entry.get("name", f"variant_{index}")))
            if "weights" in entry:
                weights = BranchRecoveryWeights.from_config({"weights": dict(entry["weights"] or {})})
            else:
                weights = BranchRecoveryWeights.from_config(entry)
        variants.append({"label": label, "index": int(index), "weights": weights})
    if not variants:
        raise ValueError("branch_weight_variants must contain at least one variant")
    return variants


def branch_weight_variants_for_json(variants: list[dict]) -> list[dict]:
    return [
        {
            "label": str(variant["label"]),
            "index": int(variant["index"]),
            "weights": variant["weights"].as_dict(),
        }
        for variant in variants
    ]


def nominal_segment_duration(cfg: ObjectiveConfig) -> float:
    if int(cfg.n_segments) <= 0:
        raise ValueError("cfg.n_segments must be positive")
    return float(cfg.tf) / float(cfg.n_segments)


def normalize_recovery_horizon_segments(value: int | np.integer) -> int:
    horizon = int(value)
    if horizon < 0:
        raise ValueError("recovery_horizon_segments must be non-negative")
    return horizon


def delayed_target_time(cfg: ObjectiveConfig, recovery_horizon_segments: int) -> float:
    return normalize_recovery_horizon_segments(recovery_horizon_segments) * nominal_segment_duration(cfg)


def branch_total_duration(cfg: ObjectiveConfig, recovery_horizon_segments: int) -> float:
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    return float(cfg.n_segments + horizon) * nominal_segment_duration(cfg)


def delayed_target_state(
    target: np.ndarray,
    cfg: ObjectiveConfig,
    recovery_horizon_segments: int,
) -> np.ndarray:
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    target = np.asarray(target, dtype=float)
    if horizon == 0:
        return target.copy()
    return propagate_ballistic(
        target,
        cfg.mu,
        delayed_target_time(cfg, horizon),
        int(cfg.substeps) * horizon,
    )


def propagate_projected_controls_for_duration(
    state0: np.ndarray,
    controls: np.ndarray,
    cfg: ObjectiveConfig,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    controls = project_controls_to_ball(np.asarray(controls, dtype=float), cfg.amax)
    finals, history = propagate_controls_batch(
        state0,
        controls,
        cfg.mu,
        float(duration),
        cfg.substeps,
        return_history=True,
    )
    return finals[0], history[:, 0, :]


def delayed_recovery_initial_controls(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    recovery_horizon_segments: int,
) -> np.ndarray:
    nominal = np.asarray(nominal_controls, dtype=float)
    start = outage_end(mask)
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    pieces = []
    if start < nominal.shape[0]:
        pieces.append(nominal[start:])
    if horizon:
        pieces.append(np.zeros((horizon, 3), dtype=float))
    if not pieces:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pieces)


def delayed_branch_full_controls(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    recovery_controls: np.ndarray,
    amax: float,
    recovery_horizon_segments: int,
) -> np.ndarray:
    nominal = project_controls_to_ball(np.asarray(nominal_controls, dtype=float), amax)
    mask = np.asarray(mask, dtype=float)
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    if mask.shape != (nominal.shape[0],):
        raise ValueError(f"mask has shape {mask.shape}, expected {(nominal.shape[0],)}")
    start = outage_end(mask)
    total_segments = nominal.shape[0] + horizon
    controls = np.zeros((total_segments, 3), dtype=float)
    if start > 0:
        controls[:start] = nominal[:start] * mask[:start, None]
    recovery_segments = total_segments - start
    if recovery_segments > 0:
        recovery = np.asarray(recovery_controls, dtype=float).reshape((recovery_segments, 3))
        controls[start:] = recovery
    return project_controls_to_ball(controls, amax)


def delayed_masked_nominal_controls(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    amax: float,
    recovery_horizon_segments: int,
) -> np.ndarray:
    recovery = delayed_recovery_initial_controls(nominal_controls, mask, recovery_horizon_segments)
    return delayed_branch_full_controls(nominal_controls, mask, recovery, amax, recovery_horizon_segments)


def terminal_error_for_duration(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    controls: np.ndarray,
    duration: float,
) -> float:
    final, _ = propagate_projected_controls_for_duration(state0, controls, cfg, duration)
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def delayed_outage_hardness_errors(
    *,
    state0: np.ndarray,
    delayed_target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    masks: np.ndarray,
    recovery_horizon_segments: int,
) -> np.ndarray:
    masks = np.asarray(masks, dtype=float)
    if masks.size == 0:
        return np.zeros(0, dtype=float)
    duration = branch_total_duration(cfg, recovery_horizon_segments)
    errors: list[float] = []
    for mask in masks:
        controls = delayed_masked_nominal_controls(
            nominal_controls,
            mask,
            cfg.amax,
            recovery_horizon_segments,
        )
        errors.append(terminal_error_for_duration(state0, delayed_target, cfg, controls, duration))
    return np.asarray(errors, dtype=float)


def select_delayed_outage_indices(
    *,
    policy,
    state0: np.ndarray,
    delayed_target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    masks: np.ndarray,
    recovery_horizon_segments: int,
    min_recovery_segments: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    parsed = normalize_selected_outage_policy(policy)
    masks = np.asarray(masks, dtype=float)
    hardness = delayed_outage_hardness_errors(
        state0=state0,
        delayed_target=delayed_target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        recovery_horizon_segments=recovery_horizon_segments,
    )
    if parsed.kind == "all_configured":
        selected = np.arange(masks.shape[0], dtype=int)
        semantics = (
            "all configured outage masks are selected for delayed-arrival independent locked-nominal recovery"
        )
        return selected, hardness, semantics
    if parsed.kind == "all_single":
        selected = np.flatnonzero(np.sum(masks < 0.5, axis=1) == 1).astype(int)
        semantics = (
            "all one-segment outage masks are selected for delayed-arrival independent locked-nominal recovery"
        )
        return selected, hardness, semantics

    count = min(int(parsed.count or 0), int(masks.shape[0]))
    if count <= 0:
        semantics = "selected_outages=0; no delayed branch recovery optimizer is run"
        return np.zeros(0, dtype=int), hardness, semantics

    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    total_segments = int(cfg.n_segments) + horizon
    eligible = np.arange(masks.shape[0], dtype=int)
    if int(min_recovery_segments) > 0:
        filtered = np.asarray(
            [
                int(index)
                for index, mask in enumerate(masks)
                if total_segments - outage_end(mask) >= int(min_recovery_segments)
            ],
            dtype=int,
        )
        if filtered.size >= count:
            eligible = filtered
    ranked = eligible[np.argsort(hardness[eligible])[::-1]]
    selected = ranked[:count].astype(int)
    semantics = (
        f"the {count} hardest eligible outage masks are selected by terminal error against the "
        "delayed target under locked nominal controls with the missed-thrust mask applied and "
        "zero acceleration during the added recovery horizon"
    )
    return selected, hardness, semantics


def optimize_delayed_locked_recovery_branch(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    recovery_horizon_segments: int,
    max_nfev: int,
    weights: BranchRecoveryWeights | dict | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    if not isinstance(weights, BranchRecoveryWeights):
        weights = BranchRecoveryWeights.from_config(weights)
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    nominal_locked = project_controls_to_ball(np.asarray(nominal_controls, dtype=float).copy(), cfg.amax)
    mask = np.asarray(mask, dtype=float)
    start = outage_end(mask)
    total_segments = int(cfg.n_segments) + horizon
    recovery_segments = total_segments - start
    dt = nominal_segment_duration(cfg)
    branch_duration = branch_total_duration(cfg, horizon)
    target_delay = delayed_target_time(cfg, horizon)
    delayed_target = delayed_target_state(target, cfg, horizon)
    scale = max(float(cfg.amax), 1e-12)

    def evaluate(recovery: np.ndarray, label: str, optimizer_result=None) -> dict:
        full = delayed_branch_full_controls(nominal_locked, mask, recovery, cfg.amax, horizon)
        final, history = propagate_projected_controls_for_duration(state0, full, cfg, branch_duration)
        error = float(state_error(final, delayed_target, cfg.position_scale, cfg.velocity_scale))
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        out = {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": int(recovery_segments),
            "recovery_horizon_segments": int(horizon),
            "branch_control_count": int(full.shape[0]),
            "nominal_segments": int(cfg.n_segments),
            "nominal_dt": float(dt),
            "original_transfer_time": float(cfg.tf),
            "delayed_target_time": float(target_delay),
            "branch_total_duration": float(branch_duration),
            "accepted_candidate": label,
            "terminal_error": error,
            "error": error,
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target propagated ballistically by delayed_target_time",
            "branch_fuel": control_fuel(full, branch_duration),
            "branch_controls": full,
            "recovery_controls": project_controls_to_ball(recovery, cfg.amax),
            "history": history,
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
        full = delayed_branch_full_controls(
            nominal_locked,
            mask,
            np.zeros((0, 3), dtype=float),
            cfg.amax,
            horizon,
        )
        error = terminal_error_for_duration(state0, delayed_target, cfg, full, branch_duration)
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        return {
            "mask_index": int(mask_index),
            "recovery_start": int(start),
            "recovery_segments": 0,
            "recovery_horizon_segments": int(horizon),
            "branch_control_count": int(full.shape[0]),
            "nominal_segments": int(cfg.n_segments),
            "nominal_dt": float(dt),
            "original_transfer_time": float(cfg.tf),
            "delayed_target_time": float(target_delay),
            "branch_total_duration": float(branch_duration),
            "accepted_candidate": "no_recovery_variables",
            "optimizer_ran": False,
            "optimizer_success": False,
            "message": "no post-outage delayed recovery controls are available",
            "cost": 0.0,
            "optimality": float("nan"),
            "nfev": 0,
            "runtime_seconds": float(time.perf_counter() - start_time),
            "terminal_error": error,
            "error": error,
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target propagated ballistically by delayed_target_time",
            "branch_fuel": control_fuel(full, branch_duration),
            "branch_controls": full,
            "recovery_controls": np.zeros((0, 3), dtype=float),
            "history": None,
            **diagnostics,
        }

    x0 = delayed_recovery_initial_controls(nominal_locked, mask, horizon).reshape(-1)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return project_controls_to_ball(np.asarray(vec, dtype=float).reshape((recovery_segments, 3)), cfg.amax)

    def residual(vec: np.ndarray) -> np.ndarray:
        recovery = unpack(vec)
        full = delayed_branch_full_controls(nominal_locked, mask, recovery, cfg.amax, horizon)
        final, _ = propagate_projected_controls_for_duration(state0, full, cfg, branch_duration)
        chunks = [weights.terminal * scaled_state_residual(final, delayed_target, cfg)]
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
                "optimizer_ran": False,
                "optimizer_success": False,
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
                "optimizer_ran": True,
                "optimizer_success": bool(result.success),
                "message": f"accepted initial recovery controls; optimizer message: {result.message}",
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted


def delayed_branch_candidate_summary_for_json(branch: dict) -> dict:
    return {
        "variant_label": str(branch["branch_weight_variant_label"]),
        "variant_index": int(branch["branch_weight_variant_index"]),
        "weights": dict(branch["branch_weights"]),
        "terminal_error": float(branch["terminal_error"]),
        "robust_threshold": float(branch["portfolio_robust_threshold"]),
        "threshold_feasible": bool(branch["portfolio_threshold_feasible"]),
        "converged_threshold_feasible": bool(branch["portfolio_converged_threshold_feasible"]),
        "branch_fuel": float(branch["branch_fuel"]),
        "nfev": int(branch["nfev"]),
        "runtime_seconds": float(branch["runtime_seconds"]),
        "optimizer_ran": bool(branch["optimizer_ran"]),
        "optimizer_success": bool(branch["optimizer_success"]),
        "accepted_candidate": str(branch["accepted_candidate"]),
        "message": str(branch["message"]),
    }


def optimize_delayed_locked_recovery_branch_portfolio(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    recovery_horizon_segments: int,
    max_nfev: int,
    robust_threshold: float,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    branch_weight_variants: list[dict] | tuple[dict, ...] | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    variants = normalize_branch_weight_variants(
        branch_weights=branch_weights,
        branch_weight_variants=branch_weight_variants,
    )
    candidate_results: list[dict] = []
    for variant in variants:
        candidate = optimize_delayed_locked_recovery_branch(
            state0=state0,
            target=target,
            cfg=cfg,
            nominal_controls=nominal_controls,
            mask=mask,
            mask_index=mask_index,
            recovery_horizon_segments=recovery_horizon_segments,
            max_nfev=max_nfev,
            weights=variant["weights"],
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
        )
        candidate["branch_weight_variant_label"] = str(variant["label"])
        candidate["branch_weight_variant_index"] = int(variant["index"])
        candidate["branch_weights"] = variant["weights"].as_dict()
        candidate["portfolio_robust_threshold"] = float(robust_threshold)
        candidate["portfolio_threshold_feasible"] = bool(float(candidate["terminal_error"]) <= float(robust_threshold))
        candidate["portfolio_converged_threshold_feasible"] = bool(
            candidate["optimizer_success"] and candidate["portfolio_threshold_feasible"]
        )
        candidate_results.append(candidate)

    converged_feasible = [
        candidate
        for candidate in candidate_results
        if bool(candidate["portfolio_converged_threshold_feasible"])
    ]
    acceptance_pool = converged_feasible if converged_feasible else candidate_results
    if converged_feasible:
        portfolio_acceptance_rule = (
            "preferred optimizer_success=True candidates with terminal_error <= robust_success; "
            "selected lowest terminal_error with fuel as tie-breaker"
        )
    else:
        portfolio_acceptance_rule = (
            "no optimizer-converged threshold-feasible candidate was available; selected lowest "
            "terminal_error with fuel as tie-breaker"
        )
    accepted = min(
        acceptance_pool,
        key=lambda item: (
            float(item["terminal_error"]),
            float(item["branch_fuel"]),
        ),
    )
    accepted = dict(accepted)
    accepted_variant_nfev = int(accepted["nfev"])
    accepted_variant_runtime = float(accepted["runtime_seconds"])
    total_nfev = int(sum(int(candidate["nfev"]) for candidate in candidate_results))
    total_runtime = float(sum(float(candidate["runtime_seconds"]) for candidate in candidate_results))
    accepted["accepted_branch_weight_variant_label"] = str(accepted["branch_weight_variant_label"])
    accepted["accepted_branch_weight_variant_index"] = int(accepted["branch_weight_variant_index"])
    accepted["accepted_branch_weights"] = dict(accepted["branch_weights"])
    accepted["accepted_variant_nfev"] = accepted_variant_nfev
    accepted["accepted_variant_runtime_seconds"] = accepted_variant_runtime
    accepted["nfev"] = total_nfev
    accepted["runtime_seconds"] = total_runtime
    accepted["branch_portfolio_enabled"] = bool(len(variants) > 1)
    accepted["branch_portfolio_variant_count"] = int(len(variants))
    accepted["portfolio_acceptance_rule"] = portfolio_acceptance_rule
    accepted["portfolio_robust_threshold"] = float(robust_threshold)
    accepted["portfolio_converged_threshold_feasible_candidate_count"] = int(len(converged_feasible))
    accepted["branch_weight_variants"] = branch_weight_variants_for_json(variants)
    accepted["branch_portfolio_candidate_results"] = [
        delayed_branch_candidate_summary_for_json(candidate)
        for candidate in candidate_results
    ]
    return accepted


def delayed_branch_summary_for_json(branch: dict) -> dict:
    summary = {
        "mask_index": int(branch["mask_index"]),
        "recovery_start": int(branch["recovery_start"]),
        "recovery_segments": int(branch["recovery_segments"]),
        "recovery_horizon_segments": int(branch["recovery_horizon_segments"]),
        "branch_control_count": int(branch["branch_control_count"]),
        "nominal_dt": float(branch["nominal_dt"]),
        "delayed_target_time": float(branch["delayed_target_time"]),
        "branch_total_duration": float(branch["branch_total_duration"]),
        "terminal_error": float(branch["terminal_error"]),
        "branch_fuel": float(branch["branch_fuel"]),
        "nfev": int(branch["nfev"]),
        "runtime_seconds": float(branch["runtime_seconds"]),
        "optimizer_ran": bool(branch["optimizer_ran"]),
        "optimizer_success": bool(branch["optimizer_success"]),
        "accepted_candidate": str(branch["accepted_candidate"]),
        "message": str(branch["message"]),
        "accepted_branch_weight_variant_label": str(branch.get("accepted_branch_weight_variant_label", "configured")),
        "accepted_branch_weight_variant_index": int(branch.get("accepted_branch_weight_variant_index", 0)),
        "accepted_branch_weights": dict(branch.get("accepted_branch_weights", branch.get("branch_weights", {}))),
        "accepted_variant_nfev": int(branch.get("accepted_variant_nfev", branch["nfev"])),
        "accepted_variant_runtime_seconds": float(
            branch.get("accepted_variant_runtime_seconds", branch["runtime_seconds"])
        ),
        "branch_portfolio_enabled": bool(branch.get("branch_portfolio_enabled", False)),
        "branch_portfolio_variant_count": int(branch.get("branch_portfolio_variant_count", 1)),
        "portfolio_acceptance_rule": str(branch.get("portfolio_acceptance_rule", "selected lowest terminal_error with fuel as tie-breaker")),
        "portfolio_robust_threshold": float(branch.get("portfolio_robust_threshold", float("nan"))),
        "portfolio_converged_threshold_feasible_candidate_count": int(
            branch.get("portfolio_converged_threshold_feasible_candidate_count", 0)
        ),
    }
    if branch.get("branch_portfolio_candidate_results") is not None:
        summary["branch_portfolio_candidate_results"] = list(branch["branch_portfolio_candidate_results"])
    return summary


def run_delayed_locked_recovery_baseline(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray,
    thresholds: dict,
    recovery_horizon_segments: int,
    selected_outages=1,
    nominal_max_nfev: int = 140,
    branch_max_nfev: int = 120,
    min_recovery_segments: int = 0,
    nominal_residual_weights: dict | None = None,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    branch_weight_variants: list[dict] | tuple[dict, ...] | None = None,
    node_initialization: str | None = "linear",
    node_initialization_blend: float | None = 0.5,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    horizon = normalize_recovery_horizon_segments(recovery_horizon_segments)
    if not isinstance(branch_weights, BranchRecoveryWeights):
        branch_weights = BranchRecoveryWeights.from_config(branch_weights)
    branch_variants = normalize_branch_weight_variants(
        branch_weights=branch_weights,
        branch_weight_variants=branch_weight_variants,
    )
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
        warm_start_info={"stage": "delayed_locked_nominal_solve"},
    )
    nominal_controls = project_controls_to_ball(np.asarray(nominal_result["controls"], dtype=float), cfg.amax)
    nominal_error = terminal_error_for_duration(state0, target, cfg, nominal_controls, cfg.tf)
    nominal_reported_error = float(nominal_result.get("nominal_error", nominal_error))
    nominal_lock_error_delta = abs(nominal_error - nominal_reported_error)
    dt = nominal_segment_duration(cfg)
    target_delay = delayed_target_time(cfg, horizon)
    total_duration = branch_total_duration(cfg, horizon)
    delayed_target = delayed_target_state(target, cfg, horizon)
    nominal_delayed_controls = np.vstack(
        [
            nominal_controls,
            np.zeros((horizon, 3), dtype=float),
        ]
    )
    nominal_delayed_coast_error = terminal_error_for_duration(
        state0,
        delayed_target,
        cfg,
        nominal_delayed_controls,
        total_duration,
    )

    selected, delayed_mask_errors, selection_semantics = select_delayed_outage_indices(
        policy=selected_outages,
        state0=state0,
        delayed_target=delayed_target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        recovery_horizon_segments=horizon,
        min_recovery_segments=int(min_recovery_segments),
    )

    nominal_threshold = float(thresholds["nominal_success"])
    selected_threshold = float(thresholds["robust_success"])
    branches: list[dict] = []
    for index in selected.astype(int).tolist():
        branches.append(
            optimize_delayed_locked_recovery_branch_portfolio(
                state0=state0,
                target=target,
                cfg=cfg,
                nominal_controls=nominal_controls,
                mask=masks[int(index)],
                mask_index=int(index),
                recovery_horizon_segments=horizon,
                max_nfev=int(branch_max_nfev),
                robust_threshold=selected_threshold,
                branch_weights=branch_weights,
                branch_weight_variants=branch_weight_variants,
                xtol=float(xtol),
                ftol=float(ftol),
                gtol=float(gtol),
            )
        )

    selected_errors = [float(branch["terminal_error"]) for branch in branches]
    selected_worst = float(np.max(selected_errors)) if selected_errors else float(nominal_delayed_coast_error)
    all_errors = delayed_mask_errors.astype(float).tolist()
    for branch in branches:
        all_errors[int(branch["mask_index"])] = float(branch["terminal_error"])
    all_worst = float(np.max(all_errors)) if all_errors else selected_worst
    branch_controls = [np.asarray(branch["branch_controls"], dtype=float) for branch in branches]
    branch_fuels = [float(branch["branch_fuel"]) for branch in branches]
    nominal_fuel = control_fuel(nominal_controls, cfg.tf)
    diagnostics = control_norm_diagnostics([nominal_controls, nominal_delayed_controls, *branch_controls], cfg.amax)
    meets_nominal = bool(nominal_error <= nominal_threshold)
    meets_selected = bool(selected_worst <= selected_threshold)
    backend_success = bool(meets_nominal and meets_selected)
    branch_optimizer_ran_by_branch = [bool(branch.get("optimizer_ran", False)) for branch in branches]
    branch_optimizer_success_by_branch = [bool(branch["optimizer_success"]) for branch in branches]
    branch_optimizer_ran = bool(any(branch_optimizer_ran_by_branch))
    branch_optimizer_all_success = bool(
        branches
        and all(ran and success for ran, success in zip(branch_optimizer_ran_by_branch, branch_optimizer_success_by_branch))
    )
    branch_portfolio_enabled = bool(len(branch_variants) > 1)
    branch_portfolio_all_success = branch_optimizer_all_success
    portfolio_acceptance_rule = (
        str(branches[0].get("portfolio_acceptance_rule"))
        if branches
        else "no selected branch portfolio candidates were evaluated"
    )
    nominal_optimizer_success = bool(nominal_result.get("optimizer_success", False))
    overall_optimizer_success = bool(nominal_optimizer_success and branch_optimizer_all_success)
    total_branch_nfev = int(sum(int(branch["nfev"]) for branch in branches))
    total_branch_runtime = float(sum(float(branch["runtime_seconds"]) for branch in branches))
    nominal_nfev = int(nominal_result.get("nfev", 0) or 0)
    nominal_runtime = float(nominal_result.get("runtime_seconds", 0.0) or 0.0)

    branch_summaries = [delayed_branch_summary_for_json(branch) for branch in branches]
    return {
        "success": backend_success,
        "backend_success": backend_success,
        "mode": MODE,
        "method_type": MODE,
        "optimizer_success": overall_optimizer_success,
        "optimizer_success_semantics": (
            "optimizer_success is nominal_optimizer_success AND branch_optimizer_all_success; it is false "
            "for nominal-only provenance rows or skipped branch optimizers because no branch optimizer converged"
        ),
        "nominal_optimizer_success": nominal_optimizer_success,
        "nominal_backend_success": bool(nominal_result.get("success", False)),
        "branch_optimizer_success": branch_optimizer_all_success,
        "branch_optimizer_all_success": branch_optimizer_all_success,
        "branch_optimizer_ran": branch_optimizer_ran,
        "branch_portfolio_enabled": branch_portfolio_enabled,
        "branch_portfolio_variant_count": int(len(branch_variants)),
        "branch_portfolio_variant_labels": [str(variant["label"]) for variant in branch_variants],
        "branch_portfolio_all_success": branch_portfolio_all_success,
        "portfolio_acceptance_rule": portfolio_acceptance_rule,
        "branch_portfolio_converged_threshold_feasible_candidate_counts": [
            int(branch.get("portfolio_converged_threshold_feasible_candidate_count", 0))
            for branch in branches
        ],
        "message": "delayed-arrival locked nominal branch recovery complete",
        "nominal_message": str(nominal_result.get("message", "")),
        "nominal_accepted_candidate": str(nominal_result.get("accepted_candidate", "optimizer")),
        "nominal_error": float(nominal_error),
        "nominal_original_error": float(nominal_error),
        "nominal_baseline_error": nominal_reported_error,
        "nominal_lock_error_delta": float(nominal_lock_error_delta),
        "nominal_delayed_coast_error": float(nominal_delayed_coast_error),
        "recovery_horizon_segments": int(horizon),
        "nominal_dt": float(dt),
        "delayed_target_time": float(target_delay),
        "branch_total_duration": float(total_duration),
        "branch_control_count": int(cfg.n_segments + horizon),
        "original_target_state": np.asarray(target, dtype=float),
        "delayed_target_state": delayed_target,
        "worst_error_semantics": (
            "selected_worst_error is the worst delayed-target terminal error among selected "
            "independently optimized delayed-arrival locked-nominal branches"
        ),
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
        "recovery_fuel_mean": float(np.mean(branch_fuels)) if branch_fuels else float(control_fuel(nominal_delayed_controls, total_duration)),
        "recovery_fuel_max": float(np.max(branch_fuels)) if branch_fuels else float(control_fuel(nominal_delayed_controls, total_duration)),
        "control_max_norm": float(diagnostics["control_max_norm"]),
        "control_bound_violation": float(diagnostics["control_bound_violation"]),
        "controls": nominal_controls,
        "nominal_controls": nominal_controls,
        "selected_outage_indices": selected.astype(int).tolist(),
        "selected_outage_errors": selected_errors,
        "all_outage_errors": all_errors,
        "nominal_masked_outage_errors": delayed_mask_errors.astype(float).tolist(),
        "branch_results": branch_summaries,
        "branch_nfev": [int(branch["nfev"]) for branch in branches],
        "branch_runtime_seconds": [float(branch["runtime_seconds"]) for branch in branches],
        "branch_optimizer_success_by_branch": branch_optimizer_success_by_branch,
        "branch_optimizer_ran_by_branch": branch_optimizer_ran_by_branch,
        "branch_accepted_weight_variant_labels": [
            str(branch.get("accepted_branch_weight_variant_label", "configured"))
            for branch in branches
        ],
        "branch_accepted_weight_variant_indices": [
            int(branch.get("accepted_branch_weight_variant_index", 0))
            for branch in branches
        ],
        "branch_accepted_weights": [
            dict(branch.get("accepted_branch_weights", branch.get("branch_weights", {})))
            for branch in branches
        ],
        "branch_accepted_variant_nfev": [
            int(branch.get("accepted_variant_nfev", branch["nfev"]))
            for branch in branches
        ],
        "branch_accepted_variant_runtime_seconds": [
            float(branch.get("accepted_variant_runtime_seconds", branch["runtime_seconds"]))
            for branch in branches
        ],
        "branch_recovery_starts": [int(branch["recovery_start"]) for branch in branches],
        "branch_recovery_segments": [int(branch["recovery_segments"]) for branch in branches],
        "branch_control_counts": [int(branch["branch_control_count"]) for branch in branches],
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
        "branch_weight_variants": branch_weight_variants_for_json(branch_variants),
        "selection_semantics": selection_semantics,
        "backend_semantics": (
            "Delayed-arrival locked-nominal continuous baseline: nominal all-windows controls are solved once "
            "to the original target at T and kept fixed; each selected missed-thrust branch then optimizes "
            "post-outage controls over the original segments plus the configured recovery horizon. If a branch "
            "weight portfolio is configured, every variant is optimized for every selected mask and all variant "
            "function evaluations are charged."
        ),
        "selected_branch_semantics": (
            "For selected masks, controls before outage_end(mask) are locked nominal controls with missed "
            "segments zeroed. Controls from outage_end(mask) through the original end plus "
            "recovery_horizon_segments additional segments are optimization variables."
        ),
        "all_mask_diagnostic_semantics": (
            "all_outage_errors evaluates every configured outage mask against the delayed target. Selected "
            "masks use optimized delayed branches; unselected masks use locked nominal masked controls and "
            "zero acceleration during the added recovery horizon."
        ),
        "control_bound_semantics": (
            "Every nominal, delayed, and recovery acceleration vector is projected to the Euclidean ball "
            "||u_i|| <= amax before propagation, residual evaluation, fuel computation, and reporting."
        ),
        "nominal_lock_semantics": (
            "Branch optimization never changes nominal_controls. nominal_error is measured at the original "
            "transfer time against the original target; delayed branch errors are measured at branch_total_duration "
            "against delayed_target_state."
        ),
        "delayed_target_semantics": (
            "delayed_target_state is the original target propagated ballistically for "
            "recovery_horizon_segments * nominal_dt; branch_total_duration is "
            "(N + recovery_horizon_segments) * nominal_dt."
        ),
    }
