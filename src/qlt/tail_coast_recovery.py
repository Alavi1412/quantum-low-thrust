from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .cr3bp import propagate_controls_batch
from .delayed_recovery import (
    branch_weight_variants_for_json,
    normalize_branch_weight_variants,
)
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


MODE = "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery"
NOMINAL_BRANCH_INITIALIZATION_LABEL = "nominal_post_outage"
NO_RECOVERY_VARIABLES_LABEL = "no_recovery_variables"


def nominal_segment_duration(cfg: ObjectiveConfig) -> float:
    if int(cfg.n_segments) <= 0:
        raise ValueError("cfg.n_segments must be positive")
    return float(cfg.tf) / float(cfg.n_segments)


def normalize_tail_coast_segments(value: int | np.integer, cfg: ObjectiveConfig) -> int:
    tail = int(value)
    if tail < 0:
        raise ValueError("tail_coast_segments must be non-negative")
    if tail > int(cfg.n_segments):
        raise ValueError("tail_coast_segments cannot exceed cfg.n_segments")
    return tail


def normalize_tail_coast_branch_initialization_fallbacks(
    fallback_initializations: list[dict] | tuple[dict, ...] | None,
) -> list[dict]:
    if not fallback_initializations:
        return []

    variants: list[dict] = []
    for index, raw in enumerate(fallback_initializations):
        entry = dict(raw or {})
        kind = str(entry.get("kind", entry.get("type", "constant_vector")))
        normalized_kind = kind.lower().replace("-", "_")
        if normalized_kind in {"constant", "constant_controls"}:
            normalized_kind = "constant_vector"
        if normalized_kind != "constant_vector":
            raise ValueError(f"unsupported tail-coast branch initialization fallback kind: {kind}")
        if "vector" not in entry:
            raise ValueError("tail-coast branch initialization fallback requires a vector")
        vector = np.asarray(entry["vector"], dtype=float).reshape(-1)
        if vector.shape != (3,):
            raise ValueError(f"tail-coast branch initialization fallback vector has shape {vector.shape}, expected (3,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("tail-coast branch initialization fallback vector must contain only finite values")
        variants.append(
            {
                "label": str(entry.get("label", entry.get("name", f"constant_vector_{index}"))),
                "index": int(index + 1),
                "fallback_index": int(index),
                "kind": "constant_vector",
                "vector": vector.astype(float),
            }
        )
    return variants


def tail_coast_branch_initialization_fallbacks_for_json(variants: list[dict]) -> list[dict]:
    return [
        {
            "label": str(variant["label"]),
            "index": int(variant["index"]),
            "fallback_index": int(variant["fallback_index"]),
            "kind": str(variant["kind"]),
            "vector": np.asarray(variant["vector"], dtype=float).tolist(),
        }
        for variant in variants
    ]


def _nominal_branch_initialization_variant() -> dict:
    return {
        "label": NOMINAL_BRANCH_INITIALIZATION_LABEL,
        "index": 0,
        "fallback_index": None,
        "kind": NOMINAL_BRANCH_INITIALIZATION_LABEL,
        "is_fallback": False,
        "vector": None,
    }


def _fallback_branch_initialization_variant(raw: dict) -> dict:
    variant = dict(raw)
    variant["is_fallback"] = True
    return variant


def _constant_recovery_initial_controls(initialization: dict, recovery_segments: int, amax: float) -> np.ndarray:
    if int(recovery_segments) <= 0:
        return np.zeros((0, 3), dtype=float)
    vector = np.asarray(initialization["vector"], dtype=float).reshape((1, 3))
    controls = np.repeat(vector, int(recovery_segments), axis=0)
    return project_controls_to_ball(controls, amax)


def tail_coast_full_controls(prefix_controls: np.ndarray, tail_coast_segments: int, cfg: ObjectiveConfig) -> np.ndarray:
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    prefix_count = int(cfg.n_segments) - tail
    prefix = np.asarray(prefix_controls, dtype=float).reshape((prefix_count, 3))
    prefix = project_controls_to_ball(prefix, cfg.amax)
    if tail <= 0:
        return prefix
    return np.vstack([prefix, np.zeros((tail, 3), dtype=float)])


def propagate_tail_controls(
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
    final, _ = propagate_tail_controls(state0, controls, cfg)
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def optimize_tail_coast_nominal(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    seed_controls: np.ndarray,
    tail_coast_segments: int,
    max_nfev: int,
    weights: BranchRecoveryWeights | dict | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    if not isinstance(weights, BranchRecoveryWeights):
        weights = BranchRecoveryWeights.from_config(weights)
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    prefix_count = int(cfg.n_segments) - tail
    seed = project_controls_to_ball(np.asarray(seed_controls, dtype=float), cfg.amax)
    if seed.shape != (int(cfg.n_segments), 3):
        raise ValueError(f"seed_controls has shape {seed.shape}, expected {(int(cfg.n_segments), 3)}")
    scale = max(float(cfg.amax), 1e-12)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return tail_coast_full_controls(np.asarray(vec, dtype=float).reshape((prefix_count, 3)), tail, cfg)

    def residual(vec: np.ndarray) -> np.ndarray:
        controls = unpack(vec)
        final, _ = propagate_tail_controls(state0, controls, cfg)
        chunks = [weights.terminal * scaled_state_residual(final, target, cfg)]
        prefix = controls[:prefix_count]
        if weights.control and prefix.size:
            chunks.append(weights.control * prefix.reshape(-1) / scale)
        if weights.smooth and prefix.shape[0] > 1:
            chunks.append(weights.smooth * np.diff(prefix, axis=0).reshape(-1) / scale)
        return np.concatenate(chunks)

    x0 = seed[:prefix_count].reshape(-1)
    initial_controls = unpack(x0)
    initial_residual = residual(x0)
    initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
    initial_error = terminal_error(state0, target, cfg, initial_controls)
    initial_eval = {
        "accepted_candidate": "initial",
        "controls": initial_controls,
        "nominal_error": initial_error,
        "error": initial_error,
        "cost": initial_cost,
        "accepted_cost": initial_cost,
        "optimality": float("nan"),
        "accepted_optimality": float("nan"),
        "optimizer_ran": False,
        "optimizer_success": False,
        "message": "tail-coast nominal optimization skipped because max_nfev <= 0",
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
    optimizer_eval = {
        "accepted_candidate": "optimizer",
        "controls": optimizer_controls,
        "nominal_error": terminal_error(state0, target, cfg, optimizer_controls),
        "error": terminal_error(state0, target, cfg, optimizer_controls),
        "cost": float(result.cost),
        "accepted_cost": float(result.cost),
        "optimality": float(result.optimality),
        "accepted_optimality": float(result.optimality),
        "optimizer_ran": True,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
    }
    accepted = min([initial_eval, optimizer_eval], key=lambda item: (float(item["nominal_error"]), float(control_fuel(item["controls"], cfg.tf))))
    accepted = dict(accepted)
    if accepted["accepted_candidate"] == "initial":
        accepted.update(
            {
                "optimizer_ran": True,
                "optimizer_success": bool(result.success),
                "message": f"accepted initial tail-coast controls; optimizer message: {result.message}",
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted


def tail_coast_branch_full_controls(
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    recovery_controls: np.ndarray,
    amax: float,
) -> np.ndarray:
    nominal = project_controls_to_ball(np.asarray(nominal_controls, dtype=float), amax)
    mask = np.asarray(mask, dtype=float)
    if mask.shape != (nominal.shape[0],):
        raise ValueError(f"mask has shape {mask.shape}, expected {(nominal.shape[0],)}")
    start = outage_end(mask)
    controls = nominal.copy()
    if start > 0:
        controls[:start] *= mask[:start, None]
    if start < controls.shape[0]:
        recovery = np.asarray(recovery_controls, dtype=float).reshape((controls.shape[0] - start, 3))
        controls[start:] = recovery
    return project_controls_to_ball(controls, amax)


def tail_coast_masked_nominal_controls(nominal_controls: np.ndarray, mask: np.ndarray, amax: float) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    return project_controls_to_ball(controls, amax)


def tail_coast_outage_hardness_errors(
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


def select_tail_coast_outage_indices(
    *,
    policy,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    parsed = normalize_selected_outage_policy(policy)
    masks = np.asarray(masks, dtype=float)
    hardness = tail_coast_outage_hardness_errors(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
    )
    if parsed.kind == "all_configured":
        return (
            np.arange(masks.shape[0], dtype=int),
            hardness,
            "all configured outage masks are selected for fixed-final-time tail-coast locked-nominal recovery",
        )
    if parsed.kind == "all_single":
        return (
            np.flatnonzero(np.sum(masks < 0.5, axis=1) == 1).astype(int),
            hardness,
            "all one-segment outage masks are selected for fixed-final-time tail-coast locked-nominal recovery",
        )
    count = min(int(parsed.count or 0), int(masks.shape[0]))
    if count <= 0:
        return np.zeros(0, dtype=int), hardness, "selected_outages=0; no fixed-final-time branch optimizer is run"
    ranked = np.argsort(hardness)[::-1]
    return (
        ranked[:count].astype(int),
        hardness,
        f"the {count} hardest outage masks are selected by fixed-final-time terminal error under tail-coast locked nominal controls",
    )


def _tail_zero_notes(nominal_controls: np.ndarray, mask: np.ndarray, tail_coast_segments: int) -> dict:
    nominal = np.asarray(nominal_controls, dtype=float)
    mask = np.asarray(mask, dtype=float)
    tail = int(tail_coast_segments)
    tail_start = max(0, nominal.shape[0] - tail)
    missed = np.flatnonzero(mask < 0.5)
    removes_zero = bool(missed.size and np.all(np.linalg.norm(nominal[missed], axis=1) <= 1e-14))
    missed_tail = [int(index) for index in missed if int(index) >= tail_start]
    return {
        "branch_controls_remove_zero_nominal": removes_zero,
        "branch_missed_tail_indices": missed_tail,
        "branch_note": (
            "missed outage is entirely in exact-zero tail controls"
            if removes_zero and len(missed_tail) == int(missed.size)
            else "missed outage includes nonzero optimized-prefix nominal controls"
        ),
    }


def optimize_tail_coast_recovery_branch(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    tail_coast_segments: int,
    max_nfev: int,
    robust_threshold: float,
    weights: BranchRecoveryWeights | dict | None = None,
    initial_recovery_controls: np.ndarray | None = None,
    initialization_label: str = NOMINAL_BRANCH_INITIALIZATION_LABEL,
    initialization_index: int = 0,
    initialization_is_fallback: bool = False,
    initialization_kind: str = NOMINAL_BRANCH_INITIALIZATION_LABEL,
    initialization_vector: np.ndarray | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    if not isinstance(weights, BranchRecoveryWeights):
        weights = BranchRecoveryWeights.from_config(weights)
    nominal_locked = project_controls_to_ball(np.asarray(nominal_controls, dtype=float).copy(), cfg.amax)
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    if tail:
        nominal_locked[-tail:] = 0.0
    mask = np.asarray(mask, dtype=float)
    start = outage_end(mask)
    recovery_segments = int(cfg.n_segments) - start
    scale = max(float(cfg.amax), 1e-12)
    notes = _tail_zero_notes(nominal_locked, mask, tail)

    def evaluate(recovery: np.ndarray, label: str, optimizer_result=None) -> dict:
        full = tail_coast_branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        final, history = propagate_tail_controls(state0, full, cfg)
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
            "branch_initialization_index": int(initialization_index),
            "branch_initialization_is_fallback": bool(initialization_is_fallback),
            "branch_initialization_kind": str(initialization_kind),
            "terminal_error": error,
            "error": error,
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target at the original fixed final time",
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": project_controls_to_ball(recovery, cfg.amax),
            "history": history,
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
        error = terminal_error(state0, target, cfg, full)
        diagnostics = control_norm_diagnostics([full], cfg.amax)
        threshold_feasible = bool(error <= float(robust_threshold))
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
            "accepted_candidate": "no_recovery_variables",
            "branch_initialization_label": NO_RECOVERY_VARIABLES_LABEL,
            "branch_initialization_index": 0,
            "branch_initialization_is_fallback": False,
            "branch_initialization_kind": NO_RECOVERY_VARIABLES_LABEL,
            "optimizer_ran": False,
            "optimizer_success": False,
            "no_recovery_variable_threshold_feasible": threshold_feasible,
            "message": "no post-outage fixed-final-time recovery controls are available; threshold feasibility is evaluated directly",
            "cost": 0.0,
            "optimality": float("nan"),
            "nfev": 0,
            "runtime_seconds": float(time.perf_counter() - start_time),
            "terminal_error": error,
            "error": error,
            "branch_weights": weights.as_dict(),
            "target_error_semantics": "terminal_error is measured against the original target at the original fixed final time",
            "branch_fuel": control_fuel(full, cfg.tf),
            "branch_controls": full,
            "recovery_controls": np.zeros((0, 3), dtype=float),
            "history": None,
            **notes,
            **diagnostics,
        }

    if initial_recovery_controls is None:
        initial_controls = nominal_locked[start:]
    else:
        initial_controls = np.asarray(initial_recovery_controls, dtype=float).reshape((recovery_segments, 3))
    initial_controls = project_controls_to_ball(initial_controls, cfg.amax)
    x0 = initial_controls.reshape(-1)

    def unpack(vec: np.ndarray) -> np.ndarray:
        return project_controls_to_ball(np.asarray(vec, dtype=float).reshape((recovery_segments, 3)), cfg.amax)

    def residual(vec: np.ndarray) -> np.ndarray:
        recovery = unpack(vec)
        full = tail_coast_branch_full_controls(nominal_locked, mask, recovery, cfg.amax)
        final, _ = propagate_tail_controls(state0, full, cfg)
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
    if initialization_vector is not None:
        initial_eval["branch_initialization_vector"] = np.asarray(initialization_vector, dtype=float).tolist()
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
    accepted = min([initial_eval, optimizer_eval], key=lambda item: (float(item["terminal_error"]), float(item["branch_fuel"])))
    accepted = dict(accepted)
    if accepted["accepted_candidate"] == "initial":
        accepted.update(
            {
                "optimizer_ran": True,
                "optimizer_success": bool(result.success),
                "message": (
                    f"accepted initial recovery controls from {initialization_label}; "
                    f"optimizer message: {result.message}"
                ),
                "optimizer_cost": float(result.cost),
                "optimizer_optimality": float(result.optimality),
                "nfev": int(result.nfev),
            }
        )
    if initialization_vector is not None:
        accepted["branch_initialization_vector"] = np.asarray(initialization_vector, dtype=float).tolist()
    accepted["runtime_seconds"] = float(time.perf_counter() - start_time)
    return accepted


def tail_coast_branch_candidate_summary_for_json(branch: dict) -> dict:
    summary = {
        "variant_label": str(branch["branch_weight_variant_label"]),
        "variant_index": int(branch["branch_weight_variant_index"]),
        "weights": dict(branch["branch_weights"]),
        "initialization_label": str(branch.get("branch_initialization_label", NOMINAL_BRANCH_INITIALIZATION_LABEL)),
        "initialization_index": int(branch.get("branch_initialization_index", 0)),
        "initialization_is_fallback": bool(branch.get("branch_initialization_is_fallback", False)),
        "initialization_kind": str(branch.get("branch_initialization_kind", NOMINAL_BRANCH_INITIALIZATION_LABEL)),
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
    if branch.get("branch_initialization_vector") is not None:
        summary["initialization_vector"] = list(branch["branch_initialization_vector"])
    return summary


def optimize_tail_coast_recovery_branch_portfolio(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    nominal_controls: np.ndarray,
    mask: np.ndarray,
    mask_index: int,
    tail_coast_segments: int,
    max_nfev: int,
    robust_threshold: float,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    branch_weight_variants: list[dict] | tuple[dict, ...] | None = None,
    branch_initialization_fallbacks: list[dict] | tuple[dict, ...] | None = None,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    variants = normalize_branch_weight_variants(
        branch_weights=branch_weights,
        branch_weight_variants=branch_weight_variants,
    )
    fallback_initializations = normalize_tail_coast_branch_initialization_fallbacks(branch_initialization_fallbacks)
    fallback_initializations_json = tail_coast_branch_initialization_fallbacks_for_json(fallback_initializations)
    candidate_results: list[dict] = []
    nominal_initialization = _nominal_branch_initialization_variant()
    fallback_possible = int(cfg.n_segments) - int(outage_end(mask)) > 0

    def evaluate_candidate(variant: dict, initialization: dict) -> dict:
        initial_controls = None
        initialization_vector = initialization.get("vector")
        if bool(initialization.get("is_fallback", False)):
            initial_controls = _constant_recovery_initial_controls(initialization, int(cfg.n_segments) - int(outage_end(mask)), cfg.amax)
        candidate = optimize_tail_coast_recovery_branch(
            state0=state0,
            target=target,
            cfg=cfg,
            nominal_controls=nominal_controls,
            mask=mask,
            mask_index=mask_index,
            tail_coast_segments=tail_coast_segments,
            max_nfev=max_nfev,
            robust_threshold=robust_threshold,
            weights=variant["weights"],
            initial_recovery_controls=initial_controls,
            initialization_label=str(initialization["label"]),
            initialization_index=int(initialization["index"]),
            initialization_is_fallback=bool(initialization.get("is_fallback", False)),
            initialization_kind=str(initialization["kind"]),
            initialization_vector=initialization_vector,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
        )
        candidate["branch_weight_variant_label"] = str(variant["label"])
        candidate["branch_weight_variant_index"] = int(variant["index"])
        candidate["branch_weights"] = variant["weights"].as_dict()
        if str(candidate.get("accepted_candidate")) == NO_RECOVERY_VARIABLES_LABEL or int(candidate.get("recovery_segments", 0)) <= 0:
            candidate["branch_initialization_label"] = NO_RECOVERY_VARIABLES_LABEL
            candidate["branch_initialization_index"] = 0
            candidate["branch_initialization_is_fallback"] = False
            candidate["branch_initialization_kind"] = NO_RECOVERY_VARIABLES_LABEL
        else:
            candidate["branch_initialization_label"] = str(initialization["label"])
            candidate["branch_initialization_index"] = int(initialization["index"])
            candidate["branch_initialization_is_fallback"] = bool(initialization.get("is_fallback", False))
            candidate["branch_initialization_kind"] = str(initialization["kind"])
        if initialization_vector is not None and candidate["branch_initialization_kind"] != NO_RECOVERY_VARIABLES_LABEL:
            candidate["branch_initialization_vector"] = np.asarray(initialization_vector, dtype=float).tolist()
        candidate["portfolio_robust_threshold"] = float(robust_threshold)
        candidate["portfolio_threshold_feasible"] = bool(float(candidate["terminal_error"]) <= float(robust_threshold))
        candidate["portfolio_converged_threshold_feasible"] = bool(
            candidate.get("optimizer_ran", False)
            and candidate["optimizer_success"]
            and candidate["portfolio_threshold_feasible"]
        )
        return candidate

    for variant in variants:
        candidate_results.append(evaluate_candidate(variant, nominal_initialization))

    nominal_converged_feasible = [
        candidate
        for candidate in candidate_results
        if bool(candidate["portfolio_converged_threshold_feasible"])
    ]
    evaluated_fallback_count = 0
    if not nominal_converged_feasible and fallback_initializations and fallback_possible:
        for initialization in fallback_initializations:
            fallback_initialization = _fallback_branch_initialization_variant(initialization)
            evaluated_fallback_count += 1
            for variant in variants:
                candidate_results.append(evaluate_candidate(variant, fallback_initialization))

    converged_feasible = [candidate for candidate in candidate_results if bool(candidate["portfolio_converged_threshold_feasible"])]
    acceptance_pool = converged_feasible if converged_feasible else candidate_results
    if nominal_converged_feasible:
        portfolio_acceptance_rule = (
            "preferred optimizer_success=True nominal-start candidates with terminal_error <= robust_success; "
            "selected lowest terminal_error with fuel as tie-breaker; fallback initializations were not evaluated"
        )
    elif converged_feasible:
        portfolio_acceptance_rule = (
            "nominal-start portfolio had no optimizer-converged threshold-feasible candidate; evaluated configured "
            "fallback initialization variants across all branch weight variants; preferred optimizer_success=True "
            "candidates with terminal_error <= robust_success; selected lowest terminal_error with fuel as tie-breaker"
        )
    elif fallback_initializations and not fallback_possible:
        portfolio_acceptance_rule = (
            "no optimizer-converged threshold-feasible candidate was available and no post-outage recovery controls "
            "exist, so fallback initialization variants were not evaluated; selected lowest terminal_error with fuel "
            "as tie-breaker"
        )
    elif evaluated_fallback_count:
        portfolio_acceptance_rule = (
            "no optimizer-converged threshold-feasible candidate was available after nominal and configured fallback "
            "initialization variants; selected lowest terminal_error with fuel as tie-breaker"
        )
    else:
        portfolio_acceptance_rule = (
            "no optimizer-converged threshold-feasible candidate was available; selected lowest terminal_error with fuel as tie-breaker"
        )
    accepted = min(acceptance_pool, key=lambda item: (float(item["terminal_error"]), float(item["branch_fuel"])))
    accepted = dict(accepted)
    accepted_variant_nfev = int(accepted["nfev"])
    accepted_variant_runtime = float(accepted["runtime_seconds"])
    total_nfev = int(sum(int(candidate["nfev"]) for candidate in candidate_results))
    total_runtime = float(sum(float(candidate["runtime_seconds"]) for candidate in candidate_results))
    fallback_candidate_count = int(sum(bool(candidate["branch_initialization_is_fallback"]) for candidate in candidate_results))
    nominal_candidate_count = int(len(candidate_results) - fallback_candidate_count)
    candidate_optimizer_success_count = int(sum(bool(candidate["optimizer_success"]) for candidate in candidate_results))
    candidate_all_optimizer_success = bool(candidate_results and candidate_optimizer_success_count == len(candidate_results))
    accepted["accepted_branch_weight_variant_label"] = str(accepted["branch_weight_variant_label"])
    accepted["accepted_branch_weight_variant_index"] = int(accepted["branch_weight_variant_index"])
    accepted["accepted_branch_weights"] = dict(accepted["branch_weights"])
    accepted["accepted_branch_initialization_label"] = str(accepted.get("branch_initialization_label", NOMINAL_BRANCH_INITIALIZATION_LABEL))
    accepted["accepted_branch_initialization_index"] = int(accepted.get("branch_initialization_index", 0))
    accepted["accepted_branch_initialization_is_fallback"] = bool(accepted.get("branch_initialization_is_fallback", False))
    accepted["accepted_branch_initialization_kind"] = str(accepted.get("branch_initialization_kind", NOMINAL_BRANCH_INITIALIZATION_LABEL))
    if accepted.get("branch_initialization_vector") is not None:
        accepted["accepted_branch_initialization_vector"] = list(accepted["branch_initialization_vector"])
    accepted["accepted_variant_nfev"] = accepted_variant_nfev
    accepted["accepted_variant_runtime_seconds"] = accepted_variant_runtime
    accepted["nfev"] = total_nfev
    accepted["runtime_seconds"] = total_runtime
    accepted["branch_portfolio_enabled"] = bool(len(variants) > 1)
    accepted["branch_portfolio_variant_count"] = int(len(variants))
    accepted["portfolio_acceptance_rule"] = portfolio_acceptance_rule
    accepted["portfolio_robust_threshold"] = float(robust_threshold)
    accepted["portfolio_converged_threshold_feasible_candidate_count"] = int(len(converged_feasible))
    accepted["portfolio_nominal_converged_threshold_feasible_candidate_count"] = int(len(nominal_converged_feasible))
    accepted["branch_portfolio_candidate_count"] = int(len(candidate_results))
    accepted["branch_portfolio_candidate_optimizer_success_count"] = candidate_optimizer_success_count
    accepted["branch_portfolio_candidate_all_optimizer_success"] = candidate_all_optimizer_success
    accepted["branch_weight_variants"] = branch_weight_variants_for_json(variants)
    accepted["branch_initialization_fallbacks"] = fallback_initializations_json
    accepted["branch_fallback_initialization_enabled"] = bool(fallback_initializations)
    accepted["branch_fallback_initialization_configured_count"] = int(len(fallback_initializations))
    accepted["branch_fallback_initialization_evaluated_count"] = int(evaluated_fallback_count)
    accepted["branch_fallback_initialization_candidate_count"] = fallback_candidate_count
    accepted["branch_nominal_initialization_candidate_count"] = nominal_candidate_count
    accepted["branch_initialization_variant_count"] = int(1 + len(fallback_initializations))
    accepted["branch_portfolio_candidate_results"] = [
        tail_coast_branch_candidate_summary_for_json(candidate)
        for candidate in candidate_results
    ]
    return accepted


def tail_coast_branch_summary_for_json(branch: dict) -> dict:
    summary = {
        "mask_index": int(branch["mask_index"]),
        "recovery_start": int(branch["recovery_start"]),
        "recovery_segments": int(branch["recovery_segments"]),
        "tail_coast_segments": int(branch["tail_coast_segments"]),
        "branch_control_count": int(branch["branch_control_count"]),
        "nominal_dt": float(branch["nominal_dt"]),
        "branch_total_duration": float(branch["branch_total_duration"]),
        "terminal_error": float(branch["terminal_error"]),
        "branch_fuel": float(branch["branch_fuel"]),
        "nfev": int(branch["nfev"]),
        "runtime_seconds": float(branch["runtime_seconds"]),
        "optimizer_ran": bool(branch["optimizer_ran"]),
        "optimizer_success": bool(branch["optimizer_success"]),
        "no_recovery_variable_threshold_feasible": bool(branch.get("no_recovery_variable_threshold_feasible", False)),
        "accepted_candidate": str(branch["accepted_candidate"]),
        "message": str(branch["message"]),
        "branch_controls_remove_zero_nominal": bool(branch.get("branch_controls_remove_zero_nominal", False)),
        "branch_missed_tail_indices": list(branch.get("branch_missed_tail_indices", [])),
        "branch_note": str(branch.get("branch_note", "")),
        "accepted_branch_weight_variant_label": str(branch.get("accepted_branch_weight_variant_label", "configured")),
        "accepted_branch_weight_variant_index": int(branch.get("accepted_branch_weight_variant_index", 0)),
        "accepted_branch_weights": dict(branch.get("accepted_branch_weights", branch.get("branch_weights", {}))),
        "accepted_branch_initialization_label": str(branch.get("accepted_branch_initialization_label", NOMINAL_BRANCH_INITIALIZATION_LABEL)),
        "accepted_branch_initialization_index": int(branch.get("accepted_branch_initialization_index", 0)),
        "accepted_branch_initialization_is_fallback": bool(branch.get("accepted_branch_initialization_is_fallback", False)),
        "accepted_branch_initialization_kind": str(branch.get("accepted_branch_initialization_kind", NOMINAL_BRANCH_INITIALIZATION_LABEL)),
        "accepted_variant_nfev": int(branch.get("accepted_variant_nfev", branch["nfev"])),
        "accepted_variant_runtime_seconds": float(branch.get("accepted_variant_runtime_seconds", branch["runtime_seconds"])),
        "branch_portfolio_enabled": bool(branch.get("branch_portfolio_enabled", False)),
        "branch_portfolio_variant_count": int(branch.get("branch_portfolio_variant_count", 1)),
        "portfolio_acceptance_rule": str(branch.get("portfolio_acceptance_rule", "selected lowest terminal_error with fuel as tie-breaker")),
        "portfolio_robust_threshold": float(branch.get("portfolio_robust_threshold", float("nan"))),
        "portfolio_converged_threshold_feasible_candidate_count": int(branch.get("portfolio_converged_threshold_feasible_candidate_count", 0)),
        "portfolio_nominal_converged_threshold_feasible_candidate_count": int(
            branch.get("portfolio_nominal_converged_threshold_feasible_candidate_count", 0)
        ),
        "branch_portfolio_candidate_count": int(branch.get("branch_portfolio_candidate_count", 0)),
        "branch_portfolio_candidate_optimizer_success_count": int(
            branch.get("branch_portfolio_candidate_optimizer_success_count", 0)
        ),
        "branch_portfolio_candidate_all_optimizer_success": bool(
            branch.get("branch_portfolio_candidate_all_optimizer_success", False)
        ),
        "branch_fallback_initialization_enabled": bool(branch.get("branch_fallback_initialization_enabled", False)),
        "branch_fallback_initialization_configured_count": int(branch.get("branch_fallback_initialization_configured_count", 0)),
        "branch_fallback_initialization_evaluated_count": int(branch.get("branch_fallback_initialization_evaluated_count", 0)),
        "branch_fallback_initialization_candidate_count": int(branch.get("branch_fallback_initialization_candidate_count", 0)),
        "branch_nominal_initialization_candidate_count": int(branch.get("branch_nominal_initialization_candidate_count", 1)),
        "branch_initialization_variant_count": int(branch.get("branch_initialization_variant_count", 1)),
    }
    if branch.get("accepted_branch_initialization_vector") is not None:
        summary["accepted_branch_initialization_vector"] = list(branch["accepted_branch_initialization_vector"])
    if branch.get("branch_initialization_fallbacks") is not None:
        summary["branch_initialization_fallbacks"] = list(branch["branch_initialization_fallbacks"])
    if branch.get("branch_portfolio_candidate_results") is not None:
        summary["branch_portfolio_candidate_results"] = list(branch["branch_portfolio_candidate_results"])
    return summary


def tail_coast_row_portfolio_acceptance_rule(branches: list[dict]) -> str:
    branch_count = int(len(branches))
    if branch_count <= 0:
        return "no selected branch portfolio candidates were evaluated"

    configured_count = int(
        max(
            [int(branch.get("branch_fallback_initialization_configured_count", 0)) for branch in branches],
            default=0,
        )
    )
    evaluated_by_branch = [
        bool(int(branch.get("branch_fallback_initialization_evaluated_count", 0)) > 0)
        for branch in branches
    ]
    accepted_by_branch = [
        bool(branch.get("accepted_branch_initialization_is_fallback", False))
        for branch in branches
    ]
    evaluated_count = int(sum(evaluated_by_branch))
    accepted_count = int(sum(accepted_by_branch))
    suffix = "per-branch portfolio_acceptance_rule entries in branch_results remain authoritative"
    if evaluated_count:
        if accepted_count:
            return (
                f"row summary across {branch_count} selected branches: fallback initializations were evaluated by "
                f"{evaluated_count} branch(es), and a fallback initialization was accepted by {accepted_count} "
                f"branch(es); {suffix}"
            )
        return (
            f"row summary across {branch_count} selected branches: fallback initializations were evaluated by "
            f"{evaluated_count} branch(es), but no branch accepted a fallback initialization; {suffix}"
        )
    if configured_count:
        return (
            f"row summary across {branch_count} selected branches: {configured_count} fallback initialization(s) were "
            f"configured, but no branch evaluated or accepted a fallback initialization; {suffix}"
        )
    return (
        f"row summary across {branch_count} selected branches: no fallback initializations were configured, evaluated, "
        f"or accepted; {suffix}"
    )


def run_tail_coast_recovery_baseline(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray,
    thresholds: dict,
    tail_coast_segments: int,
    selected_outages=1,
    nominal_max_nfev: int = 140,
    tail_nominal_max_nfev: int | None = None,
    branch_max_nfev: int = 120,
    nominal_residual_weights: dict | None = None,
    tail_nominal_weights: BranchRecoveryWeights | dict | None = None,
    branch_weights: BranchRecoveryWeights | dict | None = None,
    branch_weight_variants: list[dict] | tuple[dict, ...] | None = None,
    branch_initialization_fallbacks: list[dict] | tuple[dict, ...] | None = None,
    node_initialization: str | None = "linear",
    node_initialization_blend: float | None = 0.5,
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    gtol: float = 1e-5,
) -> dict:
    start_time = time.perf_counter()
    tail = normalize_tail_coast_segments(tail_coast_segments, cfg)
    if tail_nominal_max_nfev is None:
        tail_nominal_max_nfev = nominal_max_nfev
    if not isinstance(branch_weights, BranchRecoveryWeights):
        branch_weights = BranchRecoveryWeights.from_config(branch_weights)
    branch_variants = normalize_branch_weight_variants(
        branch_weights=branch_weights,
        branch_weight_variants=branch_weight_variants,
    )
    branch_fallback_initializations = normalize_tail_coast_branch_initialization_fallbacks(branch_initialization_fallbacks)
    branch_fallback_initializations_json = tail_coast_branch_initialization_fallbacks_for_json(branch_fallback_initializations)
    masks = np.asarray(masks, dtype=float)
    seed_result = run_multiple_shooting_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds=thresholds,
        selected_outages=0,
        max_nfev=int(nominal_max_nfev),
        min_recovery_segments=0,
        residual_weights=nominal_residual_weights,
        nominal_control_guess=None,
        selected_branch_control_guesses=None,
        node_initialization=node_initialization,
        node_initialization_blend=node_initialization_blend,
        warm_start_info={"stage": "tail_coast_seed_nominal_solve"},
    )
    seed_controls = project_controls_to_ball(np.asarray(seed_result["controls"], dtype=float), cfg.amax)
    nominal_seed_error = terminal_error(state0, target, cfg, seed_controls)
    tail_result = optimize_tail_coast_nominal(
        state0=state0,
        target=target,
        cfg=cfg,
        seed_controls=seed_controls,
        tail_coast_segments=tail,
        max_nfev=int(tail_nominal_max_nfev),
        weights=tail_nominal_weights,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
    )
    nominal_controls = np.asarray(tail_result["controls"], dtype=float)
    if tail:
        nominal_controls[-tail:] = 0.0
    nominal_error = terminal_error(state0, target, cfg, nominal_controls)
    selected, nominal_mask_errors, selection_semantics = select_tail_coast_outage_indices(
        policy=selected_outages,
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
    )

    nominal_threshold = float(thresholds["nominal_success"])
    selected_threshold = float(thresholds["robust_success"])
    branches: list[dict] = []
    for index in selected.astype(int).tolist():
        branches.append(
            optimize_tail_coast_recovery_branch_portfolio(
                state0=state0,
                target=target,
                cfg=cfg,
                nominal_controls=nominal_controls,
                mask=masks[int(index)],
                mask_index=int(index),
                tail_coast_segments=tail,
                max_nfev=int(branch_max_nfev),
                robust_threshold=selected_threshold,
                branch_weights=branch_weights,
                branch_weight_variants=branch_weight_variants,
                branch_initialization_fallbacks=branch_fallback_initializations,
                xtol=xtol,
                ftol=ftol,
                gtol=gtol,
            )
        )

    selected_errors = [float(branch["terminal_error"]) for branch in branches]
    selected_worst = float(np.max(selected_errors)) if selected_errors else float(nominal_error)
    all_errors = nominal_mask_errors.astype(float).tolist()
    for branch in branches:
        all_errors[int(branch["mask_index"])] = float(branch["terminal_error"])
    all_worst = float(np.max(all_errors)) if all_errors else selected_worst
    branch_controls = [np.asarray(branch["branch_controls"], dtype=float) for branch in branches]
    branch_fuels = [float(branch["branch_fuel"]) for branch in branches]
    nominal_fuel = control_fuel(nominal_controls, cfg.tf)
    diagnostics = control_norm_diagnostics([nominal_controls, *branch_controls], cfg.amax)
    tail_controls = nominal_controls[-tail:] if tail else np.zeros((0, 3), dtype=float)
    nominal_tail_zero_max_abs = float(np.max(np.abs(tail_controls))) if tail_controls.size else 0.0
    nominal_tail_control_norm_max = float(np.max(np.linalg.norm(tail_controls, axis=1))) if tail_controls.size else 0.0
    meets_nominal = bool(nominal_error <= nominal_threshold)
    meets_selected = bool(selected_worst <= selected_threshold)
    backend_success = bool(meets_nominal and meets_selected)
    branch_optimizer_ran_by_branch = [bool(branch.get("optimizer_ran", False)) for branch in branches]
    branch_optimizer_success_by_branch = [bool(branch["optimizer_success"]) for branch in branches]
    branch_optimizer_ran = bool(any(branch_optimizer_ran_by_branch))
    branch_optimizer_all_success = bool(branches and all(bool(branch["optimizer_success"]) for branch in branches))
    branch_portfolio_all_success = branch_optimizer_all_success
    branch_portfolio_enabled = bool(len(branch_variants) > 1)
    portfolio_acceptance_rule = tail_coast_row_portfolio_acceptance_rule(branches)
    branch_fallback_evaluated_counts = [
        int(branch.get("branch_fallback_initialization_evaluated_count", 0))
        for branch in branches
    ]
    branch_fallback_candidate_counts = [
        int(branch.get("branch_fallback_initialization_candidate_count", 0))
        for branch in branches
    ]
    branch_fallback_evaluated_by_branch = [bool(count > 0) for count in branch_fallback_evaluated_counts]
    branch_fallback_accepted_by_branch = [
        bool(branch.get("accepted_branch_initialization_is_fallback", False))
        for branch in branches
    ]
    branch_portfolio_candidate_all_optimizer_success_by_branch = [
        bool(branch.get("branch_portfolio_candidate_all_optimizer_success", False))
        for branch in branches
    ]
    nominal_optimizer_success = bool(seed_result.get("optimizer_success", False)) and bool(tail_result.get("optimizer_success", False))
    optimizer_success = bool(nominal_optimizer_success and branch_optimizer_all_success)
    total_branch_nfev = int(sum(int(branch["nfev"]) for branch in branches))
    total_branch_runtime = float(sum(float(branch["runtime_seconds"]) for branch in branches))
    seed_nfev = int(seed_result.get("nfev", 0) or 0)
    tail_nfev = int(tail_result.get("nfev", 0) or 0)
    seed_runtime = float(seed_result.get("runtime_seconds", 0.0) or 0.0)
    tail_runtime = float(tail_result.get("runtime_seconds", 0.0) or 0.0)
    branch_summaries = [tail_coast_branch_summary_for_json(branch) for branch in branches]
    branch_remove_zero = [bool(branch.get("branch_controls_remove_zero_nominal", False)) for branch in branches]

    return {
        "success": backend_success,
        "backend_success": backend_success,
        "mode": MODE,
        "method_type": MODE,
        "optimizer_success": optimizer_success,
        "optimizer_success_semantics": (
            "optimizer_success is seed nominal optimizer success AND tail-coast nominal optimizer success AND "
            "branch_optimizer_all_success. branch_optimizer_all_success requires at least one selected branch and "
            "optimizer_success=True for every selected branch's accepted portfolio-selected result; threshold-feasible "
            "no-recovery-variable branches support backend/meets_thresholds but do not count as optimizer convergence"
        ),
        "nominal_optimizer_success": nominal_optimizer_success,
        "nominal_seed_optimizer_success": bool(seed_result.get("optimizer_success", False)),
        "nominal_tail_optimizer_success": bool(tail_result.get("optimizer_success", False)),
        "nominal_backend_success": bool(seed_result.get("success", False)),
        "branch_optimizer_success": branch_optimizer_all_success,
        "branch_optimizer_all_success": branch_optimizer_all_success,
        "branch_optimizer_ran": branch_optimizer_ran,
        "branch_portfolio_enabled": branch_portfolio_enabled,
        "branch_portfolio_variant_count": int(len(branch_variants)),
        "branch_portfolio_variant_labels": [str(variant["label"]) for variant in branch_variants],
        "branch_portfolio_all_success": branch_portfolio_all_success,
        "branch_portfolio_all_success_semantics": (
            "branch_portfolio_all_success is retained for CSV compatibility and means every selected branch's "
            "accepted portfolio-selected result has optimizer_success=True; it does not mean every evaluated "
            "portfolio candidate, weight variant, or fallback initialization converged"
        ),
        "portfolio_acceptance_rule": portfolio_acceptance_rule,
        "branch_portfolio_converged_threshold_feasible_candidate_counts": [
            int(branch.get("portfolio_converged_threshold_feasible_candidate_count", 0))
            for branch in branches
        ],
        "branch_portfolio_candidate_counts": [
            int(branch.get("branch_portfolio_candidate_count", 0))
            for branch in branches
        ],
        "branch_portfolio_candidate_optimizer_success_counts": [
            int(branch.get("branch_portfolio_candidate_optimizer_success_count", 0))
            for branch in branches
        ],
        "branch_portfolio_candidate_all_optimizer_success": bool(
            branches and all(branch_portfolio_candidate_all_optimizer_success_by_branch)
        ),
        "branch_portfolio_candidate_all_optimizer_success_by_branch": branch_portfolio_candidate_all_optimizer_success_by_branch,
        "branch_fallback_initialization_enabled": bool(branch_fallback_initializations),
        "branch_fallback_initialization_configured_count": int(len(branch_fallback_initializations)),
        "branch_fallback_initialization_labels": [str(variant["label"]) for variant in branch_fallback_initializations],
        "branch_initialization_fallbacks": branch_fallback_initializations_json,
        "branch_fallback_initialization_evaluated_counts": branch_fallback_evaluated_counts,
        "branch_fallback_initialization_candidate_counts": branch_fallback_candidate_counts,
        "branch_fallback_initialization_any_evaluated": bool(any(branch_fallback_evaluated_by_branch)),
        "branch_fallback_initialization_any_accepted": bool(any(branch_fallback_accepted_by_branch)),
        "branch_fallback_initialization_evaluated_branch_count": int(sum(branch_fallback_evaluated_by_branch)),
        "branch_fallback_initialization_accepted_branch_count": int(sum(branch_fallback_accepted_by_branch)),
        "message": "fixed-final-time tail-coast locked nominal branch recovery complete",
        "nominal_message": str(tail_result.get("message", "")),
        "nominal_seed_message": str(seed_result.get("message", "")),
        "nominal_accepted_candidate": str(tail_result.get("accepted_candidate", "optimizer")),
        "nominal_error": float(nominal_error),
        "nominal_tail_coast_error": float(nominal_error),
        "nominal_seed_error": float(nominal_seed_error),
        "nominal_baseline_error": float(seed_result.get("nominal_error", nominal_seed_error)),
        "nominal_lock_error_delta": float(abs(nominal_error - float(tail_result.get("nominal_error", nominal_error)))),
        "tail_coast_segments": int(tail),
        "optimized_nominal_segments": int(cfg.n_segments - tail),
        "nominal_tail_zero_max_abs": nominal_tail_zero_max_abs,
        "nominal_tail_control_norm_max": nominal_tail_control_norm_max,
        "nominal_dt": float(nominal_segment_duration(cfg)),
        "branch_total_duration": float(cfg.tf),
        "branch_control_count": int(cfg.n_segments),
        "original_target_state": np.asarray(target, dtype=float),
        "worst_error_semantics": (
            "selected_worst_error is the worst original-target terminal error at the fixed final time among "
            "selected independently optimized tail-coast locked-nominal branches"
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
        "branch_optimizer_success_by_branch": branch_optimizer_success_by_branch,
        "branch_optimizer_ran_by_branch": branch_optimizer_ran_by_branch,
        "branch_accepted_weight_variant_labels": [str(branch.get("accepted_branch_weight_variant_label", "configured")) for branch in branches],
        "branch_accepted_weight_variant_indices": [int(branch.get("accepted_branch_weight_variant_index", 0)) for branch in branches],
        "branch_accepted_weights": [dict(branch.get("accepted_branch_weights", branch.get("branch_weights", {}))) for branch in branches],
        "branch_accepted_initialization_labels": [
            str(branch.get("accepted_branch_initialization_label", NOMINAL_BRANCH_INITIALIZATION_LABEL))
            for branch in branches
        ],
        "branch_accepted_initialization_indices": [
            int(branch.get("accepted_branch_initialization_index", 0))
            for branch in branches
        ],
        "branch_accepted_initialization_is_fallback": [
            bool(branch.get("accepted_branch_initialization_is_fallback", False))
            for branch in branches
        ],
        "branch_accepted_initialization_kinds": [
            str(branch.get("accepted_branch_initialization_kind", NOMINAL_BRANCH_INITIALIZATION_LABEL))
            for branch in branches
        ],
        "branch_accepted_variant_nfev": [int(branch.get("accepted_variant_nfev", branch["nfev"])) for branch in branches],
        "branch_accepted_variant_runtime_seconds": [
            float(branch.get("accepted_variant_runtime_seconds", branch["runtime_seconds"]))
            for branch in branches
        ],
        "branch_recovery_starts": [int(branch["recovery_start"]) for branch in branches],
        "branch_recovery_segments": [int(branch["recovery_segments"]) for branch in branches],
        "branch_control_counts": [int(branch["branch_control_count"]) for branch in branches],
        "branch_controls_remove_zero_nominal": branch_remove_zero,
        "total_branch_nfev": total_branch_nfev,
        "total_branch_runtime_seconds": total_branch_runtime,
        "nominal_seed_nfev": seed_nfev,
        "nominal_tail_nfev": tail_nfev,
        "nominal_nfev": int(seed_nfev + tail_nfev),
        "nominal_seed_runtime_seconds": seed_runtime,
        "nominal_tail_runtime_seconds": tail_runtime,
        "nominal_runtime_seconds": float(seed_runtime + tail_runtime),
        "nfev": int(seed_nfev + tail_nfev + total_branch_nfev),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "cost": float(float(tail_result.get("cost", 0.0)) + sum(float(branch.get("cost", 0.0)) for branch in branches)),
        "optimality": float(max([float(tail_result.get("optimality", 0.0)), *[float(branch.get("optimality", 0.0)) for branch in branches]])),
        "nominal_cost": float(tail_result.get("cost", 0.0)),
        "nominal_optimality": float(tail_result.get("optimality", 0.0)),
        "branch_weights": branch_weights.as_dict(),
        "branch_weight_variants": branch_weight_variants_for_json(branch_variants),
        "selection_semantics": selection_semantics,
        "backend_semantics": (
            "Fixed-final-time tail-coast locked-nominal continuous baseline: a nominal all-windows trajectory is "
            "solved first, then refined with the final tail_coast_segments nominal controls fixed exactly to zero. "
            "Each selected branch keeps the original target and original transfer time. If branch initialization "
            "fallbacks are configured, the nominal post-outage initialization is evaluated first across the branch "
            "weight portfolio and fallbacks are evaluated only when that nominal-start portfolio has no optimizer-"
            "converged threshold-feasible candidate. The row-level portfolio_acceptance_rule summarizes whether any "
            "selected branch evaluated or accepted fallback initializations; per-branch branch_results keep the "
            "branch-specific acceptance rules."
        ),
        "selected_branch_semantics": (
            "For selected masks, controls before outage_end(mask) are locked tail-coast nominal controls with missed "
            "segment(s) zeroed. Controls from outage_end(mask) through the original final time are optimization variables."
        ),
        "all_mask_diagnostic_semantics": (
            "all_outage_errors evaluates every configured outage mask at the original fixed final time. Selected masks "
            "use optimized branches; unselected masks use masked locked tail-coast nominal controls."
        ),
        "control_bound_semantics": (
            "Every nominal and recovery acceleration vector is projected to the Euclidean ball ||u_i|| <= amax before "
            "propagation, residual evaluation, fuel computation, and reporting. The nominal tail controls are then "
            "reported as exact zeros."
        ),
        "nominal_lock_semantics": (
            "Branch optimization never changes nominal_controls. nominal_tail_coast_error is measured at the original "
            "transfer time against the original target after enforcing the exact-zero nominal tail coast."
        ),
        "fixed_final_time_semantics": (
            "No delayed target state or added recovery horizon is used; branch_total_duration equals cfg.tf for every row."
        ),
    }
