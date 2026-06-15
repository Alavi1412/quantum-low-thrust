from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from .cr3bp import feedback_acceleration, propagate_controls_batch, propagate_feedback_batch
from .objective import ObjectiveConfig, state_error


def project_controls_to_ball(controls: np.ndarray, amax: float) -> np.ndarray:
    projected = np.asarray(controls, dtype=float).copy()
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(amax) / np.maximum(norms, 1e-12))
    return projected * scale


def multistart_control_guesses(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    refine_cfg: dict,
) -> list[tuple[str, np.ndarray]]:
    """Return deterministic bounded control guesses for direct refinement."""
    schedule = np.asarray(schedule, dtype=int)
    active = schedule.astype(float)[:, None]
    ms_cfg = dict(refine_cfg.get("multistart", {}) or {})
    enabled = bool(ms_cfg.get("enabled", False))
    guesses: list[tuple[str, np.ndarray]] = []

    if bool(ms_cfg.get("include_feedback", True)):
        guesses.append(("feedback", feedback_controls_for_schedule(schedule, state0, target, cfg)))
    if enabled and bool(ms_cfg.get("include_zero", True)):
        guesses.append(("zero", np.zeros((cfg.n_segments, 3), dtype=float)))

    if enabled and bool(ms_cfg.get("include_bang_bang", True)):
        feedback = feedback_controls_for_schedule(schedule, state0, target, cfg)
        norms = np.linalg.norm(feedback, axis=1, keepdims=True)
        unit = feedback / np.maximum(norms, 1e-12)
        guesses.append(("bang_feedback", active * cfg.amax * unit))

    if enabled:
        rng = np.random.default_rng(int(ms_cfg.get("seed", 12345)))
        count = int(ms_cfg.get("random_starts", 0))
        amplitude_fraction = float(ms_cfg.get("low_amplitude_fraction", 0.35))
        for i in range(max(0, count)):
            direction = rng.normal(size=(cfg.n_segments, 3))
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
            amplitude = rng.uniform(0.0, amplitude_fraction * cfg.amax, size=(cfg.n_segments, 1))
            guesses.append((f"random_low_{i}", active * amplitude * direction))

    guesses.extend(configured_initial_control_guesses(schedule, cfg, refine_cfg))

    bounded: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for label, controls in guesses:
        unique_label = label
        suffix = 1
        while unique_label in seen:
            suffix += 1
            unique_label = f"{label}_{suffix}"
        seen.add(unique_label)
        controls = np.asarray(controls, dtype=float)
        controls = controls * active
        bounded.append((unique_label, project_controls_to_ball(controls, cfg.amax)))
    return bounded


def configured_initial_control_guesses(
    schedule: np.ndarray,
    cfg: ObjectiveConfig,
    refine_cfg: dict,
) -> list[tuple[str, np.ndarray]]:
    """Load caller-supplied named control guesses from refinement config."""
    raw = refine_cfg.get("initial_control_guesses")
    if raw is None:
        return []
    if isinstance(raw, dict):
        entries = [{"name": key, "controls": value} for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        entries = list(raw)
    else:
        raise ValueError("refinement.initial_control_guesses must be a mapping or list")

    active = np.asarray(schedule, dtype=float)[:, None]
    guesses: list[tuple[str, np.ndarray]] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, dict):
            label = str(entry.get("name", entry.get("label", f"configured_{i}")))
            controls_raw = entry.get("controls")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            label = str(entry[0])
            controls_raw = entry[1]
        else:
            raise ValueError("each initial control guess must have a name and controls")
        if controls_raw is None:
            raise ValueError(f"initial control guess {label!r} is missing controls")
        controls = np.asarray(controls_raw, dtype=float)
        expected_shape = (cfg.n_segments, 3)
        if controls.shape != expected_shape:
            raise ValueError(f"initial control guess {label!r} has shape {controls.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(controls)):
            raise ValueError(f"initial control guess {label!r} contains non-finite values")
        guesses.append((label, project_controls_to_ball(controls * active, cfg.amax)))
    return guesses


def initial_controls(schedule: np.ndarray, state0: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    schedule = np.asarray(schedule, dtype=float)
    states = state0[None, :].copy()
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    dt = cfg.tf / cfg.n_segments
    for i, active in enumerate(schedule):
        if active > 0.5:
            controls[i] = feedback_acceleration(states, target, np.ones(1), cfg.amax, cfg.kr, cfg.kv)[0]
        segment_schedule = np.zeros(cfg.n_segments)
        segment_schedule[i] = active
        # Cheap one-segment update with the same integrator granularity.
        final, _ = propagate_controls_batch(states[0], controls[None, :, :] * 0.0, cfg.mu, 0.0, 1)
        del final, segment_schedule, dt
    return controls


def feedback_controls_for_schedule(schedule: np.ndarray, state0: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    schedule = np.asarray(schedule, dtype=float)
    finals, hist = propagate_feedback_batch(
        state0,
        schedule,
        target,
        cfg.mu,
        cfg.tf,
        cfg.amax,
        cfg.kr,
        cfg.kv,
        cfg.substeps,
        return_history=True,
    )
    del finals
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    stride = cfg.substeps
    for i, active in enumerate(schedule):
        if active > 0.5:
            state_i = hist[i * stride, 0, :][None, :]
            controls[i] = feedback_acceleration(state_i, target, np.ones(1), cfg.amax, cfg.kr, cfg.kv)[0]
    return controls


def control_fuel(controls: np.ndarray, tf: float) -> float:
    controls = np.asarray(controls, dtype=float)
    if controls.size == 0:
        return 0.0
    dt = float(tf) / float(controls.shape[0])
    return float(dt * np.sum(np.linalg.norm(controls, axis=-1)))


def selected_outage_indices_for_schedule(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    outage_masks: np.ndarray,
    selected_outages: int,
    min_recovery_segments: int = 0,
) -> np.ndarray:
    """Choose outages that are hardest under the binary feedback policy."""
    schedule = np.asarray(schedule, dtype=int)
    all_schedules = schedule[None, :] * outage_masks
    finals, _ = propagate_feedback_batch(
        state0, all_schedules, target, cfg.mu, cfg.tf, cfg.amax, cfg.kr, cfg.kv, cfg.substeps
    )
    outage_errors = state_error(finals, target, cfg.position_scale, cfg.velocity_scale)
    k = min(int(selected_outages), outage_masks.shape[0])
    eligible = np.arange(outage_masks.shape[0])
    if min_recovery_segments > 0:
        active_idx = np.flatnonzero(schedule > 0)
        eligible = np.asarray(
            [
                i
                for i, mask in enumerate(outage_masks)
                if np.count_nonzero(active_idx >= _outage_end(mask)) >= int(min_recovery_segments)
            ],
            dtype=int,
        )
        if eligible.size < k:
            eligible = np.arange(outage_masks.shape[0])
    ranked = eligible[np.argsort(outage_errors[eligible])]
    return ranked[-k:]


def _outage_end(mask: np.ndarray) -> int:
    missed = np.flatnonzero(np.asarray(mask, dtype=float) < 0.5)
    if missed.size == 0:
        return 0
    return int(missed[-1] + 1)


def _scaled_state_residual(final_state: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    diff = np.asarray(final_state, dtype=float) - np.asarray(target, dtype=float)
    scale = np.array(
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
    return diff / np.maximum(scale, 1e-12)


def _branch_controls_from_nominal(
    nominal_controls: np.ndarray,
    schedule: np.ndarray,
    mask: np.ndarray,
    recovery_idx: np.ndarray,
    recovery_values: np.ndarray,
    amax: float,
) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy()
    controls *= np.asarray(schedule, dtype=float)[:, None]
    controls *= np.asarray(mask, dtype=float)[:, None]
    if recovery_idx.size:
        controls[recovery_idx] = recovery_values.reshape((-1, 3))
    return project_controls_to_ball(controls, amax)


def _refine_single_sequence(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    outage_masks: np.ndarray,
    refine_cfg: dict,
    thresholds: dict,
) -> dict:
    start = time.perf_counter()
    schedule = np.asarray(schedule, dtype=int)
    active_idx = np.flatnonzero(schedule > 0)
    if active_idx.size == 0:
        return {"success": False, "message": "no active windows", "runtime_seconds": 0.0}

    base_controls = feedback_controls_for_schedule(schedule, state0, target, cfg)
    x0 = base_controls[active_idx].reshape(-1)

    selected = selected_outage_indices_for_schedule(
        schedule,
        state0,
        target,
        cfg,
        outage_masks,
        int(refine_cfg["selected_outages"]),
        int(refine_cfg.get("outage_selection_min_recovery_segments", 0)),
    )
    selected_masks = outage_masks[selected]

    def unpack(vec: np.ndarray) -> np.ndarray:
        controls = np.zeros((cfg.n_segments, 3), dtype=float)
        controls[active_idx] = vec.reshape((-1, 3))
        return project_controls_to_ball(controls, cfg.amax)

    def residual(vec: np.ndarray) -> np.ndarray:
        controls = unpack(vec)
        nominal_final, _ = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)
        res = [refine_cfg["state_residual_weight"] * _scaled_state_residual(nominal_final[0], target, cfg)]
        for mask in selected_masks:
            missed_controls = controls * (schedule * mask)[:, None]
            robust_final, _ = propagate_controls_batch(state0, missed_controls, cfg.mu, cfg.tf, cfg.substeps)
            res.append(refine_cfg["robust_residual_weight"] * _scaled_state_residual(robust_final[0], target, cfg))
        res.append(refine_cfg["fuel_residual_weight"] * controls[active_idx].reshape(-1) / cfg.amax)
        if active_idx.size > 1:
            res.append(refine_cfg["smooth_residual_weight"] * np.diff(controls[active_idx], axis=0).reshape(-1) / cfg.amax)
        res.append(refine_cfg["control_regularization"] * controls[active_idx].reshape(-1) / cfg.amax)
        return np.concatenate(res)

    result = least_squares(
        residual,
        x0,
        # Box bounds keep optimizer variables finite; unpack projects applied controls to ||u_i|| <= amax.
        bounds=(-cfg.amax, cfg.amax),
        max_nfev=int(refine_cfg["max_nfev"]),
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
        verbose=0,
    )
    controls = project_controls_to_ball(unpack(result.x), cfg.amax)
    nominal_final, _ = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)
    nominal_error = float(state_error(nominal_final[0], target, cfg.position_scale, cfg.velocity_scale))

    all_controls = np.repeat(controls[None, :, :], outage_masks.shape[0], axis=0)
    all_controls = all_controls * (schedule[None, :, None] * outage_masks[:, :, None])
    robust_finals, _ = propagate_controls_batch(state0, all_controls, cfg.mu, cfg.tf, cfg.substeps)
    robust_errors = state_error(robust_finals, target, cfg.position_scale, cfg.velocity_scale)
    worst_error = float(np.max(robust_errors))
    converged = bool(nominal_error <= thresholds["nominal_success"] and worst_error <= thresholds["robust_success"])
    nominal_fuel = control_fuel(controls, cfg.tf)

    return {
        "success": converged,
        "mode": "single_sequence",
        "optimizer_success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "runtime_seconds": float(time.perf_counter() - start),
        "nominal_error": nominal_error,
        "worst_error": worst_error,
        "selected_worst_error": float(np.max(robust_errors[selected])) if selected.size else worst_error,
        "all_mask_worst_error": worst_error,
        "active_fraction": float(np.mean(schedule)),
        "nominal_fuel": nominal_fuel,
        "recovery_fuel_mean": nominal_fuel,
        "recovery_fuel_max": nominal_fuel,
        "selected_outage_errors": robust_errors[selected].astype(float).tolist(),
        "all_outage_errors": robust_errors.astype(float).tolist(),
        "controls": controls,
        "nominal_controls": controls,
        "recovery_controls": [],
        "selected_outage_indices": selected.astype(int).tolist(),
    }


def _refine_branch_recovery(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    outage_masks: np.ndarray,
    refine_cfg: dict,
    thresholds: dict,
) -> dict:
    start = time.perf_counter()
    schedule = np.asarray(schedule, dtype=int)
    active_idx = np.flatnonzero(schedule > 0)
    if active_idx.size == 0:
        return {"success": False, "mode": "branch_recovery", "message": "no active windows", "runtime_seconds": 0.0}

    use_only_initial_guesses = bool(refine_cfg.get("use_only_initial_control_guesses", False))
    if use_only_initial_guesses:
        guess_bank = configured_initial_control_guesses(schedule, cfg, refine_cfg)
        if not guess_bank:
            raise ValueError(
                "refinement.use_only_initial_control_guesses=True requires at least one "
                "refinement.initial_control_guesses entry"
            )
    else:
        guess_bank = None

    base_controls = feedback_controls_for_schedule(schedule, state0, target, cfg)
    selected = selected_outage_indices_for_schedule(
        schedule,
        state0,
        target,
        cfg,
        outage_masks,
        int(refine_cfg["selected_outages"]),
        int(refine_cfg.get("outage_selection_min_recovery_segments", 0)),
    )
    selected_masks = outage_masks[selected]

    recovery_indices: list[np.ndarray] = []
    for mask in selected_masks:
        end = _outage_end(mask)
        recovery_indices.append(active_idx[active_idx >= end])

    def pack_guess(controls: np.ndarray) -> np.ndarray:
        controls = project_controls_to_ball(controls, cfg.amax)
        chunks: list[np.ndarray] = [controls[active_idx].reshape(-1)]
        chunks.extend(controls[idx].reshape(-1) for idx in recovery_indices)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)

    def unpack(vec: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        cursor = 0
        nominal = np.zeros((cfg.n_segments, 3), dtype=float)
        nominal_len = active_idx.size * 3
        nominal[active_idx] = vec[cursor : cursor + nominal_len].reshape((-1, 3))
        nominal = project_controls_to_ball(nominal, cfg.amax)
        cursor += nominal_len
        branches: list[np.ndarray] = []
        for idx, mask in zip(recovery_indices, selected_masks):
            branch_len = idx.size * 3
            values = vec[cursor : cursor + branch_len]
            cursor += branch_len
            branches.append(_branch_controls_from_nominal(nominal, schedule, mask, idx, values, cfg.amax))
        return nominal, branches

    def residual(vec: np.ndarray) -> np.ndarray:
        nominal, branches = unpack(vec)
        nominal_final, _ = propagate_controls_batch(state0, nominal, cfg.mu, cfg.tf, cfg.substeps)
        res = [refine_cfg["state_residual_weight"] * _scaled_state_residual(nominal_final[0], target, cfg)]
        for branch in branches:
            robust_final, _ = propagate_controls_batch(state0, branch, cfg.mu, cfg.tf, cfg.substeps)
            res.append(refine_cfg["robust_residual_weight"] * _scaled_state_residual(robust_final[0], target, cfg))
        res.append(refine_cfg["fuel_residual_weight"] * nominal[active_idx].reshape(-1) / cfg.amax)
        for idx, branch in zip(recovery_indices, branches):
            if idx.size:
                res.append(refine_cfg["fuel_residual_weight"] * branch[idx].reshape(-1) / cfg.amax)
        if active_idx.size > 1:
            res.append(refine_cfg["smooth_residual_weight"] * np.diff(nominal[active_idx], axis=0).reshape(-1) / cfg.amax)
        for idx, branch in zip(recovery_indices, branches):
            if idx.size > 1:
                res.append(refine_cfg["smooth_residual_weight"] * np.diff(branch[idx], axis=0).reshape(-1) / cfg.amax)
        res.append(refine_cfg["control_regularization"] * nominal[active_idx].reshape(-1) / cfg.amax)
        for idx, branch in zip(recovery_indices, branches):
            if idx.size:
                res.append(refine_cfg["control_regularization"] * branch[idx].reshape(-1) / cfg.amax)
        return np.concatenate(res)

    def evaluate_result(result, label: str, elapsed: float) -> dict:
        nominal_controls, selected_branch_controls = unpack(result.x)
        nominal_final, _ = propagate_controls_batch(state0, nominal_controls, cfg.mu, cfg.tf, cfg.substeps)
        nominal_error = float(state_error(nominal_final[0], target, cfg.position_scale, cfg.velocity_scale))

        selected_errors = []
        for branch in selected_branch_controls:
            final, _ = propagate_controls_batch(state0, branch, cfg.mu, cfg.tf, cfg.substeps)
            selected_errors.append(float(state_error(final[0], target, cfg.position_scale, cfg.velocity_scale)))
        selected_worst = float(np.max(selected_errors)) if selected_errors else nominal_error

        selected_lookup = {int(idx): branch for idx, branch in zip(selected, selected_branch_controls)}
        all_errors = []
        all_branch_fuels = []
        for mask_index, mask in enumerate(outage_masks):
            if mask_index in selected_lookup:
                branch_controls = selected_lookup[mask_index]
            else:
                branch_controls = nominal_controls * (schedule * mask)[:, None]
            final, _ = propagate_controls_batch(state0, branch_controls, cfg.mu, cfg.tf, cfg.substeps)
            all_errors.append(float(state_error(final[0], target, cfg.position_scale, cfg.velocity_scale)))
            all_branch_fuels.append(control_fuel(branch_controls, cfg.tf))
        all_worst = float(np.max(all_errors)) if all_errors else selected_worst
        nominal_fuel = control_fuel(nominal_controls, cfg.tf)
        recovery_fuels = [control_fuel(branch, cfg.tf) for branch in selected_branch_controls]
        converged = bool(nominal_error <= thresholds["nominal_success"] and selected_worst <= thresholds["robust_success"])
        return {
            "success": converged,
            "mode": "branch_recovery",
            "optimizer_success": bool(result.success),
            "message": result.message,
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "attempt_runtime_seconds": float(elapsed),
            "best_initial_guess": label,
            "nominal_error": nominal_error,
            "worst_error": selected_worst,
            "selected_worst_error": selected_worst,
            "all_mask_worst_error": all_worst,
            "active_fraction": float(np.mean(schedule)),
            "nominal_fuel": nominal_fuel,
            "recovery_fuel_mean": float(np.mean(recovery_fuels)) if recovery_fuels else nominal_fuel,
            "recovery_fuel_max": float(np.max(recovery_fuels)) if recovery_fuels else nominal_fuel,
            "all_mask_recovery_fuel_mean": float(np.mean(all_branch_fuels)) if all_branch_fuels else nominal_fuel,
            "selected_outage_errors": selected_errors,
            "all_outage_errors": all_errors,
            "controls": nominal_controls,
            "nominal_controls": nominal_controls,
            "recovery_controls": [branch.tolist() for branch in selected_branch_controls],
            "recovery_indices": [idx.astype(int).tolist() for idx in recovery_indices],
            "selected_outage_indices": selected.astype(int).tolist(),
        }

    attempts = []
    total_nfev = 0
    if guess_bank is None:
        guess_bank = multistart_control_guesses(schedule, state0, target, cfg, refine_cfg)
    if not guess_bank:
        guess_bank = [("feedback", base_controls)]
    for label, guess_controls in guess_bank:
        attempt_start = time.perf_counter()
        result = least_squares(
            residual,
            pack_guess(guess_controls),
            bounds=(-cfg.amax, cfg.amax),
            max_nfev=int(refine_cfg["max_nfev"]),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
            verbose=0,
        )
        total_nfev += int(result.nfev)
        attempts.append(evaluate_result(result, label, time.perf_counter() - attempt_start))

    best = min(
        attempts,
        key=lambda item: (
            not bool(item["success"]),
            max(0.0, item["nominal_error"] - thresholds["nominal_success"])
            + max(0.0, item["selected_worst_error"] - thresholds["robust_success"]),
            item["selected_worst_error"],
            item["nominal_error"],
            item["cost"],
        ),
    )
    attempt_summaries = [
        {
            "initial_guess": item["best_initial_guess"],
            "success": bool(item["success"]),
            "optimizer_success": bool(item["optimizer_success"]),
            "nominal_error": float(item["nominal_error"]),
            "selected_worst_error": float(item["selected_worst_error"]),
            "all_mask_worst_error": float(item["all_mask_worst_error"]),
            "cost": float(item["cost"]),
            "nfev": int(item["nfev"]),
            "runtime_seconds": float(item["attempt_runtime_seconds"]),
        }
        for item in attempts
    ]
    best_attempt_nfev = int(best.get("nfev", 0))
    best["nfev"] = int(total_nfev)
    best["best_attempt_nfev"] = best_attempt_nfev
    best["runtime_seconds"] = float(time.perf_counter() - start)
    best["multistart_attempts"] = attempt_summaries
    return best


def refine_schedule(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    outage_masks: np.ndarray,
    refine_cfg: dict,
    thresholds: dict,
) -> dict:
    mode = refine_cfg.get("mode", "single_sequence")
    if mode == "single_sequence":
        return _refine_single_sequence(schedule, state0, target, cfg, outage_masks, refine_cfg, thresholds)
    if mode == "branch_recovery":
        return _refine_branch_recovery(schedule, state0, target, cfg, outage_masks, refine_cfg, thresholds)
    raise ValueError(f"unknown refinement mode: {mode}")
