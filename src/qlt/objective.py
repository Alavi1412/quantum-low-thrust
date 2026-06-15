from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import numpy as np

from .cr3bp import propagate_feedback_batch


@dataclass(frozen=True)
class ObjectiveConfig:
    mu: float
    tf: float
    n_segments: int
    substeps: int
    amax: float
    kr: float
    kv: float
    position_scale: float
    velocity_scale: float
    weights: dict[str, float]
    outage_lengths: tuple[int, ...]
    target_active_fraction: float | None = None
    target_active_weight: float = 0.0


def outage_masks(n: int, lengths: tuple[int, ...]) -> np.ndarray:
    masks: list[np.ndarray] = []
    for length in lengths:
        for start in range(0, n - length + 1):
            mask = np.ones(n, dtype=float)
            mask[start : start + length] = 0.0
            masks.append(mask)
    return np.asarray(masks, dtype=float)


def state_error(final_state: np.ndarray, target: np.ndarray, position_scale: float, velocity_scale: float) -> np.ndarray:
    s = np.asarray(final_state, dtype=float)
    diff = s - target
    pos = np.linalg.norm(diff[..., :3], axis=-1) / position_scale
    vel = np.linalg.norm(diff[..., 3:], axis=-1) / velocity_scale
    return np.sqrt(pos**2 + vel**2)


def smoothness(schedule: np.ndarray) -> float:
    if schedule.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(schedule))))


def evaluate_schedule(
    schedule: np.ndarray,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray | None = None,
    return_details: bool = False,
) -> dict[str, float | np.ndarray]:
    schedule = np.asarray(schedule, dtype=float)
    if masks is None:
        masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    all_schedules = np.vstack((schedule, schedule[None, :] * masks))
    finals, _ = propagate_feedback_batch(
        state0,
        all_schedules,
        target,
        cfg.mu,
        cfg.tf,
        cfg.amax,
        cfg.kr,
        cfg.kv,
        cfg.substeps,
    )
    errors = state_error(finals, target, cfg.position_scale, cfg.velocity_scale)
    nominal = float(errors[0])
    outage_errors = errors[1:]
    worst = float(np.max(outage_errors)) if outage_errors.size else nominal
    degradation = max(0.0, worst - nominal)
    active = float(np.mean(schedule))
    sm = smoothness(schedule)
    w = cfg.weights
    target_active_fraction = cfg.target_active_fraction
    target_active_weight = float(cfg.target_active_weight)
    active_target_deviation = 0.0 if target_active_fraction is None else active - float(target_active_fraction)
    active_target_penalty = 0.0
    if target_active_fraction is not None and target_active_weight != 0.0:
        active_target_penalty = target_active_weight * active_target_deviation**2
    objective = (
        w["nominal"] * nominal
        + w["robust_worst"] * worst
        + w["robust_degradation"] * degradation
        + w["active_fraction"] * active
        + w["smoothness"] * sm
        + active_target_penalty
    )
    out: dict[str, float | np.ndarray] = {
        "objective": float(objective),
        "nominal_error": nominal,
        "worst_error": worst,
        "robust_degradation": float(degradation),
        "active_fraction": active,
        "smoothness": sm,
        "active_target_penalty": float(active_target_penalty),
        "active_target_deviation": float(active_target_deviation),
        "target_active_weight": target_active_weight,
    }
    if target_active_fraction is not None:
        out["target_active_fraction"] = float(target_active_fraction)
    if return_details:
        out["outage_errors"] = outage_errors
        out["final_states"] = finals
        out["all_schedules"] = all_schedules
    return out


class Evaluator:
    def __init__(self, state0: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig):
        self.state0 = state0
        self.target = target
        self.cfg = cfg
        self.masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
        self.count = 0

    def evaluate(self, schedule: np.ndarray, return_details: bool = False) -> dict[str, float | np.ndarray]:
        self.count += 1
        return evaluate_schedule(schedule, self.state0, self.target, self.cfg, self.masks, return_details)

    def evaluate_many(self, schedules: list[np.ndarray] | np.ndarray) -> list[dict[str, float | np.ndarray]]:
        return [self.evaluate(s) for s in schedules]


def schedule_to_string(schedule: np.ndarray) -> str:
    return "".join(str(int(v)) for v in np.asarray(schedule, dtype=int))


def string_to_schedule(bits: str) -> np.ndarray:
    return np.asarray([1 if c == "1" else 0 for c in bits.strip()], dtype=int)
