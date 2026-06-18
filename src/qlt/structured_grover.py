from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroverSubspaceResult:
    oracle: str
    iterations: int
    good_mask: np.ndarray
    probabilities: np.ndarray
    initial_probabilities: np.ndarray
    good_probability: float
    expected_cost: float


@dataclass(frozen=True)
class PredeclaredGroverProtocol:
    name: str
    good_rule: str
    iteration_rule: str
    iteration_cap: int
    initial_good_probability: float
    selected_iterations: int
    predicted_good_probability: float
    result: GroverSubspaceResult


def probability_state(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a flat vector")
    if probabilities.size == 0:
        raise ValueError("probabilities must be nonempty")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be nonnegative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    return np.sqrt(probabilities / total).astype(np.complex128)


def threshold_good_mask(costs: np.ndarray, threshold: float) -> np.ndarray:
    costs = _flat_costs(costs)
    return costs <= float(threshold)


def top_k_good_mask(costs: np.ndarray, top_k: int) -> np.ndarray:
    costs = _flat_costs(costs)
    top_k = int(top_k)
    if top_k < 1 or top_k > costs.size:
        raise ValueError("top_k must be between 1 and the number of feasible states")
    order = np.argsort(costs, kind="mergesort")
    mask = np.zeros(costs.size, dtype=bool)
    mask[order[:top_k]] = True
    return mask


def dual_candidate_good_mask(costs: np.ndarray) -> np.ndarray:
    costs = _flat_costs(costs)
    return top_k_good_mask(costs, min(2, costs.size))


def amplitude_amplification_success_probability(initial_good_probability: float, iterations: int) -> float:
    a0 = float(initial_good_probability)
    if not np.isfinite(a0) or a0 < 0.0 or a0 > 1.0:
        raise ValueError("initial_good_probability must be in [0, 1]")
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    theta = np.arcsin(np.sqrt(a0))
    return float(np.sin((2 * iterations + 1) * theta) ** 2)


def choose_amplification_iterations(initial_good_probability: float, *, cap: int = 12) -> int:
    cap = int(cap)
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    probabilities = np.array(
        [amplitude_amplification_success_probability(initial_good_probability, iteration) for iteration in range(cap + 1)],
        dtype=float,
    )
    best_probability = float(np.max(probabilities))
    tied = np.flatnonzero(np.isclose(probabilities, best_probability, rtol=0.0, atol=1e-12))
    return int(tied[0])


def grover_iteration(state: np.ndarray, reference_state: np.ndarray, good_mask: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128)
    reference_state = np.asarray(reference_state, dtype=np.complex128)
    good_mask = np.asarray(good_mask, dtype=bool)
    if state.ndim != 1 or reference_state.shape != state.shape or good_mask.shape != state.shape:
        raise ValueError("state, reference_state, and good_mask must be flat vectors of the same shape")
    norm = float(np.linalg.norm(reference_state))
    if norm <= 0.0:
        raise ValueError("reference_state must be nonzero")
    reference_state = reference_state / norm
    marked = state.copy()
    marked[good_mask] *= -1.0
    reflected = 2.0 * reference_state * np.vdot(reference_state, marked) - marked
    reflected_norm = float(np.linalg.norm(reflected))
    if reflected_norm <= 0.0:
        raise RuntimeError("Grover iteration produced a zero state")
    return reflected / reflected_norm


def amplify_subspace(
    costs: np.ndarray,
    initial_probabilities: np.ndarray,
    good_mask: np.ndarray,
    *,
    iterations: int,
    oracle: str,
) -> GroverSubspaceResult:
    costs = _flat_costs(costs)
    initial_state = probability_state(initial_probabilities)
    good_mask = np.asarray(good_mask, dtype=bool)
    if good_mask.shape != costs.shape:
        raise ValueError("good_mask must match costs")
    if not bool(np.any(good_mask)):
        raise ValueError("good_mask must mark at least one feasible state")
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    state = initial_state.copy()
    for _ in range(iterations):
        state = grover_iteration(state, initial_state, good_mask)
    probabilities = np.abs(state) ** 2
    probabilities = probabilities / probabilities.sum()
    initial_probabilities = np.abs(initial_state) ** 2
    return GroverSubspaceResult(
        oracle=str(oracle),
        iterations=iterations,
        good_mask=good_mask.copy(),
        probabilities=probabilities,
        initial_probabilities=initial_probabilities,
        good_probability=float(probabilities[good_mask].sum()),
        expected_cost=float(probabilities @ costs),
    )


def grover_sweep(
    costs: np.ndarray,
    initial_probabilities: np.ndarray,
    *,
    thresholds: list[float] | tuple[float, ...] = (),
    top_ks: list[int] | tuple[int, ...] = (),
    iterations: list[int] | tuple[int, ...],
) -> list[GroverSubspaceResult]:
    costs = _flat_costs(costs)
    rows: list[GroverSubspaceResult] = []
    for threshold in thresholds:
        mask = threshold_good_mask(costs, float(threshold))
        if not np.any(mask):
            continue
        for iteration_count in iterations:
            rows.append(
                amplify_subspace(
                    costs,
                    initial_probabilities,
                    mask,
                    iterations=int(iteration_count),
                    oracle=f"threshold<={float(threshold):.12g}",
                )
            )
    for top_k in top_ks:
        mask = top_k_good_mask(costs, int(top_k))
        for iteration_count in iterations:
            rows.append(
                amplify_subspace(
                    costs,
                    initial_probabilities,
                    mask,
                    iterations=int(iteration_count),
                    oracle=f"top_{int(top_k)}",
                )
            )
    return rows


def run_dual_candidate_warm_amplification(
    costs: np.ndarray,
    initial_probabilities: np.ndarray,
    *,
    iteration_cap: int = 12,
) -> PredeclaredGroverProtocol:
    costs = _flat_costs(costs)
    good_mask = dual_candidate_good_mask(costs)
    initial_state = probability_state(initial_probabilities)
    p0 = np.abs(initial_state) ** 2
    initial_good_probability = float(p0[good_mask].sum())
    selected_iterations = choose_amplification_iterations(initial_good_probability, cap=iteration_cap)
    result = amplify_subspace(
        costs,
        p0,
        good_mask,
        iterations=selected_iterations,
        oracle=f"top_{int(good_mask.sum())}",
    )
    return PredeclaredGroverProtocol(
        name="dual_candidate_warm_amplification",
        good_rule="top_2_recorded_selected_worst_error_or_all_states_if_M_lt_2",
        iteration_rule="argmax_r_in_0..cap sin^2((2r+1)asin(sqrt(a0))) with smaller-r tie break",
        iteration_cap=int(iteration_cap),
        initial_good_probability=initial_good_probability,
        selected_iterations=selected_iterations,
        predicted_good_probability=amplitude_amplification_success_probability(
            initial_good_probability,
            selected_iterations,
        ),
        result=result,
    )


def _flat_costs(costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1:
        raise ValueError("costs must be a flat vector")
    if costs.size == 0:
        raise ValueError("costs must be nonempty")
    if not np.all(np.isfinite(costs)):
        raise ValueError("costs must be finite")
    return costs
