from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnlineSearchResult:
    method: str
    seed: int
    budget: int
    initial_index: int
    initial_cost: float
    best_index: int
    best_cost: float
    global_best_index: int
    global_best_cost: float
    success_global_optimum: bool
    hit_top1: bool
    hit_top5: bool
    hit_top10_percent: bool
    cost_evaluations: int
    incumbent_evaluations: int
    candidate_evaluations: int
    threshold_oracle_calls: int
    total_oracle_calls: int
    iterations_attempted: int
    accepted_updates: int
    marked_count_last: int


def online_threshold_search(
    costs: np.ndarray,
    *,
    budget: int,
    seed: int,
    initial_probabilities: np.ndarray | None = None,
    growth_rate: float = 6.0 / 5.0,
    max_grover_window: int | None = None,
) -> OnlineSearchResult:
    """Simulate BBHT/Durr-Hoyer-style online threshold amplitude amplification.

    The oracle is only the current-incumbent threshold comparison C_j < C_best.
    The simulator has access to the finite cost vector to implement that oracle,
    but it does not accept or construct a known top-k mask.
    """
    costs = _flat_costs(costs)
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must allow at least one incumbent evaluation")
    if growth_rate <= 1.0:
        raise ValueError("growth_rate must be > 1")
    rng = np.random.default_rng(int(seed))
    probabilities = _probabilities(initial_probabilities, costs.size)
    max_window = int(max_grover_window) if max_grover_window is not None else max(1, int(np.ceil(np.sqrt(costs.size))))
    if max_window < 1:
        raise ValueError("max_grover_window must be positive")

    initial_index = int(rng.choice(costs.size, p=probabilities))
    incumbent_index = initial_index
    incumbent_cost = float(costs[incumbent_index])
    cost_evaluations = 1
    incumbent_evaluations = 1
    candidate_evaluations = 0
    threshold_oracle_calls = 0
    total_oracle_calls = 1
    iterations_attempted = 0
    accepted_updates = 0
    marked_count_last = int(np.count_nonzero(costs < incumbent_cost))
    grover_window = 1

    while total_oracle_calls < budget:
        remaining_after_candidate = budget - total_oracle_calls - 1
        if remaining_after_candidate < 0:
            break
        iteration_cap = min(grover_window, max_window, remaining_after_candidate)
        iterations = int(rng.integers(0, iteration_cap + 1))

        marked = costs < incumbent_cost
        marked_count_last = int(np.count_nonzero(marked))
        sample_probabilities = _amplified_probabilities(probabilities, marked, iterations)
        threshold_oracle_calls += iterations
        total_oracle_calls += iterations
        iterations_attempted += iterations

        candidate_index = int(rng.choice(costs.size, p=sample_probabilities))
        candidate_cost = float(costs[candidate_index])
        candidate_evaluations += 1
        cost_evaluations += 1
        total_oracle_calls += 1
        if candidate_cost < incumbent_cost:
            incumbent_index = candidate_index
            incumbent_cost = candidate_cost
            accepted_updates += 1
            grover_window = 1
        else:
            grover_window = min(max_window, int(np.ceil(growth_rate * grover_window)))

    return _result(
        method="online_threshold_amplitude_amplification",
        seed=int(seed),
        budget=budget,
        initial_index=initial_index,
        initial_cost=float(costs[initial_index]),
        best_index=incumbent_index,
        best_cost=incumbent_cost,
        costs=costs,
        cost_evaluations=cost_evaluations,
        incumbent_evaluations=incumbent_evaluations,
        candidate_evaluations=candidate_evaluations,
        threshold_oracle_calls=threshold_oracle_calls,
        total_oracle_calls=total_oracle_calls,
        iterations_attempted=iterations_attempted,
        accepted_updates=accepted_updates,
        marked_count_last=marked_count_last,
    )


def uniform_random_baseline(costs: np.ndarray, *, budget: int, seed: int) -> OnlineSearchResult:
    costs = _flat_costs(costs)
    budget = min(int(budget), costs.size)
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(costs.size)[:budget]
    best_pos = int(np.argmin(costs[order]))
    initial_index = int(order[0])
    best_index = int(order[best_pos])
    return _result(
        method="uniform_random_without_replacement",
        seed=int(seed),
        budget=int(budget),
        initial_index=initial_index,
        initial_cost=float(costs[initial_index]),
        best_index=best_index,
        best_cost=float(costs[best_index]),
        costs=costs,
        cost_evaluations=int(order.size),
        incumbent_evaluations=1,
        candidate_evaluations=max(0, int(order.size) - 1),
        threshold_oracle_calls=0,
        total_oracle_calls=int(order.size),
        iterations_attempted=0,
        accepted_updates=_count_incumbent_improvements(costs[order]),
        marked_count_last=0,
    )


def simulated_annealing_baseline(
    costs: np.ndarray,
    *,
    budget: int,
    seed: int,
    schedules: np.ndarray | None = None,
) -> OnlineSearchResult:
    """Matched-budget finite-subspace simulated annealing baseline.

    The state is an index in the finite constrained subspace.  When binary
    cardinality schedules are supplied, proposals use one-on/one-off Hamming
    swaps and are mapped back to the supplied finite state list.  Otherwise the
    fallback proposal is a rank/index-neighbor random walk over finite states.
    Only sampled candidates are evaluated; the full cost vector is used only to
    answer those direct evaluations and to compute the final result metrics.
    """
    costs = _flat_costs(costs)
    budget = _positive_budget(budget)
    rng = np.random.default_rng(int(seed))
    schedule_data = _prepare_schedules(schedules, costs.size)

    current_index = int(rng.integers(costs.size))
    current_cost = float(costs[current_index])
    order = [current_index]
    observed_costs = [current_cost]
    best_index = current_index
    best_cost = current_cost
    scale = max(1e-12, 0.05 * (abs(current_cost) + 1.0))

    for step in range(1, budget):
        candidate_index = _sa_proposal(current_index, costs.size, rng, schedule_data)
        candidate_cost = float(costs[candidate_index])
        order.append(candidate_index)
        observed_costs.append(candidate_cost)
        if candidate_cost < best_cost:
            best_index = candidate_index
            best_cost = candidate_cost
        temperature = max(1e-12, scale * (1.0 - step / max(1, budget)) + 1e-12)
        accept = candidate_cost <= current_cost
        if not accept:
            accept_probability = float(np.exp(-(candidate_cost - current_cost) / temperature))
            accept = bool(rng.random() < min(1.0, accept_probability))
        if accept:
            current_index = candidate_index
            current_cost = candidate_cost

    return _baseline_result(
        method="simulated_annealing_baseline",
        seed=int(seed),
        budget=budget,
        order=np.asarray(order, dtype=int),
        ordered_costs=np.asarray(observed_costs, dtype=float),
        costs=costs,
    )


def cross_entropy_baseline(
    costs: np.ndarray,
    *,
    budget: int,
    seed: int,
    schedules: np.ndarray | None = None,
) -> OnlineSearchResult:
    """Matched-budget finite-subspace cross-entropy baseline.

    With schedules, the model is an independent Bernoulli distribution over
    bits, repaired to the schedule cardinality before lookup in the supplied
    finite subspace.  Without schedules, the model is a smoothed categorical
    distribution over finite-state indices.  Updates use only evaluated elite
    candidates.
    """
    costs = _flat_costs(costs)
    budget = _positive_budget(budget)
    rng = np.random.default_rng(int(seed))
    schedule_data = _prepare_schedules(schedules, costs.size)

    order: list[int] = []
    observed_costs: list[float] = []
    batch_size = min(max(4, int(np.ceil(np.sqrt(costs.size)))), 32, budget)
    smoothing = 0.65
    if schedule_data is not None:
        probabilities = np.full(schedule_data["width"], schedule_data["cardinality"] / schedule_data["width"], dtype=float)
    else:
        probabilities = np.full(costs.size, 1.0 / costs.size, dtype=float)

    while len(order) < budget:
        remaining = budget - len(order)
        batch = min(batch_size, remaining)
        for _ in range(batch):
            if schedule_data is None:
                candidate_index = int(rng.choice(costs.size, p=probabilities))
            else:
                candidate_index = _sample_schedule_index(probabilities, rng, schedule_data)
            order.append(candidate_index)
            observed_costs.append(float(costs[candidate_index]))
        elite_count = max(1, int(np.ceil(0.25 * len(order))))
        elite_positions = np.argsort(np.asarray(observed_costs, dtype=float), kind="mergesort")[:elite_count]
        elite_indices = np.asarray(order, dtype=int)[elite_positions]
        if schedule_data is None:
            target = np.full(costs.size, 1e-6 / costs.size, dtype=float)
            np.add.at(target, elite_indices, 1.0 / elite_indices.size)
            target /= target.sum()
            probabilities = smoothing * probabilities + (1.0 - smoothing) * target
            probabilities /= probabilities.sum()
        else:
            elite_bits = schedule_data["array"][elite_indices]
            target = elite_bits.mean(axis=0)
            probabilities = smoothing * probabilities + (1.0 - smoothing) * target
            probabilities = np.clip(probabilities, 0.02, 0.98)

    return _baseline_result(
        method="cross_entropy_baseline",
        seed=int(seed),
        budget=budget,
        order=np.asarray(order, dtype=int),
        ordered_costs=np.asarray(observed_costs, dtype=float),
        costs=costs,
    )


def genetic_baseline(
    costs: np.ndarray,
    *,
    budget: int,
    seed: int,
    schedules: np.ndarray | None = None,
) -> OnlineSearchResult:
    """Matched-budget finite-subspace genetic-search baseline.

    Schedule-aware runs use uniform crossover, bit mutation, cardinality repair,
    and lookup in the supplied finite subspace.  Without schedules, the fallback
    is an index-level population with tournament selection and local/random
    mutation.  Candidate quality is known only after sampled direct evaluation.
    """
    costs = _flat_costs(costs)
    budget = _positive_budget(budget)
    rng = np.random.default_rng(int(seed))
    schedule_data = _prepare_schedules(schedules, costs.size)

    order: list[int] = []
    observed_costs: list[float] = []
    population_size = min(max(6, int(np.ceil(np.sqrt(costs.size)))), 24, budget)
    for _ in range(population_size):
        idx = _random_state_index(costs.size, rng, schedule_data)
        order.append(idx)
        observed_costs.append(float(costs[idx]))

    while len(order) < budget:
        population = _best_unique_indices(order, observed_costs, population_size)
        parent_a = _tournament_select(population, costs, rng)
        parent_b = _tournament_select(population, costs, rng)
        if schedule_data is None:
            child = _index_child(parent_a, parent_b, costs.size, rng)
        else:
            child = _schedule_child(parent_a, parent_b, rng, schedule_data)
        order.append(child)
        observed_costs.append(float(costs[child]))

    return _baseline_result(
        method="genetic_baseline",
        seed=int(seed),
        budget=budget,
        order=np.asarray(order, dtype=int),
        ordered_costs=np.asarray(observed_costs, dtype=float),
        costs=costs,
    )


def ranked_baseline(costs: np.ndarray, *, budget: int, seed: int, scores: np.ndarray | None = None) -> OnlineSearchResult:
    """Deterministic warm-start baseline with seeded tie-breaking.

    Lower scores are sampled first. If no surrogate scores are supplied, the
    baseline uses the finite costs as an optimistic ranking diagnostic.
    """
    costs = _flat_costs(costs)
    budget = min(int(budget), costs.size)
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(int(seed))
    ranking_scores = costs if scores is None else _flat_costs(scores)
    if ranking_scores.shape != costs.shape:
        raise ValueError("scores must match costs")
    jitter = rng.uniform(0.0, 1e-12, size=costs.size)
    order = np.lexsort((jitter, ranking_scores))[:budget]
    best_pos = int(np.argmin(costs[order]))
    initial_index = int(order[0])
    best_index = int(order[best_pos])
    return _result(
        method="surrogate_ranked_baseline",
        seed=int(seed),
        budget=int(budget),
        initial_index=initial_index,
        initial_cost=float(costs[initial_index]),
        best_index=best_index,
        best_cost=float(costs[best_index]),
        costs=costs,
        cost_evaluations=int(order.size),
        incumbent_evaluations=1,
        candidate_evaluations=max(0, int(order.size) - 1),
        threshold_oracle_calls=0,
        total_oracle_calls=int(order.size),
        iterations_attempted=0,
        accepted_updates=_count_incumbent_improvements(costs[order]),
        marked_count_last=0,
    )


def _amplified_probabilities(initial_probabilities: np.ndarray, marked: np.ndarray, iterations: int) -> np.ndarray:
    marked = np.asarray(marked, dtype=bool)
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    state = np.sqrt(initial_probabilities).astype(np.complex128)
    reference = state.copy()
    if iterations == 0 or not bool(np.any(marked)):
        return initial_probabilities.copy()
    for _ in range(iterations):
        state = state.copy()
        state[marked] *= -1.0
        state = 2.0 * reference * np.vdot(reference, state) - state
        state /= np.linalg.norm(state)
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def _result(
    *,
    method: str,
    seed: int,
    budget: int,
    initial_index: int,
    initial_cost: float,
    best_index: int,
    best_cost: float,
    costs: np.ndarray,
    cost_evaluations: int,
    incumbent_evaluations: int,
    candidate_evaluations: int,
    threshold_oracle_calls: int,
    total_oracle_calls: int,
    iterations_attempted: int,
    accepted_updates: int,
    marked_count_last: int,
) -> OnlineSearchResult:
    order = np.argsort(costs, kind="mergesort")
    global_best_index = int(order[0])
    top5_count = min(5, costs.size)
    top10_count = max(1, int(np.ceil(0.10 * costs.size)))
    top5 = set(int(i) for i in order[:top5_count])
    top10 = set(int(i) for i in order[:top10_count])
    success = bool(best_index == global_best_index)
    return OnlineSearchResult(
        method=method,
        seed=int(seed),
        budget=int(budget),
        initial_index=int(initial_index),
        initial_cost=float(initial_cost),
        best_index=int(best_index),
        best_cost=float(best_cost),
        global_best_index=global_best_index,
        global_best_cost=float(costs[global_best_index]),
        success_global_optimum=success,
        hit_top1=success,
        hit_top5=bool(int(best_index) in top5),
        hit_top10_percent=bool(int(best_index) in top10),
        cost_evaluations=int(cost_evaluations),
        incumbent_evaluations=int(incumbent_evaluations),
        candidate_evaluations=int(candidate_evaluations),
        threshold_oracle_calls=int(threshold_oracle_calls),
        total_oracle_calls=int(total_oracle_calls),
        iterations_attempted=int(iterations_attempted),
        accepted_updates=int(accepted_updates),
        marked_count_last=int(marked_count_last),
    )


def _probabilities(values: np.ndarray | None, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, 1.0 / size, dtype=float)
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (size,):
        raise ValueError("initial_probabilities must match costs")
    if np.any(probabilities < 0.0):
        raise ValueError("initial_probabilities must be nonnegative")
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("initial_probabilities must have positive mass")
    return probabilities / total


def _count_incumbent_improvements(ordered_costs: np.ndarray) -> int:
    ordered_costs = _flat_costs(ordered_costs)
    incumbent = float(ordered_costs[0])
    updates = 0
    for value in ordered_costs[1:]:
        candidate = float(value)
        if candidate < incumbent:
            incumbent = candidate
            updates += 1
    return updates


def _baseline_result(
    *,
    method: str,
    seed: int,
    budget: int,
    order: np.ndarray,
    ordered_costs: np.ndarray,
    costs: np.ndarray,
) -> OnlineSearchResult:
    order = np.asarray(order, dtype=int)
    ordered_costs = _flat_costs(ordered_costs)
    if order.shape != ordered_costs.shape:
        raise ValueError("order and ordered_costs must have matching shapes")
    if order.size < 1:
        raise ValueError("baseline must evaluate at least one candidate")
    best_pos = int(np.argmin(ordered_costs))
    initial_index = int(order[0])
    best_index = int(order[best_pos])
    return _result(
        method=method,
        seed=int(seed),
        budget=int(budget),
        initial_index=initial_index,
        initial_cost=float(ordered_costs[0]),
        best_index=best_index,
        best_cost=float(ordered_costs[best_pos]),
        costs=costs,
        cost_evaluations=int(order.size),
        incumbent_evaluations=1,
        candidate_evaluations=max(0, int(order.size) - 1),
        threshold_oracle_calls=0,
        total_oracle_calls=int(order.size),
        iterations_attempted=0,
        accepted_updates=_count_incumbent_improvements(ordered_costs),
        marked_count_last=0,
    )


def _positive_budget(budget: int) -> int:
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    return budget


def _prepare_schedules(schedules: np.ndarray | None, size: int) -> dict | None:
    if schedules is None:
        return None
    array = np.asarray(schedules, dtype=int)
    if array.ndim != 2:
        raise ValueError("schedules must be a 2-D array")
    if array.shape[0] != size:
        raise ValueError("schedules must have one row per cost")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("schedules must be binary")
    cardinalities = array.sum(axis=1)
    cardinality = int(cardinalities[0])
    if not np.all(cardinalities == cardinality):
        raise ValueError("schedule-aware baselines require fixed-cardinality schedules")
    lookup = {tuple(int(v) for v in row): int(i) for i, row in enumerate(array)}
    return {
        "array": array,
        "lookup": lookup,
        "width": int(array.shape[1]),
        "cardinality": cardinality,
    }


def _random_state_index(size: int, rng: np.random.Generator, schedule_data: dict | None) -> int:
    if schedule_data is None:
        return int(rng.integers(size))
    return _sample_schedule_index(
        np.full(schedule_data["width"], schedule_data["cardinality"] / schedule_data["width"], dtype=float),
        rng,
        schedule_data,
    )


def _sample_schedule_index(probabilities: np.ndarray, rng: np.random.Generator, schedule_data: dict) -> int:
    bits = (rng.random(schedule_data["width"]) < probabilities).astype(int)
    bits = _repair_cardinality(bits, schedule_data["cardinality"], rng)
    index = schedule_data["lookup"].get(tuple(int(v) for v in bits))
    if index is not None:
        return int(index)
    return int(rng.integers(schedule_data["array"].shape[0]))


def _repair_cardinality(bits: np.ndarray, cardinality: int, rng: np.random.Generator) -> np.ndarray:
    bits = np.asarray(bits, dtype=int).copy()
    active = np.flatnonzero(bits == 1)
    inactive = np.flatnonzero(bits == 0)
    if active.size > cardinality:
        turn_off = rng.choice(active, size=active.size - cardinality, replace=False)
        bits[turn_off] = 0
    elif active.size < cardinality:
        turn_on = rng.choice(inactive, size=cardinality - active.size, replace=False)
        bits[turn_on] = 1
    return bits


def _sa_proposal(current_index: int, size: int, rng: np.random.Generator, schedule_data: dict | None) -> int:
    if schedule_data is None:
        step = int(rng.choice(np.array([-2, -1, 1, 2], dtype=int)))
        if rng.random() < 0.25:
            return int(rng.integers(size))
        return int((current_index + step) % size)
    bits = schedule_data["array"][current_index].copy()
    active = np.flatnonzero(bits == 1)
    inactive = np.flatnonzero(bits == 0)
    if active.size == 0 or inactive.size == 0:
        return int(rng.integers(size))
    bits[int(rng.choice(active))] = 0
    bits[int(rng.choice(inactive))] = 1
    return int(schedule_data["lookup"].get(tuple(int(v) for v in bits), rng.integers(size)))


def _best_unique_indices(order: list[int], observed_costs: list[float], limit: int) -> list[int]:
    best_by_index: dict[int, float] = {}
    for idx, cost in zip(order, observed_costs):
        idx = int(idx)
        cost = float(cost)
        if idx not in best_by_index or cost < best_by_index[idx]:
            best_by_index[idx] = cost
    ranked = sorted(best_by_index, key=lambda idx: (best_by_index[idx], idx))
    return [int(idx) for idx in ranked[: max(1, int(limit))]]


def _tournament_select(population: list[int], costs: np.ndarray, rng: np.random.Generator) -> int:
    contenders = rng.choice(np.asarray(population, dtype=int), size=min(3, len(population)), replace=True)
    return int(contenders[int(np.argmin(costs[contenders]))])


def _index_child(parent_a: int, parent_b: int, size: int, rng: np.random.Generator) -> int:
    if rng.random() < 0.20:
        return int(rng.integers(size))
    midpoint = int(round((int(parent_a) + int(parent_b)) / 2.0))
    mutation = int(rng.choice(np.array([-2, -1, 0, 1, 2], dtype=int)))
    return int((midpoint + mutation) % size)


def _schedule_child(parent_a: int, parent_b: int, rng: np.random.Generator, schedule_data: dict) -> int:
    a = schedule_data["array"][parent_a]
    b = schedule_data["array"][parent_b]
    mask = rng.random(schedule_data["width"]) < 0.5
    child = np.where(mask, a, b).astype(int)
    mutate = rng.random(schedule_data["width"]) < (1.0 / max(1, schedule_data["width"]))
    child[mutate] = 1 - child[mutate]
    child = _repair_cardinality(child, schedule_data["cardinality"], rng)
    return int(schedule_data["lookup"].get(tuple(int(v) for v in child), rng.integers(schedule_data["array"].shape[0])))


def _flat_costs(costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1:
        raise ValueError("costs must be a flat vector")
    if costs.size == 0:
        raise ValueError("costs must be nonempty")
    if not np.all(np.isfinite(costs)):
        raise ValueError("costs must be finite")
    return costs
