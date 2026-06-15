from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .objective import Evaluator
from .surrogate import QuboModel, all_bitstrings


@dataclass
class SolverResult:
    method: str
    best_schedule: np.ndarray
    best_metrics: dict
    evaluated: list[tuple[np.ndarray, dict]]
    true_evaluations: int
    runtime_seconds: float


def random_schedule(rng: np.random.Generator, n: int, p: float = 0.5) -> np.ndarray:
    return (rng.random(n) < p).astype(int)


def _finish(method: str, start_count: int, evaluator: Evaluator, start_time: float, rows: list[tuple[np.ndarray, dict]]) -> SolverResult:
    best_schedule, best_metrics = min(rows, key=lambda item: item[1]["objective"])
    return SolverResult(method, best_schedule.copy(), dict(best_metrics), rows, evaluator.count - start_count, time.perf_counter() - start_time)


def solve_random(evaluator: Evaluator, rng: np.random.Generator, n: int, samples: int) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    rows = []
    for _ in range(samples):
        s = random_schedule(rng, n)
        rows.append((s, evaluator.evaluate(s)))
    return _finish("random", start_count, evaluator, start_time, rows)


def solve_cem(evaluator: Evaluator, rng: np.random.Generator, n: int, cfg: dict) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    p = np.full(n, 0.5)
    rows: list[tuple[np.ndarray, dict]] = []
    elite_n = max(2, int(round(cfg["elite_fraction"] * cfg["batch_size"])))
    smoothing = float(cfg["smoothing"])
    for _ in range(cfg["iterations"]):
        batch = (rng.random((cfg["batch_size"], n)) < p).astype(int)
        batch_rows = [(s, evaluator.evaluate(s)) for s in batch]
        rows.extend(batch_rows)
        elites = [s for s, _ in sorted(batch_rows, key=lambda item: item[1]["objective"])[:elite_n]]
        elite_mean = np.mean(elites, axis=0)
        p = smoothing * p + (1.0 - smoothing) * np.clip(elite_mean, 0.05, 0.95)
    return _finish("cross_entropy", start_count, evaluator, start_time, rows)


def solve_genetic(evaluator: Evaluator, rng: np.random.Generator, n: int, cfg: dict) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    population = int(cfg["population"])
    elite_count = int(cfg["elite"])
    generations = int(cfg["generations"])
    mutation_rate = float(cfg["mutation_rate"])
    max_evaluations = cfg.get("max_evaluations", cfg.get("evaluation_budget"))
    if max_evaluations is not None:
        max_evaluations = int(max_evaluations)
        if max_evaluations < 1:
            raise ValueError("genetic max_evaluations must be at least 1")

    def has_budget() -> bool:
        return max_evaluations is None or evaluator.count - start_count < max_evaluations

    pop = (rng.random((population, n)) < 0.5).astype(int)
    rows: list[tuple[np.ndarray, dict]] = []
    scored = []
    for s in pop:
        if not has_budget():
            break
        item = (s, evaluator.evaluate(s))
        scored.append(item)
        rows.append(item)
    for _ in range(generations):
        if not has_budget():
            break
        ranked = sorted(scored, key=lambda item: item[1]["objective"])
        elites = [s.copy() for s, _ in ranked[:elite_count]]
        children = elites.copy()
        while len(children) < population:
            a, b = rng.choice(len(elites), size=2, replace=True)
            cut = int(rng.integers(1, n))
            child = np.concatenate((elites[a][:cut], elites[b][cut:]))
            flips = rng.random(n) < mutation_rate
            child[flips] = 1 - child[flips]
            children.append(child)
        pop = np.asarray(children, dtype=int)
        scored = [(s, m) for s, m in ranked[:elite_count]]
        for s in pop[elite_count:]:
            if not has_budget():
                break
            item = (s, evaluator.evaluate(s))
            scored.append(item)
            rows.append(item)
    return _finish("genetic", start_count, evaluator, start_time, rows)


def solve_true_sa(evaluator: Evaluator, rng: np.random.Generator, n: int, cfg: dict) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    current = random_schedule(rng, n)
    current_metrics = evaluator.evaluate(current)
    rows = [(current.copy(), current_metrics)]
    best = (current.copy(), current_metrics)
    steps = int(cfg["steps"])
    for k in range(1, steps):
        temp = cfg["temp0"] * (cfg["temp_min"] / cfg["temp0"]) ** (k / max(1, steps - 1))
        proposal = current.copy()
        proposal[int(rng.integers(n))] ^= 1
        metrics = evaluator.evaluate(proposal)
        delta = metrics["objective"] - current_metrics["objective"]
        if delta <= 0.0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            current, current_metrics = proposal, metrics
        if current_metrics["objective"] < best[1]["objective"]:
            best = (current.copy(), current_metrics)
        rows.append((proposal.copy(), metrics))
    return _finish("true_sa", start_count, evaluator, start_time, rows + [best])


def solve_surrogate_sa(
    evaluator: Evaluator,
    rng: np.random.Generator,
    n: int,
    cfg: dict,
    qubo: QuboModel,
) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    current = random_schedule(rng, n)
    current_e = float(qubo.energy(current)[0])
    candidates: dict[str, np.ndarray] = {"".join(map(str, current)): current.copy()}
    steps = int(cfg["steps"])
    for k in range(1, steps + 1):
        temp = cfg["temp0"] * (cfg["temp_min"] / cfg["temp0"]) ** (k / max(1, steps))
        proposal = current.copy()
        proposal[int(rng.integers(n))] ^= 1
        e = float(qubo.energy(proposal)[0])
        if e <= current_e or rng.random() < np.exp(-(e - current_e) / max(temp, 1e-12)):
            current, current_e = proposal, e
        candidates["".join(map(str, current))] = current.copy()
        candidates["".join(map(str, proposal))] = proposal.copy()
    ranked = sorted(candidates.values(), key=lambda s: float(qubo.energy(s)[0]))
    rows = [(s, evaluator.evaluate(s)) for s in ranked[: cfg["candidates_to_evaluate"]]]
    return _finish("surrogate_qubo_sa", start_count, evaluator, start_time, rows)


def apply_rx_all_qubits(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    psi = state
    for q in range(n):
        reshaped = psi.reshape((2**q, 2, -1))
        zero = reshaped[:, 0, :].copy()
        one = reshaped[:, 1, :].copy()
        reshaped[:, 0, :] = c * zero + s * one
        reshaped[:, 1, :] = s * zero + c * one
        psi = reshaped.reshape(-1)
    return psi


def _qaoa_angle_arrays(gamma, beta) -> tuple[np.ndarray, np.ndarray]:
    gammas = np.atleast_1d(np.asarray(gamma, dtype=float))
    betas = np.atleast_1d(np.asarray(beta, dtype=float))
    if gammas.shape != betas.shape:
        raise ValueError("QAOA gamma and beta must have the same number of layers")
    if gammas.size < 1:
        raise ValueError("QAOA depth p must be at least 1")
    return gammas, betas


def qaoa_probabilities_from_energies(energies: np.ndarray, n: int, gamma, beta) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    state = np.full(2**n, 1.0 / np.sqrt(2**n), dtype=np.complex128)
    gammas, betas = _qaoa_angle_arrays(gamma, beta)
    for layer_gamma, layer_beta in zip(gammas, betas):
        state *= np.exp(-1j * layer_gamma * energies)
        state = apply_rx_all_qubits(state, layer_beta, n)
    probs = np.abs(state) ** 2
    probs /= probs.sum()
    return probs


def qaoa_distribution(qubo: QuboModel, gamma, beta) -> tuple[np.ndarray, np.ndarray]:
    n = qubo.n
    bits = all_bitstrings(n)
    energies = qubo.energy(bits)
    probs = qaoa_probabilities_from_energies(energies, n, gamma, beta)
    return probs, energies


def qaoa_expected_energy_from_energies(energies: np.ndarray, n: int, angles: np.ndarray, p: int) -> float:
    probs = qaoa_probabilities_from_energies(energies, n, angles[:p], angles[p:])
    return float(probs @ energies)


def qaoa_expected_energy(qubo: QuboModel, angles: np.ndarray | list[float] | tuple[float, ...], p: int | None = None) -> float:
    values = np.asarray(angles, dtype=float)
    if values.ndim != 1:
        raise ValueError("QAOA angles must be a flat vector")
    if p is None:
        if values.size % 2 != 0:
            raise ValueError("QAOA angle vector length must be even")
        p = values.size // 2
    p = int(p)
    if p < 1 or values.size != 2 * p:
        raise ValueError("QAOA angle vector must contain p gammas followed by p betas")
    bits = all_bitstrings(qubo.n)
    energies = qubo.energy(bits)
    return qaoa_expected_energy_from_energies(energies, qubo.n, values, p)


def random_qaoa_angles(rng: np.random.Generator, p: int) -> np.ndarray:
    p = int(p)
    if p < 1:
        raise ValueError("QAOA depth p must be at least 1")
    return np.concatenate(
        (
            rng.uniform(0.0, np.pi, size=p),
            rng.uniform(0.0, np.pi / 2.0, size=p),
        )
    )


def optimize_qaoa_angles(
    qubo: QuboModel,
    rng: np.random.Generator,
    cfg: dict,
) -> tuple[np.ndarray, float, dict]:
    p = int(cfg.get("p", 1))
    if p not in {1, 2}:
        raise ValueError("optimized QAOA currently supports p=1 or p=2")
    restarts = max(1, int(cfg.get("angle_restarts", cfg.get("angle_trials", 1))))
    maxiter = max(1, int(cfg.get("maxiter", 100)))
    bounds = [(0.0, np.pi)] * p + [(0.0, np.pi / 2.0)] * p
    energies = qubo.energy(all_bitstrings(qubo.n))
    best_angles: np.ndarray | None = None
    best_energy = np.inf
    objective_calls = 0

    def objective(angles: np.ndarray) -> float:
        nonlocal objective_calls
        objective_calls += 1
        clipped = np.asarray(
            [np.clip(value, low, high) for value, (low, high) in zip(angles, bounds)],
            dtype=float,
        )
        return qaoa_expected_energy_from_energies(energies, qubo.n, clipped, p)

    for _ in range(restarts):
        initial = random_qaoa_angles(rng, p)
        initial_energy = objective(initial)
        if initial_energy < best_energy:
            best_angles = initial.copy()
            best_energy = initial_energy
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        candidate_angles = np.asarray(result.x, dtype=float)
        candidate_angles = np.asarray(
            [np.clip(value, low, high) for value, (low, high) in zip(candidate_angles, bounds)],
            dtype=float,
        )
        candidate_energy = qaoa_expected_energy_from_energies(energies, qubo.n, candidate_angles, p)
        if candidate_energy < best_energy:
            best_angles = candidate_angles
            best_energy = candidate_energy

    assert best_angles is not None
    return best_angles, float(best_energy), {
        "optimizer": "scipy_minimize",
        "p": p,
        "angle_restarts": restarts,
        "maxiter": maxiter,
        "objective_calls": int(objective_calls),
    }


def select_qaoa_angles(
    qubo: QuboModel,
    rng: np.random.Generator,
    cfg: dict,
) -> tuple[np.ndarray, float, dict]:
    optimizer = cfg.get("optimizer")
    p = int(cfg.get("p", 1))
    energies = qubo.energy(all_bitstrings(qubo.n))
    if optimizer in {None, "", "random", "random_trials"}:
        best_angles = None
        best_exp = np.inf
        angle_trials = int(cfg["angle_trials"])
        for _ in range(angle_trials):
            angles = random_qaoa_angles(rng, p)
            exp_e = qaoa_expected_energy_from_energies(energies, qubo.n, angles, p)
            if exp_e < best_exp:
                best_exp = exp_e
                best_angles = angles
        assert best_angles is not None
        return best_angles, float(best_exp), {"optimizer": "random_trials", "p": p, "angle_trials": angle_trials}
    if optimizer == "scipy_minimize":
        return optimize_qaoa_angles(qubo, rng, cfg)
    raise ValueError(f"unknown QAOA optimizer: {optimizer}")


def solve_qaoa(
    evaluator: Evaluator,
    rng: np.random.Generator,
    cfg: dict,
    qubo: QuboModel,
) -> SolverResult:
    start_count, start_time = evaluator.count, time.perf_counter()
    n = qubo.n
    p = int(cfg.get("p", 1))
    if p < 1:
        raise ValueError("QAOA depth p must be at least 1")
    angles, _, angle_info = select_qaoa_angles(qubo, rng, cfg)
    best_probs, best_energies = qaoa_distribution(qubo, angles[:p], angles[p:])
    n_sample = min(int(cfg["sample_count"]), 2**n)
    idx = rng.choice(2**n, size=n_sample, replace=True, p=best_probs)
    unique = np.unique(idx)
    ranked_idx = sorted(unique, key=lambda i: (best_energies[i], -best_probs[i]))
    if len(ranked_idx) < cfg["candidates_to_evaluate"]:
        fill = np.argsort(best_energies)
        seen = set(ranked_idx)
        ranked_idx.extend([int(i) for i in fill if int(i) not in seen][: cfg["candidates_to_evaluate"] - len(ranked_idx)])
    bits = all_bitstrings(n)
    rows = [(bits[i].astype(int), evaluator.evaluate(bits[i])) for i in ranked_idx[: cfg["candidates_to_evaluate"]]]
    result = _finish("qaoa_statevector", start_count, evaluator, start_time, rows)
    result.best_metrics = {
        **result.best_metrics,
        "qaoa_expected_qubo_energy": float(angle_info.get("expected_energy", qaoa_expected_energy(qubo, angles, p))),
        "qaoa_angle_optimizer": angle_info["optimizer"],
        "qaoa_depth": int(p),
    }
    return result
