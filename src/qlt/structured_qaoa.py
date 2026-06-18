from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class StructuredQaoaResult:
    mixer: str
    depth: int
    angles: np.ndarray
    expected_cost: float
    probabilities: np.ndarray
    initial_probabilities: np.ndarray


def one_coast_schedules(n_segments: int) -> np.ndarray:
    """Return all schedules with exactly one coast window in coast-index order."""
    schedules = np.ones((int(n_segments), int(n_segments)), dtype=int)
    for coast_index in range(int(n_segments)):
        schedules[coast_index, coast_index] = 0
    return schedules


def adjacency_mixer(n_states: int, topology: str = "path") -> np.ndarray:
    """Build a feasible-subspace mixer adjacency matrix.

    For the one-coast benchmark, each basis state is identified by the coast-window
    position. The path topology swaps the coast with an adjacent thrust window, the
    Hamming-weight-preserving XY move restricted to this cardinality sector.
    """
    n_states = int(n_states)
    mixer = np.zeros((n_states, n_states), dtype=float)
    if topology == "path":
        for index in range(n_states - 1):
            mixer[index, index + 1] = 1.0
            mixer[index + 1, index] = 1.0
    elif topology == "cycle":
        for index in range(n_states):
            mixer[index, (index + 1) % n_states] = 1.0
            mixer[(index + 1) % n_states, index] = 1.0
    elif topology == "complete":
        mixer[:] = 1.0
        np.fill_diagonal(mixer, 0.0)
    else:
        raise ValueError(f"unknown structured-QAOA mixer topology: {topology}")
    return mixer


def inverse_cost_initial_probabilities(costs: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1:
        raise ValueError("costs must be a flat vector")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    weights = 1.0 / (costs + float(epsilon))
    probabilities = weights / weights.sum()
    return probabilities


def exponential_cost_initial_probabilities(costs: np.ndarray, beta: float = 1.0) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1:
        raise ValueError("costs must be a flat vector")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    shifted = costs - float(np.min(costs))
    weights = np.exp(-float(beta) * shifted)
    probabilities = weights / weights.sum()
    return probabilities


def cost_biased_initial_probabilities(
    costs: np.ndarray,
    *,
    mode: str = "inverse",
    epsilon: float = 0.05,
    beta: float = 1.0,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "inverse":
        return inverse_cost_initial_probabilities(costs, epsilon=epsilon)
    if mode in {"exponential", "exp"}:
        return exponential_cost_initial_probabilities(costs, beta=beta)
    raise ValueError(f"unknown warm-start mode: {mode}")


def qaoa_probabilities_subspace(
    costs: np.ndarray,
    mixer: np.ndarray,
    angles: np.ndarray,
    depth: int,
    initial_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    mixer = np.asarray(mixer, dtype=float)
    angles = np.asarray(angles, dtype=float)
    depth = int(depth)
    if costs.ndim != 1:
        raise ValueError("costs must be a flat vector")
    if mixer.shape != (costs.size, costs.size):
        raise ValueError("mixer shape must match the number of feasible states")
    if angles.shape != (2 * depth,):
        raise ValueError("angles must contain depth gammas followed by depth betas")
    if initial_probabilities is None:
        initial_probabilities = np.full(costs.size, 1.0 / costs.size, dtype=float)
    initial_probabilities = np.asarray(initial_probabilities, dtype=float)
    if initial_probabilities.shape != costs.shape:
        raise ValueError("initial probabilities must match the number of feasible states")
    if np.any(initial_probabilities < 0.0) or initial_probabilities.sum() <= 0.0:
        raise ValueError("initial probabilities must be nonnegative and nonzero")
    initial_probabilities = initial_probabilities / initial_probabilities.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(mixer)
    state = np.sqrt(initial_probabilities).astype(np.complex128)
    for gamma, beta in zip(angles[:depth], angles[depth:]):
        state *= np.exp(-1j * float(gamma) * costs)
        state = eigenvectors @ (np.exp(-1j * float(beta) * eigenvalues) * (eigenvectors.T.conj() @ state))
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def expected_subspace_cost(
    costs: np.ndarray,
    mixer: np.ndarray,
    angles: np.ndarray,
    depth: int,
    initial_probabilities: np.ndarray | None = None,
) -> float:
    probabilities = qaoa_probabilities_subspace(costs, mixer, angles, depth, initial_probabilities)
    return float(probabilities @ np.asarray(costs, dtype=float))


def optimize_subspace_qaoa(
    costs: np.ndarray,
    mixer: np.ndarray,
    *,
    depth: int,
    restarts: int = 32,
    maxiter: int = 400,
    seed: int = 20260618,
    initial_probabilities: np.ndarray | None = None,
) -> StructuredQaoaResult:
    depth = int(depth)
    if depth < 1:
        raise ValueError("structured-QAOA depth must be at least 1")
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 2.0 * np.pi)] * depth + [(0.0, np.pi)] * depth
    best_angles: np.ndarray | None = None
    best_cost = np.inf

    if initial_probabilities is None:
        initial_probabilities = np.full(np.asarray(costs).size, 1.0 / np.asarray(costs).size, dtype=float)
    initial_probabilities = np.asarray(initial_probabilities, dtype=float)
    initial_probabilities = initial_probabilities / initial_probabilities.sum()
    mixer_eigenvalues, mixer_eigenvectors = np.linalg.eigh(np.asarray(mixer, dtype=float))

    def objective(angles: np.ndarray) -> float:
        clipped = np.asarray(
            [np.clip(value, low, high) for value, (low, high) in zip(angles, bounds)],
            dtype=float,
        )
        probabilities = _qaoa_probabilities_with_mixer_eigendecomposition(
            costs,
            mixer_eigenvalues,
            mixer_eigenvectors,
            clipped,
            depth,
            initial_probabilities,
        )
        return float(probabilities @ np.asarray(costs, dtype=float))

    starts = [np.full(2 * depth, 0.25, dtype=float)]
    starts.extend(rng.uniform([low for low, _ in bounds], [high for _, high in bounds]) for _ in range(max(0, int(restarts) - 1)))
    for initial in starts:
        initial_cost = objective(initial)
        if initial_cost < best_cost:
            best_angles = np.asarray(initial, dtype=float)
            best_cost = float(initial_cost)
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        candidate = np.asarray(result.x, dtype=float)
        candidate_cost = objective(candidate)
        if candidate_cost < best_cost:
            best_angles = candidate
            best_cost = float(candidate_cost)

    assert best_angles is not None
    return StructuredQaoaResult(
        mixer="subspace",
        depth=depth,
        angles=best_angles,
        expected_cost=best_cost,
        probabilities=_qaoa_probabilities_with_mixer_eigendecomposition(
            costs,
            mixer_eigenvalues,
            mixer_eigenvectors,
            best_angles,
            depth,
            initial_probabilities,
        ),
        initial_probabilities=initial_probabilities.copy(),
    )


def _qaoa_probabilities_with_mixer_eigendecomposition(
    costs: np.ndarray,
    mixer_eigenvalues: np.ndarray,
    mixer_eigenvectors: np.ndarray,
    angles: np.ndarray,
    depth: int,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    angles = np.asarray(angles, dtype=float)
    state = np.sqrt(np.asarray(initial_probabilities, dtype=float)).astype(np.complex128)
    for gamma, beta in zip(angles[:depth], angles[depth:]):
        state *= np.exp(-1j * float(gamma) * costs)
        state = mixer_eigenvectors @ (
            np.exp(-1j * float(beta) * mixer_eigenvalues) * (mixer_eigenvectors.T.conj() @ state)
        )
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()
