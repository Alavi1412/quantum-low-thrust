from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.structured_grover import (
    amplify_subspace,
    amplitude_amplification_success_probability,
    choose_amplification_iterations,
    run_dual_candidate_warm_amplification,
    threshold_good_mask,
    top_k_good_mask,
)


def test_uniform_four_state_grover_finds_single_marked_state() -> None:
    costs = np.array([0.0, 1.0, 2.0, 3.0])
    initial = np.full(4, 0.25)
    result = amplify_subspace(
        costs,
        initial,
        top_k_good_mask(costs, 1),
        iterations=1,
        oracle="top_1",
    )
    assert result.good_probability == 1.0
    np.testing.assert_allclose(result.probabilities, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-14)
    assert result.expected_cost == 0.0


def test_threshold_oracle_marks_all_costs_at_or_below_threshold() -> None:
    costs = np.array([0.4, 0.2, 0.2, 0.9])
    mask = threshold_good_mask(costs, 0.2)
    np.testing.assert_array_equal(mask, np.array([False, True, True, False]))


def test_amplification_preserves_normalized_feasible_subspace_distribution() -> None:
    costs = np.array([0.1, 0.2, 0.5])
    initial = np.array([4.0, 1.0, 1.0])
    result = amplify_subspace(
        costs,
        initial,
        top_k_good_mask(costs, 2),
        iterations=3,
        oracle="top_2",
    )
    assert result.probabilities.shape == costs.shape
    assert np.all(result.probabilities >= 0.0)
    np.testing.assert_allclose(result.probabilities.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(result.initial_probabilities.sum(), 1.0, atol=1e-14)


def test_predeclared_iteration_rule_uses_initial_good_mass_with_smallest_tie() -> None:
    a0 = 0.25
    selected = choose_amplification_iterations(a0, cap=12)
    expected = max(
        range(13),
        key=lambda r: (amplitude_amplification_success_probability(a0, r), -r),
    )
    assert selected == expected
    assert selected == 1


def test_dual_candidate_protocol_matches_exact_top_two_amplification() -> None:
    costs = np.array([0.0, 0.05, 0.8, 1.0])
    initial = np.array([0.30, 0.20, 0.25, 0.25])
    protocol = run_dual_candidate_warm_amplification(costs, initial, iteration_cap=12)

    assert protocol.name == "dual_candidate_warm_amplification"
    assert protocol.result.oracle == "top_2"
    assert protocol.selected_iterations == 0
    np.testing.assert_array_equal(protocol.result.good_mask, np.array([True, True, False, False]))
    np.testing.assert_allclose(protocol.initial_good_probability, 0.5, atol=1e-14)
    np.testing.assert_allclose(
        protocol.result.good_probability,
        amplitude_amplification_success_probability(0.5, protocol.selected_iterations),
        atol=1e-14,
    )
    np.testing.assert_allclose(protocol.result.probabilities.sum(), 1.0, atol=1e-14)
