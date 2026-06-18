from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_online_quantum_search import (
    exact_sign_test_pvalue,
    paired_comparisons,
    wilson_interval,
)
from qlt.online_quantum_search import (
    cross_entropy_baseline,
    genetic_baseline,
    online_threshold_search,
    ranked_baseline,
    simulated_annealing_baseline,
    uniform_random_baseline,
)


def test_wilson_interval_known_half_success_case_is_symmetric() -> None:
    lower, upper = wilson_interval(5, 10)

    assert np.isclose(lower, 0.23659309051256394)
    assert np.isclose(upper, 0.7634069094874361)
    assert np.isclose(lower + upper, 1.0)
    assert lower < 0.5 < upper


def test_exact_sign_test_pvalue_counts_wins_losses_and_ties() -> None:
    wins, losses, ties, pvalue = exact_sign_test_pvalue(np.array([-2.0, -1.0, -0.5, 0.0, 1.0, np.nan]))

    assert wins == 3
    assert losses == 1
    assert ties == 1
    assert np.isclose(pvalue, 0.625)


def test_paired_comparisons_reports_success_delta_and_cost_diff_sign() -> None:
    raw = pd.DataFrame(
        [
            _comparison_row("online_threshold_amplitude_amplification", 0, True, 1.0),
            _comparison_row("online_threshold_amplitude_amplification", 1, False, 2.0),
            _comparison_row("uniform_random_without_replacement", 0, False, 1.5),
            _comparison_row("uniform_random_without_replacement", 1, False, 2.5),
            _comparison_row("genetic_baseline", 0, True, 0.8),
            _comparison_row("genetic_baseline", 1, True, 1.9),
        ]
    )

    comparisons = paired_comparisons(raw).set_index("baseline_method")

    uniform = comparisons.loc["uniform_random_without_replacement"]
    assert uniform["paired_seeds"] == 2
    assert uniform["paired_success_delta_sum"] == 1
    assert np.isclose(uniform["paired_success_delta_mean"], 0.5)
    assert uniform["best_cost_diff_mean_online_minus_baseline"] < 0.0
    assert uniform["best_cost_online_wins"] == 2
    assert bool(uniform["best_cost_diff_negative_favors_online"])

    genetic = comparisons.loc["genetic_baseline"]
    assert genetic["paired_success_delta_sum"] == -1
    assert genetic["best_cost_diff_mean_online_minus_baseline"] > 0.0
    assert genetic["best_cost_online_losses"] == 2


def _comparison_row(method: str, seed: int, success: bool, best_cost: float) -> dict[str, object]:
    return {
        "evidence_tier": "unit",
        "case_id": "tiny",
        "n_segments": 3,
        "active_windows": 2,
        "M": 3,
        "budget_label": "sqrtM",
        "budget": 2,
        "method": method,
        "seed": seed,
        "success_global_optimum": success,
        "best_cost": best_cost,
    }


def test_threshold_search_does_not_require_top_k_mask_input() -> None:
    costs = np.array([0.3, 0.1, 0.4, 0.2])
    result = online_threshold_search(costs, budget=8, seed=7)

    assert result.method == "online_threshold_amplitude_amplification"
    assert 0 <= result.best_index < costs.size
    assert result.best_cost <= result.initial_cost


def test_query_accounting_is_monotone_and_bounded() -> None:
    costs = np.array([0.9, 0.7, 0.1, 0.4, 0.2])
    result = online_threshold_search(costs, budget=9, seed=3)

    assert result.total_oracle_calls == result.budget
    assert result.cost_evaluations == result.incumbent_evaluations + result.candidate_evaluations
    assert result.total_oracle_calls == result.cost_evaluations + result.threshold_oracle_calls
    assert result.incumbent_evaluations == 1
    assert result.best_cost <= result.initial_cost


def test_threshold_search_consumes_remaining_budget_by_capping_iterations() -> None:
    costs = np.array([0.8, 0.6, 0.4, 0.2, 0.0])

    for budget in range(1, 12):
        result = online_threshold_search(costs, budget=budget, seed=5, max_grover_window=10)
        assert result.total_oracle_calls == budget
        assert result.total_oracle_calls == result.cost_evaluations + result.threshold_oracle_calls


def test_four_state_fixed_seed_can_improve_over_initial_incumbent() -> None:
    costs = np.array([0.0, 0.4, 0.8, 1.0])
    result = online_threshold_search(costs, budget=6, seed=0)

    assert result.initial_index != result.global_best_index
    assert result.best_cost < result.initial_cost
    assert result.hit_top1


def test_baselines_are_deterministic_with_fixed_seed() -> None:
    costs = np.array([0.6, 0.2, 0.5, 0.1, 0.4])

    first_uniform = uniform_random_baseline(costs, budget=3, seed=11)
    second_uniform = uniform_random_baseline(costs, budget=3, seed=11)
    assert first_uniform == second_uniform

    scores = np.array([0.5, 0.1, 0.3, 0.2, 0.4])
    first_ranked = ranked_baseline(costs, budget=3, seed=13, scores=scores)
    second_ranked = ranked_baseline(costs, budget=3, seed=13, scores=scores)
    assert first_ranked == second_ranked

    schedules = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    for baseline in (
        simulated_annealing_baseline,
        cross_entropy_baseline,
        genetic_baseline,
    ):
        first = baseline(costs, budget=8, seed=17, schedules=schedules)
        second = baseline(costs, budget=8, seed=17, schedules=schedules)
        assert first == second


def test_baselines_count_strict_running_incumbent_updates() -> None:
    costs = np.array([5.0, 2.0, 4.0, 1.0, 3.0])

    uniform = uniform_random_baseline(costs, budget=5, seed=13)
    assert uniform.initial_index == 0
    assert uniform.accepted_updates == 2

    scores = np.arange(costs.size, dtype=float)
    ranked = ranked_baseline(costs, budget=5, seed=0, scores=scores)
    assert ranked.initial_index == 0
    assert ranked.accepted_updates == 2


def test_deployable_baselines_respect_evaluation_budget() -> None:
    costs = np.array([0.9, 0.6, 0.4, 0.2, 0.3, 0.1])
    schedules = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    for baseline in (
        uniform_random_baseline,
        simulated_annealing_baseline,
        cross_entropy_baseline,
        genetic_baseline,
    ):
        if baseline is uniform_random_baseline:
            result = baseline(costs, budget=4, seed=23)
        else:
            result = baseline(costs, budget=4, seed=23, schedules=schedules)
        assert result.cost_evaluations <= result.budget
        assert result.total_oracle_calls == result.cost_evaluations
        assert result.threshold_oracle_calls == 0
        assert result.cost_evaluations == result.incumbent_evaluations + result.candidate_evaluations
        assert result.best_cost <= result.initial_cost


def test_schedule_aware_baselines_handle_full_budget_accounting() -> None:
    costs = np.array([0.8, 0.7, 0.2, 0.6, 0.4, 0.1])
    schedules = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    for baseline in (
        simulated_annealing_baseline,
        cross_entropy_baseline,
        genetic_baseline,
    ):
        result = baseline(costs, budget=9, seed=5, schedules=schedules)
        assert result.cost_evaluations == 9
        assert result.total_oracle_calls == 9
        assert 0 <= result.initial_index < costs.size
        assert 0 <= result.best_index < costs.size
