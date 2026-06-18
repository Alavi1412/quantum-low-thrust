from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_online_branch_refinement import (
    config_for_case,
    parse_case_id,
    select_top_screening_rows,
    serialize_refinement_row,
    threshold_dimensional_bounds,
    write_latex_table,
)


def test_parse_and_configure_feedback_case() -> None:
    n_segments, active_count = parse_case_id("feedback_N20_k16")

    assert n_segments == 20
    assert active_count == 16

    config = {
        "benchmark": {"segments": 12},
        "objective": {"target_active_fraction": 11 / 12},
    }
    prepared, n_case, k_case = config_for_case(config, "feedback_N16_k12")

    assert n_case == 16
    assert k_case == 12
    assert prepared["benchmark"]["segments"] == 16
    assert np.isclose(prepared["objective"]["target_active_fraction"], 0.75)
    assert config["benchmark"]["segments"] == 12


def test_select_top_screening_rows_is_stable_and_tracks_source_index() -> None:
    data = pd.DataFrame(
        {
            "schedule": ["1110", "1101", "1011", "0111"],
            "feedback_objective": [0.4, 0.2, 0.2, 0.3],
            "feedback_nominal_error": [0.1, 0.1, 0.1, 0.1],
            "feedback_worst_error": [0.5, 0.4, 0.3, 0.2],
        }
    )
    data["_source_index"] = range(len(data))

    selected = select_top_screening_rows(data, 2)

    assert selected["schedule"].to_list() == ["1011", "1101"]
    assert selected["screening_rank"].to_list() == [1, 2]
    assert selected["_source_index"].to_list() == [2, 1]


def test_serialize_refinement_row_preserves_screening_and_branch_metrics() -> None:
    candidate = pd.Series(
        {
            "screening_rank": 1,
            "_source_index": 42,
            "schedule": "0000111111111111",
            "feedback_objective": 0.7,
            "feedback_nominal_error": 0.33,
            "feedback_worst_error": 0.35,
        }
    )
    refined = {
        "nominal_error": 0.134,
        "selected_worst_error": 0.154,
        "all_mask_worst_error": 0.158,
        "nominal_fuel": 0.075,
        "optimizer_success": False,
        "success": False,
        "nfev": 35,
        "best_attempt_nfev": 35,
        "selected_outage_indices": [9],
        "selected_outage_errors": [0.154],
        "runtime_seconds": 68.0,
    }

    row = serialize_refinement_row(
        case_id="feedback_N16_k12",
        source_path=Path("data/results/online_quantum_search/feedback_N16_k12_subspace_costs.csv"),
        candidate=candidate,
        m_size=1820,
        active_count=12,
        screening_fuel=0.08,
        refined=refined,
        thresholds={"nominal_success": 0.09, "robust_success": 0.17},
    )

    assert row["case_id"] == "feedback_N16_k12"
    assert row["subspace_index"] == 42
    assert row["active_count"] == 12
    assert row["M"] == 1820
    assert row["branch_nominal_error"] == 0.134
    assert row["branch_selected_worst_error"] == 0.154
    assert row["optimizer_success"] is False
    assert row["success"] is False
    assert row["selected_outage_indices"] == "[9]"


def test_write_latex_table_contains_branch_columns(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "case_id": "feedback_N16_k12",
                "M": 1820,
                "schedule_bits": "0000111111111111",
                "screening_objective": 0.7,
                "screening_nominal_error": 0.33,
                "screening_robust_worst_error": 0.35,
                "branch_nominal_error": 0.134,
                "branch_selected_worst_error": 0.154,
                "branch_all_mask_worst_error": 0.158,
                "branch_nominal_fuel": 0.075,
                "optimizer_success": False,
                "success": False,
                "nfev": 35,
                "selected_outage_indices": "[9]",
            }
        ]
    )

    table_path = write_latex_table(rows, tmp_path)

    assert table_path.is_file()
    text = table_path.read_text(encoding="utf-8")
    assert "Branch nom." in text
    assert "All-mask worst" in text
    assert "feedback\\_N16\\_k12" in text


def test_threshold_dimensional_bounds_are_component_upper_bounds() -> None:
    rows = threshold_dimensional_bounds(
        {
            "objective": {
                "position_scale": 1.0,
                "velocity_scale": 0.35,
                "thresholds": {"nominal_success": 0.09, "robust_success": 0.17},
            }
        }
    )

    nominal = rows[0]
    robust = rows[1]
    assert nominal["threshold"] == "nominal"
    assert np.isclose(nominal["position_bound_km_if_velocity_error_zero"], 34596.0)
    assert np.isclose(nominal["velocity_bound_m_per_s_if_position_error_zero"], 32.282, atol=0.01)
    assert robust["position_bound_km_if_velocity_error_zero"] > nominal["position_bound_km_if_velocity_error_zero"]
