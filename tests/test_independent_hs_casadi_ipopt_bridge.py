from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results" / "independent_hs_casadi_ipopt_bridge"
CSV_PATH = RESULTS_DIR / "independent_hs_casadi_ipopt_bridge.csv"
METADATA_PATH = RESULTS_DIR / "independent_hs_casadi_ipopt_bridge_metadata.json"
TABLE_PATH = ROOT / "tables" / "independent_hs_casadi_ipopt_bridge" / "independent_hs_casadi_ipopt_bridge_table.tex"


def test_casadi_ipopt_smoke_solves_tiny_nlp_when_available():
    ca = pytest.importorskip("casadi")

    x = ca.MX.sym("x")
    nlp = {"x": x, "f": (x - 1.0) ** 2, "g": ca.vertcat(x)}
    solver = ca.nlpsol(
        "casadi_ipopt_bridge_smoke",
        "ipopt",
        nlp,
        {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 20,
        },
    )
    solution = solver(x0=[0.0], lbx=[-2.0], ubx=[2.0], lbg=[-2.0], ubg=[2.0])

    assert solver.stats()["success"] is True
    assert float(solution["x"]) == pytest.approx(1.0, abs=1.0e-8)


def test_independent_hs_casadi_ipopt_bridge_artifacts_are_scoped_and_stable():
    assert CSV_PATH.is_file()
    assert METADATA_PATH.is_file()
    assert TABLE_PATH.is_file()

    df = pd.read_csv(CSV_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    summary = metadata["polish_case_summary"]

    assert len(df) == 9
    assert int((df["record_type"] == "nominal").sum()) == 1
    assert int((df["record_type"] == "branch").sum()) == 8
    assert int(summary["row_count"]) == 9
    assert int(summary["nominal_row_count"]) == 1
    assert int(summary["branch_row_count"]) == 8
    assert int(summary["ipopt_success_count"]) == 9
    assert int(summary["bridge_pass_count"]) == 9
    assert summary["all_ipopt_success"] is True
    assert summary["all_rows_pass_configured_threshold"] is True

    assert float(summary["source_nominal_replay_terminal_error"]) == pytest.approx(
        0.011333095366088189, abs=1.0e-12
    )
    assert float(summary["source_branch_replay_worst_terminal_error"]) == pytest.approx(
        0.07792080291839382, abs=1.0e-12
    )
    assert float(summary["ipopt_refined_nominal_terminal_error"]) == pytest.approx(
        0.009138565365046585, abs=1.0e-10
    )
    assert float(summary["ipopt_refined_branch_worst_terminal_error"]) == pytest.approx(
        0.015534969964216154, abs=1.0e-10
    )
    assert float(summary["max_branch_prefix_control_delta_from_refined_nominal_masked"]) <= 1.0e-12
    assert int(summary["zero_variable_branch_count"]) == 1
    assert float(summary["max_control_bound_violation"]) <= 1.0e-10
    assert float(summary["max_control_norm"]) <= 0.2 + 1.0e-9

    assert metadata["casadi_refinement"] is True
    assert metadata["optimization_rerun"] is True
    assert metadata["production_solver_parity_claim"] is True
    assert "mature NLP backend bridge check" in metadata["production_solver_parity_claim_scope"]
    for flag_name in (
        "high_fidelity_validation",
        "high_fidelity_flight_validation",
        "fuel_optimality_claim",
        "doi_claim",
        "quantum_advantage_claim",
    ):
        assert metadata[flag_name] is False
        assert not df[flag_name].astype(bool).any()

    assert df["casadi_refinement"].astype(bool).all()
    assert df["optimization_rerun"].astype(bool).all()
    assert df["production_solver_parity_claim"].astype(bool).all()
    assert df["ipopt_success"].astype(bool).all()
    assert df["ipopt_refined_passes_configured_threshold"].astype(bool).all()
    assert df["bridge_passes_configured_threshold"].astype(bool).all()
    assert df["max_control_bound_violation"].astype(float).max() <= 1.0e-10
    assert set(df["configured_threshold"].round(2)) == {0.09, 0.17}

    branch_masks = df.loc[df["record_type"] == "branch", "active_control_mask"].map(json.loads)
    assert len(branch_masks) == 8
    assert all(len(mask) == 8 for mask in branch_masks)
    assert all(sum(int(value) == 0 for value in mask) == 1 for mask in branch_masks)

    sidecars = metadata["artifacts"]["casadi_ipopt_bridge_control_sidecars"]
    assert len(sidecars) == 9
    sidecar_payloads = []
    for sidecar in sidecars:
        path = ROOT / sidecar["path"]
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        sidecar_payloads.append(payload)
        assert payload["casadi_refinement"] is True
        assert payload["optimization_rerun"] is True
        assert payload["production_solver_parity_claim"] is True
        assert payload["high_fidelity_validation"] is False
        assert payload["fuel_optimality_claim"] is False
        assert payload["quantum_advantage_claim"] is False

    nominal_payload = next(payload for payload in sidecar_payloads if payload["record_type"] == "nominal")
    nominal_endpoint = np.asarray(nominal_payload["refined_endpoint_controls"], dtype=float)
    nominal_midpoint = np.asarray(nominal_payload["refined_midpoint_controls"], dtype=float)
    branch_payloads = {
        (int(payload["branch_order"]), int(payload["mask_index"])): payload
        for payload in sidecar_payloads
        if payload["record_type"] == "branch"
    }
    assert len(branch_payloads) == 8

    branch_rows = df[df["record_type"] == "branch"]
    for row in branch_rows.to_dict(orient="records"):
        outage_mask = [int(value) for value in json.loads(row["outage_mask"])]
        active_mask = [int(value) for value in json.loads(row["active_control_mask"])]
        variable_mask = [int(value) for value in json.loads(row["variable_control_mask"])]
        recovery_start = int(row["recovery_start"])
        expected_variable_mask = [0] * len(outage_mask)
        expected_variable_mask[recovery_start:] = outage_mask[recovery_start:]

        assert active_mask == outage_mask
        assert variable_mask == expected_variable_mask
        assert float(row["max_prefix_control_delta_from_refined_nominal_masked"]) <= 1.0e-12

        payload = branch_payloads[(int(row["branch_order"]), int(row["mask_index"]))]
        assert payload["recovery_start"] == recovery_start
        assert payload["active_control_mask"] == outage_mask
        assert payload["variable_control_mask"] == expected_variable_mask
        assert payload["max_prefix_control_delta_from_refined_nominal_masked"] <= 1.0e-12

        branch_endpoint = np.asarray(payload["refined_endpoint_controls"], dtype=float)
        branch_midpoint = np.asarray(payload["refined_midpoint_controls"], dtype=float)
        mask_array = np.asarray(outage_mask, dtype=float)[:, None]
        np.testing.assert_allclose(
            branch_endpoint[:recovery_start],
            nominal_endpoint[:recovery_start] * mask_array[:recovery_start],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            branch_midpoint[:recovery_start],
            nominal_midpoint[:recovery_start] * mask_array[:recovery_start],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            branch_endpoint[np.asarray(outage_mask, dtype=int) == 0],
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            branch_midpoint[np.asarray(outage_mask, dtype=int) == 0],
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        )

    assert int((branch_rows["variable_control_mask"].map(json.loads).map(sum) == 0).sum()) == 1

    table_text = TABLE_PATH.read_text(encoding="utf-8")
    assert "Scoped CasADi/IPOPT bridge summary" in table_text
