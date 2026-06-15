from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.objective import ObjectiveConfig, outage_masks
import qlt.locked_recovery as locked


def small_cfg(**overrides) -> ObjectiveConfig:
    params = {
        "mu": 0.01215058560962404,
        "tf": 0.08,
        "n_segments": 4,
        "substeps": 1,
        "amax": 0.05,
        "kr": 0.1,
        "kv": 0.1,
        "position_scale": 1.0,
        "velocity_scale": 1.0,
        "weights": {
            "nominal": 1.0,
            "robust_worst": 0.85,
            "robust_degradation": 0.55,
            "active_fraction": 0.08,
            "smoothness": 0.04,
        },
        "outage_lengths": (1,),
    }
    params.update(overrides)
    return ObjectiveConfig(**params)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def state_pair():
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([1.01, 0.015, 0.018, 0.005, 0.035, -0.015], dtype=float)
    return state0, target


def test_selected_outages_zero_locks_nominal_and_runs_no_branches(monkeypatch):
    cfg = small_cfg(n_segments=3)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    nominal_error = locked.terminal_error(state0, target, cfg, controls)

    def fake_multiple_shooting(**kwargs):
        assert kwargs["selected_outages"] == 0
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": nominal_error,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
        }

    monkeypatch.setattr(locked, "run_multiple_shooting_baseline", fake_multiple_shooting)

    result = locked.run_locked_nominal_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        selected_outages=0,
        nominal_max_nfev=1,
        branch_max_nfev=1,
    )

    assert result["selected_outage_indices"] == []
    assert result["branch_results"] == []
    assert result["branch_nfev"] == []
    assert result["selected_worst_error"] == pytest.approx(result["nominal_error"])
    assert result["selected_recovery_worst_error"] == pytest.approx(result["nominal_error"])
    assert result["optimizer_success"] is True


def test_branch_recovery_keeps_nominal_controls_unchanged_and_bounded():
    cfg = small_cfg(n_segments=4, tf=0.1, amax=0.05)
    state0, target = state_pair()
    nominal = np.array(
        [
            [0.02, 0.00, 0.00],
            [0.03, 0.01, 0.00],
            [0.01, -0.02, 0.00],
            [0.00, 0.01, -0.02],
        ],
        dtype=float,
    )
    original = nominal.copy()
    mask = np.ones(cfg.n_segments, dtype=float)
    mask[1] = 0.0

    result = locked.optimize_locked_recovery_branch(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal,
        mask=mask,
        mask_index=1,
        max_nfev=2,
        weights={"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
    )

    assert np.allclose(nominal, original)
    full = np.asarray(result["branch_controls"], dtype=float)
    start = locked.outage_end(mask)
    assert np.allclose(full[:start], original[:start] * mask[:start, None])
    assert np.max(np.linalg.norm(full, axis=1)) <= cfg.amax + 1e-12
    assert result["control_bound_violation"] == 0.0


def test_branch_recovery_reports_initial_candidate_cost(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    nominal = np.zeros((cfg.n_segments, 3), dtype=float)
    mask = np.ones(cfg.n_segments, dtype=float)
    mask[0] = 0.0

    class FakeOptimizerResult:
        x = np.zeros((cfg.n_segments - 1) * 3, dtype=float)
        success = False
        message = "fixture optimizer was worse"
        cost = 123.0
        optimality = 456.0
        nfev = 7

    monkeypatch.setattr(locked, "least_squares", lambda *args, **kwargs: FakeOptimizerResult())

    result = locked.optimize_locked_recovery_branch(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal,
        mask=mask,
        mask_index=0,
        max_nfev=10,
        weights={"weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
    )

    full = locked.branch_full_controls(nominal, mask, np.zeros((cfg.n_segments - 1, 3), dtype=float), cfg.amax)
    final, _ = locked.propagate_projected_controls(state0, full, cfg)
    residual = locked.scaled_state_residual(final, target, cfg)
    expected_cost = 0.5 * float(np.dot(residual, residual))
    assert result["accepted_candidate"] == "initial"
    assert result["cost"] == pytest.approx(expected_cost)
    assert result["accepted_cost"] == pytest.approx(expected_cost)
    assert np.isnan(result["optimality"])
    assert np.isnan(result["accepted_optimality"])
    assert result["optimizer_cost"] == pytest.approx(123.0)
    assert result["optimizer_optimality"] == pytest.approx(456.0)
    assert result["optimizer_success"] is False
    assert result["nfev"] == 7


def test_selected_outage_policy_parsing_int_all_single_and_all_configured():
    cfg = small_cfg(n_segments=4, outage_lengths=(1, 2))
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)

    selected, hardness, semantics = locked.select_outage_indices(
        policy=2,
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=controls,
        masks=masks,
        min_recovery_segments=1,
    )
    assert len(selected) == 2
    assert hardness.shape == (masks.shape[0],)
    assert "hardest" in semantics

    selected_single, _, semantics_single = locked.select_outage_indices(
        policy="all_single",
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=controls,
        masks=masks,
    )
    assert len(selected_single) == cfg.n_segments
    assert np.all(np.sum(masks[selected_single] < 0.5, axis=1) == 1)
    assert "one-segment" in semantics_single

    selected_all, _, semantics_all = locked.select_outage_indices(
        policy="all_configured",
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=controls,
        masks=masks,
    )
    assert np.array_equal(selected_all, np.arange(masks.shape[0]))
    assert "all configured" in semantics_all

    assert locked.selected_outage_count_for_policy("all", masks) == masks.shape[0]


def test_script_artifact_generation_with_fake_backend(monkeypatch, tmp_path):
    script = load_script_module(
        "run_locked_nominal_recovery_test",
        ROOT / "scripts" / "run_locked_nominal_recovery.py",
    )
    config = {
        "run": {"label": "locked_test", "output_subdir": "locked_test"},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_dro_phase",
            "transfer_time": 0.1,
            "segments": 3,
            "substeps_per_segment": 1,
            "amax": 0.2,
            "steering": {"kr": 0.75, "kv": 1.45},
        },
        "objective": {
            "position_scale": 1.0,
            "velocity_scale": 0.35,
            "weights": {
                "nominal": 1.0,
                "robust_worst": 0.85,
                "robust_degradation": 0.55,
                "active_fraction": 0.08,
                "smoothness": 0.04,
            },
            "thresholds": {"nominal_success": 0.09, "robust_success": 0.17},
        },
        "outages": {"block_lengths": [1]},
        "locked_recovery": {
            "nominal": {"max_nfev": 1, "node_initialization": "linear", "node_initialization_blend": 0.5},
            "branch": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "cases": [
                {
                    "case_id": "locked_unit",
                    "purpose": "unit test",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 3,
                    "selected_outages": 0,
                    "nominal_max_nfev": 1,
                    "branch_max_nfev": 0,
                }
            ],
        },
    }
    config_path = tmp_path / "locked_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=np.zeros(6, dtype=float),
            target=np.ones(6, dtype=float) * 0.01,
            target_metadata={"target_state_generation": "fixture"},
        )

    def fake_backend(**kwargs):
        cfg = kwargs["cfg"]
        controls = np.zeros((cfg.n_segments, 3), dtype=float)
        return {
            "success": True,
            "backend_success": True,
            "mode": "locked_nominal_independent_branch_recovery",
            "method_type": "locked_nominal_independent_branch_recovery",
            "optimizer_success": True,
            "nominal_optimizer_success": True,
            "nominal_backend_success": True,
            "message": "fixture ok",
            "nominal_message": "fixture nominal ok",
            "nominal_accepted_candidate": "fixture",
            "nominal_error": 0.01,
            "nominal_baseline_error": 0.01,
            "nominal_lock_error_delta": 0.0,
            "selected_recovery_worst_error": 0.01,
            "selected_worst_error": 0.01,
            "all_outage_worst_error": 0.2,
            "all_mask_worst_error": 0.2,
            "nominal_threshold": 0.09,
            "selected_recovery_threshold": 0.17,
            "selected_worst_threshold": 0.17,
            "meets_nominal_threshold": True,
            "meets_selected_recovery_threshold": True,
            "meets_selected_worst_threshold": True,
            "meets_thresholds": True,
            "nominal_fuel": 0.0,
            "recovery_fuel_mean": 0.0,
            "recovery_fuel_max": 0.0,
            "control_max_norm": 0.0,
            "control_bound_violation": 0.0,
            "controls": controls,
            "nominal_controls": controls,
            "selected_outage_indices": [],
            "selected_outage_errors": [],
            "all_outage_errors": [0.2, 0.18, 0.16],
            "nominal_masked_outage_errors": [0.2, 0.18, 0.16],
            "branch_results": [],
            "branch_nfev": [],
            "branch_runtime_seconds": [],
            "branch_optimizer_success": [],
            "branch_recovery_starts": [],
            "branch_recovery_segments": [],
            "total_branch_nfev": 0,
            "total_branch_runtime_seconds": 0.0,
            "nominal_nfev": 1,
            "nominal_runtime_seconds": 0.01,
            "nfev": 1,
            "runtime_seconds": 0.02,
            "cost": 0.0,
            "optimality": 0.0,
            "nominal_cost": 0.0,
            "nominal_optimality": 0.0,
            "backend_semantics": "fixture backend",
            "selection_semantics": "fixture selection",
            "selected_branch_semantics": "fixture selected",
            "all_mask_diagnostic_semantics": "fixture all mask",
            "control_bound_semantics": "fixture bound",
            "nominal_lock_semantics": "fixture lock",
            "worst_error_semantics": "fixture worst",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)
    monkeypatch.setattr(script, "run_locked_nominal_recovery_baseline", fake_backend)

    args = script.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )
    df = script.run(args)

    results_dir = tmp_path / "data" / "results" / "locked_test"
    tables_dir = tmp_path / "tables" / "locked_test"
    figures_dir = tmp_path / "figures" / "locked_test"

    assert len(df) == 1
    assert (results_dir / "locked_nominal_recovery.csv").exists()
    assert (results_dir / "locked_nominal_recovery_metadata.json").exists()
    assert (tables_dir / "locked_nominal_recovery_table.tex").exists()
    assert (figures_dir / "locked_nominal_recovery.png").exists()
    assert (figures_dir / "locked_nominal_recovery.pdf").exists()

    csv_df = pd.read_csv(results_dir / "locked_nominal_recovery.csv")
    assert csv_df.iloc[0]["selected_worst_error"] == pytest.approx(0.01)
    assert json.loads(csv_df.iloc[0]["branch_results"]) == []
    metadata = json.loads((results_dir / "locked_nominal_recovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_count"] == 1
    assert "Locked-nominal" in metadata["semantics"]["backend"]
