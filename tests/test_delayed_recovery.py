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

from qlt.cr3bp import propagate_ballistic
from qlt.objective import ObjectiveConfig, outage_masks
import qlt.delayed_recovery as delayed


def small_cfg(**overrides) -> ObjectiveConfig:
    params = {
        "mu": 0.01215058560962404,
        "tf": 0.08,
        "n_segments": 4,
        "substeps": 2,
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


def test_delayed_target_and_total_duration_semantics_are_exposed():
    cfg = small_cfg(n_segments=4, tf=0.08, substeps=2)
    _, target = state_pair()
    horizon = 2
    dt = cfg.tf / cfg.n_segments
    expected_target = propagate_ballistic(target, cfg.mu, horizon * dt, cfg.substeps * horizon)

    assert delayed.nominal_segment_duration(cfg) == pytest.approx(0.02)
    assert delayed.delayed_target_time(cfg, horizon) == pytest.approx(0.04)
    assert delayed.branch_total_duration(cfg, horizon) == pytest.approx(0.12)
    assert np.allclose(delayed.delayed_target_state(target, cfg, horizon), expected_target)


def test_selected_outages_zero_runs_no_branch_optimizer(monkeypatch):
    cfg = small_cfg(n_segments=3)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    nominal_error = delayed.terminal_error_for_duration(state0, target, cfg, controls, cfg.tf)

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

    def fail_branch_optimizer(**kwargs):
        raise AssertionError("branch optimizer should not run when selected_outages=0")

    monkeypatch.setattr(delayed, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(delayed, "optimize_delayed_locked_recovery_branch", fail_branch_optimizer)

    result = delayed.run_delayed_locked_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        recovery_horizon_segments=2,
        selected_outages=0,
        nominal_max_nfev=1,
        branch_max_nfev=1,
    )

    assert result["selected_outage_indices"] == []
    assert result["branch_results"] == []
    assert result["branch_nfev"] == []
    assert result["total_branch_nfev"] == 0
    assert result["selected_worst_error"] == pytest.approx(result["nominal_delayed_coast_error"])
    assert result["nominal_optimizer_success"] is True
    assert result["branch_optimizer_ran"] is False
    assert result["branch_optimizer_success"] is False
    assert result["branch_optimizer_all_success"] is False
    assert result["optimizer_success"] is False
    assert result["branch_optimizer_success_by_branch"] == []
    assert result["branch_optimizer_ran_by_branch"] == []
    assert result["branch_portfolio_enabled"] is False
    assert result["branch_portfolio_variant_count"] == 1
    assert result["branch_portfolio_all_success"] is False
    assert result["branch_control_count"] == cfg.n_segments + 2


def test_branch_controls_include_horizon_controls_and_enforce_bounds():
    cfg = small_cfg(n_segments=4, tf=0.1, amax=0.05)
    state0, target = state_pair()
    nominal = np.array(
        [
            [0.20, 0.00, 0.00],
            [0.03, 0.01, 0.00],
            [0.01, -0.02, 0.00],
            [0.00, 0.01, -0.02],
        ],
        dtype=float,
    )
    bounded_nominal = delayed.project_controls_to_ball(nominal, cfg.amax)
    mask = np.ones(cfg.n_segments, dtype=float)
    mask[1] = 0.0
    horizon = 2

    result = delayed.optimize_delayed_locked_recovery_branch(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal,
        mask=mask,
        mask_index=1,
        recovery_horizon_segments=horizon,
        max_nfev=0,
        weights={"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
    )

    full = np.asarray(result["branch_controls"], dtype=float)
    start = delayed.outage_end(mask)
    assert full.shape == (cfg.n_segments + horizon, 3)
    assert result["branch_control_count"] == cfg.n_segments + horizon
    assert result["recovery_segments"] == cfg.n_segments + horizon - start
    assert result["optimizer_ran"] is False
    assert result["optimizer_success"] is False
    assert np.allclose(full[:start], bounded_nominal[:start] * mask[:start, None])
    assert np.max(np.linalg.norm(full, axis=1)) <= cfg.amax + 1e-12
    assert result["control_bound_violation"] == 0.0
    assert result["branch_total_duration"] == pytest.approx((cfg.n_segments + horizon) * cfg.tf / cfg.n_segments)


def test_branch_recovery_reports_initial_candidate_without_conflating_optimizer(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    nominal = np.zeros((cfg.n_segments, 3), dtype=float)
    mask = np.ones(cfg.n_segments, dtype=float)
    mask[0] = 0.0
    horizon = 1
    recovery_segments = cfg.n_segments + horizon - delayed.outage_end(mask)

    class FakeOptimizerResult:
        x = np.zeros(recovery_segments * 3, dtype=float)
        success = False
        message = "fixture nfev cap"
        cost = 123.0
        optimality = 456.0
        nfev = 7

    monkeypatch.setattr(delayed, "least_squares", lambda *args, **kwargs: FakeOptimizerResult())

    result = delayed.optimize_delayed_locked_recovery_branch(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=nominal,
        mask=mask,
        mask_index=0,
        recovery_horizon_segments=horizon,
        max_nfev=10,
        weights={"weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
    )

    recovery = np.zeros((recovery_segments, 3), dtype=float)
    full = delayed.delayed_branch_full_controls(nominal, mask, recovery, cfg.amax, horizon)
    target_delayed = delayed.delayed_target_state(target, cfg, horizon)
    final, _ = delayed.propagate_projected_controls_for_duration(
        state0,
        full,
        cfg,
        delayed.branch_total_duration(cfg, horizon),
    )
    residual = delayed.scaled_state_residual(final, target_delayed, cfg)
    expected_cost = 0.5 * float(np.dot(residual, residual))
    assert result["accepted_candidate"] == "initial"
    assert result["cost"] == pytest.approx(expected_cost)
    assert result["accepted_cost"] == pytest.approx(expected_cost)
    assert np.isnan(result["optimality"])
    assert np.isnan(result["accepted_optimality"])
    assert result["optimizer_cost"] == pytest.approx(123.0)
    assert result["optimizer_optimality"] == pytest.approx(456.0)
    assert result["optimizer_ran"] is True
    assert result["optimizer_success"] is False
    assert result["nfev"] == 7


def test_branch_weight_portfolio_charges_all_variants_and_reports_accepted(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    nominal_error = delayed.terminal_error_for_duration(state0, target, cfg, controls, cfg.tf)
    horizon = 1

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": nominal_error,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 2,
            "runtime_seconds": 0.01,
        }

    def fake_branch_optimizer(**kwargs):
        weights = kwargs["weights"]
        if weights.control > 0.0:
            terminal_error = 2.0
            fuel = 0.10
            nfev = 3
            runtime = 0.2
            success = True
            message = "regularized fixture"
        else:
            terminal_error = 0.5
            fuel = 0.30
            nfev = 5
            runtime = 0.4
            success = False
            message = "terminal-only fixture nfev cap"
        total_segments = kwargs["cfg"].n_segments + int(kwargs["recovery_horizon_segments"])
        return {
            "mask_index": int(kwargs["mask_index"]),
            "recovery_start": delayed.outage_end(kwargs["mask"]),
            "recovery_segments": total_segments - delayed.outage_end(kwargs["mask"]),
            "recovery_horizon_segments": int(kwargs["recovery_horizon_segments"]),
            "branch_control_count": total_segments,
            "nominal_dt": kwargs["cfg"].tf / kwargs["cfg"].n_segments,
            "delayed_target_time": kwargs["recovery_horizon_segments"] * kwargs["cfg"].tf / kwargs["cfg"].n_segments,
            "branch_total_duration": delayed.branch_total_duration(kwargs["cfg"], kwargs["recovery_horizon_segments"]),
            "terminal_error": terminal_error,
            "error": terminal_error,
            "branch_fuel": fuel,
            "nfev": nfev,
            "runtime_seconds": runtime,
            "optimizer_ran": True,
            "optimizer_success": success,
            "accepted_candidate": "optimizer",
            "message": message,
            "cost": terminal_error,
            "optimality": 0.0,
            "branch_weights": weights.as_dict(),
            "branch_controls": np.zeros((total_segments, 3), dtype=float),
            "recovery_controls": np.zeros((total_segments - delayed.outage_end(kwargs["mask"]), 3), dtype=float),
            "history": None,
            "control_max_norm": 0.0,
            "control_bound_violation": 0.0,
        }

    monkeypatch.setattr(delayed, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(delayed, "optimize_delayed_locked_recovery_branch", fake_branch_optimizer)

    result = delayed.run_delayed_locked_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        recovery_horizon_segments=horizon,
        selected_outages=1,
        nominal_max_nfev=1,
        branch_max_nfev=9,
        branch_weight_variants=[
            {
                "label": "regularized_001",
                "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0},
            },
            {
                "label": "terminal_only",
                "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
            },
        ],
    )

    assert result["branch_portfolio_enabled"] is True
    assert result["branch_portfolio_variant_count"] == 2
    assert result["branch_portfolio_variant_labels"] == ["regularized_001", "terminal_only"]
    assert result["branch_nfev"] == [8]
    assert result["total_branch_nfev"] == 8
    assert result["nfev"] == 10
    assert result["branch_runtime_seconds"] == [pytest.approx(0.6)]
    assert result["branch_accepted_weight_variant_labels"] == ["regularized_001"]
    assert result["branch_accepted_variant_nfev"] == [3]
    assert result["branch_optimizer_success_by_branch"] == [True]
    assert result["branch_portfolio_all_success"] is True
    assert result["branch_portfolio_converged_threshold_feasible_candidate_counts"] == [1]
    assert "preferred optimizer_success=True" in result["portfolio_acceptance_rule"]
    branch_summary = result["branch_results"][0]
    assert branch_summary["accepted_branch_weight_variant_label"] == "regularized_001"
    assert branch_summary["portfolio_converged_threshold_feasible_candidate_count"] == 1
    candidates = branch_summary["branch_portfolio_candidate_results"]
    assert [candidate["variant_label"] for candidate in candidates] == ["regularized_001", "terminal_only"]
    assert [candidate["nfev"] for candidate in candidates] == [3, 5]
    assert [candidate["converged_threshold_feasible"] for candidate in candidates] == [True, False]


def test_select_delayed_outage_policy_int_all_single_and_all_configured():
    cfg = small_cfg(n_segments=4, outage_lengths=(1, 2), tf=0.12, amax=0.08)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    nominal_controls = np.array(
        [
            [0.02, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.00, 0.00, -0.04],
            [0.01, -0.02, 0.01],
        ],
        dtype=float,
    )
    horizon = 2
    delayed_target = delayed.delayed_target_state(target, cfg, horizon)

    selected, hardness, semantics = delayed.select_delayed_outage_indices(
        policy=2,
        state0=state0,
        delayed_target=delayed_target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        recovery_horizon_segments=horizon,
        min_recovery_segments=1,
    )
    expected = np.argsort(hardness)[::-1][:2]
    assert np.array_equal(selected, expected)
    assert hardness.shape == (masks.shape[0],)
    assert np.all(np.isfinite(hardness))
    assert "delayed target" in semantics

    selected_single, hardness_single, semantics_single = delayed.select_delayed_outage_indices(
        policy="all_single",
        state0=state0,
        delayed_target=delayed_target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        recovery_horizon_segments=horizon,
    )
    assert np.array_equal(selected_single, np.arange(cfg.n_segments))
    assert np.array_equal(hardness_single, hardness)
    assert np.all(np.sum(masks[selected_single] < 0.5, axis=1) == 1)
    assert "one-segment" in semantics_single

    selected_all, hardness_all, semantics_all = delayed.select_delayed_outage_indices(
        policy="all_configured",
        state0=state0,
        delayed_target=delayed_target,
        cfg=cfg,
        nominal_controls=nominal_controls,
        masks=masks,
        recovery_horizon_segments=horizon,
    )
    assert np.array_equal(selected_all, np.arange(masks.shape[0]))
    assert np.array_equal(hardness_all, hardness)
    assert "all configured" in semantics_all


def test_script_artifact_generation_with_fake_backend(monkeypatch, tmp_path):
    script = load_script_module(
        "run_delayed_locked_recovery_test",
        ROOT / "scripts" / "run_delayed_locked_recovery.py",
    )
    config = {
        "run": {"label": "delayed_test", "output_subdir": "delayed_test"},
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
        "delayed_recovery": {
            "recovery_horizon_segments": 2,
            "nominal": {"max_nfev": 1, "node_initialization": "linear", "node_initialization_blend": 0.5},
            "branch": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "cases": [
                {
                    "case_id": "delayed_unit",
                    "purpose": "unit test",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 3,
                    "recovery_horizon_segments": 2,
                    "selected_outages": 0,
                    "nominal_max_nfev": 1,
                    "branch_max_nfev": 0,
                }
            ],
        },
    }
    config_path = tmp_path / "delayed_config.yaml"
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
        horizon = int(kwargs["recovery_horizon_segments"])
        controls = np.zeros((cfg.n_segments, 3), dtype=float)
        return {
            "success": True,
            "backend_success": True,
            "mode": "delayed_arrival_locked_nominal_independent_branch_recovery",
            "method_type": "delayed_arrival_locked_nominal_independent_branch_recovery",
            "optimizer_success": False,
            "optimizer_success_semantics": "fixture overall optimizer success",
            "nominal_optimizer_success": True,
            "nominal_backend_success": True,
            "branch_optimizer_success": False,
            "branch_optimizer_all_success": False,
            "branch_optimizer_ran": False,
            "branch_portfolio_enabled": False,
            "branch_portfolio_variant_count": 1,
            "branch_portfolio_variant_labels": ["configured"],
            "branch_portfolio_all_success": False,
            "portfolio_acceptance_rule": "fixture no selected branch portfolio candidates",
            "branch_portfolio_converged_threshold_feasible_candidate_counts": [],
            "branch_weight_variants": [
                {
                    "label": "configured",
                    "index": 0,
                    "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
                }
            ],
            "message": "fixture ok",
            "nominal_message": "fixture nominal ok",
            "nominal_accepted_candidate": "fixture",
            "nominal_error": 0.01,
            "nominal_original_error": 0.01,
            "nominal_baseline_error": 0.01,
            "nominal_lock_error_delta": 0.0,
            "nominal_delayed_coast_error": 0.012,
            "recovery_horizon_segments": horizon,
            "nominal_dt": cfg.tf / cfg.n_segments,
            "delayed_target_time": horizon * cfg.tf / cfg.n_segments,
            "branch_total_duration": (cfg.n_segments + horizon) * cfg.tf / cfg.n_segments,
            "branch_control_count": cfg.n_segments + horizon,
            "original_target_state": np.ones(6, dtype=float) * 0.01,
            "delayed_target_state": np.ones(6, dtype=float) * 0.02,
            "worst_error_semantics": "fixture worst",
            "worst_error": 0.012,
            "selected_recovery_worst_error": 0.012,
            "selected_worst_error": 0.012,
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
            "branch_optimizer_success_by_branch": [],
            "branch_optimizer_ran_by_branch": [],
            "branch_accepted_weight_variant_labels": [],
            "branch_accepted_weight_variant_indices": [],
            "branch_accepted_weights": [],
            "branch_accepted_variant_nfev": [],
            "branch_accepted_variant_runtime_seconds": [],
            "branch_recovery_starts": [],
            "branch_recovery_segments": [],
            "branch_control_counts": [],
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
            "delayed_target_semantics": "fixture delayed target",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)
    monkeypatch.setattr(script, "run_delayed_locked_recovery_baseline", fake_backend)

    args = script.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )
    df = script.run(args)

    results_dir = tmp_path / "data" / "results" / "delayed_test"
    tables_dir = tmp_path / "tables" / "delayed_test"
    figures_dir = tmp_path / "figures" / "delayed_test"

    assert len(df) == 1
    assert (results_dir / "delayed_locked_recovery.csv").exists()
    assert (results_dir / "delayed_locked_recovery_metadata.json").exists()
    assert (tables_dir / "delayed_locked_recovery_table.tex").exists()
    assert (figures_dir / "delayed_locked_recovery.png").exists()
    assert (figures_dir / "delayed_locked_recovery.pdf").exists()

    csv_df = pd.read_csv(results_dir / "delayed_locked_recovery.csv")
    assert csv_df.iloc[0]["recovery_horizon_segments"] == 2
    assert csv_df.iloc[0]["branch_control_count"] == 5
    assert bool(csv_df.iloc[0]["branch_optimizer_ran"]) is False
    assert bool(csv_df.iloc[0]["branch_optimizer_all_success"]) is False
    assert bool(csv_df.iloc[0]["branch_portfolio_enabled"]) is False
    assert csv_df.iloc[0]["branch_portfolio_variant_count"] == 1
    assert json.loads(csv_df.iloc[0]["branch_portfolio_variant_labels"]) == ["configured"]
    assert "fixture" in csv_df.iloc[0]["portfolio_acceptance_rule"]
    assert json.loads(csv_df.iloc[0]["branch_results"]) == []
    metadata = json.loads((results_dir / "delayed_locked_recovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_count"] == 1
    assert "implementation_identities" in metadata
    assert "delayed_recovery_module" in metadata["implementation_identities"]
    assert metadata["branch_weight_variants_by_case"]["delayed_unit"][0]["label"] == "configured"
    assert "Delayed-arrival" in metadata["semantics"]["backend"]


def test_script_effective_settings_include_branch_weight_variants(tmp_path):
    script = load_script_module(
        "run_delayed_locked_recovery_settings_test",
        ROOT / "scripts" / "run_delayed_locked_recovery.py",
    )
    config = {
        "run": {"label": "delayed_test", "output_subdir": "delayed_test"},
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
        "delayed_recovery": {
            "branch": {
                "weight_variants": [
                    {
                        "label": "regularized_001",
                        "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0},
                    },
                    {
                        "label": "terminal_only",
                        "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
                    },
                ]
            },
            "cases": [
                {
                    "case_id": "portfolio_settings",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 3,
                    "recovery_horizon_segments": 2,
                    "selected_outages": 0,
                }
            ],
        },
    }
    args = SimpleNamespace(source_states=tmp_path / "missing_source_states.json")
    case = script._suite_cases(config)[0]
    settings = script._effective_settings(config, args, case)
    variants = settings["branch_weight_variants"]

    assert [variant["label"] for variant in variants] == ["regularized_001", "terminal_only"]
    assert variants[1]["weights"]["control"] == pytest.approx(0.0)

    changed = json.loads(json.dumps(config))
    changed["delayed_recovery"]["branch"]["weight_variants"][0]["weights"]["control"] = 0.02
    changed_case = script._suite_cases(changed)[0]
    changed_settings = script._effective_settings(changed, args, changed_case)
    assert script.settings_fingerprint(settings) != script.settings_fingerprint(changed_settings)
