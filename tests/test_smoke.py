from __future__ import annotations

import importlib.util
import copy
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.cr3bp import load_source_states, propagate_ballistic, teacher_control_schedule
import qlt.experiment as experiment
import qlt.feasibility as feasibility_module
import qlt.multiple_shooting as multiple_shooting_module
from qlt.feasibility import FEASIBILITY_COLUMNS, run_case
from qlt.experiment import clean_generated_outputs, make_objective_config, run_metadata_extra, run_seed
from qlt.objective import Evaluator, ObjectiveConfig, outage_masks
from qlt.refinement import multistart_control_guesses, project_controls_to_ball, refine_schedule
from qlt.multiple_shooting import MULTIPLE_SHOOTING_COLUMNS, run_case as run_multiple_shooting_case
import qlt.reporting as reporting_module
from qlt.reporting import summarize, write_latex_tables, write_metadata
from qlt.solvers import SolverResult, optimize_qaoa_angles, qaoa_expected_energy, random_qaoa_angles, solve_genetic, solve_qaoa
from qlt.surrogate import QuboModel, fit_qubo


def small_objective_config(**overrides) -> ObjectiveConfig:
    params = {
        "mu": 0.01215058560962404,
        "tf": 0.1,
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


def multiple_shooting_fixture_row(**overrides):
    row = {
        "transfer_time": 4.0,
        "amax": 0.2,
        "segments": 14,
        "max_nfev": 120,
        "substeps_per_segment": 7,
        "selected_outages": 3,
        "node_initialization": "linear",
        "node_initialization_blend": 0.5,
        "nominal_first_homotopy": False,
        "schedule": "11111111111111",
        "base_nominal_error": 1.0,
        "base_worst_error": 2.0,
        "nominal_error": 0.1,
        "selected_recovery_worst_error": 0.2,
        "selected_worst_error": 0.2,
        "all_outage_worst_error": 0.3,
        "all_mask_worst_error": 0.3,
        "nominal_threshold": 0.09,
        "selected_recovery_threshold": 0.17,
        "selected_worst_threshold": 0.17,
        "meets_nominal_threshold": False,
        "meets_selected_recovery_threshold": False,
        "meets_selected_worst_threshold": False,
        "meets_thresholds": False,
        "optimizer_success": False,
        "multiple_shooting_success": False,
        "cost": 1.0,
        "optimality": 0.5,
        "nfev": 120,
        "runtime_seconds": 0.01,
        "control_max_norm": 0.2,
        "accepted_candidate": "optimizer",
        "warm_start_refinement_nfev": 0,
        "warm_start_nominal_error": np.nan,
        "warm_start_selected_worst_error": np.nan,
        "nominal_first_nfev": 0,
        "nominal_first_nominal_error": np.nan,
        "nominal_first_selected_worst_error": np.nan,
        "nominal_first_all_mask_worst_error": np.nan,
        "nominal_fuel": 0.8,
        "recovery_fuel_mean": 0.7,
        "recovery_fuel_max": 0.75,
        "selected_outage_indices": "[0]",
        "selected_outage_errors": "[0.2]",
        "all_outage_errors": "[0.3]",
        "message": "fixture",
    }
    row.update(overrides)
    return multiple_shooting_module.normalize_multiple_shooting_row(row)


def robust_margin_fixture_config(output_subdir="robust_margin_test", groups=None):
    if groups is None:
        groups = {
            "thrust_margin_selected": {
                "purpose": "test selected branch",
                "phase_times": [0.2],
                "transfer_times": [0.5],
                "amax": [0.2],
                "segments": [4],
                "selected_outages": [1],
                "max_nfev": [1],
                "min_recovery_segments": 1,
            },
            "all_single_outage_margin": {
                "purpose": "test all single outages",
                "phase_times": [0.3],
                "transfer_times": [0.5],
                "amax": [0.3],
                "segments": [3],
                "selected_outages": "all_single_outage_masks",
                "max_nfev": [1],
                "min_recovery_segments": 1,
            },
        }
    return {
        "run": {"label": "robust_margin_test", "output_subdir": output_subdir},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_halo_phase_shift",
            "transfer_time": 0.5,
            "phase_time": 0.2,
            "segments": 4,
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
        "suite": {
            "residual_weights": {
                "initial_weight": 10.0,
                "defect_weight": 1.0,
                "terminal_weight": 4.0,
                "branch_terminal_weight": 4.0,
                "branch_start_weight": 5.0,
                "control_weight": 0.01,
                "smooth_weight": 0.012,
            },
            "groups": groups,
        },
        "refinement": {
            "mode": "multiple_shooting_branch_recovery",
            "solver_mode": "bounded_projected_multiple_shooting",
            "max_nfev": 1,
            "selected_outages": 1,
            "outage_selection_min_recovery_segments": 1,
            "node_initialization": "linear",
            "node_initialization_blend": 0.5,
            "nominal_first_homotopy": False,
            "nominal_first_nfev": 1,
        },
    }


def fake_robust_margin_backend_row(base_config, transfer_time, amax, segments, max_nfev, args):
    settings = multiple_shooting_module.settings_values_for_case(
        base_config,
        transfer_time,
        amax,
        segments,
        max_nfev,
        args,
    )
    selected = int(args.selected_outages)
    errors = [0.02 + 0.001 * index for index in range(selected)]
    all_errors = [0.03 + 0.001 * index for index in range(int(segments))]
    row = multiple_shooting_fixture_row(
        **settings,
        substeps_per_segment=int(base_config["benchmark"]["substeps_per_segment"]),
        selected_outage_indices=json.dumps(list(range(selected))),
        selected_outage_errors=json.dumps(errors),
        all_outage_errors=json.dumps(all_errors),
        nominal_error=0.01 + float(amax) * 0.01,
        selected_recovery_worst_error=max(errors) if errors else 0.0,
        selected_worst_error=max(errors) if errors else 0.0,
        all_outage_worst_error=max(all_errors) if all_errors else 0.0,
        all_mask_worst_error=max(all_errors) if all_errors else 0.0,
        nominal_threshold=float(base_config["objective"]["thresholds"]["nominal_success"]),
        selected_recovery_threshold=float(base_config["objective"]["thresholds"]["robust_success"]),
        selected_worst_threshold=float(base_config["objective"]["thresholds"]["robust_success"]),
        meets_nominal_threshold=True,
        meets_selected_recovery_threshold=True,
        meets_selected_worst_threshold=True,
        meets_thresholds=True,
        optimizer_success=True,
        multiple_shooting_success=True,
        nfev=int(max_nfev),
        runtime_seconds=0.01,
        control_max_norm=float(amax) * 0.5,
        control_bound_violation=0.0,
        accepted_candidate="fixture",
        message="fixture ok",
    )
    row["settings_fingerprint"] = multiple_shooting_module._settings_fingerprint(settings)
    return row


def continuation_margin_fixture_config(output_subdir="continuation_margin_test", groups=None):
    if groups is None:
        groups = {
            "phase_continuation_all_single": {
                "purpose": "test continuation",
                "outage_lengths": [1],
                "cases": [
                    {
                        "case_id": "source_p03",
                        "phase_time": 0.3,
                        "transfer_time": 0.5,
                        "amax": 0.3,
                        "segments": 3,
                        "selected_outages": "all_configured_outage_masks",
                        "max_nfev": 2,
                        "warm_start_kind": "cold",
                    },
                    {
                        "case_id": "warm_p02",
                        "phase_time": 0.2,
                        "transfer_time": 0.5,
                        "amax": 0.3,
                        "segments": 3,
                        "selected_outages": "all_configured_outage_masks",
                        "max_nfev": 2,
                        "warm_start_kind": "nominal_controls",
                        "warm_start_from_case_id": "source_p03",
                    },
                ],
            }
        }
    return {
        "run": {"label": "continuation_margin_test", "output_subdir": output_subdir},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_halo_phase_shift",
            "transfer_time": 0.5,
            "phase_time": 0.3,
            "segments": 3,
            "substeps_per_segment": 1,
            "amax": 0.3,
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
        "suite": {
            "residual_weights": {
                "initial_weight": 10.0,
                "defect_weight": 1.0,
                "terminal_weight": 4.0,
                "branch_terminal_weight": 4.0,
                "branch_start_weight": 5.0,
                "control_weight": 0.01,
                "smooth_weight": 0.012,
            },
            "groups": groups,
        },
        "refinement": {
            "mode": "multiple_shooting_branch_recovery",
            "solver_mode": "bounded_projected_multiple_shooting",
            "max_nfev": 2,
            "selected_outages": "all_configured_outage_masks",
            "outage_selection_min_recovery_segments": 1,
            "node_initialization": "linear",
            "node_initialization_blend": 0.5,
        },
    }


def fake_continuation_states(root, config, source_states):
    del root, config, source_states
    return SimpleNamespace(
        mu=0.01215058560962404,
        initial=np.zeros(6, dtype=float),
        target=np.ones(6, dtype=float) * 0.01,
        target_metadata={"target_state_generation": "fixture"},
    )


def fake_continuation_backend(calls):
    def _fake_backend(
        *,
        state0,
        target,
        cfg,
        masks,
        thresholds,
        selected_outages,
        max_nfev,
        min_recovery_segments,
        residual_weights,
        nominal_control_guess,
        selected_branch_control_guesses,
        node_initialization,
        node_initialization_blend,
        warm_start_info,
    ):
        del state0, target, thresholds, max_nfev, min_recovery_segments, residual_weights
        del selected_branch_control_guesses, node_initialization, node_initialization_blend, warm_start_info
        guess = None if nominal_control_guess is None else np.asarray(nominal_control_guess, dtype=float).copy()
        calls.append({"guess": guess, "selected_outages": int(selected_outages), "mask_count": int(masks.shape[0])})
        if guess is None:
            controls = np.full((cfg.n_segments, 3), 0.01 * len(calls), dtype=float)
        else:
            controls = guess + 0.001
        selected_errors = [0.02 + 0.001 * index for index in range(int(selected_outages))]
        all_errors = [0.03 + 0.001 * index for index in range(int(masks.shape[0]))]
        return {
            "success": True,
            "mode": "multiple_shooting_branch_recovery",
            "solver_mode": "bounded_projected_multiple_shooting",
            "optimizer_success": True,
            "message": "fixture ok",
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": int(len(calls)),
            "runtime_seconds": 0.01,
            "nominal_error": 0.01,
            "selected_recovery_worst_error": max(selected_errors) if selected_errors else 0.01,
            "selected_worst_error": max(selected_errors) if selected_errors else 0.01,
            "all_outage_worst_error": max(all_errors) if all_errors else 0.01,
            "all_mask_worst_error": max(all_errors) if all_errors else 0.01,
            "nominal_fuel": 0.01,
            "recovery_fuel_mean": 0.01,
            "recovery_fuel_max": 0.01,
            "control_max_norm": float(np.linalg.norm(controls, axis=1).max()),
            "control_bound_violation": 0.0,
            "controls": controls,
            "accepted_candidate": "fixture",
            "selected_outage_indices": list(range(int(selected_outages))),
            "selected_outage_errors": selected_errors,
            "all_outage_errors": all_errors,
        }

    return _fake_backend


def test_multiple_shooting_node_initialization_modes_endpoint_behavior():
    cfg = small_objective_config(n_segments=4, tf=0.2, amax=0.04)
    state0 = np.array([1.05, 0.02, -0.01, 0.0, 0.08, 0.02], dtype=float)
    target = np.array([0.88, -0.04, 0.03, -0.03, 0.02, -0.01], dtype=float)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    selected_masks = outage_masks(cfg.n_segments, cfg.outage_lengths)[:1]

    rollout_layout, rollout_vec = multiple_shooting_module._initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        nominal_control_guess=controls,
    )
    explicit_layout, explicit_vec = multiple_shooting_module._initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        nominal_control_guess=controls,
        node_initialization="rollout",
    )
    assert rollout_layout.size == explicit_layout.size
    assert np.allclose(rollout_vec, explicit_vec)

    linear_layout, linear_vec = multiple_shooting_module._initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        nominal_control_guess=controls,
        node_initialization="linear",
    )
    nominal_nodes, _, branch_nodes, _ = linear_layout.unpack(linear_vec)
    assert np.allclose(nominal_nodes[0], state0)
    assert np.allclose(nominal_nodes[-1], target)
    assert np.allclose(branch_nodes[0][-1], target)

    branch_start = multiple_shooting_module._rollout_nodes(
        state0,
        controls * selected_masks[0, :, None],
        cfg,
    )[multiple_shooting_module._outage_end(selected_masks[0])]
    assert np.allclose(branch_nodes[0][0], branch_start)


def test_multiple_shooting_blended_node_initialization_matches_weighted_nodes():
    cfg = small_objective_config(n_segments=3, tf=0.15, amax=0.04)
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.95, -0.02, 0.01, -0.01, 0.02, 0.03], dtype=float)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    blend = 0.25

    layout, vec = multiple_shooting_module._initial_guess(
        state0,
        target,
        cfg,
        np.zeros((0, cfg.n_segments), dtype=float),
        nominal_control_guess=controls,
        node_initialization="blend",
        node_initialization_blend=blend,
    )
    nominal_nodes, _, _, _ = layout.unpack(vec)
    rollout_nodes = multiple_shooting_module._rollout_nodes(state0, controls, cfg)
    linear_nodes = multiple_shooting_module._linear_nodes(state0, target, cfg.n_segments + 1)
    assert np.allclose(nominal_nodes, (1.0 - blend) * rollout_nodes + blend * linear_nodes)


def test_bounded_multiple_shooting_residual_projects_controls_before_propagation(monkeypatch):
    cfg = small_objective_config(n_segments=3, tf=0.15, amax=0.05)
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.95, -0.02, 0.01, -0.01, 0.02, 0.03], dtype=float)
    masks = outage_masks(cfg.n_segments, (1,))
    selected = np.array([0], dtype=int)
    selected_masks = masks[selected]
    layout, x0 = multiple_shooting_module._initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        nominal_control_guess=np.zeros((cfg.n_segments, 3), dtype=float),
    )
    raw = x0.copy()
    raw[layout.nominal_controls] = 2.0
    for control_slice in layout.branch_controls:
        raw[control_slice] = -3.0

    propagated_controls = []

    def fake_propagate_segment(state, control, used_cfg):
        assert used_cfg is cfg
        propagated_controls.append(np.asarray(control, dtype=float).copy())
        return np.asarray(state, dtype=float)

    monkeypatch.setattr(multiple_shooting_module, "_propagate_segment", fake_propagate_segment)
    problem = multiple_shooting_module._BoundedMultipleShootingProblem(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        selected=selected,
        selected_masks=selected_masks,
        layout=layout,
        weights={
            "initial": 10.0,
            "defect": 1.0,
            "terminal": 4.0,
            "branch_terminal": 4.0,
            "branch_start": 5.0,
            "control": 0.01,
            "smooth": 0.01,
        },
    )

    residual = problem.residual(raw)
    evaluated = problem.evaluate_vector(raw)
    residual_diagnostics = problem.residual_control_diagnostics(raw)

    assert np.all(np.isfinite(residual))
    assert propagated_controls
    assert max(np.linalg.norm(control) for control in propagated_controls) <= cfg.amax + 1e-12
    assert residual_diagnostics["residual_control_max_norm"] <= cfg.amax + 1e-12
    assert evaluated["control_max_norm"] <= cfg.amax + 1e-12
    assert evaluated["control_bound_violation"] == 0.0


def test_cr3bp_objective_and_qubo_smoke():
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    states = load_source_states(ROOT / "data" / "source_states.json", config["benchmark"]["transfer_time"], substeps=64)
    cfg = make_objective_config(config, states.mu)
    evaluator = Evaluator(states.initial, states.target, cfg)
    schedules = np.array(
        [
            np.zeros(cfg.n_segments, dtype=int),
            np.ones(cfg.n_segments, dtype=int),
            np.arange(cfg.n_segments) % 2,
            (np.arange(cfg.n_segments) % 3 == 0).astype(int),
            (np.arange(cfg.n_segments) % 4 < 2).astype(int),
            (np.arange(cfg.n_segments) > cfg.n_segments // 2).astype(int),
        ]
    )
    metrics = [evaluator.evaluate(s) for s in schedules]
    objectives = np.array([m["objective"] for m in metrics])
    assert np.all(np.isfinite(objectives))
    model, _ = fit_qubo(schedules, objectives, validation_fraction=0.34, seed=123)
    pred = model.energy(schedules)
    assert pred.shape == objectives.shape
    assert np.all(np.isfinite(pred))


def test_active_target_objective_prior_is_disabled_by_default():
    cfg = small_objective_config()
    state0 = np.array([0.8, 0.0, 0.0, 0.0, 0.2, 0.0])
    target = state0 + np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0])
    schedule = np.array([1, 1, 0, 0], dtype=int)

    metrics = Evaluator(state0, target, cfg).evaluate(schedule)
    expected = (
        cfg.weights["nominal"] * metrics["nominal_error"]
        + cfg.weights["robust_worst"] * metrics["worst_error"]
        + cfg.weights["robust_degradation"] * metrics["robust_degradation"]
        + cfg.weights["active_fraction"] * metrics["active_fraction"]
        + cfg.weights["smoothness"] * metrics["smoothness"]
    )

    assert cfg.target_active_fraction is None
    assert cfg.target_active_weight == 0.0
    assert metrics["active_target_penalty"] == 0.0
    assert np.isclose(metrics["objective"], expected)


def test_active_target_objective_prior_adds_squared_deviation_penalty():
    state0 = np.array([0.8, 0.0, 0.0, 0.0, 0.2, 0.0])
    target = state0 + np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.0])
    schedule = np.array([1, 1, 1, 0], dtype=int)
    base_cfg = small_objective_config()
    prior_cfg = small_objective_config(target_active_fraction=0.5, target_active_weight=8.0)

    base_metrics = Evaluator(state0, target, base_cfg).evaluate(schedule)
    prior_metrics = Evaluator(state0, target, prior_cfg).evaluate(schedule)
    expected_penalty = 8.0 * (0.75 - 0.5) ** 2

    assert np.isclose(prior_metrics["active_target_penalty"], expected_penalty)
    assert prior_metrics["target_active_fraction"] == 0.5
    assert prior_metrics["target_active_weight"] == 8.0
    assert np.isclose(prior_metrics["active_target_deviation"], 0.25)
    assert np.isclose(prior_metrics["objective"] - base_metrics["objective"], expected_penalty)


def test_objective_config_reads_active_target_prior_and_metadata():
    config = yaml.safe_load((ROOT / "configs" / "q1_phase_shift.yaml").read_text(encoding="utf-8"))
    config["objective"]["target_active_fraction"] = 11.0 / 12.0
    config["objective"]["weights"]["active_target"] = 12.0

    cfg = make_objective_config(config, config["benchmark"]["mu"])
    states = SimpleNamespace(
        target=np.zeros(6),
        target_mode="catalog_halo_phase_shift",
        target_metadata={"target_state_generation": "unit-test"},
    )
    metadata = run_metadata_extra(config, states)

    assert cfg.target_active_fraction == 11.0 / 12.0
    assert cfg.target_active_weight == 12.0
    assert metadata["objective_prior"] == {
        "target_active_fraction": 11.0 / 12.0,
        "target_active_weight": 12.0,
        "active_target_enabled": True,
    }


def test_qubo_ranking_diagnostics_are_finite_and_monotonic():
    schedules = np.array(list(np.ndindex((2, 2, 2, 2))), dtype=int)
    objectives = schedules @ np.array([1.0, 2.0, 4.0, 8.0])

    model, _ = fit_qubo(schedules, objectives, ridge=1e-8, validation_fraction=0.34, seed=1)
    diagnostics = model.diagnostics

    for key in ["val_spearman", "val_pairwise_order_accuracy", "val_top_k_recall"]:
        assert np.isfinite(diagnostics[key])
    assert diagnostics["val_spearman"] > 0.99
    assert diagnostics["val_pairwise_order_accuracy"] == 1.0
    assert diagnostics["val_top_k_recall"] == 1.0


def test_teacher_controlled_target_generation_records_schedule_and_bounds():
    controls, metadata = teacher_control_schedule(
        n_segments=16,
        amax=0.065,
        teacher_cfg={"active_windows": [5, 6, 7, 8, 10, 12, 14], "amax_fraction": 0.6},
    )

    assert controls.shape == (16, 3)
    assert metadata["active_schedule"] == "0000011110101010"
    assert metadata["active_windows_zero_based"] == [5, 6, 7, 8, 10, 12, 14]
    assert np.max(np.linalg.norm(controls, axis=1)) <= 0.065 + 1e-12
    assert 0.0 < metadata["max_control_norm"] <= 0.065


def test_load_source_states_teacher_mode_generates_metadata():
    states = load_source_states(
        ROOT / "data" / "source_states.json",
        transfer_time=3.0,
        substeps=4,
        target_mode="teacher_controlled",
        segments=8,
        amax=0.05,
        teacher={"active_windows": [2, 3, 5, 7], "amax_fraction": 0.5},
    )

    assert states.target_mode == "teacher_controlled"
    assert states.teacher_controls is not None
    assert states.teacher_controls.shape == (8, 3)
    assert np.max(np.linalg.norm(states.teacher_controls, axis=1)) <= 0.05 + 1e-12
    teacher = states.target_metadata["teacher"]
    assert teacher["active_schedule"] == "00110101"
    assert teacher["fuel"] > 0.0
    assert np.allclose(np.asarray(teacher["target_state"], dtype=float), states.target)


def test_load_source_states_catalog_halo_phase_shift_records_non_teacher_metadata():
    states = load_source_states(
        ROOT / "data" / "source_states.json",
        transfer_time=0.5,
        substeps=4,
        target_mode="catalog_halo_phase_shift",
        segments=8,
        amax=0.2,
        phase_time=0.3,
    )

    expected = propagate_ballistic(states.initial, states.mu, 0.3, 4)
    metadata = states.target_metadata

    assert states.target_mode == "catalog_halo_phase_shift"
    assert states.teacher_controls is None
    assert np.allclose(states.target, expected)
    assert metadata["source_state"] == "initial_nrho_like_l2_southern_halo"
    assert metadata["phase_time"] == 0.3
    assert metadata["generation_substeps"] == 4
    assert metadata["teacher_generated"] is False
    assert "teacher" not in metadata
    assert "teacher" not in metadata["target_state_generation"].lower()


def test_catalog_halo_phase_shift_is_not_zero_thrust_trivial_and_all_windows_recovers():
    config = yaml.safe_load((ROOT / "configs" / "q1_phase_shift.yaml").read_text(encoding="utf-8"))
    states = experiment.load_configured_states(ROOT, config)
    cfg = make_objective_config(config, states.mu)
    thresholds = config["objective"]["thresholds"]
    evaluator = Evaluator(states.initial, states.target, cfg)

    zero_metrics = evaluator.evaluate(np.zeros(cfg.n_segments, dtype=int))
    assert zero_metrics["nominal_error"] > thresholds["nominal_success"]

    smoke_refine_cfg = dict(config["refinement"])
    smoke_refine_cfg["max_nfev"] = 5
    schedule = np.ones(cfg.n_segments, dtype=int)
    refined = refine_schedule(
        schedule,
        states.initial,
        states.target,
        cfg,
        outage_masks(cfg.n_segments, cfg.outage_lengths),
        smoke_refine_cfg,
        thresholds,
    )

    assert refined["nominal_error"] <= thresholds["nominal_success"]
    assert refined["selected_worst_error"] <= thresholds["robust_success"]
    assert refined["all_mask_worst_error"] <= thresholds["robust_success"]


def test_run_metadata_extra_records_teacher_target_details():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    config["run"]["label"] = "metadata_teacher_test"
    config["benchmark"]["target_mode"] = "teacher_controlled"
    config["benchmark"]["teacher"] = {"active_windows": [2, 3, 5, 7], "amax_fraction": 0.5}
    config["benchmark"]["segments"] = 8
    states = load_source_states(
        ROOT / "data" / "source_states.json",
        transfer_time=float(config["benchmark"]["transfer_time"]),
        substeps=int(config["benchmark"]["substeps_per_segment"]),
        target_mode="teacher_controlled",
        segments=int(config["benchmark"]["segments"]),
        amax=float(config["benchmark"]["amax"]),
        teacher=config["benchmark"]["teacher"],
    )

    metadata = run_metadata_extra(config, states)

    assert metadata["benchmark_label"] == "metadata_teacher_test"
    assert metadata["target_mode"] == "teacher_controlled"
    assert metadata["target_metadata"]["teacher"]["active_schedule"] == "00110101"
    assert metadata["target_metadata"]["teacher"]["controls"]


def test_write_metadata_keeps_manifest_anchor_when_git_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(reporting_module, "_git_output", lambda args: None)

    metadata_path = tmp_path / "run_metadata.json"
    write_metadata(metadata_path, "unit-test", {"run": {"label": "metadata"}})

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["git_head"] is None
    assert data["git_status_short"] is None
    assert data["tracked_file_count"] is None
    assert data["untracked_file_count"] is None
    assert data["project_manifest_file_count"] > 0
    assert len(data["project_manifest_hash"]) == 64


def test_regeneration_scripts_respect_output_subdir_config():
    plot_results = load_script_module("plot_results_test", ROOT / "scripts" / "plot_results.py")
    generate_tables = load_script_module("generate_tables_test", ROOT / "scripts" / "generate_tables.py")

    config_path = ROOT / "configs" / "q1_teacher_feasible.yaml"
    plot_results_dir, plot_figures_dir, plot_tables_dir = plot_results.configured_output_paths(config_path)
    table_results_dir, table_figures_dir, table_tables_dir = generate_tables.configured_output_paths(config_path)

    assert plot_results_dir == ROOT / "data" / "results" / "teacher_feasible"
    assert plot_figures_dir == ROOT / "figures" / "teacher_feasible"
    assert plot_tables_dir == ROOT / "tables" / "teacher_feasible"
    assert (table_results_dir, table_figures_dir, table_tables_dir) == (plot_results_dir, plot_figures_dir, plot_tables_dir)

    default_results_dir, default_figures_dir, default_tables_dir = plot_results.configured_output_paths(ROOT / "configs" / "default.yaml")
    assert default_results_dir == ROOT / "data" / "results"
    assert default_figures_dir == ROOT / "figures"
    assert default_tables_dir == ROOT / "tables"

    override_results_dir, override_figures_dir, override_tables_dir = generate_tables.configured_output_paths(
        ROOT / "configs" / "default.yaml",
        "teacher_feasible",
    )
    assert override_results_dir == ROOT / "data" / "results" / "teacher_feasible"
    assert override_figures_dir == ROOT / "figures" / "teacher_feasible"
    assert override_tables_dir == ROOT / "tables" / "teacher_feasible"


def test_cardinality_ablation_enumerates_inactive_windows_deterministically():
    ablation = load_script_module("cardinality_ablation_test", ROOT / "scripts" / "run_cardinality_ablation.py")

    schedules = ablation.enumerate_cardinality_schedules(4, 3)
    bits = ["".join(str(int(v)) for v in schedule) for schedule in schedules]

    assert bits == ["0111", "1011", "1101", "1110"]
    assert ablation.coast_windows(schedules[0]) == "0"
    assert ablation.coast_windows(np.ones(4, dtype=int)) == "-"


def test_cardinality_ablation_method_frontier_comparison_counts_exact_schedules():
    ablation = load_script_module("cardinality_ablation_compare_test", ROOT / "scripts" / "run_cardinality_ablation.py")
    one_coast = pd.DataFrame(
        [
            {
                "schedule": "011111111111",
                "refinement_success": True,
                "one_coast_pareto_frontier": True,
                "refined_selected_worst_error": 0.10,
                "refined_nominal_fuel": 0.09,
            },
            {
                "schedule": "101111111111",
                "refinement_success": True,
                "one_coast_pareto_frontier": False,
                "refined_selected_worst_error": 0.104,
                "refined_nominal_fuel": 0.09,
            },
            {
                "schedule": "110111111111",
                "refinement_success": True,
                "one_coast_pareto_frontier": False,
                "refined_selected_worst_error": 0.13,
                "refined_nominal_fuel": 0.09,
            },
        ]
    )
    method_results = pd.DataFrame(
        [
            {
                "method": "surrogate_qubo_sa",
                "normalized_schedule": "011111111111",
                "normalized_active_count": 11,
                "refined_selected_worst_error": 0.11,
                "refined_nominal_fuel": 0.09,
            },
            {
                "method": "surrogate_qubo_sa",
                "normalized_schedule": "101111111111",
                "normalized_active_count": 11,
                "refined_selected_worst_error": 0.12,
                "refined_nominal_fuel": 0.09,
            },
            {
                "method": "surrogate_qubo_sa",
                "normalized_schedule": "111111111111",
                "normalized_active_count": 12,
                "refined_selected_worst_error": 0.08,
                "refined_nominal_fuel": 0.10,
            },
        ]
    )

    comparison = ablation.compare_methods_to_frontier(method_results, one_coast, near_selected_tolerance=0.005)
    row = comparison.loc[comparison["method"] == "surrogate_qubo_sa"].iloc[0]

    assert row["runs"] == 3
    assert row["one_coast_runs"] == 2
    assert row["one_coast_success_runs"] == 2
    assert row["one_coast_frontier_runs"] == 1
    assert row["near_best_selected_runs"] == 2
    assert np.isclose(row["median_exhaustive_selected_worst_error"], 0.102)


def test_refined_controls_respect_euclidean_norm_bound():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    states = load_source_states(ROOT / "data" / "source_states.json", config["benchmark"]["transfer_time"], substeps=64)
    cfg = make_objective_config(config, states.mu)
    schedule = np.ones(cfg.n_segments, dtype=int)
    refine_cfg = dict(config["refinement"])
    refine_cfg["max_nfev"] = 3
    refined = refine_schedule(
        schedule,
        states.initial,
        states.target,
        cfg,
        outage_masks(cfg.n_segments, cfg.outage_lengths),
        refine_cfg,
        config["objective"]["thresholds"],
    )
    controls = refined["controls"]
    assert np.max(np.linalg.norm(controls, axis=1)) <= cfg.amax + 1e-10
    assert refined["mode"] == "branch_recovery"
    for branch in refined["recovery_controls"]:
        branch_controls = np.asarray(branch, dtype=float)
        assert np.max(np.linalg.norm(branch_controls, axis=1)) <= cfg.amax + 1e-10


def test_branch_recovery_uses_zero_thrust_during_missed_segments_and_recovers_after():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    states = load_source_states(ROOT / "data" / "source_states.json", config["benchmark"]["transfer_time"], substeps=64)
    cfg = make_objective_config(config, states.mu)
    schedule = np.ones(cfg.n_segments, dtype=int)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    refine_cfg = dict(config["refinement"])
    refine_cfg["selected_outages"] = 1
    refine_cfg["max_nfev"] = 2

    refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, refine_cfg, config["objective"]["thresholds"])

    assert refined["mode"] == "branch_recovery"
    assert refined["recovery_controls"]
    mask = masks[refined["selected_outage_indices"][0]]
    branch = np.asarray(refined["recovery_controls"][0], dtype=float)
    assert np.allclose(branch[mask < 0.5], 0.0)
    assert any(i >= int(np.flatnonzero(mask < 0.5)[-1] + 1) for i in refined["recovery_indices"][0])


def test_projection_enforces_ball_not_component_box():
    controls = np.array([[1.0, 1.0, 1.0], [0.1, 0.0, 0.0]])
    projected = project_controls_to_ball(controls, 0.5)
    assert np.all(np.linalg.norm(projected, axis=1) <= 0.5 + 1e-12)
    assert np.isclose(np.linalg.norm(projected[0]), 0.5)


def test_multistart_control_guesses_respect_norm_bound():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    states = load_source_states(ROOT / "data" / "source_states.json", config["benchmark"]["transfer_time"], substeps=64)
    cfg = make_objective_config(config, states.mu)
    schedule = np.ones(cfg.n_segments, dtype=int)
    refine_cfg = dict(config["refinement"])
    refine_cfg["multistart"] = {
        "enabled": True,
        "include_feedback": True,
        "include_zero": True,
        "include_bang_bang": True,
        "random_starts": 2,
        "low_amplitude_fraction": 0.4,
        "seed": 999,
    }

    guesses = multistart_control_guesses(schedule, states.initial, states.target, cfg, refine_cfg)

    assert {label for label, _ in guesses}.issuperset({"feedback", "zero", "bang_feedback", "random_low_0"})
    for _, controls in guesses:
        assert controls.shape == (cfg.n_segments, 3)
        assert np.max(np.linalg.norm(controls, axis=1)) <= cfg.amax + 1e-12


def test_teacher_control_initial_guess_can_be_injected_and_projected():
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "robust_worst": 1.0, "robust_degradation": 0.0, "active_fraction": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    schedule = np.array([1, 0, 1], dtype=int)
    teacher_controls = np.array([[1.0, 1.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, -2.0]])
    refine_cfg = {
        "multistart": {"enabled": False, "include_feedback": False},
        "initial_control_guesses": [{"name": "teacher_controls_oracle", "controls": teacher_controls}],
    }

    guesses = multistart_control_guesses(schedule, np.zeros(6), np.ones(6), cfg, refine_cfg)

    assert [label for label, _ in guesses] == ["teacher_controls_oracle"]
    controls = guesses[0][1]
    assert np.allclose(controls[1], 0.0)
    assert np.max(np.linalg.norm(controls, axis=1)) <= cfg.amax + 1e-12


def test_branch_recovery_configured_guess_only_mode_requires_configured_guesses():
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "robust_worst": 1.0, "robust_degradation": 0.0, "active_fraction": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    schedule = np.array([1, 0, 1], dtype=int)
    refine_cfg = {
        "mode": "branch_recovery",
        "use_only_initial_control_guesses": True,
        "initial_control_guesses": [],
    }

    with pytest.raises(ValueError, match="use_only_initial_control_guesses"):
        refine_schedule(
            schedule,
            np.zeros(6),
            np.ones(6),
            cfg,
            outage_masks(cfg.n_segments, cfg.outage_lengths),
            refine_cfg,
            {"nominal_success": 0.1, "robust_success": 0.1},
        )


def test_feasibility_sweep_result_schema():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    args = SimpleNamespace(
        source_states=ROOT / "data" / "source_states.json",
        selected_outages=1,
        min_recovery_segments=2,
        multistart=False,
        random_starts=0,
        low_amplitude_fraction=0.35,
        include_bang_bang=False,
        multistart_seed=123,
        state_residual_weight=None,
        robust_residual_weight=None,
        fuel_residual_weight=None,
        smooth_residual_weight=None,
        control_regularization=None,
    )

    row = run_case(
        config,
        transfer_time=float(config["benchmark"]["transfer_time"]),
        amax=float(config["benchmark"]["amax"]),
        segments=6,
        max_nfev=1,
        args=args,
    )

    assert set(FEASIBILITY_COLUMNS).issubset(row)
    assert row["schedule"] == "111111"
    assert row["nominal_threshold"] == config["objective"]["thresholds"]["nominal_success"]
    assert row["selected_worst_threshold"] == config["objective"]["thresholds"]["robust_success"]
    assert isinstance(row["meets_thresholds"], bool)
    assert row["control_max_norm"] <= row["amax"] + 1e-10


def test_feasibility_run_case_uses_configured_state_loader_and_min_recovery_override(monkeypatch):
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    config["benchmark"]["target_mode"] = "teacher_controlled"
    config["benchmark"]["teacher"] = {"active_windows": [1, 3], "amax_fraction": 0.4}
    config["refinement"]["outage_selection_min_recovery_segments"] = 99
    captured = {}

    def fake_load_configured_states(root, configured, source_states):
        captured["root"] = root
        captured["config"] = configured
        captured["source_states"] = source_states
        return SimpleNamespace(mu=0.01215, initial=np.zeros(6), target=np.ones(6), target_mode=configured["benchmark"]["target_mode"])

    class FakeEvaluator:
        def __init__(self, *args):
            del args

        def evaluate(self, schedule):
            return {"nominal_error": 1.0, "worst_error": 2.0}

    def fake_refine_schedule(schedule, state0, target, cfg, masks, refine_cfg, thresholds):
        del schedule, state0, target, cfg, masks, thresholds
        captured["refine_cfg"] = refine_cfg
        return {
            "nominal_error": 0.1,
            "selected_worst_error": 0.2,
            "all_mask_worst_error": 0.3,
            "optimizer_success": True,
            "success": True,
            "controls": np.zeros((configured_segments, 3)),
        }

    configured_segments = 6
    source_path = ROOT / "data" / "source_states.json"
    args = SimpleNamespace(
        source_states=source_path,
        selected_outages=1,
        min_recovery_segments=2,
        multistart=False,
        random_starts=0,
        low_amplitude_fraction=0.35,
        include_bang_bang=False,
        multistart_seed=123,
        state_residual_weight=None,
        robust_residual_weight=None,
        fuel_residual_weight=None,
        smooth_residual_weight=None,
        control_regularization=None,
    )
    monkeypatch.setattr(feasibility_module, "load_configured_states", fake_load_configured_states)
    monkeypatch.setattr(feasibility_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(feasibility_module, "refine_schedule", fake_refine_schedule)

    row = feasibility_module.run_case(config, 2.5, 0.07, configured_segments, 1, args)

    assert captured["source_states"] == source_path
    assert captured["config"]["benchmark"]["target_mode"] == "teacher_controlled"
    assert captured["config"]["benchmark"]["teacher"] == {"active_windows": [1, 3], "amax_fraction": 0.4}
    assert captured["config"]["benchmark"]["transfer_time"] == 2.5
    assert captured["config"]["benchmark"]["segments"] == configured_segments
    assert captured["refine_cfg"]["outage_selection_min_recovery_segments"] == 2
    assert row["selected_recovery_worst_error"] == row["selected_worst_error"] == 0.2
    assert row["all_outage_worst_error"] == row["all_mask_worst_error"] == 0.3


def test_feasibility_resume_zero_cases_backfills_legacy_csv(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"run": {"label": "legacy_feasibility"}}), encoding="utf-8")
    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "transfer_time": 3.0,
                "amax": 0.065,
                "segments": 14,
                "max_nfev": 120,
                "substeps_per_segment": 7,
                "selected_outages": 3,
                "multistart_enabled": True,
                "random_starts": 1,
                "schedule": "11111111111111",
                "base_nominal_error": 1.0,
                "base_worst_error": 2.0,
                "nominal_error": 0.1,
                "selected_worst_error": 0.2,
                "all_mask_worst_error": 0.3,
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
                "meets_nominal_threshold": False,
                "meets_selected_worst_threshold": False,
                "meets_thresholds": False,
                "optimizer_success": True,
                "refinement_success": False,
                "best_initial_guess": "zero",
                "cost": 1.0,
                "nfev": 2,
                "best_attempt_nfev": 2,
                "runtime_seconds": 0.01,
                "control_max_norm": 0.05,
                "selected_outage_indices": "[0]",
                "selected_outage_errors": "[0.2]",
                "all_outage_errors": "[0.3]",
                "message": "legacy",
            }
        ]
    ).to_csv(results_dir / "feasibility_sweep.csv", index=False)

    def fail_run_case(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume --max-cases 0 must not run new feasibility cases")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(feasibility_module, "run_case", fail_run_case)
    args = SimpleNamespace(
        config=config_path,
        source_states=tmp_path / "source_states.json",
        transfer_times="3.0",
        amax="0.065",
        segments="14",
        max_nfev="120",
        selected_outages=3,
        min_recovery_segments=4,
        multistart=False,
        random_starts=0,
        low_amplitude_fraction=0.35,
        include_bang_bang=False,
        multistart_seed=123,
        state_residual_weight=None,
        robust_residual_weight=None,
        fuel_residual_weight=None,
        smooth_residual_weight=None,
        control_regularization=None,
        max_cases=0,
        resume=True,
    )

    df = feasibility_module.run(args)
    rewritten = pd.read_csv(results_dir / "feasibility_sweep.csv")
    metadata = json.loads((results_dir / "feasibility_metadata.json").read_text(encoding="utf-8"))

    assert df.loc[0, "selected_recovery_worst_error"] == 0.2
    assert df.loc[0, "all_outage_worst_error"] == 0.3
    assert "selected_recovery_worst_error" in rewritten.columns
    assert "all_outage_worst_error" in rewritten.columns
    assert "project_manifest_hash" in metadata
    assert "project_manifest_file_count" in metadata
    assert metadata["feasibility_sweep_rows"] == 1
    assert (tmp_path / "tables" / "feasibility_table.tex").exists()


def test_multiple_shooting_run_case_uses_configured_state_loader(monkeypatch):
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    config["benchmark"]["target_mode"] = "teacher_controlled"
    config["benchmark"]["teacher"] = {"active_windows": [1, 3], "amax_fraction": 0.4}
    captured = {}

    def fake_load_configured_states(root, configured, source_states):
        captured["root"] = root
        captured["config"] = configured
        captured["source_states"] = source_states
        return SimpleNamespace(mu=0.01215, initial=np.zeros(6), target=np.ones(6), target_mode=configured["benchmark"]["target_mode"])

    class FakeEvaluator:
        def __init__(self, *args):
            del args

        def evaluate(self, schedule):
            return {"nominal_error": 1.0, "worst_error": 2.0}

    def fake_multiple_shooting_baseline(**kwargs):
        captured["baseline_kwargs"] = kwargs
        return {
            "nominal_error": 0.1,
            "selected_worst_error": 0.2,
            "all_mask_worst_error": 0.3,
            "optimizer_success": True,
            "success": True,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
            "control_max_norm": 0.0,
            "accepted_candidate": "initial",
            "nominal_fuel": 0.0,
            "recovery_fuel_mean": 0.0,
            "recovery_fuel_max": 0.0,
            "selected_outage_indices": [0],
            "selected_outage_errors": [0.2],
            "all_outage_errors": [0.3],
            "message": "ok",
        }

    source_path = ROOT / "data" / "source_states.json"
    args = SimpleNamespace(
        source_states=source_path,
        selected_outages=1,
        min_recovery_segments=2,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=2.0,
        branch_terminal_weight=2.0,
        branch_start_weight=5.0,
        control_weight=0.01,
        smooth_weight=0.01,
        warm_start_refinement=False,
    )
    monkeypatch.setattr(multiple_shooting_module, "load_configured_states", fake_load_configured_states)
    monkeypatch.setattr(multiple_shooting_module, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(multiple_shooting_module, "run_multiple_shooting_baseline", fake_multiple_shooting_baseline)

    row = multiple_shooting_module.run_case(config, 2.5, 0.07, 6, 1, args)

    assert captured["source_states"] == source_path
    assert captured["config"]["benchmark"]["target_mode"] == "teacher_controlled"
    assert captured["config"]["benchmark"]["teacher"] == {"active_windows": [1, 3], "amax_fraction": 0.4}
    assert captured["config"]["benchmark"]["transfer_time"] == 2.5
    assert captured["config"]["benchmark"]["segments"] == 6
    assert captured["baseline_kwargs"]["min_recovery_segments"] == 2
    assert captured["baseline_kwargs"]["node_initialization"] == "rollout"
    assert captured["baseline_kwargs"]["node_initialization_blend"] == 0.5
    assert row["selected_recovery_worst_error"] == row["selected_worst_error"] == 0.2
    assert row["all_outage_worst_error"] == row["all_mask_worst_error"] == 0.3


def test_multiple_shooting_resume_zero_cases_backfills_legacy_csv(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"label": "legacy_multiple_shooting"},
                "benchmark": {"transfer_time": 3.0, "amax": 0.065, "segments": 14},
                "refinement": {"mode": "branch_recovery", "max_nfev": 90},
            }
        ),
        encoding="utf-8",
    )
    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "transfer_time": 4.0,
                "amax": 0.2,
                "segments": 14,
                "max_nfev": 120,
                "substeps_per_segment": 7,
                "selected_outages": 3,
                "schedule": "11111111111111",
                "base_nominal_error": 1.0,
                "base_worst_error": 2.0,
                "nominal_error": 0.1,
                "selected_worst_error": 0.2,
                "all_mask_worst_error": 0.3,
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
                "meets_nominal_threshold": False,
                "meets_selected_worst_threshold": False,
                "meets_thresholds": False,
                "optimizer_success": False,
                "multiple_shooting_success": False,
                "cost": 1.0,
                "optimality": 0.5,
                "nfev": 120,
                "runtime_seconds": 0.01,
                "control_max_norm": 0.2,
                "nominal_fuel": 0.8,
                "recovery_fuel_mean": 0.7,
                "recovery_fuel_max": 0.75,
                "selected_outage_indices": "[0]",
                "selected_outage_errors": "[0.2]",
                "all_outage_errors": "[0.3]",
                "message": "legacy",
            }
        ]
    ).to_csv(results_dir / "multiple_shooting_feasibility.csv", index=False)

    def fail_run_case(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume --max-cases 0 must not run new multiple-shooting cases")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiple_shooting_module, "run_case", fail_run_case)
    args = SimpleNamespace(
        config=config_path,
        source_states=tmp_path / "source_states.json",
        transfer_times="4.0",
        amax="0.2",
        segments="14",
        max_nfev="120",
        selected_outages=3,
        min_recovery_segments=4,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=4.0,
        branch_terminal_weight=4.0,
        branch_start_weight=5.0,
        control_weight=0.01,
        smooth_weight=0.01,
        warm_start_refinement=False,
        warm_start_nfev=90,
        max_cases=0,
        resume=True,
    )

    df = multiple_shooting_module.run(args)
    rewritten = pd.read_csv(results_dir / "multiple_shooting_feasibility.csv")
    metadata_text = (results_dir / "multiple_shooting_metadata.json").read_text(encoding="utf-8")
    summary_text = (results_dir / "multiple_shooting_feasibility_metadata.json").read_text(encoding="utf-8")
    non_finite_token = r"(?<![\w\"])(?:NaN|-?Infinity)(?![\w\"])"
    metadata = json.loads(metadata_text)
    summary_metadata = json.loads(summary_text)

    assert df.loc[0, "selected_recovery_worst_error"] == 0.2
    assert df.loc[0, "all_outage_worst_error"] == 0.3
    assert df.loc[0, "accepted_candidate"] == "unknown"
    assert re.search(non_finite_token, metadata_text) is None
    assert re.search(non_finite_token, summary_text) is None
    assert summary_metadata["best_cases"][0]["warm_start_nominal_error"] is None
    assert summary_metadata["best_cases"][0]["warm_start_selected_worst_error"] is None
    assert metadata["config"]["benchmark"]["transfer_time"] == 4.0
    assert metadata["config"]["benchmark"]["amax"] == 0.2
    assert metadata["config"]["benchmark"]["segments"] == 14
    assert metadata["config"]["refinement"]["mode"] == "multiple_shooting_branch_recovery"
    assert metadata["config"]["refinement"]["max_nfev"] == 120
    assert metadata["config"]["refinement"]["selected_outages"] == 3
    assert "selected_recovery_worst_error" in rewritten.columns
    assert "all_outage_worst_error" in rewritten.columns
    assert "project_manifest_hash" in metadata
    assert "project_manifest_file_count" in metadata
    assert metadata["multiple_shooting_rows"] == 1
    assert (tmp_path / "tables" / "multiple_shooting_feasibility_table.tex").exists()


def test_multiple_shooting_resume_metadata_uses_mixed_row_settings(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"label": "mixed_multiple_shooting"},
                "benchmark": {"transfer_time": 3.0, "amax": 0.065, "segments": 14},
                "refinement": {
                    "mode": "branch_recovery",
                    "max_nfev": 90,
                    "node_initialization": "rollout",
                    "nominal_first_homotopy": False,
                    "nominal_first_nfev": 90,
                },
            }
        ),
        encoding="utf-8",
    )
    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True)
    rows = [
        multiple_shooting_fixture_row(
            transfer_time=4.0,
            amax=0.3,
            max_nfev=120,
            selected_outages=3,
            node_initialization="linear",
            node_initialization_blend=0.5,
            nominal_first_homotopy=False,
        ),
        multiple_shooting_fixture_row(
            transfer_time=5.0,
            amax=0.5,
            max_nfev=250,
            selected_outages=0,
            node_initialization="blend",
            node_initialization_blend=0.25,
            nominal_first_homotopy=True,
            nominal_first_nfev=17,
            nominal_first_nominal_error=0.11,
            nominal_first_selected_worst_error=0.11,
            nominal_first_all_mask_worst_error=0.42,
            selected_outage_indices="[]",
            selected_outage_errors="[]",
        ),
    ]
    pd.DataFrame(rows).to_csv(results_dir / "multiple_shooting_feasibility.csv", index=False)

    def fail_run_case(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume --max-cases 0 must only regenerate artifacts")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiple_shooting_module, "run_case", fail_run_case)
    args = SimpleNamespace(
        config=config_path,
        source_states=tmp_path / "source_states.json",
        transfer_times="4.0,5.0,6.0",
        amax="0.3,0.5",
        segments="14,18",
        max_nfev="120,250,400",
        selected_outages=3,
        min_recovery_segments=4,
        node_initialization="rollout",
        node_initialization_blend=None,
        nominal_first_homotopy=False,
        nominal_first=None,
        nominal_first_nfev=90,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=4.0,
        branch_terminal_weight=4.0,
        branch_start_weight=5.0,
        control_weight=0.01,
        smooth_weight=0.01,
        warm_start_refinement=False,
        warm_start_nfev=90,
        max_cases=0,
        resume=True,
    )

    multiple_shooting_module.run(args)
    metadata = json.loads((results_dir / "multiple_shooting_metadata.json").read_text(encoding="utf-8"))
    summary_metadata = json.loads(
        (results_dir / "multiple_shooting_feasibility_metadata.json").read_text(encoding="utf-8")
    )
    effective_cases = metadata["effective_cases"]

    assert effective_cases[0]["config"]["refinement"]["node_initialization"] == "linear"
    assert effective_cases[0]["config"]["refinement"]["nominal_first_homotopy"] is False
    assert effective_cases[1]["config"]["refinement"]["node_initialization"] == "blend"
    assert effective_cases[1]["config"]["refinement"]["node_initialization_blend"] == 0.25
    assert effective_cases[1]["config"]["refinement"]["nominal_first_homotopy"] is True
    assert effective_cases[1]["config"]["refinement"]["nominal_first_nfev"] == 17
    assert effective_cases[1]["config"]["refinement"]["selected_outages"] == 0
    assert summary_metadata["targeted_cases"]["transfer_time"] == [4.0, 5.0]
    assert summary_metadata["targeted_cases"]["amax"] == [0.3, 0.5]
    assert summary_metadata["targeted_cases"]["segments"] == [14]
    assert summary_metadata["targeted_cases"]["max_nfev"] == [120, 250]
    assert summary_metadata["targeted_cases"]["selected_outages"] == [0, 3]
    assert summary_metadata["command_case_grid"]["segments"] == [14, 18]
    assert summary_metadata["command_case_grid"]["max_nfev"] == [120, 250, 400]


def test_multiple_shooting_resume_fingerprint_reruns_changed_settings(monkeypatch, tmp_path):
    config = {
        "run": {"label": "fingerprint_multiple_shooting"},
        "benchmark": {"transfer_time": 4.0, "amax": 0.2, "segments": 14},
        "refinement": {"mode": "branch_recovery", "max_nfev": 120},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True)

    old_args = SimpleNamespace(
        selected_outages=3,
        min_recovery_segments=4,
        node_initialization="linear",
        node_initialization_blend=0.5,
        nominal_first_homotopy=False,
        nominal_first=None,
        nominal_first_nfev=90,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=4.0,
        branch_terminal_weight=4.0,
        branch_start_weight=5.0,
        control_weight=0.01,
        smooth_weight=0.01,
        warm_start_refinement=False,
        warm_start_nfev=90,
        source_states=source_states,
    )
    old_row = multiple_shooting_fixture_row(
        **multiple_shooting_module.settings_values_for_case(config, 4.0, 0.2, 14, 120, old_args)
    )
    old_row["settings_fingerprint"] = multiple_shooting_module._settings_fingerprint(old_row)
    pd.DataFrame([old_row]).to_csv(results_dir / "multiple_shooting_feasibility.csv", index=False)

    run_calls = []
    new_args = SimpleNamespace(
        config=config_path,
        source_states=source_states,
        transfer_times="4.0",
        amax="0.2",
        segments="14",
        max_nfev="120",
        selected_outages=3,
        min_recovery_segments=5,
        node_initialization="linear",
        node_initialization_blend=0.5,
        nominal_first_homotopy=False,
        nominal_first=None,
        nominal_first_nfev=90,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=4.0,
        branch_terminal_weight=4.0,
        branch_start_weight=5.0,
        control_weight=0.02,
        smooth_weight=0.01,
        warm_start_refinement=False,
        warm_start_nfev=90,
        max_cases=None,
        resume=True,
    )

    def fake_run_case(base_config, transfer_time, amax, segments, max_nfev, args):
        run_calls.append((transfer_time, amax, segments, max_nfev))
        row = multiple_shooting_fixture_row(
            **multiple_shooting_module.settings_values_for_case(
                base_config,
                transfer_time,
                amax,
                segments,
                max_nfev,
                args,
            )
        )
        row["settings_fingerprint"] = multiple_shooting_module._settings_fingerprint(row)
        return row

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(multiple_shooting_module, "run_case", fake_run_case)

    df = multiple_shooting_module.run(new_args)

    assert run_calls == [(4.0, 0.2, 14, 120)]
    assert len(df) == 2
    assert sorted(df["min_recovery_segments"].astype(int).tolist()) == [4, 5]


def test_robust_margin_suite_schema_metadata_and_artifacts(monkeypatch, tmp_path):
    robust_module = load_script_module(
        "run_robust_margin_suite_schema_test",
        ROOT / "scripts" / "run_robust_margin_suite.py",
    )
    config = robust_margin_fixture_config()
    config_path = tmp_path / "robust_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(robust_module.multiple_shooting, "run_case", fake_robust_margin_backend_row)
    args = robust_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )

    df = robust_module.run(args)
    results_dir = tmp_path / "data" / "results" / "robust_margin_test"
    tables_dir = tmp_path / "tables" / "robust_margin_test"
    figures_dir = tmp_path / "figures" / "robust_margin_test"
    metadata = json.loads((results_dir / "robust_margin_suite_metadata.json").read_text(encoding="utf-8"))
    csv_df = pd.read_csv(results_dir / "robust_margin_suite.csv")

    required = {
        "case_group",
        "phase_time",
        "transfer_time",
        "amax",
        "segments",
        "selected_outages",
        "outage_count",
        "selected_all_outages",
        "nominal_error",
        "selected_worst_error",
        "all_mask_worst_error",
        "thresholds",
        "meets_thresholds",
        "nfev",
        "runtime_seconds",
        "optimizer_success",
        "accepted_candidate",
        "control_max_norm",
        "control_bound_violation",
        "selected_outage_indices",
        "selected_outage_errors",
        "all_outage_errors",
        "settings_fingerprint",
        "config_hash",
        "source_states_id",
        "message",
    }
    all_single = df[df["case_group"] == "all_single_outage_margin"].iloc[0]

    assert required.issubset(df.columns)
    assert len(df) == 2
    assert len(csv_df) == 2
    assert bool(all_single["selected_all_outages"]) is True
    assert int(all_single["selected_outages"]) == int(all_single["outage_count"]) == 3
    assert json.loads(all_single["thresholds"]) == {"nominal_success": 0.09, "robust_success": 0.17}
    assert metadata["row_count"] == 2
    assert "all_mask" in metadata["semantics"]
    assert "one-segment outage masks only" in metadata["limitations"][2]
    assert (tables_dir / "robust_margin_suite_table.tex").exists()
    assert (figures_dir / "robust_margin_suite.png").exists()


def test_robust_margin_suite_resume_rejects_stale_rows(monkeypatch, tmp_path):
    robust_module = load_script_module(
        "run_robust_margin_suite_resume_test",
        ROOT / "scripts" / "run_robust_margin_suite.py",
    )
    groups = {
        "thrust_margin_selected": {
            "purpose": "test stale resume",
            "phase_times": [0.2],
            "transfer_times": [0.5],
            "amax": [0.2],
            "segments": [4],
            "selected_outages": [1],
            "max_nfev": [1],
            "min_recovery_segments": 1,
        }
    }
    config = robust_margin_fixture_config(output_subdir="robust_margin_resume_test", groups=groups)
    config_path = tmp_path / "robust_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    args = robust_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )
    cases = robust_module._suite_cases(config)
    expected = robust_module._expected_case(config, args, cases[0])
    results_dir = tmp_path / "data" / "results" / "robust_margin_resume_test"
    results_dir.mkdir(parents=True)
    stale = {column: None for column in robust_module.ROBUST_MARGIN_COLUMNS}
    stale.update(
        {
            "case_id": expected["case_id"],
            "case_group": "thrust_margin_selected",
            "settings_fingerprint": "stale-fingerprint",
            "config_hash": expected["config_hash"],
            "source_states_id": expected["source_states_id"],
        }
    )
    pd.DataFrame([stale]).to_csv(results_dir / "robust_margin_suite.csv", index=False)

    run_calls = []

    def fake_run_case(*call_args, **call_kwargs):
        del call_kwargs
        run_calls.append(call_args[1:5])
        return fake_robust_margin_backend_row(*call_args)

    monkeypatch.setattr(robust_module.multiple_shooting, "run_case", fake_run_case)

    df = robust_module.run(args)
    metadata = json.loads((results_dir / "robust_margin_suite_metadata.json").read_text(encoding="utf-8"))
    rewritten = pd.read_csv(results_dir / "robust_margin_suite.csv")

    assert run_calls == [(0.5, 0.2, 4, 1)]
    assert len(df) == 1
    assert rewritten.loc[0, "settings_fingerprint"] != "stale-fingerprint"
    assert metadata["resume_rejected_rows"][0]["reason"] == "settings_fingerprint missing or mismatched"


def test_continuation_margin_suite_schema_controls_and_warm_start(monkeypatch, tmp_path):
    continuation_module = load_script_module(
        "run_continuation_margin_suite_schema_test",
        ROOT / "scripts" / "run_continuation_margin_suite.py",
    )
    config = continuation_margin_fixture_config()
    config_path = tmp_path / "continuation_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(continuation_module, "load_configured_states", fake_continuation_states)
    monkeypatch.setattr(
        continuation_module.multiple_shooting,
        "run_multiple_shooting_baseline",
        fake_continuation_backend(calls),
    )
    args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )

    df = continuation_module.run(args)
    results_dir = tmp_path / "data" / "results" / "continuation_margin_test"
    tables_dir = tmp_path / "tables" / "continuation_margin_test"
    figures_dir = tmp_path / "figures" / "continuation_margin_test"
    metadata = json.loads((results_dir / "continuation_margin_suite_metadata.json").read_text(encoding="utf-8"))
    csv_df = pd.read_csv(results_dir / "continuation_margin_suite.csv")
    warm_row = df[df["case_id"] == "warm_p02"].iloc[0]

    required = {
        "case_id",
        "case_order",
        "case_group",
        "phase_time",
        "transfer_time",
        "amax",
        "segments",
        "outage_lengths",
        "selected_outages",
        "outage_count",
        "selected_all_outages",
        "warm_start_from_case_id",
        "warm_start_from_phase_time",
        "warm_start_kind",
        "max_nfev",
        "nominal_error",
        "selected_worst_error",
        "all_mask_worst_error",
        "thresholds",
        "meets_thresholds",
        "optimizer_success",
        "multiple_shooting_success",
        "accepted_candidate",
        "nfev",
        "runtime_seconds",
        "control_max_norm",
        "control_bound_violation",
        "nominal_fuel",
        "selected_outage_indices",
        "selected_outage_errors",
        "all_outage_errors",
        "settings_fingerprint",
        "config_hash",
        "source_states_id",
        "message",
    }

    assert required.issubset(df.columns)
    assert len(df) == 2
    assert len(csv_df) == 2
    assert df["case_order"].astype(int).tolist() == [0, 1]
    assert csv_df["case_order"].astype(int).tolist() == [0, 1]
    assert [int(row["case_order"]) for row in metadata["rows"]] == [0, 1]
    assert calls[0]["guess"] is None
    assert np.allclose(calls[1]["guess"], np.full((3, 3), 0.01))
    assert bool(warm_row["selected_all_outages"]) is True
    assert int(warm_row["selected_outages"]) == int(warm_row["outage_count"]) == 3
    assert str(warm_row["warm_start_from_case_id"]) == "source_p03"
    assert str(warm_row["warm_start_source_control_hash"])
    assert (results_dir / "controls" / "source_p03_nominal_controls.json").exists()
    assert (results_dir / "controls" / "warm_p02_nominal_controls.json").exists()
    assert "continuous-backend direct multiple-shooting" in metadata["semantics"]["backend"]
    assert metadata["row_count"] == 2
    assert (tables_dir / "continuation_margin_suite_table.tex").exists()
    assert (figures_dir / "continuation_margin_suite.png").exists()


def test_continuation_margin_suite_all_mask_count_for_two_segment_outages(monkeypatch, tmp_path):
    continuation_module = load_script_module(
        "run_continuation_margin_suite_count_test",
        ROOT / "scripts" / "run_continuation_margin_suite.py",
    )
    groups = {
        "single_segment_all_mask_diagnostic": {
            "purpose": "test all one-segment outages",
            "outage_lengths": [1],
            "cases": [
                {
                    "case_id": "single_segment_source",
                    "phase_time": 0.3,
                    "transfer_time": 0.5,
                    "amax": 0.3,
                    "segments": 8,
                    "selected_outages": "all_configured_outage_masks",
                    "max_nfev": 2,
                    "warm_start_kind": "cold",
                }
            ],
        },
        "two_segment_all_mask_diagnostic": {
            "purpose": "test all one/two-segment outages",
            "outage_lengths": [1, 2],
            "cases": [
                {
                    "case_id": "two_segment_source",
                    "phase_time": 0.3,
                    "transfer_time": 0.5,
                    "amax": 0.3,
                    "segments": 6,
                    "selected_outages": "all_configured_outage_masks",
                    "max_nfev": 2,
                    "warm_start_kind": "cold",
                }
            ],
        }
    }
    config = continuation_margin_fixture_config(output_subdir="continuation_margin_count_test", groups=groups)
    config_path = tmp_path / "continuation_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(continuation_module, "load_configured_states", fake_continuation_states)
    monkeypatch.setattr(
        continuation_module.multiple_shooting,
        "run_multiple_shooting_baseline",
        fake_continuation_backend(calls),
    )
    args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )

    df = continuation_module.run(args)
    single_row = df[df["case_id"] == "single_segment_source"].iloc[0]
    two_segment_row = df[df["case_id"] == "two_segment_source"].iloc[0]

    assert continuation_module._outage_count(8, [1]) == 8
    assert continuation_module._selected_outages("all_configured_outage_masks", 8, [1]) == 8
    assert continuation_module._outage_count(6, [1, 2]) == 11
    assert continuation_module._selected_outages("all_configured_outage_masks", 6, [1, 2]) == 11
    assert calls[0]["mask_count"] == 8
    assert calls[0]["selected_outages"] == 8
    assert calls[1]["mask_count"] == 11
    assert calls[1]["selected_outages"] == 11
    assert int(single_row["outage_count"]) == 8
    assert int(single_row["selected_outages"]) == 8
    assert bool(single_row["selected_all_outages"]) is True
    assert int(two_segment_row["outage_count"]) == 11
    assert int(two_segment_row["selected_outages"]) == 11
    assert bool(two_segment_row["selected_all_outages"]) is True


def test_continuation_margin_suite_resume_uses_persisted_controls(monkeypatch, tmp_path):
    continuation_module = load_script_module(
        "run_continuation_margin_suite_resume_test",
        ROOT / "scripts" / "run_continuation_margin_suite.py",
    )
    config = continuation_margin_fixture_config(output_subdir="continuation_margin_resume_test")
    config_path = tmp_path / "continuation_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(continuation_module, "load_configured_states", fake_continuation_states)
    monkeypatch.setattr(
        continuation_module.multiple_shooting,
        "run_multiple_shooting_baseline",
        fake_continuation_backend(calls),
    )
    first_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )
    continuation_module.run(first_args)
    assert len(calls) == 2
    results_dir = tmp_path / "data" / "results" / "continuation_margin_resume_test"
    csv_path = results_dir / "continuation_margin_suite.csv"
    stale_csv = pd.read_csv(csv_path)
    stale_csv["case_order"] = ""
    stale_csv.to_csv(csv_path, index=False)

    def fail_backend(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume should use compatible CSV rows and persisted controls")

    calls.clear()
    monkeypatch.setattr(continuation_module.multiple_shooting, "run_multiple_shooting_baseline", fail_backend)
    resume_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )

    df = continuation_module.run(resume_args)
    metadata = json.loads((results_dir / "continuation_margin_suite_metadata.json").read_text(encoding="utf-8"))
    rewritten = pd.read_csv(csv_path)

    assert len(df) == 2
    assert calls == []
    assert df["case_order"].astype(int).tolist() == [0, 1]
    assert rewritten["case_order"].astype(int).tolist() == [0, 1]
    assert [int(row["case_order"]) for row in metadata["rows"]] == [0, 1]
    assert metadata["resume_rejected_rows"] == []


def test_continuation_margin_suite_resume_reruns_when_source_controls_missing(monkeypatch, tmp_path):
    continuation_module = load_script_module(
        "run_continuation_margin_suite_missing_source_controls_test",
        ROOT / "scripts" / "run_continuation_margin_suite.py",
    )
    config = continuation_margin_fixture_config(output_subdir="continuation_margin_missing_source_controls_test")
    config_path = tmp_path / "continuation_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(continuation_module, "load_configured_states", fake_continuation_states)
    monkeypatch.setattr(
        continuation_module.multiple_shooting,
        "run_multiple_shooting_baseline",
        fake_continuation_backend(calls),
    )
    first_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )
    continuation_module.run(first_args)
    assert len(calls) == 2

    results_dir = tmp_path / "data" / "results" / "continuation_margin_missing_source_controls_test"
    (results_dir / "controls" / "source_p03_nominal_controls.json").unlink()

    calls.clear()
    resume_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )

    df = continuation_module.run(resume_args)
    metadata = json.loads((results_dir / "continuation_margin_suite_metadata.json").read_text(encoding="utf-8"))

    assert len(df) == 2
    assert len(calls) == 2
    assert calls[0]["guess"] is None
    assert np.allclose(calls[1]["guess"], np.full((3, 3), 0.01))
    assert [
        (row["case_id"], row["reason"])
        for row in metadata["resume_rejected_rows"]
    ] == [
        ("source_p03", "nominal-control sidecar missing, stale, or hash mismatched"),
        ("warm_p02", "warm-start source controls are unavailable or stale"),
    ]
    assert metadata["resume_rejected_rows"][1]["source_case_id"] == "source_p03"


@pytest.mark.parametrize("corrupt_kind", ["truncated_json", "invalid_controls"])
def test_continuation_margin_suite_resume_reruns_when_source_controls_corrupt(monkeypatch, tmp_path, corrupt_kind):
    continuation_module = load_script_module(
        f"run_continuation_margin_suite_corrupt_source_controls_test_{corrupt_kind}",
        ROOT / "scripts" / "run_continuation_margin_suite.py",
    )
    config = continuation_margin_fixture_config(output_subdir=f"continuation_margin_corrupt_source_controls_test_{corrupt_kind}")
    config_path = tmp_path / "continuation_margin.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(continuation_module, "load_configured_states", fake_continuation_states)
    monkeypatch.setattr(
        continuation_module.multiple_shooting,
        "run_multiple_shooting_baseline",
        fake_continuation_backend(calls),
    )
    first_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )
    continuation_module.run(first_args)
    assert len(calls) == 2

    results_dir = tmp_path / "data" / "results" / f"continuation_margin_corrupt_source_controls_test_{corrupt_kind}"
    controls_dir = results_dir / "controls"
    csv_path = results_dir / "continuation_margin_suite.csv"
    source_sidecar = controls_dir / "source_p03_nominal_controls.json"
    if corrupt_kind == "truncated_json":
        source_sidecar.write_text('{"case_id": "source_p03",', encoding="utf-8")
    elif corrupt_kind == "invalid_controls":
        payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
        payload["controls"] = [["not-a-float"]]
        source_sidecar.write_text(json.dumps(payload), encoding="utf-8")
    else:
        raise AssertionError(f"unhandled corrupt_kind {corrupt_kind}")

    calls.clear()
    resume_args = continuation_module.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states), "--resume"]
    )

    df = continuation_module.run(resume_args)
    metadata = json.loads((results_dir / "continuation_margin_suite_metadata.json").read_text(encoding="utf-8"))
    csv_df = pd.read_csv(csv_path)

    assert len(df) == 2
    assert len(csv_df) == 2
    assert len(calls) == 2
    assert calls[0]["guess"] is None
    assert np.allclose(calls[1]["guess"], np.full((3, 3), 0.01))
    assert df["case_order"].astype(int).tolist() == [0, 1]
    assert csv_df["case_order"].astype(int).tolist() == [0, 1]
    assert metadata["row_count"] == 2
    assert [int(row["case_order"]) for row in metadata["rows"]] == [0, 1]
    assert [
        (row["case_id"], row["reason"])
        for row in metadata["resume_rejected_rows"]
    ] == [
        ("source_p03", "nominal-control sidecar missing, stale, or hash mismatched"),
        ("warm_p02", "warm-start source controls are unavailable or stale"),
    ]
    assert metadata["resume_rejected_rows"][1]["source_case_id"] == "source_p03"

    for case_id in ("source_p03", "warm_p02"):
        sidecar_path = controls_dir / f"{case_id}_nominal_controls.json"
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        row = csv_df[csv_df["case_id"] == case_id].iloc[0]
        loaded = continuation_module._load_controls(controls_dir, case_id, str(row["settings_fingerprint"]))
        assert loaded is not None
        controls, control_hash, control_path = loaded
        assert payload["case_id"] == case_id
        assert payload["control_hash"] == control_hash
        assert control_path == sidecar_path
        assert controls.shape == (3, 3)
        assert control_hash == str(row["nominal_control_hash"])


def test_multiple_shooting_result_schema_and_control_bound():
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    args = SimpleNamespace(
        source_states=ROOT / "data" / "source_states.json",
        selected_outages=1,
        min_recovery_segments=1,
        initial_weight=10.0,
        defect_weight=1.0,
        terminal_weight=2.0,
        branch_terminal_weight=2.0,
        branch_start_weight=5.0,
        control_weight=0.01,
        smooth_weight=0.01,
    )

    row = run_multiple_shooting_case(
        config,
        transfer_time=1.0,
        amax=0.08,
        segments=4,
        max_nfev=1,
        args=args,
    )

    assert set(MULTIPLE_SHOOTING_COLUMNS).issubset(row)
    assert row["schedule"] == "1111"
    assert row["nominal_threshold"] == config["objective"]["thresholds"]["nominal_success"]
    assert row["selected_worst_threshold"] == config["objective"]["thresholds"]["robust_success"]
    assert isinstance(row["meets_thresholds"], bool)
    assert row["control_max_norm"] <= row["amax"] + 1e-10
    assert row["solver_mode"] == "bounded_projected_multiple_shooting"
    assert row["bounded_controls_in_residual"] is True
    assert row["residual_control_max_norm"] <= row["amax"] + 1e-10
    assert row["evaluation_control_max_norm"] <= row["amax"] + 1e-10
    assert row["control_bound_violation"] <= 1e-10
    assert row["nfev"] >= 1


def test_summary_preserves_budget_accounting_columns():
    raw = pd.DataFrame(
        [
            {
                "method": "random",
                "objective": 1.0,
                "nominal_error": 0.1,
                "worst_error": 0.2,
                "robust_degradation": 0.1,
                "active_fraction": 0.5,
                "solver_true_evaluations": 24,
                "shared_qubo_training_evaluations": 0,
                "total_true_evaluations_including_training": 24,
                "true_evaluations": 24,
                "runtime_seconds": 1.0,
                "refined_nominal_error": 0.05,
                "refined_worst_error": 0.1,
                "refined_selected_worst_error": 0.1,
                "refined_all_mask_worst_error": 0.15,
                "refinement_nfev": 3,
                "refined_nominal_fuel": 0.01,
                "refined_recovery_fuel_mean": 0.02,
                "refined_recovery_fuel_max": 0.03,
                "refinement_success": True,
            },
            {
                "method": "surrogate_qubo_sa",
                "objective": 0.8,
                "nominal_error": 0.08,
                "worst_error": 0.18,
                "robust_degradation": 0.1,
                "active_fraction": 0.5,
                "solver_true_evaluations": 24,
                "shared_qubo_training_evaluations": 48,
                "total_true_evaluations_including_training": 72,
                "true_evaluations": 72,
                "runtime_seconds": 1.0,
                "refined_nominal_error": 0.04,
                "refined_worst_error": 0.09,
                "refined_selected_worst_error": 0.09,
                "refined_all_mask_worst_error": 0.11,
                "refinement_nfev": 4,
                "refined_nominal_fuel": 0.01,
                "refined_recovery_fuel_mean": 0.02,
                "refined_recovery_fuel_max": 0.03,
                "refinement_success": True,
            },
        ]
    )
    summary = summarize(raw)
    qubo_row = summary.loc[summary["method"] == "surrogate_qubo_sa"].iloc[0]
    assert qubo_row["solver_true_evaluations_median"] == 24
    assert qubo_row["shared_qubo_training_evaluations_median"] == 48
    assert qubo_row["total_true_evaluations_including_training_median"] == 72
    assert qubo_row["true_evaluations_median"] == 72


def test_summary_exposes_repair_schedule_provenance_and_active_fractions():
    base = {
        "objective": 1.0,
        "nominal_error": 0.1,
        "worst_error": 0.2,
        "robust_degradation": 0.1,
        "solver_true_evaluations": 1,
        "shared_qubo_training_evaluations": 0,
        "total_true_evaluations_including_training": 1,
        "true_evaluations": 1,
        "runtime_seconds": 0.01,
        "refined_nominal_error": 0.05,
        "refined_worst_error": 0.1,
        "refined_selected_worst_error": 0.1,
        "refined_all_mask_worst_error": 0.1,
        "refinement_nfev": 1,
        "refined_nominal_fuel": 0.01,
        "refined_recovery_fuel_mean": 0.01,
        "refined_recovery_fuel_max": 0.01,
        "refinement_success": True,
        "schedule_repair": "prefix_to_last_active",
        "refined_schedule_source": "heuristic dense-availability envelope",
    }
    raw = pd.DataFrame(
        [
            {
                **base,
                "method": "repaired",
                "schedule": "00000101",
                "refinement_candidate_schedule": "00000101",
                "refined_schedule": "11111111",
                "active_fraction": 0.25,
            },
            {
                **base,
                "method": "repaired",
                "schedule": "00000111",
                "refinement_candidate_schedule": "00000111",
                "refined_schedule": "11111111",
                "active_fraction": 0.375,
            },
        ]
    )

    summary = summarize(raw)
    row = summary.loc[summary["method"] == "repaired"].iloc[0]

    assert row["schedule"] == "mixed"
    assert row["refinement_candidate_schedule"] == "mixed"
    assert row["refined_schedule"] == "11111111"
    assert row["active_fraction_median"] == 0.3125
    assert row["refinement_candidate_active_fraction_median"] == 0.3125
    assert row["refined_active_fraction_median"] == 1.0
    assert row["schedule_repair"] == "prefix_to_last_active"


def test_summary_keeps_empty_diagnostic_notes_blank_and_mixes_nonempty_notes():
    base = {
        "objective": 1.0,
        "nominal_error": 0.1,
        "worst_error": 0.2,
        "robust_degradation": 0.1,
        "active_fraction": 0.5,
        "solver_true_evaluations": 1,
        "shared_qubo_training_evaluations": 0,
        "total_true_evaluations_including_training": 1,
        "true_evaluations": 1,
        "runtime_seconds": 0.01,
        "refined_nominal_error": 0.05,
        "refined_worst_error": 0.1,
        "refined_selected_worst_error": 0.1,
        "refined_all_mask_worst_error": 0.1,
        "refinement_nfev": 1,
        "refined_nominal_fuel": 0.01,
        "refined_recovery_fuel_mean": 0.01,
        "refined_recovery_fuel_max": 0.01,
        "refinement_success": True,
    }
    raw = pd.DataFrame(
        [
            {**base, "method": "normal", "diagnostic_notes": None},
            {**base, "method": "normal", "diagnostic_notes": ""},
            {**base, "method": "diagnostic", "diagnostic_notes": "oracle initialization"},
            {**base, "method": "diagnostic", "diagnostic_notes": "extra diagnostic"},
        ]
    )

    summary = summarize(raw)

    normal_notes = summary.loc[summary["method"] == "normal", "diagnostic_notes"].iloc[0]
    diagnostic_notes = summary.loc[summary["method"] == "diagnostic", "diagnostic_notes"].iloc[0]
    assert pd.isna(normal_notes)
    assert diagnostic_notes == "mixed"


def test_latex_results_table_uses_explicit_budget_accounting_columns(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "method": "surrogate_qubo_sa",
                "nominal_error_median": 0.1,
                "worst_error_median": 0.2,
                "refined_nominal_error_median": 0.05,
                "refined_selected_worst_error_median": 0.08,
                "refined_all_mask_worst_error_median": 0.12,
                "active_fraction_median": 0.5,
                "refinement_candidate_active_fraction_median": 0.375,
                "refined_active_fraction_median": 1.0,
                "schedule_repair": "prefix_to_last_active",
                "refined_schedule_source": "heuristic dense-availability envelope",
                "schedule": "00000101",
                "refinement_candidate_schedule": "00000111",
                "refined_schedule": "11111111",
                "solver_true_evaluations_median": 24,
                "shared_qubo_training_evaluations_median": 48,
                "total_true_evaluations_including_training_median": 72,
                "true_evaluations_median": 999,
                "refined_nominal_fuel_median": 0.01,
                "refined_recovery_fuel_mean_median": 0.02,
                "refined_recovery_fuel_max_median": 0.03,
                "refinement_nfev_median": 4,
                "refinement_success_rate": 1.0,
            }
        ]
    )
    write_latex_tables(summary, pd.DataFrame([{"seed": 1, "train_rmse": 0.0}]), tmp_path)

    table = (tmp_path / "results_table.tex").read_text(encoding="utf-8")
    assert "Solver true evals" in table
    assert "Shared QUBO train evals" in table
    assert "Total true evals" in table
    assert "Selected recovery worst error" in table
    assert "Solver best schedule" in table
    assert "Refine candidate schedule" in table
    assert "Refined schedule" in table
    assert "Solver best active fraction" in table
    assert "Refine candidate active fraction" in table
    assert "Refined active fraction" in table
    assert "00000111" in table
    assert "11111111" in table
    assert " & 48 & " in table
    assert " & 72 & " in table
    assert "999.0000" not in table
    recovery = (tmp_path / "recovery_table.tex").read_text(encoding="utf-8")
    assert "Schedule repair" in recovery
    assert "Refined schedule" in recovery
    assert "prefix\\_to\\_last\\_active" in recovery


def test_trajectory_example_selection_excludes_diagnostics_until_fallback():
    raw = pd.DataFrame(
        [
            {
                "method": "teacher_controls_oracle_diagnostic",
                "comparison_group": "oracle_diagnostic",
                "refined_worst_error": 0.01,
                "worst_error": 0.01,
            },
            {
                "method": "random",
                "comparison_group": "method_comparison",
                "refined_worst_error": 0.2,
                "worst_error": 0.3,
            },
        ]
    )

    row, metadata = experiment.select_trajectory_example_row(raw)

    assert row["method"] == "random"
    assert metadata["used_diagnostic_fallback"] is False

    fallback_row, fallback_metadata = experiment.select_trajectory_example_row(raw.iloc[[0]])
    assert fallback_row["method"] == "teacher_controls_oracle_diagnostic"
    assert fallback_metadata["used_diagnostic_fallback"] is True
    assert "fallback_reason" in fallback_metadata


def test_cleanup_removes_stale_qubo_artifacts_without_source_files(tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir()

    stale_fit = results_dir / "qubo_fit_seed999.csv"
    stale_coefficients = results_dir / "qubo_coefficients_seed999.json"
    source_file = results_dir / "source_states.json"
    unrelated = results_dir / "manual_notes.csv"
    for path in (stale_fit, stale_coefficients, source_file, unrelated):
        path.write_text("keep or delete by pattern", encoding="utf-8")

    clean_generated_outputs(results_dir, figures_dir, tables_dir)

    assert not stale_fit.exists()
    assert not stale_coefficients.exists()
    assert source_file.exists()
    assert unrelated.exists()


def test_schedule_repair_variants_are_deterministic_and_disabled_by_default():
    schedule = np.array([0, 1, 0, 1, 0], dtype=int)

    disabled = experiment.schedule_repair_variants(schedule, {"experiments": {}})
    assert [(name, bits.tolist()) for name, bits, _ in disabled] == [("identity", [0, 1, 0, 1, 0])]

    enabled = experiment.schedule_repair_variants(
        schedule,
        {"experiments": {"schedule_repairs": {"enabled": True, "variants": ["identity", "prefix_to_last_active"]}}},
    )
    assert [(name, bits.tolist()) for name, bits, _ in enabled] == [
        ("identity", [0, 1, 0, 1, 0]),
        ("prefix_to_last_active", [1, 1, 1, 1, 0]),
    ]
    assert experiment.repair_schedule_variant(np.zeros(4, dtype=int), "prefix_to_last_active").tolist() == [0, 0, 0, 0]
    assert experiment.repair_schedule_variant(np.array([0, 1, 0, 0]), "prefix_to_last_active").tolist() == [1, 1, 0, 0]


def test_run_seed_keeps_refined_schedule_and_controls_consistent_for_multiple_candidates(monkeypatch, tmp_path):
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "worst": 1.0, "active": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    config = {
        "experiments": {"top_refine_per_method": 2},
        "objective": {"thresholds": {}},
        "refinement": {},
        "solvers": {
            "random": {"samples": 1},
            "cem": {},
            "genetic": {},
            "true_sa": {},
            "surrogate_sa": {},
            "qaoa": {},
        },
    }
    states = SimpleNamespace(initial=np.zeros(6), target=np.ones(6))
    schedules = [np.array([1, 0, 0]), np.array([0, 1, 1])]

    def fake_train_qubo(evaluator, rng, config, seed, out_dir):
        del evaluator, rng, config, seed
        fit_path = out_dir / "qubo_fit_seed1.csv"
        fit_path.write_text("split,true,predicted\nval,1,1\n", encoding="utf-8")
        return SimpleNamespace(diagnostics={}), fit_path, 7

    def fake_solver(method):
        return SolverResult(
            method=method,
            best_schedule=schedules[0],
            best_metrics={
                "objective": 0.0,
                "nominal_error": 1.0,
                "worst_error": 2.0,
                "robust_degradation": 1.0,
                "active_fraction": 1.0 / 3.0,
                "smoothness": 0.0,
            },
            evaluated=[
                (schedules[0], {"objective": 0.0}),
                (schedules[1], {"objective": 1.0}),
            ],
            true_evaluations=2,
            runtime_seconds=0.01,
        )

    def fake_refine_schedule(schedule, *args):
        del args
        marker = float(schedule.sum())
        return {
            "success": True,
            "nominal_error": marker,
            "worst_error": 10.0 - marker,
            "nfev": 1,
            "runtime_seconds": 0.01,
            "controls": np.full((3, 3), marker),
        }

    monkeypatch.setattr(experiment, "train_qubo", fake_train_qubo)
    monkeypatch.setattr(experiment, "solve_random", lambda *args: fake_solver("random"))
    monkeypatch.setattr(experiment, "solve_cem", lambda *args: fake_solver("cross_entropy"))
    monkeypatch.setattr(experiment, "solve_genetic", lambda *args: fake_solver("genetic"))
    monkeypatch.setattr(experiment, "solve_true_sa", lambda *args: fake_solver("true_sa"))
    monkeypatch.setattr(experiment, "solve_surrogate_sa", lambda *args: fake_solver("surrogate_qubo_sa"))
    monkeypatch.setattr(experiment, "solve_qaoa", lambda *args: fake_solver("qaoa_statevector"))
    monkeypatch.setattr(experiment, "refine_schedule", fake_refine_schedule)

    rows, _, _ = run_seed(1, states, cfg, config, tmp_path)

    for row in rows:
        controls = np.asarray(row["refined_controls"], dtype=float)
        assert row["schedule"] == "100"
        assert row["refined_schedule"] == "011"
        assert row["refinement_candidate_schedule"] == "011"
        assert row["active_fraction"] == 1.0 / 3.0
        assert row["refinement_candidate_active_fraction"] == 2.0 / 3.0
        assert row["refined_active_fraction"] == 2.0 / 3.0
        assert row["schedule_repair"] == "identity"
        assert np.all(controls == 2.0)


def test_run_seed_records_schedule_repair_winner_metadata(monkeypatch, tmp_path):
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "worst": 1.0, "active": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    config = {
        "experiments": {
            "top_refine_per_method": 1,
            "schedule_repairs": {"enabled": True, "variants": ["identity", "prefix_to_last_active"]},
        },
        "objective": {"thresholds": {}},
        "refinement": {},
        "solvers": {
            "random": {"samples": 1},
            "cem": {},
            "genetic": {},
            "true_sa": {},
            "surrogate_sa": {},
            "qaoa": {},
        },
    }
    states = SimpleNamespace(initial=np.zeros(6), target=np.ones(6))
    schedule = np.array([0, 0, 1])

    def fake_train_qubo(evaluator, rng, config, seed, out_dir):
        del evaluator, rng, config, seed
        fit_path = out_dir / "qubo_fit_seed1.csv"
        fit_path.write_text("split,true,predicted\nval,1,1\n", encoding="utf-8")
        return SimpleNamespace(diagnostics={}), fit_path, 7

    def fake_solver(method):
        return SolverResult(
            method=method,
            best_schedule=schedule,
            best_metrics={
                "objective": 0.0,
                "nominal_error": 1.0,
                "worst_error": 2.0,
                "robust_degradation": 1.0,
                "active_fraction": 1.0 / 3.0,
                "smoothness": 0.0,
            },
            evaluated=[(schedule, {"objective": 0.0})],
            true_evaluations=2,
            runtime_seconds=0.01,
        )

    def fake_refine_schedule(refine_schedule_arg, *args):
        del args
        bits = "".join(str(int(v)) for v in refine_schedule_arg)
        worst_error = 1.0 if bits == "111" else 5.0
        return {
            "success": bits == "111",
            "nominal_error": worst_error,
            "worst_error": worst_error,
            "nfev": 1,
            "runtime_seconds": 0.01,
            "controls": np.full((3, 3), float(refine_schedule_arg.sum())),
        }

    monkeypatch.setattr(experiment, "train_qubo", fake_train_qubo)
    monkeypatch.setattr(experiment, "solve_random", lambda *args: fake_solver("random"))
    monkeypatch.setattr(experiment, "solve_cem", lambda *args: fake_solver("cross_entropy"))
    monkeypatch.setattr(experiment, "solve_genetic", lambda *args: fake_solver("genetic"))
    monkeypatch.setattr(experiment, "solve_true_sa", lambda *args: fake_solver("true_sa"))
    monkeypatch.setattr(experiment, "solve_surrogate_sa", lambda *args: fake_solver("surrogate_qubo_sa"))
    monkeypatch.setattr(experiment, "solve_qaoa", lambda *args: fake_solver("qaoa_statevector"))
    monkeypatch.setattr(experiment, "refine_schedule", fake_refine_schedule)

    rows, _, _ = run_seed(1, states, cfg, config, tmp_path)

    assert len(rows) == 6
    for row in rows:
        assert row["schedule"] == "001"
        assert row["refinement_candidate_schedule"] == "001"
        assert row["refined_schedule"] == "111"
        assert row["active_fraction"] == 1.0 / 3.0
        assert row["refinement_candidate_active_fraction"] == 1.0 / 3.0
        assert row["refined_active_fraction"] == 1.0
        assert row["schedule_repair"] == "prefix_to_last_active"
        assert "dense-availability envelope" in row["refined_schedule_source"]
        assert row["refinement_success"] is True


def test_run_seed_adds_all_windows_continuous_baseline(monkeypatch, tmp_path):
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "robust_worst": 1.0, "robust_degradation": 0.0, "active_fraction": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    config = {
        "experiments": {"top_refine_per_method": 1, "equal_total_budget": False},
        "objective": {"thresholds": {}},
        "refinement": {},
        "baselines": {"all_windows_continuous": {"enabled": True}},
        "solvers": {
            "random": {"samples": 1},
            "cem": {},
            "genetic": {},
            "true_sa": {},
            "surrogate_sa": {},
            "qaoa": {},
        },
    }
    states = SimpleNamespace(initial=np.zeros(6), target=np.ones(6))

    def fake_train_qubo(evaluator, rng, config, seed, out_dir):
        del evaluator, rng, config, seed
        fit_path = out_dir / "qubo_fit_seed1.csv"
        fit_path.write_text("split,true,predicted\nval,1,1\n", encoding="utf-8")
        return SimpleNamespace(diagnostics={}), fit_path, 7

    def fake_solver(method):
        schedule = np.array([1, 0, 0])
        return SolverResult(
            method=method,
            best_schedule=schedule,
            best_metrics={
                "objective": 0.0,
                "nominal_error": 1.0,
                "worst_error": 2.0,
                "robust_degradation": 1.0,
                "active_fraction": 1.0 / 3.0,
                "smoothness": 0.0,
            },
            evaluated=[(schedule, {"objective": 0.0})],
            true_evaluations=2,
            runtime_seconds=0.01,
        )

    def fake_refine_schedule(schedule, *args):
        del args
        return {
            "success": True,
            "mode": "branch_recovery",
            "nominal_error": 0.1,
            "worst_error": 0.2,
            "selected_worst_error": 0.2,
            "all_mask_worst_error": 0.3,
            "nfev": 5,
            "runtime_seconds": 0.01,
            "nominal_fuel": 0.01,
            "recovery_fuel_mean": 0.02,
            "recovery_fuel_max": 0.03,
            "controls": np.full((3, 3), float(schedule.sum())),
        }

    monkeypatch.setattr(experiment, "train_qubo", fake_train_qubo)
    monkeypatch.setattr(experiment, "solve_random", lambda *args: fake_solver("random"))
    monkeypatch.setattr(experiment, "solve_cem", lambda *args: fake_solver("cross_entropy"))
    monkeypatch.setattr(experiment, "solve_genetic", lambda *args: fake_solver("genetic"))
    monkeypatch.setattr(experiment, "solve_true_sa", lambda *args: fake_solver("true_sa"))
    monkeypatch.setattr(experiment, "solve_surrogate_sa", lambda *args: fake_solver("surrogate_qubo_sa"))
    monkeypatch.setattr(experiment, "solve_qaoa", lambda *args: fake_solver("qaoa_statevector"))
    monkeypatch.setattr(experiment, "refine_schedule", fake_refine_schedule)

    rows, _, _ = run_seed(1, states, cfg, config, tmp_path)

    baseline = [row for row in rows if row["method"] == "all_windows_continuous"]
    assert len(baseline) == 1
    assert baseline[0]["schedule"] == "111"
    assert baseline[0]["solver_true_evaluations"] == 5
    assert baseline[0]["refinement_mode"] == "branch_recovery"


def test_run_seed_emits_teacher_controls_oracle_diagnostic(monkeypatch, tmp_path):
    cfg = ObjectiveConfig(
        mu=0.01215,
        tf=0.1,
        n_segments=3,
        substeps=1,
        amax=0.2,
        kr=0.1,
        kv=0.1,
        position_scale=1.0,
        velocity_scale=1.0,
        weights={"nominal": 1.0, "robust_worst": 1.0, "robust_degradation": 0.0, "active_fraction": 0.0, "smoothness": 0.0},
        outage_lengths=(1,),
    )
    config = {
        "run": {"label": "teacher_oracle_test"},
        "experiments": {"top_refine_per_method": 1, "equal_total_budget": False},
        "objective": {"thresholds": {}},
        "refinement": {"mode": "branch_recovery", "max_nfev": 1},
        "baselines": {"teacher_controls_oracle": {"enabled": True}},
        "solvers": {
            "random": {"samples": 1},
            "cem": {},
            "genetic": {},
            "true_sa": {},
            "surrogate_sa": {},
            "qaoa": {},
        },
    }
    teacher_controls = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]])
    states = SimpleNamespace(
        initial=np.zeros(6),
        target=np.ones(6),
        target_mode="teacher_controlled",
        target_metadata={"teacher": {"active_schedule": "101", "controls": teacher_controls.tolist()}},
        teacher_controls=teacher_controls,
    )
    captured = {}

    def fake_train_qubo(evaluator, rng, config, seed, out_dir):
        del evaluator, rng, config, seed
        fit_path = out_dir / "qubo_fit_seed1.csv"
        fit_path.write_text("split,true,predicted\nval,1,1\n", encoding="utf-8")
        return SimpleNamespace(diagnostics={}), fit_path, 7

    def fake_solver(method):
        schedule = np.array([1, 0, 0])
        return SolverResult(
            method=method,
            best_schedule=schedule,
            best_metrics={
                "objective": 0.0,
                "nominal_error": 1.0,
                "worst_error": 2.0,
                "robust_degradation": 1.0,
                "active_fraction": 1.0 / 3.0,
                "smoothness": 0.0,
            },
            evaluated=[(schedule, {"objective": 0.0})],
            true_evaluations=2,
            runtime_seconds=0.01,
        )

    def fake_refine_schedule(schedule, state0, target, cfg, masks, refine_cfg, thresholds):
        del state0, target, cfg, masks, thresholds
        if refine_cfg.get("use_only_initial_control_guesses"):
            captured["oracle_schedule"] = schedule.copy()
            captured["oracle_refine_cfg"] = refine_cfg
        return {
            "success": True,
            "mode": "branch_recovery",
            "nominal_error": 0.1,
            "worst_error": 0.2,
            "selected_worst_error": 0.2,
            "all_mask_worst_error": 0.3,
            "nfev": 5,
            "runtime_seconds": 0.01,
            "nominal_fuel": 0.01,
            "recovery_fuel_mean": 0.02,
            "recovery_fuel_max": 0.03,
            "controls": np.full((3, 3), float(schedule.sum())),
        }

    monkeypatch.setattr(experiment, "train_qubo", fake_train_qubo)
    monkeypatch.setattr(experiment, "solve_random", lambda *args: fake_solver("random"))
    monkeypatch.setattr(experiment, "solve_cem", lambda *args: fake_solver("cross_entropy"))
    monkeypatch.setattr(experiment, "solve_genetic", lambda *args: fake_solver("genetic"))
    monkeypatch.setattr(experiment, "solve_true_sa", lambda *args: fake_solver("true_sa"))
    monkeypatch.setattr(experiment, "solve_surrogate_sa", lambda *args: fake_solver("surrogate_qubo_sa"))
    monkeypatch.setattr(experiment, "solve_qaoa", lambda *args: fake_solver("qaoa_statevector"))
    monkeypatch.setattr(experiment, "refine_schedule", fake_refine_schedule)

    rows, _, _ = run_seed(1, states, cfg, config, tmp_path)

    oracle = [row for row in rows if row["method"] == "teacher_controls_oracle_diagnostic"]
    assert len(oracle) == 1
    assert oracle[0]["comparison_group"] == "oracle_diagnostic"
    assert oracle[0]["schedule"] == "101"
    assert captured["oracle_schedule"].tolist() == [1, 0, 1]
    assert captured["oracle_refine_cfg"]["use_only_initial_control_guesses"] is True
    guess = captured["oracle_refine_cfg"]["initial_control_guesses"][0]
    assert guess["name"] == "teacher_controls_oracle"
    assert np.allclose(guess["controls"], teacher_controls)


def test_equal_total_budget_expands_classical_solver_configs():
    config = yaml.safe_load((ROOT / "configs" / "q1_candidate.yaml").read_text(encoding="utf-8"))
    solvers = experiment.solver_configs_for_budget(config, qubo_training_evaluations=72)
    target_total = config["experiments"]["solver_true_budget"] + 72
    assert solvers["random"]["samples"] == target_total
    assert solvers["true_sa"]["steps"] == target_total
    assert solvers["genetic"]["max_evaluations"] == target_total
    assert solvers["surrogate_sa"]["candidates_to_evaluate"] == config["solvers"]["surrogate_sa"]["candidates_to_evaluate"]


def test_genetic_respects_equal_total_budget_cap_for_phase_shift_cardinality():
    config = yaml.safe_load((ROOT / "configs" / "q1_phase_shift_cardinality.yaml").read_text(encoding="utf-8"))
    qubo_training_evaluations = int(config["experiments"]["qubo_train_samples"])
    target_total = int(config["experiments"]["solver_true_budget"]) + qubo_training_evaluations
    solver_cfg = experiment.solver_configs_for_budget(config, qubo_training_evaluations)["genetic"]

    class CountingEvaluator:
        def __init__(self):
            self.count = 0

        def evaluate(self, schedule):
            self.count += 1
            return {"objective": float(np.sum(schedule))}

    evaluator = CountingEvaluator()
    result = solve_genetic(
        evaluator,
        np.random.default_rng(123),
        int(config["benchmark"]["segments"]),
        solver_cfg,
    )

    assert solver_cfg["max_evaluations"] == target_total
    assert evaluator.count == target_total
    assert result.true_evaluations == target_total
    assert len(result.evaluated) == target_total


def small_qaoa_test_qubo() -> QuboModel:
    return QuboModel(
        n=3,
        intercept=0.0,
        linear=np.array([0.4, -1.1, 0.7], dtype=float),
        quadratic=np.array(
            [
                [0.0, 0.3, -0.2],
                [0.3, 0.0, 0.5],
                [-0.2, 0.5, 0.0],
            ],
            dtype=float,
        ),
        diagnostics={},
    )


def test_optimized_qaoa_angles_do_not_increase_expected_qubo_energy_from_initial_angles():
    qubo = small_qaoa_test_qubo()
    seed = 20260615
    initial_angles = random_qaoa_angles(np.random.default_rng(seed), p=2)
    initial_energy = qaoa_expected_energy(qubo, initial_angles, p=2)

    optimized_angles, optimized_energy, info = optimize_qaoa_angles(
        qubo,
        np.random.default_rng(seed),
        {"p": 2, "optimizer": "scipy_minimize", "angle_restarts": 1, "maxiter": 30},
    )

    assert optimized_angles.shape == initial_angles.shape
    assert info["optimizer"] == "scipy_minimize"
    assert info["p"] == 2
    assert optimized_energy <= initial_energy + 1e-10


def test_optimized_qaoa_true_candidate_evaluation_count_is_bounded():
    qubo = small_qaoa_test_qubo()

    class CountingEvaluator:
        def __init__(self):
            self.count = 0

        def evaluate(self, schedule):
            self.count += 1
            return {"objective": float(qubo.energy(schedule)[0])}

    evaluator = CountingEvaluator()
    result = solve_qaoa(
        evaluator,
        np.random.default_rng(7),
        {
            "p": 1,
            "optimizer": "scipy_minimize",
            "angle_restarts": 2,
            "maxiter": 12,
            "sample_count": 64,
            "candidates_to_evaluate": 3,
        },
        qubo,
    )

    assert evaluator.count == 3
    assert result.true_evaluations == 3
    assert len(result.evaluated) == 3
    assert result.best_metrics["qaoa_angle_optimizer"] == "scipy_minimize"


def test_direct_collocation_evaluation_projects_reported_controls_to_bound():
    from qlt.direct_collocation import DirectCollocationProblem, DirectCollocationWeights, initial_guess

    cfg = small_objective_config(n_segments=3, tf=0.12, amax=0.04)
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.95, -0.02, 0.01, -0.01, 0.02, 0.03], dtype=float)
    masks = outage_masks(cfg.n_segments, (1,))
    selected = np.array([0], dtype=int)
    layout, vec = initial_guess(state0, target, cfg, masks[selected], node_initialization="linear")
    raw = vec.copy()
    raw[layout.nominal_controls] = 2.0
    for control_slice in layout.branch_controls:
        raw[control_slice] = -3.0

    problem = DirectCollocationProblem(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        selected=selected,
        layout=layout,
        weights=DirectCollocationWeights(),
    )
    residual = problem.residual(raw)
    evaluated = problem.evaluate_vector(raw, {"nominal_success": 1.0, "robust_success": 1.0})

    assert np.all(np.isfinite(residual))
    assert evaluated["control_max_norm"] <= cfg.amax + 1e-12
    assert evaluated["control_bound_violation"] == 0.0
    assert np.max(np.linalg.norm(evaluated["nominal_controls"], axis=1)) <= cfg.amax + 1e-12
    for controls in evaluated["selected_branch_controls"]:
        assert np.max(np.linalg.norm(controls, axis=1)) <= cfg.amax + 1e-12


def test_direct_collocation_hermite_simpson_defect_is_finite_and_distinct():
    from qlt.direct_collocation import hermite_simpson_defect, trapezoidal_defect

    cfg = small_objective_config(n_segments=3, tf=0.18, amax=0.05)
    left = np.array([0.92, 0.07, -0.02, 0.01, 0.08, -0.03], dtype=float)
    right = np.array([0.89, 0.10, -0.01, -0.02, 0.06, 0.01], dtype=float)
    control = np.array([0.015, -0.012, 0.006], dtype=float)

    trap = trapezoidal_defect(left, right, control, cfg)
    hs = hermite_simpson_defect(left, right, control, cfg)

    assert np.all(np.isfinite(hs))
    assert np.all(np.isfinite(trap))
    assert not np.allclose(hs, trap)


def test_run_direct_collocation_baseline_hermite_simpson_reports_method_and_bounds():
    from qlt.direct_collocation import run_direct_collocation_baseline

    cfg = small_objective_config(n_segments=2, tf=0.08, amax=0.04)
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.99, -0.01, 0.01, -0.005, 0.02, 0.01], dtype=float)
    masks = outage_masks(cfg.n_segments, (1,))

    result = run_direct_collocation_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 1.0, "robust_success": 1.0},
        selected_outages=0,
        max_nfev=1,
        min_recovery_segments=1,
        collocation_config={"method": "hermite_simpson", "node_initialization": "linear"},
    )

    assert result["collocation_method"] == "hermite_simpson"
    assert result["method_type"] == "bounded_projected_constant_control_hermite_simpson_direct_collocation"
    assert "constant-control Hermite-Simpson" in result["collocation_scheme_semantics"]
    selected_branch_semantics = result["selected_branch_semantics"].lower()
    assert "only the nominal trajectory is optimized" in selected_branch_semantics
    assert "selected outage branches are optimized" not in selected_branch_semantics
    assert result["selected_outage_indices"] == []
    assert np.isfinite(result["nominal_error"])
    assert np.isfinite(result["selected_worst_error"])
    assert result["selected_worst_error"] == result["nominal_error"]
    assert np.isfinite(result["all_mask_worst_error"])
    assert result["control_max_norm"] <= cfg.amax + 1e-12
    assert result["control_bound_violation"] == 0.0


def test_direct_collocation_resume_fingerprint_changes_for_thresholds_and_settings():
    module = load_script_module(
        "run_direct_collocation_baseline_fingerprint_test",
        ROOT / "scripts" / "run_direct_collocation_baseline.py",
    )
    config = yaml.safe_load((ROOT / "configs" / "direct_collocation_baseline.yaml").read_text(encoding="utf-8"))
    args = SimpleNamespace(source_states=ROOT / "data" / "source_states.json")
    case = module._suite_cases(config)[0]

    base = module.settings_fingerprint(module._effective_settings(config, args, case))
    changed_thresholds = copy.deepcopy(config)
    changed_thresholds["objective"]["thresholds"]["nominal_success"] = 0.123
    changed_nfev = copy.deepcopy(config)
    changed_case = dict(case)
    changed_case["max_nfev"] = int(case["max_nfev"]) + 1
    changed_weight = copy.deepcopy(config)
    changed_weight["direct_collocation"]["weights"]["terminal"] = 7.0

    assert module.settings_fingerprint(module._effective_settings(changed_thresholds, args, case)) != base
    assert module.settings_fingerprint(module._effective_settings(changed_nfev, args, changed_case)) != base
    assert module.settings_fingerprint(module._effective_settings(changed_weight, args, case)) != base


def test_direct_collocation_table_and_metadata_generation_smoke(tmp_path):
    module = load_script_module(
        "run_direct_collocation_baseline_artifact_test",
        ROOT / "scripts" / "run_direct_collocation_baseline.py",
    )
    config = yaml.safe_load((ROOT / "configs" / "direct_collocation_baseline.yaml").read_text(encoding="utf-8"))
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir()

    row = {column: None for column in module.DIRECT_COLLOCATION_COLUMNS}
    row.update(
        {
            "suite_case_id": "unit",
            "case_type": "selected_branch",
            "target_mode": "catalog_halo_phase_shift",
            "target_generation": "unit-test",
            "phase_time": 0.2,
            "transfer_time": 0.5,
            "amax": 0.2,
            "segments": 3,
            "max_nfev": 1,
            "selected_outages": 1,
            "min_recovery_segments": 1,
            "settings_fingerprint": "fp",
            "config_hash": "cfg",
            "source_states_id": "source",
            "method_type": "bounded_projected_trapezoidal_direct_collocation",
            "collocation_method": "trapezoidal",
            "nominal_error": 0.1,
            "selected_worst_error": 0.2,
            "all_mask_worst_error": 0.25,
            "nominal_threshold": 0.09,
            "selected_worst_threshold": 0.17,
            "thresholds": "{}",
            "meets_nominal_threshold": False,
            "meets_selected_worst_threshold": False,
            "meets_thresholds": False,
            "control_max_norm": 0.2,
            "control_bound_violation": 0.0,
            "nominal_fuel": 0.01,
            "recovery_fuel_mean": 0.02,
            "recovery_fuel_max": 0.03,
            "cost": 1.0,
            "optimality": 0.5,
            "nfev": 1,
            "runtime_seconds": 0.01,
            "selected_outage_indices": "[0]",
            "selected_outage_errors": "[0.2]",
            "all_outage_errors": "[0.25]",
            "optimizer_success": False,
            "direct_collocation_success": False,
            "selected_branch_semantics": "selected",
            "all_mask_diagnostic_semantics": "all",
            "control_bound_semantics": "bounded",
            "message": "unit",
        }
    )

    module.regenerate(
        pd.DataFrame([row], columns=module.DIRECT_COLLOCATION_COLUMNS),
        results_dir,
        figures_dir,
        tables_dir,
        config,
        "unit-test",
        [],
        None,
    )

    assert (results_dir / "direct_collocation_baseline.csv").exists()
    assert (results_dir / "direct_collocation_baseline_metadata.json").exists()
    assert (tables_dir / "direct_collocation_baseline_table.tex").exists()
    assert (figures_dir / "direct_collocation_baseline.png").exists()
    metadata = json.loads((results_dir / "direct_collocation_baseline_metadata.json").read_text(encoding="utf-8"))
    assert metadata["method_type"] == "bounded_projected_trapezoidal_direct_collocation"
    assert metadata["limitations"]


def test_catalog_feasibility_envelope_resume_rejects_stale_and_writes_artifacts(tmp_path):
    module = load_script_module(
        "run_catalog_feasibility_envelope_artifact_test",
        ROOT / "scripts" / "run_catalog_feasibility_envelope.py",
    )
    config = yaml.safe_load((ROOT / "configs" / "catalog_feasibility_envelope.yaml").read_text(encoding="utf-8"))
    args = SimpleNamespace(source_states=ROOT / "data" / "source_states.json")
    cases = module._suite_cases(config)[:1]
    expected = module._expected_index(config, args, cases)
    case = cases[0]

    changed_target = copy.deepcopy(config)
    changed_target["benchmark"]["target_mode"] = "catalog_halo_phase_shift"
    assert module._config_for_case(changed_target, case)["benchmark"]["target_mode"] == "catalog_dro_phase"

    stale = {column: None for column in module.CATALOG_FEASIBILITY_ENVELOPE_COLUMNS}
    stale.update(
        {
            "suite_case_id": case["suite_case_id"],
            "settings_fingerprint": "stale-fingerprint",
            "config_hash": expected[case["suite_case_id"]]["config_hash"],
            "source_states_id": expected[case["suite_case_id"]]["source_states_id"],
        }
    )
    kept, rejected = module._compatible_existing_rows(
        pd.DataFrame([stale], columns=module.CATALOG_FEASIBILITY_ENVELOPE_COLUMNS),
        expected,
    )
    assert kept.empty
    assert rejected[0]["reason"] == "settings_fingerprint missing or mismatched"

    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir()

    row = {column: None for column in module.CATALOG_FEASIBILITY_ENVELOPE_COLUMNS}
    row.update(
        {
            "suite_case_id": case["suite_case_id"],
            "case_type": "nominal_only",
            "purpose": "unit",
            "target_mode": "catalog_dro_phase",
            "target_generation": "fixture",
            "backend": "multiple_shooting",
            "method": "multiple_shooting",
            "method_type": "bounded_projected_multiple_shooting",
            "collocation_method": "",
            "transfer_time": 4.0,
            "amax": 1.0,
            "segments": 14,
            "substeps_per_segment": 7,
            "max_nfev": 1,
            "selected_outages": 0,
            "min_recovery_segments": 4,
            "node_initialization": "linear",
            "node_initialization_blend": 0.5,
            "settings_fingerprint": expected[case["suite_case_id"]]["settings_fingerprint"],
            "config_hash": expected[case["suite_case_id"]]["config_hash"],
            "source_states_id": expected[case["suite_case_id"]]["source_states_id"],
            "nominal_error": 0.01,
            "selected_worst_error": 0.01,
            "all_mask_worst_error": 8.5,
            "nominal_threshold": 0.09,
            "selected_worst_threshold": 0.17,
            "thresholds": "{}",
            "meets_nominal_threshold": True,
            "meets_selected_worst_threshold": True,
            "meets_thresholds": True,
            "control_max_norm": 1.0,
            "control_bound_violation": 0.0,
            "nominal_fuel": 0.1,
            "recovery_fuel_mean": 0.1,
            "recovery_fuel_max": 0.1,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
            "selected_outage_indices": "[]",
            "selected_outage_errors": "[]",
            "all_outage_errors": "[8.5]",
            "optimizer_success": True,
            "backend_success": True,
            "accepted_candidate": "optimizer",
            "selected_branch_semantics": "nominal only",
            "all_mask_diagnostic_semantics": "diagnostic",
            "control_bound_semantics": "bounded",
            "nominal_only_semantics": "selected_outages=0 rows optimize only the nominal trajectory",
            "message": "unit",
        }
    )

    module.regenerate(
        pd.DataFrame([row], columns=module.CATALOG_FEASIBILITY_ENVELOPE_COLUMNS),
        results_dir,
        figures_dir,
        tables_dir,
        config,
        "unit-test",
        rejected,
    )

    assert (results_dir / "catalog_feasibility_envelope.csv").exists()
    assert (results_dir / "catalog_feasibility_envelope_metadata.json").exists()
    assert (tables_dir / "catalog_feasibility_envelope_table.tex").exists()
    assert (figures_dir / "catalog_feasibility_envelope.png").exists()
    assert (figures_dir / "catalog_feasibility_envelope.pdf").exists()
    metadata = json.loads((results_dir / "catalog_feasibility_envelope_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_count"] == 1
    assert metadata["resume_rejected_rows"][0]["reason"] == "settings_fingerprint missing or mismatched"
    assert "Nominal-only rows" in metadata["limitations"][0]
