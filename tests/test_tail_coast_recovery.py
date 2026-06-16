from __future__ import annotations

import importlib.util
import hashlib
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
import qlt.tail_coast_recovery as tail


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


def state_pair():
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([1.01, 0.015, 0.018, 0.005, 0.035, -0.015], dtype=float)
    return state0, target


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_deterministic_json(path: Path, data: dict) -> str:
    payload = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_hard_catalog_combined_one_two_segment_tail_coast_case_selects_all_configured_masks():
    script = load_script_module(
        "run_tail_coast_recovery_combined_case_test",
        ROOT / "scripts" / "run_tail_coast_recovery.py",
    )
    config = yaml.safe_load((ROOT / "configs" / "hard_catalog_tail_coast_recovery.yaml").read_text(encoding="utf-8"))

    cases = {case["suite_case_id"]: case for case in script._suite_cases(config)}
    case = cases["tail_coast_all_one_two_segment_t5_portfolio"]

    assert case["transfer_time"] == 4.0
    assert case["amax"] == 1.0
    assert case["segments"] == 14
    assert case["tail_coast_segments"] == 5
    assert case["outage_lengths"] == [1, 2]
    assert case["selected_outage_policy"] == "all_configured"
    assert case["outage_count"] == 27
    assert case["selected_outage_count"] == 27
    assert len(outage_masks(case["segments"], tuple(case["outage_lengths"]))) == 27

    branch = case["case_raw"]["branch"]
    assert branch["xtol"] == 1.0e-4
    assert branch["ftol"] == 1.0e-4
    assert branch["gtol"] == 1.0e-4
    assert [fallback["label"] for fallback in branch["fallback_initializations"]] == [
        "constant_y_plus_0p5",
        "constant_y_minus_0p5",
        "constant_x_plus_0p5",
        "constant_x_minus_0p5",
        "constant_y_plus_1",
        "constant_y_minus_1",
    ]
    assert [variant["label"] for variant in branch["weight_variants"]] == ["regularized_001", "terminal_only"]


def test_tail_coast_table_preserves_input_case_order(tmp_path):
    script = load_script_module(
        "run_tail_coast_recovery_table_order_test",
        ROOT / "scripts" / "run_tail_coast_recovery.py",
    )
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    df = pd.DataFrame(
        [
            {
                "suite_case_id": "tail_coast_z_config_order_first",
                "selected_outage_policy": "all_single",
                "outage_lengths": "[1]",
                "selected_outage_count": 1,
                "tail_coast_segments": 2,
                "branch_portfolio_variant_count": 1,
                "branch_fallback_initialization_evaluated_branch_count": 0,
                "branch_fallback_initialization_accepted_branch_count": 0,
                "branch_recovery_segments": "[1]",
                "branch_optimizer_success_by_branch": "[true]",
                "branch_optimizer_ran_by_branch": "[true]",
                "nominal_tail_coast_error": 0.01,
                "selected_worst_error": 0.02,
                "all_mask_worst_error": 0.03,
                "control_max_norm": 0.04,
                "control_bound_violation": 0.0,
                "nfev": 10,
                "meets_thresholds": True,
                "branch_optimizer_ran": True,
                "branch_optimizer_all_success": True,
            },
            {
                "suite_case_id": "tail_coast_a_config_order_second",
                "selected_outage_policy": "all_configured",
                "outage_lengths": "[1, 2]",
                "selected_outage_count": 2,
                "tail_coast_segments": 2,
                "branch_portfolio_variant_count": 1,
                "branch_fallback_initialization_evaluated_branch_count": 0,
                "branch_fallback_initialization_accepted_branch_count": 0,
                "branch_recovery_segments": "[1, 0]",
                "branch_optimizer_success_by_branch": "[true, false]",
                "branch_optimizer_ran_by_branch": "[true, false]",
                "nominal_tail_coast_error": 0.01,
                "selected_worst_error": 0.02,
                "all_mask_worst_error": 0.03,
                "control_max_norm": 0.04,
                "control_bound_violation": 0.0,
                "nfev": 20,
                "meets_thresholds": True,
                "branch_optimizer_ran": True,
                "branch_optimizer_all_success": False,
            },
        ]
    )

    script.write_table(df, tables_dir)

    text = (tables_dir / "tail_coast_recovery_table.tex").read_text(encoding="utf-8")
    first = text.index("tail\\_coast\\_z\\_config\\_order\\_first")
    second = text.index("tail\\_coast\\_a\\_config\\_order\\_second")
    assert first < second
    assert "tail\\_coast\\_a\\_config\\_order\\_second & all configured & [1, 2] & 2 & 2 & 1 & 0 & 0 & 1/1 & 1" in text


def test_regenerate_artifacts_only_uses_existing_csv_without_running_backend(monkeypatch, tmp_path):
    script = load_script_module(
        "run_tail_coast_recovery_regenerate_only_test",
        ROOT / "scripts" / "run_tail_coast_recovery.py",
    )
    config = {
        "run": {"label": "tail_test", "output_subdir": "tail_test"},
        "benchmark": {
            "target_mode": "catalog_dro_phase",
            "transfer_time": 0.1,
            "segments": 3,
            "substeps_per_segment": 1,
            "amax": 0.2,
        },
        "objective": {"thresholds": {"nominal_success": 0.09, "robust_success": 0.17}},
        "outages": {"block_lengths": [1]},
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "cases": [
                {"case_id": "tail_z_config_order_first", "selected_outages": 0},
                {"case_id": "tail_a_config_order_second", "selected_outages": 0},
            ],
        },
    }
    config_path = tmp_path / "tail_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    results_dir = tmp_path / "data" / "results" / "tail_test"
    results_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    args = script.build_parser().parse_args(["--config", str(config_path), "--regenerate-artifacts-only"])
    cases = script._suite_cases(config)
    expected = script._expected_index(config, args, cases)

    def row(case_id: str, nfev: int) -> dict:
        expected_row = expected[case_id]
        out = {column: None for column in script.TAIL_COAST_COLUMNS}
        out.update(
            {
                "suite_case_id": case_id,
                "selected_outage_policy": "0",
                "outage_lengths": "[1]",
                "selected_outage_count": 0,
                "tail_coast_segments": 1,
                "branch_portfolio_variant_count": 1,
                "branch_fallback_initialization_evaluated_branch_count": 0,
                "branch_fallback_initialization_accepted_branch_count": 0,
                "branch_recovery_segments": "[]",
                "branch_optimizer_success_by_branch": "[]",
                "branch_optimizer_ran_by_branch": "[]",
                "nominal_tail_coast_error": 0.01,
                "selected_worst_error": 0.02,
                "all_mask_worst_error": 0.03,
                "control_max_norm": 0.04,
                "control_bound_violation": 0.0,
                "nfev": nfev,
                "meets_thresholds": True,
                "branch_optimizer_ran": False,
                "branch_optimizer_all_success": False,
                "settings_fingerprint": expected_row["settings_fingerprint"],
                "config_hash": expected_row["config_hash"],
                "source_states_id": expected_row["source_states_id"],
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
            }
        )
        return out

    pd.DataFrame(
        [
            row("tail_a_config_order_second", 20),
            row("tail_z_config_order_first", 10),
        ],
        columns=script.TAIL_COAST_COLUMNS,
    ).to_csv(results_dir / "tail_coast_recovery.csv", index=False)
    csv_before = (results_dir / "tail_coast_recovery.csv").read_bytes()
    monkeypatch.setattr(
        script,
        "run_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts\\run_tail_coast_recovery.py", "--config", str(config_path), "--regenerate-artifacts-only"],
    )

    df = script.run(args)

    assert df["suite_case_id"].tolist() == ["tail_z_config_order_first", "tail_a_config_order_second"]
    csv_df = pd.read_csv(results_dir / "tail_coast_recovery.csv")
    assert csv_df["suite_case_id"].tolist() == ["tail_a_config_order_second", "tail_z_config_order_first"]
    assert (results_dir / "tail_coast_recovery.csv").read_bytes() == csv_before
    table = (tmp_path / "tables" / "tail_test" / "tail_coast_recovery_table.tex").read_text(encoding="utf-8")
    assert table.index("tail\\_z\\_config\\_order\\_first") < table.index("tail\\_a\\_config\\_order\\_second")
    assert "tail\\_z\\_config\\_order\\_first & none" in table
    metadata = json.loads((results_dir / "tail_coast_recovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["raw_csv_written"] is False
    assert "leaves the existing tail_coast_recovery.csv bytes untouched" in metadata["raw_csv_write_semantics"]
    assert metadata["resume_rejected_rows"] == []
    assert metadata["skipped_cases"] == []
    assert metadata["artifact_refresh_command"] == metadata["command"]
    assert "--regenerate-artifacts-only" in metadata["artifact_refresh_command"]
    assert "--resume" in metadata["evidence_replay_command"]
    assert "--regenerate-artifacts-only" not in metadata["evidence_replay_command"]


def test_regenerate_artifacts_only_rejects_stale_rows_and_records_provenance(monkeypatch, tmp_path):
    script = load_script_module(
        "run_tail_coast_recovery_regenerate_only_stale_test",
        ROOT / "scripts" / "run_tail_coast_recovery.py",
    )
    config = {
        "run": {"label": "tail_test", "output_subdir": "tail_test"},
        "benchmark": {
            "target_mode": "catalog_dro_phase",
            "transfer_time": 0.1,
            "segments": 3,
            "substeps_per_segment": 1,
            "amax": 0.2,
        },
        "objective": {"thresholds": {"nominal_success": 0.09, "robust_success": 0.17}},
        "outages": {"block_lengths": [1]},
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "cases": [
                {"case_id": "tail_keep", "selected_outages": 0},
                {"case_id": "tail_stale", "selected_outages": 0},
            ],
        },
    }
    config_path = tmp_path / "tail_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    results_dir = tmp_path / "data" / "results" / "tail_test"
    results_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    args = script.build_parser().parse_args(["--config", str(config_path), "--regenerate-artifacts-only"])
    cases = script._suite_cases(config)
    expected = script._expected_index(config, args, cases)

    def row(case_id: str, nfev: int, **overrides) -> dict:
        expected_row = expected[case_id]
        out = {column: None for column in script.TAIL_COAST_COLUMNS}
        out.update(
            {
                "suite_case_id": case_id,
                "selected_outage_policy": "0",
                "outage_lengths": "[1]",
                "selected_outage_count": 0,
                "tail_coast_segments": 1,
                "branch_portfolio_variant_count": 1,
                "branch_fallback_initialization_evaluated_branch_count": 0,
                "branch_fallback_initialization_accepted_branch_count": 0,
                "branch_recovery_segments": "[]",
                "branch_optimizer_success_by_branch": "[]",
                "branch_optimizer_ran_by_branch": "[]",
                "nominal_tail_coast_error": 0.01,
                "selected_worst_error": 0.02,
                "all_mask_worst_error": 0.03,
                "control_max_norm": 0.04,
                "control_bound_violation": 0.0,
                "nfev": nfev,
                "meets_thresholds": True,
                "branch_optimizer_ran": False,
                "branch_optimizer_all_success": False,
                "settings_fingerprint": expected_row["settings_fingerprint"],
                "config_hash": expected_row["config_hash"],
                "source_states_id": expected_row["source_states_id"],
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
            }
        )
        out.update(overrides)
        return out

    pd.DataFrame(
        [
            row("tail_stale", 30, settings_fingerprint="stale-fingerprint"),
            row("tail_stale", 31, config_hash="stale-config", source_states_id="stale-source"),
            row("tail_keep", 10),
        ],
        columns=script.TAIL_COAST_COLUMNS,
    ).to_csv(results_dir / "tail_coast_recovery.csv", index=False)
    csv_before = (results_dir / "tail_coast_recovery.csv").read_bytes()
    monkeypatch.setattr(
        script,
        "run_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend should not run")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scripts\\run_tail_coast_recovery.py", "--config", str(config_path), "--regenerate-artifacts-only"],
    )

    df = script.run(args)

    assert df["suite_case_id"].tolist() == ["tail_keep"]
    csv_df = pd.read_csv(results_dir / "tail_coast_recovery.csv")
    assert csv_df["suite_case_id"].tolist() == ["tail_stale", "tail_stale", "tail_keep"]
    assert (results_dir / "tail_coast_recovery.csv").read_bytes() == csv_before
    metadata = json.loads((results_dir / "tail_coast_recovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["raw_csv_written"] is False
    assert metadata["row_count"] == 1
    assert metadata["completed_case_count"] == 1
    assert metadata["expected_case_count"] == 2
    assert [case["suite_case_id"] for case in metadata["skipped_cases"]] == ["tail_stale"]
    assert metadata["skipped_cases"][0]["reason"] == "no compatible existing row for regenerate-artifacts-only"
    assert len(metadata["resume_rejected_rows"]) == 2
    assert metadata["resume_rejected_rows"][0]["reason"] == "settings_fingerprint missing or mismatched"
    assert metadata["resume_rejected_rows"][1]["reason"] == "provenance field mismatch"
    assert metadata["resume_rejected_rows"][1]["mismatched_fields"] == ["config_hash", "source_states_id"]
    assert metadata["artifact_refresh_command"] == metadata["command"]
    assert "--regenerate-artifacts-only" in metadata["artifact_refresh_command"]
    assert metadata["evidence_replay_command"].startswith("py -3.11 scripts\\run_tail_coast_recovery.py --config ")
    assert "--resume" in metadata["evidence_replay_command"]
    assert "--regenerate-artifacts-only" not in metadata["evidence_replay_command"]


def test_tail_controls_are_exact_zero_and_reported(monkeypatch):
    cfg = small_cfg(n_segments=5, tf=0.1, amax=0.05)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    seed_controls = np.full((cfg.n_segments, 3), 0.01, dtype=float)

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture seed",
            "accepted_candidate": "fixture",
            "controls": seed_controls.copy(),
            "nominal_error": 0.2,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 2,
            "runtime_seconds": 0.01,
        }

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        tail_coast_segments=2,
        selected_outages=0,
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=0,
    )

    controls = np.asarray(result["nominal_controls"], dtype=float)
    assert np.array_equal(controls[-2:], np.zeros((2, 3), dtype=float))
    assert result["tail_coast_segments"] == 2
    assert result["optimized_nominal_segments"] == 3
    assert result["nominal_tail_zero_max_abs"] == 0.0
    assert result["nominal_tail_control_norm_max"] == 0.0
    assert result["branch_total_duration"] == pytest.approx(cfg.tf)


def test_selected_outages_zero_runs_no_branch_optimizer_and_does_not_claim_convergence(monkeypatch):
    cfg = small_cfg(n_segments=3)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)

    def fake_multiple_shooting(**kwargs):
        assert kwargs["selected_outages"] == 0
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": tail.terminal_error(state0, target, cfg, controls),
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
        }

    def fail_branch_optimizer(**kwargs):
        raise AssertionError("branch optimizer should not run when selected_outages=0")

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(tail, "optimize_tail_coast_recovery_branch_portfolio", fail_branch_optimizer)

    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        tail_coast_segments=1,
        selected_outages=0,
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=1,
    )

    assert result["selected_outage_indices"] == []
    assert result["branch_results"] == []
    assert result["branch_nfev"] == []
    assert result["total_branch_nfev"] == 0
    assert result["branch_optimizer_ran"] is False
    assert result["branch_optimizer_all_success"] is False
    assert result["branch_optimizer_success"] is False
    assert result["optimizer_success"] is False
    assert result["branch_optimizer_success_by_branch"] == []
    assert result["branch_optimizer_ran_by_branch"] == []


def test_selected_outages_all_single_selects_only_one_segment_masks_end_to_end(monkeypatch):
    cfg = small_cfg(n_segments=4, outage_lengths=(1, 2))
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    selected_by_branch = []

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
        }

    def fake_tail_nominal(**kwargs):
        return {
            "controls": controls.copy(),
            "optimizer_success": True,
            "message": "fixture tail nominal",
            "accepted_candidate": "fixture",
            "nfev": 0,
            "runtime_seconds": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
        }

    def fake_branch_portfolio(**kwargs):
        mask_index = int(kwargs["mask_index"])
        selected_by_branch.append(mask_index)
        branch_cfg = kwargs["cfg"]
        start = tail.outage_end(kwargs["mask"])
        recovery_segments = int(branch_cfg.n_segments) - int(start)
        optimizer_ran = bool(recovery_segments > 0)
        branch_weights = kwargs["branch_weights"]
        return {
            "mask_index": mask_index,
            "recovery_start": int(start),
            "recovery_segments": recovery_segments,
            "tail_coast_segments": int(kwargs["tail_coast_segments"]),
            "branch_control_count": int(branch_cfg.n_segments),
            "nominal_segments": int(branch_cfg.n_segments),
            "nominal_dt": branch_cfg.tf / branch_cfg.n_segments,
            "original_transfer_time": branch_cfg.tf,
            "branch_total_duration": branch_cfg.tf,
            "terminal_error": 0.01,
            "error": 0.01,
            "branch_fuel": 0.0,
            "nfev": 1 if optimizer_ran else 0,
            "runtime_seconds": 0.01,
            "optimizer_ran": optimizer_ran,
            "optimizer_success": optimizer_ran,
            "no_recovery_variable_threshold_feasible": not optimizer_ran,
            "accepted_candidate": "optimizer" if optimizer_ran else "no_recovery_variables",
            "message": "fixture branch",
            "cost": 0.0,
            "optimality": 0.0,
            "branch_weights": branch_weights.as_dict(),
            "branch_controls": np.zeros((branch_cfg.n_segments, 3), dtype=float),
            "branch_controls_remove_zero_nominal": False,
            "branch_missed_tail_indices": [],
            "branch_note": "fixture",
        }

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(tail, "optimize_tail_coast_nominal", fake_tail_nominal)
    monkeypatch.setattr(tail, "optimize_tail_coast_recovery_branch_portfolio", fake_branch_portfolio)

    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        tail_coast_segments=1,
        selected_outages="all_single",
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=1,
    )

    expected_single = np.flatnonzero(np.sum(masks < 0.5, axis=1) == 1).astype(int).tolist()
    length_two = set(np.flatnonzero(np.sum(masks < 0.5, axis=1) == 2).astype(int).tolist())
    assert result["selected_outage_indices"] == expected_single
    assert selected_by_branch == expected_single
    assert set(result["selected_outage_indices"]).isdisjoint(length_two)


def test_all_single_row_portfolio_acceptance_rule_aggregates_fallback_use(monkeypatch):
    cfg = small_cfg(n_segments=4)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    expected_single = np.flatnonzero(np.sum(masks < 0.5, axis=1) == 1).astype(int).tolist()
    fallback_branch_index = expected_single[1]

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 1,
            "runtime_seconds": 0.01,
        }

    def fake_tail_nominal(**kwargs):
        return {
            "controls": controls.copy(),
            "optimizer_success": True,
            "message": "fixture tail nominal",
            "accepted_candidate": "fixture",
            "nfev": 0,
            "runtime_seconds": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
        }

    def fake_branch_portfolio(**kwargs):
        mask_index = int(kwargs["mask_index"])
        start = tail.outage_end(kwargs["mask"])
        recovery_segments = int(kwargs["cfg"].n_segments) - int(start)
        optimizer_ran = bool(recovery_segments > 0)
        accepted_fallback = bool(mask_index == fallback_branch_index and optimizer_ran)
        if accepted_fallback:
            init_label = "constant_y_plus_0p5"
            init_kind = "constant_vector"
        elif optimizer_ran:
            init_label = "nominal_post_outage"
            init_kind = "nominal_post_outage"
        else:
            init_label = "no_recovery_variables"
            init_kind = "no_recovery_variables"
        return {
            "mask_index": mask_index,
            "recovery_start": int(start),
            "recovery_segments": recovery_segments,
            "tail_coast_segments": int(kwargs["tail_coast_segments"]),
            "branch_control_count": int(kwargs["cfg"].n_segments),
            "nominal_segments": int(kwargs["cfg"].n_segments),
            "nominal_dt": kwargs["cfg"].tf / kwargs["cfg"].n_segments,
            "original_transfer_time": kwargs["cfg"].tf,
            "branch_total_duration": kwargs["cfg"].tf,
            "terminal_error": 0.01,
            "error": 0.01,
            "branch_fuel": 0.0,
            "nfev": 0 if not optimizer_ran else (2 if accepted_fallback else 1),
            "runtime_seconds": 0.02 if accepted_fallback else 0.01,
            "optimizer_ran": optimizer_ran,
            "optimizer_success": optimizer_ran,
            "accepted_candidate": "optimizer" if optimizer_ran else "no_recovery_variables",
            "message": "fixture branch",
            "cost": 0.0,
            "optimality": 0.0,
            "branch_controls": np.zeros((kwargs["cfg"].n_segments, 3), dtype=float),
            "branch_controls_remove_zero_nominal": False,
            "branch_missed_tail_indices": [],
            "branch_note": "fixture",
            "accepted_branch_weight_variant_label": "configured",
            "accepted_branch_weight_variant_index": 0,
            "accepted_branch_weights": {},
            "accepted_branch_initialization_label": init_label,
            "accepted_branch_initialization_index": 1 if accepted_fallback else 0,
            "accepted_branch_initialization_is_fallback": accepted_fallback,
            "accepted_branch_initialization_kind": init_kind,
            "accepted_variant_nfev": 1 if optimizer_ran else 0,
            "accepted_variant_runtime_seconds": 0.01,
            "branch_portfolio_enabled": False,
            "branch_portfolio_variant_count": 1,
            "portfolio_acceptance_rule": (
                "fixture later branch evaluated and accepted fallback initialization"
                if accepted_fallback
                else "fixture branch fallback initializations were not evaluated"
            ),
            "portfolio_robust_threshold": 10.0,
            "portfolio_converged_threshold_feasible_candidate_count": 1 if optimizer_ran else 0,
            "portfolio_nominal_converged_threshold_feasible_candidate_count": 0 if accepted_fallback or not optimizer_ran else 1,
            "branch_portfolio_candidate_count": 2 if accepted_fallback else 1,
            "branch_portfolio_candidate_optimizer_success_count": 2 if accepted_fallback else int(optimizer_ran),
            "branch_portfolio_candidate_all_optimizer_success": optimizer_ran,
            "branch_fallback_initialization_enabled": True,
            "branch_fallback_initialization_configured_count": 1,
            "branch_fallback_initialization_evaluated_count": 1 if accepted_fallback else 0,
            "branch_fallback_initialization_candidate_count": 1 if accepted_fallback else 0,
            "branch_nominal_initialization_candidate_count": 1,
            "branch_initialization_variant_count": 2,
        }

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(tail, "optimize_tail_coast_nominal", fake_tail_nominal)
    monkeypatch.setattr(tail, "optimize_tail_coast_recovery_branch_portfolio", fake_branch_portfolio)

    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 10.0},
        tail_coast_segments=1,
        selected_outages="all_single",
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=1,
        branch_initialization_fallbacks=[{"label": "constant_y_plus_0p5", "vector": [0.0, 0.5, 0.0]}],
    )

    assert result["selected_outage_indices"] == expected_single
    assert result["branch_fallback_initialization_any_evaluated"] is True
    assert result["branch_fallback_initialization_any_accepted"] is True
    assert result["branch_fallback_initialization_evaluated_branch_count"] == 1
    assert result["branch_fallback_initialization_accepted_branch_count"] == 1
    assert "fallback initializations were evaluated by 1 branch(es)" in result["portfolio_acceptance_rule"]
    assert "fallback initialization was accepted by 1 branch(es)" in result["portfolio_acceptance_rule"]
    assert result["portfolio_acceptance_rule"] != result["branch_results"][0]["portfolio_acceptance_rule"]
    assert result["branch_results"][1]["accepted_branch_initialization_is_fallback"] is True


def test_branch_portfolio_charges_all_variants_and_accepts_converged_threshold_candidate(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    controls = np.zeros((cfg.n_segments, 3), dtype=float)

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 2,
            "runtime_seconds": 0.01,
        }

    def fake_branch_optimizer(**kwargs):
        weights = kwargs["weights"]
        if weights.control > 0.0:
            terminal_error = 0.08
            fuel = 0.10
            nfev = 3
            runtime = 0.2
            success = True
            message = "regularized fixture"
        else:
            terminal_error = 0.03
            fuel = 0.30
            nfev = 5
            runtime = 0.4
            success = False
            message = "terminal-only fixture nfev cap"
        start = tail.outage_end(kwargs["mask"])
        return {
            "mask_index": int(kwargs["mask_index"]),
            "recovery_start": start,
            "recovery_segments": kwargs["cfg"].n_segments - start,
            "tail_coast_segments": int(kwargs["tail_coast_segments"]),
            "branch_control_count": kwargs["cfg"].n_segments,
            "nominal_segments": kwargs["cfg"].n_segments,
            "nominal_dt": kwargs["cfg"].tf / kwargs["cfg"].n_segments,
            "original_transfer_time": kwargs["cfg"].tf,
            "branch_total_duration": kwargs["cfg"].tf,
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
            "branch_controls": np.zeros((kwargs["cfg"].n_segments, 3), dtype=float),
            "recovery_controls": np.zeros((kwargs["cfg"].n_segments - start, 3), dtype=float),
            "history": None,
            "branch_controls_remove_zero_nominal": False,
            "branch_missed_tail_indices": [],
            "branch_note": "fixture",
            "control_max_norm": 0.0,
            "control_bound_violation": 0.0,
        }

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(tail, "optimize_tail_coast_recovery_branch", fake_branch_optimizer)

    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 10.0, "robust_success": 0.17},
        tail_coast_segments=1,
        selected_outages=1,
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=9,
        branch_weight_variants=[
            {"label": "regularized_001", "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0}},
            {"label": "terminal_only", "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
        ],
    )

    assert result["branch_portfolio_enabled"] is True
    assert result["branch_portfolio_variant_count"] == 2
    assert result["branch_nfev"] == [8]
    assert result["total_branch_nfev"] == 8
    assert result["branch_runtime_seconds"] == [pytest.approx(0.6)]
    assert result["branch_accepted_weight_variant_labels"] == ["regularized_001"]
    assert result["branch_accepted_variant_nfev"] == [3]
    assert result["branch_optimizer_success_by_branch"] == [True]
    assert result["branch_optimizer_all_success"] is True
    assert result["branch_portfolio_all_success"] is True
    assert result["branch_portfolio_candidate_all_optimizer_success"] is False
    assert result["branch_portfolio_candidate_all_optimizer_success_by_branch"] == [False]
    assert "accepted portfolio-selected result" in result["branch_portfolio_all_success_semantics"]
    assert "does not mean every evaluated portfolio candidate" in result["branch_portfolio_all_success_semantics"]
    candidates = result["branch_results"][0]["branch_portfolio_candidate_results"]
    assert "preferred optimizer_success=True" in result["branch_results"][0]["portfolio_acceptance_rule"]
    assert result["branch_results"][0]["branch_portfolio_candidate_all_optimizer_success"] is False
    assert result["branch_results"][0]["branch_portfolio_candidate_count"] == 2
    assert result["branch_results"][0]["branch_portfolio_candidate_optimizer_success_count"] == 1
    assert [candidate["variant_label"] for candidate in candidates] == ["regularized_001", "terminal_only"]
    assert [candidate["nfev"] for candidate in candidates] == [3, 5]


def test_branch_portfolio_fallback_initializations_are_gated_and_charged(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    mask = outage_masks(cfg.n_segments, cfg.outage_lengths)[0]
    controls = np.zeros((cfg.n_segments, 3), dtype=float)
    variants = [
        {"label": "regularized_001", "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0}},
        {"label": "terminal_only", "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
    ]
    fallbacks = [
        {"label": "constant_y_plus_0p5", "vector": [0.0, 0.5, 0.0]},
        {"label": "constant_y_minus_0p5", "vector": [0.0, -0.5, 0.0]},
    ]
    calls = []
    nominal_feasible = {"value": True}

    def fake_branch_optimizer(**kwargs):
        weights = kwargs["weights"]
        is_regularized = bool(weights.control > 0.0)
        is_fallback = bool(kwargs["initialization_is_fallback"])
        init_label = str(kwargs["initialization_label"])
        calls.append(
            {
                "label": init_label,
                "is_fallback": is_fallback,
                "control_weight": float(weights.control),
                "initial_controls": kwargs.get("initial_recovery_controls"),
            }
        )
        if not is_fallback and nominal_feasible["value"] and is_regularized:
            terminal_error, success, nfev, runtime, fuel = 0.05, True, 2, 0.2, 0.30
        elif not is_fallback:
            terminal_error = 0.12 if is_regularized else 0.08
            success = bool(is_regularized)
            nfev = 2 if is_regularized else 3
            runtime = 0.2 if is_regularized else 0.3
            fuel = 0.20 if is_regularized else 0.25
        elif init_label == "constant_y_plus_0p5" and is_regularized:
            terminal_error, success, nfev, runtime, fuel = 0.04, True, 7, 0.7, 0.40
        elif init_label == "constant_y_plus_0p5":
            terminal_error, success, nfev, runtime, fuel = 0.06, True, 8, 0.8, 0.30
        elif is_regularized:
            terminal_error, success, nfev, runtime, fuel = 0.18, True, 5, 0.5, 0.35
        else:
            terminal_error, success, nfev, runtime, fuel = 0.09, False, 6, 0.6, 0.28
        start = tail.outage_end(kwargs["mask"])
        recovery_segments = int(kwargs["cfg"].n_segments) - int(start)
        return {
            "mask_index": int(kwargs["mask_index"]),
            "recovery_start": int(start),
            "recovery_segments": recovery_segments,
            "tail_coast_segments": int(kwargs["tail_coast_segments"]),
            "branch_control_count": int(kwargs["cfg"].n_segments),
            "nominal_segments": int(kwargs["cfg"].n_segments),
            "nominal_dt": kwargs["cfg"].tf / kwargs["cfg"].n_segments,
            "original_transfer_time": kwargs["cfg"].tf,
            "branch_total_duration": kwargs["cfg"].tf,
            "terminal_error": terminal_error,
            "error": terminal_error,
            "branch_fuel": fuel,
            "nfev": nfev,
            "runtime_seconds": runtime,
            "optimizer_ran": True,
            "optimizer_success": success,
            "accepted_candidate": "optimizer",
            "message": f"fixture {init_label}",
            "cost": terminal_error,
            "optimality": 0.0,
            "branch_weights": weights.as_dict(),
            "branch_controls": np.zeros((kwargs["cfg"].n_segments, 3), dtype=float),
            "recovery_controls": np.zeros((recovery_segments, 3), dtype=float),
            "history": None,
            "branch_controls_remove_zero_nominal": False,
            "branch_missed_tail_indices": [],
            "branch_note": "fixture",
            "control_max_norm": 0.0,
            "control_bound_violation": 0.0,
        }

    monkeypatch.setattr(tail, "optimize_tail_coast_recovery_branch", fake_branch_optimizer)

    skipped = tail.optimize_tail_coast_recovery_branch_portfolio(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=controls,
        mask=mask,
        mask_index=0,
        tail_coast_segments=1,
        max_nfev=9,
        robust_threshold=0.1,
        branch_weight_variants=variants,
        branch_initialization_fallbacks=fallbacks,
    )

    assert [call["label"] for call in calls] == ["nominal_post_outage", "nominal_post_outage"]
    assert [call["is_fallback"] for call in calls] == [False, False]
    assert skipped["accepted_branch_initialization_label"] == "nominal_post_outage"
    assert skipped["accepted_branch_initialization_is_fallback"] is False
    assert skipped["branch_fallback_initialization_configured_count"] == 2
    assert skipped["branch_fallback_initialization_evaluated_count"] == 0
    assert skipped["branch_fallback_initialization_candidate_count"] == 0

    calls.clear()
    nominal_feasible["value"] = False
    recovered = tail.optimize_tail_coast_recovery_branch_portfolio(
        state0=state0,
        target=target,
        cfg=cfg,
        nominal_controls=controls,
        mask=mask,
        mask_index=0,
        tail_coast_segments=1,
        max_nfev=9,
        robust_threshold=0.1,
        branch_weight_variants=variants,
        branch_initialization_fallbacks=fallbacks,
    )

    assert [call["label"] for call in calls] == [
        "nominal_post_outage",
        "nominal_post_outage",
        "constant_y_plus_0p5",
        "constant_y_plus_0p5",
        "constant_y_minus_0p5",
        "constant_y_minus_0p5",
    ]
    assert [call["is_fallback"] for call in calls] == [False, False, True, True, True, True]
    assert calls[2]["initial_controls"].shape == (cfg.n_segments - tail.outage_end(mask), 3)
    assert np.max(np.linalg.norm(calls[2]["initial_controls"], axis=1)) <= cfg.amax
    assert recovered["nfev"] == 31
    assert recovered["runtime_seconds"] == pytest.approx(3.1)
    assert recovered["accepted_variant_nfev"] == 7
    assert recovered["accepted_variant_runtime_seconds"] == pytest.approx(0.7)
    assert recovered["accepted_branch_weight_variant_label"] == "regularized_001"
    assert recovered["accepted_branch_initialization_label"] == "constant_y_plus_0p5"
    assert recovered["accepted_branch_initialization_index"] == 1
    assert recovered["accepted_branch_initialization_is_fallback"] is True
    assert recovered["branch_fallback_initialization_evaluated_count"] == 2
    assert recovered["branch_fallback_initialization_candidate_count"] == 4
    assert recovered["branch_nominal_initialization_candidate_count"] == 2
    assert recovered["portfolio_converged_threshold_feasible_candidate_count"] == 2
    summaries = recovered["branch_portfolio_candidate_results"]
    assert [item["initialization_label"] for item in summaries] == [call["label"] for call in calls]
    assert [item["initialization_is_fallback"] for item in summaries] == [call["is_fallback"] for call in calls]


def test_tail_coast_portfolio_no_recovery_variables_are_not_converged(monkeypatch):
    cfg = small_cfg(n_segments=3, tf=0.1, amax=0.05)
    state0, target = state_pair()
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    final_mask_index = next(
        index for index, mask in enumerate(masks)
        if tail.outage_end(mask) == cfg.n_segments
    )
    controls = np.zeros((cfg.n_segments, 3), dtype=float)

    def fake_multiple_shooting(**kwargs):
        return {
            "success": True,
            "optimizer_success": True,
            "message": "fixture nominal",
            "accepted_candidate": "fixture",
            "controls": controls.copy(),
            "nominal_error": 0.01,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 2,
            "runtime_seconds": 0.01,
        }

    def fake_tail_nominal(**kwargs):
        return {
            "controls": controls.copy(),
            "optimizer_success": True,
            "message": "fixture tail nominal",
            "accepted_candidate": "fixture",
            "nfev": 0,
            "runtime_seconds": 0.01,
        }

    def fake_select(**kwargs):
        return (
            np.array([final_mask_index], dtype=int),
            np.full(masks.shape[0], 0.01, dtype=float),
            "fixture selects final outage",
        )

    monkeypatch.setattr(tail, "run_multiple_shooting_baseline", fake_multiple_shooting)
    monkeypatch.setattr(tail, "optimize_tail_coast_nominal", fake_tail_nominal)
    monkeypatch.setattr(tail, "select_tail_coast_outage_indices", fake_select)

    result = tail.run_tail_coast_recovery_baseline(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        thresholds={"nominal_success": 1.0e9, "robust_success": 1.0e9},
        tail_coast_segments=1,
        selected_outages=1,
        nominal_max_nfev=1,
        tail_nominal_max_nfev=0,
        branch_max_nfev=9,
        branch_weight_variants=[
            {"label": "regularized_001", "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0}},
            {"label": "terminal_only", "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
        ],
    )

    assert result["selected_outage_indices"] == [final_mask_index]
    assert result["branch_recovery_segments"] == [0]
    assert result["branch_optimizer_ran_by_branch"] == [False]
    assert result["branch_optimizer_success_by_branch"] == [False]
    assert result["branch_optimizer_all_success"] is False
    assert result["branch_optimizer_success"] is False
    assert result["branch_portfolio_all_success"] is False
    assert result["optimizer_success"] is False
    assert result["backend_success"] is True
    assert result["meets_thresholds"] is True
    assert result["branch_portfolio_converged_threshold_feasible_candidate_counts"] == [0]
    assert result["branch_accepted_initialization_labels"] == ["no_recovery_variables"]
    assert result["branch_accepted_initialization_kinds"] == ["no_recovery_variables"]
    assert "no fallback initializations were configured" in result["portfolio_acceptance_rule"]

    branch = result["branch_results"][0]
    assert branch["optimizer_ran"] is False
    assert branch["optimizer_success"] is False
    assert branch["no_recovery_variable_threshold_feasible"] is True
    assert branch["accepted_branch_initialization_label"] == "no_recovery_variables"
    assert branch["accepted_branch_initialization_kind"] == "no_recovery_variables"
    assert branch["portfolio_converged_threshold_feasible_candidate_count"] == 0
    assert "no optimizer-converged threshold-feasible candidate" in branch["portfolio_acceptance_rule"]
    candidates = branch["branch_portfolio_candidate_results"]
    assert [candidate["variant_label"] for candidate in candidates] == ["regularized_001", "terminal_only"]
    assert [candidate["initialization_label"] for candidate in candidates] == ["no_recovery_variables", "no_recovery_variables"]
    assert [candidate["initialization_kind"] for candidate in candidates] == ["no_recovery_variables", "no_recovery_variables"]
    assert [candidate["optimizer_ran"] for candidate in candidates] == [False, False]
    assert [candidate["optimizer_success"] for candidate in candidates] == [False, False]
    assert [candidate["threshold_feasible"] for candidate in candidates] == [True, True]
    assert [candidate["converged_threshold_feasible"] for candidate in candidates] == [False, False]


def test_runner_artifact_generation_with_fake_backend(monkeypatch, tmp_path):
    script = load_script_module("run_tail_coast_recovery_test", ROOT / "scripts" / "run_tail_coast_recovery.py")
    config = {
        "run": {"label": "tail_test", "output_subdir": "tail_test"},
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
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "nominal": {"max_nfev": 1, "node_initialization": "linear", "node_initialization_blend": 0.5},
            "tail_nominal": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "branch": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "cases": [
                {
                    "case_id": "tail_unit",
                    "purpose": "unit test",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 3,
                    "tail_coast_segments": 1,
                    "selected_outages": 0,
                    "nominal_max_nfev": 1,
                    "tail_nominal_max_nfev": 1,
                    "branch_max_nfev": 0,
                }
            ],
        },
    }
    config_path = tmp_path / "tail_config.yaml"
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
            "mode": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "method_type": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "optimizer_success": False,
            "optimizer_success_semantics": "fixture overall optimizer success",
            "nominal_optimizer_success": True,
            "nominal_seed_optimizer_success": True,
            "nominal_tail_optimizer_success": True,
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
            "message": "fixture ok",
            "nominal_message": "fixture nominal ok",
            "nominal_seed_message": "fixture seed ok",
            "nominal_accepted_candidate": "fixture",
            "nominal_error": 0.01,
            "nominal_tail_coast_error": 0.01,
            "nominal_seed_error": 0.02,
            "nominal_baseline_error": 0.02,
            "nominal_lock_error_delta": 0.0,
            "tail_coast_segments": 1,
            "optimized_nominal_segments": cfg.n_segments - 1,
            "nominal_tail_zero_max_abs": 0.0,
            "nominal_tail_control_norm_max": 0.0,
            "nominal_dt": cfg.tf / cfg.n_segments,
            "branch_total_duration": cfg.tf,
            "branch_control_count": cfg.n_segments,
            "original_target_state": np.ones(6, dtype=float) * 0.01,
            "worst_error_semantics": "fixture worst",
            "worst_error": 0.01,
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
            "branch_controls_remove_zero_nominal": [],
            "total_branch_nfev": 0,
            "total_branch_runtime_seconds": 0.0,
            "nominal_seed_nfev": 1,
            "nominal_tail_nfev": 1,
            "nominal_nfev": 2,
            "nominal_seed_runtime_seconds": 0.01,
            "nominal_tail_runtime_seconds": 0.01,
            "nominal_runtime_seconds": 0.02,
            "nfev": 2,
            "runtime_seconds": 0.02,
            "cost": 0.0,
            "optimality": 0.0,
            "nominal_cost": 0.0,
            "nominal_optimality": 0.0,
            "backend_semantics": "fixture backend fixed-final-time",
            "selection_semantics": "fixture selection",
            "selected_branch_semantics": "fixture selected",
            "all_mask_diagnostic_semantics": "fixture all mask",
            "control_bound_semantics": "fixture bound",
            "nominal_lock_semantics": "fixture lock",
            "fixed_final_time_semantics": "fixture fixed final time",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)
    monkeypatch.setattr(script, "run_tail_coast_recovery_baseline", fake_backend)
    args = script.build_parser().parse_args(["--config", str(config_path), "--source-states", str(source_states), "--resume"])
    df = script.run(args)

    results_dir = tmp_path / "data" / "results" / "tail_test"
    tables_dir = tmp_path / "tables" / "tail_test"
    figures_dir = tmp_path / "figures" / "tail_test"
    assert len(df) == 1
    assert (results_dir / "tail_coast_recovery.csv").exists()
    assert (results_dir / "tail_coast_recovery_metadata.json").exists()
    assert (tables_dir / "tail_coast_recovery_table.tex").exists()
    assert (figures_dir / "tail_coast_recovery.png").exists()
    assert (figures_dir / "tail_coast_recovery.pdf").exists()

    csv_df = pd.read_csv(results_dir / "tail_coast_recovery.csv")
    assert csv_df.iloc[0]["tail_coast_segments"] == 1
    assert bool(csv_df.iloc[0]["branch_optimizer_ran"]) is False
    assert bool(csv_df.iloc[0]["branch_optimizer_all_success"]) is False
    assert json.loads(csv_df.iloc[0]["branch_results"]) == []
    metadata = json.loads((results_dir / "tail_coast_recovery_metadata.json").read_text(encoding="utf-8"))
    assert metadata["row_count"] == 1
    assert "tail_coast_recovery_module" in metadata["implementation_identities"]
    assert "Fixed-final-time" in metadata["semantics"]["backend"]


def test_runner_writes_branch_control_sidecars_with_fake_backend(monkeypatch, tmp_path):
    script = load_script_module("run_tail_coast_recovery_sidecar_test", ROOT / "scripts" / "run_tail_coast_recovery.py")
    config = {
        "run": {"label": "tail_sidecar_test", "output_subdir": "tail_sidecar_test"},
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
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "nominal": {"max_nfev": 1, "node_initialization": "linear", "node_initialization_blend": 0.5},
            "tail_nominal": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "branch": {"max_nfev": 1, "weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
            "cases": [
                {
                    "case_id": "tail_sidecar_unit",
                    "purpose": "unit sidecar test",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 3,
                    "tail_coast_segments": 1,
                    "selected_outages": "all_configured",
                    "nominal_max_nfev": 1,
                    "tail_nominal_max_nfev": 1,
                    "branch_max_nfev": 1,
                }
            ],
        },
    }
    config_path = tmp_path / "tail_sidecar_config.yaml"
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
        nominal = np.array(
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.02, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        branch0 = nominal.copy()
        branch0[0] = 0.0
        branch1 = nominal.copy()
        branch1[1:] = np.array([[0.03, 0.0, 0.0], [0.0, 0.0, 0.0]])
        branch_records = [
            {
                "mask_index": 0,
                "recovery_start": 1,
                "recovery_segments": 2,
                "tail_coast_segments": 1,
                "branch_control_count": cfg.n_segments,
                "nominal_dt": cfg.tf / cfg.n_segments,
                "branch_total_duration": cfg.tf,
                "terminal_error": 0.03,
                "branch_fuel": 0.001,
                "nfev": 2,
                "runtime_seconds": 0.01,
                "optimizer_ran": True,
                "optimizer_success": True,
                "accepted_candidate": "optimizer",
                "message": "fixture branch 0",
                "accepted_branch_weight_variant_label": "regularized_001",
                "accepted_branch_weight_variant_index": 0,
                "accepted_branch_weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
                "accepted_branch_initialization_label": "nominal_post_outage",
                "accepted_branch_initialization_index": 0,
                "accepted_branch_initialization_is_fallback": False,
                "accepted_branch_initialization_kind": "nominal_post_outage",
                "accepted_variant_nfev": 2,
                "accepted_variant_runtime_seconds": 0.01,
                "branch_controls": branch0,
                "recovery_controls": branch0[1:],
                "control_max_norm": 0.02,
                "control_bound_violation": 0.0,
            },
            {
                "mask_index": 1,
                "recovery_start": 2,
                "recovery_segments": 1,
                "tail_coast_segments": 1,
                "branch_control_count": cfg.n_segments,
                "nominal_dt": cfg.tf / cfg.n_segments,
                "branch_total_duration": cfg.tf,
                "terminal_error": 0.04,
                "branch_fuel": 0.002,
                "nfev": 3,
                "runtime_seconds": 0.02,
                "optimizer_ran": True,
                "optimizer_success": True,
                "accepted_candidate": "optimizer",
                "message": "fixture branch 1",
                "accepted_branch_weight_variant_label": "terminal_only",
                "accepted_branch_weight_variant_index": 1,
                "accepted_branch_weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
                "accepted_branch_initialization_label": "constant_y_plus_0p5",
                "accepted_branch_initialization_index": 1,
                "accepted_branch_initialization_is_fallback": True,
                "accepted_branch_initialization_kind": "constant_vector",
                "accepted_branch_initialization_vector": [0.0, 0.5, 0.0],
                "accepted_variant_nfev": 3,
                "accepted_variant_runtime_seconds": 0.02,
                "branch_controls": branch1,
                "recovery_controls": branch1[2:],
                "control_max_norm": 0.03,
                "control_bound_violation": 0.0,
            },
        ]
        branch_results = [
            {key: value for key, value in record.items() if key not in {"branch_controls", "recovery_controls"}}
            for record in branch_records
        ]
        return {
            "success": True,
            "backend_success": True,
            "mode": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "method_type": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "optimizer_success": True,
            "optimizer_success_semantics": "fixture optimizer success",
            "nominal_optimizer_success": True,
            "nominal_seed_optimizer_success": True,
            "nominal_tail_optimizer_success": True,
            "nominal_backend_success": True,
            "branch_optimizer_success": True,
            "branch_optimizer_all_success": True,
            "branch_optimizer_ran": True,
            "branch_portfolio_enabled": True,
            "branch_portfolio_variant_count": 2,
            "branch_portfolio_variant_labels": ["regularized_001", "terminal_only"],
            "branch_portfolio_all_success": True,
            "branch_portfolio_all_success_semantics": "fixture",
            "portfolio_acceptance_rule": "fixture",
            "branch_portfolio_converged_threshold_feasible_candidate_counts": [1, 1],
            "branch_portfolio_candidate_counts": [2, 2],
            "branch_portfolio_candidate_optimizer_success_counts": [2, 2],
            "branch_portfolio_candidate_all_optimizer_success": True,
            "branch_portfolio_candidate_all_optimizer_success_by_branch": [True, True],
            "message": "fixture ok",
            "nominal_message": "fixture nominal ok",
            "nominal_seed_message": "fixture seed ok",
            "nominal_accepted_candidate": "fixture",
            "nominal_error": 0.01,
            "nominal_tail_coast_error": 0.01,
            "nominal_seed_error": 0.02,
            "nominal_baseline_error": 0.02,
            "nominal_lock_error_delta": 0.0,
            "tail_coast_segments": 1,
            "optimized_nominal_segments": cfg.n_segments - 1,
            "nominal_tail_zero_max_abs": 0.0,
            "nominal_tail_control_norm_max": 0.0,
            "nominal_dt": cfg.tf / cfg.n_segments,
            "branch_total_duration": cfg.tf,
            "branch_control_count": cfg.n_segments,
            "original_target_state": np.ones(6, dtype=float) * 0.01,
            "worst_error_semantics": "fixture worst",
            "worst_error": 0.04,
            "selected_recovery_worst_error": 0.04,
            "selected_worst_error": 0.04,
            "all_outage_worst_error": 0.04,
            "all_mask_worst_error": 0.04,
            "nominal_threshold": 0.09,
            "selected_recovery_threshold": 0.17,
            "selected_worst_threshold": 0.17,
            "meets_nominal_threshold": True,
            "meets_selected_recovery_threshold": True,
            "meets_selected_worst_threshold": True,
            "meets_thresholds": True,
            "nominal_fuel": 0.001,
            "recovery_fuel_mean": 0.0015,
            "recovery_fuel_max": 0.002,
            "control_max_norm": 0.03,
            "control_bound_violation": 0.0,
            "controls": nominal,
            "nominal_controls": nominal,
            "accepted_branch_control_results": branch_records,
            "selected_outage_indices": [0, 1],
            "selected_outage_errors": [0.03, 0.04],
            "all_outage_errors": [0.03, 0.04, 0.02],
            "nominal_masked_outage_errors": [0.2, 0.18, 0.16],
            "branch_results": branch_results,
            "branch_nfev": [2, 3],
            "branch_runtime_seconds": [0.01, 0.02],
            "branch_optimizer_success_by_branch": [True, True],
            "branch_optimizer_ran_by_branch": [True, True],
            "branch_accepted_weight_variant_labels": ["regularized_001", "terminal_only"],
            "branch_accepted_weight_variant_indices": [0, 1],
            "branch_accepted_weights": [
                {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
                {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
            ],
            "branch_accepted_initialization_labels": ["nominal_post_outage", "constant_y_plus_0p5"],
            "branch_accepted_initialization_indices": [0, 1],
            "branch_accepted_initialization_is_fallback": [False, True],
            "branch_accepted_initialization_kinds": ["nominal_post_outage", "constant_vector"],
            "branch_accepted_variant_nfev": [2, 3],
            "branch_accepted_variant_runtime_seconds": [0.01, 0.02],
            "branch_recovery_starts": [1, 2],
            "branch_recovery_segments": [2, 1],
            "branch_control_counts": [3, 3],
            "branch_controls_remove_zero_nominal": [False, False],
            "total_branch_nfev": 5,
            "total_branch_runtime_seconds": 0.03,
            "nominal_seed_nfev": 1,
            "nominal_tail_nfev": 1,
            "nominal_nfev": 2,
            "nominal_seed_runtime_seconds": 0.01,
            "nominal_tail_runtime_seconds": 0.01,
            "nominal_runtime_seconds": 0.02,
            "nfev": 7,
            "runtime_seconds": 0.05,
            "cost": 0.0,
            "optimality": 0.0,
            "nominal_cost": 0.0,
            "nominal_optimality": 0.0,
            "backend_semantics": "fixture backend fixed-final-time",
            "selection_semantics": "fixture selection",
            "selected_branch_semantics": "fixture selected",
            "all_mask_diagnostic_semantics": "fixture all mask",
            "control_bound_semantics": "fixture bound",
            "nominal_lock_semantics": "fixture lock",
            "fixed_final_time_semantics": "fixture fixed final time",
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)
    monkeypatch.setattr(script, "run_tail_coast_recovery_baseline", fake_backend)
    args = script.build_parser().parse_args(["--config", str(config_path), "--source-states", str(source_states), "--resume"])
    df = script.run(args)

    results_dir = tmp_path / "data" / "results" / "tail_sidecar_test"
    csv_df = pd.read_csv(results_dir / "tail_coast_recovery.csv")
    row = csv_df.iloc[0]
    assert len(df) == 1
    assert int(row["branch_control_sidecar_count"]) == 2
    assert str(row["branch_control_replay_ready"]).lower() == "true"
    assert len(str(row["nominal_control_sha256"])) == 64
    assert len(str(row["branch_control_manifest_sha256"])) == 64

    nominal_path = tmp_path / str(row["nominal_control_path"])
    manifest_path = tmp_path / str(row["branch_control_manifest_path"])
    assert hashlib.sha256(nominal_path.read_bytes()).hexdigest() == row["nominal_control_sha256"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == row["branch_control_manifest_sha256"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["branch_count"] == 2
    assert manifest["branch_control_replay_ready"] is True
    assert "no high-fidelity validation" in manifest["replay_semantics"]
    assert manifest["nominal_control_sha256"] == row["nominal_control_sha256"]
    assert len(manifest["branch_control_sidecars"]) == 2
    first_sidecar = manifest["branch_control_sidecars"][0]
    first_path = tmp_path / first_sidecar["path"]
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_sidecar["sha256"]
    first_branch = json.loads(first_path.read_text(encoding="utf-8"))
    assert first_branch["mask_index"] == 0
    assert first_branch["outage_mask"] == [0, 1, 1]
    assert first_branch["recovery_start"] == 1
    assert first_branch["recovery_segments"] == 2
    assert first_branch["branch_controls"][0] == [0.0, 0.0, 0.0]
    assert first_branch["recovery_controls"] == first_branch["branch_controls"][1:]
    assert first_branch["optimizer_ran"] is True
    assert first_branch["optimizer_success"] is True


def test_runner_resume_reuses_incremental_branch_sidecars(monkeypatch, tmp_path):
    script = load_script_module(
        "run_tail_coast_recovery_incremental_resume_test",
        ROOT / "scripts" / "run_tail_coast_recovery.py",
    )
    config = {
        "run": {"label": "tail_resume_test", "output_subdir": "tail_resume_test"},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_dro_phase",
            "transfer_time": 0.12,
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
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "cases": [
                {
                    "case_id": "tail_resume_unit",
                    "purpose": "unit resume",
                    "transfer_time": 0.12,
                    "amax": 0.2,
                    "segments": 3,
                    "tail_coast_segments": 1,
                    "selected_outages": 2,
                    "nominal_max_nfev": 1,
                    "tail_nominal_max_nfev": 1,
                    "branch_max_nfev": 1,
                }
            ],
        },
    }
    config_path = tmp_path / "tail_resume_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    cfg = script.make_objective_config(config, 0.01215058560962404)
    nominal = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    branch0 = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
    branch1 = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=float)

    def branch_record(mask_index: int, controls: np.ndarray, terminal_error: float, nfev: int) -> dict:
        return {
            "mask_index": mask_index,
            "recovery_start": mask_index + 1,
            "recovery_segments": cfg.n_segments - (mask_index + 1),
            "tail_coast_segments": 1,
            "branch_control_count": cfg.n_segments,
            "nominal_dt": cfg.tf / cfg.n_segments,
            "branch_total_duration": cfg.tf,
            "accepted_candidate": "optimizer",
            "optimizer_ran": True,
            "optimizer_success": True,
            "message": "fixture branch",
            "nfev": nfev,
            "runtime_seconds": 0.1 * nfev,
            "terminal_error": terminal_error,
            "branch_fuel": 0.001 * (nfev + 1),
            "accepted_branch_weight_variant_label": "fixture",
            "accepted_branch_weight_variant_index": 0,
            "accepted_branch_weights": {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
            "accepted_branch_initialization_label": "nominal_post_outage",
            "accepted_branch_initialization_index": 0,
            "accepted_branch_initialization_is_fallback": False,
            "accepted_branch_initialization_kind": "nominal_post_outage",
            "accepted_variant_nfev": nfev,
            "accepted_variant_runtime_seconds": 0.1 * nfev,
            "branch_controls": controls,
            "recovery_controls": controls[mask_index + 1 :],
            "control_max_norm": float(np.linalg.norm(controls, axis=1).max()),
            "control_bound_violation": 0.0,
            "branch_controls_remove_zero_nominal": False,
            "branch_missed_tail_indices": [],
            "branch_note": "fixture",
        }

    branch0_record = branch_record(0, branch0, 0.03, 2)
    branch1_record = branch_record(1, branch1, 0.04, 3)

    def make_result(records: list[dict]) -> dict:
        branch_results = [
            {key: value for key, value in record.items() if key not in {"branch_controls", "recovery_controls"}}
            for record in records
        ]
        return {
            "success": True,
            "backend_success": True,
            "mode": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "method_type": "fixed_final_time_tail_coast_locked_nominal_independent_branch_recovery",
            "optimizer_success": True,
            "optimizer_success_semantics": "fixture",
            "nominal_optimizer_success": True,
            "nominal_seed_optimizer_success": True,
            "nominal_tail_optimizer_success": True,
            "nominal_backend_success": True,
            "branch_optimizer_success": True,
            "branch_optimizer_all_success": True,
            "branch_optimizer_ran": bool(records),
            "branch_portfolio_enabled": False,
            "branch_portfolio_variant_count": 1,
            "branch_portfolio_variant_labels": ["fixture"],
            "branch_portfolio_all_success": True,
            "branch_portfolio_all_success_semantics": "fixture",
            "portfolio_acceptance_rule": "fixture",
            "branch_portfolio_converged_threshold_feasible_candidate_counts": [1 for _ in records],
            "branch_portfolio_candidate_counts": [1 for _ in records],
            "branch_portfolio_candidate_optimizer_success_counts": [1 for _ in records],
            "branch_portfolio_candidate_all_optimizer_success": True,
            "branch_portfolio_candidate_all_optimizer_success_by_branch": [True for _ in records],
            "message": "fixture ok",
            "nominal_message": "fixture nominal",
            "nominal_seed_message": "fixture seed",
            "nominal_accepted_candidate": "fixture",
            "nominal_error": 0.01,
            "nominal_tail_coast_error": 0.01,
            "nominal_seed_error": 0.02,
            "nominal_baseline_error": 0.02,
            "nominal_lock_error_delta": 0.0,
            "tail_coast_segments": 1,
            "optimized_nominal_segments": cfg.n_segments - 1,
            "nominal_tail_zero_max_abs": 0.0,
            "nominal_tail_control_norm_max": 0.0,
            "nominal_dt": cfg.tf / cfg.n_segments,
            "branch_total_duration": cfg.tf,
            "branch_control_count": cfg.n_segments,
            "original_target_state": np.ones(6, dtype=float) * 0.01,
            "worst_error_semantics": "fixture",
            "worst_error": max([float(record["terminal_error"]) for record in records], default=0.01),
            "selected_recovery_worst_error": max([float(record["terminal_error"]) for record in records], default=0.01),
            "selected_worst_error": max([float(record["terminal_error"]) for record in records], default=0.01),
            "all_outage_worst_error": max([float(record["terminal_error"]) for record in records], default=0.01),
            "all_mask_worst_error": max([float(record["terminal_error"]) for record in records], default=0.01),
            "nominal_threshold": 0.09,
            "selected_recovery_threshold": 0.17,
            "selected_worst_threshold": 0.17,
            "meets_nominal_threshold": True,
            "meets_selected_recovery_threshold": True,
            "meets_selected_worst_threshold": True,
            "meets_thresholds": True,
            "nominal_fuel": 0.001,
            "recovery_fuel_mean": 0.002,
            "recovery_fuel_max": 0.003,
            "control_max_norm": 0.03,
            "control_bound_violation": 0.0,
            "controls": nominal,
            "nominal_controls": nominal,
            "accepted_branch_control_results": records,
            "selected_outage_count": 2,
            "selected_outage_indices": [0, 1],
            "selected_outage_errors": [float(record["terminal_error"]) for record in records],
            "all_outage_errors": [0.03, 0.04, 0.02],
            "nominal_masked_outage_errors": [0.2, 0.18, 0.16],
            "branch_results": branch_results,
            "branch_nfev": [int(record["nfev"]) for record in records],
            "branch_runtime_seconds": [float(record["runtime_seconds"]) for record in records],
            "branch_optimizer_success_by_branch": [True for _ in records],
            "branch_optimizer_ran_by_branch": [True for _ in records],
            "branch_accepted_weight_variant_labels": ["fixture" for _ in records],
            "branch_accepted_weight_variant_indices": [0 for _ in records],
            "branch_accepted_weights": [
                {"terminal": 1.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}
                for _ in records
            ],
            "branch_accepted_initialization_labels": ["nominal_post_outage" for _ in records],
            "branch_accepted_initialization_indices": [0 for _ in records],
            "branch_accepted_initialization_is_fallback": [False for _ in records],
            "branch_accepted_initialization_kinds": ["nominal_post_outage" for _ in records],
            "branch_accepted_variant_nfev": [int(record["nfev"]) for record in records],
            "branch_accepted_variant_runtime_seconds": [float(record["runtime_seconds"]) for record in records],
            "branch_recovery_starts": [int(record["recovery_start"]) for record in records],
            "branch_recovery_segments": [int(record["recovery_segments"]) for record in records],
            "branch_control_counts": [cfg.n_segments for _ in records],
            "branch_controls_remove_zero_nominal": [False for _ in records],
            "total_branch_nfev": sum(int(record["nfev"]) for record in records),
            "total_branch_runtime_seconds": sum(float(record["runtime_seconds"]) for record in records),
            "nominal_seed_nfev": 1,
            "nominal_tail_nfev": 1,
            "nominal_nfev": 2,
            "nominal_seed_runtime_seconds": 0.01,
            "nominal_tail_runtime_seconds": 0.01,
            "nominal_runtime_seconds": 0.02,
            "nfev": 2 + sum(int(record["nfev"]) for record in records),
            "runtime_seconds": 0.02 + sum(float(record["runtime_seconds"]) for record in records),
            "cost": 0.0,
            "optimality": 0.0,
            "nominal_cost": 0.0,
            "nominal_optimality": 0.0,
            "backend_semantics": "fixture",
            "selection_semantics": "fixture",
            "selected_branch_semantics": "fixture",
            "all_mask_diagnostic_semantics": "fixture",
            "control_bound_semantics": "fixture",
            "nominal_lock_semantics": "fixture",
            "fixed_final_time_semantics": "fixture",
        }

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=np.zeros(6, dtype=float),
            target=np.ones(6, dtype=float) * 0.01,
            target_metadata={"target_state_generation": "fixture"},
        )

    def interrupted_backend(**kwargs):
        assert kwargs["branch_result_overrides"] == {}
        kwargs["accepted_nominal_callback"](make_result([]))
        kwargs["accepted_branch_callback"](branch0_record, 0)
        raise RuntimeError("fixture interrupt")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)
    monkeypatch.setattr(script, "run_tail_coast_recovery_baseline", interrupted_backend)
    args = script.build_parser().parse_args(["--config", str(config_path), "--source-states", str(source_states), "--resume"])
    with pytest.raises(RuntimeError, match="fixture interrupt"):
        script.run(args)

    results_dir = tmp_path / "data" / "results" / "tail_resume_test"
    controls_dir = results_dir / "controls"
    manifest_path = controls_dir / "tail_resume_unit_branch_control_manifest.json"
    progress_path = controls_dir / "tail_resume_unit_branch_control_progress.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["progress_state"] == "in_progress"
    assert manifest["branch_control_replay_ready"] is False
    assert manifest["branch_control_sidecar_count"] == 1
    progress = pd.read_csv(progress_path)
    assert progress["record_type"].tolist() == ["nominal", "branch"]
    assert progress.iloc[-1]["status"] == "last_completed_checkpoint"

    resume_seen = {"mask0_override": False}

    def resumed_backend(**kwargs):
        overrides = kwargs["branch_result_overrides"]
        assert sorted(overrides) == [0]
        assert int(overrides[0]["mask_index"]) == 0
        resume_seen["mask0_override"] = True
        kwargs["accepted_nominal_callback"](make_result([]))
        kwargs["accepted_branch_callback"](branch1_record, 1)
        return make_result([overrides[0], branch1_record])

    monkeypatch.setattr(script, "run_tail_coast_recovery_baseline", resumed_backend)
    df = script.run(args)

    assert resume_seen["mask0_override"] is True
    row = df.iloc[0]
    assert int(row["branch_control_sidecar_count"]) == 2
    assert str(row["branch_control_replay_ready"]).lower() == "true"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["progress_state"] == "complete"
    assert manifest["branch_control_replay_ready"] is True
    assert manifest["branch_control_sidecar_count"] == 2
    progress = pd.read_csv(progress_path)
    assert progress["record_type"].tolist() == ["nominal", "branch", "branch"]
    assert progress["status"].tolist() == ["complete", "complete", "complete"]


def test_runner_effective_settings_include_implementation_config_and_variant_settings(tmp_path):
    script = load_script_module("run_tail_coast_recovery_settings_test", ROOT / "scripts" / "run_tail_coast_recovery.py")
    config = {
        "run": {"label": "tail_test", "output_subdir": "tail_test"},
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
            "weights": {"nominal": 1.0, "robust_worst": 0.85, "robust_degradation": 0.55, "active_fraction": 0.08, "smoothness": 0.04},
            "thresholds": {"nominal_success": 0.09, "robust_success": 0.17},
        },
        "outages": {"block_lengths": [1]},
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "branch": {
                "fallback_initializations": [
                    {"label": "constant_y_plus_0p5", "vector": [0.0, 0.5, 0.0]},
                    {"label": "constant_y_minus_0p5", "vector": [0.0, -0.5, 0.0]},
                ],
                "weight_variants": [
                    {"label": "regularized_001", "weights": {"terminal": 4.0, "control": 0.01, "smooth": 0.01, "continuity": 0.0}},
                    {"label": "terminal_only", "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0}},
                ]
            },
            "cases": [{"case_id": "settings", "transfer_time": 0.1, "amax": 0.2, "segments": 3, "tail_coast_segments": 1, "selected_outages": 0}],
        },
    }
    args = SimpleNamespace(source_states=tmp_path / "missing_source_states.json")
    case = script._suite_cases(config)[0]
    settings = script._effective_settings(config, args, case)
    assert "implementation_identities" in settings
    assert "tail_coast_recovery_module" in settings["implementation_identities"]
    assert "cr3bp_module" in settings["implementation_identities"]
    assert "multiple_shooting_module" in settings["implementation_identities"]
    assert "objective_module" in settings["implementation_identities"]
    assert "refinement_module" in settings["implementation_identities"]
    assert settings["case"]["tail_coast_segments"] == 1
    assert [variant["label"] for variant in settings["branch_weight_variants"]] == ["regularized_001", "terminal_only"]
    assert [variant["label"] for variant in settings["branch_initialization_fallbacks"]] == [
        "constant_y_plus_0p5",
        "constant_y_minus_0p5",
    ]
    assert settings["branch_initialization_fallbacks"][0]["vector"] == [0.0, 0.5, 0.0]

    changed = json.loads(json.dumps(config))
    changed["tail_coast_recovery"]["branch"]["weight_variants"][0]["weights"]["control"] = 0.02
    changed_case = script._suite_cases(changed)[0]
    changed_settings = script._effective_settings(changed, args, changed_case)
    assert script.settings_fingerprint(settings) != script.settings_fingerprint(changed_settings)

    changed_fallback = json.loads(json.dumps(config))
    changed_fallback["tail_coast_recovery"]["branch"]["fallback_initializations"][0]["vector"] = [0.0, 0.25, 0.0]
    changed_fallback_case = script._suite_cases(changed_fallback)[0]
    changed_fallback_settings = script._effective_settings(changed_fallback, args, changed_fallback_case)
    assert script.settings_fingerprint(settings) != script.settings_fingerprint(changed_fallback_settings)


def test_tail_coast_branch_control_replay_script_with_generated_sidecars(monkeypatch, tmp_path):
    replay = load_script_module(
        "run_tail_coast_branch_control_replay_unit",
        ROOT / "scripts" / "run_tail_coast_branch_control_replay.py",
    )
    config = {
        "run": {"label": "tail_replay_test", "output_subdir": "tail_replay_test"},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_dro_phase",
            "transfer_time": 0.1,
            "segments": 2,
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
        "tail_coast_recovery": {
            "tail_coast_segments": 1,
            "cases": [
                {
                    "case_id": "tail_replay_unit",
                    "purpose": "unit replay",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 2,
                    "tail_coast_segments": 1,
                    "selected_outages": "all_configured",
                    "nominal_max_nfev": 1,
                    "tail_nominal_max_nfev": 1,
                    "branch_max_nfev": 1,
                }
            ],
        },
    }
    config_path = tmp_path / "tail_replay_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "data" / "results" / "tail_replay_test"
    controls_dir = results_dir / "controls"
    controls_dir.mkdir(parents=True)

    cfg = replay.make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    nominal_controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    target = replay.propagate_controls_batch(initial, nominal_controls, cfg.mu, cfg.tf, cfg.substeps)[0][0]
    branch0_controls = np.array([[0.0, 0.0, 0.0], [0.015, 0.0, 0.0]], dtype=float)
    branch1_controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
    nominal_error = replay._replay_terminal_error(initial, target, cfg, nominal_controls)
    branch0_error = replay._replay_terminal_error(initial, target, cfg, branch0_controls)
    branch1_error = replay._replay_terminal_error(initial, target, cfg, branch1_controls)

    nominal_path = controls_dir / "tail_replay_unit_nominal_controls.json"
    nominal_sha = write_deterministic_json(
        nominal_path,
        {
            "schema_version": 1,
            "sidecar_type": "tail_coast_nominal_controls",
            "suite_case_id": "tail_replay_unit",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "nominal_error": nominal_error,
            "controls": nominal_controls.tolist(),
        },
    )
    branch_entries = []
    for order, (mask_index, mask, controls, error, recovery_start, recovery_segments) in enumerate(
        [
            (0, [0, 1], branch0_controls, branch0_error, 1, 1),
            (1, [1, 0], branch1_controls, branch1_error, 2, 0),
        ]
    ):
        branch_path = controls_dir / f"tail_replay_unit_branch_{order:03d}_mask_{mask_index:03d}_controls.json"
        branch_sha = write_deterministic_json(
            branch_path,
            {
                "schema_version": 1,
                "sidecar_type": "tail_coast_branch_controls",
                "suite_case_id": "tail_replay_unit",
                "branch_order": order,
                "mask_index": mask_index,
                "outage_mask": mask,
                "target_state": target.tolist(),
                "thresholds": config["objective"]["thresholds"],
                "recovery_start": recovery_start,
                "recovery_segments": recovery_segments,
                "optimizer_ran": recovery_segments > 0,
                "optimizer_success": recovery_segments > 0,
                "accepted_candidate": "optimizer" if recovery_segments > 0 else "no_recovery_variables",
                "accepted_branch_weight_variant_label": "fixture",
                "accepted_branch_initialization_label": "fixture",
                "terminal_error": error,
                "branch_fuel": 0.0,
                "branch_controls": controls.tolist(),
                "recovery_controls": controls[recovery_start:].tolist() if recovery_start < controls.shape[0] else [],
            },
        )
        branch_entries.append(
            {
                "branch_order": order,
                "mask_index": mask_index,
                "outage_mask": mask,
                "recovery_start": recovery_start,
                "recovery_segments": recovery_segments,
                "terminal_error": error,
                "branch_fuel": 0.0,
                "optimizer_ran": recovery_segments > 0,
                "optimizer_success": recovery_segments > 0,
                "path": f"data/results/tail_replay_test/controls/{branch_path.name}",
                "sha256": branch_sha,
            }
        )

    manifest_path = controls_dir / "tail_replay_unit_branch_control_manifest.json"
    manifest_sha = write_deterministic_json(
        manifest_path,
        {
            "schema_version": 1,
            "sidecar_type": "tail_coast_branch_control_manifest",
            "suite_case_id": "tail_replay_unit",
            "target_mode": "catalog_dro_phase",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "settings_fingerprint": "fixture-fingerprint",
            "nominal_control_path": f"data/results/tail_replay_test/controls/{nominal_path.name}",
            "nominal_control_sha256": nominal_sha,
            "branch_count": 2,
            "branch_control_sidecar_count": 2,
            "branch_control_replay_ready": True,
            "branch_control_sidecars": branch_entries,
            "limitations": ["fixture normalized CR3BP replay only"],
        },
    )
    pd.DataFrame(
        [
            {
                "suite_case_id": "tail_replay_unit",
                "target_mode": "catalog_dro_phase",
                "nominal_error": nominal_error,
                "selected_worst_error": max(branch0_error, branch1_error),
                "all_mask_worst_error": max(branch0_error, branch1_error),
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
                "meets_thresholds": True,
                "branch_control_manifest_path": f"data/results/tail_replay_test/controls/{manifest_path.name}",
                "branch_control_manifest_sha256": manifest_sha,
                "nominal_control_path": f"data/results/tail_replay_test/controls/{nominal_path.name}",
                "nominal_control_sha256": nominal_sha,
                "branch_control_sidecar_count": 2,
                "branch_control_replay_ready": True,
            }
        ]
    ).to_csv(results_dir / "tail_coast_recovery.csv", index=False)

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=initial,
            target=target,
            target_metadata={"target_state_generation": "fixture"},
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(replay, "load_configured_states", fake_load_states)
    args = replay.build_parser().parse_args(["--config", str(config_path), "--source-states", str(source_states)])
    df = replay.run(args)

    metadata = json.loads((results_dir / "tail_coast_branch_control_replay_metadata.json").read_text(encoding="utf-8"))
    assert len(df) == 3
    assert int((df["record_type"] == "branch").sum()) == 2
    assert metadata["optimization_rerun"] is False
    assert metadata["branch_control_replay"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["branch_row_count"] == 2
    assert metadata["max_branch_terminal_error_delta"] <= 1.0e-12
    assert metadata["passes_tolerance"] is True
    assert (results_dir / "tail_coast_branch_control_replay.csv").exists()
    assert (tmp_path / "tables" / "tail_replay_test" / "tail_coast_branch_control_replay_table.tex").exists()


def test_real_tail_coast_branch_control_replay_package_metadata_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "hard_catalog_tail_coast_branch_control_replay"
        / "tail_coast_branch_control_replay_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("focused branch-control replay package has not been generated")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["optimization_rerun"] is False
    assert metadata["branch_control_replay"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["branch_row_count"] == 27
    assert metadata["passes_tolerance"] is True
