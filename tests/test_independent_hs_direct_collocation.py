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


def small_cfg(**overrides) -> ObjectiveConfig:
    params = {
        "mu": 0.01215058560962404,
        "tf": 0.12,
        "n_segments": 3,
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


def independent_minimal_config(output_subdir: str = "independent_hs_test") -> dict:
    return {
        "run": {"label": "independent_hs_test", "output_subdir": output_subdir},
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
        "direct_collocation": {
            "method": "hermite_simpson_midpoint",
            "node_initialization": "blend",
            "node_initialization_blend": 0.35,
            "xtol": 1e-5,
            "ftol": 1e-5,
            "gtol": 1e-5,
            "weights": {
                "initial": 10.0,
                "defect": 1.0,
                "terminal": 5.0,
                "branch_start": 8.0,
                "branch_defect": 1.0,
                "branch_terminal": 5.0,
                "control": 0.01,
                "smooth": 0.012,
            },
        },
        "suite": {
            "runtime_budget_seconds": 600,
            "selected_outages": 1,
            "min_recovery_segments": 1,
            "groups": {
                "phase_group": {
                    "purpose": "unit-test independent HS group",
                    "target_mode": "catalog_halo_phase_shift",
                    "outage_lengths": [1],
                    "cases": [
                        {
                            "case_id": "ihs_cold_p03",
                            "phase_time": 0.3,
                            "transfer_time": 0.5,
                            "amax": 0.3,
                            "segments": 3,
                            "selected_outages": 1,
                            "max_nfev": 2,
                            "warm_start_kind": "cold",
                        },
                        {
                            "case_id": "ihs_warm_p02_from_p03",
                            "phase_time": 0.2,
                            "transfer_time": 0.5,
                            "amax": 0.3,
                            "segments": 3,
                            "selected_outages": 1,
                            "max_nfev": 2,
                            "warm_start_kind": "nominal_controls",
                            "warm_start_from_case_id": "ihs_cold_p03",
                        },
                    ],
                }
            },
        },
    }


def test_independent_hs_method_normalization_type_and_aliases():
    from qlt.direct_collocation import (
        collocation_method_type,
        collocation_method_uses_midpoint_controls,
        normalize_collocation_method,
    )

    assert normalize_collocation_method("hermite-simpson-midpoint") == "hermite_simpson_midpoint"
    assert normalize_collocation_method("hermite_simpson_independent") == "hermite_simpson_midpoint"
    assert collocation_method_uses_midpoint_controls("hs_independent") is True
    assert collocation_method_uses_midpoint_controls("hermite_simpson") is False
    assert collocation_method_type("hermite_simpson_midpoint") == (
        "bounded_projected_independent_midpoint_control_hermite_simpson_direct_collocation"
    )
    with pytest.raises(ValueError, match="hermite_simpson_midpoint"):
        normalize_collocation_method("not_a_method")


def test_independent_hs_wrapper_supports_configured_artifact_family():
    script = load_script_module(
        "run_independent_hs_names_test",
        ROOT / "scripts" / "run_independent_hs_continuation.py",
    )

    assert script._configured_names({}) == (
        "independent_hs_continuation_baseline",
        "independent_hs_continuation_baseline",
    )
    assert script._configured_names(
        {
            "run": {
                "suite_name": "independent_hs_all_configured_headroom",
                "artifact_stem": "independent_hs_all_configured_headroom",
            }
        }
    ) == (
        "independent_hs_all_configured_headroom",
        "independent_hs_all_configured_headroom",
    )


def test_independent_hs_layout_adds_midpoint_controls_only_for_independent_method():
    from qlt.direct_collocation import DirectCollocationLayout, initial_guess

    cfg = small_cfg(n_segments=4)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)[:1]

    constant_layout = DirectCollocationLayout(cfg, masks, method="hermite_simpson")
    independent_layout = DirectCollocationLayout(cfg, masks, method="hermite_simpson_midpoint")

    assert constant_layout.has_midpoint_controls is False
    assert constant_layout.nominal_midpoint_controls is None
    assert all(item is None for item in constant_layout.branch_midpoint_controls)

    assert independent_layout.has_midpoint_controls is True
    assert independent_layout.nominal_midpoint_controls is not None
    assert all(item is not None for item in independent_layout.branch_midpoint_controls)
    assert independent_layout.size > constant_layout.size

    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.95, -0.02, 0.01, -0.01, 0.02, 0.03], dtype=float)
    layout, vec = initial_guess(
        state0,
        target,
        cfg,
        masks,
        method="hermite_simpson_midpoint",
        nominal_control_guess=np.zeros((cfg.n_segments, 3), dtype=float),
    )
    decision = layout.unpack_decision(vec)

    assert decision.nominal_midpoint_controls is not None
    assert decision.nominal_midpoint_controls.shape == (cfg.n_segments, 3)
    assert len(decision.branch_midpoint_controls) == 1
    assert decision.branch_midpoint_controls[0] is not None
    assert decision.branch_midpoint_controls[0].shape == decision.branch_controls[0].shape


def test_independent_hs_residual_and_evaluation_use_midpoint_controls():
    from qlt.direct_collocation import DirectCollocationProblem, DirectCollocationWeights, initial_guess

    cfg = small_cfg(n_segments=3, tf=0.18, amax=0.05)
    state0 = np.array([1.03, 0.01, 0.02, 0.01, 0.04, -0.02], dtype=float)
    target = np.array([0.99, -0.01, 0.01, -0.005, 0.02, 0.01], dtype=float)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    selected = np.zeros(0, dtype=int)
    layout, vec = initial_guess(
        state0,
        target,
        cfg,
        np.zeros((0, cfg.n_segments), dtype=float),
        method="hermite_simpson_midpoint",
        nominal_control_guess=np.zeros((cfg.n_segments, 3), dtype=float),
        node_initialization="linear",
    )
    assert layout.nominal_midpoint_controls is not None

    problem = DirectCollocationProblem(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        selected=selected,
        layout=layout,
        weights=DirectCollocationWeights(),
        method="hermite_simpson_midpoint",
    )

    changed = vec.copy()
    midpoint_values = np.zeros((cfg.n_segments, 3), dtype=float)
    midpoint_values[:, 0] = 0.04
    changed[layout.nominal_midpoint_controls] = midpoint_values.reshape(-1)

    residual_base = problem.residual(vec)
    residual_changed = problem.residual(changed)
    eval_base = problem.evaluate_vector(vec, {"nominal_success": 10.0, "robust_success": 10.0})
    eval_changed = problem.evaluate_vector(changed, {"nominal_success": 10.0, "robust_success": 10.0})

    assert not np.allclose(residual_base, residual_changed)
    assert not np.allclose(eval_base["nominal_history"], eval_changed["nominal_history"])
    assert eval_changed["nominal_fuel"] > eval_base["nominal_fuel"]
    assert eval_changed["fuel_quadrature"] == "simpson_endpoint_midpoint_endpoint"


def test_independent_hs_evidence_script_fake_backend_writes_artifacts(monkeypatch, tmp_path):
    script = load_script_module(
        "run_independent_hs_continuation_test",
        ROOT / "scripts" / "run_independent_hs_continuation.py",
    )
    config = independent_minimal_config("independent_hs_e2e_test")
    config_path = tmp_path / "independent_hs_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=np.zeros(6, dtype=float),
            target=np.ones(6, dtype=float) * 0.01,
            target_metadata={"target_state_generation": "fixture target generation"},
        )

    def fake_run_direct_collocation(
        *,
        state0,
        target,
        cfg,
        masks,
        thresholds,
        selected_outages,
        max_nfev,
        min_recovery_segments,
        collocation_config,
        nominal_control_guess,
        selected_branch_control_guesses,
        warm_start_info,
        nominal_midpoint_control_guess=None,
    ):
        del state0, target, masks, thresholds, selected_outages, max_nfev, min_recovery_segments
        del selected_branch_control_guesses
        calls.append(
            {
                "method": collocation_config.get("method"),
                "warm_start_enabled": bool(warm_start_info.get("enabled")),
                "guess_is_none": nominal_control_guess is None,
                "midpoint_guess_is_none": nominal_midpoint_control_guess is None,
            }
        )
        controls = (
            np.zeros((cfg.n_segments, 3), dtype=float)
            if nominal_control_guess is None
            else np.asarray(nominal_control_guess, dtype=float).copy()
        )
        midpoint_controls = controls + 0.001
        if nominal_midpoint_control_guess is not None:
            midpoint_controls = np.asarray(nominal_midpoint_control_guess, dtype=float).copy()
        return {
            "success": True,
            "nominal_controls": controls,
            "nominal_midpoint_controls": midpoint_controls,
            "selected_branch_controls": [],
            "selected_branch_midpoint_controls": [],
            "nominal_history": None,
            "nominal_fuel": 0.1,
            "recovery_fuel_mean": 0.1,
            "recovery_fuel_max": 0.1,
            "all_mask_recovery_fuel_mean": 0.1,
            "nominal_error": 0.04,
            "selected_worst_error": 0.06,
            "all_mask_worst_error": 0.08,
            "selected_outage_errors": [0.06],
            "all_outage_errors": [0.07, 0.08],
            "control_max_norm": 0.1,
            "control_bound_violation": 0.0,
            "fuel_quadrature": "simpson_endpoint_midpoint_endpoint",
            "method_type": "bounded_projected_independent_midpoint_control_hermite_simpson_direct_collocation",
            "collocation_method": "hermite_simpson_midpoint",
            "collocation_scheme_semantics": "fixture independent midpoint HS",
            "selected_branch_semantics": "fixture selected",
            "all_mask_diagnostic_semantics": "fixture all mask",
            "control_bound_semantics": "fixture bound",
            "optimizer_success": True,
            "message": "fixture ok",
            "cost": 0.001,
            "optimality": 1e-6,
            "nfev": 2,
            "runtime_seconds": 0.01,
            "selected_outage_indices": [0],
            "weights": {},
            "warm_start_info": warm_start_info,
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script, "run_direct_collocation_baseline", fake_run_direct_collocation)
    monkeypatch.setattr(script, "load_configured_states", fake_load_states)

    args = script.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )
    df = script.run(args)

    results_dir = tmp_path / "data" / "results" / "independent_hs_e2e_test"
    csv_path = results_dir / "independent_hs_continuation_baseline.csv"
    meta_path = results_dir / "independent_hs_continuation_baseline_metadata.json"
    tables_dir = tmp_path / "tables" / "independent_hs_e2e_test"
    figures_dir = tmp_path / "figures" / "independent_hs_e2e_test"

    assert len(df) == 2
    assert csv_path.exists()
    assert meta_path.exists()
    assert (tables_dir / "independent_hs_continuation_baseline_table.tex").exists()
    assert (figures_dir / "independent_hs_continuation_baseline.png").exists()
    assert (figures_dir / "independent_hs_continuation_baseline.pdf").exists()

    assert calls[0]["method"] == "hermite_simpson_midpoint"
    assert calls[0]["guess_is_none"] is True
    assert calls[0]["warm_start_enabled"] is False
    assert calls[1]["method"] == "hermite_simpson_midpoint"
    assert calls[1]["guess_is_none"] is False
    assert calls[1]["midpoint_guess_is_none"] is False
    assert calls[1]["warm_start_enabled"] is True

    csv_df = pd.read_csv(csv_path)
    assert "collocation_scheme_semantics" in csv_df.columns
    assert "fuel_quadrature" in csv_df.columns
    assert "nominal_control_sidecar_hash" in csv_df.columns
    assert "nominal_endpoint_control_hash" in csv_df.columns
    assert "nominal_midpoint_control_hash" in csv_df.columns
    assert "nominal_midpoint_control_present" in csv_df.columns
    assert set(csv_df["collocation_method"]) == {"hermite_simpson_midpoint"}
    assert csv_df["nominal_midpoint_control_present"].map(bool).all()
    assert (
        csv_df["nominal_control_hash"].astype(str)
        == csv_df["nominal_control_sidecar_hash"].astype(str)
    ).all()
    assert (
        csv_df["nominal_control_hash"].astype(str)
        != csv_df["nominal_endpoint_control_hash"].astype(str)
    ).all()

    controls_dir = results_dir / "controls"
    cold_sidecar = controls_dir / "ihs_cold_p03_nominal_controls.json"
    sidecar_payload = json.loads(cold_sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["midpoint_control_present"] is True
    assert sidecar_payload["nominal_midpoint_controls"] is not None
    assert sidecar_payload["sidecar_hash"] == csv_df.iloc[0]["nominal_control_hash"]

    loaded = script.load_control_sidecar(
        controls_dir,
        "ihs_cold_p03",
        csv_df.iloc[0]["settings_fingerprint"],
        require_midpoint_controls=True,
    )
    assert loaded is not None
    assert loaded.midpoint_controls is not None
    assert loaded.midpoint_control_present is True
    assert np.allclose(loaded.midpoint_controls, np.ones((3, 3), dtype=float) * 0.001)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 2
    assert "independent-midpoint-control" in metadata["semantics"]["backend"]
    assert "hermite_simpson_midpoint" in metadata["semantics"]
    assert "sidecar_schema" in metadata["semantics"]


def test_independent_hs_sidecar_requires_and_validates_midpoint_controls(tmp_path):
    script = load_script_module(
        "run_independent_hs_sidecar_test",
        ROOT / "scripts" / "run_independent_hs_continuation.py",
    )
    controls_dir = tmp_path / "controls"
    endpoint_controls = np.zeros((3, 3), dtype=float)
    midpoint_controls = np.ones((3, 3), dtype=float) * 0.02
    row = {
        "phase_time": 0.3,
        "transfer_time": 0.5,
        "amax": 0.3,
        "segments": 3,
        "outage_lengths": "[1]",
        "warm_start_kind": "cold",
        "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson_midpoint",
    }

    sidecar = script.write_control_sidecar(
        controls_dir,
        case_id="ihs_case",
        settings_fingerprint="fp",
        controls=endpoint_controls,
        midpoint_controls=midpoint_controls,
        row=row,
    )
    assert sidecar["control_hash"] == sidecar["sidecar_hash"]
    assert sidecar["control_hash"] != sidecar["endpoint_control_hash"]
    assert sidecar["midpoint_control_present"] is True

    loaded = script.load_control_sidecar(
        controls_dir,
        "ihs_case",
        "fp",
        require_midpoint_controls=True,
    )
    assert loaded is not None
    assert np.allclose(loaded.controls, endpoint_controls)
    assert loaded.midpoint_controls is not None
    assert np.allclose(loaded.midpoint_controls, midpoint_controls)
    assert loaded.control_hash == sidecar["sidecar_hash"]
    assert loaded.endpoint_control_hash == sidecar["endpoint_control_hash"]
    assert loaded.midpoint_control_hash == sidecar["midpoint_control_hash"]

    sidecar_path = controls_dir / "ihs_case_nominal_controls.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["midpoint_control_hash"] = "deadbeef" * 8
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    assert script.load_control_sidecar(
        controls_dir,
        "ihs_case",
        "fp",
        require_midpoint_controls=True,
    ) is None

    endpoint_only = script.write_control_sidecar(
        controls_dir,
        case_id="endpoint_only",
        settings_fingerprint="fp2",
        controls=endpoint_controls,
        row={**row, "collocation_method": "hermite_simpson"},
    )
    assert endpoint_only["control_hash"] == endpoint_only["endpoint_control_hash"]
    assert script.load_control_sidecar(
        controls_dir,
        "endpoint_only",
        "fp2",
        require_midpoint_controls=False,
    ) is not None
    assert script.load_control_sidecar(
        controls_dir,
        "endpoint_only",
        "fp2",
        require_midpoint_controls=True,
    ) is None
