from __future__ import annotations

import hashlib
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

from qlt.bicircular import SolarTidalParameters, propagate_piecewise_controls_bicircular
from qlt.direct_collocation import propagate_piecewise_controls


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_deterministic_json(path: Path, data: dict) -> str:
    payload = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_midpoint_bicircular_matches_cr3bp_at_zero_solar_mu_and_changes_when_active():
    state0 = np.array([1.02, 0.01, 0.02, 0.0, 0.03, -0.01], dtype=float)
    endpoint = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, -0.01]], dtype=float)
    midpoint = np.array([[0.02, 0.01, 0.0], [-0.01, 0.025, -0.005]], dtype=float)
    mu = 0.01215058560962404

    cr3bp_final, _ = propagate_piecewise_controls(
        state0,
        endpoint,
        mu,
        0.12,
        3,
        midpoint_controls=midpoint,
    )
    zero_sun_final, zero_nodes = propagate_piecewise_controls_bicircular(
        state0,
        endpoint,
        mu,
        0.12,
        3,
        midpoint_controls=midpoint,
        phase_rad=0.6,
        parameters=SolarTidalParameters(sun_mu_ratio=0.0),
        return_nodes=True,
    )
    active_final, _ = propagate_piecewise_controls_bicircular(
        state0,
        endpoint,
        mu,
        0.12,
        3,
        midpoint_controls=midpoint,
        phase_rad=0.6,
        parameters=SolarTidalParameters(sun_distance_lu=20.0, sun_mu_ratio=5.0),
    )
    active_repeat, _ = propagate_piecewise_controls_bicircular(
        state0,
        endpoint,
        mu,
        0.12,
        3,
        midpoint_controls=midpoint,
        phase_rad=0.6,
        parameters=SolarTidalParameters(sun_distance_lu=20.0, sun_mu_ratio=5.0),
    )

    assert np.allclose(zero_sun_final, cr3bp_final, atol=1.0e-14, rtol=1.0e-14)
    assert zero_nodes is not None
    assert zero_nodes.shape == (3, 6)
    assert not np.allclose(active_final, cr3bp_final, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(active_repeat, active_final, atol=0.0, rtol=0.0)


def minimal_config() -> dict:
    return {
        "run": {
            "label": "ihs_phase_stress_source",
            "suite_name": "independent_hs_all_configured_headroom",
            "artifact_stem": "independent_hs_all_configured_headroom",
            "output_subdir": "ihs_phase_stress_source",
        },
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_halo_phase_shift",
            "transfer_time": 0.08,
            "phase_time": 0.4,
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
            "thresholds": {"nominal_success": 10.0, "robust_success": 10.0},
        },
        "outages": {"block_lengths": [1]},
        "direct_collocation": {"method": "hermite_simpson_midpoint"},
        "suite": {
            "selected_outages": "all_configured_outage_masks",
            "min_recovery_segments": 1,
            "groups": {
                "phase_group": {
                    "purpose": "unit all-configured phase stress",
                    "target_mode": "catalog_halo_phase_shift",
                    "outage_lengths": [1],
                    "cases": [
                        {
                            "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
                            "phase_time": 0.4,
                            "transfer_time": 0.08,
                            "amax": 0.2,
                            "segments": 2,
                            "selected_outages": "all_configured_outage_masks",
                            "max_nfev": 1,
                            "warm_start_kind": "cold",
                            "persist_branch_controls": True,
                        }
                    ],
                }
            },
        },
    }


def build_phase_stress_fixture(tmp_path: Path, module):
    config = minimal_config()
    config_path = tmp_path / "ihs_phase_stress_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "data" / "results" / "ihs_phase_stress_source"
    controls_dir = results_dir / "controls"
    controls_dir.mkdir(parents=True)

    cfg = module.make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    nominal_endpoint = np.zeros((2, 3), dtype=float)
    nominal_midpoint = np.zeros((2, 3), dtype=float)
    target, _ = module.propagate_piecewise_controls(
        initial,
        nominal_endpoint,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=nominal_midpoint,
    )
    nominal_error = module._cr3bp_terminal_error(
        state0=initial,
        target=target,
        cfg=cfg,
        endpoint_controls=nominal_endpoint,
        midpoint_controls=nominal_midpoint,
    )

    nominal_path = controls_dir / "ihs_all_single_p04_amax02_polish_from_p04_nominal_controls.json"
    nominal_sha = write_deterministic_json(
        nominal_path,
        {
            "schema_version": 2,
            "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
            "settings_fingerprint": "fixture-fingerprint",
            "nominal_endpoint_controls": nominal_endpoint.tolist(),
            "nominal_midpoint_controls": nominal_midpoint.tolist(),
            "midpoint_control_present": True,
        },
    )

    branch_entries = []
    for order in range(2):
        mask = [1, 1]
        mask[order] = 0
        branch_path = controls_dir / (
            f"ihs_all_single_p04_amax02_polish_from_p04_branch_{order:03d}_mask_{order:03d}_controls.json"
        )
        branch_sha = write_deterministic_json(
            branch_path,
            {
                "schema_version": 1,
                "sidecar_type": "hs_direct_collocation_branch_controls",
                "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
                "branch_order": order,
                "mask_index": order,
                "outage_mask": mask,
                "target_state": target.tolist(),
                "thresholds": config["objective"]["thresholds"],
                "recovery_start": order + 1,
                "recovery_segments": 2 - (order + 1),
                "recorded_branch_terminal_error": nominal_error,
                "recorded_selected_worst_error": nominal_error,
                "recorded_all_mask_worst_error": nominal_error,
                "branch_endpoint_controls": nominal_endpoint.tolist(),
                "branch_controls": nominal_endpoint.tolist(),
                "branch_midpoint_controls": nominal_midpoint.tolist(),
            },
        )
        branch_entries.append(
            {
                "branch_order": order,
                "mask_index": order,
                "outage_mask": mask,
                "recovery_start": order + 1,
                "recovery_segments": 2 - (order + 1),
                "recorded_branch_terminal_error": nominal_error,
                "path": f"data/results/ihs_phase_stress_source/controls/{branch_path.name}",
                "sha256": branch_sha,
            }
        )

    manifest_path = controls_dir / "ihs_all_single_p04_amax02_polish_from_p04_branch_control_manifest.json"
    manifest_sha = write_deterministic_json(
        manifest_path,
        {
            "schema_version": 1,
            "sidecar_type": "hs_direct_collocation_branch_control_manifest",
            "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
            "settings_fingerprint": "fixture-fingerprint",
            "target_mode": "catalog_halo_phase_shift",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "nominal_control_path": f"data/results/ihs_phase_stress_source/controls/{nominal_path.name}",
            "nominal_control_sha256": nominal_sha,
            "nominal_error": nominal_error,
            "selected_worst_error": nominal_error,
            "all_mask_worst_error": nominal_error,
            "expected_branch_count": 2,
            "branch_count": 2,
            "branch_control_sidecar_count": 2,
            "branch_control_replay_ready": True,
            "selected_outage_indices": [0, 1],
            "selected_outage_errors": [nominal_error, nominal_error],
            "all_outage_errors": [nominal_error, nominal_error],
            "branch_control_sidecars": branch_entries,
        },
    )

    pd.DataFrame(
        [
            {
                "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
                "target_mode": "catalog_halo_phase_shift",
                "phase_time": 0.4,
                "transfer_time": 0.08,
                "amax": 0.2,
                "segments": 2,
                "nominal_error": nominal_error,
                "selected_worst_error": nominal_error,
                "all_mask_worst_error": nominal_error,
                "nominal_threshold": 10.0,
                "selected_worst_threshold": 10.0,
                "meets_thresholds": True,
                "optimizer_success": True,
                "nfev": 1,
                "branch_control_manifest_path": f"data/results/ihs_phase_stress_source/controls/{manifest_path.name}",
                "branch_control_manifest_hash": manifest_sha,
                "branch_control_sidecar_count": 2,
                "branch_control_replay_ready": True,
            }
        ]
    ).to_csv(results_dir / "independent_hs_all_configured_headroom.csv", index=False)
    return config_path, source_states, initial, target


def test_phase_stress_script_with_generated_sidecars(monkeypatch, tmp_path):
    module = load_script_module(
        "run_independent_hs_bicircular_phase_stress_unit",
        ROOT / "scripts" / "run_independent_hs_bicircular_phase_stress.py",
    )
    config_path, source_states, initial, target = build_phase_stress_fixture(tmp_path, module)

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=initial,
            target=target,
            target_metadata={"target_state_generation": "fixture"},
        )

    output_results = tmp_path / "phase_stress_results"
    output_tables = tmp_path / "phase_stress_tables"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_configured_states", fake_load_states)
    args = module.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--source-states",
            str(source_states),
            "--results-dir",
            str(output_results),
            "--tables-dir",
            str(output_tables),
            "--phases-degrees",
            "0",
            "180",
        ]
    )
    first = module.run(args)
    first_bytes = {
        "csv": (output_results / "independent_hs_bicircular_phase_stress.csv").read_bytes(),
        "metadata": (output_results / "independent_hs_bicircular_phase_stress_metadata.json").read_bytes(),
        "table": (output_tables / "independent_hs_bicircular_phase_stress_table.tex").read_bytes(),
    }
    second = module.run(args)

    assert len(first) == 6
    assert len(second) == 6
    assert (output_results / "independent_hs_bicircular_phase_stress.csv").read_bytes() == first_bytes["csv"]
    assert (output_results / "independent_hs_bicircular_phase_stress_metadata.json").read_bytes() == first_bytes["metadata"]
    assert (output_tables / "independent_hs_bicircular_phase_stress_table.tex").read_bytes() == first_bytes["table"]

    metadata = json.loads(
        (output_results / "independent_hs_bicircular_phase_stress_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_recorded_artifacts_only"] is True
    assert metadata["independent_hs_bicircular_phase_stress_probe"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["phase_count"] == 2
    assert metadata["branch_source_record_count"] == 2
    assert metadata["baseline_reproduction"]["max_cr3bp_delta_from_recorded"] <= 1.0e-10
    assert "not SPICE ephemeris validation" in " ".join(metadata["interpretation_limits"])

    csv_df = pd.read_csv(output_results / "independent_hs_bicircular_phase_stress.csv")
    assert set(csv_df["phase_degrees"]) == {0.0, 180.0}
    assert set(csv_df["record_type"]) == {"nominal", "branch"}
    assert csv_df["controls_sha256"].str.len().eq(64).all()
    assert csv_df["midpoint_controls_replayed"].map(bool).all()


def test_real_polished_independent_hs_phase_stress_package_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "independent_hs_bicircular_phase_stress"
        / "independent_hs_bicircular_phase_stress_metadata.json"
    )
    csv_path = metadata_path.with_name("independent_hs_bicircular_phase_stress.csv")
    if not metadata_path.exists() or not csv_path.exists():
        pytest.skip("independent-HS bicircular phase-stress package has not been generated")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    polish = metadata["polish_case_summary"]
    assert metadata["phase_degrees"] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert polish["nominal_phase_count"] == 8
    assert polish["branch_phase_count"] == 64
    assert polish["nominal_bicircular_pass_count"] == 8
    assert polish["branch_bicircular_pass_count"] == 64
    assert polish["max_nominal_bicircular_terminal_error"] == pytest.approx(0.022138676654057693)
    assert polish["max_branch_bicircular_terminal_error"] == pytest.approx(0.08557051343145317)

    df = pd.read_csv(csv_path)
    polish_rows = df[df["case_id"] == "ihs_all_single_p04_amax02_polish_from_p04"]
    nominal = polish_rows[polish_rows["record_type"] == "nominal"]
    branch = polish_rows[polish_rows["record_type"] == "branch"]
    assert len(nominal) == 8
    assert len(branch) == 64
    assert nominal["bicircular_passes_configured_threshold"].map(bool).all()
    assert branch["bicircular_passes_configured_threshold"].map(bool).all()
