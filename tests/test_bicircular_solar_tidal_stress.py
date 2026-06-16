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

from qlt.bicircular import (
    SolarTidalParameters,
    propagate_controls_batch_bicircular,
    solar_tidal_acceleration,
    sun_position_rotating,
)
from qlt.cr3bp import propagate_controls_batch


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


def test_bicircular_zero_solar_mu_matches_cr3bp_control_propagation():
    state0 = np.array([1.02, 0.01, 0.02, 0.0, 0.03, -0.01], dtype=float)
    controls = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.0, 0.02, -0.01],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    mu = 0.01215058560962404
    cr3bp_final, _ = propagate_controls_batch(state0, controls, mu, 0.12, 2)
    bicircular_final, _ = propagate_controls_batch_bicircular(
        state0,
        controls,
        mu,
        0.12,
        2,
        phase_rad=0.7,
        parameters=SolarTidalParameters(sun_mu_ratio=0.0),
    )

    assert np.allclose(bicircular_final, cr3bp_final, atol=1.0e-14, rtol=1.0e-14)


def test_solar_tidal_acceleration_uses_rotating_phase_and_differential_term():
    params = SolarTidalParameters(sun_distance_lu=10.0, sun_mu_ratio=2.0, sun_inertial_angular_rate_ratio=0.25)

    assert np.allclose(
        sun_position_rotating(0.0, phase_rad=np.pi / 2.0, parameters=params),
        np.array([0.0, 10.0, 0.0]),
        atol=1.0e-14,
    )
    assert np.allclose(
        solar_tidal_acceleration(np.zeros(3), 0.0, phase_rad=0.0, parameters=params),
        np.zeros(3),
        atol=1.0e-14,
    )

    accel = solar_tidal_acceleration(np.array([1.0, 0.0, 0.0]), 0.0, phase_rad=0.0, parameters=params)
    assert accel.shape == (3,)
    assert accel[0] > 0.0


def test_bicircular_solar_tidal_stress_script_with_generated_sidecars(monkeypatch, tmp_path):
    module = load_script_module(
        "run_bicircular_solar_tidal_stress_unit",
        ROOT / "scripts" / "run_bicircular_solar_tidal_stress.py",
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
            "thresholds": {"nominal_success": 10.0, "robust_success": 10.0},
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

    cfg = module.make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    nominal_controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    target = module.propagate_controls_batch(initial, nominal_controls, cfg.mu, cfg.tf, cfg.substeps)[0][0]
    branch0_controls = np.array([[0.0, 0.0, 0.0], [0.015, 0.0, 0.0]], dtype=float)
    branch1_controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
    nominal_error = module._cr3bp_terminal_error(initial, target, cfg, nominal_controls)
    branch0_error = module._cr3bp_terminal_error(initial, target, cfg, branch0_controls)
    branch1_error = module._cr3bp_terminal_error(initial, target, cfg, branch1_controls)

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
                "nominal_threshold": 10.0,
                "selected_worst_threshold": 10.0,
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

    output_results = tmp_path / "bicircular_results"
    output_tables = tmp_path / "bicircular_tables"
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
        "csv": (output_results / "bicircular_solar_tidal_stress.csv").read_bytes(),
        "metadata": (output_results / "bicircular_solar_tidal_stress_metadata.json").read_bytes(),
        "table": (output_tables / "bicircular_solar_tidal_stress_table.tex").read_bytes(),
    }
    second = module.run(args)

    assert len(first) == 6
    assert len(second) == 6
    assert (output_results / "bicircular_solar_tidal_stress.csv").read_bytes() == first_bytes["csv"]
    assert (output_results / "bicircular_solar_tidal_stress_metadata.json").read_bytes() == first_bytes["metadata"]
    assert (output_tables / "bicircular_solar_tidal_stress_table.tex").read_bytes() == first_bytes["table"]

    metadata = json.loads((output_results / "bicircular_solar_tidal_stress_metadata.json").read_text(encoding="utf-8"))
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_recorded_artifacts_only"] is True
    assert metadata["bicircular_solar_tidal_stress_probe"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["branch_source_record_count"] == 2
    assert metadata["row_count"] == 6
    assert metadata["baseline_reproduction"]["max_cr3bp_delta_from_source"] <= 1.0e-10
    assert metadata["solar_tidal_parameters"]["rotating_frame_phase_rate"] == pytest.approx(-0.9252)
    assert "not SPICE ephemeris validation" in " ".join(metadata["interpretation_limits"])

    csv_df = pd.read_csv(output_results / "bicircular_solar_tidal_stress.csv")
    assert set(csv_df["phase_degrees"]) == {0.0, 180.0}
    assert set(csv_df["record_type"]) == {"nominal", "branch"}
    assert csv_df["controls_sha256"].str.len().eq(64).all()
    assert (csv_df["solar_tidal_delta_from_cr3bp"].astype(float) > 0.0).any()


def test_real_bicircular_solar_tidal_stress_package_metadata_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "bicircular_solar_tidal_stress"
        / "bicircular_solar_tidal_stress_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("bicircular solar-tidal stress package has not been generated")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_recorded_artifacts_only"] is True
    assert metadata["bicircular_solar_tidal_stress_probe"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["branch_source_record_count"] == 27
    assert metadata["phase_degrees"] == [0.0, 90.0, 180.0, 270.0]
