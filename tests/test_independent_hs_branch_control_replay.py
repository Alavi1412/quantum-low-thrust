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


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_deterministic_json(path: Path, data: dict) -> str:
    payload = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def minimal_config() -> dict:
    return {
        "run": {
            "label": "ihs_replay_source",
            "suite_name": "independent_hs_all_configured_headroom",
            "artifact_stem": "independent_hs_all_configured_headroom",
            "output_subdir": "ihs_replay_source",
        },
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_halo_phase_shift",
            "transfer_time": 0.08,
            "phase_time": 0.4,
            "segments": 8,
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
        "direct_collocation": {"method": "hermite_simpson_midpoint"},
        "suite": {
            "selected_outages": "all_configured_outage_masks",
            "min_recovery_segments": 1,
            "groups": {
                "phase_group": {
                    "purpose": "unit all-configured replay",
                    "target_mode": "catalog_halo_phase_shift",
                    "outage_lengths": [1],
                    "cases": [
                        {
                            "case_id": "ihs_all_single_p04_amax02_warm_from_p03",
                            "phase_time": 0.4,
                            "transfer_time": 0.08,
                            "amax": 0.2,
                            "segments": 8,
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


def build_replay_fixture(tmp_path: Path, replay_module):
    config = minimal_config()
    config_path = tmp_path / "ihs_replay_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    results_dir = tmp_path / "data" / "results" / "ihs_replay_source"
    controls_dir = results_dir / "controls"
    controls_dir.mkdir(parents=True)

    cfg = replay_module.make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    nominal_endpoint = np.zeros((8, 3), dtype=float)
    nominal_midpoint = np.zeros((8, 3), dtype=float)
    target, _ = replay_module.propagate_piecewise_controls(
        initial,
        nominal_endpoint,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=nominal_midpoint,
    )
    nominal_error = replay_module._terminal_error(
        state0=initial,
        target=target,
        cfg=cfg,
        endpoint_controls=nominal_endpoint,
        midpoint_controls=nominal_midpoint,
    )

    nominal_path = controls_dir / "ihs_all_single_p04_amax02_warm_from_p03_nominal_controls.json"
    nominal_sha = write_deterministic_json(
        nominal_path,
        {
            "schema_version": 2,
            "case_id": "ihs_all_single_p04_amax02_warm_from_p03",
            "settings_fingerprint": "fixture-fingerprint",
            "nominal_endpoint_controls": nominal_endpoint.tolist(),
            "nominal_midpoint_controls": nominal_midpoint.tolist(),
            "midpoint_control_present": True,
            "sidecar_hash": "fixture-logical-sidecar-hash",
        },
    )

    branch_entries = []
    all_errors = []
    for order in range(8):
        mask = [1] * 8
        mask[order] = 0
        endpoint = nominal_endpoint.copy()
        midpoint = nominal_midpoint.copy()
        replay_error = replay_module._terminal_error(
            state0=initial,
            target=target,
            cfg=cfg,
            endpoint_controls=endpoint,
            midpoint_controls=midpoint,
        )
        all_errors.append(replay_error)
        branch_path = controls_dir / (
            f"ihs_all_single_p04_amax02_warm_from_p03_branch_{order:03d}_mask_{order:03d}_controls.json"
        )
        branch_sha = write_deterministic_json(
            branch_path,
            {
                "schema_version": 1,
                "sidecar_type": "hs_direct_collocation_branch_controls",
                "case_id": "ihs_all_single_p04_amax02_warm_from_p03",
                "branch_order": order,
                "mask_index": order,
                "outage_mask": mask,
                "target_state": target.tolist(),
                "thresholds": config["objective"]["thresholds"],
                "recovery_start": order + 1,
                "recovery_segments": 8 - (order + 1),
                "recorded_branch_terminal_error": replay_error,
                "recorded_selected_worst_error": 0.0,
                "recorded_all_mask_worst_error": 0.0,
                "branch_endpoint_controls": endpoint.tolist(),
                "branch_controls": endpoint.tolist(),
                "branch_midpoint_controls": midpoint.tolist(),
            },
        )
        branch_entries.append(
            {
                "branch_order": order,
                "mask_index": order,
                "outage_mask": mask,
                "recovery_start": order + 1,
                "recovery_segments": 8 - (order + 1),
                "recorded_branch_terminal_error": replay_error,
                "path": f"data/results/ihs_replay_source/controls/{branch_path.name}",
                "sha256": branch_sha,
            }
        )

    manifest_path = controls_dir / "ihs_all_single_p04_amax02_warm_from_p03_branch_control_manifest.json"
    manifest_sha = write_deterministic_json(
        manifest_path,
        {
            "schema_version": 1,
            "sidecar_type": "hs_direct_collocation_branch_control_manifest",
            "case_id": "ihs_all_single_p04_amax02_warm_from_p03",
            "settings_fingerprint": "fixture-fingerprint",
            "target_mode": "catalog_halo_phase_shift",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "nominal_control_path": f"data/results/ihs_replay_source/controls/{nominal_path.name}",
            "nominal_control_sha256": nominal_sha,
            "nominal_error": nominal_error,
            "selected_worst_error": max(all_errors),
            "all_mask_worst_error": max(all_errors),
            "expected_branch_count": 8,
            "branch_count": 8,
            "branch_control_sidecar_count": 8,
            "branch_control_replay_ready": True,
            "selected_outage_indices": list(range(8)),
            "selected_outage_errors": all_errors,
            "all_outage_errors": all_errors,
            "branch_control_sidecars": branch_entries,
            "limitations": ["fixture normalized CR3BP replay only; no high-fidelity validation"],
        },
    )

    pd.DataFrame(
        [
            {
                "case_id": "ihs_all_single_p04_amax02_warm_from_p03",
                "target_mode": "catalog_halo_phase_shift",
                "phase_time": 0.4,
                "transfer_time": 0.08,
                "amax": 0.2,
                "segments": 8,
                "nominal_error": nominal_error,
                "selected_worst_error": max(all_errors),
                "all_mask_worst_error": max(all_errors),
                "nominal_threshold": 0.09,
                "selected_worst_threshold": 0.17,
                "meets_thresholds": True,
                "branch_control_manifest_path": f"data/results/ihs_replay_source/controls/{manifest_path.name}",
                "branch_control_manifest_hash": manifest_sha,
                "branch_control_sidecar_count": 8,
                "branch_control_replay_ready": True,
            }
        ]
    ).to_csv(results_dir / "independent_hs_all_configured_headroom.csv", index=False)
    return config_path, source_states, initial, target, branch_entries[0]["path"]


def test_independent_hs_branch_control_replay_validates_hashes_and_replays_all_8(monkeypatch, tmp_path):
    replay = load_script_module(
        "run_independent_hs_branch_control_replay_unit",
        ROOT / "scripts" / "run_independent_hs_branch_control_replay.py",
    )
    config_path, source_states, initial, target, first_branch_path = build_replay_fixture(tmp_path, replay)

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

    metadata = json.loads(
        (tmp_path / "data" / "results" / "independent_hs_branch_control_replay" / "independent_hs_branch_control_replay_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(df) == 9
    assert int((df["record_type"] == "branch").sum()) == 8
    assert metadata["branch_row_count"] == 8
    assert metadata["optimization_rerun"] is False
    assert metadata["high_fidelity_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["passes_tolerance"] is True
    assert metadata["max_branch_terminal_error_delta"] <= 1.0e-12
    assert metadata["summary"][0]["branch_row_count"] == 8
    assert metadata["summary"][0]["selected_worst_replay_delta"] <= 1.0e-12
    assert (tmp_path / "tables" / "independent_hs_branch_control_replay" / "independent_hs_branch_control_replay_table.tex").exists()

    branch_path = tmp_path / first_branch_path
    payload = json.loads(branch_path.read_text(encoding="utf-8"))
    payload["recorded_branch_terminal_error"] = 123.0
    branch_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        replay.run(args)
