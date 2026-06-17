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

from qlt.bicircular import SolarTidalParameters
from qlt.bicircular_tail_coast_recovery import terminal_error_bicircular
from qlt.cr3bp import propagate_controls_batch
from qlt.experiment import make_objective_config
from qlt.objective import state_error


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


def test_bicircular_terminal_error_zero_solar_mu_matches_cr3bp():
    cfg = SimpleNamespace(
        mu=0.01215058560962404,
        tf=0.12,
        n_segments=3,
        substeps=2,
        amax=0.2,
        position_scale=1.0,
        velocity_scale=0.35,
    )
    state0 = np.array([1.02, 0.01, 0.02, 0.0, 0.03, -0.01], dtype=float)
    controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, -0.01], [0.0, 0.0, 0.0]], dtype=float)
    target = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)[0][0]

    assert terminal_error_bicircular(
        state0,
        target,
        cfg,
        controls,
        phase_rad=0.5,
        parameters=SolarTidalParameters(sun_mu_ratio=0.0),
    ) == pytest.approx(0.0, abs=1.0e-14)


def make_bicircular_tail_coast_recovery_fixture(monkeypatch, tmp_path, module_name: str):
    module = load_script_module(
        module_name,
        ROOT / "scripts" / "run_bicircular_tail_coast_recovery.py",
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
            "tail_nominal": {"max_nfev": 1, "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0}},
            "branch": {"max_nfev": 1, "weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0}},
            "cases": [
                {
                    "case_id": "tail_replay_unit",
                    "purpose": "unit retuning",
                    "transfer_time": 0.1,
                    "amax": 0.2,
                    "segments": 2,
                    "tail_coast_segments": 1,
                    "outage_lengths": [1],
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

    cfg = make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    nominal_controls = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    target = propagate_controls_batch(initial, nominal_controls, cfg.mu, cfg.tf, cfg.substeps)[0][0]
    branch_controls = [
        np.array([[0.0, 0.0, 0.0], [0.015, 0.0, 0.0]], dtype=float),
        np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float),
    ]

    nominal_path = controls_dir / "tail_replay_unit_nominal_controls.json"
    nominal_error = float(state_error(target, target, cfg.position_scale, cfg.velocity_scale))
    nominal_sha = write_deterministic_json(
        nominal_path,
        {
            "sidecar_type": "tail_coast_nominal_controls",
            "suite_case_id": "tail_replay_unit",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "nominal_error": nominal_error,
            "controls": nominal_controls.tolist(),
        },
    )
    branch_entries = []
    for order, (mask_index, mask, controls) in enumerate(
        [
            (0, [0, 1], branch_controls[0]),
            (1, [1, 0], branch_controls[1]),
        ]
    ):
        branch_path = controls_dir / f"tail_replay_unit_branch_{order:03d}_mask_{mask_index:03d}_controls.json"
        branch_error = float(
            state_error(
                propagate_controls_batch(initial, controls, cfg.mu, cfg.tf, cfg.substeps)[0][0],
                target,
                cfg.position_scale,
                cfg.velocity_scale,
            )
        )
        branch_sha = write_deterministic_json(
            branch_path,
            {
                "sidecar_type": "tail_coast_branch_controls",
                "suite_case_id": "tail_replay_unit",
                "branch_order": order,
                "mask_index": mask_index,
                "outage_mask": mask,
                "target_state": target.tolist(),
                "thresholds": config["objective"]["thresholds"],
                "recovery_start": 1 + order,
                "recovery_segments": 1 - order,
                "terminal_error": branch_error,
                "branch_controls": controls.tolist(),
                "accepted_branch_weights": {"terminal": 4.0, "control": 0.0, "smooth": 0.0, "continuity": 0.0},
            },
        )
        branch_entries.append(
            {
                "branch_order": order,
                "mask_index": mask_index,
                "outage_mask": mask,
                "path": f"data/results/tail_replay_test/controls/{branch_path.name}",
                "sha256": branch_sha,
            }
        )

    manifest_path = controls_dir / "tail_replay_unit_branch_control_manifest.json"
    manifest_sha = write_deterministic_json(
        manifest_path,
        {
            "sidecar_type": "tail_coast_branch_control_manifest",
            "suite_case_id": "tail_replay_unit",
            "target_state": target.tolist(),
            "thresholds": config["objective"]["thresholds"],
            "nominal_control_path": f"data/results/tail_replay_test/controls/{nominal_path.name}",
            "nominal_control_sha256": nominal_sha,
            "branch_control_sidecar_count": 2,
            "branch_control_sidecars": branch_entries,
        },
    )
    pd.DataFrame(
        [
            {
                "suite_case_id": "tail_replay_unit",
                "branch_control_replay_ready": True,
                "branch_control_manifest_path": f"data/results/tail_replay_test/controls/{manifest_path.name}",
                "branch_control_manifest_sha256": manifest_sha,
                "nominal_error": nominal_error,
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
            "--case-id",
            "tail_replay_unit",
            "--results-dir",
            str(output_results),
            "--tables-dir",
            str(output_tables),
            "--nominal-max-nfev",
            "0",
            "--branch-max-nfev",
            "0",
            "--max-branches",
            "1",
        ]
    )
    resumed_args = module.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--source-states",
            str(source_states),
            "--case-id",
            "tail_replay_unit",
            "--results-dir",
            str(output_results),
            "--tables-dir",
            str(output_tables),
            "--nominal-max-nfev",
            "0",
            "--branch-max-nfev",
            "0",
            "--resume",
        ]
    )
    return SimpleNamespace(
        module=module,
        first_args=args,
        resumed_args=resumed_args,
        config_path=config_path,
        source_states=source_states,
        output_results=output_results,
        output_tables=output_tables,
    )


def resolve_fixture_path(tmp_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else tmp_path / path


def test_bicircular_tail_coast_recovery_script_checkpoint_resume(monkeypatch, tmp_path):
    fixture = make_bicircular_tail_coast_recovery_fixture(
        monkeypatch,
        tmp_path,
        "run_bicircular_tail_coast_recovery_unit_resume_valid",
    )
    module = fixture.module
    first = module.run(fixture.first_args)
    assert len(first) == 2
    checkpoint = json.loads((fixture.output_results / module.CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert checkpoint["complete"] is False
    assert checkpoint["row_count"] == 2
    branch_row = next(row for row in checkpoint["rows"] if row["record_type"] == "branch")
    branch_sidecar = json.loads(resolve_fixture_path(tmp_path, branch_row["retuned_controls_path"]).read_text(encoding="utf-8"))
    assert branch_sidecar["record_type"] == "branch"

    second = module.run(fixture.resumed_args)
    assert len(second) == 3
    metadata = json.loads((fixture.output_results / module.METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["bicircular_tail_coast_retuned_recovery"] is True
    assert metadata["retuning_optimization_rerun"] is True
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["high_fidelity_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["final_summary"]["package_complete"] is True
    assert metadata["final_summary"]["branch_row_count"] == 2
    assert (fixture.output_tables / module.TABLE_NAME).is_file()

    budgeted_completed_resume_args = module.build_parser().parse_args(
        [
            "--config",
            str(fixture.config_path),
            "--source-states",
            str(fixture.source_states),
            "--case-id",
            "tail_replay_unit",
            "--results-dir",
            str(fixture.output_results),
            "--tables-dir",
            str(fixture.output_tables),
            "--nominal-max-nfev",
            "0",
            "--branch-max-nfev",
            "0",
            "--resume",
            "--runtime-budget-seconds",
            "0.001",
        ]
    )
    third = module.run(budgeted_completed_resume_args)
    assert len(third) == 3
    checkpoint = json.loads((fixture.output_results / module.CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert checkpoint["complete"] is True


def test_bicircular_tail_coast_recovery_resume_accepts_legacy_branch_sidecar_without_record_type(
    monkeypatch,
    tmp_path,
):
    fixture = make_bicircular_tail_coast_recovery_fixture(
        monkeypatch,
        tmp_path,
        "run_bicircular_tail_coast_recovery_unit_resume_legacy",
    )
    module = fixture.module
    first = module.run(fixture.first_args)
    assert len(first) == 2

    checkpoint_path = fixture.output_results / module.CHECKPOINT_NAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    branch_row = next(row for row in checkpoint["rows"] if row["record_type"] == "branch")
    branch_sidecar_path = resolve_fixture_path(tmp_path, branch_row["retuned_controls_path"])
    branch_sidecar = json.loads(branch_sidecar_path.read_text(encoding="utf-8"))
    assert branch_sidecar.pop("record_type") == "branch"
    legacy_sha = write_deterministic_json(branch_sidecar_path, branch_sidecar)
    branch_row["retuned_controls_sha256"] = legacy_sha
    write_deterministic_json(checkpoint_path, checkpoint)

    second = module.run(fixture.resumed_args)

    assert len(second) == 3
    metadata = json.loads((fixture.output_results / module.METADATA_NAME).read_text(encoding="utf-8"))
    assert metadata["final_summary"]["package_complete"] is True
    assert metadata["final_summary"]["branch_row_count"] == 2


def test_bicircular_tail_coast_recovery_resume_rejects_legacy_branch_identity_mismatch(
    monkeypatch,
    tmp_path,
):
    fixture = make_bicircular_tail_coast_recovery_fixture(
        monkeypatch,
        tmp_path,
        "run_bicircular_tail_coast_recovery_unit_resume_legacy_mismatch",
    )
    module = fixture.module
    first = module.run(fixture.first_args)
    assert len(first) == 2

    checkpoint_path = fixture.output_results / module.CHECKPOINT_NAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    branch_row = next(row for row in checkpoint["rows"] if row["record_type"] == "branch")
    branch_sidecar_path = resolve_fixture_path(tmp_path, branch_row["retuned_controls_path"])
    branch_sidecar = json.loads(branch_sidecar_path.read_text(encoding="utf-8"))
    branch_sidecar.pop("record_type")
    branch_sidecar["mask_index"] = int(branch_row["mask_index"]) + 1
    branch_row["retuned_controls_sha256"] = write_deterministic_json(branch_sidecar_path, branch_sidecar)
    write_deterministic_json(checkpoint_path, checkpoint)

    csv_path = fixture.output_results / module.RESULT_CSV_NAME
    csv_before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    skipped_branch_path = fixture.output_results / "controls" / "tail_replay_unit_phase_0_branch_001_mask_001_controls.json"
    assert not skipped_branch_path.exists()

    with pytest.raises(RuntimeError, match="invalid resume checkpoint row.*mask_index.*mismatch"):
        module.run(fixture.resumed_args)

    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == csv_before
    assert not skipped_branch_path.exists()


def test_bicircular_tail_coast_recovery_resume_rejects_corrupt_checkpoint_branch_sidecar(
    monkeypatch,
    tmp_path,
):
    fixture = make_bicircular_tail_coast_recovery_fixture(
        monkeypatch,
        tmp_path,
        "run_bicircular_tail_coast_recovery_unit_resume_corrupt",
    )
    module = fixture.module
    first = module.run(fixture.first_args)
    assert len(first) == 2

    checkpoint_path = fixture.output_results / module.CHECKPOINT_NAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    branch_row = next(row for row in checkpoint["rows"] if row["record_type"] == "branch")
    branch_sidecar_path = resolve_fixture_path(tmp_path, branch_row["retuned_controls_path"])
    branch_sidecar = json.loads(branch_sidecar_path.read_text(encoding="utf-8"))
    branch_sidecar["settings_fingerprint"] = "corrupted-after-checkpoint"
    branch_sidecar_path.write_text(
        json.dumps(branch_sidecar, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = fixture.output_results / module.RESULT_CSV_NAME
    csv_before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    skipped_branch_path = fixture.output_results / "controls" / "tail_replay_unit_phase_0_branch_001_mask_001_controls.json"
    assert not skipped_branch_path.exists()

    with pytest.raises(RuntimeError, match="invalid resume checkpoint row.*sha256 mismatch"):
        module.run(fixture.resumed_args)

    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == csv_before
    assert not skipped_branch_path.exists()


def test_real_bicircular_tail_coast_recovery_package_metadata_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "bicircular_tail_coast_recovery"
        / "bicircular_tail_coast_recovery_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("bicircular tail-coast recovery package has not been generated")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["bicircular_tail_coast_retuned_recovery"] is True
    assert metadata["retuning_optimization_rerun"] is True
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["high_fidelity_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["scope"]["phase_degrees"] == [0.0]
    assert metadata["scope"]["fixed_final_time"] is True
    assert metadata["scope"]["outage_lengths"] == [1, 2]
    assert metadata["final_summary"]["package_complete"] is True
    assert metadata["final_summary"]["nominal_error"] == pytest.approx(0.31677192167859453)
    assert metadata["final_summary"]["nominal_pass"] is False
    assert metadata["final_summary"]["branch_row_count"] == 27
    assert metadata["final_summary"]["expected_branch_count"] == 27
    assert metadata["final_summary"]["branch_pass_count"] == 19
    assert metadata["final_summary"]["max_branch_error"] == pytest.approx(6.029904532225566)
    assert metadata["final_summary"]["meets_thresholds"] is False
    assert metadata["final_summary"]["strict_branch_pass_count"] == 16
    assert metadata["final_summary"]["strict_meets_thresholds"] is False
    assert metadata["final_summary"]["total_nfev"] == 1027
    assert metadata["final_summary"]["optimizer_success_count"] == 25
    assert metadata["final_summary"]["optimizer_ran_count"] == 26
    assert "not SPICE ephemeris validation" in metadata["interpretation_limits"][1]
