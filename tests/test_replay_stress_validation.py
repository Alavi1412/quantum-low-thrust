from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_stress_validation_reproduces_recorded_nominal_baselines():
    module = load_script_module(
        "run_replay_stress_validation_unit",
        ROOT / "scripts" / "run_replay_stress_validation.py",
    )

    df, skipped = module.build_replay_stress_validation()

    assert skipped == []
    assert len(df) == 15
    assert set(df["case"]) == {
        "all_single_p04_warm_from_p03",
        "two_segment_n8_p03_cold",
        "ihs_phase_p04_amax02_warm_from_p03",
    }
    assert set(df["replay_variant"]) == {
        "baseline_source_substeps",
        "refined_2x_substeps",
        "refined_4x_substeps",
        "accel_scale_0p99_source_substeps",
        "accel_scale_1p01_source_substeps",
    }

    baseline = df[df["replay_variant"] == "baseline_source_substeps"].copy()
    assert len(baseline) == 3
    assert baseline["baseline_reproduces_recorded"].map(bool).all()
    assert (baseline["absolute_delta_from_recorded_nominal_error"] <= 1.0e-12).all()

    expected = {
        "all_single_p04_warm_from_p03": 0.0530980832118395,
        "two_segment_n8_p03_cold": 0.0612012101866208,
        "ihs_phase_p04_amax02_warm_from_p03": 0.0197147568098046,
    }
    for case, recorded in expected.items():
        row = baseline.loc[baseline["case"] == case].iloc[0]
        assert abs(float(row["nominal_error"]) - recorded) <= 1.0e-12
        assert abs(float(row["recorded_nominal_error"]) - recorded) <= 1.0e-15

    ihs = df[df["case"] == "ihs_phase_p04_amax02_warm_from_p03"]
    assert ihs["midpoint_controls_replayed"].map(bool).all()
    continuation = df[df["case"] != "ihs_phase_p04_amax02_warm_from_p03"]
    assert not continuation["midpoint_controls_replayed"].map(bool).any()


def test_replay_stress_validation_writes_deterministic_artifacts_without_optimization(tmp_path):
    module = load_script_module(
        "run_replay_stress_validation_artifact_unit",
        ROOT / "scripts" / "run_replay_stress_validation.py",
    )

    kwargs = {
        "results_dir": tmp_path / "results",
        "tables_dir": tmp_path / "tables",
        "command": "unit-test",
    }
    first = module.write_artifacts(**kwargs)
    first_bytes = {
        "csv": first["csv_path"].read_bytes(),
        "metadata": first["metadata_path"].read_bytes(),
        "table": first["table_path"].read_bytes(),
    }
    second = module.write_artifacts(**kwargs)

    assert second["csv_path"].read_bytes() == first_bytes["csv"]
    assert second["metadata_path"].read_bytes() == first_bytes["metadata"]
    assert second["table_path"].read_bytes() == first_bytes["table"]

    metadata = json.loads(second["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_only_recorded_control_sidecars"] is True
    assert metadata["high_fidelity_claim"] is False
    assert "Recorded nominal-control sidecars" in metadata["source_mode"]
    assert "no least-squares optimization" in metadata["source_mode"]
    assert "not a high-fidelity validation model" in " ".join(metadata["interpretation_limits"])
    assert metadata["baseline_reproduction"]["all_baselines_reproduced"] is True
    assert metadata["baseline_reproduction"]["max_abs_delta"] <= 1.0e-12

    input_paths = {item["path"] for item in metadata["input_artifacts"]}
    assert "data/results/continuation_extension_suite/controls/all_single_p04_warm_from_p03_nominal_controls.json" in input_paths
    assert "data/results/continuation_extension_suite/controls/two_segment_n8_p03_cold_nominal_controls.json" in input_paths
    assert (
        "data/results/independent_hs_continuation_baseline/controls/"
        "ihs_phase_p04_amax02_warm_from_p03_nominal_controls.json"
    ) in input_paths

    csv_df = pd.read_csv(second["csv_path"])
    assert len(csv_df) == 15
    assert "accel_scale_0p99_source_substeps" in set(csv_df["replay_variant"])
    assert "accel_scale_1p01_source_substeps" in set(csv_df["replay_variant"])
