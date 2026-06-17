from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evidence_synthesis_replays_representative_recorded_rows():
    module = load_script_module(
        "run_evidence_synthesis_unit",
        ROOT / "scripts" / "run_evidence_synthesis.py",
    )

    synthesis = module.build_synthesis()
    has_ihs_replay = module.independent_hs_branch_replay_available()
    has_ihs_bicircular = module.independent_hs_bicircular_phase_stress_available()
    has_ihs_horizons = module.independent_hs_horizons_solar_tidal_replay_available()
    has_ihs_point_mass = module.independent_hs_horizons_point_mass_retuning_available()
    has_ihs_multi_point_mass = module.independent_hs_horizons_multi_epoch_point_mass_retuning_available()

    expected_rows = {
        "phase_shift_tight_threshold_counts",
        "continuation_all_single_p04",
        "continuation_two_segment_n8_p03",
        "direct_collocation_selected_p04",
        "ihs_tighter_thrust_p04",
        "ihs_all_configured_headroom_p04_amax02",
        "tail_coast_hard_catalog_all_one_two",
        "ihs_hard_catalog_selected_failure",
    }
    if has_ihs_replay:
        expected_rows.add("ihs_branch_control_replay_p04_amax02")
    if has_ihs_bicircular:
        expected_rows.add("ihs_bicircular_phase_stress_polish_p04_amax02")
    if has_ihs_horizons:
        expected_rows.add("ihs_horizons_solar_tidal_replay_polish_p04_amax02")
    if has_ihs_point_mass:
        expected_rows.add("ihs_horizons_point_mass_retuning_polish_p04_amax02")
    if has_ihs_multi_point_mass:
        expected_rows.add("ihs_horizons_multi_epoch_point_mass_retuning_polish_p04_amax02")
    assert len(synthesis) == len(expected_rows)
    assert set(synthesis["row_id"]) == expected_rows

    def row(row_id: str) -> pd.Series:
        return synthesis.loc[synthesis["row_id"] == row_id].iloc[0]

    threshold = row("phase_shift_tight_threshold_counts")
    assert "random=0/30" in threshold["pass_status_note"]
    assert "qaoa_statevector=0/30" in threshold["pass_status_note"]
    assert "all_windows_continuous=30/30" in threshold["pass_status_note"]
    assert threshold["tight_0p05_0p09_all_mask_pass"] == "sampled methods 0/30; all-windows 30/30"

    all_single = row("continuation_all_single_p04")
    assert all_single["nominal_error"] == "0.0530980832118395"
    assert all_single["selected_worst_error"] == "0.0139134347944667"
    assert all_single["all_mask_worst_error"] == "0.0139134347944667"
    assert all_single["stringent_0p065_0p10_all_mask_pass"] == "True"
    assert all_single["tight_0p05_0p09_all_mask_pass"] == "False"

    two_segment = row("continuation_two_segment_n8_p03")
    assert two_segment["nominal_error"] == "0.0612012101866208"
    assert two_segment["selected_worst_error"] == "0.0434908055815499"
    assert two_segment["all_mask_worst_error"] == "0.0434908055815499"
    assert two_segment["stringent_0p065_0p10_all_mask_pass"] == "True"

    direct = row("direct_collocation_selected_p04")
    assert direct["nominal_error"] == "0.03567443236113212"
    assert direct["selected_worst_error"] == "0.019117085167272115"
    assert direct["all_mask_worst_error"] == "0.06024618745953626"
    assert direct["tight_0p05_0p09_all_mask_pass"] == "True"

    ihs_phase = row("ihs_tighter_thrust_p04")
    assert ihs_phase["nominal_error"] == "0.0197147568098046"
    assert ihs_phase["selected_worst_error"] == "0.0187821107883081"
    assert ihs_phase["all_mask_worst_error"] == "0.0531572965780589"
    assert ihs_phase["tight_0p05_0p09_all_mask_pass"] == "True"

    ihs_all = row("ihs_all_configured_headroom_p04_amax02")
    assert ihs_all["nominal_error"] == "0.011115187774142957"
    assert ihs_all["selected_worst_error"] == "0.07741645121655767"
    assert ihs_all["all_mask_worst_error"] == "0.07741645121655767"
    assert ihs_all["configured_pass"] == "True"
    assert ihs_all["tight_0p05_0p09_all_mask_pass"] == "True"
    assert "8/8 configured one-segment masks" in ihs_all["mask_scope"]

    if has_ihs_replay:
        ihs_replay = row("ihs_branch_control_replay_p04_amax02")
        assert ihs_replay["configured_pass"] == "True"
        assert "16 branch replay rows" in ihs_replay["mask_scope"]
        assert "max replay delta=0.0" in ihs_replay["pass_status_note"]
        assert "without adding high-fidelity" in ihs_replay["practitioner_interpretation"]

    if has_ihs_bicircular:
        ihs_stress = row("ihs_bicircular_phase_stress_polish_p04_amax02")
        assert ihs_stress["configured_pass"] == "True"
        assert ihs_stress["nominal_error"] == "0.022138676654057693"
        assert ihs_stress["selected_worst_error"] == "0.08557051343145317"
        assert ihs_stress["all_mask_worst_error"] == "0.08557051343145317"
        assert ihs_stress["tight_0p05_0p09_all_mask_pass"] == "True"
        assert "branch 64/64" in ihs_stress["pass_status_note"]
        assert "not flight validation" in ihs_stress["practitioner_interpretation"]

    if has_ihs_horizons:
        ihs_horizons = row("ihs_horizons_solar_tidal_replay_polish_p04_amax02")
        assert ihs_horizons["configured_pass"] == "True"
        assert float(ihs_horizons["nominal_error"]) < 0.09
        assert float(ihs_horizons["selected_worst_error"]) < 0.17
        assert ihs_horizons["all_mask_worst_error"] == ihs_horizons["selected_worst_error"]
        assert ihs_horizons["tight_0p05_0p09_all_mask_pass"] == "True"
        assert "branch 8/8" in ihs_horizons["pass_status_note"]
        assert "cached JPL Horizons vectors" in ihs_horizons["practitioner_interpretation"]
        assert "not SPICE/high-fidelity/flight validation" in ihs_horizons["practitioner_interpretation"]

    if has_ihs_point_mass:
        ihs_point_mass = row("ihs_horizons_point_mass_retuning_polish_p04_amax02")
        assert ihs_point_mass["configured_pass"] == "True"
        assert float(ihs_point_mass["nominal_error"]) < 0.09
        assert float(ihs_point_mass["selected_worst_error"]) < 0.17
        assert ihs_point_mass["all_mask_worst_error"] == ihs_point_mass["selected_worst_error"]
        assert ihs_point_mass["tight_0p05_0p09_all_mask_pass"] == "True"
        assert "persisted replay failed" in ihs_point_mass["pass_status_note"]
        assert "retuned branch 8/8" in ihs_point_mass["pass_status_note"]
        assert "Persisted controls fail direct ephemeris point-mass replay" in ihs_point_mass["practitioner_interpretation"]
        assert "not SPICE/full high-fidelity/flight validation" in ihs_point_mass["practitioner_interpretation"]

    if has_ihs_multi_point_mass:
        ihs_multi_point_mass = row("ihs_horizons_multi_epoch_point_mass_retuning_polish_p04_amax02")
        assert ihs_multi_point_mass["configured_pass"] == "True"
        assert ihs_multi_point_mass["nominal_error"] == "0.02143944130524006"
        assert ihs_multi_point_mass["selected_worst_error"] == "0.02473065115224942"
        assert ihs_multi_point_mass["all_mask_worst_error"] == ihs_multi_point_mass["selected_worst_error"]
        assert ihs_multi_point_mass["tight_0p05_0p09_all_mask_pass"] == "True"
        assert "4 representative 2026 epochs" in ihs_multi_point_mass["mask_scope"]
        assert "nominal direct replay fails in 4/4 epochs" in ihs_multi_point_mass["pass_status_note"]
        assert "direct branch pass 18/32 (July 8/8)" in ihs_multi_point_mass["pass_status_note"]
        assert "retuned branch 32/32" in ihs_multi_point_mass["pass_status_note"]
        assert "no longer a single representative-epoch check" in ihs_multi_point_mass["practitioner_interpretation"]
        assert "not SPICE/full high-fidelity/flight validation" in ihs_multi_point_mass[
            "practitioner_interpretation"
        ]

    tail = row("tail_coast_hard_catalog_all_one_two")
    assert tail["nominal_error"] == "0.02299233817855882"
    assert tail["selected_worst_error"] == "0.0936063931709301"
    assert tail["all_mask_worst_error"] == "0.0936063931709301"
    assert tail["configured_pass"] == "True"
    assert tail["near_tight_0p05_0p10_all_mask_pass"] == "True"
    assert tail["tight_0p05_0p09_all_mask_pass"] == "False"

    ihs_hard = row("ihs_hard_catalog_selected_failure")
    assert ihs_hard["nominal_error"] == "4.264391829064117"
    assert ihs_hard["selected_worst_error"] == "11.084255309095791"
    assert ihs_hard["all_mask_worst_error"] == "11.084255309095791"
    assert ihs_hard["configured_pass"] == "False"


def test_evidence_synthesis_writes_deterministic_artifacts_without_optimization(tmp_path):
    module = load_script_module(
        "run_evidence_synthesis_artifact_unit",
        ROOT / "scripts" / "run_evidence_synthesis.py",
    )

    artifacts = module.write_artifacts(
        results_dir=tmp_path / "results",
        tables_dir=tmp_path / "tables",
        command="unit-test",
    )

    csv_path = artifacts["csv_path"]
    metadata_path = artifacts["metadata_path"]
    table_path = artifacts["evidence_synthesis_table_tex"]
    lessons_path = artifacts["practitioner_lessons_table_tex"]

    assert csv_path.exists()
    assert metadata_path.exists()
    assert table_path.exists()
    assert lessons_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    has_ihs_replay = module.independent_hs_branch_replay_available()
    has_ihs_bicircular = module.independent_hs_bicircular_phase_stress_available()
    has_ihs_horizons = module.independent_hs_horizons_solar_tidal_replay_available()
    has_ihs_point_mass = module.independent_hs_horizons_point_mass_retuning_available()
    has_ihs_multi_point_mass = module.independent_hs_horizons_multi_epoch_point_mass_retuning_available()
    expected_rows = (
        8
        + int(has_ihs_replay)
        + int(has_ihs_bicircular)
        + int(has_ihs_horizons)
        + int(has_ihs_point_mass)
        + int(has_ihs_multi_point_mass)
    )
    expected_inputs = (
        12
        + (2 if has_ihs_replay else 0)
        + (2 if has_ihs_bicircular else 0)
        + (2 if has_ihs_horizons else 0)
        + (2 if has_ihs_point_mass else 0)
        + (2 if has_ihs_multi_point_mass else 0)
    )
    assert metadata["optimization_rerun"] is False
    assert metadata["row_count"] == expected_rows
    assert "Recorded CSV/JSON artifacts only" in metadata["source_mode"]
    assert "Runtime is intentionally omitted" in metadata["determinism_note"]
    assert len(metadata["input_artifacts"]) == expected_inputs

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == expected_rows
    assert "tail_coast_hard_catalog_all_one_two" in set(csv_df["row_id"])
    assert "ihs_all_configured_headroom_p04_amax02" in set(csv_df["row_id"])
    if has_ihs_replay:
        assert "ihs_branch_control_replay_p04_amax02" in set(csv_df["row_id"])
    if has_ihs_bicircular:
        assert "ihs_bicircular_phase_stress_polish_p04_amax02" in set(csv_df["row_id"])
    if has_ihs_horizons:
        assert "ihs_horizons_solar_tidal_replay_polish_p04_amax02" in set(csv_df["row_id"])
    if has_ihs_point_mass:
        assert "ihs_horizons_point_mass_retuning_polish_p04_amax02" in set(csv_df["row_id"])
    if has_ihs_multi_point_mass:
        assert "ihs_horizons_multi_epoch_point_mass_retuning_polish_p04_amax02" in set(csv_df["row_id"])
    table = table_path.read_text(encoding="utf-8")
    assert "0.05/0.10: True; 0.05/0.09: False" in table
    assert "sampled methods 0/30; all-windows 30/30" in table
