from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_path(relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else ROOT / path


def test_multi_epoch_summary_aggregation_smoke():
    module = load_script_module(
        "run_independent_hs_horizons_multi_epoch_point_mass_retuning_unit",
        ROOT / "scripts" / "run_independent_hs_horizons_multi_epoch_point_mass_retuning.py",
    )

    summary = module._aggregate_overall_summary(
        [
            {
                "row_count": 9,
                "nominal_row_count": 1,
                "branch_row_count": 8,
                "direct_replay_failure_recorded": True,
                "retuned_all_rows_pass": True,
                "retuned_nominal_point_mass_error": 0.02,
                "retuned_branch_point_mass_worst_error": 0.03,
                "persisted_nominal_point_mass_replay_error": 0.20,
                "persisted_branch_point_mass_replay_worst_error": 0.30,
                "retuned_branch_pass_count": 8,
                "scipy_success_count": 9,
                "total_nfev": 11,
                "total_runtime_seconds": 1.5,
            },
            {
                "row_count": 9,
                "nominal_row_count": 1,
                "branch_row_count": 8,
                "direct_replay_failure_recorded": True,
                "retuned_all_rows_pass": True,
                "retuned_nominal_point_mass_error": 0.01,
                "retuned_branch_point_mass_worst_error": 0.04,
                "persisted_nominal_point_mass_replay_error": 0.40,
                "persisted_branch_point_mass_replay_worst_error": 0.25,
                "retuned_branch_pass_count": 8,
                "scipy_success_count": 9,
                "total_nfev": 13,
                "total_runtime_seconds": 2.5,
            },
        ]
    )

    assert summary["epoch_count"] == 2
    assert summary["row_count"] == 18
    assert summary["branch_rows_per_epoch"] == [8, 8]
    assert summary["direct_replay_failure_epoch_count"] == 2
    assert summary["all_epochs_record_direct_replay_failure"] is True
    assert summary["retuned_all_epochs_all_rows_pass"] is True
    assert summary["retuned_nominal_worst_over_epochs"] == pytest.approx(0.02)
    assert summary["retuned_branch_worst_over_epochs"] == pytest.approx(0.04)
    assert summary["persisted_nominal_replay_worst_over_epochs"] == pytest.approx(0.40)
    assert summary["persisted_branch_replay_worst_over_epochs"] == pytest.approx(0.30)
    assert summary["retuned_branch_pass_count_total"] == 16
    assert summary["scipy_success_count_total"] == 18
    assert summary["total_nfev"] == 24


def test_real_multi_epoch_point_mass_retuning_package_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "independent_hs_horizons_multi_epoch_point_mass_retuning"
        / "independent_hs_horizons_multi_epoch_point_mass_retuning_metadata.json"
    )
    csv_path = metadata_path.with_name("independent_hs_horizons_multi_epoch_point_mass_retuning.csv")
    if not metadata_path.exists() or not csv_path.exists():
        pytest.skip("multi-epoch independent-HS cached-Horizons point-mass retuning package has not been generated")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    nominal = df[df["record_type"] == "nominal"]
    branch = df[df["record_type"] == "branch"]
    overall = metadata["overall_summary"]

    assert metadata["cached_horizons_earth_moon_sun_multi_epoch_point_mass_retuning"] is True
    assert metadata["multi_epoch_cached_horizons_point_mass_stress_retuning"] is True
    assert metadata["optimization_rerun"] is True
    assert metadata["retuning"] is True
    assert metadata["uses_cached_horizons_vectors"] is True
    assert metadata["runtime_network_dependency"] is False
    for flag_name in (
        "spice_ephemeris_validation",
        "high_fidelity_validation",
        "high_fidelity_flight_validation",
        "production_solver_parity_claim",
        "fuel_optimality_claim",
        "doi_claim",
        "quantum_advantage_claim",
    ):
        assert metadata[flag_name] is False

    assert metadata["epoch_count"] == 4
    assert metadata["row_count"] == 36
    assert metadata["nominal_row_count"] == 4
    assert metadata["branch_row_count"] == 32
    assert len(metadata["epochs"]) == 4
    assert len(df) == metadata["row_count"]
    assert len(nominal) == metadata["nominal_row_count"]
    assert len(branch) == metadata["branch_row_count"]
    assert set(df["epoch_id"]) == {"2026jan01", "2026apr01", "2026jul01", "2026oct01"}
    assert df.groupby(["epoch_id", "record_type"]).size().to_dict() == {
        ("2026apr01", "branch"): 8,
        ("2026apr01", "nominal"): 1,
        ("2026jan01", "branch"): 8,
        ("2026jan01", "nominal"): 1,
        ("2026jul01", "branch"): 8,
        ("2026jul01", "nominal"): 1,
        ("2026oct01", "branch"): 8,
        ("2026oct01", "nominal"): 1,
    }

    assert overall["epoch_count"] == 4
    assert overall["row_count"] == 36
    assert overall["nominal_row_count"] == 4
    assert overall["branch_row_count"] == 32
    assert overall["branch_rows_per_epoch"] == [8, 8, 8, 8]
    assert overall["direct_replay_failure_epoch_count"] == 4
    assert overall["all_epochs_record_direct_replay_failure"] is True
    assert overall["retuned_all_epochs_all_rows_pass"] is True
    assert overall["retuned_branch_pass_count_total"] == 32
    assert overall["branch_row_count_total"] == 32
    assert overall["scipy_success_count_total"] == 36
    assert overall["total_nfev"] == 197
    assert overall["persisted_nominal_replay_worst_over_epochs"] == pytest.approx(0.3812580376880591, abs=1e-12)
    assert overall["persisted_branch_replay_worst_over_epochs"] == pytest.approx(0.3797450961017463, abs=1e-12)
    assert overall["retuned_nominal_worst_over_epochs"] == pytest.approx(0.02143944130524006, abs=1e-12)
    assert overall["retuned_branch_worst_over_epochs"] == pytest.approx(0.02473065115224942, abs=1e-12)

    assert nominal["point_mass_replay_passes_configured_threshold"].map(bool).sum() == 0
    assert branch["point_mass_replay_passes_configured_threshold"].map(bool).sum() == 18
    assert df["point_mass_retuned_passes_configured_threshold"].map(bool).all()
    assert branch["point_mass_retuned_passes_configured_threshold"].map(bool).sum() == 32
    assert df["scipy_success"].map(bool).all()
    assert nominal["point_mass_retuned_terminal_error"].max() == pytest.approx(
        overall["retuned_nominal_worst_over_epochs"], abs=1e-12
    )
    assert branch["point_mass_retuned_terminal_error"].max() == pytest.approx(
        overall["retuned_branch_worst_over_epochs"], abs=1e-12
    )

    cache_hashes = metadata["cache_sha256_by_epoch"]
    for epoch in metadata["epochs"]:
        assert epoch["row_count"] == 9
        assert epoch["nominal_row_count"] == 1
        assert epoch["branch_row_count"] == 8
        assert epoch["direct_replay_failure_recorded"] is True
        assert epoch["retuned_all_rows_pass"] is True
        assert epoch["retuned_branch_pass_count"] == 8
        assert epoch["scipy_success_count"] == 9
        assert cache_hashes[epoch["epoch_id"]] == epoch["cache_sha256"]
        cache_path = root_path(epoch["cache_path"])
        assert cache_path.is_file()
        assert sha256(cache_path) == epoch["cache_sha256"]
        assert root_path(epoch["single_epoch_metadata_json"]).is_file()
        assert root_path(epoch["single_epoch_results_dir"]).is_dir()

    for column, hash_column in (
        ("source_controls_path", "source_controls_sha256"),
        ("retuned_controls_path", "retuned_controls_sha256"),
    ):
        for row in df[[column, hash_column]].dropna().to_dict(orient="records"):
            path = root_path(str(row[column]))
            assert path.is_file()
            assert sha256(path) == str(row[hash_column])

    table_path = ROOT / "tables" / "independent_hs_horizons_multi_epoch_point_mass_retuning"
    table_path = table_path / "independent_hs_horizons_multi_epoch_point_mass_retuning_table.tex"
    assert table_path.is_file()
