from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.ephemeris_contrast import (
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    HorizonsVectorSample,
    canonical_node_jds,
    canonical_node_times,
    load_spice_cache,
    sample_to_cache_row,
    spice_point_mass_profile_from_cache,
    validate_spice_cache,
    validate_spice_cache_compatibility,
)


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


def sample(jd: float, position_km: list[float], velocity_km_s: list[float]) -> HorizonsVectorSample:
    position = np.asarray(position_km, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    distance = float(np.linalg.norm(position))
    return HorizonsVectorSample(
        jd_tdb=float(jd),
        calendar_tdb=f"JD TDB {float(jd):.9f}",
        position_km=position,
        velocity_km_s=velocity,
        light_time_s=0.0,
        range_km=distance,
        range_rate_km_s=float(np.dot(position, velocity) / max(distance, 1.0e-15)),
    )


def synthetic_spice_cache_fixture() -> tuple[dict[str, object], dict[str, object]]:
    start_jd_tdb = 2461041.5
    tf = 0.5
    n_segments = 2
    node_times = canonical_node_times(tf, n_segments)
    node_jds = canonical_node_jds(
        start_jd_tdb=start_jd_tdb,
        tf=tf,
        n_segments=n_segments,
        canonical_time_unit_seconds=DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    )
    moon = [
        sample(jd, [384000.0 + 100.0 * index, 1000.0 * index, 0.0], [0.0, 1.0, 0.0])
        for index, jd in enumerate(node_jds)
    ]
    sun = [
        sample(jd, [147000000.0 + 1000.0 * index, 0.0, 0.0], [0.0, 30.0, 0.0])
        for index, jd in enumerate(node_jds)
    ]
    kernels = {
        key: {
            "filename": f"{key}.kernel",
            "url": f"https://example.invalid/{key}.kernel",
            "sha256": f"{key}" * 16,
            "bytes": 1,
        }
        for key in ("lsk", "spk", "gm")
    }
    cache = {
        "schema_version": 1,
        "cache_type": "spice_geocentric_moon_sun_vectors",
        "metadata": {
            "start_jd_tdb": start_jd_tdb,
            "canonical_transfer_time": tf,
            "segments": n_segments,
            "canonical_node_times": [float(value) for value in node_times],
            "node_jd_tdb": [float(value) for value in node_jds],
            "canonical_time_unit_seconds": DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
            "reference_distance_km": DEFAULT_REFERENCE_DISTANCE_KM,
            "frame": "J2000",
            "observer": "EARTH",
            "targets": ["MOON", "SUN"],
            "aberration_correction": "NONE",
        },
        "spice": {
            "spiceypy_version": "test",
            "cspice_toolkit_version": "CSPICE_TEST",
        },
        "kernels": kernels,
        "bodies": {
            "moon_geocentric": {
                "target": "MOON",
                "observer": "EARTH",
                "frame": "J2000",
                "aberration_correction": "NONE",
                "samples": [sample_to_cache_row(row) for row in moon],
            },
            "sun_geocentric": {
                "target": "SUN",
                "observer": "EARTH",
                "frame": "J2000",
                "aberration_correction": "NONE",
                "samples": [sample_to_cache_row(row) for row in sun],
            },
        },
    }
    params = {
        "start_jd_tdb": start_jd_tdb,
        "tf": tf,
        "n_segments": n_segments,
        "canonical_time_unit_seconds": DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
        "reference_distance_km": DEFAULT_REFERENCE_DISTANCE_KM,
    }
    return cache, params


def test_spice_cache_validation_and_profile_construction_smoke(tmp_path):
    cache, params = synthetic_spice_cache_fixture()
    validate_spice_cache(cache)
    validate_spice_cache_compatibility(cache, **params)
    cache_path = tmp_path / "tiny_spice_cache.json"
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    loaded = load_spice_cache(cache_path)
    profile = spice_point_mass_profile_from_cache(loaded)

    assert profile.canonical_time.tolist() == pytest.approx([0.0, 0.25, 0.5])
    assert profile.moon_position_km.shape == (3, 3)
    assert profile.sun_position_km.shape == (3, 3)
    assert profile.moon_position_km[0, 0] == pytest.approx(384000.0)
    assert profile.sun_velocity_km_s[-1, 1] == pytest.approx(30.0)

    stale = json.loads(json.dumps(cache))
    stale["metadata"]["node_jd_tdb"] = [float(value) + 0.01 for value in stale["metadata"]["node_jd_tdb"]]
    with pytest.raises(ValueError, match="node_jd_tdb"):
        validate_spice_cache_compatibility(stale, **params)


def test_spice_replay_summary_aggregation_smoke():
    module = load_script_module(
        "run_independent_hs_spice_ephemeris_replay_unit",
        ROOT / "scripts" / "run_independent_hs_spice_ephemeris_replay.py",
    )
    summary = module._overall_summary(
        [
            {
                "row_count": 9,
                "nominal_row_count": 1,
                "branch_row_count": 8,
                "nominal_spice_replay_error": 0.02,
                "branch_spice_replay_worst_error": 0.03,
                "branch_spice_replay_pass_count": 8,
                "nominal_spice_replay_pass": True,
                "max_abs_delta_from_horizons_retuned": 1.0e-9,
                "source_horizons_retuned_nominal_error": 0.021,
                "source_horizons_retuned_branch_worst_error": 0.031,
            },
            {
                "row_count": 9,
                "nominal_row_count": 1,
                "branch_row_count": 8,
                "nominal_spice_replay_error": 0.04,
                "branch_spice_replay_worst_error": 0.05,
                "branch_spice_replay_pass_count": 7,
                "nominal_spice_replay_pass": False,
                "max_abs_delta_from_horizons_retuned": 2.0e-9,
                "source_horizons_retuned_nominal_error": 0.041,
                "source_horizons_retuned_branch_worst_error": 0.051,
            },
        ]
    )

    assert summary["epoch_count"] == 2
    assert summary["row_count"] == 18
    assert summary["branch_row_count_total"] == 16
    assert summary["branch_spice_replay_pass_count_total"] == 15
    assert summary["nominal_spice_replay_pass_count_total"] == 1
    assert summary["nominal_spice_replay_worst_over_epochs"] == pytest.approx(0.04)
    assert summary["branch_spice_replay_worst_over_epochs"] == pytest.approx(0.05)
    assert summary["max_abs_delta_from_horizons_retuned"] == pytest.approx(2.0e-9)


def test_real_independent_hs_spice_ephemeris_replay_package_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "independent_hs_spice_ephemeris_replay"
        / "independent_hs_spice_ephemeris_replay_metadata.json"
    )
    csv_path = metadata_path.with_name("independent_hs_spice_ephemeris_replay.csv")
    if not metadata_path.exists() or not csv_path.exists():
        pytest.skip("independent-HS SPICE ephemeris replay package has not been generated")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    df = pd.read_csv(csv_path)
    nominal = df[df["record_type"] == "nominal"]
    branch = df[df["record_type"] == "branch"]
    overall = metadata["overall_summary"]

    assert metadata["spice_ephemeris_validation"] is True
    assert metadata["spice_derived_ephemeris_replay"] is True
    assert metadata["retuning"] is False
    assert metadata["optimization_rerun"] is False
    assert metadata["high_fidelity_validation"] is False
    assert metadata["high_fidelity_flight_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["doi_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["runtime_network_dependency"] is False
    assert metadata["uses_committed_spice_vector_caches"] is True

    assert metadata["row_count"] == 36
    assert metadata["nominal_row_count"] == 4
    assert metadata["branch_row_count"] == 32
    assert len(df) == 36
    assert len(nominal) == 4
    assert len(branch) == 32
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
    assert overall["branch_spice_replay_pass_count_total"] == 32
    assert overall["branch_row_count_total"] == 32
    assert overall["nominal_spice_replay_pass_count_total"] == 4
    assert overall["nominal_spice_replay_worst_over_epochs"] == pytest.approx(
        0.021439441253166033,
        abs=1.0e-12,
    )
    assert overall["branch_spice_replay_worst_over_epochs"] == pytest.approx(
        0.024730650824609506,
        abs=1.0e-12,
    )
    assert overall["max_abs_delta_from_horizons_retuned"] == pytest.approx(
        3.2763991519857427e-10,
        abs=1.0e-13,
    )
    assert branch["spice_replay_passes_configured_threshold"].map(bool).sum() == 32
    assert nominal["spice_replay_passes_configured_threshold"].map(bool).sum() == 4
    assert df["source_horizons_retuned_passes_configured_threshold"].map(bool).all()
    assert df["no_retune_semantics"].str.contains("No optimization or retuning").all()

    expected_kernel_hashes = {
        "lsk": "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
        "spk": "54d97562a5b094d298b1b8eafa5a2e17e3e010ce85e1a366d07f003ad159323c",
        "gm": "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140",
    }
    for epoch in metadata["epochs"]:
        assert epoch["row_count"] == 9
        assert epoch["nominal_row_count"] == 1
        assert epoch["branch_row_count"] == 8
        assert epoch["branch_spice_replay_pass_count"] == 8
        assert epoch["nominal_spice_replay_pass"] is True
        assert epoch["kernel_sha256"] == expected_kernel_hashes
        cache_path = root_path(epoch["cache_path"])
        assert cache_path.is_file()
        assert sha256(cache_path) == epoch["cache_sha256"]
        cache = load_spice_cache(cache_path)
        assert cache["spice"]["cspice_toolkit_version"] == "CSPICE_N0067"
        assert cache["metadata"]["aberration_correction"] == "NONE"
        assert cache["metadata"]["observer"] == "EARTH"

    for row in df[["source_horizons_retuned_controls_path", "source_horizons_retuned_controls_sha256"]].to_dict(
        orient="records"
    ):
        path = root_path(str(row["source_horizons_retuned_controls_path"]))
        assert path.is_file()
        assert sha256(path) == str(row["source_horizons_retuned_controls_sha256"])

    table_path = (
        ROOT
        / "tables"
        / "independent_hs_spice_ephemeris_replay"
        / "independent_hs_spice_ephemeris_replay_table.tex"
    )
    assert table_path.is_file()
    assert not (ROOT / "data" / "cache" / "spice" / "kernels").exists()
    assert not list((ROOT / "data" / "cache" / "spice").glob("*.bsp"))
    assert not list((ROOT / "data" / "cache" / "spice").glob("*.tls"))
    assert not list((ROOT / "data" / "cache" / "spice").glob("*.tpc"))
