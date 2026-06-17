from __future__ import annotations

import copy
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
    DEFAULT_CACHE_START_JD_TDB,
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    HorizonsVectorSample,
    canonical_node_jds,
    canonical_node_times,
    horizons_rotating_geometry,
    load_horizons_cache,
    parse_horizons_vectors_result,
    sample_to_cache_row,
    samples_from_cache,
    solar_tidal_acceleration_from_sun_vector,
    validate_horizons_cache,
    validate_horizons_cache_compatibility,
)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample(jd: float, position_km: list[float], velocity_km_s: list[float]) -> HorizonsVectorSample:
    position = np.asarray(position_km, dtype=float)
    velocity = np.asarray(velocity_km_s, dtype=float)
    return HorizonsVectorSample(
        jd_tdb=float(jd),
        calendar_tdb=f"JD {float(jd):.9f}",
        position_km=position,
        velocity_km_s=velocity,
        light_time_s=0.0,
        range_km=float(np.linalg.norm(position)),
        range_rate_km_s=0.0,
    )


def compatible_cache_fixture() -> tuple[dict[str, object], dict[str, object]]:
    start_jd_tdb = DEFAULT_CACHE_START_JD_TDB
    tf = 4.0
    n_segments = 2
    canonical_time_unit_seconds = DEFAULT_CANONICAL_TIME_UNIT_SECONDS
    reference_distance_km = DEFAULT_REFERENCE_DISTANCE_KM
    node_times = canonical_node_times(tf, n_segments)
    node_jds = canonical_node_jds(
        start_jd_tdb=start_jd_tdb,
        tf=tf,
        n_segments=n_segments,
        canonical_time_unit_seconds=canonical_time_unit_seconds,
    )
    moon = [
        sample(jd, [384000.0 + 1000.0 * i, 0.0, 0.0], [0.0, 1.0, 0.0])
        for i, jd in enumerate(node_jds)
    ]
    sun = [
        sample(jd, [147000000.0 + 1000.0 * i, 0.0, 0.0], [0.0, 30.0, 0.0])
        for i, jd in enumerate(node_jds)
    ]
    tlist = "\n".join(f"{float(jd):.9f}" for jd in node_jds)
    body_template = {
        "query_url": "https://ssd.jpl.nasa.gov/api/horizons.api?EPHEM_TYPE=VECTORS",
        "request_params": {"TLIST": tlist},
        "response_signature": {"source": "NASA/JPL Horizons API"},
    }
    cache = {
        "schema_version": 1,
        "cache_type": "jpl_horizons_geocentric_moon_sun_vectors",
        "metadata": {
            "start_jd_tdb": start_jd_tdb,
            "canonical_transfer_time": tf,
            "segments": n_segments,
            "canonical_node_times": [float(value) for value in node_times],
            "node_jd_tdb": [float(value) for value in node_jds],
            "canonical_time_unit_seconds": canonical_time_unit_seconds,
            "reference_distance_km": reference_distance_km,
        },
        "bodies": {
            "moon_geocentric": {
                **body_template,
                "samples": [sample_to_cache_row(row) for row in moon],
            },
            "sun_geocentric": {
                **body_template,
                "samples": [sample_to_cache_row(row) for row in sun],
            },
        },
    }
    params = {
        "start_jd_tdb": start_jd_tdb,
        "tf": tf,
        "n_segments": n_segments,
        "canonical_time_unit_seconds": canonical_time_unit_seconds,
        "reference_distance_km": reference_distance_km,
    }
    return cache, params


def test_horizons_rotating_geometry_uses_fixed_reference_distance_for_sun_lu():
    moon = [
        sample(1.0, [300.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        sample(2.0, [500.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ]
    sun = [
        sample(1.0, [1000.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        sample(2.0, [1000.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ]

    geometry = horizons_rotating_geometry(
        moon_samples=moon,
        sun_samples=sun,
        mu=0.0,
        reference_distance_km=200.0,
    )

    assert np.allclose(geometry["em_distance_ratio_to_mean"], [0.75, 1.25])
    assert np.allclose(geometry["em_distance_ratio_to_reference"], [1.5, 2.5])
    assert np.allclose(geometry["sun_barycenter_rotating_lu"][:, 0], [5.0, 5.0])
    assert np.allclose(geometry["sun_distance_lu"], [5.0, 5.0])


def test_horizons_cache_compatibility_rejects_stale_metadata_with_same_node_count():
    cache, params = compatible_cache_fixture()
    validate_horizons_cache(cache)
    validate_horizons_cache_compatibility(cache, **params)

    stale_reference = copy.deepcopy(cache)
    stale_reference["metadata"]["reference_distance_km"] = DEFAULT_REFERENCE_DISTANCE_KM + 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="reference_distance_km"):
        validate_horizons_cache_compatibility(stale_reference, **params)

    stale_nodes = copy.deepcopy(cache)
    stale_nodes["metadata"]["node_jd_tdb"] = [  # type: ignore[index]
        float(value) + 0.01 for value in stale_nodes["metadata"]["node_jd_tdb"]  # type: ignore[index]
    ]
    with pytest.raises(ValueError, match="node_jd_tdb"):
        validate_horizons_cache_compatibility(stale_nodes, **params)

    with pytest.raises(ValueError, match="canonical_transfer_time"):
        validate_horizons_cache_compatibility(cache, **{**params, "tf": float(params["tf"]) + 0.1})


def test_horizons_cache_integrity_and_raw_parser_when_present():
    cache_path = ROOT / "data" / "cache" / "horizons" / "hard_catalog_tail_coast_2026jan01_vectors.json"
    if not cache_path.exists():
        pytest.skip("Horizons cache has not been generated")
    cache = load_horizons_cache(cache_path)
    assert cache["cache_type"] == "jpl_horizons_geocentric_moon_sun_vectors"
    metadata = cache["metadata"]
    assert metadata["start_jd_tdb"] == pytest.approx(2461041.5)
    assert metadata["canonical_transfer_time"] == pytest.approx(4.0)
    assert metadata["segments"] == 14
    assert "horizons_api" in metadata["query_docs"]

    moon = samples_from_cache(cache, "moon_geocentric")
    sun = samples_from_cache(cache, "sun_geocentric")
    assert len(moon) == 15
    assert len(sun) == 15
    assert np.allclose([m.jd_tdb for m in moon], [s.jd_tdb for s in sun], atol=5.0e-10, rtol=0.0)

    moon_raw = cache["bodies"]["moon_geocentric"]["raw_result"]
    reparsed = parse_horizons_vectors_result(moon_raw)
    assert len(reparsed) == len(moon)
    assert reparsed[0].calendar_tdb == moon[0].calendar_tdb
    assert np.allclose(reparsed[0].position_km, moon[0].position_km, atol=1.0e-9, rtol=0.0)

    for body in ("moon_geocentric", "sun_geocentric"):
        body_data = cache["bodies"][body]
        assert body_data["response_signature"]["source"] == "NASA/JPL Horizons API"
        assert "EPHEM_TYPE=VECTORS" in body_data["query_url"]
        assert body_data["request_params"]["CSV_FORMAT"] == "YES"


def test_solar_tidal_acceleration_from_sun_vector_has_differential_origin_zero():
    sun = np.array([10.0, 0.0, 0.0], dtype=float)
    assert np.allclose(
        solar_tidal_acceleration_from_sun_vector(np.zeros(3), sun, sun_mu_ratio=2.0),
        np.zeros(3),
        atol=1.0e-14,
    )
    accel = solar_tidal_acceleration_from_sun_vector(np.array([1.0, 0.0, 0.0]), sun, sun_mu_ratio=2.0)
    assert accel.shape == (3,)
    assert accel[0] > 0.0


def test_horizons_ephemeris_force_model_contrast_script_is_offline_deterministic(tmp_path):
    module = load_script_module(
        "run_horizons_ephemeris_force_model_contrast_unit",
        ROOT / "scripts" / "run_horizons_ephemeris_force_model_contrast.py",
    )
    cache_path = ROOT / "data" / "cache" / "horizons" / "hard_catalog_tail_coast_2026jan01_vectors.json"
    if not cache_path.exists():
        pytest.skip("Horizons cache has not been generated")
    output_results = tmp_path / "results"
    output_tables = tmp_path / "tables"
    args = module.build_parser().parse_args(
        [
            "--cache",
            str(cache_path),
            "--results-dir",
            str(output_results),
            "--tables-dir",
            str(output_tables),
        ]
    )
    first = module.run(args)
    first_bytes = {
        "csv": (output_results / "horizons_ephemeris_force_model_contrast.csv").read_bytes(),
        "metadata": (output_results / "horizons_ephemeris_force_model_contrast_metadata.json").read_bytes(),
        "table": (output_tables / "horizons_ephemeris_force_model_contrast_table.tex").read_bytes(),
    }
    second = module.run(args)

    assert len(first) == 15
    assert len(second) == 15
    assert (output_results / "horizons_ephemeris_force_model_contrast.csv").read_bytes() == first_bytes["csv"]
    assert (output_results / "horizons_ephemeris_force_model_contrast_metadata.json").read_bytes() == first_bytes["metadata"]
    assert (output_tables / "horizons_ephemeris_force_model_contrast_table.tex").read_bytes() == first_bytes["table"]

    metadata = json.loads(
        (output_results / "horizons_ephemeris_force_model_contrast_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["optimization_rerun"] is False
    assert metadata["uses_cached_horizons_vectors"] is True
    assert metadata["runtime_network_dependency"] is False
    assert metadata["force_model_contrast_only"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["accepted_control_high_fidelity_replay"] is False
    assert metadata["row_count"] == 15
    assert metadata["length_unit_scaling"]["reference_distance_km"] == pytest.approx(DEFAULT_REFERENCE_DISTANCE_KM)
    assert metadata["geometry_summary"]["reference_distance_km"] == pytest.approx(DEFAULT_REFERENCE_DISTANCE_KM)
    assert 382.0 < metadata["geometry_summary"]["sun_distance_lu_min"] < 383.0
    assert 382.0 < metadata["geometry_summary"]["sun_distance_lu_max"] < 383.0
    assert metadata["geometry_summary"]["sun_distance_lu_max"] > metadata["geometry_summary"]["sun_distance_lu_min"]
    assert "window mean" in metadata["length_unit_scaling"]["earth_moon_distance_ratio_to_mean"]
    assert metadata["geometry_summary"]["em_distance_ratio_min"] < 1.0
    assert metadata["geometry_summary"]["em_distance_ratio_max"] > 1.0
    assert "not SPICE validation" in " ".join(metadata["interpretation_limits"])

    df = pd.read_csv(output_results / "horizons_ephemeris_force_model_contrast.csv")
    assert len(df) == 15
    assert df["node_index"].tolist() == list(range(15))
    assert df["representative_branch_mask_index"].nunique() == 1
    assert int(df["representative_branch_mask_index"].iloc[0]) == metadata["representative_states"]["branch_mask_index"]
    assert (df["nominal_tidal_accel_delta_norm"] > 0.0).any()
    assert df["sun_distance_lu"].between(382.0, 383.0).all()

    stale_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    stale_cache["metadata"]["node_jd_tdb"] = [
        float(value) + 0.01 for value in stale_cache["metadata"]["node_jd_tdb"]
    ]
    stale_cache_path = tmp_path / "stale_same_node_count_horizons_cache.json"
    stale_cache_path.write_text(json.dumps(stale_cache, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    stale_args = module.build_parser().parse_args(
        [
            "--cache",
            str(stale_cache_path),
            "--results-dir",
            str(tmp_path / "stale_results"),
            "--tables-dir",
            str(tmp_path / "stale_tables"),
        ]
    )
    with pytest.raises(ValueError, match="node_jd_tdb"):
        module.run(stale_args)


def test_real_horizons_ephemeris_force_model_contrast_metadata_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "horizons_ephemeris_force_model_contrast"
        / "horizons_ephemeris_force_model_contrast_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("Horizons ephemeris force-model contrast package has not been generated")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["force_model_contrast_only"] is True
    assert metadata["high_fidelity_validation"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["accepted_control_high_fidelity_replay"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["row_count"] == 15
    assert metadata["geometry_summary"]["reference_distance_km"] == pytest.approx(DEFAULT_REFERENCE_DISTANCE_KM)
    assert 382.0 < metadata["geometry_summary"]["sun_distance_lu_min"] < 383.0
    assert 382.0 < metadata["geometry_summary"]["sun_distance_lu_max"] < 383.0
