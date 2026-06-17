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

from qlt.ephemeris_contrast import (
    DEFAULT_CACHE_START_JD_TDB,
    DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    DEFAULT_REFERENCE_DISTANCE_KM,
    HorizonsVectorSample,
    canonical_node_jds,
    canonical_node_times,
    canonical_point_mass_scales,
    geocentric_inertial_state_to_rotating,
    horizons_point_mass_profile_from_cache,
    propagate_piecewise_controls_horizons_point_mass,
    rotating_state_to_geocentric_inertial,
    sample_to_cache_row,
)


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


def cache_fixture(*, tf: float = 0.08, n_segments: int = 2) -> dict[str, object]:
    node_times = canonical_node_times(tf, n_segments)
    node_jds = canonical_node_jds(
        start_jd_tdb=DEFAULT_CACHE_START_JD_TDB,
        tf=tf,
        n_segments=n_segments,
        canonical_time_unit_seconds=DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    )
    moon = [
        sample(jd, [384400.0, 600.0 * i, 20.0], [0.0, 1.024, 0.0])
        for i, jd in enumerate(node_jds)
    ]
    sun = [
        sample(jd, [147100000.0, 80000.0 * i, 500.0], [0.0, 29.8, 0.0])
        for i, jd in enumerate(node_jds)
    ]
    tlist = "\n".join(f"{float(jd):.9f}" for jd in node_jds)
    body_template = {
        "query_url": "https://ssd.jpl.nasa.gov/api/horizons.api?EPHEM_TYPE=VECTORS",
        "request_params": {"TLIST": tlist},
        "response_signature": {"source": "NASA/JPL Horizons API"},
    }
    return {
        "schema_version": 1,
        "cache_type": "jpl_horizons_geocentric_moon_sun_vectors",
        "metadata": {
            "start_jd_tdb": DEFAULT_CACHE_START_JD_TDB,
            "canonical_transfer_time": tf,
            "segments": n_segments,
            "canonical_node_times": [float(value) for value in node_times],
            "node_jd_tdb": [float(value) for value in node_jds],
            "canonical_time_unit_seconds": DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
            "reference_distance_km": DEFAULT_REFERENCE_DISTANCE_KM,
        },
        "bodies": {
            "moon_geocentric": {**body_template, "samples": [sample_to_cache_row(row) for row in moon]},
            "sun_geocentric": {**body_template, "samples": [sample_to_cache_row(row) for row in sun]},
        },
    }


def minimal_config() -> dict:
    return {
        "run": {
            "label": "ihs_point_mass_source",
            "suite_name": "independent_hs_all_configured_headroom",
            "artifact_stem": "independent_hs_all_configured_headroom",
            "output_subdir": "ihs_point_mass_source",
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
                    "purpose": "unit cached-Horizons point-mass retune",
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


def test_rotating_geocentric_transform_round_trip_on_cache_fixture():
    cache = cache_fixture()
    profile = horizons_point_mass_profile_from_cache(cache)
    moon_sample = sample(
        float(profile.jd_tdb[1]),
        profile.moon_position_km[1].tolist(),
        profile.moon_velocity_km_s[1].tolist(),
    )
    mu = 0.01215058560962404
    scales = canonical_point_mass_scales(mu=mu)
    state = np.asarray([1.02, -0.03, 0.04, 0.02, -0.01, 0.05], dtype=float)

    inertial = rotating_state_to_geocentric_inertial(state, moon_sample, mu=mu, scales=scales)
    recovered = geocentric_inertial_state_to_rotating(inertial, moon_sample, mu=mu, scales=scales)

    assert np.allclose(recovered, state, rtol=0.0, atol=1.0e-12)


def build_retuning_fixture(tmp_path: Path, module):
    config = minimal_config()
    config_path = tmp_path / "ihs_point_mass_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    cache_path = tmp_path / "horizons_cache.json"
    cache = cache_fixture(tf=0.08, n_segments=2)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    results_dir = tmp_path / "data" / "results" / "ihs_point_mass_source"
    controls_dir = results_dir / "controls"
    controls_dir.mkdir(parents=True)
    cfg = module.make_objective_config(config, 0.01215058560962404)
    initial = np.array([1.02, 0.01, 0.0, 0.0, 0.03, 0.0], dtype=float)
    endpoint = np.zeros((2, 3), dtype=float)
    midpoint = np.zeros((2, 3), dtype=float)
    profile = horizons_point_mass_profile_from_cache(cache)
    target, _ = propagate_piecewise_controls_horizons_point_mass(
        initial,
        endpoint,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        profile=profile,
        midpoint_controls=midpoint,
    )
    nominal_path = controls_dir / "ihs_all_single_p04_amax02_polish_from_p04_nominal_controls.json"
    nominal_sha = write_deterministic_json(
        nominal_path,
        {
            "schema_version": 2,
            "case_id": "ihs_all_single_p04_amax02_polish_from_p04",
            "settings_fingerprint": "fixture-fingerprint",
            "nominal_endpoint_controls": endpoint.tolist(),
            "nominal_midpoint_controls": midpoint.tolist(),
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
        branch_endpoint = endpoint * np.asarray(mask, dtype=float)[:, None]
        branch_midpoint = midpoint * np.asarray(mask, dtype=float)[:, None]
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
                "recorded_branch_terminal_error": 0.0,
                "recorded_selected_worst_error": 0.0,
                "recorded_all_mask_worst_error": 0.0,
                "branch_endpoint_controls": branch_endpoint.tolist(),
                "branch_controls": branch_endpoint.tolist(),
                "branch_midpoint_controls": branch_midpoint.tolist(),
            },
        )
        branch_entries.append(
            {
                "branch_order": order,
                "mask_index": order,
                "outage_mask": mask,
                "recovery_start": order + 1,
                "recovery_segments": 2 - (order + 1),
                "recorded_branch_terminal_error": 0.0,
                "path": f"data/results/ihs_point_mass_source/controls/{branch_path.name}",
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
            "nominal_control_path": f"data/results/ihs_point_mass_source/controls/{nominal_path.name}",
            "nominal_control_sha256": nominal_sha,
            "nominal_error": 0.0,
            "selected_worst_error": 0.0,
            "all_mask_worst_error": 0.0,
            "expected_branch_count": 2,
            "branch_count": 2,
            "branch_control_sidecar_count": 2,
            "branch_control_replay_ready": True,
            "selected_outage_indices": [0, 1],
            "selected_outage_errors": [0.0, 0.0],
            "all_outage_errors": [0.0, 0.0],
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
                "nominal_error": 0.0,
                "selected_worst_error": 0.0,
                "all_mask_worst_error": 0.0,
                "nominal_threshold": 10.0,
                "selected_worst_threshold": 10.0,
                "meets_thresholds": True,
                "optimizer_success": True,
                "nfev": 1,
                "branch_control_manifest_path": f"data/results/ihs_point_mass_source/controls/{manifest_path.name}",
                "branch_control_manifest_hash": manifest_sha,
                "branch_control_sidecar_count": 2,
                "branch_control_replay_ready": True,
            }
        ]
    ).to_csv(results_dir / "independent_hs_all_configured_headroom.csv", index=False)
    return config_path, source_states, cache_path, initial, target


def test_point_mass_retuning_script_smoke_and_metadata(monkeypatch, tmp_path):
    module = load_script_module(
        "run_independent_hs_horizons_point_mass_retuning_unit",
        ROOT / "scripts" / "run_independent_hs_horizons_point_mass_retuning.py",
    )
    config_path, source_states, cache_path, initial, target = build_retuning_fixture(tmp_path, module)

    def fake_load_states(root, case_config, source_states_path):
        del root, case_config, source_states_path
        return SimpleNamespace(
            mu=0.01215058560962404,
            initial=initial,
            target=target,
            target_metadata={"target_state_generation": "fixture"},
        )

    output_results = tmp_path / "point_mass_results"
    output_tables = tmp_path / "point_mass_tables"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_configured_states", fake_load_states)
    args = module.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--source-states",
            str(source_states),
            "--cache",
            str(cache_path),
            "--expected-cache-sha256",
            "",
            "--results-dir",
            str(output_results),
            "--tables-dir",
            str(output_tables),
            "--max-nfev",
            "2",
        ]
    )
    df = module.run(args)

    assert len(df) == 3
    metadata = json.loads(
        (output_results / "independent_hs_horizons_point_mass_retuning_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["optimization_rerun"] is True
    assert metadata["retuning"] is True
    assert metadata["uses_cached_horizons_vectors"] is True
    assert metadata["runtime_network_dependency"] is False
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["high_fidelity_flight_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["doi_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    assert metadata["point_mass_retuned_summary"]["retuned_branch_pass_count"] == 2
    assert metadata["point_mass_retuned_summary"]["branch_row_count"] == 2
    assert "retuned_control_manifest_json" in metadata["artifacts"]
    assert (output_tables / "independent_hs_horizons_point_mass_retuning_table.tex").is_file()


def test_real_polished_independent_hs_horizons_point_mass_retuning_package_when_present():
    metadata_path = (
        ROOT
        / "data"
        / "results"
        / "independent_hs_horizons_point_mass_retuning"
        / "independent_hs_horizons_point_mass_retuning_metadata.json"
    )
    csv_path = metadata_path.with_name("independent_hs_horizons_point_mass_retuning.csv")
    if not metadata_path.exists() or not csv_path.exists():
        pytest.skip("independent-HS cached-Horizons point-mass retuning package has not been generated")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["cached_horizons_earth_moon_sun_point_mass_retuning"] is True
    assert metadata["optimization_rerun"] is True
    assert metadata["retuning"] is True
    assert metadata["spice_ephemeris_validation"] is False
    assert metadata["high_fidelity_flight_validation"] is False
    assert metadata["production_solver_parity_claim"] is False
    assert metadata["fuel_optimality_claim"] is False
    assert metadata["doi_claim"] is False
    assert metadata["quantum_advantage_claim"] is False
    summary = metadata["polish_case_summary"]
    assert summary["persisted_nominal_point_mass_replay_error"] > 0.09
    assert summary["persisted_nominal_point_mass_replay_error"] == pytest.approx(0.3812580376880591, abs=5.0e-12)
    assert summary["persisted_branch_replay_pass_count"] == 0
    assert summary["retuned_nominal_point_mass_error"] < 0.09
    assert summary["retuned_branch_point_mass_worst_error"] < 0.17
    assert summary["retuned_branch_pass_count"] == 8
    assert summary["branch_row_count"] == 8
    assert summary["retuned_branch_point_mass_worst_error"] == pytest.approx(0.02473065115224942, abs=5.0e-5)
    assert len(metadata["artifacts"]["retuned_control_sidecars"]) == 9

    df = pd.read_csv(csv_path)
    polish = df[df["case_id"] == "ihs_all_single_p04_amax02_polish_from_p04"]
    branch = polish[polish["record_type"] == "branch"]
    assert len(branch) == 8
    assert not bool(polish[polish["record_type"] == "nominal"]["point_mass_replay_passes_configured_threshold"].iloc[0])
    assert branch["point_mass_retuned_passes_configured_threshold"].map(bool).all()
    assert branch["point_mass_replay_passes_configured_threshold"].map(bool).sum() == 0
