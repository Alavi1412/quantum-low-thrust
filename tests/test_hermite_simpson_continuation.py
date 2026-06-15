"""Fast focused tests for the Hermite-Simpson continuation baseline script.

Covers:
- warm-start controls projection/pass-through in initial_guess
- continuation sidecar write/load round-trip and hash verification
- settings fingerprint sensitivity to source hashes and case parameters
- resume compatibility: stale rows and missing/hash-mismatched sidecars rejected
- _suite_cases / _case_by_id / _source_case helpers
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Load the script module under test
# ---------------------------------------------------------------------------

def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


SCRIPT_PATH = ROOT / "scripts" / "run_hermite_simpson_continuation.py"
hs_script = _load_script("run_hermite_simpson_continuation", SCRIPT_PATH)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _minimal_config(output_subdir: str = "hs_test") -> dict:
    return {
        "run": {"label": "hs_test", "output_subdir": output_subdir},
        "benchmark": {
            "mu": 0.01215058560962404,
            "target_mode": "catalog_halo_phase_shift",
            "transfer_time": 0.5,
            "phase_time": 0.3,
            "segments": 4,
            "substeps_per_segment": 1,
            "amax": 0.3,
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
        "direct_collocation": {
            "method": "hermite_simpson",
            "node_initialization": "blend",
            "node_initialization_blend": 0.35,
            "xtol": 1e-5,
            "ftol": 1e-5,
            "gtol": 1e-5,
            "weights": {
                "initial": 10.0,
                "defect": 1.0,
                "terminal": 5.0,
                "branch_start": 8.0,
                "branch_defect": 1.0,
                "branch_terminal": 5.0,
                "control": 0.01,
                "smooth": 0.012,
            },
        },
        "suite": {
            "runtime_budget_seconds": 600,
            "selected_outages": 1,
            "min_recovery_segments": 1,
            "groups": {
                "phase_group": {
                    "purpose": "unit-test group",
                    "outage_lengths": [1],
                    "cases": [
                        {
                            "case_id": "cold_p03",
                            "phase_time": 0.3,
                            "transfer_time": 0.5,
                            "amax": 0.3,
                            "segments": 4,
                            "selected_outages": 1,
                            "max_nfev": 2,
                            "warm_start_kind": "cold",
                        },
                        {
                            "case_id": "warm_p02_from_p03",
                            "phase_time": 0.2,
                            "transfer_time": 0.5,
                            "amax": 0.3,
                            "segments": 4,
                            "selected_outages": 1,
                            "max_nfev": 2,
                            "warm_start_kind": "nominal_controls",
                            "warm_start_from_case_id": "cold_p03",
                        },
                    ],
                }
            },
        },
    }


def _fake_source_states():
    return SimpleNamespace(
        mu=0.01215058560962404,
        initial=np.zeros(6, dtype=float),
        target=np.ones(6, dtype=float) * 0.01,
        target_metadata={"target_state_generation": "fixture"},
    )


# ---------------------------------------------------------------------------
# 1. _suite_cases produces correct case list
# ---------------------------------------------------------------------------

def test_suite_cases_parses_warm_start_dependencies():
    config = _minimal_config()
    cases = hs_script._suite_cases(config)

    assert len(cases) == 2
    cold = cases[0]
    warm = cases[1]

    assert cold["case_id"] == "cold_p03"
    assert cold["warm_start_kind"] == "cold"
    assert cold["warm_start_from_case_id"] is None
    assert cold["case_order"] == 0

    assert warm["case_id"] == "warm_p02_from_p03"
    assert warm["warm_start_kind"] == "nominal_controls"
    assert warm["warm_start_from_case_id"] == "cold_p03"
    assert warm["case_order"] == 1


def test_suite_cases_rejects_out_of_range_selected_outages():
    config = _minimal_config()
    config["suite"]["groups"]["phase_group"]["cases"][0]["selected_outages"] = 999
    with pytest.raises(ValueError, match="selected_outages"):
        hs_script._suite_cases(config)


def test_case_by_id_detects_duplicates():
    config = _minimal_config()
    cases = hs_script._suite_cases(config)
    by_id = hs_script._case_by_id(cases)
    assert "cold_p03" in by_id
    assert "warm_p02_from_p03" in by_id

    # Force duplicate
    dup_cases = cases + [cases[0]]
    with pytest.raises(ValueError, match="duplicate"):
        hs_script._case_by_id(dup_cases)


def test_source_case_returns_none_for_cold_start():
    config = _minimal_config()
    cases = hs_script._suite_cases(config)
    by_id = hs_script._case_by_id(cases)
    cold = cases[0]
    warm = cases[1]
    assert hs_script._source_case(by_id, cold) is None
    source = hs_script._source_case(by_id, warm)
    assert source is not None
    assert source["case_id"] == "cold_p03"


def test_source_case_raises_on_missing_dependency():
    config = _minimal_config()
    cases = hs_script._suite_cases(config)
    # Build an index that is missing the cold source (cold_p03 is cases[0])
    by_id_without_source: dict[str, dict] = {}  # empty index has neither case
    with pytest.raises(ValueError, match="warm-start source"):
        hs_script._source_case(by_id_without_source, cases[1])


# ---------------------------------------------------------------------------
# 2. Settings fingerprint is sensitive to key parameters
# ---------------------------------------------------------------------------

def test_fingerprint_changes_when_source_control_hash_changes():
    config = _minimal_config()
    source_states = Path("data/source_states.json")
    cases = hs_script._suite_cases(config)
    warm_case = cases[1]
    warm_case["warm_start_from_phase_time"] = 0.3

    fp_a = hs_script.compute_settings_fingerprint(
        config,
        source_states,
        warm_case,
        source_settings_fingerprint="abc123",
        source_control_hash="hash_v1",
    )
    fp_b = hs_script.compute_settings_fingerprint(
        config,
        source_states,
        warm_case,
        source_settings_fingerprint="abc123",
        source_control_hash="hash_v2",
    )
    assert fp_a != fp_b


def test_fingerprint_changes_when_source_settings_fingerprint_changes():
    config = _minimal_config()
    source_states = Path("data/source_states.json")
    cases = hs_script._suite_cases(config)
    warm_case = cases[1]

    fp_a = hs_script.compute_settings_fingerprint(
        config, source_states, warm_case, source_settings_fingerprint="fp_v1", source_control_hash="ctrl_hash"
    )
    fp_b = hs_script.compute_settings_fingerprint(
        config, source_states, warm_case, source_settings_fingerprint="fp_v2", source_control_hash="ctrl_hash"
    )
    assert fp_a != fp_b


def test_fingerprint_changes_when_max_nfev_changes():
    config = _minimal_config()
    source_states = Path("data/source_states.json")
    cases = hs_script._suite_cases(config)
    cold_case = dict(cases[0])

    cold_a = {**cold_case, "max_nfev": 10}
    cold_b = {**cold_case, "max_nfev": 99}

    fp_a = hs_script.compute_settings_fingerprint(config, source_states, cold_a)
    fp_b = hs_script.compute_settings_fingerprint(config, source_states, cold_b)
    assert fp_a != fp_b


def test_fingerprint_changes_when_collocation_method_changes():
    config_hs = _minimal_config()
    config_trap = _minimal_config()
    config_trap["direct_collocation"]["method"] = "trapezoidal"
    source_states = Path("data/source_states.json")
    cases = hs_script._suite_cases(config_hs)
    cold_case = cases[0]

    fp_hs = hs_script.compute_settings_fingerprint(config_hs, source_states, cold_case)
    fp_trap = hs_script.compute_settings_fingerprint(config_trap, source_states, cold_case)
    assert fp_hs != fp_trap


def test_cold_fingerprint_does_not_depend_on_none_source_hashes():
    """Cold case with no source dependency: None vs None must be deterministic."""
    config = _minimal_config()
    source_states = Path("data/source_states.json")
    cases = hs_script._suite_cases(config)
    cold_case = cases[0]

    fp1 = hs_script.compute_settings_fingerprint(config, source_states, cold_case)
    fp2 = hs_script.compute_settings_fingerprint(
        config, source_states, cold_case,
        source_settings_fingerprint=None, source_control_hash=None,
    )
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# 3. Control sidecar write / load round-trip
# ---------------------------------------------------------------------------

def test_control_sidecar_roundtrip(tmp_path):
    controls_dir = tmp_path / "controls"
    controls = np.random.default_rng(42).standard_normal((4, 3)) * 0.1
    fp = "testfp_abc123"
    row = {
        "phase_time": 0.3,
        "transfer_time": 0.5,
        "amax": 0.3,
        "segments": 4,
        "outage_lengths": "[1]",
        "warm_start_kind": "cold",
        "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson",
    }
    sidecar = hs_script.write_control_sidecar(
        controls_dir, case_id="test_case", settings_fingerprint=fp, controls=controls, row=row
    )
    loaded = hs_script.load_control_sidecar(controls_dir, "test_case", fp)
    assert loaded is not None
    loaded_controls, loaded_hash, loaded_path = loaded
    assert np.allclose(loaded_controls, controls)
    assert loaded_hash == sidecar["control_hash"]
    assert loaded_path.exists()


def test_control_sidecar_returns_none_for_missing_file(tmp_path):
    controls_dir = tmp_path / "controls"
    controls_dir.mkdir()
    result = hs_script.load_control_sidecar(controls_dir, "nonexistent_case", "some_fp")
    assert result is None


def test_control_sidecar_returns_none_for_wrong_fingerprint(tmp_path):
    controls_dir = tmp_path / "controls"
    controls = np.ones((4, 3), dtype=float) * 0.05
    row = {
        "phase_time": 0.3, "transfer_time": 0.5, "amax": 0.3, "segments": 4,
        "outage_lengths": "[1]", "warm_start_kind": "cold", "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson",
    }
    hs_script.write_control_sidecar(
        controls_dir, case_id="case_a", settings_fingerprint="fp_correct", controls=controls, row=row
    )
    result = hs_script.load_control_sidecar(controls_dir, "case_a", "fp_wrong")
    assert result is None


def test_control_sidecar_returns_none_for_tampered_hash(tmp_path):
    controls_dir = tmp_path / "controls"
    controls = np.ones((4, 3), dtype=float) * 0.07
    row = {
        "phase_time": 0.3, "transfer_time": 0.5, "amax": 0.3, "segments": 4,
        "outage_lengths": "[1]", "warm_start_kind": "cold", "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson",
    }
    hs_script.write_control_sidecar(
        controls_dir, case_id="tamper_case", settings_fingerprint="fp_ok", controls=controls, row=row
    )
    sidecar_path = controls_dir / "tamper_case_nominal_controls.json"
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    data["control_hash"] = "deadbeef" * 8
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")

    result = hs_script.load_control_sidecar(controls_dir, "tamper_case", "fp_ok")
    assert result is None


def test_control_sidecar_returns_none_for_wrong_case_id(tmp_path):
    controls_dir = tmp_path / "controls"
    controls = np.zeros((3, 3), dtype=float)
    row = {
        "phase_time": 0.2, "transfer_time": 0.5, "amax": 0.2, "segments": 3,
        "outage_lengths": "[1]", "warm_start_kind": "cold", "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson",
    }
    hs_script.write_control_sidecar(
        controls_dir, case_id="real_case", settings_fingerprint="fp", controls=controls, row=row
    )
    # Request with a different case_id but the same path implicitly (case_id mismatch inside file)
    result = hs_script.load_control_sidecar(controls_dir, "other_case", "fp")
    assert result is None


# ---------------------------------------------------------------------------
# 4. Warm-start projection pass-through via initial_guess
# ---------------------------------------------------------------------------

def test_initial_guess_accepts_nominal_control_guess():
    """Verify initial_guess passes the warm-start controls into the decision vector."""
    from qlt.direct_collocation import initial_guess, DirectCollocationLayout
    from qlt.objective import ObjectiveConfig, outage_masks

    cfg = ObjectiveConfig(
        mu=0.01215058560962404,
        tf=0.5,
        n_segments=4,
        substeps=1,
        amax=0.3,
        kr=0.75,
        kv=1.45,
        position_scale=1.0,
        velocity_scale=0.35,
        weights={"nominal": 1.0, "robust_worst": 0.85, "robust_degradation": 0.55, "active_fraction": 0.08, "smoothness": 0.04},
        outage_lengths=(1,),
    )
    state0 = np.zeros(6, dtype=float)
    target = np.ones(6, dtype=float) * 0.01
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)[:1]

    warm_controls = np.ones((cfg.n_segments, 3), dtype=float) * 0.15
    layout_cold, vec_cold = initial_guess(state0, target, cfg, masks)
    layout_warm, vec_warm = initial_guess(state0, target, cfg, masks, nominal_control_guess=warm_controls)

    # Extract nominal controls from the warm-start vector
    nom_controls_warm = vec_warm[layout_warm.nominal_controls].reshape((cfg.n_segments, 3))
    assert np.allclose(nom_controls_warm, warm_controls)

    # Cold start should differ
    nom_controls_cold = vec_cold[layout_cold.nominal_controls].reshape((cfg.n_segments, 3))
    assert not np.allclose(nom_controls_cold, warm_controls)


def test_initial_guess_projects_oversized_warm_controls():
    """Warm controls exceeding amax must be projected to the ball."""
    from qlt.direct_collocation import initial_guess
    from qlt.objective import ObjectiveConfig, outage_masks

    amax = 0.2
    cfg = ObjectiveConfig(
        mu=0.01215058560962404, tf=0.5, n_segments=3, substeps=1, amax=amax,
        kr=0.75, kv=1.45, position_scale=1.0, velocity_scale=0.35,
        weights={"nominal": 1.0, "robust_worst": 0.85, "robust_degradation": 0.55, "active_fraction": 0.08, "smoothness": 0.04},
        outage_lengths=(1,),
    )
    state0 = np.zeros(6, dtype=float)
    target = np.ones(6, dtype=float) * 0.01
    masks = np.zeros((0, cfg.n_segments), dtype=float)

    # Controls that violate the ball constraint
    oversized = np.ones((cfg.n_segments, 3), dtype=float) * 5.0
    layout, vec = initial_guess(state0, target, cfg, masks, nominal_control_guess=oversized)
    nom_controls = vec[layout.nominal_controls].reshape((cfg.n_segments, 3))
    norms = np.linalg.norm(nom_controls, axis=1)
    assert np.all(norms <= amax + 1e-10)


def test_initial_guess_raises_on_wrong_shape():
    """Warm controls with the wrong shape should raise a ValueError."""
    from qlt.direct_collocation import initial_guess
    from qlt.objective import ObjectiveConfig, outage_masks

    cfg = ObjectiveConfig(
        mu=0.01215058560962404, tf=0.5, n_segments=4, substeps=1, amax=0.3,
        kr=0.75, kv=1.45, position_scale=1.0, velocity_scale=0.35,
        weights={"nominal": 1.0, "robust_worst": 0.85, "robust_degradation": 0.55, "active_fraction": 0.08, "smoothness": 0.04},
        outage_lengths=(1,),
    )
    state0 = np.zeros(6, dtype=float)
    target = np.zeros(6, dtype=float)
    masks = np.zeros((0, cfg.n_segments), dtype=float)

    wrong_shape = np.ones((3, 3), dtype=float)  # wrong n_segments
    with pytest.raises(ValueError, match="nominal_control_guess shape"):
        initial_guess(state0, target, cfg, masks, nominal_control_guess=wrong_shape)


# ---------------------------------------------------------------------------
# 5. Resume compatibility: stale and missing-sidecar rows are rejected
# ---------------------------------------------------------------------------

def _make_fixture_row(case_id: str, settings_fp: str, config_hash: str, source_states_id: str, **overrides) -> dict:
    row: dict = {col: None for col in hs_script.HS_CONTINUATION_COLUMNS}
    row.update(
        {
            "case_id": case_id,
            "case_order": 0,
            "case_group": "phase_group",
            "group_purpose": "test",
            "phase_time": 0.3,
            "transfer_time": 0.5,
            "amax": 0.3,
            "segments": 4,
            "outage_lengths": "[1]",
            "selected_outages": 1,
            "outage_count": 4,
            "selected_all_outages": False,
            "warm_start_from_case_id": "",
            "warm_start_from_phase_time": float("nan"),
            "warm_start_kind": "cold",
            "warm_start_source_settings_fingerprint": "",
            "warm_start_source_control_hash": "",
            "max_nfev": 2,
            "min_recovery_segments": 1,
            "node_initialization": "blend",
            "node_initialization_blend": 0.35,
            "method_type": "bounded_projected_constant_control_hermite_simpson_direct_collocation",
            "collocation_method": "hermite_simpson",
            "nominal_error": 0.05,
            "selected_worst_error": 0.08,
            "all_mask_worst_error": 0.09,
            "thresholds": '{"nominal_success": 0.09, "robust_success": 0.17}',
            "nominal_threshold": 0.09,
            "selected_worst_threshold": 0.17,
            "meets_nominal_threshold": True,
            "meets_selected_worst_threshold": True,
            "meets_thresholds": True,
            "optimizer_success": True,
            "direct_collocation_success": True,
            "cost": 0.001,
            "optimality": 1e-5,
            "nfev": 2,
            "runtime_seconds": 0.1,
            "control_max_norm": 0.25,
            "control_bound_violation": 0.0,
            "nominal_fuel": 0.1,
            "recovery_fuel_mean": 0.1,
            "recovery_fuel_max": 0.1,
            "selected_outage_indices": "[0]",
            "selected_outage_errors": "[0.08]",
            "all_outage_errors": "[0.09]",
            "selected_branch_semantics": "fixture",
            "all_mask_diagnostic_semantics": "fixture",
            "control_bound_semantics": "fixture",
            "nominal_control_path": "",
            "nominal_control_hash": "",
            "settings_fingerprint": settings_fp,
            "config_hash": config_hash,
            "source_states_id": source_states_id,
            "message": "fixture ok",
        }
    )
    row.update(overrides)
    return row


def _write_sidecar_for_row(controls_dir: Path, case_id: str, settings_fp: str, segments: int) -> tuple[np.ndarray, str]:
    controls = np.ones((segments, 3), dtype=float) * 0.05
    row = {
        "phase_time": 0.3, "transfer_time": 0.5, "amax": 0.3, "segments": segments,
        "outage_lengths": "[1]", "warm_start_kind": "cold", "warm_start_from_case_id": "",
        "collocation_method": "hermite_simpson",
    }
    sidecar = hs_script.write_control_sidecar(
        controls_dir, case_id=case_id, settings_fingerprint=settings_fp, controls=controls, row=row
    )
    return controls, sidecar["control_hash"]


def test_resume_keeps_compatible_row(tmp_path, monkeypatch):
    config = _minimal_config("hs_resume_test")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    controls_dir = tmp_path / "data" / "results" / "hs_resume_test" / "controls"

    cases = hs_script._suite_cases(config)
    cold_case = cases[0]

    # Compute the correct fingerprint
    monkeypatch.chdir(tmp_path)
    source_states_id = hs_script._file_identity(source_states)
    config_hash_val = hs_script._config_hash(config)
    fp = hs_script.compute_settings_fingerprint(config, source_states, cold_case)

    # Write sidecar
    controls, ctrl_hash = _write_sidecar_for_row(controls_dir, "cold_p03", fp, 4)

    # Build existing df row
    row = _make_fixture_row("cold_p03", fp, config_hash_val, source_states_id)
    df = pd.DataFrame([row], columns=hs_script.HS_CONTINUATION_COLUMNS)

    kept, controls_by_id, rejected = hs_script._compatible_existing_rows(
        df, base_config=config, source_states=source_states, cases=cases, controls_dir=controls_dir
    )
    assert len(kept) == 1
    assert "cold_p03" in controls_by_id
    assert len(rejected) == 0


def test_resume_rejects_stale_fingerprint(tmp_path, monkeypatch):
    config = _minimal_config("hs_stale_test")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    controls_dir = tmp_path / "data" / "results" / "hs_stale_test" / "controls"

    cases = hs_script._suite_cases(config)
    monkeypatch.chdir(tmp_path)
    config_hash_val = hs_script._config_hash(config)
    source_states_id = hs_script._file_identity(source_states)

    stale_fp = "stale_fingerprint_0000"
    row = _make_fixture_row("cold_p03", stale_fp, config_hash_val, source_states_id)
    df = pd.DataFrame([row], columns=hs_script.HS_CONTINUATION_COLUMNS)

    kept, controls_by_id, rejected = hs_script._compatible_existing_rows(
        df, base_config=config, source_states=source_states, cases=cases, controls_dir=controls_dir
    )
    assert len(kept) == 0
    assert len(controls_by_id) == 0
    assert any("settings_fingerprint" in r.get("reason", "") for r in rejected)


def test_resume_rejects_row_with_missing_sidecar(tmp_path, monkeypatch):
    config = _minimal_config("hs_nosidecar_test")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    controls_dir = tmp_path / "data" / "results" / "hs_nosidecar_test" / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    cases = hs_script._suite_cases(config)
    monkeypatch.chdir(tmp_path)
    config_hash_val = hs_script._config_hash(config)
    source_states_id = hs_script._file_identity(source_states)
    fp = hs_script.compute_settings_fingerprint(config, source_states, cases[0])

    # Row has correct fingerprint but no sidecar file
    row = _make_fixture_row("cold_p03", fp, config_hash_val, source_states_id)
    df = pd.DataFrame([row], columns=hs_script.HS_CONTINUATION_COLUMNS)

    kept, controls_by_id, rejected = hs_script._compatible_existing_rows(
        df, base_config=config, source_states=source_states, cases=cases, controls_dir=controls_dir
    )
    assert len(kept) == 0
    assert any("sidecar" in r.get("reason", "") for r in rejected)


def test_resume_rejects_warm_row_if_source_not_yet_kept(tmp_path, monkeypatch):
    """A warm-start row must be rejected if its source case is not in kept rows."""
    config = _minimal_config("hs_no_source_test")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")
    controls_dir = tmp_path / "data" / "results" / "hs_no_source_test" / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    cases = hs_script._suite_cases(config)
    warm_case = cases[1]
    monkeypatch.chdir(tmp_path)
    config_hash_val = hs_script._config_hash(config)
    source_states_id = hs_script._file_identity(source_states)

    # Only include the warm row (no cold source) in the existing df
    stale_warm_fp = "warm_stale_0000"
    row = _make_fixture_row(
        "warm_p02_from_p03",
        stale_warm_fp,
        config_hash_val,
        source_states_id,
        case_order=1,
        warm_start_kind="nominal_controls",
        warm_start_from_case_id="cold_p03",
    )
    df = pd.DataFrame([row], columns=hs_script.HS_CONTINUATION_COLUMNS)

    kept, controls_by_id, rejected = hs_script._compatible_existing_rows(
        df, base_config=config, source_states=source_states, cases=cases, controls_dir=controls_dir
    )
    assert "warm_p02_from_p03" not in controls_by_id
    assert any("source" in r.get("reason", "").lower() for r in rejected)


# ---------------------------------------------------------------------------
# 6. End-to-end fake-backend run via monkeypatch (2 mini cases)
# ---------------------------------------------------------------------------

def test_run_produces_csv_and_metadata(monkeypatch, tmp_path):
    """End-to-end smoke test: fake backend, 2 cases, check CSV + metadata."""
    import qlt.direct_collocation as dc_module

    config = _minimal_config("hs_e2e_test")
    config_path = tmp_path / "hs_config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_states = tmp_path / "source_states.json"
    source_states.write_text("{}", encoding="utf-8")

    call_log: list[dict] = []

    def fake_run_direct_collocation(
        *,
        state0,
        target,
        cfg,
        masks,
        thresholds,
        selected_outages,
        max_nfev,
        min_recovery_segments,
        collocation_config,
        nominal_control_guess,
        selected_branch_control_guesses,
        warm_start_info,
    ):
        call_log.append({
            "warm_start_enabled": warm_start_info.get("enabled"),
            "nominal_control_guess_is_none": nominal_control_guess is None,
            "collocation_method": collocation_config.get("method", "hermite_simpson"),
        })
        n = cfg.n_segments
        if nominal_control_guess is not None:
            nom = np.asarray(nominal_control_guess, dtype=float).copy()
        else:
            nom = np.zeros((n, 3), dtype=float)
        return {
            "success": True,
            "nominal_controls": nom,
            "selected_branch_controls": [],
            "nominal_history": None,
            "nominal_fuel": 0.1,
            "recovery_fuel_mean": 0.1,
            "recovery_fuel_max": 0.1,
            "all_mask_recovery_fuel_mean": 0.1,
            "nominal_error": 0.04,
            "selected_worst_error": 0.06,
            "all_mask_worst_error": 0.08,
            "selected_outage_errors": [0.06],
            "all_outage_errors": [0.07, 0.08],
            "control_max_norm": 0.1,
            "control_bound_violation": 0.0,
            "method_type": "bounded_projected_constant_control_hermite_simpson_direct_collocation",
            "collocation_method": "hermite_simpson",
            "collocation_scheme_semantics": "fixture",
            "selected_branch_semantics": "fixture selected",
            "all_mask_diagnostic_semantics": "fixture all mask",
            "control_bound_semantics": "fixture bound",
            "optimizer_success": True,
            "message": "fixture ok",
            "cost": 0.001,
            "optimality": 1e-6,
            "nfev": 2,
            "runtime_seconds": 0.01,
            "selected_outage_indices": [0],
            "weights": {},
            "warm_start_info": warm_start_info,
        }

    def fake_load_states(root, case_config, source_states_path):
        return _fake_source_states()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hs_script, "run_direct_collocation_baseline", fake_run_direct_collocation)
    monkeypatch.setattr(hs_script, "load_configured_states", fake_load_states)

    args = hs_script.build_parser().parse_args(
        ["--config", str(config_path), "--source-states", str(source_states)]
    )
    df = hs_script.run(args)

    results_dir = tmp_path / "data" / "results" / "hs_e2e_test"
    csv_path = results_dir / "hermite_simpson_continuation_baseline.csv"
    meta_path = results_dir / "hermite_simpson_continuation_baseline_metadata.json"

    assert csv_path.exists()
    assert meta_path.exists()
    assert len(df) == 2

    # Verify call log: first call is cold (no warm-start), second has warm-start
    assert len(call_log) == 2
    assert call_log[0]["nominal_control_guess_is_none"] is True
    assert call_log[0]["warm_start_enabled"] is False
    assert call_log[1]["nominal_control_guess_is_none"] is False
    assert call_log[1]["warm_start_enabled"] is True
    assert call_log[1]["collocation_method"] == "hermite_simpson"

    # Check CSV columns
    csv_df = pd.read_csv(csv_path)
    assert set(hs_script.HS_CONTINUATION_COLUMNS).issubset(csv_df.columns)
    assert "method_type" in csv_df.columns
    assert "collocation_method" in csv_df.columns
    assert "direct_collocation_success" in csv_df.columns

    # Metadata
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["row_count"] == 2
    assert "Hermite-Simpson" in meta["semantics"]["backend"]
    assert "NOT a quantum" in meta["semantics"]["backend"]

    # Sidecars
    controls_dir = results_dir / "controls"
    assert (controls_dir / "cold_p03_nominal_controls.json").exists()
    assert (controls_dir / "warm_p02_from_p03_nominal_controls.json").exists()

    # Tables and figures
    tables_dir = tmp_path / "tables" / "hs_e2e_test"
    figures_dir = tmp_path / "figures" / "hs_e2e_test"
    assert (tables_dir / "hermite_simpson_continuation_baseline_table.tex").exists()
    assert (figures_dir / "hermite_simpson_continuation_baseline.png").exists()
    assert (figures_dir / "hermite_simpson_continuation_baseline.pdf").exists()
