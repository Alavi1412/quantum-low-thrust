"""Independent-midpoint-control Hermite-Simpson continuation evidence.

This runner reuses the Hermite-Simpson continuation harness but configures it
for ``method: hermite_simpson_midpoint`` and a distinct artifact family:
``independent_hs_continuation_baseline``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE_SCRIPT = ROOT / "scripts" / "run_hermite_simpson_continuation.py"
_SPEC = importlib.util.spec_from_file_location("_independent_hs_base", BASE_SCRIPT)
_base = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_base)

SUITE_NAME = "independent_hs_continuation_baseline"
ARTIFACT_STEM = "independent_hs_continuation_baseline"
DEFAULT_CONFIG_PATH = Path("configs/independent_hs_continuation_baseline.yaml")
DEFAULT_COLLOCATION_METHOD = "hermite_simpson_midpoint"
SCRIPT_DESCRIPTION = (
    "Independent-midpoint-control Hermite-Simpson direct-collocation continuation evidence. "
    "Continuous-backend probe; not a quantum/discrete result."
)
METADATA_BACKEND_SEMANTICS = (
    "This is a continuous-backend independent-midpoint-control Hermite-Simpson direct-collocation "
    "evidence package with persisted nominal-control warm starts and trajectory stacking; it is NOT "
    "a quantum, QUBO, QAOA, or discrete schedule-search result."
)
METADATA_METHOD_KEY = "hermite_simpson_midpoint"
METADATA_METHOD_SEMANTICS = (
    "Hermite-Simpson direct transcription with independent midpoint control variables for the nominal "
    "trajectory and selected outage branch recovery arcs. Defect residuals use endpoint/segment controls "
    "for f_left and f_right, independent midpoint controls for f_mid, and reporting propagation uses "
    "quadratic endpoint-midpoint-endpoint control interpolation."
)
METADATA_LIMITATIONS = [
    "Normalized Earth-Moon CR3BP only; not a flight-ready trajectory optimization.",
    "Independent midpoint controls mature the continuous HS baseline but do not make selected-branch rows universal robustness evidence over all outage masks.",
    "Warm starts load persisted nominal endpoint and midpoint controls from the named source row when midpoint controls are available and required; cold starts initialize midpoint controls from endpoint controls.",
    "Fuel for independent-midpoint rows is Simpson-style endpoint-midpoint-endpoint quadrature over control norms; old constant-control HS rows retain rectangle semantics.",
    "Failed/negative diagnostics are preserved honestly and not extrapolated.",
    "The optimizer may stop at max_nfev; such rows are retained with optimizer_success=False.",
]

HS_CONTINUATION_COLUMNS = _base.HS_CONTINUATION_COLUMNS
INDEPENDENT_HS_CONTINUATION_COLUMNS = HS_CONTINUATION_COLUMNS
TARGET_GENERATION = _base.TARGET_GENERATION

run_direct_collocation_baseline = _base.run_direct_collocation_baseline
load_configured_states = _base.load_configured_states
make_objective_config = _base.make_objective_config
output_directories = _base.output_directories
outage_masks = _base.outage_masks
write_json = _base.write_json


def _configure_base() -> None:
    _base.SUITE_NAME = SUITE_NAME
    _base.ARTIFACT_STEM = ARTIFACT_STEM
    _base.DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
    _base.DEFAULT_COLLOCATION_METHOD = DEFAULT_COLLOCATION_METHOD
    _base.FORCE_COLLOCATION_METHOD = DEFAULT_COLLOCATION_METHOD
    _base.SCRIPT_DESCRIPTION = SCRIPT_DESCRIPTION
    _base.METADATA_BACKEND_SEMANTICS = METADATA_BACKEND_SEMANTICS
    _base.METADATA_METHOD_KEY = METADATA_METHOD_KEY
    _base.METADATA_METHOD_SEMANTICS = METADATA_METHOD_SEMANTICS
    _base.METADATA_LIMITATIONS = METADATA_LIMITATIONS
    _base.run_direct_collocation_baseline = run_direct_collocation_baseline
    _base.load_configured_states = load_configured_states
    _base.make_objective_config = make_objective_config
    _base.output_directories = output_directories
    _base.outage_masks = outage_masks
    _base.write_json = write_json


def _force_independent_method(config: dict) -> dict:
    config = dict(config)
    direct = dict(config.get("direct_collocation", {}) or {})
    direct["method"] = DEFAULT_COLLOCATION_METHOD
    config["direct_collocation"] = direct
    return config


def _suite_cases(config: dict) -> list[dict]:
    _configure_base()
    return _base._suite_cases(_force_independent_method(config))


def _case_by_id(cases: list[dict]) -> dict[str, dict]:
    _configure_base()
    return _base._case_by_id(cases)


def _source_case(cases_by_id: dict[str, dict], case: dict) -> dict | None:
    _configure_base()
    return _base._source_case(cases_by_id, case)


def compute_settings_fingerprint(*args, **kwargs) -> str:
    _configure_base()
    if args:
        args = (_force_independent_method(args[0]), *args[1:])
    elif "base_config" in kwargs:
        kwargs["base_config"] = _force_independent_method(kwargs["base_config"])
    return _base.compute_settings_fingerprint(*args, **kwargs)


def write_control_sidecar(*args, **kwargs) -> dict:
    _configure_base()
    return _base.write_control_sidecar(*args, **kwargs)


def load_control_sidecar(*args, **kwargs):
    _configure_base()
    return _base.load_control_sidecar(*args, **kwargs)


def run(args) -> "_base.pd.DataFrame":
    _configure_base()
    return _base.run(args)


def build_parser() -> argparse.ArgumentParser:
    _configure_base()
    return _base.build_parser()


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
