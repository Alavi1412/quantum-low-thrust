"""Run a scoped CasADi/IPOPT bridge check for the independent-HS polish case.

This experiment exports the accepted independent-midpoint Hermite-Simpson
endpoint-plus-midpoint control sidecars for
``ihs_all_single_p04_amax02_polish_from_p04`` to a small CasADi direct-shooting
NLP. It refines active controls only under the same normalized CR3BP target,
scales, thresholds, and branch-mask semantics used by the source artifacts.

The output is intentionally narrow: it is a scoped CasADi/IPOPT mature NLP
backend bridge check, not production flight validation, high-fidelity
propagation, global or fuel optimality evidence, DOI evidence, or
quantum-advantage evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_independent_hs_continuation as ihs_runner
from qlt.direct_collocation import propagate_piecewise_controls
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.refinement import project_controls_to_ball
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "independent_hs_all_configured_headroom.yaml"
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_casadi_ipopt_bridge")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_casadi_ipopt_bridge")
POLISH_CASE_ID = "ihs_all_single_p04_amax02_polish_from_p04"
BRIDGE_CSV_NAME = "independent_hs_casadi_ipopt_bridge.csv"
BRIDGE_METADATA_NAME = "independent_hs_casadi_ipopt_bridge_metadata.json"
BRIDGE_TABLE_NAME = "independent_hs_casadi_ipopt_bridge_table.tex"
BRIDGE_CONTROL_MANIFEST_NAME = "independent_hs_casadi_ipopt_bridge_control_manifest.json"

BRIDGE_COLUMNS = [
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "source_controls_path",
    "source_controls_sha256",
    "casadi_refined_controls_path",
    "casadi_refined_controls_sha256",
    "active_control_mask",
    "recovery_start",
    "variable_control_mask",
    "max_prefix_control_delta_from_refined_nominal_masked",
    "source_recorded_cr3bp_terminal_error",
    "source_replay_terminal_error",
    "ipopt_refined_terminal_error",
    "terminal_error_delta_from_source_replay",
    "terminal_error_improvement_from_source_replay",
    "configured_threshold",
    "source_replay_passes_configured_threshold",
    "ipopt_refined_passes_configured_threshold",
    "bridge_passes_configured_threshold",
    "casadi_refinement",
    "optimization_rerun",
    "production_solver_parity_claim",
    "high_fidelity_validation",
    "high_fidelity_flight_validation",
    "fuel_optimality_claim",
    "doi_claim",
    "quantum_advantage_claim",
    "ipopt_success",
    "ipopt_status",
    "ipopt_iterations",
    "ipopt_objective",
    "runtime_seconds",
    "control_bound",
    "refined_endpoint_norm_max",
    "refined_midpoint_norm_max",
    "max_control_norm",
    "max_control_bound_violation",
    "substeps_per_segment",
    "transfer_time",
    "bridge_semantics",
]

_CSV_FIELD_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_FIELD_LIMIT)
        break
    except OverflowError:
        _CSV_FIELD_LIMIT //= 10


def _json_bytes(data: object) -> bytes:
    text = json.dumps(
        sanitize_json(data),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(data))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    for base in (Path.cwd(), ROOT):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _resolve_existing_path(value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("expected artifact path, got blank")
    path = Path(text)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _read_json_verified(path_text: object, expected_sha256: object | None = None) -> tuple[dict[str, Any], Path, str]:
    path = _resolve_existing_path(path_text)
    if not path.is_file():
        raise RuntimeError(f"sidecar not found: {path}")
    actual = _sha256(path)
    if expected_sha256 not in (None, "") and str(expected_sha256).strip() != actual:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return json.loads(path.read_text(encoding="utf-8")), path, actual


def _read_input_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"independent-HS continuation CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _case_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in ihs_runner._suite_cases(config)}


def _case_config(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    ihs_runner._configure_base(config)
    return ihs_runner._base._case_config(config, case)


def _append_artifact(items: list[dict[str, object]], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    items.append({"path": _relative_or_absolute(path), "sha256": _sha256(path), "bytes": path.stat().st_size})


def _require_casadi():
    try:
        return importlib.import_module("casadi")
    except ImportError as exc:
        raise RuntimeError(
            "CasADi is required for the independent-HS CasADi/IPOPT bridge. "
            "Install casadi==3.7.2 or rerun in the verified Python 3.11 environment."
        ) from exc


def _terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray,
) -> float:
    final, _ = propagate_piecewise_controls(
        state0,
        endpoint_controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=midpoint_controls,
    )
    return float(state_error(final, target, cfg.position_scale, cfg.velocity_scale))


def _scaled_residual(final_state: np.ndarray, target: np.ndarray, *, position_scale: float, velocity_scale: float) -> np.ndarray:
    scale = np.asarray(
        [position_scale, position_scale, position_scale, velocity_scale, velocity_scale, velocity_scale],
        dtype=float,
    )
    return (np.asarray(final_state, dtype=float) - np.asarray(target, dtype=float)) / np.maximum(scale, 1.0e-12)


def _control_norm_diagnostics(
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray,
    *,
    amax: float,
) -> dict[str, float]:
    endpoint_norms = np.linalg.norm(np.asarray(endpoint_controls, dtype=float), axis=1)
    midpoint_norms = np.linalg.norm(np.asarray(midpoint_controls, dtype=float), axis=1)
    endpoint_max = float(endpoint_norms.max()) if endpoint_norms.size else 0.0
    midpoint_max = float(midpoint_norms.max()) if midpoint_norms.size else 0.0
    max_norm = max(endpoint_max, midpoint_max)
    violation = max(0.0, max_norm - float(amax))
    if violation <= 1.0e-10:
        violation = 0.0
        max_norm = min(max_norm, float(amax))
        endpoint_max = min(endpoint_max, float(amax))
        midpoint_max = min(midpoint_max, float(amax))
    return {
        "endpoint_norm_max": endpoint_max,
        "midpoint_norm_max": midpoint_max,
        "max_control_norm": max_norm,
        "max_control_bound_violation": violation,
    }


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def _infer_recovery_start(outage_mask: list[int]) -> int:
    mask = np.asarray(outage_mask, dtype=int)
    zero_indices = np.flatnonzero(mask == 0)
    if zero_indices.size == 0:
        return 0
    return int(zero_indices[-1]) + 1


def _branch_recovery_start(
    *,
    branch: dict[str, Any],
    entry: dict[str, Any],
    outage_mask: list[int],
    n_segments: int,
) -> int:
    explicit = _int_or_none(branch.get("recovery_start"))
    if explicit is None:
        explicit = _int_or_none(entry.get("recovery_start"))
    recovery_start = _infer_recovery_start(outage_mask) if explicit is None else explicit
    if recovery_start < 0 or recovery_start > int(n_segments):
        raise RuntimeError(f"recovery_start {recovery_start} is outside [0, {int(n_segments)}]")
    return int(recovery_start)


def _branch_variable_control_mask(outage_mask: list[int], recovery_start: int) -> np.ndarray:
    active = np.asarray(outage_mask, dtype=int)
    variable = np.zeros_like(active, dtype=int)
    variable[int(recovery_start) :] = active[int(recovery_start) :]
    return variable


def _prefix_invariant_diagnostics(
    *,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray,
    nominal_endpoint_controls: np.ndarray | None,
    nominal_midpoint_controls: np.ndarray | None,
    outage_mask: list[int] | None,
    recovery_start: int | None,
) -> dict[str, float]:
    if (
        nominal_endpoint_controls is None
        or nominal_midpoint_controls is None
        or outage_mask is None
        or recovery_start is None
        or int(recovery_start) <= 0
    ):
        return {
            "max_prefix_endpoint_delta_from_refined_nominal_masked": 0.0,
            "max_prefix_midpoint_delta_from_refined_nominal_masked": 0.0,
            "max_prefix_control_delta_from_refined_nominal_masked": 0.0,
        }
    prefix_end = int(recovery_start)
    mask = np.asarray(outage_mask, dtype=float)[:prefix_end, None]
    expected_endpoint = np.asarray(nominal_endpoint_controls, dtype=float)[:prefix_end] * mask
    expected_midpoint = np.asarray(nominal_midpoint_controls, dtype=float)[:prefix_end] * mask
    endpoint_delta = np.asarray(endpoint_controls, dtype=float)[:prefix_end] - expected_endpoint
    midpoint_delta = np.asarray(midpoint_controls, dtype=float)[:prefix_end] - expected_midpoint
    endpoint_max = float(np.max(np.abs(endpoint_delta))) if endpoint_delta.size else 0.0
    midpoint_max = float(np.max(np.abs(midpoint_delta))) if midpoint_delta.size else 0.0
    return {
        "max_prefix_endpoint_delta_from_refined_nominal_masked": endpoint_max,
        "max_prefix_midpoint_delta_from_refined_nominal_masked": midpoint_max,
        "max_prefix_control_delta_from_refined_nominal_masked": max(endpoint_max, midpoint_max),
    }


def _casadi_cr3bp_derivative(ca, state, mu: float, accel):
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    r1 = ca.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = ca.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3 + accel[0]
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1**3 - mu * y / r2**3 + accel[1]
    az = -(1.0 - mu) * z / r1**3 - mu * z / r2**3 + accel[2]
    return ca.vertcat(vx, vy, vz, ax, ay, az)


def _casadi_quadratic_control(ca, endpoint_control, midpoint_control, tau: float):
    return endpoint_control + 4.0 * float(tau) * (1.0 - float(tau)) * (midpoint_control - endpoint_control)


def _casadi_rk4_step_quadratic_control(
    ca,
    state,
    *,
    mu: float,
    dt: float,
    endpoint_control,
    midpoint_control,
    tau0: float,
    tau1: float,
):
    tau_mid = 0.5 * (float(tau0) + float(tau1))
    u1 = _casadi_quadratic_control(ca, endpoint_control, midpoint_control, tau0)
    u2 = _casadi_quadratic_control(ca, endpoint_control, midpoint_control, tau_mid)
    u4 = _casadi_quadratic_control(ca, endpoint_control, midpoint_control, tau1)
    k1 = _casadi_cr3bp_derivative(ca, state, mu, u1)
    k2 = _casadi_cr3bp_derivative(ca, state + 0.5 * dt * k1, mu, u2)
    k3 = _casadi_cr3bp_derivative(ca, state + 0.5 * dt * k2, mu, u2)
    k4 = _casadi_cr3bp_derivative(ca, state + dt * k3, mu, u4)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _casadi_propagate(ca, state0: np.ndarray, endpoint_controls: list, midpoint_controls: list, *, mu: float, tf: float, substeps: int):
    state = ca.DM(np.asarray(state0, dtype=float).reshape((6, 1)))
    n_segments = len(endpoint_controls)
    h = float(tf) / float(n_segments)
    dt = h / float(substeps)
    for segment_index in range(n_segments):
        endpoint = endpoint_controls[segment_index]
        midpoint = midpoint_controls[segment_index]
        for step_index in range(int(substeps)):
            tau0 = step_index / float(substeps)
            tau1 = (step_index + 1) / float(substeps)
            state = _casadi_rk4_step_quadratic_control(
                ca,
                state,
                mu=float(mu),
                dt=dt,
                endpoint_control=endpoint,
                midpoint_control=midpoint,
                tau0=tau0,
                tau1=tau1,
            )
    return state


def _solve_casadi_ipopt_refinement(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_seed: np.ndarray,
    midpoint_seed: np.ndarray,
    variable_control_mask: np.ndarray,
    fixed_endpoint_controls: np.ndarray | None = None,
    fixed_midpoint_controls: np.ndarray | None = None,
    args: argparse.Namespace,
) -> dict[str, object]:
    ca = _require_casadi()
    amax = float(cfg.amax)
    endpoint_seed = project_controls_to_ball(np.asarray(endpoint_seed, dtype=float), amax)
    midpoint_seed = project_controls_to_ball(np.asarray(midpoint_seed, dtype=float), amax)
    variable = np.asarray(variable_control_mask, dtype=int) > 0
    fixed_endpoint = (
        np.zeros_like(endpoint_seed)
        if fixed_endpoint_controls is None
        else project_controls_to_ball(np.asarray(fixed_endpoint_controls, dtype=float), amax)
    )
    fixed_midpoint = (
        np.zeros_like(midpoint_seed)
        if fixed_midpoint_controls is None
        else project_controls_to_ball(np.asarray(fixed_midpoint_controls, dtype=float), amax)
    )
    if endpoint_seed.shape != (int(cfg.n_segments), 3):
        raise ValueError(f"endpoint seed shape {endpoint_seed.shape} does not match {(int(cfg.n_segments), 3)}")
    if midpoint_seed.shape != endpoint_seed.shape:
        raise ValueError(f"midpoint seed shape {midpoint_seed.shape} does not match {endpoint_seed.shape}")
    if fixed_endpoint.shape != endpoint_seed.shape:
        raise ValueError(f"fixed endpoint shape {fixed_endpoint.shape} does not match {endpoint_seed.shape}")
    if fixed_midpoint.shape != midpoint_seed.shape:
        raise ValueError(f"fixed midpoint shape {fixed_midpoint.shape} does not match {midpoint_seed.shape}")
    if variable.shape != (int(cfg.n_segments),):
        raise ValueError(f"variable mask shape {variable.shape} does not match {(int(cfg.n_segments),)}")
    if not np.any(variable):
        endpoint = fixed_endpoint.copy()
        midpoint = fixed_midpoint.copy()
        terminal_error = _terminal_error(
            state0=np.asarray(state0, dtype=float),
            target=np.asarray(target, dtype=float),
            cfg=cfg,
            endpoint_controls=endpoint,
            midpoint_controls=midpoint,
        )
        final, _ = propagate_piecewise_controls(
            state0,
            endpoint,
            cfg.mu,
            cfg.tf,
            cfg.substeps,
            midpoint_controls=midpoint,
        )
        residual = _scaled_residual(
            final,
            target,
            position_scale=float(cfg.position_scale),
            velocity_scale=float(cfg.velocity_scale),
        )
        return {
            "endpoint_controls": endpoint,
            "midpoint_controls": midpoint,
            "final_state": final,
            "terminal_error": terminal_error,
            "ipopt_success": True,
            "ipopt_status": "No_Variable_Fixed_Schedule",
            "ipopt_iterations": 0,
            "ipopt_objective": float(args.terminal_residual_weight) * float(np.sum(residual * residual)),
            "runtime_seconds": 0.0,
            "casadi_version": str(ca.__version__),
        }

    variable_indices = [int(value) for value in np.flatnonzero(variable)]
    variable_count = len(variable_indices)
    endpoint_var_count = variable_count * 3
    x0 = np.concatenate((endpoint_seed[variable].reshape(-1), midpoint_seed[variable].reshape(-1)))
    x = ca.MX.sym("u_active", int(x0.size))

    endpoint_controls = []
    midpoint_controls = []
    endpoint_by_index = {}
    midpoint_by_index = {}
    for rank, segment_index in enumerate(variable_indices):
        endpoint_by_index[segment_index] = x[3 * rank : 3 * rank + 3]
        start = endpoint_var_count + 3 * rank
        midpoint_by_index[segment_index] = x[start : start + 3]
    for segment_index in range(int(cfg.n_segments)):
        if segment_index in endpoint_by_index:
            endpoint_controls.append(endpoint_by_index[segment_index])
            midpoint_controls.append(midpoint_by_index[segment_index])
        else:
            endpoint_controls.append(ca.DM(fixed_endpoint[segment_index].reshape((3, 1))))
            midpoint_controls.append(ca.DM(fixed_midpoint[segment_index].reshape((3, 1))))

    final_state = _casadi_propagate(
        ca,
        state0,
        endpoint_controls,
        midpoint_controls,
        mu=float(cfg.mu),
        tf=float(cfg.tf),
        substeps=int(cfg.substeps),
    )
    scale = ca.DM(
        np.asarray(
            [cfg.position_scale, cfg.position_scale, cfg.position_scale, cfg.velocity_scale, cfg.velocity_scale, cfg.velocity_scale],
            dtype=float,
        ).reshape((6, 1))
    )
    residual = (final_state - ca.DM(np.asarray(target, dtype=float).reshape((6, 1)))) / ca.fmax(scale, 1.0e-12)
    objective = float(args.terminal_residual_weight) * ca.sumsqr(residual)
    scaled_amax = max(amax, 1.0e-12)
    for segment_index in variable_indices:
        endpoint = endpoint_by_index[segment_index]
        midpoint = midpoint_by_index[segment_index]
        endpoint_ref = ca.DM(endpoint_seed[segment_index].reshape((3, 1)))
        midpoint_ref = ca.DM(midpoint_seed[segment_index].reshape((3, 1)))
        objective += float(args.seed_deviation_regularization_weight) * ca.sumsqr((endpoint - endpoint_ref) / scaled_amax)
        objective += float(args.seed_deviation_regularization_weight) * ca.sumsqr((midpoint - midpoint_ref) / scaled_amax)
        objective += float(args.control_regularization_weight) * ca.sumsqr(endpoint / scaled_amax)
        objective += float(args.control_regularization_weight) * ca.sumsqr(midpoint / scaled_amax)
    for left, right in zip(variable_indices[:-1], variable_indices[1:]):
        objective += float(args.smooth_regularization_weight) * ca.sumsqr(
            (endpoint_by_index[right] - endpoint_by_index[left]) / scaled_amax
        )
        objective += float(args.smooth_regularization_weight) * ca.sumsqr(
            (midpoint_by_index[right] - midpoint_by_index[left]) / scaled_amax
        )

    constraints = []
    for segment_index in variable_indices:
        constraints.append(ca.sumsqr(endpoint_by_index[segment_index]))
        constraints.append(ca.sumsqr(midpoint_by_index[segment_index]))
    nlp = {"x": x, "f": objective, "g": ca.vertcat(*constraints)}
    options = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": int(args.max_iter),
        "ipopt.tol": float(args.tol),
        "ipopt.acceptable_tol": float(args.acceptable_tol),
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
    }
    solver = ca.nlpsol("independent_hs_casadi_ipopt_bridge_solver", "ipopt", nlp, options)
    lower = np.full_like(x0, -amax, dtype=float)
    upper = np.full_like(x0, amax, dtype=float)
    lbg = np.zeros(len(constraints), dtype=float)
    ubg = np.full(len(constraints), amax * amax, dtype=float)

    start = time.perf_counter()
    try:
        solution = solver(x0=x0, lbx=lower, ubx=upper, lbg=lbg, ubg=ubg)
        runtime_seconds = float(time.perf_counter() - start)
        stats = solver.stats()
        solved = np.asarray(solution["x"], dtype=float).reshape(-1)
        ipopt_objective = float(solution["f"])
        status = str(stats.get("return_status", "unknown"))
        success = bool(stats.get("success", False))
        iterations = int(stats.get("iter_count", -1))
    except Exception as exc:  # pragma: no cover - exercised only on solver runtime failure
        runtime_seconds = float(time.perf_counter() - start)
        solved = x0.copy()
        ipopt_objective = float("nan")
        status = f"exception: {exc}"
        success = False
        iterations = -1

    endpoint = fixed_endpoint.copy()
    midpoint = fixed_midpoint.copy()
    endpoint[variable] = project_controls_to_ball(solved[:endpoint_var_count].reshape((-1, 3)), amax)
    midpoint[variable] = project_controls_to_ball(solved[endpoint_var_count:].reshape((-1, 3)), amax)
    terminal_error = _terminal_error(
        state0=np.asarray(state0, dtype=float),
        target=np.asarray(target, dtype=float),
        cfg=cfg,
        endpoint_controls=endpoint,
        midpoint_controls=midpoint,
    )
    final, _ = propagate_piecewise_controls(
        state0,
        endpoint,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=midpoint,
    )
    return {
        "endpoint_controls": endpoint,
        "midpoint_controls": midpoint,
        "final_state": final,
        "terminal_error": terminal_error,
        "ipopt_success": success,
        "ipopt_status": status,
        "ipopt_iterations": iterations,
        "ipopt_objective": ipopt_objective,
        "runtime_seconds": runtime_seconds,
        "casadi_version": str(ca.__version__),
    }


def _write_refined_sidecar(
    *,
    controls_dir: Path,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    active_mask: np.ndarray,
    recovery_start: int | None,
    variable_control_mask: np.ndarray,
    prefix_invariant_diagnostics: dict[str, float],
    source_controls_path: Path,
    source_controls_sha256: str,
    source_recorded_cr3bp_terminal_error: float,
    source_replay_terminal_error: float,
    configured_threshold: float,
    refine_result: dict[str, object],
    cfg,
) -> tuple[Path, str]:
    if record_type == "nominal":
        filename = f"{case_id}_nominal_casadi_ipopt_refined_controls.json"
    else:
        filename = f"{case_id}_branch_{int(branch_order):03d}_mask_{int(mask_index):03d}_casadi_ipopt_refined_controls.json"
    path = controls_dir / filename
    endpoint = np.asarray(refine_result["endpoint_controls"], dtype=float)
    midpoint = np.asarray(refine_result["midpoint_controls"], dtype=float)
    diagnostics = _control_norm_diagnostics(endpoint, midpoint, amax=float(cfg.amax))
    refined_error = float(refine_result["terminal_error"])
    limitations = [
        "Scoped normalized-CR3BP CasADi/IPOPT direct-shooting bridge for one accepted independent-HS case.",
        "Branch variables are restricted to post-recovery active endpoint-plus-midpoint controls.",
        "Pre-recovery branch prefixes are fixed to the refined nominal controls with the outage mask applied.",
        "Outage-masked branch segments are fixed inactive/zero.",
        "This is a local mature-backend bridge check, not global optimality, fuel optimality, production flight validation, DOI evidence, or quantum advantage evidence.",
        "The force model remains the original normalized CR3BP target and scales; this is not high-fidelity or flight validation.",
    ]
    payload = {
        "schema_version": 1,
        "sidecar_type": "independent_hs_casadi_ipopt_bridge_refined_controls",
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else [int(value) for value in outage_mask],
        "active_control_mask": [int(value) for value in np.asarray(active_mask, dtype=int).tolist()],
        "recovery_start": "" if recovery_start is None else int(recovery_start),
        "variable_control_mask": [int(value) for value in np.asarray(variable_control_mask, dtype=int).tolist()],
        "max_prefix_control_delta_from_refined_nominal_masked": float(
            prefix_invariant_diagnostics["max_prefix_control_delta_from_refined_nominal_masked"]
        ),
        "prefix_invariant_diagnostics": {
            key: float(value) for key, value in prefix_invariant_diagnostics.items()
        },
        "source_controls_path": _relative_or_absolute(source_controls_path),
        "source_controls_sha256": source_controls_sha256,
        "source_recorded_cr3bp_terminal_error": float(source_recorded_cr3bp_terminal_error),
        "source_replay_terminal_error": float(source_replay_terminal_error),
        "ipopt_refined_terminal_error": refined_error,
        "configured_threshold": float(configured_threshold),
        "ipopt_refined_passes_configured_threshold": bool(refined_error <= float(configured_threshold)),
        "amax": float(cfg.amax),
        "substeps_per_segment": int(cfg.substeps),
        "transfer_time": float(cfg.tf),
        "casadi_refinement": True,
        "optimization_rerun": True,
        "production_solver_parity_claim": True,
        "production_solver_parity_claim_scope": (
            "Scoped CasADi/IPOPT mature NLP backend bridge check for one normalized-CR3BP accepted case only."
        ),
        "high_fidelity_validation": False,
        "high_fidelity_flight_validation": False,
        "fuel_optimality_claim": False,
        "doi_claim": False,
        "quantum_advantage_claim": False,
        "casadi_version": str(refine_result["casadi_version"]),
        "ipopt_success": bool(refine_result["ipopt_success"]),
        "ipopt_status": str(refine_result["ipopt_status"]),
        "ipopt_iterations": int(refine_result["ipopt_iterations"]),
        "ipopt_objective": float(refine_result["ipopt_objective"]),
        "runtime_seconds": float(refine_result["runtime_seconds"]),
        "refined_endpoint_controls": endpoint.tolist(),
        "refined_midpoint_controls": midpoint.tolist(),
        "final_state": np.asarray(refine_result["final_state"], dtype=float).tolist(),
        "control_norm_diagnostics": diagnostics,
        "bridge_semantics": (
            "CasADi/IPOPT local direct-shooting refinement of endpoint-plus-midpoint controls with "
            "Euclidean norm constraints and the original normalized CR3BP target/scales. Branch rows "
            "fix the pre-recovery prefix to the refined nominal masked controls and expose only "
            "post-recovery active controls as IPOPT variables."
        ),
        "limitations": limitations,
    }
    _write_json(path, payload)
    return path, _sha256(path)


def _row_for_result(
    *,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    source_controls_path: Path,
    source_controls_sha256: str,
    refined_controls_path: Path,
    refined_controls_sha256: str,
    active_mask: np.ndarray,
    recovery_start: int | None,
    variable_control_mask: np.ndarray,
    prefix_invariant_diagnostics: dict[str, float],
    source_recorded_cr3bp_terminal_error: float,
    source_replay_terminal_error: float,
    configured_threshold: float,
    refine_result: dict[str, object],
    cfg,
) -> dict[str, object]:
    endpoint = np.asarray(refine_result["endpoint_controls"], dtype=float)
    midpoint = np.asarray(refine_result["midpoint_controls"], dtype=float)
    diagnostics = _control_norm_diagnostics(endpoint, midpoint, amax=float(cfg.amax))
    refined_error = float(refine_result["terminal_error"])
    delta = refined_error - float(source_replay_terminal_error)
    improvement = float(source_replay_terminal_error) - refined_error
    pass_threshold = bool(refined_error <= float(configured_threshold))
    return {
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
        "source_controls_path": _relative_or_absolute(source_controls_path),
        "source_controls_sha256": source_controls_sha256,
        "casadi_refined_controls_path": _relative_or_absolute(refined_controls_path),
        "casadi_refined_controls_sha256": refined_controls_sha256,
        "active_control_mask": json.dumps([int(value) for value in np.asarray(active_mask, dtype=int).tolist()]),
        "recovery_start": "" if recovery_start is None else int(recovery_start),
        "variable_control_mask": json.dumps(
            [int(value) for value in np.asarray(variable_control_mask, dtype=int).tolist()]
        ),
        "max_prefix_control_delta_from_refined_nominal_masked": float(
            prefix_invariant_diagnostics["max_prefix_control_delta_from_refined_nominal_masked"]
        ),
        "source_recorded_cr3bp_terminal_error": float(source_recorded_cr3bp_terminal_error),
        "source_replay_terminal_error": float(source_replay_terminal_error),
        "ipopt_refined_terminal_error": refined_error,
        "terminal_error_delta_from_source_replay": delta,
        "terminal_error_improvement_from_source_replay": improvement,
        "configured_threshold": float(configured_threshold),
        "source_replay_passes_configured_threshold": bool(float(source_replay_terminal_error) <= float(configured_threshold)),
        "ipopt_refined_passes_configured_threshold": pass_threshold,
        "bridge_passes_configured_threshold": bool(pass_threshold and bool(refine_result["ipopt_success"])),
        "casadi_refinement": True,
        "optimization_rerun": True,
        "production_solver_parity_claim": True,
        "high_fidelity_validation": False,
        "high_fidelity_flight_validation": False,
        "fuel_optimality_claim": False,
        "doi_claim": False,
        "quantum_advantage_claim": False,
        "ipopt_success": bool(refine_result["ipopt_success"]),
        "ipopt_status": str(refine_result["ipopt_status"]),
        "ipopt_iterations": int(refine_result["ipopt_iterations"]),
        "ipopt_objective": float(refine_result["ipopt_objective"]),
        "runtime_seconds": float(refine_result["runtime_seconds"]),
        "control_bound": float(cfg.amax),
        "refined_endpoint_norm_max": diagnostics["endpoint_norm_max"],
        "refined_midpoint_norm_max": diagnostics["midpoint_norm_max"],
        "max_control_norm": diagnostics["max_control_norm"],
        "max_control_bound_violation": diagnostics["max_control_bound_violation"],
        "substeps_per_segment": int(cfg.substeps),
        "transfer_time": float(cfg.tf),
        "bridge_semantics": (
            "Scoped CasADi/IPOPT direct-shooting refinement under normalized CR3BP; "
            "branch prefixes are fixed to refined nominal masked controls and only post-recovery active "
            "controls are variables; "
            "not high-fidelity validation, flight validation, fuel optimality, DOI evidence, or quantum evidence."
        ),
    }


def _write_table(summary: dict[str, object], rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / BRIDGE_TABLE_NAME
    if not rows:
        path.write_text("% No independent-HS CasADi/IPOPT bridge rows.\n", encoding="utf-8")
        return path
    frame = pd.DataFrame(rows)
    display = frame[
        [
            "record_type",
            "mask_index",
            "recovery_start",
            "source_replay_terminal_error",
            "ipopt_refined_terminal_error",
            "configured_threshold",
            "bridge_passes_configured_threshold",
            "ipopt_success",
            "ipopt_iterations",
        ]
    ].rename(
        columns={
            "record_type": "Type",
            "mask_index": "Mask",
            "recovery_start": "Recovery start",
            "source_replay_terminal_error": "Source replay",
            "ipopt_refined_terminal_error": "IPOPT refined",
            "configured_threshold": "Threshold",
            "bridge_passes_configured_threshold": "Bridge pass",
            "ipopt_success": "IPOPT success",
            "ipopt_iterations": "Iterations",
        }
    )
    caption = (
        "% Scoped CasADi/IPOPT bridge summary: "
        f"worst refined={float(summary['worst_ipopt_refined_terminal_error']):.6g}, "
        f"pass count={int(summary['ipopt_refined_pass_count'])}/{int(summary['row_count'])}.\n"
    )
    path.write_text(caption + display.to_latex(index=False, float_format="%.6g", escape=True), encoding="utf-8")
    return path


def _selected_rows(input_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    case_ids = set(args.case_id or [POLISH_CASE_ID])
    rows = []
    for row in input_rows:
        if case_ids and str(row.get("case_id", "")) not in case_ids:
            continue
        if not _bool_value(row.get("branch_control_replay_ready", "")):
            continue
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> pd.DataFrame:
    ca = _require_casadi()
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_results_dir, _, _ = output_directories(Path.cwd(), config)
    input_csv = input_results_dir / "independent_hs_all_configured_headroom.csv"
    if not input_csv.is_file():
        artifact_stem = str((config.get("run", {}) or {}).get("artifact_stem", "independent_hs_continuation_baseline"))
        input_csv = input_results_dir / f"{artifact_stem}.csv"
    input_rows = _read_input_rows(input_csv)
    selected_rows = _selected_rows(input_rows, args)
    if not selected_rows:
        raise RuntimeError(f"no replay-ready independent-HS rows selected for case IDs {sorted(args.case_id or [POLISH_CASE_ID])}")

    cases = _case_by_id(config)
    results_dir = _resolve_output_path(args.results_dir)
    tables_dir = _resolve_output_path(args.tables_dir)
    controls_dir = results_dir / "controls"
    results_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)

    input_artifacts: list[dict[str, object]] = []
    seen_artifacts: set[Path] = set()
    for path in (config_path, source_states, input_csv):
        _append_artifact(input_artifacts, seen_artifacts, path)

    bridge_rows: list[dict[str, object]] = []
    refined_sidecars: list[dict[str, object]] = []
    manifest_path: Path | None = None
    manifest_sha = ""

    for row in selected_rows:
        case_id = str(row["case_id"])
        if case_id not in cases:
            raise RuntimeError(f"selected case {case_id} is not present in config suite")
        case_config = _case_config(config, cases[case_id])
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        target = np.asarray(states.target, dtype=float)
        thresholds = case_config["objective"]["thresholds"]
        nominal_threshold = float(thresholds["nominal_success"])
        robust_threshold = float(thresholds["robust_success"])
        manifest, manifest_path, manifest_sha = _read_json_verified(
            row["branch_control_manifest_path"],
            row.get("branch_control_manifest_hash"),
        )
        nominal, nominal_path, nominal_sha = _read_json_verified(
            manifest["nominal_control_path"],
            manifest.get("nominal_control_sha256"),
        )
        _append_artifact(input_artifacts, seen_artifacts, manifest_path)
        _append_artifact(input_artifacts, seen_artifacts, nominal_path)
        recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
        if recorded_target.shape != target.shape:
            raise RuntimeError(f"manifest target_state shape mismatch for {case_id}")
        target_delta = float(np.max(np.abs(target - recorded_target)))
        if target_delta > float(args.baseline_tolerance):
            raise RuntimeError(f"target_state mismatch for {case_id}: max abs delta {target_delta}")

        nominal_endpoint = np.asarray(nominal.get("nominal_endpoint_controls", nominal.get("controls")), dtype=float)
        nominal_midpoint = np.asarray(nominal.get("nominal_midpoint_controls"), dtype=float)
        nominal_recorded = float(manifest.get("nominal_error", row.get("nominal_error")))
        nominal_replay = _terminal_error(
            state0=np.asarray(states.initial, dtype=float),
            target=target,
            cfg=cfg,
            endpoint_controls=nominal_endpoint,
            midpoint_controls=nominal_midpoint,
        )
        nominal_active = np.ones(int(cfg.n_segments), dtype=int)
        nominal_variable_mask = nominal_active.copy()
        nominal_refined = _solve_casadi_ipopt_refinement(
            state0=np.asarray(states.initial, dtype=float),
            target=target,
            cfg=cfg,
            endpoint_seed=nominal_endpoint,
            midpoint_seed=nominal_midpoint,
            variable_control_mask=nominal_variable_mask,
            args=args,
        )
        nominal_refined_endpoint = np.asarray(nominal_refined["endpoint_controls"], dtype=float)
        nominal_refined_midpoint = np.asarray(nominal_refined["midpoint_controls"], dtype=float)
        nominal_prefix_diagnostics = _prefix_invariant_diagnostics(
            endpoint_controls=nominal_refined_endpoint,
            midpoint_controls=nominal_refined_midpoint,
            nominal_endpoint_controls=None,
            nominal_midpoint_controls=None,
            outage_mask=None,
            recovery_start=None,
        )
        nominal_sidecar_path, nominal_sidecar_sha = _write_refined_sidecar(
            controls_dir=controls_dir,
            case_id=case_id,
            record_type="nominal",
            branch_order=None,
            mask_index=None,
            outage_mask=None,
            active_mask=nominal_active,
            recovery_start=None,
            variable_control_mask=nominal_variable_mask,
            prefix_invariant_diagnostics=nominal_prefix_diagnostics,
            source_controls_path=nominal_path,
            source_controls_sha256=nominal_sha,
            source_recorded_cr3bp_terminal_error=nominal_recorded,
            source_replay_terminal_error=nominal_replay,
            configured_threshold=nominal_threshold,
            refine_result=nominal_refined,
            cfg=cfg,
        )
        refined_sidecars.append(
            {
                "record_type": "nominal",
                "path": _relative_or_absolute(nominal_sidecar_path),
                "sha256": nominal_sidecar_sha,
                "ipopt_refined_terminal_error": float(nominal_refined["terminal_error"]),
            }
        )
        bridge_rows.append(
            _row_for_result(
                case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                source_controls_path=nominal_path,
                source_controls_sha256=nominal_sha,
                refined_controls_path=nominal_sidecar_path,
                refined_controls_sha256=nominal_sidecar_sha,
                active_mask=nominal_active,
                recovery_start=None,
                variable_control_mask=nominal_variable_mask,
                prefix_invariant_diagnostics=nominal_prefix_diagnostics,
                source_recorded_cr3bp_terminal_error=nominal_recorded,
                source_replay_terminal_error=nominal_replay,
                configured_threshold=nominal_threshold,
                refine_result=nominal_refined,
                cfg=cfg,
            )
        )

        branch_entries = list(manifest.get("branch_control_sidecars", []))
        expected_branch_count = int(manifest.get("expected_branch_count", row.get("branch_control_sidecar_count", 0)))
        if len(branch_entries) != expected_branch_count:
            raise RuntimeError(
                f"manifest branch sidecar count {len(branch_entries)} does not match expected {expected_branch_count}"
            )
        for entry in branch_entries:
            branch, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            _append_artifact(input_artifacts, seen_artifacts, branch_path)
            endpoint_controls = np.asarray(branch.get("branch_endpoint_controls", branch.get("branch_controls")), dtype=float)
            midpoint_controls = np.asarray(branch.get("branch_midpoint_controls"), dtype=float)
            outage_mask = [int(value) for value in branch["outage_mask"]]
            active_mask = np.asarray(outage_mask, dtype=int)
            recovery_start = _branch_recovery_start(
                branch=branch,
                entry=entry,
                outage_mask=outage_mask,
                n_segments=int(cfg.n_segments),
            )
            variable_control_mask = _branch_variable_control_mask(outage_mask, recovery_start)
            fixed_endpoint = np.zeros_like(endpoint_controls)
            fixed_midpoint = np.zeros_like(midpoint_controls)
            if recovery_start > 0:
                prefix_mask = active_mask[:recovery_start, None]
                fixed_endpoint[:recovery_start] = nominal_refined_endpoint[:recovery_start] * prefix_mask
                fixed_midpoint[:recovery_start] = nominal_refined_midpoint[:recovery_start] * prefix_mask
            branch_recorded = float(branch["recorded_branch_terminal_error"])
            branch_replay = _terminal_error(
                state0=np.asarray(states.initial, dtype=float),
                target=target,
                cfg=cfg,
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
            )
            branch_refined = _solve_casadi_ipopt_refinement(
                state0=np.asarray(states.initial, dtype=float),
                target=target,
                cfg=cfg,
                endpoint_seed=endpoint_controls,
                midpoint_seed=midpoint_controls,
                variable_control_mask=variable_control_mask,
                fixed_endpoint_controls=fixed_endpoint,
                fixed_midpoint_controls=fixed_midpoint,
                args=args,
            )
            branch_prefix_diagnostics = _prefix_invariant_diagnostics(
                endpoint_controls=np.asarray(branch_refined["endpoint_controls"], dtype=float),
                midpoint_controls=np.asarray(branch_refined["midpoint_controls"], dtype=float),
                nominal_endpoint_controls=nominal_refined_endpoint,
                nominal_midpoint_controls=nominal_refined_midpoint,
                outage_mask=outage_mask,
                recovery_start=recovery_start,
            )
            sidecar_path, sidecar_sha = _write_refined_sidecar(
                controls_dir=controls_dir,
                case_id=case_id,
                record_type="branch",
                branch_order=int(branch["branch_order"]),
                mask_index=int(branch["mask_index"]),
                outage_mask=outage_mask,
                active_mask=active_mask,
                recovery_start=recovery_start,
                variable_control_mask=variable_control_mask,
                prefix_invariant_diagnostics=branch_prefix_diagnostics,
                source_controls_path=branch_path,
                source_controls_sha256=branch_sha,
                source_recorded_cr3bp_terminal_error=branch_recorded,
                source_replay_terminal_error=branch_replay,
                configured_threshold=robust_threshold,
                refine_result=branch_refined,
                cfg=cfg,
            )
            refined_sidecars.append(
                {
                    "record_type": "branch",
                    "branch_order": int(branch["branch_order"]),
                    "mask_index": int(branch["mask_index"]),
                    "recovery_start": recovery_start,
                    "variable_control_mask": [
                        int(value) for value in np.asarray(variable_control_mask, dtype=int).tolist()
                    ],
                    "max_prefix_control_delta_from_refined_nominal_masked": float(
                        branch_prefix_diagnostics["max_prefix_control_delta_from_refined_nominal_masked"]
                    ),
                    "path": _relative_or_absolute(sidecar_path),
                    "sha256": sidecar_sha,
                    "ipopt_refined_terminal_error": float(branch_refined["terminal_error"]),
                }
            )
            bridge_rows.append(
                _row_for_result(
                    case_id=case_id,
                    record_type="branch",
                    branch_order=int(branch["branch_order"]),
                    mask_index=int(branch["mask_index"]),
                    outage_mask=outage_mask,
                    source_controls_path=branch_path,
                    source_controls_sha256=branch_sha,
                    refined_controls_path=sidecar_path,
                    refined_controls_sha256=sidecar_sha,
                    active_mask=active_mask,
                    recovery_start=recovery_start,
                    variable_control_mask=variable_control_mask,
                    prefix_invariant_diagnostics=branch_prefix_diagnostics,
                    source_recorded_cr3bp_terminal_error=branch_recorded,
                    source_replay_terminal_error=branch_replay,
                    configured_threshold=robust_threshold,
                    refine_result=branch_refined,
                    cfg=cfg,
                )
            )

    df = pd.DataFrame(bridge_rows, columns=BRIDGE_COLUMNS)
    csv_path = results_dir / BRIDGE_CSV_NAME
    df.to_csv(csv_path, index=False, float_format="%.17g")

    nominal_rows = df[df["record_type"] == "nominal"]
    branch_rows = df[df["record_type"] == "branch"]
    summary = {
        "case_id": POLISH_CASE_ID,
        "row_count": int(len(df)),
        "nominal_row_count": int(len(nominal_rows)),
        "branch_row_count": int(len(branch_rows)),
        "source_nominal_replay_terminal_error": float(nominal_rows["source_replay_terminal_error"].max()),
        "source_branch_replay_worst_terminal_error": float(branch_rows["source_replay_terminal_error"].max()),
        "worst_source_replay_terminal_error": float(df["source_replay_terminal_error"].max()),
        "ipopt_refined_nominal_terminal_error": float(nominal_rows["ipopt_refined_terminal_error"].max()),
        "ipopt_refined_branch_worst_terminal_error": float(branch_rows["ipopt_refined_terminal_error"].max()),
        "worst_ipopt_refined_terminal_error": float(df["ipopt_refined_terminal_error"].max()),
        "ipopt_refined_pass_count": int(df["ipopt_refined_passes_configured_threshold"].map(bool).sum()),
        "bridge_pass_count": int(df["bridge_passes_configured_threshold"].map(bool).sum()),
        "ipopt_success_count": int(df["ipopt_success"].map(bool).sum()),
        "ipopt_total_iterations": int(df["ipopt_iterations"].astype(int).clip(lower=0).sum()),
        "total_runtime_seconds": float(df["runtime_seconds"].astype(float).sum()),
        "max_control_norm": float(df["max_control_norm"].astype(float).max()),
        "max_control_bound_violation": float(df["max_control_bound_violation"].astype(float).max()),
        "max_branch_prefix_control_delta_from_refined_nominal_masked": (
            float(branch_rows["max_prefix_control_delta_from_refined_nominal_masked"].astype(float).max())
            if len(branch_rows)
            else 0.0
        ),
        "zero_variable_branch_count": int(
            branch_rows["variable_control_mask"].map(lambda value: sum(json.loads(str(value))) == 0).sum()
        )
        if len(branch_rows)
        else 0,
        "max_abs_terminal_error_delta_from_source_replay": float(
            df["terminal_error_delta_from_source_replay"].astype(float).abs().max()
        ),
        "min_terminal_error_improvement_from_source_replay": float(
            df["terminal_error_improvement_from_source_replay"].astype(float).min()
        ),
        "all_rows_pass_configured_threshold": bool(df["ipopt_refined_passes_configured_threshold"].map(bool).all()),
        "all_ipopt_success": bool(df["ipopt_success"].map(bool).all()),
    }
    table_path = _write_table(summary, bridge_rows, tables_dir)

    control_manifest_path = controls_dir / BRIDGE_CONTROL_MANIFEST_NAME
    control_manifest_payload = {
        "schema_version": 1,
        "manifest_type": "independent_hs_casadi_ipopt_bridge_control_manifest",
        "case_id": POLISH_CASE_ID,
        "refined_sidecar_count": int(len(refined_sidecars)),
        "refined_control_sidecars": refined_sidecars,
        "source_manifest_path": "" if manifest_path is None else _relative_or_absolute(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "summary": summary,
        "limitations": [
            "Scoped CasADi/IPOPT direct-shooting bridge for the normalized CR3BP polish case only.",
            "Branch rows fix pre-recovery prefixes to refined nominal masked controls and expose only post-recovery active controls.",
            "This is not high-fidelity validation, production mission design, global/fuel optimality, DOI evidence, or quantum evidence.",
        ],
    }
    _write_json(control_manifest_path, control_manifest_payload)
    control_manifest_sha = _sha256(control_manifest_path)

    metadata_path = results_dir / BRIDGE_METADATA_NAME
    limitations = [
        "CasADi/IPOPT is used as a mature NLP backend for a scoped direct-shooting bridge check only.",
        "The benchmark remains normalized Earth-Moon CR3BP with the original target, scales, thresholds, and branch masks.",
        "This local refinement does not establish production mission design, global optimality, fuel optimality, or high-fidelity validation.",
        "No DOI, quantum advantage, QUBO, or QAOA claim is supported by this artifact.",
    ]
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "nominal_row_count": int(len(nominal_rows)),
        "branch_row_count": int(len(branch_rows)),
        "casadi_version": str(ca.__version__),
        "ipopt_backend": "casadi.nlpsol(ipopt)",
        "casadi_refinement": True,
        "optimization_rerun": True,
        "production_solver_parity_claim": True,
        "production_solver_parity_claim_scope": (
            "Scoped CasADi/IPOPT mature NLP backend bridge check for one normalized-CR3BP "
            "accepted independent-HS benchmark case; not a production flight-design claim."
        ),
        "high_fidelity_validation": False,
        "high_fidelity_flight_validation": False,
        "fuel_optimality_claim": False,
        "doi_claim": False,
        "quantum_advantage_claim": False,
        "runtime_network_dependency": False,
        "uses_normalized_cr3bp": True,
        "branch_outage_segments_fixed_inactive": True,
        "branch_prefix_fixed_to_refined_nominal_masked_controls": True,
        "branch_variable_controls_recovery_only": True,
        "branch_recovery_start_source": (
            "Source branch sidecar or manifest explicit recovery_start when present; otherwise one plus the last "
            "zero in the outage mask."
        ),
        "refinement_settings": {
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "acceptable_tol": float(args.acceptable_tol),
            "terminal_residual_weight": float(args.terminal_residual_weight),
            "seed_deviation_regularization_weight": float(args.seed_deviation_regularization_weight),
            "control_regularization_weight": float(args.control_regularization_weight),
            "smooth_regularization_weight": float(args.smooth_regularization_weight),
        },
        "polish_case_summary": summary,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_casadi_ipopt_bridge_csv": _relative_or_absolute(csv_path),
            "independent_hs_casadi_ipopt_bridge_metadata_json": _relative_or_absolute(metadata_path),
            "independent_hs_casadi_ipopt_bridge_table_tex": _relative_or_absolute(table_path),
            "casadi_ipopt_bridge_control_manifest_json": _relative_or_absolute(control_manifest_path),
            "casadi_ipopt_bridge_control_manifest_sha256": control_manifest_sha,
            "casadi_ipopt_bridge_control_sidecars": refined_sidecars,
        },
        "limitations": limitations,
        "interpretation_limits": limitations,
    }
    _write_json(metadata_path, metadata)
    print(
        "Completed independent-HS CasADi/IPOPT bridge "
        f"with {summary['row_count']} rows, refined nominal={summary['ipopt_refined_nominal_terminal_error']:.6g}, "
        f"refined branch worst={summary['ipopt_refined_branch_worst_terminal_error']:.6g}, "
        f"IPOPT success={summary['ipopt_success_count']}/{summary['row_count']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a scoped CasADi/IPOPT direct-shooting bridge for the independent-HS polish case."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--baseline-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--acceptable-tol", type=float, default=1.0e-7)
    parser.add_argument("--terminal-residual-weight", type=float, default=1.0)
    parser.add_argument("--seed-deviation-regularization-weight", type=float, default=1.0e-4)
    parser.add_argument("--control-regularization-weight", type=float, default=1.0e-5)
    parser.add_argument("--smooth-regularization-weight", type=float, default=5.0e-5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
