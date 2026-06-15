from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .cr3bp import cr3bp_derivative, rk4_step
from .objective import ObjectiveConfig, state_error
from .refinement import control_fuel, feedback_controls_for_schedule, project_controls_to_ball, selected_outage_indices_for_schedule


METHOD_TYPE = "bounded_projected_trapezoidal_direct_collocation"
COLLOCATION_METHOD_TYPES = {
    "trapezoidal": METHOD_TYPE,
    "hermite_simpson": "bounded_projected_constant_control_hermite_simpson_direct_collocation",
}


def normalize_collocation_method(method: str | None) -> str:
    normalized = str(method or "trapezoidal").lower().replace("-", "_")
    if normalized not in COLLOCATION_METHOD_TYPES:
        allowed = ", ".join(sorted(COLLOCATION_METHOD_TYPES))
        raise ValueError(f"direct-collocation method must be one of: {allowed}")
    return normalized


def collocation_method_type(method: str | None) -> str:
    return COLLOCATION_METHOD_TYPES[normalize_collocation_method(method)]


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path | str | None) -> str:
    if path is None:
        return "default"
    raw = Path(path)
    try:
        resolved = raw.resolve(strict=False)
    except OSError:
        resolved = raw
    digest = "missing"
    try:
        if raw.is_file():
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    except OSError:
        digest = "unreadable"
    return f"{resolved.as_posix()}|sha256={digest}"


def settings_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scale_vector(cfg: ObjectiveConfig) -> np.ndarray:
    return np.array(
        [
            cfg.position_scale,
            cfg.position_scale,
            cfg.position_scale,
            cfg.velocity_scale,
            cfg.velocity_scale,
            cfg.velocity_scale,
        ],
        dtype=float,
    )


def scaled_state_residual(state: np.ndarray, target: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    return (np.asarray(state, dtype=float) - np.asarray(target, dtype=float)) / np.maximum(scale_vector(cfg), 1e-12)


def outage_end(mask: np.ndarray) -> int:
    missed = np.flatnonzero(np.asarray(mask, dtype=float) < 0.5)
    if missed.size == 0:
        return 0
    return int(missed[-1] + 1)


def propagate_piecewise_controls(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    tf: float,
    substeps_per_segment: int,
    *,
    return_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    controls = project_controls_to_ball(np.asarray(controls, dtype=float), np.inf)
    state = np.asarray(state0, dtype=float).copy()
    n_segments = int(controls.shape[0])
    if n_segments == 0:
        nodes = np.asarray([state.copy()]) if return_nodes else None
        return state, nodes
    h = float(tf) / float(n_segments)
    dt = h / float(substeps_per_segment)
    nodes = [state.copy()] if return_nodes else None
    for control in controls:
        for _ in range(int(substeps_per_segment)):
            state = rk4_step(state, mu, dt, control)
        if return_nodes:
            nodes.append(state.copy())
    return state, np.asarray(nodes) if return_nodes else None


def propagate_prefix(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    segment_dt: float,
    substeps_per_segment: int,
) -> np.ndarray:
    state = np.asarray(state0, dtype=float).copy()
    if len(controls) == 0:
        return state
    dt = float(segment_dt) / float(substeps_per_segment)
    for control in np.asarray(controls, dtype=float):
        for _ in range(int(substeps_per_segment)):
            state = rk4_step(state, mu, dt, control)
    return state


def trapezoidal_defect(left: np.ndarray, right: np.ndarray, control: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    h = float(cfg.tf) / float(cfg.n_segments)
    f_left = cr3bp_derivative(left, cfg.mu, control)
    f_right = cr3bp_derivative(right, cfg.mu, control)
    return np.asarray(right, dtype=float) - np.asarray(left, dtype=float) - 0.5 * h * (f_left + f_right)


def hermite_simpson_defect(left: np.ndarray, right: np.ndarray, control: np.ndarray, cfg: ObjectiveConfig) -> np.ndarray:
    h = float(cfg.tf) / float(cfg.n_segments)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    control = np.asarray(control, dtype=float)
    f_left = cr3bp_derivative(left, cfg.mu, control)
    f_right = cr3bp_derivative(right, cfg.mu, control)
    x_mid = 0.5 * (left + right) + h / 8.0 * (f_left - f_right)
    f_mid = cr3bp_derivative(x_mid, cfg.mu, control)
    return right - left - h / 6.0 * (f_left + 4.0 * f_mid + f_right)


def collocation_defect(
    left: np.ndarray,
    right: np.ndarray,
    control: np.ndarray,
    cfg: ObjectiveConfig,
    method: str,
) -> np.ndarray:
    normalized = normalize_collocation_method(method)
    if normalized == "trapezoidal":
        return trapezoidal_defect(left, right, control, cfg)
    return hermite_simpson_defect(left, right, control, cfg)


@dataclass(frozen=True)
class DirectCollocationWeights:
    initial: float = 10.0
    defect: float = 1.0
    terminal: float = 5.0
    branch_start: float = 8.0
    branch_defect: float = 1.0
    branch_terminal: float = 5.0
    control: float = 0.01
    smooth: float = 0.01

    @classmethod
    def from_config(cls, config: dict | None) -> "DirectCollocationWeights":
        cfg = dict(config or {})
        weights = dict(cfg.get("weights", {}) or {})
        return cls(
            initial=float(weights.get("initial", cfg.get("initial_weight", cls.initial))),
            defect=float(weights.get("defect", cfg.get("defect_weight", cls.defect))),
            terminal=float(weights.get("terminal", cfg.get("terminal_weight", cls.terminal))),
            branch_start=float(weights.get("branch_start", cfg.get("branch_start_weight", cls.branch_start))),
            branch_defect=float(weights.get("branch_defect", cfg.get("branch_defect_weight", cls.branch_defect))),
            branch_terminal=float(weights.get("branch_terminal", cfg.get("branch_terminal_weight", cls.branch_terminal))),
            control=float(weights.get("control", cfg.get("control_weight", cls.control))),
            smooth=float(weights.get("smooth", cfg.get("smooth_weight", cls.smooth))),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "initial": self.initial,
            "defect": self.defect,
            "terminal": self.terminal,
            "branch_start": self.branch_start,
            "branch_defect": self.branch_defect,
            "branch_terminal": self.branch_terminal,
            "control": self.control,
            "smooth": self.smooth,
        }


class DirectCollocationLayout:
    def __init__(self, cfg: ObjectiveConfig, selected_masks: np.ndarray):
        self.cfg = cfg
        self.selected_masks = np.asarray(selected_masks, dtype=float)
        self.starts = [outage_end(mask) for mask in self.selected_masks]
        cursor = 0
        self.nominal_nodes = slice(cursor, cursor + (cfg.n_segments + 1) * 6)
        cursor = self.nominal_nodes.stop
        self.nominal_controls = slice(cursor, cursor + cfg.n_segments * 3)
        cursor = self.nominal_controls.stop
        self.branch_nodes: list[slice] = []
        self.branch_controls: list[slice] = []
        for start in self.starts:
            node_count = cfg.n_segments - start + 1
            control_count = cfg.n_segments - start
            self.branch_nodes.append(slice(cursor, cursor + node_count * 6))
            cursor = self.branch_nodes[-1].stop
            self.branch_controls.append(slice(cursor, cursor + control_count * 3))
            cursor = self.branch_controls[-1].stop
        self.size = cursor

    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        cfg = self.cfg
        nominal_nodes = np.asarray(vec[self.nominal_nodes], dtype=float).reshape((cfg.n_segments + 1, 6))
        nominal_controls = project_controls_to_ball(
            np.asarray(vec[self.nominal_controls], dtype=float).reshape((cfg.n_segments, 3)),
            cfg.amax,
        )
        branch_nodes = []
        branch_controls = []
        for start, node_slice, control_slice in zip(self.starts, self.branch_nodes, self.branch_controls):
            branch_nodes.append(np.asarray(vec[node_slice], dtype=float).reshape((cfg.n_segments - start + 1, 6)))
            control_count = cfg.n_segments - start
            if control_count:
                controls = np.asarray(vec[control_slice], dtype=float).reshape((control_count, 3))
            else:
                controls = np.zeros((0, 3), dtype=float)
            branch_controls.append(project_controls_to_ball(controls, cfg.amax))
        return nominal_nodes, nominal_controls, branch_nodes, branch_controls

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.size, -np.inf, dtype=float)
        upper = np.full(self.size, np.inf, dtype=float)
        for control_slice in [self.nominal_controls, *self.branch_controls]:
            lower[control_slice] = -self.cfg.amax
            upper[control_slice] = self.cfg.amax
        return lower, upper


def _linear_nodes(start: np.ndarray, target: np.ndarray, count: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, int(count), dtype=float)[:, None]
    return (1.0 - weights) * np.asarray(start, dtype=float)[None, :] + weights * np.asarray(target, dtype=float)[None, :]


def initial_guess(
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    selected_masks: np.ndarray,
    *,
    node_initialization: str = "blend",
    node_initialization_blend: float = 0.35,
) -> tuple[DirectCollocationLayout, np.ndarray]:
    layout = DirectCollocationLayout(cfg, selected_masks)
    schedule = np.ones(cfg.n_segments, dtype=int)
    nominal_controls = feedback_controls_for_schedule(schedule, state0, target, cfg)
    _, rollout_nodes = propagate_piecewise_controls(state0, nominal_controls, cfg.mu, cfg.tf, cfg.substeps, return_nodes=True)
    assert rollout_nodes is not None
    linear_nodes = _linear_nodes(state0, target, cfg.n_segments + 1)
    mode = str(node_initialization or "blend").lower().replace("_", "-")
    if mode == "rollout":
        nominal_nodes = rollout_nodes
    elif mode == "linear":
        nominal_nodes = linear_nodes
    elif mode == "blend":
        blend = min(1.0, max(0.0, float(node_initialization_blend)))
        nominal_nodes = (1.0 - blend) * rollout_nodes + blend * linear_nodes
    else:
        raise ValueError("node_initialization must be one of rollout, linear, blend")

    chunks = [nominal_nodes.reshape(-1), nominal_controls.reshape(-1)]
    h = float(cfg.tf) / float(cfg.n_segments)
    for mask, start in zip(np.asarray(selected_masks, dtype=float), layout.starts):
        prefix_controls = nominal_controls[:start] * mask[:start, None]
        branch_start = propagate_prefix(state0, prefix_controls, cfg.mu, h, cfg.substeps)
        branch_nodes = _linear_nodes(branch_start, target, cfg.n_segments - start + 1)
        branch_controls = nominal_controls[start:].copy()
        chunks.append(branch_nodes.reshape(-1))
        chunks.append(branch_controls.reshape(-1))
    return layout, np.concatenate(chunks)


def branch_full_controls(nominal_controls: np.ndarray, mask: np.ndarray, start: int, recovery_controls: np.ndarray, amax: float) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    if int(start) < controls.shape[0]:
        controls[int(start) :] = np.asarray(recovery_controls, dtype=float)
    return project_controls_to_ball(controls, amax)


class DirectCollocationProblem:
    def __init__(
        self,
        *,
        state0: np.ndarray,
        target: np.ndarray,
        cfg: ObjectiveConfig,
        masks: np.ndarray,
        selected: np.ndarray,
        layout: DirectCollocationLayout,
        weights: DirectCollocationWeights,
        method: str = "trapezoidal",
    ):
        self.state0 = np.asarray(state0, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.cfg = cfg
        self.masks = np.asarray(masks, dtype=float)
        self.selected = np.asarray(selected, dtype=int)
        self.layout = layout
        self.weights = weights
        self.method = normalize_collocation_method(method)

    def residual(self, vec: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        nodes, controls, branch_nodes, branch_controls = self.layout.unpack(vec)
        scale = np.maximum(scale_vector(cfg), 1e-12)
        chunks = [
            self.weights.initial * scaled_state_residual(nodes[0], self.state0, cfg),
        ]
        for i in range(cfg.n_segments):
            chunks.append(
                self.weights.defect
                * collocation_defect(nodes[i], nodes[i + 1], controls[i], cfg, self.method)
                / scale
            )
        chunks.append(self.weights.terminal * scaled_state_residual(nodes[-1], self.target, cfg))
        chunks.append(self.weights.control * controls.reshape(-1) / max(cfg.amax, 1e-12))
        if cfg.n_segments > 1:
            chunks.append(self.weights.smooth * np.diff(controls, axis=0).reshape(-1) / max(cfg.amax, 1e-12))

        h = float(cfg.tf) / float(cfg.n_segments)
        for mask_index, mask, start, nodes_b, controls_b in zip(
            self.selected,
            self.layout.selected_masks,
            self.layout.starts,
            branch_nodes,
            branch_controls,
        ):
            del mask_index
            branch_start = propagate_prefix(self.state0, controls[:start] * mask[:start, None], cfg.mu, h, cfg.substeps)
            chunks.append(self.weights.branch_start * scaled_state_residual(nodes_b[0], branch_start, cfg))
            for j in range(cfg.n_segments - start):
                chunks.append(
                    self.weights.branch_defect
                    * collocation_defect(nodes_b[j], nodes_b[j + 1], controls_b[j], cfg, self.method)
                    / scale
                )
            chunks.append(self.weights.branch_terminal * scaled_state_residual(nodes_b[-1], self.target, cfg))
            if controls_b.size:
                chunks.append(self.weights.control * controls_b.reshape(-1) / max(cfg.amax, 1e-12))
            if controls_b.shape[0] > 1:
                chunks.append(self.weights.smooth * np.diff(controls_b, axis=0).reshape(-1) / max(cfg.amax, 1e-12))
        return np.concatenate(chunks)

    def evaluate_vector(self, vec: np.ndarray, thresholds: dict) -> dict:
        cfg = self.cfg
        _, nominal_controls, _, selected_branch_controls = self.layout.unpack(vec)
        nominal_final, nominal_history = propagate_piecewise_controls(
            self.state0,
            nominal_controls,
            cfg.mu,
            cfg.tf,
            cfg.substeps,
            return_nodes=True,
        )
        nominal_error = float(state_error(nominal_final, self.target, cfg.position_scale, cfg.velocity_scale))

        selected_lookup = {
            int(mask_index): branch_full_controls(nominal_controls, mask, start, controls, cfg.amax)
            for mask_index, mask, start, controls in zip(
                self.selected,
                self.layout.selected_masks,
                self.layout.starts,
                selected_branch_controls,
            )
        }
        all_errors: list[float] = []
        all_fuels: list[float] = []
        selected_errors: list[float] = []
        selected_fuels: list[float] = []
        for mask_index, mask in enumerate(self.masks):
            if mask_index in selected_lookup:
                controls = selected_lookup[mask_index]
            else:
                controls = project_controls_to_ball(nominal_controls * mask[:, None], cfg.amax)
            final, _ = propagate_piecewise_controls(self.state0, controls, cfg.mu, cfg.tf, cfg.substeps)
            err = float(state_error(final, self.target, cfg.position_scale, cfg.velocity_scale))
            all_errors.append(err)
            all_fuels.append(control_fuel(controls, cfg.tf))
            if mask_index in set(int(v) for v in self.selected.tolist()):
                selected_errors.append(err)
                selected_fuels.append(control_fuel(controls, cfg.tf))

        selected_worst = float(np.max(selected_errors)) if selected_errors else nominal_error
        all_worst = float(np.max(all_errors)) if all_errors else selected_worst
        all_controls = [nominal_controls, *selected_branch_controls]
        max_norm = max(
            [0.0]
            + [
                float(np.max(np.linalg.norm(np.asarray(item, dtype=float), axis=1)))
                for item in all_controls
                if np.asarray(item, dtype=float).size
            ]
        )
        bound_violation = max(0.0, max_norm - float(cfg.amax))
        if bound_violation <= 1e-12:
            bound_violation = 0.0
            max_norm = min(max_norm, float(cfg.amax))
        converged = bool(
            nominal_error <= float(thresholds["nominal_success"])
            and selected_worst <= float(thresholds["robust_success"])
        )
        return {
            "success": converged,
            "nominal_error": nominal_error,
            "selected_worst_error": selected_worst,
            "all_mask_worst_error": all_worst,
            "selected_outage_errors": selected_errors,
            "all_outage_errors": all_errors,
            "nominal_controls": nominal_controls,
            "selected_branch_controls": selected_branch_controls,
            "nominal_history": nominal_history,
            "nominal_fuel": control_fuel(nominal_controls, cfg.tf),
            "recovery_fuel_mean": float(np.mean(selected_fuels)) if selected_fuels else control_fuel(nominal_controls, cfg.tf),
            "recovery_fuel_max": float(np.max(selected_fuels)) if selected_fuels else control_fuel(nominal_controls, cfg.tf),
            "all_mask_recovery_fuel_mean": float(np.mean(all_fuels)) if all_fuels else control_fuel(nominal_controls, cfg.tf),
            "control_max_norm": max_norm,
            "control_bound_violation": bound_violation,
        }


def run_direct_collocation_baseline(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg: ObjectiveConfig,
    masks: np.ndarray,
    thresholds: dict,
    selected_outages: int,
    max_nfev: int,
    min_recovery_segments: int = 1,
    collocation_config: dict | None = None,
) -> dict:
    start = time.perf_counter()
    collocation_config = dict(collocation_config or {})
    method = normalize_collocation_method(collocation_config.get("method", "trapezoidal"))
    schedule = np.ones(cfg.n_segments, dtype=int)
    if int(selected_outages) <= 0:
        selected = np.zeros(0, dtype=int)
    else:
        selected = selected_outage_indices_for_schedule(
            schedule,
            state0,
            target,
            cfg,
            masks,
            int(selected_outages),
            int(min_recovery_segments),
        )
    selected_masks = masks[selected] if selected.size else np.zeros((0, cfg.n_segments), dtype=float)
    layout, x0 = initial_guess(
        state0,
        target,
        cfg,
        selected_masks,
        node_initialization=str(collocation_config.get("node_initialization", "blend")),
        node_initialization_blend=float(collocation_config.get("node_initialization_blend", 0.35)),
    )
    problem = DirectCollocationProblem(
        state0=state0,
        target=target,
        cfg=cfg,
        masks=masks,
        selected=selected,
        layout=layout,
        weights=DirectCollocationWeights.from_config(collocation_config),
        method=method,
    )
    lower, upper = layout.bounds()
    result = least_squares(
        problem.residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=float(collocation_config.get("xtol", 1e-5)),
        ftol=float(collocation_config.get("ftol", 1e-5)),
        gtol=float(collocation_config.get("gtol", 1e-5)),
        verbose=0,
    )
    evaluation = problem.evaluate_vector(result.x, thresholds)
    selected_branch_semantics = (
        "selected_outages=0/no selected outage branches: only the nominal trajectory is optimized; "
        "selected_worst_error equals nominal_error; all outage masks are diagnostic under masked nominal controls"
        if selected.size == 0
        else "nominal trajectory and selected outage branches are optimized in one least-squares direct transcription; branch starts are fixed by RK4 propagation through the missed segment(s), then branch controls are re-optimized after the outage"
    )
    all_mask_diagnostic_semantics = (
        "all outage masks are diagnostic under masked nominal controls only; no selected outage branch recovery controls are optimized"
        if selected.size == 0
        else "all outage masks are evaluated after optimization; selected masks use optimized branch recovery controls and unselected masks use masked nominal controls only"
    )
    return {
        **evaluation,
        "method_type": collocation_method_type(method),
        "collocation_method": method,
        "collocation_scheme_semantics": (
            "trapezoidal direct transcription with segment-constant controls"
            if method == "trapezoidal"
            else "constant-control Hermite-Simpson direct transcription; controls are held constant over each segment and no independent midpoint control variables are optimized"
        ),
        "selected_branch_semantics": selected_branch_semantics,
        "all_mask_diagnostic_semantics": all_mask_diagnostic_semantics,
        "control_bound_semantics": "all nominal and branch controls are Euclidean projected to ||u_i|| <= amax inside residual evaluation, RK4 reporting propagation, fuel computation, and output diagnostics; scalar optimizer bounds are finite guards, not the scientific bound",
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "runtime_seconds": float(time.perf_counter() - start),
        "selected_outage_indices": selected.astype(int).tolist(),
        "weights": problem.weights.as_dict(),
    }
