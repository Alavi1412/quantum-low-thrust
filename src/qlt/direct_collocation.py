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
    "hermite_simpson_midpoint": "bounded_projected_independent_midpoint_control_hermite_simpson_direct_collocation",
}
COLLOCATION_METHOD_ALIASES = {
    "hermite_simpson_independent": "hermite_simpson_midpoint",
    "independent_hermite_simpson": "hermite_simpson_midpoint",
    "independent_hs": "hermite_simpson_midpoint",
    "hs_independent": "hermite_simpson_midpoint",
    "hs_midpoint": "hermite_simpson_midpoint",
}


def normalize_collocation_method(method: str | None) -> str:
    normalized = str(method or "trapezoidal").lower().replace("-", "_")
    normalized = COLLOCATION_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in COLLOCATION_METHOD_TYPES:
        allowed = ", ".join(sorted(COLLOCATION_METHOD_TYPES))
        raise ValueError(f"direct-collocation method must be one of: {allowed}")
    return normalized


def collocation_method_type(method: str | None) -> str:
    return COLLOCATION_METHOD_TYPES[normalize_collocation_method(method)]


def collocation_method_uses_midpoint_controls(method: str | None) -> bool:
    return normalize_collocation_method(method) == "hermite_simpson_midpoint"


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


def quadratic_midpoint_control(endpoint_control: np.ndarray, midpoint_control: np.ndarray, tau: float) -> np.ndarray:
    """Control interpolation with u(0)=u(1)=endpoint and u(0.5)=midpoint."""
    tau = float(tau)
    endpoint = np.asarray(endpoint_control, dtype=float)
    midpoint = np.asarray(midpoint_control, dtype=float)
    return endpoint + 4.0 * tau * (1.0 - tau) * (midpoint - endpoint)


def _rk4_step_quadratic_control(
    state: np.ndarray,
    mu: float,
    dt: float,
    endpoint_control: np.ndarray,
    midpoint_control: np.ndarray,
    tau0: float,
    tau1: float,
) -> np.ndarray:
    tau_mid = 0.5 * (float(tau0) + float(tau1))
    u1 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau0)
    u2 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau_mid)
    u4 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau1)
    k1 = cr3bp_derivative(state, mu, u1)
    k2 = cr3bp_derivative(state + 0.5 * dt * k1, mu, u2)
    k3 = cr3bp_derivative(state + 0.5 * dt * k2, mu, u2)
    k4 = cr3bp_derivative(state + dt * k3, mu, u4)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_piecewise_controls(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    tf: float,
    substeps_per_segment: int,
    *,
    midpoint_controls: np.ndarray | None = None,
    return_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    controls = project_controls_to_ball(np.asarray(controls, dtype=float), np.inf)
    if midpoint_controls is not None:
        midpoint_controls = project_controls_to_ball(np.asarray(midpoint_controls, dtype=float), np.inf)
        if midpoint_controls.shape != controls.shape:
            raise ValueError(
                f"midpoint_controls shape {midpoint_controls.shape} does not match controls shape {controls.shape}"
            )
    state = np.asarray(state0, dtype=float).copy()
    n_segments = int(controls.shape[0])
    if n_segments == 0:
        nodes = np.asarray([state.copy()]) if return_nodes else None
        return state, nodes
    h = float(tf) / float(n_segments)
    dt = h / float(substeps_per_segment)
    nodes = [state.copy()] if return_nodes else None
    steps = int(substeps_per_segment)
    for segment_index, control in enumerate(controls):
        if midpoint_controls is None:
            for _ in range(steps):
                state = rk4_step(state, mu, dt, control)
        else:
            midpoint_control = midpoint_controls[segment_index]
            for step_index in range(steps):
                tau0 = step_index / float(steps)
                tau1 = (step_index + 1) / float(steps)
                state = _rk4_step_quadratic_control(state, mu, dt, control, midpoint_control, tau0, tau1)
        if return_nodes:
            nodes.append(state.copy())
    return state, np.asarray(nodes) if return_nodes else None


def propagate_prefix(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    segment_dt: float,
    substeps_per_segment: int,
    *,
    midpoint_controls: np.ndarray | None = None,
) -> np.ndarray:
    state = np.asarray(state0, dtype=float).copy()
    controls = np.asarray(controls, dtype=float)
    if midpoint_controls is not None:
        midpoint_controls = np.asarray(midpoint_controls, dtype=float)
        if midpoint_controls.shape != controls.shape:
            raise ValueError(
                f"midpoint_controls shape {midpoint_controls.shape} does not match controls shape {controls.shape}"
            )
    if len(controls) == 0:
        return state
    dt = float(segment_dt) / float(substeps_per_segment)
    steps = int(substeps_per_segment)
    for segment_index, control in enumerate(controls):
        if midpoint_controls is None:
            for _ in range(steps):
                state = rk4_step(state, mu, dt, control)
        else:
            midpoint_control = midpoint_controls[segment_index]
            for step_index in range(steps):
                tau0 = step_index / float(steps)
                tau1 = (step_index + 1) / float(steps)
                state = _rk4_step_quadratic_control(state, mu, dt, control, midpoint_control, tau0, tau1)
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


def hermite_simpson_midpoint_defect(
    left: np.ndarray,
    right: np.ndarray,
    endpoint_control: np.ndarray,
    midpoint_control: np.ndarray,
    cfg: ObjectiveConfig,
) -> np.ndarray:
    h = float(cfg.tf) / float(cfg.n_segments)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    endpoint_control = np.asarray(endpoint_control, dtype=float)
    midpoint_control = np.asarray(midpoint_control, dtype=float)
    f_left = cr3bp_derivative(left, cfg.mu, endpoint_control)
    f_right = cr3bp_derivative(right, cfg.mu, endpoint_control)
    x_mid = 0.5 * (left + right) + h / 8.0 * (f_left - f_right)
    f_mid = cr3bp_derivative(x_mid, cfg.mu, midpoint_control)
    return right - left - h / 6.0 * (f_left + 4.0 * f_mid + f_right)


def collocation_defect(
    left: np.ndarray,
    right: np.ndarray,
    control: np.ndarray,
    cfg: ObjectiveConfig,
    method: str,
    *,
    midpoint_control: np.ndarray | None = None,
) -> np.ndarray:
    normalized = normalize_collocation_method(method)
    if normalized == "trapezoidal":
        return trapezoidal_defect(left, right, control, cfg)
    if normalized == "hermite_simpson_midpoint":
        if midpoint_control is None:
            midpoint_control = control
        return hermite_simpson_midpoint_defect(left, right, control, midpoint_control, cfg)
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


@dataclass(frozen=True)
class DirectCollocationDecision:
    nominal_nodes: np.ndarray
    nominal_controls: np.ndarray
    nominal_midpoint_controls: np.ndarray | None
    branch_nodes: list[np.ndarray]
    branch_controls: list[np.ndarray]
    branch_midpoint_controls: list[np.ndarray | None]


class DirectCollocationLayout:
    def __init__(self, cfg: ObjectiveConfig, selected_masks: np.ndarray, method: str | None = "trapezoidal"):
        self.cfg = cfg
        self.method = normalize_collocation_method(method)
        self.has_midpoint_controls = collocation_method_uses_midpoint_controls(self.method)
        self.selected_masks = np.asarray(selected_masks, dtype=float)
        self.starts = [outage_end(mask) for mask in self.selected_masks]
        cursor = 0
        self.nominal_nodes = slice(cursor, cursor + (cfg.n_segments + 1) * 6)
        cursor = self.nominal_nodes.stop
        self.nominal_controls = slice(cursor, cursor + cfg.n_segments * 3)
        cursor = self.nominal_controls.stop
        self.nominal_midpoint_controls: slice | None = None
        if self.has_midpoint_controls:
            self.nominal_midpoint_controls = slice(cursor, cursor + cfg.n_segments * 3)
            cursor = self.nominal_midpoint_controls.stop
        self.branch_nodes: list[slice] = []
        self.branch_controls: list[slice] = []
        self.branch_midpoint_controls: list[slice | None] = []
        for start in self.starts:
            node_count = cfg.n_segments - start + 1
            control_count = cfg.n_segments - start
            self.branch_nodes.append(slice(cursor, cursor + node_count * 6))
            cursor = self.branch_nodes[-1].stop
            self.branch_controls.append(slice(cursor, cursor + control_count * 3))
            cursor = self.branch_controls[-1].stop
            if self.has_midpoint_controls:
                self.branch_midpoint_controls.append(slice(cursor, cursor + control_count * 3))
                cursor = self.branch_midpoint_controls[-1].stop
            else:
                self.branch_midpoint_controls.append(None)
        self.size = cursor

    def unpack_decision(self, vec: np.ndarray) -> DirectCollocationDecision:
        cfg = self.cfg
        nominal_nodes = np.asarray(vec[self.nominal_nodes], dtype=float).reshape((cfg.n_segments + 1, 6))
        nominal_controls = project_controls_to_ball(
            np.asarray(vec[self.nominal_controls], dtype=float).reshape((cfg.n_segments, 3)),
            cfg.amax,
        )
        nominal_midpoint_controls = None
        if self.nominal_midpoint_controls is not None:
            nominal_midpoint_controls = project_controls_to_ball(
                np.asarray(vec[self.nominal_midpoint_controls], dtype=float).reshape((cfg.n_segments, 3)),
                cfg.amax,
            )
        branch_nodes = []
        branch_controls = []
        branch_midpoint_controls: list[np.ndarray | None] = []
        for start, node_slice, control_slice, midpoint_slice in zip(
            self.starts,
            self.branch_nodes,
            self.branch_controls,
            self.branch_midpoint_controls,
        ):
            branch_nodes.append(np.asarray(vec[node_slice], dtype=float).reshape((cfg.n_segments - start + 1, 6)))
            control_count = cfg.n_segments - start
            if control_count:
                controls = np.asarray(vec[control_slice], dtype=float).reshape((control_count, 3))
            else:
                controls = np.zeros((0, 3), dtype=float)
            branch_controls.append(project_controls_to_ball(controls, cfg.amax))
            if midpoint_slice is not None:
                if control_count:
                    midpoint_controls = np.asarray(vec[midpoint_slice], dtype=float).reshape((control_count, 3))
                else:
                    midpoint_controls = np.zeros((0, 3), dtype=float)
                branch_midpoint_controls.append(project_controls_to_ball(midpoint_controls, cfg.amax))
            else:
                branch_midpoint_controls.append(None)
        return DirectCollocationDecision(
            nominal_nodes=nominal_nodes,
            nominal_controls=nominal_controls,
            nominal_midpoint_controls=nominal_midpoint_controls,
            branch_nodes=branch_nodes,
            branch_controls=branch_controls,
            branch_midpoint_controls=branch_midpoint_controls,
        )

    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        decision = self.unpack_decision(vec)
        return decision.nominal_nodes, decision.nominal_controls, decision.branch_nodes, decision.branch_controls

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.size, -np.inf, dtype=float)
        upper = np.full(self.size, np.inf, dtype=float)
        midpoint_slices = []
        if self.nominal_midpoint_controls is not None:
            midpoint_slices.append(self.nominal_midpoint_controls)
        midpoint_slices.extend(item for item in self.branch_midpoint_controls if item is not None)
        for control_slice in [self.nominal_controls, *self.branch_controls, *midpoint_slices]:
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
    nominal_control_guess: np.ndarray | None = None,
    nominal_midpoint_control_guess: np.ndarray | None = None,
    selected_branch_control_guesses: list[np.ndarray] | None = None,
    selected_branch_midpoint_control_guesses: list[np.ndarray] | None = None,
    method: str | None = "trapezoidal",
    node_initialization: str = "blend",
    node_initialization_blend: float = 0.35,
) -> tuple[DirectCollocationLayout, np.ndarray]:
    """Build the direct-collocation decision vector.

    Optional warm starts mirror :mod:`qlt.multiple_shooting`:

    - ``nominal_control_guess`` is a full ``(n_segments, 3)`` acceleration
      schedule, Euclidean projected to ``||u_i|| <= amax``.
    - ``nominal_midpoint_control_guess`` optionally seeds the independent
      midpoint controls for ``hermite_simpson_midpoint``.  When omitted,
      midpoint controls are initialized from the endpoint controls.
    - ``selected_branch_control_guesses`` is an optional list of full-length
      ``(n_segments, 3)`` schedules whose post-outage tail seeds each selected
      branch's recovery controls.
    - ``selected_branch_midpoint_control_guesses`` does the same for independent
      midpoint branch controls when ``method`` uses midpoint controls.

    When omitted the historical feedback-rollout guess is used unchanged.
    """
    layout = DirectCollocationLayout(cfg, selected_masks, method=method)
    schedule = np.ones(cfg.n_segments, dtype=int)
    if nominal_control_guess is None:
        nominal_controls = feedback_controls_for_schedule(schedule, state0, target, cfg)
    else:
        nominal_controls = np.asarray(nominal_control_guess, dtype=float)
        if nominal_controls.shape != (cfg.n_segments, 3):
            raise ValueError(
                f"nominal_control_guess shape {nominal_controls.shape} does not match {(cfg.n_segments, 3)}"
            )
        nominal_controls = project_controls_to_ball(nominal_controls, cfg.amax)
    nominal_midpoint_controls = None
    if layout.has_midpoint_controls:
        if nominal_midpoint_control_guess is None:
            nominal_midpoint_controls = nominal_controls.copy()
        else:
            nominal_midpoint_controls = np.asarray(nominal_midpoint_control_guess, dtype=float)
            if nominal_midpoint_controls.shape != (cfg.n_segments, 3):
                raise ValueError(
                    "nominal_midpoint_control_guess shape "
                    f"{nominal_midpoint_controls.shape} does not match {(cfg.n_segments, 3)}"
                )
            nominal_midpoint_controls = project_controls_to_ball(nominal_midpoint_controls, cfg.amax)
    _, rollout_nodes = propagate_piecewise_controls(
        state0,
        nominal_controls,
        cfg.mu,
        cfg.tf,
        cfg.substeps,
        midpoint_controls=nominal_midpoint_controls,
        return_nodes=True,
    )
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

    branch_guesses = list(selected_branch_control_guesses or [])
    for guess in branch_guesses:
        guess_arr = np.asarray(guess, dtype=float)
        if guess_arr.shape != (cfg.n_segments, 3):
            raise ValueError(
                f"selected_branch_control_guess shape {guess_arr.shape} does not match {(cfg.n_segments, 3)}"
            )
    branch_midpoint_guesses = list(selected_branch_midpoint_control_guesses or [])
    for guess in branch_midpoint_guesses:
        if guess is None:
            continue
        guess_arr = np.asarray(guess, dtype=float)
        if guess_arr.shape != (cfg.n_segments, 3):
            raise ValueError(
                "selected_branch_midpoint_control_guess shape "
                f"{guess_arr.shape} does not match {(cfg.n_segments, 3)}"
            )

    chunks = [nominal_nodes.reshape(-1), nominal_controls.reshape(-1)]
    if nominal_midpoint_controls is not None:
        chunks.append(nominal_midpoint_controls.reshape(-1))
    h = float(cfg.tf) / float(cfg.n_segments)
    for branch_index, (mask, start) in enumerate(zip(np.asarray(selected_masks, dtype=float), layout.starts)):
        prefix_controls = nominal_controls[:start] * mask[:start, None]
        prefix_midpoint_controls = (
            nominal_midpoint_controls[:start] * mask[:start, None]
            if nominal_midpoint_controls is not None
            else None
        )
        branch_start = propagate_prefix(
            state0,
            prefix_controls,
            cfg.mu,
            h,
            cfg.substeps,
            midpoint_controls=prefix_midpoint_controls,
        )
        branch_nodes = _linear_nodes(branch_start, target, cfg.n_segments - start + 1)
        if branch_index < len(branch_guesses):
            branch_full = project_controls_to_ball(np.asarray(branch_guesses[branch_index], dtype=float), cfg.amax)
            branch_controls = branch_full[start:].copy()
        else:
            branch_controls = nominal_controls[start:].copy()
        branch_midpoint_controls = None
        if layout.has_midpoint_controls:
            if branch_index < len(branch_midpoint_guesses) and branch_midpoint_guesses[branch_index] is not None:
                midpoint_full = project_controls_to_ball(
                    np.asarray(branch_midpoint_guesses[branch_index], dtype=float),
                    cfg.amax,
                )
                branch_midpoint_controls = midpoint_full[start:].copy()
            else:
                branch_midpoint_controls = branch_controls.copy()
        chunks.append(branch_nodes.reshape(-1))
        chunks.append(branch_controls.reshape(-1))
        if branch_midpoint_controls is not None:
            chunks.append(branch_midpoint_controls.reshape(-1))
    return layout, np.concatenate(chunks)


def branch_full_controls(nominal_controls: np.ndarray, mask: np.ndarray, start: int, recovery_controls: np.ndarray, amax: float) -> np.ndarray:
    controls = np.asarray(nominal_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    if int(start) < controls.shape[0]:
        controls[int(start) :] = np.asarray(recovery_controls, dtype=float)
    return project_controls_to_ball(controls, amax)


def branch_full_midpoint_controls(
    nominal_midpoint_controls: np.ndarray | None,
    mask: np.ndarray,
    start: int,
    recovery_midpoint_controls: np.ndarray | None,
    amax: float,
) -> np.ndarray | None:
    if nominal_midpoint_controls is None:
        return None
    controls = np.asarray(nominal_midpoint_controls, dtype=float).copy() * np.asarray(mask, dtype=float)[:, None]
    if int(start) < controls.shape[0] and recovery_midpoint_controls is not None:
        controls[int(start) :] = np.asarray(recovery_midpoint_controls, dtype=float)
    return project_controls_to_ball(controls, amax)


def simpson_control_fuel(controls: np.ndarray, midpoint_controls: np.ndarray | None, tf: float) -> float:
    controls = np.asarray(controls, dtype=float)
    if midpoint_controls is None:
        return control_fuel(controls, tf)
    midpoint_controls = np.asarray(midpoint_controls, dtype=float)
    if controls.size == 0:
        return 0.0
    if midpoint_controls.shape != controls.shape:
        raise ValueError(
            f"midpoint_controls shape {midpoint_controls.shape} does not match controls shape {controls.shape}"
        )
    h = float(tf) / float(controls.shape[0])
    endpoint_norms = np.linalg.norm(controls, axis=-1)
    midpoint_norms = np.linalg.norm(midpoint_controls, axis=-1)
    return float((h / 6.0) * np.sum(endpoint_norms + 4.0 * midpoint_norms + endpoint_norms))


def _control_arrays_for_diagnostics(
    controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
    branch_controls: list[np.ndarray],
    branch_midpoint_controls: list[np.ndarray | None],
) -> list[np.ndarray]:
    arrays = [controls, *branch_controls]
    if midpoint_controls is not None:
        arrays.append(midpoint_controls)
    arrays.extend(item for item in branch_midpoint_controls if item is not None)
    return arrays


def _append_control_regularization(
    chunks: list[np.ndarray],
    *,
    controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
    cfg: ObjectiveConfig,
    weights: DirectCollocationWeights,
) -> None:
    scale = max(cfg.amax, 1e-12)
    if controls.size:
        chunks.append(weights.control * controls.reshape(-1) / scale)
    if controls.shape[0] > 1:
        chunks.append(weights.smooth * np.diff(controls, axis=0).reshape(-1) / scale)
    if midpoint_controls is not None and midpoint_controls.size:
        chunks.append(weights.control * midpoint_controls.reshape(-1) / scale)
        if midpoint_controls.shape[0] > 1:
            chunks.append(weights.smooth * np.diff(midpoint_controls, axis=0).reshape(-1) / scale)


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
        if collocation_method_uses_midpoint_controls(self.method) and not self.layout.has_midpoint_controls:
            raise ValueError("hermite_simpson_midpoint requires a DirectCollocationLayout with midpoint controls")

    def residual(self, vec: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        decision = self.layout.unpack_decision(vec)
        nodes = decision.nominal_nodes
        controls = decision.nominal_controls
        midpoint_controls = decision.nominal_midpoint_controls
        branch_nodes = decision.branch_nodes
        branch_controls = decision.branch_controls
        branch_midpoint_controls = decision.branch_midpoint_controls
        scale = np.maximum(scale_vector(cfg), 1e-12)
        chunks = [
            self.weights.initial * scaled_state_residual(nodes[0], self.state0, cfg),
        ]
        for i in range(cfg.n_segments):
            midpoint_control = None if midpoint_controls is None else midpoint_controls[i]
            chunks.append(
                self.weights.defect
                * collocation_defect(
                    nodes[i],
                    nodes[i + 1],
                    controls[i],
                    cfg,
                    self.method,
                    midpoint_control=midpoint_control,
                )
                / scale
            )
        chunks.append(self.weights.terminal * scaled_state_residual(nodes[-1], self.target, cfg))
        _append_control_regularization(
            chunks,
            controls=controls,
            midpoint_controls=midpoint_controls,
            cfg=cfg,
            weights=self.weights,
        )

        h = float(cfg.tf) / float(cfg.n_segments)
        for mask_index, mask, start, nodes_b, controls_b, midpoint_controls_b in zip(
            self.selected,
            self.layout.selected_masks,
            self.layout.starts,
            branch_nodes,
            branch_controls,
            branch_midpoint_controls,
        ):
            del mask_index
            prefix_midpoint_controls = (
                midpoint_controls[:start] * mask[:start, None]
                if midpoint_controls is not None
                else None
            )
            branch_start = propagate_prefix(
                self.state0,
                controls[:start] * mask[:start, None],
                cfg.mu,
                h,
                cfg.substeps,
                midpoint_controls=prefix_midpoint_controls,
            )
            chunks.append(self.weights.branch_start * scaled_state_residual(nodes_b[0], branch_start, cfg))
            for j in range(cfg.n_segments - start):
                midpoint_control_b = None if midpoint_controls_b is None else midpoint_controls_b[j]
                chunks.append(
                    self.weights.branch_defect
                    * collocation_defect(
                        nodes_b[j],
                        nodes_b[j + 1],
                        controls_b[j],
                        cfg,
                        self.method,
                        midpoint_control=midpoint_control_b,
                    )
                    / scale
                )
            chunks.append(self.weights.branch_terminal * scaled_state_residual(nodes_b[-1], self.target, cfg))
            _append_control_regularization(
                chunks,
                controls=controls_b,
                midpoint_controls=midpoint_controls_b,
                cfg=cfg,
                weights=self.weights,
            )
        return np.concatenate(chunks)

    def evaluate_vector(self, vec: np.ndarray, thresholds: dict) -> dict:
        cfg = self.cfg
        decision = self.layout.unpack_decision(vec)
        nominal_controls = decision.nominal_controls
        nominal_midpoint_controls = decision.nominal_midpoint_controls
        selected_branch_controls = decision.branch_controls
        selected_branch_midpoint_controls = decision.branch_midpoint_controls
        nominal_final, nominal_history = propagate_piecewise_controls(
            self.state0,
            nominal_controls,
            cfg.mu,
            cfg.tf,
            cfg.substeps,
            midpoint_controls=nominal_midpoint_controls,
            return_nodes=True,
        )
        nominal_error = float(state_error(nominal_final, self.target, cfg.position_scale, cfg.velocity_scale))

        selected_lookup = {
            int(mask_index): (
                branch_full_controls(nominal_controls, mask, start, controls, cfg.amax),
                branch_full_midpoint_controls(
                    nominal_midpoint_controls,
                    mask,
                    start,
                    midpoint_controls_b,
                    cfg.amax,
                ),
            )
            for mask_index, mask, start, controls, midpoint_controls_b in zip(
                self.selected,
                self.layout.selected_masks,
                self.layout.starts,
                selected_branch_controls,
                selected_branch_midpoint_controls,
            )
        }
        all_errors: list[float] = []
        all_fuels: list[float] = []
        selected_errors: list[float] = []
        selected_fuels: list[float] = []
        for mask_index, mask in enumerate(self.masks):
            if mask_index in selected_lookup:
                controls, midpoint_controls = selected_lookup[mask_index]
            else:
                controls = project_controls_to_ball(nominal_controls * mask[:, None], cfg.amax)
                midpoint_controls = (
                    project_controls_to_ball(nominal_midpoint_controls * mask[:, None], cfg.amax)
                    if nominal_midpoint_controls is not None
                    else None
                )
            final, _ = propagate_piecewise_controls(
                self.state0,
                controls,
                cfg.mu,
                cfg.tf,
                cfg.substeps,
                midpoint_controls=midpoint_controls,
            )
            err = float(state_error(final, self.target, cfg.position_scale, cfg.velocity_scale))
            fuel = simpson_control_fuel(controls, midpoint_controls, cfg.tf)
            all_errors.append(err)
            all_fuels.append(fuel)
            if mask_index in set(int(v) for v in self.selected.tolist()):
                selected_errors.append(err)
                selected_fuels.append(fuel)

        selected_worst = float(np.max(selected_errors)) if selected_errors else nominal_error
        all_worst = float(np.max(all_errors)) if all_errors else selected_worst
        all_controls = _control_arrays_for_diagnostics(
            nominal_controls,
            nominal_midpoint_controls,
            selected_branch_controls,
            selected_branch_midpoint_controls,
        )
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
            "nominal_midpoint_controls": nominal_midpoint_controls,
            "selected_branch_controls": selected_branch_controls,
            "selected_branch_midpoint_controls": selected_branch_midpoint_controls,
            "nominal_history": nominal_history,
            "nominal_fuel": simpson_control_fuel(nominal_controls, nominal_midpoint_controls, cfg.tf),
            "recovery_fuel_mean": float(np.mean(selected_fuels)) if selected_fuels else simpson_control_fuel(nominal_controls, nominal_midpoint_controls, cfg.tf),
            "recovery_fuel_max": float(np.max(selected_fuels)) if selected_fuels else simpson_control_fuel(nominal_controls, nominal_midpoint_controls, cfg.tf),
            "all_mask_recovery_fuel_mean": float(np.mean(all_fuels)) if all_fuels else simpson_control_fuel(nominal_controls, nominal_midpoint_controls, cfg.tf),
            "fuel_quadrature": (
                "simpson_endpoint_midpoint_endpoint"
                if nominal_midpoint_controls is not None
                else "segment_rectangle_endpoint_controls"
            ),
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
    nominal_control_guess: np.ndarray | None = None,
    nominal_midpoint_control_guess: np.ndarray | None = None,
    selected_branch_control_guesses: list[np.ndarray] | None = None,
    selected_branch_midpoint_control_guesses: list[np.ndarray] | None = None,
    warm_start_info: dict | None = None,
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
        nominal_control_guess=nominal_control_guess,
        nominal_midpoint_control_guess=nominal_midpoint_control_guess,
        selected_branch_control_guesses=selected_branch_control_guesses,
        selected_branch_midpoint_control_guesses=selected_branch_midpoint_control_guesses,
        method=method,
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
    collocation_scheme_semantics = {
        "trapezoidal": "trapezoidal direct transcription with segment-constant controls",
        "hermite_simpson": (
            "constant-control Hermite-Simpson direct transcription; controls are held constant over each "
            "segment and no independent midpoint control variables are optimized"
        ),
        "hermite_simpson_midpoint": (
            "independent-midpoint-control Hermite-Simpson direct transcription; each segment optimizes an "
            "endpoint/segment control and a separate midpoint control, f_left/f_right use the endpoint control, "
            "f_mid uses the midpoint control, and reporting propagation uses RK4 substeps with quadratic "
            "control interpolation satisfying u(0)=u(1)=endpoint and u(0.5)=midpoint"
        ),
    }[method]
    control_bound_semantics = (
        "all nominal and branch endpoint controls are Euclidean projected to ||u_i|| <= amax inside residual "
        "evaluation, RK4 reporting propagation, fuel computation, and output diagnostics; scalar optimizer "
        "bounds are finite guards, not the scientific bound"
        if method != "hermite_simpson_midpoint"
        else "all nominal and branch endpoint and independent midpoint controls are Euclidean projected to "
        "||u_i|| <= amax inside residual evaluation, RK4 reporting propagation, Simpson-style fuel "
        "computation, and output diagnostics; scalar optimizer bounds are finite guards, not the scientific bound"
    )
    return {
        **evaluation,
        "method_type": collocation_method_type(method),
        "collocation_method": method,
        "collocation_scheme_semantics": collocation_scheme_semantics,
        "selected_branch_semantics": selected_branch_semantics,
        "all_mask_diagnostic_semantics": all_mask_diagnostic_semantics,
        "control_bound_semantics": control_bound_semantics,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "runtime_seconds": float(time.perf_counter() - start),
        "selected_outage_indices": selected.astype(int).tolist(),
        "weights": problem.weights.as_dict(),
        "warm_start_info": dict(warm_start_info or {}),
    }
