from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cr3bp import cr3bp_derivative
from .direct_collocation import quadratic_midpoint_control
from .refinement import project_controls_to_ball


DEFAULT_SUN_DISTANCE_LU = 389.17
DEFAULT_SUN_MU_RATIO = 328900.56
DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO = 0.0748


@dataclass(frozen=True)
class SolarTidalParameters:
    """Canonical nondimensional solar-tidal stress-probe parameters.

    The values are intentionally fixed defaults for a bicircular stress model in
    Earth-Moon canonical units. They are not ephemeris or SPICE state data.
    """

    sun_distance_lu: float = DEFAULT_SUN_DISTANCE_LU
    sun_mu_ratio: float = DEFAULT_SUN_MU_RATIO
    sun_inertial_angular_rate_ratio: float = DEFAULT_SUN_INERTIAL_ANGULAR_RATE_RATIO

    @property
    def rotating_frame_phase_rate(self) -> float:
        return float(self.sun_inertial_angular_rate_ratio) - 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "sun_distance_lu": float(self.sun_distance_lu),
            "sun_mu_ratio": float(self.sun_mu_ratio),
            "sun_inertial_angular_rate_ratio": float(self.sun_inertial_angular_rate_ratio),
            "rotating_frame_phase_rate": float(self.rotating_frame_phase_rate),
            "distance_semantics": "constant circular Sun distance in Earth-Moon distance units",
            "mu_semantics": "GM_sun/(GM_earth+GM_moon) nondimensional mass ratio",
            "rate_semantics": "Sun inertial mean-motion ratio to Earth-Moon mean motion; rotating-frame phase rate is n_sun - 1",
            "model_scope": "simple bicircular solar third-body tidal stress model; not SPICE ephemeris validation",
        }


def sun_position_rotating(
    t: float,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> np.ndarray:
    params = parameters or SolarTidalParameters()
    angle = float(phase_rad) + float(params.rotating_frame_phase_rate) * float(t)
    return float(params.sun_distance_lu) * np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)


def solar_tidal_acceleration(
    positions: np.ndarray,
    t: float,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> np.ndarray:
    params = parameters or SolarTidalParameters()
    r = np.asarray(positions, dtype=float)
    scalar = r.ndim == 1
    if scalar:
        r = r[None, :]
    if r.shape[1] != 3:
        raise ValueError(f"positions must have shape (..., 3), got {r.shape}")

    r_s = sun_position_rotating(t, phase_rad=phase_rad, parameters=params)
    rel = r_s[None, :] - r
    rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
    sun_norm = max(float(np.linalg.norm(r_s)), 1e-12)
    accel = float(params.sun_mu_ratio) * (rel / rel_norm**3 - r_s[None, :] / sun_norm**3)
    return accel[0] if scalar else accel


def bicircular_derivative(
    t: float,
    states: np.ndarray,
    mu: float,
    *,
    control_accel: np.ndarray | None = None,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> np.ndarray:
    s = np.asarray(states, dtype=float)
    scalar = s.ndim == 1
    working = s[None, :] if scalar else s
    if working.shape[1] != 6:
        raise ValueError(f"states must have shape (..., 6), got {working.shape}")

    solar_accel = solar_tidal_acceleration(
        working[:, :3],
        t,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    if control_accel is None:
        accel = solar_accel
    else:
        control = np.asarray(control_accel, dtype=float)
        if control.ndim == 1:
            control = control[None, :]
        accel = solar_accel + control
    out = cr3bp_derivative(working, mu, accel)
    return out[0] if scalar else out


def rk4_bicircular_step(
    states: np.ndarray,
    mu: float,
    t: float,
    dt: float,
    *,
    control_accel: np.ndarray | None = None,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> np.ndarray:
    s = np.asarray(states, dtype=float)
    k1 = bicircular_derivative(
        t,
        s,
        mu,
        control_accel=control_accel,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k2 = bicircular_derivative(
        t + 0.5 * dt,
        s + 0.5 * dt * k1,
        mu,
        control_accel=control_accel,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k3 = bicircular_derivative(
        t + 0.5 * dt,
        s + 0.5 * dt * k2,
        mu,
        control_accel=control_accel,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k4 = bicircular_derivative(
        t + dt,
        s + dt * k3,
        mu,
        control_accel=control_accel,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_bicircular_step_quadratic_control(
    state: np.ndarray,
    mu: float,
    t: float,
    dt: float,
    endpoint_control: np.ndarray,
    midpoint_control: np.ndarray,
    tau0: float,
    tau1: float,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
) -> np.ndarray:
    tau_mid = 0.5 * (float(tau0) + float(tau1))
    u1 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau0)
    u2 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau_mid)
    u4 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau1)
    k1 = bicircular_derivative(
        t,
        state,
        mu,
        control_accel=u1,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k2 = bicircular_derivative(
        t + 0.5 * dt,
        state + 0.5 * dt * k1,
        mu,
        control_accel=u2,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k3 = bicircular_derivative(
        t + 0.5 * dt,
        state + 0.5 * dt * k2,
        mu,
        control_accel=u2,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    k4 = bicircular_derivative(
        t + dt,
        state + dt * k3,
        mu,
        control_accel=u4,
        phase_rad=phase_rad,
        parameters=parameters,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_piecewise_controls_bicircular(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    tf: float,
    substeps_per_segment: int,
    *,
    midpoint_controls: np.ndarray | None = None,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    return_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Propagate endpoint/midpoint controls under the simple bicircular stress model.

    This mirrors :func:`qlt.direct_collocation.propagate_piecewise_controls`
    for endpoint controls and optional independent midpoint controls, but uses
    :func:`bicircular_derivative` with the circular solar-tidal perturbation.
    It is a deterministic stress-probe propagator, not SPICE ephemeris or
    high-fidelity force-model validation.
    """

    controls = project_controls_to_ball(np.asarray(controls, dtype=float), np.inf)
    if controls.ndim != 2 or controls.shape[1] != 3:
        raise ValueError(f"controls must have shape (segments, 3), got {controls.shape}")
    if midpoint_controls is not None:
        midpoint_controls = project_controls_to_ball(np.asarray(midpoint_controls, dtype=float), np.inf)
        if midpoint_controls.shape != controls.shape:
            raise ValueError(
                f"midpoint_controls shape {midpoint_controls.shape} does not match controls shape {controls.shape}"
            )
    if int(substeps_per_segment) <= 0:
        raise ValueError("substeps_per_segment must be positive")

    state = np.asarray(state0, dtype=float).copy()
    n_segments = int(controls.shape[0])
    if n_segments == 0:
        nodes = np.asarray([state.copy()]) if return_nodes else None
        return state, nodes

    h = float(tf) / float(n_segments)
    dt = h / float(substeps_per_segment)
    steps = int(substeps_per_segment)
    t = 0.0
    nodes = [state.copy()] if return_nodes else None
    params = parameters or SolarTidalParameters()

    for segment_index, control in enumerate(controls):
        if midpoint_controls is None:
            for _ in range(steps):
                state = rk4_bicircular_step(
                    state,
                    mu,
                    t,
                    dt,
                    control_accel=control,
                    phase_rad=phase_rad,
                    parameters=params,
                )
                t += dt
        else:
            midpoint_control = midpoint_controls[segment_index]
            for step_index in range(steps):
                tau0 = step_index / float(steps)
                tau1 = (step_index + 1) / float(steps)
                state = _rk4_bicircular_step_quadratic_control(
                    state,
                    mu,
                    t,
                    dt,
                    control,
                    midpoint_control,
                    tau0,
                    tau1,
                    phase_rad=phase_rad,
                    parameters=params,
                )
                t += dt
        if return_nodes:
            nodes.append(state.copy())
    return state, np.asarray(nodes) if return_nodes else None


def propagate_controls_batch_bicircular(
    state0: np.ndarray,
    control_schedules: np.ndarray,
    mu: float,
    tf: float,
    substeps_per_segment: int,
    *,
    phase_rad: float = 0.0,
    parameters: SolarTidalParameters | None = None,
    return_history: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    controls = np.asarray(control_schedules, dtype=float)
    if controls.ndim == 2:
        controls = controls[None, :, :]
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError(f"control_schedules must have shape (..., segments, 3), got {controls.shape}")
    if int(substeps_per_segment) <= 0:
        raise ValueError("substeps_per_segment must be positive")

    m, n, _ = controls.shape
    states = np.repeat(np.asarray(state0, dtype=float)[None, :], m, axis=0)
    dt = float(tf) / float(n * int(substeps_per_segment))
    t = 0.0
    history = [states.copy()] if return_history else None
    params = parameters or SolarTidalParameters()

    for i in range(n):
        accel = controls[:, i, :]
        for _ in range(int(substeps_per_segment)):
            states = rk4_bicircular_step(
                states,
                mu,
                t,
                dt,
                control_accel=accel,
                phase_rad=phase_rad,
                parameters=params,
            )
            t += dt
            if return_history:
                history.append(states.copy())
    if return_history:
        return states, np.asarray(history)
    return states, None
