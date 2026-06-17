from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from .cr3bp import cr3bp_derivative
from .direct_collocation import quadratic_midpoint_control
from .refinement import project_controls_to_ball


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
DEFAULT_CACHE_START_JD_TDB = 2461041.5
DEFAULT_CANONICAL_TIME_UNIT_SECONDS = 375190.259
DEFAULT_REFERENCE_DISTANCE_KM = 384400.0


@dataclass(frozen=True)
class HorizonsVectorSample:
    jd_tdb: float
    calendar_tdb: str
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    light_time_s: float
    range_km: float
    range_rate_km_s: float


def canonical_node_times(tf: float, n_segments: int) -> np.ndarray:
    if int(n_segments) <= 0:
        raise ValueError("n_segments must be positive")
    return np.linspace(0.0, float(tf), int(n_segments) + 1)


def canonical_node_jds(
    *,
    start_jd_tdb: float,
    tf: float,
    n_segments: int,
    canonical_time_unit_seconds: float = DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
) -> list[float]:
    nodes = canonical_node_times(tf, n_segments)
    return [
        float(start_jd_tdb) + float(t) * float(canonical_time_unit_seconds) / 86400.0
        for t in nodes
    ]


def horizons_vectors_query_params(command: str, center: str, jd_tdb: Iterable[float]) -> dict[str, str]:
    return {
        "format": "json",
        "COMMAND": str(command),
        "CENTER": str(center),
        "EPHEM_TYPE": "VECTORS",
        "TLIST": "\n".join(f"{float(jd):.9f}" for jd in jd_tdb),
        "VEC_TABLE": "3",
        "OBJ_DATA": "NO",
        "CSV_FORMAT": "YES",
    }


def horizons_query_url(params: dict[str, str]) -> str:
    return HORIZONS_API_URL + "?" + urlencode(params)


def fetch_horizons_vectors(params: dict[str, str], *, timeout_seconds: float = 60.0) -> dict[str, object]:
    url = horizons_query_url(params)
    with urlopen(url, timeout=float(timeout_seconds)) as response:
        payload = response.read().decode("utf-8", errors="replace")
    data = json.loads(payload)
    if "error" in data:
        raise RuntimeError(f"Horizons API error: {data['error']}")
    if "result" not in data:
        raise RuntimeError("Horizons API response missing result field")
    return {
        "query_url": url,
        "request_params": dict(params),
        "response_signature": data.get("signature", {}),
        "result_sha256": hashlib.sha256(str(data["result"]).encode("utf-8")).hexdigest(),
        "result": data["result"],
    }


def parse_horizons_vectors_result(result: str) -> list[HorizonsVectorSample]:
    text = str(result)
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("Horizons result missing $$SOE/$$EOE vector block")
    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    samples: list[HorizonsVectorSample] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = next(csv.reader(StringIO(line), skipinitialspace=True))
        if len(row) < 11:
            raise ValueError(f"Horizons vector row has {len(row)} columns, expected at least 11: {line}")
        samples.append(
            HorizonsVectorSample(
                jd_tdb=float(row[0]),
                calendar_tdb=str(row[1]).strip(),
                position_km=np.asarray([float(row[2]), float(row[3]), float(row[4])], dtype=float),
                velocity_km_s=np.asarray([float(row[5]), float(row[6]), float(row[7])], dtype=float),
                light_time_s=float(row[8]),
                range_km=float(row[9]),
                range_rate_km_s=float(row[10]),
            )
        )
    if not samples:
        raise ValueError("Horizons vector block contains no samples")
    return samples


def sample_to_cache_row(sample: HorizonsVectorSample) -> dict[str, object]:
    return {
        "jd_tdb": float(sample.jd_tdb),
        "calendar_tdb": sample.calendar_tdb,
        "position_km": [float(v) for v in sample.position_km],
        "velocity_km_s": [float(v) for v in sample.velocity_km_s],
        "light_time_s": float(sample.light_time_s),
        "range_km": float(sample.range_km),
        "range_rate_km_s": float(sample.range_rate_km_s),
    }


def sample_from_cache_row(row: dict[str, object]) -> HorizonsVectorSample:
    return HorizonsVectorSample(
        jd_tdb=float(row["jd_tdb"]),
        calendar_tdb=str(row["calendar_tdb"]),
        position_km=np.asarray(row["position_km"], dtype=float),
        velocity_km_s=np.asarray(row["velocity_km_s"], dtype=float),
        light_time_s=float(row["light_time_s"]),
        range_km=float(row["range_km"]),
        range_rate_km_s=float(row["range_rate_km_s"]),
    )


def write_horizons_cache(
    *,
    path: Path,
    moon_response: dict[str, object],
    sun_response: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    moon_samples = parse_horizons_vectors_result(str(moon_response["result"]))
    sun_samples = parse_horizons_vectors_result(str(sun_response["result"]))
    cache = {
        "schema_version": 1,
        "cache_type": "jpl_horizons_geocentric_moon_sun_vectors",
        "metadata": metadata,
        "bodies": {
            "moon_geocentric": {
                **{k: v for k, v in moon_response.items() if k != "result"},
                "raw_result": moon_response["result"],
                "samples": [sample_to_cache_row(sample) for sample in moon_samples],
            },
            "sun_geocentric": {
                **{k: v for k, v in sun_response.items() if k != "result"},
                "raw_result": sun_response["result"],
                "samples": [sample_to_cache_row(sample) for sample in sun_samples],
            },
        },
        "interpretation_limits": [
            "Cached JPL Horizons vectors are used for an offline force-model contrast only.",
            "The cache does not define a mission epoch or flight-validation data product.",
            "The contrast is not SPICE validation, high-fidelity trajectory replay, or accepted-control retuning.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return cache


def load_horizons_cache(path: Path) -> dict[str, object]:
    cache = json.loads(path.read_text(encoding="utf-8"))
    validate_horizons_cache(cache)
    return cache


def samples_from_cache(cache: dict[str, object], body: str) -> list[HorizonsVectorSample]:
    bodies = cache.get("bodies", {})
    if not isinstance(bodies, dict) or body not in bodies:
        raise ValueError(f"Horizons cache missing body {body!r}")
    body_data = bodies[body]
    if not isinstance(body_data, dict):
        raise ValueError(f"Horizons cache body {body!r} is not an object")
    raw_samples = body_data.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"Horizons cache body {body!r} missing samples")
    samples = [sample_from_cache_row(row) for row in raw_samples]
    raw_result = body_data.get("raw_result")
    if raw_result is not None:
        reparsed = parse_horizons_vectors_result(str(raw_result))
        if len(reparsed) != len(samples):
            raise ValueError(f"Horizons cache raw/sample count mismatch for {body!r}")
        for parsed, stored in zip(reparsed, samples):
            if abs(parsed.jd_tdb - stored.jd_tdb) > 5.0e-10:
                raise ValueError(f"Horizons cache JD mismatch for {body!r}")
            if not np.allclose(parsed.position_km, stored.position_km, rtol=0.0, atol=1.0e-9):
                raise ValueError(f"Horizons cache position mismatch for {body!r}")
            if not np.allclose(parsed.velocity_km_s, stored.velocity_km_s, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"Horizons cache velocity mismatch for {body!r}")
    return samples


def validate_horizons_cache(cache: dict[str, object]) -> None:
    if int(cache.get("schema_version", 0)) != 1:
        raise ValueError("unsupported Horizons cache schema_version")
    bodies = cache.get("bodies")
    if not isinstance(bodies, dict):
        raise ValueError("Horizons cache missing bodies")
    for body in ("moon_geocentric", "sun_geocentric"):
        body_data = bodies.get(body)
        if not isinstance(body_data, dict):
            raise ValueError(f"Horizons cache missing body {body}")
        signature = body_data.get("response_signature", {})
        if not isinstance(signature, dict) or signature.get("source") != "NASA/JPL Horizons API":
            raise ValueError(f"Horizons cache {body} missing NASA/JPL Horizons API signature")
        if "query_url" not in body_data or "request_params" not in body_data:
            raise ValueError(f"Horizons cache {body} missing query metadata")
    moon = samples_from_cache(cache, "moon_geocentric")
    sun = samples_from_cache(cache, "sun_geocentric")
    if len(moon) != len(sun):
        raise ValueError("Moon/Sun Horizons caches have different sample counts")
    moon_jd = np.asarray([sample.jd_tdb for sample in moon], dtype=float)
    sun_jd = np.asarray([sample.jd_tdb for sample in sun], dtype=float)
    if not np.allclose(moon_jd, sun_jd, rtol=0.0, atol=5.0e-10):
        raise ValueError("Moon/Sun Horizons caches have different JD grids")


def _metadata_float(metadata: dict[str, object], key: str) -> float:
    if key not in metadata:
        raise ValueError(f"Horizons cache metadata missing {key}")
    try:
        return float(metadata[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Horizons cache metadata {key} is not numeric") from exc


def _metadata_int(metadata: dict[str, object], key: str) -> int:
    if key not in metadata:
        raise ValueError(f"Horizons cache metadata missing {key}")
    try:
        value = int(metadata[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Horizons cache metadata {key} is not an integer") from exc
    if float(metadata[key]) != float(value):
        raise ValueError(f"Horizons cache metadata {key} is not integral")
    return value


def _metadata_float_array(metadata: dict[str, object], key: str) -> np.ndarray:
    if key not in metadata:
        raise ValueError(f"Horizons cache metadata missing {key}")
    value = metadata[key]
    if not isinstance(value, list):
        raise ValueError(f"Horizons cache metadata {key} is not a list")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Horizons cache metadata {key} is not numeric") from exc


def _require_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=float(atol)):
        raise ValueError(
            f"Horizons cache metadata mismatch for {name}: "
            f"expected {float(expected):.17g}, got {float(actual):.17g}"
        )


def _require_allclose(name: str, actual: np.ndarray, expected: np.ndarray, *, atol: float) -> None:
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    if actual_arr.shape != expected_arr.shape:
        raise ValueError(
            f"Horizons cache metadata mismatch for {name}: "
            f"expected shape {expected_arr.shape}, got {actual_arr.shape}"
        )
    if not np.allclose(actual_arr, expected_arr, rtol=0.0, atol=float(atol)):
        max_delta = float(np.max(np.abs(actual_arr - expected_arr))) if actual_arr.size else 0.0
        raise ValueError(
            f"Horizons cache metadata mismatch for {name}: "
            f"max abs delta {max_delta:.17g} exceeds {float(atol):.17g}"
        )


def validate_horizons_cache_compatibility(
    cache: dict[str, object],
    *,
    start_jd_tdb: float,
    tf: float,
    n_segments: int,
    canonical_time_unit_seconds: float = DEFAULT_CANONICAL_TIME_UNIT_SECONDS,
    reference_distance_km: float = DEFAULT_REFERENCE_DISTANCE_KM,
) -> None:
    """Validate that a Horizons cache matches the active contrast configuration."""

    metadata = cache.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Horizons cache missing metadata")

    expected_nodes = canonical_node_times(float(tf), int(n_segments))
    expected_jds = np.asarray(
        canonical_node_jds(
            start_jd_tdb=float(start_jd_tdb),
            tf=float(tf),
            n_segments=int(n_segments),
            canonical_time_unit_seconds=float(canonical_time_unit_seconds),
        ),
        dtype=float,
    )

    _require_close("start_jd_tdb", _metadata_float(metadata, "start_jd_tdb"), float(start_jd_tdb), atol=5.0e-10)
    _require_close(
        "canonical_transfer_time",
        _metadata_float(metadata, "canonical_transfer_time"),
        float(tf),
        atol=1.0e-12,
    )
    actual_segments = _metadata_int(metadata, "segments")
    if actual_segments != int(n_segments):
        raise ValueError(
            f"Horizons cache metadata mismatch for segments: expected {int(n_segments)}, got {actual_segments}"
        )
    _require_close(
        "canonical_time_unit_seconds",
        _metadata_float(metadata, "canonical_time_unit_seconds"),
        float(canonical_time_unit_seconds),
        atol=1.0e-9,
    )
    _require_close(
        "reference_distance_km",
        _metadata_float(metadata, "reference_distance_km"),
        float(reference_distance_km),
        atol=1.0e-9,
    )
    _require_allclose(
        "canonical_node_times",
        _metadata_float_array(metadata, "canonical_node_times"),
        expected_nodes,
        atol=1.0e-12,
    )
    _require_allclose(
        "node_jd_tdb",
        _metadata_float_array(metadata, "node_jd_tdb"),
        expected_jds,
        atol=5.0e-10,
    )

    for body in ("moon_geocentric", "sun_geocentric"):
        samples = samples_from_cache(cache, body)
        sample_jds = np.asarray([sample.jd_tdb for sample in samples], dtype=float)
        _require_allclose(f"{body} sample JD grid", sample_jds, expected_jds, atol=1.0e-9)
        body_data = cache["bodies"][body]  # type: ignore[index]
        if not isinstance(body_data, dict):
            raise ValueError(f"Horizons cache body {body!r} is not an object")
        request_params = body_data.get("request_params", {})
        if isinstance(request_params, dict) and "TLIST" in request_params:
            expected_tlist = "\n".join(f"{float(jd):.9f}" for jd in expected_jds)
            if str(request_params["TLIST"]) != expected_tlist:
                raise ValueError(f"Horizons cache request TLIST mismatch for {body}")


def rotating_basis_from_moon(sample: HorizonsVectorSample) -> np.ndarray:
    x_axis = np.asarray(sample.position_km, dtype=float)
    distance = float(np.linalg.norm(x_axis))
    if distance <= 0.0:
        raise ValueError("Moon geocentric position has zero norm")
    x_hat = x_axis / distance
    h_vec = np.cross(sample.position_km, sample.velocity_km_s)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm <= 0.0:
        raise ValueError("Moon geocentric angular momentum has zero norm")
    z_hat = h_vec / h_norm
    y_hat = np.cross(z_hat, x_hat)
    y_hat /= max(float(np.linalg.norm(y_hat)), 1.0e-15)
    return np.column_stack((x_hat, y_hat, z_hat))


def inertial_to_rotating(vector_km: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(basis, dtype=float).T @ np.asarray(vector_km, dtype=float)


def wrapped_angle_delta(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def horizons_rotating_geometry(
    *,
    moon_samples: list[HorizonsVectorSample],
    sun_samples: list[HorizonsVectorSample],
    mu: float,
    reference_distance_km: float,
) -> dict[str, np.ndarray]:
    if len(moon_samples) != len(sun_samples):
        raise ValueError("Moon and Sun sample counts must match")
    fixed_reference_distance_km = float(reference_distance_km)
    if fixed_reference_distance_km <= 0.0:
        raise ValueError("reference_distance_km must be positive")
    moon_pos = np.asarray([sample.position_km for sample in moon_samples], dtype=float)
    moon_vel = np.asarray([sample.velocity_km_s for sample in moon_samples], dtype=float)
    sun_pos = np.asarray([sample.position_km for sample in sun_samples], dtype=float)
    jd = np.asarray([sample.jd_tdb for sample in moon_samples], dtype=float)
    em_distance = np.linalg.norm(moon_pos, axis=1)
    mean_em_distance = float(np.mean(em_distance))
    angular_rate = np.linalg.norm(np.cross(moon_pos, moon_vel), axis=1) / np.maximum(em_distance, 1.0e-15) ** 2
    mean_angular_rate = float(np.mean(angular_rate))
    sun_bary_rot_lu = []
    for moon, sun in zip(moon_samples, sun_samples):
        basis = rotating_basis_from_moon(moon)
        barycenter_geocentric = float(mu) * moon.position_km
        sun_bary_inertial = sun.position_km - barycenter_geocentric
        sun_bary_rot_lu.append(inertial_to_rotating(sun_bary_inertial, basis) / fixed_reference_distance_km)
    sun_bary_rot_lu_arr = np.asarray(sun_bary_rot_lu, dtype=float)
    sun_phase = np.unwrap(np.arctan2(sun_bary_rot_lu_arr[:, 1], sun_bary_rot_lu_arr[:, 0]))
    return {
        "jd_tdb": jd,
        "em_distance_km": em_distance,
        "mean_em_distance_km": np.asarray(mean_em_distance),
        "em_distance_ratio_to_mean": em_distance / mean_em_distance,
        "reference_distance_km": np.asarray(fixed_reference_distance_km),
        "em_distance_ratio_to_reference": em_distance / fixed_reference_distance_km,
        "em_angular_rate_rad_s": angular_rate,
        "mean_em_angular_rate_rad_s": np.asarray(mean_angular_rate),
        "em_angular_rate_ratio_to_mean": angular_rate / mean_angular_rate,
        "sun_barycenter_rotating_lu": sun_bary_rot_lu_arr,
        "sun_distance_lu": np.linalg.norm(sun_bary_rot_lu_arr, axis=1),
        "sun_phase_rotating_rad": sun_phase,
    }


def solar_tidal_acceleration_from_sun_vector(
    position_rotating_lu: np.ndarray,
    sun_rotating_lu: np.ndarray,
    *,
    sun_mu_ratio: float,
) -> np.ndarray:
    r = np.asarray(position_rotating_lu, dtype=float)
    scalar = r.ndim == 1
    if scalar:
        r = r[None, :]
    s = np.asarray(sun_rotating_lu, dtype=float)
    if s.ndim == 1:
        s = np.repeat(s[None, :], r.shape[0], axis=0)
    if r.shape != s.shape:
        raise ValueError(f"position and Sun vectors must have matching shape, got {r.shape} and {s.shape}")
    rel = s - r
    rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1.0e-15)
    sun_norm = np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1.0e-15)
    accel = float(sun_mu_ratio) * (rel / rel_norm**3 - s / sun_norm**3)
    return accel[0] if scalar else accel


def horizons_sun_vector_profile_from_cache(
    cache: dict[str, object],
    *,
    mu: float,
    reference_distance_km: float = DEFAULT_REFERENCE_DISTANCE_KM,
) -> dict[str, np.ndarray]:
    """Build canonical-time Sun vectors for cached-Horizons-derived replay.

    The returned Sun vectors are JPL-Horizons-derived Earth/Moon/Sun geometry
    transformed into the Earth-Moon rotating barycentric frame and divided by a
    fixed CR3BP reference distance. This is a replay/stress-probe input, not a
    SPICE or high-fidelity propagation product.
    """

    metadata = cache.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Horizons cache missing metadata")
    canonical_times = _metadata_float_array(metadata, "canonical_node_times")
    moon_samples = samples_from_cache(cache, "moon_geocentric")
    sun_samples = samples_from_cache(cache, "sun_geocentric")
    geometry = horizons_rotating_geometry(
        moon_samples=moon_samples,
        sun_samples=sun_samples,
        mu=float(mu),
        reference_distance_km=float(reference_distance_km),
    )
    sun_vectors = np.asarray(geometry["sun_barycenter_rotating_lu"], dtype=float)
    if canonical_times.shape != (sun_vectors.shape[0],):
        raise ValueError(
            "Horizons cache canonical_node_times length does not match Sun-vector sample count"
        )
    if np.any(np.diff(canonical_times) <= 0.0):
        raise ValueError("Horizons cache canonical_node_times must be strictly increasing")
    return {
        "canonical_time": canonical_times,
        "sun_barycenter_rotating_lu": sun_vectors,
        "sun_distance_lu": np.asarray(geometry["sun_distance_lu"], dtype=float),
        "jd_tdb": np.asarray(geometry["jd_tdb"], dtype=float),
    }


def interpolate_horizons_sun_vector(
    t: float,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Linearly interpolate cached Horizons Sun vectors over canonical time."""

    times = np.asarray(canonical_times, dtype=float)
    vectors = np.asarray(sun_vectors_rotating_lu, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"canonical_times must be one-dimensional, got {times.shape}")
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"sun_vectors_rotating_lu must have shape (n, 3), got {vectors.shape}")
    if vectors.shape[0] != times.shape[0]:
        raise ValueError(
            f"Sun-vector sample count {vectors.shape[0]} does not match time count {times.shape[0]}"
        )
    if times.shape[0] < 2:
        raise ValueError("at least two Sun-vector samples are required for interpolation")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("canonical_times must be strictly increasing")

    value = float(t)
    lower = float(times[0])
    upper = float(times[-1])
    if value < lower - float(tolerance) or value > upper + float(tolerance):
        raise ValueError(f"canonical time {value:.17g} is outside cached Horizons range {lower:.17g}--{upper:.17g}")
    value = min(max(value, lower), upper)
    return np.asarray([np.interp(value, times, vectors[:, axis]) for axis in range(3)], dtype=float)


def horizons_solar_tidal_derivative(
    t: float,
    states: np.ndarray,
    mu: float,
    *,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    sun_mu_ratio: float,
    control_accel: np.ndarray | None = None,
) -> np.ndarray:
    """CR3BP derivative plus cached-Horizons-derived solar-tidal acceleration."""

    s = np.asarray(states, dtype=float)
    scalar = s.ndim == 1
    working = s[None, :] if scalar else s
    if working.ndim != 2 or working.shape[1] != 6:
        raise ValueError(f"states must have shape (..., 6), got {working.shape}")

    sun_vector = interpolate_horizons_sun_vector(t, canonical_times, sun_vectors_rotating_lu)
    solar_accel = solar_tidal_acceleration_from_sun_vector(
        working[:, :3],
        sun_vector,
        sun_mu_ratio=float(sun_mu_ratio),
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


def rk4_horizons_solar_tidal_step(
    states: np.ndarray,
    mu: float,
    t: float,
    dt: float,
    *,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    sun_mu_ratio: float,
    control_accel: np.ndarray | None = None,
) -> np.ndarray:
    """One RK4 step for the cached-Horizons-derived solar-tidal stress probe."""

    s = np.asarray(states, dtype=float)
    k1 = horizons_solar_tidal_derivative(
        t,
        s,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=control_accel,
    )
    k2 = horizons_solar_tidal_derivative(
        t + 0.5 * dt,
        s + 0.5 * dt * k1,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=control_accel,
    )
    k3 = horizons_solar_tidal_derivative(
        t + 0.5 * dt,
        s + 0.5 * dt * k2,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=control_accel,
    )
    k4 = horizons_solar_tidal_derivative(
        t + dt,
        s + dt * k3,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=control_accel,
    )
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_horizons_solar_tidal_step_quadratic_control(
    state: np.ndarray,
    mu: float,
    t: float,
    dt: float,
    endpoint_control: np.ndarray,
    midpoint_control: np.ndarray,
    tau0: float,
    tau1: float,
    *,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    sun_mu_ratio: float,
) -> np.ndarray:
    tau_mid = 0.5 * (float(tau0) + float(tau1))
    u1 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau0)
    u2 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau_mid)
    u4 = quadratic_midpoint_control(endpoint_control, midpoint_control, tau1)
    k1 = horizons_solar_tidal_derivative(
        t,
        state,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=u1,
    )
    k2 = horizons_solar_tidal_derivative(
        t + 0.5 * dt,
        state + 0.5 * dt * k1,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=u2,
    )
    k3 = horizons_solar_tidal_derivative(
        t + 0.5 * dt,
        state + 0.5 * dt * k2,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=u2,
    )
    k4 = horizons_solar_tidal_derivative(
        t + dt,
        state + dt * k3,
        mu,
        canonical_times=canonical_times,
        sun_vectors_rotating_lu=sun_vectors_rotating_lu,
        sun_mu_ratio=sun_mu_ratio,
        control_accel=u4,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_piecewise_controls_horizons_solar_tidal(
    state0: np.ndarray,
    controls: np.ndarray,
    mu: float,
    tf: float,
    substeps_per_segment: int,
    *,
    canonical_times: np.ndarray,
    sun_vectors_rotating_lu: np.ndarray,
    sun_mu_ratio: float,
    midpoint_controls: np.ndarray | None = None,
    return_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Replay endpoint/midpoint controls with cached-Horizons-derived Sun vectors.

    This mirrors the independent-HS reporting propagation semantics: endpoint
    controls are segment controls and optional midpoint controls use the same
    quadratic interpolation satisfying ``u(0)=u(1)=endpoint`` and
    ``u(0.5)=midpoint``. The force perturbation is only a simplified solar
    tidal acceleration from linearly interpolated cached JPL Horizons geometry;
    it is not SPICE validation, high-fidelity propagation, flight validation,
    or production solver parity.
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

    for segment_index, control in enumerate(controls):
        if midpoint_controls is None:
            for _ in range(steps):
                state = rk4_horizons_solar_tidal_step(
                    state,
                    mu,
                    t,
                    dt,
                    canonical_times=canonical_times,
                    sun_vectors_rotating_lu=sun_vectors_rotating_lu,
                    sun_mu_ratio=sun_mu_ratio,
                    control_accel=control,
                )
                t += dt
        else:
            midpoint_control = midpoint_controls[segment_index]
            for step_index in range(steps):
                tau0 = step_index / float(steps)
                tau1 = (step_index + 1) / float(steps)
                state = _rk4_horizons_solar_tidal_step_quadratic_control(
                    state,
                    mu,
                    t,
                    dt,
                    control,
                    midpoint_control,
                    tau0,
                    tau1,
                    canonical_times=canonical_times,
                    sun_vectors_rotating_lu=sun_vectors_rotating_lu,
                    sun_mu_ratio=sun_mu_ratio,
                )
                t += dt
        if return_nodes:
            nodes.append(state.copy())
    return state, np.asarray(nodes) if return_nodes else None
