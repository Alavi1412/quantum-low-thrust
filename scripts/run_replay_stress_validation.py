"""Replay recorded nominal controls under deterministic integration variants.

This postprocessor reads persisted nominal-control sidecars from representative
continuous-backend rows and repropagates them with the existing CR3BP utilities.
It does not launch trajectory optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.cr3bp import load_source_states, propagate_controls_batch
from qlt.direct_collocation import propagate_piecewise_controls
from qlt.objective import state_error


DEFAULT_RESULTS_DIR = ROOT / "data" / "results" / "replay_stress_validation"
DEFAULT_TABLES_DIR = ROOT / "tables" / "replay_stress_validation"

CONFIGURED_NOMINAL_THRESHOLD = 0.09
TIGHTER_NOMINAL_THRESHOLD = 0.065
TIGHT_NOMINAL_THRESHOLD = 0.05
BASELINE_REPRODUCTION_TOLERANCE = 1.0e-12

REPLAY_COLUMNS = [
    "case",
    "source_artifact",
    "source_control_path",
    "source_control_sha256",
    "role_backend",
    "target_mode",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "source_substeps_per_segment",
    "replay_substeps_per_segment",
    "acceleration_scale",
    "replay_control_max_norm",
    "replay_control_bound_violation",
    "replay_variant",
    "recorded_nominal_error",
    "nominal_error",
    "delta_from_recorded_nominal_error",
    "absolute_delta_from_recorded_nominal_error",
    "configured_nominal_pass",
    "tighter_nominal_pass",
    "tight_nominal_pass",
    "baseline_reproduces_recorded",
    "midpoint_controls_replayed",
    "interpretation_limits",
]


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    source_csv: Path
    source_config: Path
    source_row_key: str
    role_backend: str
    require_midpoint_controls: bool = False


@dataclass(frozen=True)
class ReplayVariant:
    label: str
    substep_factor: int = 1
    acceleration_scale: float = 1.0
    interpretation_limits: str = ""


REPLAY_CASES = [
    ReplayCase(
        case_id="all_single_p04_warm_from_p03",
        source_csv=ROOT / "data" / "results" / "continuation_extension_suite" / "continuation_margin_suite.csv",
        source_config=ROOT / "configs" / "continuation_extension_suite.yaml",
        source_row_key="case_id",
        role_backend="bounded multiple-shooting continuous continuation; endpoint nominal controls",
    ),
    ReplayCase(
        case_id="two_segment_n8_p03_cold",
        source_csv=ROOT / "data" / "results" / "continuation_extension_suite" / "continuation_margin_suite.csv",
        source_config=ROOT / "configs" / "continuation_extension_suite.yaml",
        source_row_key="case_id",
        role_backend="bounded multiple-shooting continuous continuation; endpoint nominal controls",
    ),
    ReplayCase(
        case_id="ihs_phase_p04_amax02_warm_from_p03",
        source_csv=ROOT
        / "data"
        / "results"
        / "independent_hs_continuation_baseline"
        / "independent_hs_continuation_baseline.csv",
        source_config=ROOT / "configs" / "independent_hs_continuation_baseline.yaml",
        source_row_key="case_id",
        role_backend="independent-midpoint Hermite-Simpson collocation; endpoint and midpoint nominal controls",
        require_midpoint_controls=True,
    ),
    ReplayCase(
        case_id="ihs_all_single_p04_amax02_warm_from_p03",
        source_csv=ROOT
        / "data"
        / "results"
        / "independent_hs_all_configured_headroom"
        / "independent_hs_all_configured_headroom.csv",
        source_config=ROOT / "configs" / "independent_hs_all_configured_headroom.yaml",
        source_row_key="case_id",
        role_backend=(
            "all-configured independent-midpoint Hermite-Simpson; "
            "endpoint and midpoint nominal controls"
        ),
        require_midpoint_controls=True,
    ),
]

REPLAY_VARIANTS = [
    ReplayVariant(
        label="baseline_source_substeps",
        interpretation_limits="Deterministic source-grid nominal-control replay; branch recovery controls are not replayed.",
    ),
    ReplayVariant(
        label="refined_2x_substeps",
        substep_factor=2,
        interpretation_limits="Integration-grid refinement only; target definition and recorded controls are fixed.",
    ),
    ReplayVariant(
        label="refined_4x_substeps",
        substep_factor=4,
        interpretation_limits="Integration-grid refinement only; target definition and recorded controls are fixed.",
    ),
    ReplayVariant(
        label="accel_scale_0p99_source_substeps",
        acceleration_scale=0.99,
        interpretation_limits="Small direct acceleration-scale stress around recorded controls; not high-fidelity validation.",
    ),
    ReplayVariant(
        label="accel_scale_1p01_source_substeps",
        acceleration_scale=1.01,
        interpretation_limits=(
            "Small direct acceleration-scale stress around recorded controls; may exceed the configured acceleration "
            "bound and is not high-fidelity validation."
        ),
    ),
]


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_recorded_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"source CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_substeps(case: ReplayCase, row: dict[str, str]) -> int:
    value = row.get("substeps_per_segment")
    if value not in (None, ""):
        return int(float(value))
    config = yaml.safe_load(case.source_config.read_text(encoding="utf-8"))
    return int(config["benchmark"]["substeps_per_segment"])


def _source_row(case: ReplayCase) -> dict[str, str]:
    rows = _read_csv_rows(case.source_csv)
    for row in rows:
        if str(row.get(case.source_row_key, "")) == case.case_id:
            return row
    raise RuntimeError(f"no row in {_relative_or_absolute(case.source_csv)} for {case.source_row_key}={case.case_id}")


def _load_sidecar(
    path: Path,
    *,
    expected_segments: int,
    expected_case_id: str,
    require_midpoint_controls: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"recorded nominal-control sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sidecar_case = str(payload.get("case_id", ""))
    if sidecar_case and sidecar_case != expected_case_id:
        raise RuntimeError(f"sidecar case_id {sidecar_case!r} does not match expected {expected_case_id!r}")
    controls = np.asarray(payload.get("nominal_endpoint_controls", payload.get("controls")), dtype=float)
    if controls.shape != (int(expected_segments), 3):
        raise RuntimeError(f"sidecar controls shape {controls.shape} does not match {(expected_segments, 3)}")

    midpoint_controls = None
    midpoint_raw = payload.get("nominal_midpoint_controls")
    if midpoint_raw is not None:
        midpoint_controls = np.asarray(midpoint_raw, dtype=float)
        if midpoint_controls.shape != controls.shape:
            raise RuntimeError(f"midpoint controls shape {midpoint_controls.shape} does not match {controls.shape}")
    if require_midpoint_controls and midpoint_controls is None:
        raise RuntimeError(
            f"{expected_case_id} requires independent midpoint controls but sidecar does not contain them"
        )
    return controls, midpoint_controls, payload


def _load_states(row: dict[str, str], source_substeps: int):
    return load_source_states(
        ROOT / "data" / "source_states.json",
        transfer_time=float(row["transfer_time"]),
        substeps=int(source_substeps),
        target_mode=str(row.get("target_mode", "catalog_halo_phase_shift")),
        segments=int(row["segments"]),
        amax=float(row["amax"]),
        phase_time=float(row["phase_time"]) if str(row.get("phase_time", "")).strip() else None,
    )


def _replay_nominal_error(
    *,
    row: dict[str, str],
    controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
    source_substeps: int,
    variant: ReplayVariant,
) -> float:
    states = _load_states(row, source_substeps)
    scaled_controls = controls * float(variant.acceleration_scale)
    replay_substeps = int(source_substeps) * int(variant.substep_factor)
    if midpoint_controls is None:
        final, _ = propagate_controls_batch(
            states.initial,
            scaled_controls,
            states.mu,
            float(row["transfer_time"]),
            replay_substeps,
        )
        final_state = final[0]
    else:
        scaled_midpoint_controls = midpoint_controls * float(variant.acceleration_scale)
        final_state, _ = propagate_piecewise_controls(
            states.initial,
            scaled_controls,
            states.mu,
            float(row["transfer_time"]),
            replay_substeps,
            midpoint_controls=scaled_midpoint_controls,
        )
    return float(state_error(final_state, states.target, 1.0, 0.35))


def _case_rows(case: ReplayCase) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    row = _source_row(case)
    source_substeps = _source_substeps(case, row)
    sidecar_path = _resolve_recorded_path(str(row.get("nominal_control_path", "")))
    try:
        controls, midpoint_controls, _ = _load_sidecar(
            sidecar_path,
            expected_segments=int(row["segments"]),
            expected_case_id=case.case_id,
            require_midpoint_controls=case.require_midpoint_controls,
        )
    except RuntimeError as exc:
        return [], {
            "case": case.case_id,
            "source_artifact": _relative_or_absolute(case.source_csv),
            "source_control_path": _relative_or_absolute(sidecar_path),
            "reason": str(exc),
        }

    recorded_nominal = float(row["nominal_error"])
    case_rows: list[dict[str, object]] = []
    for variant in REPLAY_VARIANTS:
        replay_substeps = source_substeps * int(variant.substep_factor)
        scaled_controls = controls * float(variant.acceleration_scale)
        scaled_midpoint_controls = (
            midpoint_controls * float(variant.acceleration_scale) if midpoint_controls is not None else None
        )
        max_norms = [float(np.max(np.linalg.norm(scaled_controls, axis=1)))]
        if scaled_midpoint_controls is not None:
            max_norms.append(float(np.max(np.linalg.norm(scaled_midpoint_controls, axis=1))))
        replay_control_max_norm = max(max_norms)
        replay_control_bound_violation = max(0.0, replay_control_max_norm - float(row["amax"]))
        if replay_control_bound_violation <= 1.0e-12:
            replay_control_bound_violation = 0.0
        nominal_error = _replay_nominal_error(
            row=row,
            controls=controls,
            midpoint_controls=midpoint_controls,
            source_substeps=source_substeps,
            variant=variant,
        )
        delta = nominal_error - recorded_nominal
        baseline = variant.label == "baseline_source_substeps"
        case_rows.append(
            {
                "case": case.case_id,
                "source_artifact": _relative_or_absolute(case.source_csv),
                "source_control_path": _relative_or_absolute(sidecar_path),
                "source_control_sha256": _sha256(sidecar_path),
                "role_backend": case.role_backend,
                "target_mode": str(row.get("target_mode", "")),
                "phase_time": float(row["phase_time"]) if str(row.get("phase_time", "")).strip() else np.nan,
                "transfer_time": float(row["transfer_time"]),
                "amax": float(row["amax"]),
                "segments": int(row["segments"]),
                "source_substeps_per_segment": int(source_substeps),
                "replay_substeps_per_segment": int(replay_substeps),
                "acceleration_scale": float(variant.acceleration_scale),
                "replay_control_max_norm": replay_control_max_norm,
                "replay_control_bound_violation": replay_control_bound_violation,
                "replay_variant": variant.label,
                "recorded_nominal_error": recorded_nominal,
                "nominal_error": nominal_error,
                "delta_from_recorded_nominal_error": delta,
                "absolute_delta_from_recorded_nominal_error": abs(delta),
                "configured_nominal_pass": bool(nominal_error <= CONFIGURED_NOMINAL_THRESHOLD),
                "tighter_nominal_pass": bool(nominal_error <= TIGHTER_NOMINAL_THRESHOLD),
                "tight_nominal_pass": bool(nominal_error <= TIGHT_NOMINAL_THRESHOLD),
                "baseline_reproduces_recorded": bool(
                    baseline and abs(delta) <= BASELINE_REPRODUCTION_TOLERANCE
                ),
                "midpoint_controls_replayed": bool(midpoint_controls is not None),
                "interpretation_limits": variant.interpretation_limits,
            }
        )
    return case_rows, None


def build_replay_stress_validation() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for case in REPLAY_CASES:
        case_rows, skipped_case = _case_rows(case)
        rows.extend(case_rows)
        if skipped_case is not None:
            skipped.append(skipped_case)
    return pd.DataFrame(rows, columns=REPLAY_COLUMNS), skipped


def write_latex_table(df: pd.DataFrame, tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "replay_stress_validation_table.tex"
    if df.empty:
        path.write_text("% No replay/stress validation rows.\n", encoding="utf-8")
        return path
    table = df[
        [
            "case",
            "role_backend",
            "replay_variant",
            "nominal_error",
            "delta_from_recorded_nominal_error",
            "configured_nominal_pass",
            "tighter_nominal_pass",
            "tight_nominal_pass",
            "interpretation_limits",
        ]
    ].copy()
    table.columns = [
        "Case",
        "Role/backend",
        "Replay variant",
        "Nominal error",
        "Delta from recorded",
        "Configured nominal pass",
        "Tighter nominal pass",
        "Tight nominal pass",
        "Interpretation/limits",
    ]
    path.write_text(table.to_latex(index=False, float_format="%.6g", escape=True), encoding="utf-8")
    return path


def _input_artifacts(df: pd.DataFrame) -> list[Path]:
    paths = {
        ROOT / "data" / "source_states.json",
        *[case.source_csv for case in REPLAY_CASES],
        *[case.source_config for case in REPLAY_CASES],
    }
    if not df.empty:
        for value in df["source_control_path"].dropna().unique():
            paths.add(_resolve_recorded_path(str(value)))
    return sorted(paths, key=lambda path: _relative_or_absolute(path))


def write_artifacts(
    *,
    results_dir: Path,
    tables_dir: Path,
    command: str,
) -> dict[str, object]:
    df, skipped = build_replay_stress_validation()
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "replay_stress_validation.csv"
    metadata_path = results_dir / "replay_stress_validation_metadata.json"
    table_path = write_latex_table(df, tables_dir)
    df.to_csv(csv_path, index=False)

    baseline = df[df["replay_variant"] == "baseline_source_substeps"] if not df.empty else df
    metadata = {
        "command": command,
        "row_count": int(len(df)),
        "case_count": int(df["case"].nunique()) if not df.empty else 0,
        "skipped_cases": skipped,
        "optimization_rerun": False,
        "uses_only_recorded_control_sidecars": True,
        "high_fidelity_claim": False,
        "source_mode": (
            "Recorded nominal-control sidecars are repropagated with existing CR3BP utilities only; "
            "no least-squares optimization, schedule search, branch recovery optimization, or high-fidelity force model is launched."
        ),
        "determinism_note": (
            "Runtime and wall-clock timestamps are intentionally omitted so identical inputs and command text "
            "produce byte-stable CSV, table, and metadata outputs."
        ),
        "thresholds": {
            "configured_nominal": CONFIGURED_NOMINAL_THRESHOLD,
            "tighter_nominal": TIGHTER_NOMINAL_THRESHOLD,
            "tight_nominal": TIGHT_NOMINAL_THRESHOLD,
        },
        "baseline_reproduction_tolerance": BASELINE_REPRODUCTION_TOLERANCE,
        "baseline_reproduction": {
            "all_baselines_reproduced": bool(
                (baseline["absolute_delta_from_recorded_nominal_error"] <= BASELINE_REPRODUCTION_TOLERANCE).all()
            )
            if not baseline.empty
            else False,
            "max_abs_delta": float(baseline["absolute_delta_from_recorded_nominal_error"].max())
            if not baseline.empty
            else None,
        },
        "variant_semantics": [
            {
                "replay_variant": variant.label,
                "substep_factor": int(variant.substep_factor),
                "acceleration_scale": float(variant.acceleration_scale),
                "interpretation_limits": variant.interpretation_limits,
            }
            for variant in REPLAY_VARIANTS
        ],
        "input_artifacts": [
            {
                "path": _relative_or_absolute(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in _input_artifacts(df)
        ],
        "artifacts": {
            "replay_stress_validation_csv": _relative_or_absolute(csv_path),
            "replay_stress_validation_metadata_json": _relative_or_absolute(metadata_path),
            "replay_stress_validation_table_tex": _relative_or_absolute(table_path),
        },
        "interpretation_limits": [
            "Rows are nominal-control replays only; selected branch recovery controls are not replayed.",
            "The acceleration-scale rows are small direct perturbations of recorded controls, not a high-fidelity validation model; scale-up rows may exceed the configured acceleration bound.",
            "Pass flags use nominal-error thresholds only because this script does not rerun missed-thrust branch recovery.",
            "The target definition is fixed to the source artifact's CR3BP target generation settings.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "dataframe": df,
        "metadata": metadata,
        "csv_path": csv_path,
        "metadata_path": metadata_path,
        "table_path": table_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded nominal controls under deterministic CR3BP stress variants."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else ROOT / args.tables_dir
    artifacts = write_artifacts(
        results_dir=results_dir,
        tables_dir=tables_dir,
        command=" ".join(sys.argv),
    )
    metadata = artifacts["metadata"]
    print(
        "Completed replay/stress validation "
        f"with {metadata['row_count']} rows; optimization_rerun={metadata['optimization_rerun']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
