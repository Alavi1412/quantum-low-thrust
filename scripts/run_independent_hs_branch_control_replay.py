"""Replay persisted independent-HS branch controls under normalized CR3BP.

This postprocessor reads branch-control sidecars written by
scripts/run_independent_hs_continuation.py for independent-midpoint
Hermite-Simpson rows. It repropagates recorded nominal and branch full-control
schedules only; it does not rerun optimization, branch selection, schedule
search, high-fidelity validation, or production-solver parity checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

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
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "independent_hs_all_configured_headroom.yaml"
DEFAULT_RESULTS_DIR = Path("data/results/independent_hs_branch_control_replay")
DEFAULT_TABLES_DIR = Path("tables/independent_hs_branch_control_replay")
REPLAY_CSV_NAME = "independent_hs_branch_control_replay.csv"
REPLAY_METADATA_NAME = "independent_hs_branch_control_replay_metadata.json"
REPLAY_TABLE_NAME = "independent_hs_branch_control_replay_table.tex"

REPLAY_COLUMNS = [
    "case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "controls_path",
    "controls_sha256",
    "midpoint_controls_replayed",
    "recorded_terminal_error",
    "replay_terminal_error",
    "terminal_error_delta",
    "tolerance",
    "passes_tolerance",
    "recovery_start",
    "recovery_segments",
    "replay_semantics",
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


def _resolve_artifact_path(value: object) -> Path:
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


def _read_json_verified(path_text: object, expected_sha256: object | None = None) -> tuple[dict, Path, str]:
    path = _resolve_artifact_path(path_text)
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


def _case_by_id(config: dict) -> dict[str, dict]:
    return {str(case["case_id"]): case for case in ihs_runner._suite_cases(config)}


def _case_config(config: dict, case: dict) -> dict:
    ihs_runner._configure_base(config)
    return ihs_runner._base._case_config(config, case)


def _terminal_error(
    *,
    state0: np.ndarray,
    target: np.ndarray,
    cfg,
    endpoint_controls: np.ndarray,
    midpoint_controls: np.ndarray | None,
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


def _replay_row(
    *,
    case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    controls_path: Path,
    controls_sha256: str,
    midpoint_controls_replayed: bool,
    recorded_terminal_error: float,
    replay_terminal_error: float,
    tolerance: float,
    recovery_start: int | None = None,
    recovery_segments: int | None = None,
) -> dict[str, object]:
    delta = abs(float(replay_terminal_error) - float(recorded_terminal_error))
    return {
        "case_id": case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
        "controls_path": _relative_or_absolute(controls_path),
        "controls_sha256": controls_sha256,
        "midpoint_controls_replayed": bool(midpoint_controls_replayed),
        "recorded_terminal_error": float(recorded_terminal_error),
        "replay_terminal_error": float(replay_terminal_error),
        "terminal_error_delta": float(delta),
        "tolerance": float(tolerance),
        "passes_tolerance": bool(delta <= float(tolerance)),
        "recovery_start": "" if recovery_start is None else int(recovery_start),
        "recovery_segments": "" if recovery_segments is None else int(recovery_segments),
        "replay_semantics": (
            "normalized CR3BP persisted-control replay only; no optimization rerun; "
            "no high-fidelity validation; no production solver parity"
        ),
    }


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / REPLAY_TABLE_NAME
    table = pd.DataFrame(summary_rows)
    if table.empty:
        path.write_text("% No independent-HS branch-control replay rows.\n", encoding="utf-8")
        return path
    display = table[
        [
            "case_id",
            "branch_row_count",
            "nominal_terminal_error_delta",
            "max_branch_terminal_error_delta",
            "max_terminal_error_delta",
            "selected_worst_replay_delta",
            "all_mask_worst_replay_delta",
            "tolerance",
            "passes_tolerance",
        ]
    ].rename(
        columns={
            "case_id": "Case",
            "branch_row_count": "Branch rows",
            "nominal_terminal_error_delta": "Nominal delta",
            "max_branch_terminal_error_delta": "Max branch delta",
            "max_terminal_error_delta": "Max delta",
            "selected_worst_replay_delta": "Selected-worst delta",
            "all_mask_worst_replay_delta": "All-mask-worst delta",
            "tolerance": "Tolerance",
            "passes_tolerance": "Pass",
        }
    )
    display.to_latex(path, index=False, float_format="%.3e", escape=True)
    return path


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    input_results_dir, _, _ = output_directories(Path.cwd(), config)
    input_csv = input_results_dir / "independent_hs_all_configured_headroom.csv"
    if not input_csv.is_file():
        artifact_stem = str((config.get("run", {}) or {}).get("artifact_stem", "independent_hs_continuation_baseline"))
        input_csv = input_results_dir / f"{artifact_stem}.csv"
    input_rows = _read_input_rows(input_csv)
    cases = _case_by_id(config)
    output_results_dir = args.results_dir if args.results_dir.is_absolute() else Path.cwd() / args.results_dir
    output_tables_dir = args.tables_dir if args.tables_dir.is_absolute() else Path.cwd() / args.tables_dir
    output_results_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    replay_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    input_artifacts: list[dict[str, object]] = [
        {"path": _relative_or_absolute(config_path), "sha256": _sha256(config_path), "bytes": config_path.stat().st_size},
        {"path": _relative_or_absolute(source_states), "sha256": _sha256(source_states), "bytes": source_states.stat().st_size},
        {"path": _relative_or_absolute(input_csv), "sha256": _sha256(input_csv), "bytes": input_csv.stat().st_size},
    ]

    for row in input_rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in cases:
            continue
        if args.case_id and case_id not in set(args.case_id):
            continue
        if not _bool_value(row.get("branch_control_replay_ready", "")):
            continue
        case = cases[case_id]
        case_config = _case_config(config, case)
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        manifest, manifest_path, manifest_sha = _read_json_verified(
            row["branch_control_manifest_path"],
            row.get("branch_control_manifest_hash"),
        )
        nominal, nominal_path, nominal_sha = _read_json_verified(
            manifest["nominal_control_path"],
            manifest.get("nominal_control_sha256"),
        )
        input_artifacts.extend(
            [
                {"path": _relative_or_absolute(manifest_path), "sha256": manifest_sha, "bytes": manifest_path.stat().st_size},
                {"path": _relative_or_absolute(nominal_path), "sha256": nominal_sha, "bytes": nominal_path.stat().st_size},
            ]
        )
        target_state = np.asarray(states.target, dtype=float)
        recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
        if recorded_target.shape != target_state.shape:
            raise RuntimeError(f"manifest target_state shape mismatch for {case_id}")
        target_state_max_abs_delta = float(np.max(np.abs(target_state - recorded_target)))

        nominal_endpoint = np.asarray(nominal.get("nominal_endpoint_controls", nominal.get("controls")), dtype=float)
        nominal_midpoint_raw = nominal.get("nominal_midpoint_controls")
        nominal_midpoint = None if nominal_midpoint_raw is None else np.asarray(nominal_midpoint_raw, dtype=float)
        nominal_error = _terminal_error(
            state0=np.asarray(states.initial, dtype=float),
            target=target_state,
            cfg=cfg,
            endpoint_controls=nominal_endpoint,
            midpoint_controls=nominal_midpoint,
        )
        nominal_recorded = float(manifest.get("nominal_error", row.get("nominal_error")))
        replay_rows.append(
            _replay_row(
                case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                controls_path=nominal_path,
                controls_sha256=nominal_sha,
                midpoint_controls_replayed=nominal_midpoint is not None,
                recorded_terminal_error=nominal_recorded,
                replay_terminal_error=nominal_error,
                tolerance=float(args.tolerance),
            )
        )

        branch_replay_errors: list[float] = []
        branch_deltas: list[float] = []
        branch_entries = list(manifest.get("branch_control_sidecars", []))
        for entry in branch_entries:
            branch, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            input_artifacts.append(
                {"path": _relative_or_absolute(branch_path), "sha256": branch_sha, "bytes": branch_path.stat().st_size}
            )
            endpoint_controls = np.asarray(branch.get("branch_endpoint_controls", branch.get("branch_controls")), dtype=float)
            midpoint_raw = branch.get("branch_midpoint_controls")
            midpoint_controls = None if midpoint_raw is None else np.asarray(midpoint_raw, dtype=float)
            replay_error = _terminal_error(
                state0=np.asarray(states.initial, dtype=float),
                target=target_state,
                cfg=cfg,
                endpoint_controls=endpoint_controls,
                midpoint_controls=midpoint_controls,
            )
            recorded_error = float(branch["recorded_branch_terminal_error"])
            branch_row = _replay_row(
                case_id=case_id,
                record_type="branch",
                branch_order=int(branch["branch_order"]),
                mask_index=int(branch["mask_index"]),
                outage_mask=[int(value) for value in branch["outage_mask"]],
                controls_path=branch_path,
                controls_sha256=branch_sha,
                midpoint_controls_replayed=midpoint_controls is not None,
                recorded_terminal_error=recorded_error,
                replay_terminal_error=replay_error,
                tolerance=float(args.tolerance),
                recovery_start=int(branch["recovery_start"]),
                recovery_segments=int(branch["recovery_segments"]),
            )
            branch_deltas.append(float(branch_row["terminal_error_delta"]))
            branch_replay_errors.append(float(replay_error))
            replay_rows.append(branch_row)

        case_rows = [item for item in replay_rows if str(item["case_id"]) == case_id]
        selected_worst_replay = float(max(branch_replay_errors)) if branch_replay_errors else nominal_error
        recorded_selected_worst = float(manifest.get("selected_worst_error", row.get("selected_worst_error")))
        recorded_all_worst = float(manifest.get("all_mask_worst_error", row.get("all_mask_worst_error")))
        nominal_delta = float(case_rows[0]["terminal_error_delta"])
        max_branch_delta = float(max(branch_deltas)) if branch_deltas else 0.0
        max_delta = float(max([float(item["terminal_error_delta"]) for item in case_rows], default=0.0))
        selected_delta = abs(selected_worst_replay - recorded_selected_worst)
        all_delta = abs(selected_worst_replay - recorded_all_worst)
        expected_branch_count = int(manifest.get("expected_branch_count", len(branch_entries)))
        passes = bool(
            max_delta <= float(args.tolerance)
            and selected_delta <= float(args.tolerance)
            and all_delta <= float(args.tolerance)
            and target_state_max_abs_delta <= float(args.tolerance)
            and len(branch_entries) == expected_branch_count
        )
        summary_rows.append(
            {
                "case_id": case_id,
                "branch_row_count": int(len(branch_entries)),
                "expected_branch_count": expected_branch_count,
                "nominal_terminal_error_delta": nominal_delta,
                "max_branch_terminal_error_delta": max_branch_delta,
                "max_terminal_error_delta": max_delta,
                "selected_worst_replay_terminal_error": selected_worst_replay,
                "recorded_selected_worst_error": recorded_selected_worst,
                "selected_worst_replay_delta": selected_delta,
                "recorded_all_mask_worst_error": recorded_all_worst,
                "all_mask_worst_replay_delta": all_delta,
                "target_state_max_abs_delta": target_state_max_abs_delta,
                "tolerance": float(args.tolerance),
                "passes_tolerance": passes,
            }
        )

    if not replay_rows:
        raise RuntimeError("no replay-ready independent-HS branch-control rows found")

    df = pd.DataFrame(replay_rows, columns=REPLAY_COLUMNS)
    replay_csv = output_results_dir / REPLAY_CSV_NAME
    df.to_csv(replay_csv, index=False)
    table_path = _write_table(summary_rows, output_tables_dir)
    branch_rows = df[df["record_type"] == "branch"]
    max_delta = float(df["terminal_error_delta"].astype(float).max()) if len(df) else 0.0
    max_branch_delta = float(branch_rows["terminal_error_delta"].astype(float).max()) if len(branch_rows) else 0.0
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
        "case_count": int(df["case_id"].nunique()),
        "nominal_row_count": int((df["record_type"] == "nominal").sum()),
        "branch_row_count": int(len(branch_rows)),
        "max_terminal_error_delta": max_delta,
        "max_branch_terminal_error_delta": max_branch_delta,
        "tolerance": float(args.tolerance),
        "passes_tolerance": bool(all(row["passes_tolerance"] for row in summary_rows)),
        "optimization_rerun": False,
        "branch_control_replay": True,
        "uses_recorded_artifacts_only": True,
        "high_fidelity_validation": False,
        "production_solver_parity_claim": False,
        "fuel_optimality_claim": False,
        "quantum_advantage_claim": False,
        "replay_semantics": (
            "Normalized CR3BP replay of persisted independent-HS nominal and branch full-control schedules. "
            "The script validates manifest/sidecar SHA-256 hashes and compares replayed terminal errors "
            "to recorded nominal, selected-worst, and all-mask-worst errors without rerunning optimization."
        ),
        "limitations": ihs_runner._base._branch_control_replay_limitations(),
        "summary": summary_rows,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "independent_hs_branch_control_replay_csv": _relative_or_absolute(replay_csv),
            "independent_hs_branch_control_replay_metadata_json": _relative_or_absolute(
                output_results_dir / REPLAY_METADATA_NAME
            ),
            "independent_hs_branch_control_replay_table_tex": _relative_or_absolute(table_path),
        },
    }
    _write_json(output_results_dir / REPLAY_METADATA_NAME, metadata)
    print(
        "Completed independent-HS branch-control replay "
        f"with {int(len(branch_rows))} branch rows, max_branch_delta={max_branch_delta:.3e}, "
        f"passes_tolerance={metadata['passes_tolerance']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay persisted independent-HS branch controls under normalized CR3BP."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
