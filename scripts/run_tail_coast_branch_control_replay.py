"""Replay persisted tail-coast accepted branch controls under normalized CR3BP.

This postprocessor reads accepted-control sidecars written by
scripts/run_tail_coast_recovery.py. It repropagates the recorded nominal and
accepted branch full-control schedules only; it does not rerun optimization,
portfolio selection, schedule search, or high-fidelity validation.
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

import run_tail_coast_recovery as tail_runner
from qlt.cr3bp import propagate_controls_batch
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import state_error
from qlt.reporting import sanitize_json


DEFAULT_CONFIG = ROOT / "configs" / "hard_catalog_tail_coast_branch_control_replay.yaml"
REPLAY_CSV_NAME = "tail_coast_branch_control_replay.csv"
REPLAY_METADATA_NAME = "tail_coast_branch_control_replay_metadata.json"
REPLAY_TABLE_NAME = "tail_coast_branch_control_replay_table.tex"

REPLAY_COLUMNS = [
    "suite_case_id",
    "record_type",
    "branch_order",
    "mask_index",
    "outage_mask",
    "controls_path",
    "controls_sha256",
    "recorded_terminal_error",
    "replay_terminal_error",
    "terminal_error_delta",
    "tolerance",
    "passes_tolerance",
    "recovery_start",
    "recovery_segments",
    "optimizer_ran",
    "optimizer_success",
    "accepted_candidate",
    "accepted_weight_variant",
    "accepted_initialization",
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
    text = json.dumps(sanitize_json(data), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
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
    return str(path)


def _resolve_artifact_path(value: object) -> Path:
    text = str(value).strip()
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
        raise RuntimeError(f"tail-coast recovery CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float_value(value: object) -> float:
    return float(str(value).strip())


def _replay_terminal_error(state0: np.ndarray, target: np.ndarray, cfg, controls: np.ndarray) -> float:
    controls = np.asarray(controls, dtype=float).reshape((int(cfg.n_segments), 3))
    final, _ = propagate_controls_batch(state0, controls, cfg.mu, cfg.tf, cfg.substeps)
    return float(state_error(final[0], target, cfg.position_scale, cfg.velocity_scale))


def _replay_row(
    *,
    suite_case_id: str,
    record_type: str,
    branch_order: int | None,
    mask_index: int | None,
    outage_mask: list[int] | None,
    controls_path: Path,
    controls_sha256: str,
    recorded_terminal_error: float,
    replay_terminal_error: float,
    tolerance: float,
    recovery_start: int | None = None,
    recovery_segments: int | None = None,
    optimizer_ran: bool | None = None,
    optimizer_success: bool | None = None,
    accepted_candidate: str = "",
    accepted_weight_variant: str = "",
    accepted_initialization: str = "",
) -> dict[str, object]:
    delta = abs(float(replay_terminal_error) - float(recorded_terminal_error))
    return {
        "suite_case_id": suite_case_id,
        "record_type": record_type,
        "branch_order": "" if branch_order is None else int(branch_order),
        "mask_index": "" if mask_index is None else int(mask_index),
        "outage_mask": "" if outage_mask is None else json.dumps([int(value) for value in outage_mask]),
        "controls_path": _relative_or_absolute(controls_path),
        "controls_sha256": controls_sha256,
        "recorded_terminal_error": float(recorded_terminal_error),
        "replay_terminal_error": float(replay_terminal_error),
        "terminal_error_delta": float(delta),
        "tolerance": float(tolerance),
        "passes_tolerance": bool(delta <= float(tolerance)),
        "recovery_start": "" if recovery_start is None else int(recovery_start),
        "recovery_segments": "" if recovery_segments is None else int(recovery_segments),
        "optimizer_ran": "" if optimizer_ran is None else bool(optimizer_ran),
        "optimizer_success": "" if optimizer_success is None else bool(optimizer_success),
        "accepted_candidate": accepted_candidate,
        "accepted_weight_variant": accepted_weight_variant,
        "accepted_initialization": accepted_initialization,
        "replay_semantics": (
            "normalized CR3BP accepted-control replay only; no optimization rerun; "
            "no high-fidelity validation"
        ),
    }


def _case_by_id(config: dict) -> dict[str, dict]:
    return {str(case["suite_case_id"]): case for case in tail_runner._suite_cases(config)}


def _write_table(summary_rows: list[dict[str, object]], tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / REPLAY_TABLE_NAME
    table = pd.DataFrame(summary_rows)
    if table.empty:
        path.write_text("% No tail-coast branch-control replay rows.\n", encoding="utf-8")
        return path
    display = table[
        [
            "suite_case_id",
            "branch_row_count",
            "nominal_terminal_error_delta",
            "max_branch_terminal_error_delta",
            "max_terminal_error_delta",
            "tolerance",
            "passes_tolerance",
        ]
    ].rename(
        columns={
            "suite_case_id": "Case",
            "branch_row_count": "Branch rows",
            "nominal_terminal_error_delta": "Nominal delta",
            "max_branch_terminal_error_delta": "Max branch delta",
            "max_terminal_error_delta": "Max delta",
            "tolerance": "Tolerance",
            "passes_tolerance": "Pass",
        }
    )
    display.to_latex(path, index=False, float_format="%.3e", escape=True)
    return path


def run(args: argparse.Namespace) -> pd.DataFrame:
    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir, _, tables_dir = output_directories(Path.cwd(), config)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    source_states = args.source_states if args.source_states.is_absolute() else Path.cwd() / args.source_states
    tail_csv = results_dir / "tail_coast_recovery.csv"
    input_rows = _read_input_rows(tail_csv)
    cases = _case_by_id(config)
    replay_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    input_artifacts: list[dict[str, object]] = [
        {"path": _relative_or_absolute(config_path), "sha256": _sha256(config_path), "bytes": config_path.stat().st_size},
        {"path": _relative_or_absolute(source_states), "sha256": _sha256(source_states), "bytes": source_states.stat().st_size},
        {"path": _relative_or_absolute(tail_csv), "sha256": _sha256(tail_csv), "bytes": tail_csv.stat().st_size},
    ]

    for row in input_rows:
        case_id = str(row.get("suite_case_id", ""))
        if case_id not in cases:
            continue
        if not _bool_value(row.get("branch_control_replay_ready", "")):
            continue
        case = cases[case_id]
        case_config = tail_runner._case_config(config, case)
        states = load_configured_states(Path.cwd(), case_config, source_states)
        cfg = make_objective_config(case_config, states.mu)
        manifest, manifest_path, manifest_sha = _read_json_verified(
            row["branch_control_manifest_path"],
            row.get("branch_control_manifest_sha256"),
        )
        nominal, nominal_path, nominal_sha = _read_json_verified(
            manifest["nominal_control_path"],
            manifest.get("nominal_control_sha256"),
        )
        input_artifacts.append(
            {"path": _relative_or_absolute(manifest_path), "sha256": manifest_sha, "bytes": manifest_path.stat().st_size}
        )
        input_artifacts.append(
            {"path": _relative_or_absolute(nominal_path), "sha256": nominal_sha, "bytes": nominal_path.stat().st_size}
        )
        target_state = np.asarray(states.target, dtype=float)
        recorded_target = np.asarray(manifest.get("target_state", []), dtype=float)
        if recorded_target.shape != target_state.shape:
            raise RuntimeError(f"manifest target_state shape mismatch for {case_id}")
        target_state_max_abs_delta = float(np.max(np.abs(target_state - recorded_target)))
        nominal_controls = np.asarray(nominal["controls"], dtype=float)
        nominal_error = _replay_terminal_error(states.initial, target_state, cfg, nominal_controls)
        nominal_recorded = float(nominal.get("nominal_error", row.get("nominal_error")))
        replay_rows.append(
            _replay_row(
                suite_case_id=case_id,
                record_type="nominal",
                branch_order=None,
                mask_index=None,
                outage_mask=None,
                controls_path=nominal_path,
                controls_sha256=nominal_sha,
                recorded_terminal_error=nominal_recorded,
                replay_terminal_error=nominal_error,
                tolerance=float(args.tolerance),
                accepted_candidate=str(nominal.get("nominal_accepted_candidate", "")),
            )
        )

        branch_deltas: list[float] = []
        branch_entries = list(manifest.get("branch_control_sidecars", []))
        for entry in branch_entries:
            branch, branch_path, branch_sha = _read_json_verified(entry["path"], entry.get("sha256"))
            input_artifacts.append(
                {"path": _relative_or_absolute(branch_path), "sha256": branch_sha, "bytes": branch_path.stat().st_size}
            )
            branch_controls = np.asarray(branch["branch_controls"], dtype=float)
            replay_error = _replay_terminal_error(states.initial, target_state, cfg, branch_controls)
            recorded_error = float(branch["terminal_error"])
            branch_row = _replay_row(
                suite_case_id=case_id,
                record_type="branch",
                branch_order=int(branch["branch_order"]),
                mask_index=int(branch["mask_index"]),
                outage_mask=[int(value) for value in branch["outage_mask"]],
                controls_path=branch_path,
                controls_sha256=branch_sha,
                recorded_terminal_error=recorded_error,
                replay_terminal_error=replay_error,
                tolerance=float(args.tolerance),
                recovery_start=int(branch["recovery_start"]),
                recovery_segments=int(branch["recovery_segments"]),
                optimizer_ran=bool(branch.get("optimizer_ran", False)),
                optimizer_success=bool(branch.get("optimizer_success", False)),
                accepted_candidate=str(branch.get("accepted_candidate", "")),
                accepted_weight_variant=str(branch.get("accepted_branch_weight_variant_label", "")),
                accepted_initialization=str(branch.get("accepted_branch_initialization_label", "")),
            )
            branch_deltas.append(float(branch_row["terminal_error_delta"]))
            replay_rows.append(branch_row)

        case_rows = [item for item in replay_rows if str(item["suite_case_id"]) == case_id]
        nominal_delta = float(case_rows[0]["terminal_error_delta"])
        max_branch_delta = float(max(branch_deltas)) if branch_deltas else 0.0
        max_delta = float(max([float(item["terminal_error_delta"]) for item in case_rows], default=0.0))
        passes = bool(
            max_delta <= float(args.tolerance)
            and target_state_max_abs_delta <= float(args.tolerance)
            and len(branch_entries) == int(row.get("branch_control_sidecar_count", len(branch_entries)))
        )
        summary_rows.append(
            {
                "suite_case_id": case_id,
                "branch_row_count": int(len(branch_entries)),
                "nominal_terminal_error_delta": nominal_delta,
                "max_branch_terminal_error_delta": max_branch_delta,
                "max_terminal_error_delta": max_delta,
                "target_state_max_abs_delta": target_state_max_abs_delta,
                "tolerance": float(args.tolerance),
                "passes_tolerance": passes,
            }
        )

    if not replay_rows:
        raise RuntimeError("no branch-control replay-ready rows found in focused tail-coast recovery CSV")

    df = pd.DataFrame(replay_rows, columns=REPLAY_COLUMNS)
    replay_csv = results_dir / REPLAY_CSV_NAME
    df.to_csv(replay_csv, index=False)
    table_path = _write_table(summary_rows, tables_dir)
    branch_rows = df[df["record_type"] == "branch"]
    max_delta = float(df["terminal_error_delta"].astype(float).max()) if len(df) else 0.0
    max_branch_delta = float(branch_rows["terminal_error_delta"].astype(float).max()) if len(branch_rows) else 0.0
    metadata = {
        "command": " ".join(sys.argv),
        "row_count": int(len(df)),
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
        "fuel_optimality_claim": False,
        "production_solver_parity_claim": False,
        "quantum_advantage_claim": False,
        "replay_semantics": (
            "Normalized CR3BP replay of persisted accepted nominal and branch full-control schedules. "
            "The script compares replayed terminal errors to recorded terminal errors and does not "
            "rerun any optimizer or branch-control search."
        ),
        "limitations": tail_runner._branch_control_replay_limitations(),
        "summary": summary_rows,
        "input_artifacts": input_artifacts,
        "artifacts": {
            "tail_coast_branch_control_replay_csv": _relative_or_absolute(replay_csv),
            "tail_coast_branch_control_replay_metadata_json": _relative_or_absolute(results_dir / REPLAY_METADATA_NAME),
            "tail_coast_branch_control_replay_table_tex": _relative_or_absolute(table_path),
        },
    }
    _write_json(results_dir / REPLAY_METADATA_NAME, metadata)
    print(
        "Completed tail-coast branch-control replay "
        f"with {int(len(branch_rows))} branch rows, max_branch_delta={max_branch_delta:.3e}, "
        f"passes_tolerance={metadata['passes_tolerance']}.",
        flush=True,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay persisted hard-catalog tail-coast accepted branch controls under normalized CR3BP."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
