from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import load_configured_states, make_objective_config
from qlt.objective import outage_masks, string_to_schedule
from qlt.refinement import control_fuel, feedback_controls_for_schedule, refine_schedule
from qlt.reporting import package_versions, revision_metadata, write_json


DEFAULT_CONFIG = Path("configs/q1_phase_shift_cardinality.yaml")
DEFAULT_INPUT_DIR = Path("data/results/online_quantum_search")
DEFAULT_RESULTS_DIR = Path("data/results/online_branch_refinement")
DEFAULT_TABLES_DIR = Path("tables/online_branch_refinement")
DEFAULT_CASES = ("feedback_N16_k12", "feedback_N20_k16")
CANONICAL_LENGTH_UNIT_KM = 384400.0
CANONICAL_TIME_UNIT_SECONDS = 375190.259

OUTPUT_COLUMNS = [
    "case_id",
    "source_subspace_costs",
    "screening_rank",
    "subspace_index",
    "schedule_bits",
    "active_count",
    "M",
    "screening_objective",
    "screening_nominal_error",
    "screening_robust_worst_error",
    "screening_fuel",
    "branch_nominal_error",
    "branch_selected_worst_error",
    "branch_all_mask_worst_error",
    "branch_nominal_fuel",
    "optimizer_success",
    "success",
    "nfev",
    "best_attempt_nfev",
    "selected_outage_indices",
    "selected_outage_errors",
    "nominal_threshold",
    "robust_threshold",
    "runtime_seconds",
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _command_string() -> str:
    argv = getattr(sys, "orig_argv", sys.argv)
    return subprocess.list2cmdline([str(arg) for arg in argv])


def _artifact_path(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_case_id(case_id: str) -> tuple[int, int]:
    parts = str(case_id).split("_")
    if len(parts) != 3 or parts[0] != "feedback" or not parts[1].startswith("N") or not parts[2].startswith("k"):
        raise ValueError(f"unsupported feedback case id: {case_id}")
    return int(parts[1][1:]), int(parts[2][1:])


def load_feedback_case(input_dir: Path, case_id: str) -> pd.DataFrame:
    path = input_dir / f"{case_id}_subspace_costs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing feedback subspace costs: {path}")
    data = pd.read_csv(path, dtype={"schedule": str})
    required = {
        "schedule",
        "feedback_objective",
        "feedback_nominal_error",
        "feedback_worst_error",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    n_segments, active_count = parse_case_id(case_id)
    data = data.copy()
    data["schedule"] = data["schedule"].astype(str).str.zfill(n_segments)
    data["_source_index"] = range(len(data))
    data["_active_count"] = data["schedule"].map(lambda bits: bits.count("1"))
    bad = data[data["_active_count"] != active_count]
    if not bad.empty:
        raise ValueError(f"{path} contains schedules outside expected active count k={active_count}")
    return data


def select_top_screening_rows(case: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    ranked = case.sort_values(["feedback_objective", "feedback_worst_error", "schedule"], kind="mergesort").head(top_n)
    ranked = ranked.copy().reset_index(drop=True)
    ranked["screening_rank"] = range(1, len(ranked) + 1)
    return ranked


def config_for_case(config: dict, case_id: str) -> tuple[dict, int, int]:
    n_segments, active_count = parse_case_id(case_id)
    prepared = copy.deepcopy(config)
    prepared["benchmark"]["segments"] = int(n_segments)
    prepared["objective"]["target_active_fraction"] = float(active_count) / float(n_segments)
    return prepared, n_segments, active_count


def threshold_dimensional_bounds(config: dict) -> list[dict[str, float | str]]:
    position_scale = float(config["objective"]["position_scale"])
    velocity_scale = float(config["objective"]["velocity_scale"])
    velocity_unit_km_s = CANONICAL_LENGTH_UNIT_KM / CANONICAL_TIME_UNIT_SECONDS
    thresholds = config["objective"]["thresholds"]
    rows = []
    for key, label in [("nominal_success", "nominal"), ("robust_success", "robust")]:
        tau = float(thresholds[key])
        rows.append(
            {
                "threshold": label,
                "nondimensional_error": tau,
                "position_bound_km_if_velocity_error_zero": tau * position_scale * CANONICAL_LENGTH_UNIT_KM,
                "velocity_bound_m_per_s_if_position_error_zero": tau * velocity_scale * velocity_unit_km_s * 1000.0,
            }
        )
    return rows


def branch_refinement_row(
    *,
    case_id: str,
    source_path: Path,
    candidate: pd.Series,
    m_size: int,
    active_count: int,
    states,
    cfg,
    masks,
    refine_cfg: dict,
    thresholds: dict,
) -> dict:
    schedule_bits = str(candidate["schedule"])
    schedule = string_to_schedule(schedule_bits)
    screening_fuel = control_fuel(feedback_controls_for_schedule(schedule, states.initial, states.target, cfg), cfg.tf)
    refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, refine_cfg, thresholds)
    return serialize_refinement_row(
        case_id=case_id,
        source_path=source_path,
        candidate=candidate,
        m_size=m_size,
        active_count=active_count,
        screening_fuel=screening_fuel,
        refined=refined,
        thresholds=thresholds,
    )


def serialize_refinement_row(
    *,
    case_id: str,
    source_path: Path,
    candidate: pd.Series,
    m_size: int,
    active_count: int,
    screening_fuel: float,
    refined: dict,
    thresholds: dict,
) -> dict:
    selected = refined.get("selected_outage_indices", [])
    selected_errors = refined.get("selected_outage_errors", [])
    return {
        "case_id": case_id,
        "source_subspace_costs": _artifact_path(source_path),
        "screening_rank": int(candidate["screening_rank"]),
        "subspace_index": int(candidate["_source_index"]),
        "schedule_bits": str(candidate["schedule"]),
        "active_count": int(active_count),
        "M": int(m_size),
        "screening_objective": float(candidate["feedback_objective"]),
        "screening_nominal_error": float(candidate["feedback_nominal_error"]),
        "screening_robust_worst_error": float(candidate["feedback_worst_error"]),
        "screening_fuel": float(screening_fuel),
        "branch_nominal_error": float(refined.get("nominal_error", float("nan"))),
        "branch_selected_worst_error": float(refined.get("selected_worst_error", refined.get("worst_error", float("nan")))),
        "branch_all_mask_worst_error": float(refined.get("all_mask_worst_error", float("nan"))),
        "branch_nominal_fuel": float(refined.get("nominal_fuel", float("nan"))),
        "optimizer_success": bool(refined.get("optimizer_success", False)),
        "success": bool(refined.get("success", False)),
        "nfev": int(refined.get("nfev", 0) or 0),
        "best_attempt_nfev": int(refined.get("best_attempt_nfev", refined.get("nfev", 0)) or 0),
        "selected_outage_indices": json.dumps([int(value) for value in selected]),
        "selected_outage_errors": json.dumps([float(value) for value in selected_errors]),
        "nominal_threshold": float(thresholds["nominal_success"]),
        "robust_threshold": float(thresholds["robust_success"]),
        "runtime_seconds": float(refined.get("runtime_seconds", 0.0) or 0.0),
    }


def write_latex_table(rows: pd.DataFrame, tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "online_branch_refinement_table.tex"
    table = rows[
        [
            "case_id",
            "M",
            "schedule_bits",
            "screening_objective",
            "screening_nominal_error",
            "screening_robust_worst_error",
            "branch_nominal_error",
            "branch_selected_worst_error",
            "branch_all_mask_worst_error",
            "branch_nominal_fuel",
            "optimizer_success",
            "success",
            "nfev",
            "selected_outage_indices",
        ]
    ].copy()
    table.columns = [
        "Case",
        "M",
        "Schedule",
        "Screen obj.",
        "Screen nom.",
        "Screen worst",
        "Branch nom.",
        "Selected worst",
        "All-mask worst",
        "Nom. fuel",
        "Opt. success",
        "Pass",
        "nfev",
        "Outage(s)",
    ]
    table.to_latex(path, index=False, float_format="%.6f", escape=True)
    return path


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    input_dir = _resolve(args.input_dir)
    results_dir = _resolve(args.results_dir)
    tables_dir = _resolve(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    rows = []
    case_metadata = []
    for case_id in args.cases:
        case_start = time.perf_counter()
        case_data = load_feedback_case(input_dir, case_id)
        selected = select_top_screening_rows(case_data, int(args.top_n))
        case_config, n_segments, active_count = config_for_case(config, case_id)
        states = load_configured_states(ROOT, case_config)
        cfg = make_objective_config(case_config, states.mu)
        masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
        thresholds = case_config["objective"]["thresholds"]
        refine_cfg = copy.deepcopy(case_config["refinement"])
        if args.max_nfev is not None:
            refine_cfg["max_nfev"] = int(args.max_nfev)
        source_path = input_dir / f"{case_id}_subspace_costs.csv"
        for _, candidate in selected.iterrows():
            row = branch_refinement_row(
                case_id=case_id,
                source_path=source_path,
                candidate=candidate,
                m_size=len(case_data),
                active_count=active_count,
                states=states,
                cfg=cfg,
                masks=masks,
                refine_cfg=refine_cfg,
                thresholds=thresholds,
            )
            rows.append(row)
            print(
                f"{case_id} rank {row['screening_rank']} bits={row['schedule_bits']} "
                f"success={row['success']} opt={row['optimizer_success']} "
                f"nom={row['branch_nominal_error']:.6f} selected={row['branch_selected_worst_error']:.6f} "
                f"all={row['branch_all_mask_worst_error']:.6f} nfev={row['nfev']}",
                flush=True,
            )
        case_metadata.append(
            {
                "case_id": case_id,
                "n_segments": int(n_segments),
                "active_count": int(active_count),
                "M": int(len(case_data)),
                "top_n": int(len(selected)),
                "runtime_seconds": float(time.perf_counter() - case_start),
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    csv_path = results_dir / "online_branch_refinement.csv"
    metadata_path = results_dir / "online_branch_refinement_metadata.json"
    df.to_csv(csv_path, index=False)
    table_path = write_latex_table(df, tables_dir)
    metadata = {
        "command": _command_string(),
        "python": sys.version,
        "packages": package_versions(["numpy", "pandas", "pyyaml", "scipy"]),
        **revision_metadata(),
        "config": _artifact_path(config_path),
        "input_dir": _artifact_path(input_dir),
        "cases": case_metadata,
        "default_policy": (
            "Default run refines only the top feedback-screening schedule per large case with the configured "
            "branch_recovery max_nfev cap; no multistart expansion is enabled."
        ),
        "thresholds": copy.deepcopy(config["objective"]["thresholds"]),
        "threshold_dimensional_bounds": threshold_dimensional_bounds(config),
        "threshold_dimensional_caveat": (
            "The reported error is sqrt((||dr||/position_scale)^2 + (||dv||/velocity_scale)^2). "
            "Dimensional rows are component upper-bound interpretations when the other component is zero, "
            "not flight-dynamics targeting tolerances."
        ),
        "claim_limits": [
            "This is a targeted diagnostic for feedback-screening winners, not a full branch-refined enumeration.",
            "The larger online-search rows remain screening evidence unless branch-refined feasible seeds are demonstrated.",
            "Optimizer status is reported separately from threshold pass/fail.",
            "All-mask worst is a diagnostic over outage masks not selected for refinement.",
        ],
        "runtime_seconds": float(time.perf_counter() - start),
        "artifact_paths": {
            "csv": _artifact_path(csv_path),
            "metadata_json": _artifact_path(metadata_path),
            "table_tex": _artifact_path(table_path),
        },
    }
    write_json(metadata_path, metadata)
    return {"rows": df, "metadata": metadata}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Branch-refine top large online-search feedback-screening schedules.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
