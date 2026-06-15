from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .experiment import load_configured_states, make_objective_config, output_directories
from .objective import Evaluator, outage_masks, schedule_to_string
from .refinement import refine_schedule
from .reporting import write_metadata


FEASIBILITY_COLUMNS = [
    "transfer_time",
    "amax",
    "segments",
    "max_nfev",
    "substeps_per_segment",
    "selected_outages",
    "multistart_enabled",
    "random_starts",
    "schedule",
    "base_nominal_error",
    "base_worst_error",
    "nominal_error",
    "selected_recovery_worst_error",
    "selected_worst_error",
    "all_outage_worst_error",
    "all_mask_worst_error",
    "nominal_threshold",
    "selected_recovery_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_recovery_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "optimizer_success",
    "refinement_success",
    "best_initial_guess",
    "cost",
    "nfev",
    "best_attempt_nfev",
    "runtime_seconds",
    "control_max_norm",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "message",
]


def _has_value(row: dict, key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _coalesce(row: dict, *keys: str, default=np.nan):
    for key in keys:
        if _has_value(row, key):
            return row[key]
    return default


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def normalize_feasibility_row(row: dict) -> dict:
    normalized = dict(row)
    selected_worst = _coalesce(
        normalized,
        "selected_recovery_worst_error",
        "selected_worst_error",
        "worst_error",
    )
    normalized["selected_recovery_worst_error"] = selected_worst
    normalized["selected_worst_error"] = _coalesce(
        normalized,
        "selected_worst_error",
        "selected_recovery_worst_error",
        "worst_error",
        default=selected_worst,
    )
    all_outage_worst = _coalesce(
        normalized,
        "all_outage_worst_error",
        "all_mask_worst_error",
        default=normalized["selected_worst_error"],
    )
    normalized["all_outage_worst_error"] = all_outage_worst
    normalized["all_mask_worst_error"] = _coalesce(
        normalized,
        "all_mask_worst_error",
        "all_outage_worst_error",
        default=all_outage_worst,
    )

    robust_threshold = _coalesce(
        normalized,
        "selected_recovery_threshold",
        "selected_worst_threshold",
    )
    normalized["selected_recovery_threshold"] = robust_threshold
    normalized["selected_worst_threshold"] = _coalesce(
        normalized,
        "selected_worst_threshold",
        "selected_recovery_threshold",
        default=robust_threshold,
    )

    if not _has_value(normalized, "meets_nominal_threshold"):
        normalized["meets_nominal_threshold"] = bool(
            float(normalized["nominal_error"]) <= float(normalized["nominal_threshold"])
        )
    if not _has_value(normalized, "meets_selected_recovery_threshold"):
        normalized["meets_selected_recovery_threshold"] = bool(
            float(normalized["selected_recovery_worst_error"]) <= float(normalized["selected_recovery_threshold"])
        )
    normalized["meets_selected_worst_threshold"] = _coalesce(
        normalized,
        "meets_selected_worst_threshold",
        "meets_selected_recovery_threshold",
        default=normalized["meets_selected_recovery_threshold"],
    )
    if not _has_value(normalized, "meets_thresholds"):
        normalized["meets_thresholds"] = bool(
            _as_bool(normalized["meets_nominal_threshold"])
            and _as_bool(normalized["meets_selected_recovery_threshold"])
        )

    defaults = {
        "multistart_enabled": False,
        "random_starts": 0,
        "selected_outage_indices": "[]",
        "selected_outage_errors": "[]",
        "all_outage_errors": "[]",
        "message": "",
    }
    for key, value in defaults.items():
        if not _has_value(normalized, key):
            normalized[key] = value
    return {column: normalized.get(column, np.nan) for column in FEASIBILITY_COLUMNS}


def normalize_feasibility_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FEASIBILITY_COLUMNS)
    rows = [normalize_feasibility_row(row) for row in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=FEASIBILITY_COLUMNS)


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def config_for_case(base_config: dict, transfer_time: float, amax: float, segments: int, max_nfev: int, args) -> dict:
    config = copy.deepcopy(base_config)
    config["benchmark"]["transfer_time"] = float(transfer_time)
    config["benchmark"]["amax"] = float(amax)
    config["benchmark"]["segments"] = int(segments)
    config["refinement"]["mode"] = "branch_recovery"
    config["refinement"]["max_nfev"] = int(max_nfev)
    config["refinement"]["selected_outages"] = int(args.selected_outages)
    config["refinement"]["outage_selection_min_recovery_segments"] = int(args.min_recovery_segments)
    for arg_name, cfg_name in [
        ("state_residual_weight", "state_residual_weight"),
        ("robust_residual_weight", "robust_residual_weight"),
        ("fuel_residual_weight", "fuel_residual_weight"),
        ("smooth_residual_weight", "smooth_residual_weight"),
        ("control_regularization", "control_regularization"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config["refinement"][cfg_name] = float(value)
    config["refinement"]["multistart"] = {
        "enabled": bool(args.multistart),
        "include_feedback": True,
        "include_zero": bool(args.multistart),
        "include_bang_bang": bool(args.include_bang_bang),
        "random_starts": int(args.random_starts) if args.multistart else 0,
        "low_amplitude_fraction": float(args.low_amplitude_fraction),
        "seed": int(args.multistart_seed),
    }
    return config


def run_case(base_config: dict, transfer_time: float, amax: float, segments: int, max_nfev: int, args) -> dict:
    config = config_for_case(base_config, transfer_time, amax, segments, max_nfev, args)
    states = load_configured_states(Path.cwd(), config, args.source_states)
    cfg = make_objective_config(config, states.mu)
    schedule = np.ones(cfg.n_segments, dtype=int)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = config["objective"]["thresholds"]

    evaluator = Evaluator(states.initial, states.target, cfg)
    base = evaluator.evaluate(schedule)
    start = time.perf_counter()
    refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, config["refinement"], thresholds)
    elapsed = time.perf_counter() - start
    controls = np.asarray(refined.get("controls", np.zeros((cfg.n_segments, 3))), dtype=float)
    max_norm = float(np.max(np.linalg.norm(controls, axis=1))) if controls.size else 0.0

    nominal_threshold = float(thresholds["nominal_success"])
    robust_threshold = float(thresholds["robust_success"])
    nominal_error = float(refined.get("nominal_error", np.inf))
    selected_worst = float(refined.get("selected_worst_error", refined.get("worst_error", np.inf)))
    return {
        "transfer_time": float(transfer_time),
        "amax": float(amax),
        "segments": int(segments),
        "max_nfev": int(max_nfev),
        "substeps_per_segment": int(cfg.substeps),
        "selected_outages": int(config["refinement"]["selected_outages"]),
        "multistart_enabled": bool(config["refinement"]["multistart"]["enabled"]),
        "random_starts": int(config["refinement"]["multistart"]["random_starts"]),
        "schedule": schedule_to_string(schedule),
        "base_nominal_error": float(base["nominal_error"]),
        "base_worst_error": float(base["worst_error"]),
        "nominal_error": nominal_error,
        "selected_recovery_worst_error": selected_worst,
        "selected_worst_error": selected_worst,
        "all_outage_worst_error": float(refined.get("all_mask_worst_error", selected_worst)),
        "all_mask_worst_error": float(refined.get("all_mask_worst_error", selected_worst)),
        "nominal_threshold": nominal_threshold,
        "selected_recovery_threshold": robust_threshold,
        "selected_worst_threshold": robust_threshold,
        "meets_nominal_threshold": bool(nominal_error <= nominal_threshold),
        "meets_selected_recovery_threshold": bool(selected_worst <= robust_threshold),
        "meets_selected_worst_threshold": bool(selected_worst <= robust_threshold),
        "meets_thresholds": bool(nominal_error <= nominal_threshold and selected_worst <= robust_threshold),
        "optimizer_success": bool(refined.get("optimizer_success", False)),
        "refinement_success": bool(refined.get("success", False)),
        "best_initial_guess": refined.get("best_initial_guess", "unknown"),
        "cost": float(refined.get("cost", np.nan)),
        "nfev": int(refined.get("nfev", 0) or 0),
        "best_attempt_nfev": int(refined.get("best_attempt_nfev", refined.get("nfev", 0)) or 0),
        "runtime_seconds": float(refined.get("runtime_seconds", elapsed) or elapsed),
        "control_max_norm": max_norm,
        "selected_outage_indices": json.dumps(refined.get("selected_outage_indices", [])),
        "selected_outage_errors": json.dumps(refined.get("selected_outage_errors", [])),
        "all_outage_errors": json.dumps(refined.get("all_outage_errors", [])),
        "message": str(refined.get("message", "")),
    }


def write_feasibility_table(df: pd.DataFrame, tables_dir: Path) -> None:
    ordered = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True])
    table = ordered[
        [
            "transfer_time",
            "amax",
            "segments",
            "max_nfev",
            "nominal_error",
            "selected_recovery_worst_error",
            "all_outage_worst_error",
            "meets_thresholds",
            "best_initial_guess",
            "nfev",
            "runtime_seconds",
        ]
    ]
    table.columns = [
        "Transfer time",
        "amax",
        "Segments",
        "Max nfev",
        "Nominal error",
        "Selected recovery worst error",
        "All-outage diagnostic worst error",
        "Meets thresholds",
        "Best start",
        "nfev",
        "Runtime (s)",
    ]
    tables_dir.joinpath("feasibility_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def plot_feasibility_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    agg = (
        df.sort_values(["selected_worst_error", "nominal_error"])
        .groupby(["transfer_time", "amax"], as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    values = agg["selected_worst_error"].to_numpy(dtype=float)
    scatter = ax.scatter(
        agg["transfer_time"],
        agg["amax"],
        c=values,
        s=140,
        cmap="viridis_r",
        edgecolors=np.where(agg["meets_thresholds"], "black", "white"),
        linewidths=np.where(agg["meets_thresholds"], 2.0, 0.8),
    )
    for _, row in agg.iterrows():
        label = f"N={int(row['segments'])}\n{row['selected_worst_error']:.2f}"
        ax.annotate(label, (row["transfer_time"], row["amax"]), textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.axhline(0.065, color="0.65", linestyle=":", linewidth=1)
    ax.set_xlabel("Transfer time")
    ax.set_ylabel("amax")
    ax.set_title("Best selected-worst recovery error by sweep point")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Selected-worst error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"feasibility_heatmap{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: Path, args) -> None:
    feasible = df[df["meets_thresholds"]].copy()
    best = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True]).head(5)
    summary = {
        "command": " ".join(sys.argv),
        "case_count": int(len(df)),
        "feasible_case_count": int(len(feasible)),
        "thresholds": {
            "nominal_success": float(df["nominal_threshold"].iloc[0]) if not df.empty else None,
            "selected_recovery_success": float(df["selected_worst_threshold"].iloc[0]) if not df.empty else None,
        },
        "grid": {
            "transfer_time": parse_float_grid(args.transfer_times),
            "amax": parse_float_grid(args.amax),
            "segments": parse_int_grid(args.segments),
            "max_nfev": parse_int_grid(args.max_nfev),
        },
        "best_cases": best.to_dict(orient="records"),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def regenerate_artifacts(df: pd.DataFrame, results_dir: Path, figures_dir: Path, tables_dir: Path, args, config: dict) -> None:
    if df.empty:
        return
    write_summary(df, results_dir / "summary_feasibility.json", args)
    write_feasibility_table(df, tables_dir)
    plot_feasibility_heatmap(df, figures_dir)
    write_metadata(
        results_dir / "feasibility_metadata.json",
        " ".join(sys.argv),
        config,
        {
            "feasibility_sweep_rows": int(len(df)),
            "feasibility_output": str(results_dir / "feasibility_sweep.csv"),
            "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_recovery_worst_error <= robust_success",
            "mask_semantics": "selected_recovery_worst_error/selected_worst_error evaluate only the selected outage masks optimized with branch recovery; all_outage_worst_error/all_mask_worst_error evaluate every outage mask as a diagnostic. selected_worst_error is retained as a backwards-compatible alias.",
        },
    )


def run(args) -> pd.DataFrame:
    root = Path.cwd()
    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    results_dir, figures_dir, tables_dir = output_directories(root, base_config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "feasibility_sweep.csv"

    rows: list[dict] = []
    completed: set[tuple[float, float, int, int]] = set()
    if args.resume and csv_path.exists():
        existing = normalize_feasibility_dataframe(pd.read_csv(csv_path))
        rows.extend(existing.to_dict(orient="records"))
        completed = {
            (float(row["transfer_time"]), float(row["amax"]), int(row["segments"]), int(row["max_nfev"]))
            for row in rows
        }

    cases = [
        (tf, amax, segments, max_nfev)
        for tf in parse_float_grid(args.transfer_times)
        for amax in parse_float_grid(args.amax)
        for segments in parse_int_grid(args.segments)
        for max_nfev in parse_int_grid(args.max_nfev)
    ]
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    for transfer_time, amax, segments, max_nfev in cases:
        key = (float(transfer_time), float(amax), int(segments), int(max_nfev))
        if key in completed:
            continue
        row = run_case(base_config, transfer_time, amax, segments, max_nfev, args)
        rows.append(row)
        df = normalize_feasibility_dataframe(pd.DataFrame(rows))
        df.to_csv(csv_path, index=False)
        regenerate_artifacts(df, results_dir, figures_dir, tables_dir, args, base_config)
        print(
            "case "
            f"tf={transfer_time} amax={amax} N={segments} max_nfev={max_nfev}: "
            f"nominal={row['nominal_error']:.4f}, selected_worst={row['selected_worst_error']:.4f}, "
            f"met={row['meets_thresholds']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    df = normalize_feasibility_dataframe(pd.DataFrame(rows))
    df.to_csv(csv_path, index=False)
    regenerate_artifacts(df, results_dir, figures_dir, tables_dir, args, base_config)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep cislunar all-windows branch-recovery feasibility.")
    parser.add_argument("--config", type=Path, default=Path("configs/q1_candidate.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--transfer-times", default="3.0,4.0,5.0")
    parser.add_argument("--amax", default="0.065,0.1,0.15,0.2")
    parser.add_argument("--segments", default="14,18,22")
    parser.add_argument("--max-nfev", default="120")
    parser.add_argument("--selected-outages", type=int, default=3)
    parser.add_argument("--min-recovery-segments", type=int, default=4)
    parser.add_argument("--multistart", action="store_true")
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--low-amplitude-fraction", type=float, default=0.35)
    parser.add_argument("--include-bang-bang", action="store_true")
    parser.add_argument("--multistart-seed", type=int, default=12345)
    parser.add_argument("--state-residual-weight", type=float, default=None)
    parser.add_argument("--robust-residual-weight", type=float, default=None)
    parser.add_argument("--fuel-residual-weight", type=float, default=None)
    parser.add_argument("--smooth-residual-weight", type=float, default=None)
    parser.add_argument("--control-regularization", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
