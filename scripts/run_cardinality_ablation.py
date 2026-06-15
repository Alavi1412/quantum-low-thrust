from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import load_configured_states, make_objective_config, state_target_mode
from qlt.objective import Evaluator, outage_masks, schedule_to_string
from qlt.refinement import refine_schedule
from qlt.reporting import package_versions, revision_metadata, write_json


DEFAULT_CONFIG = Path("configs/q1_phase_shift_cardinality.yaml")
DEFAULT_FUEL_WEIGHTS = [0.018, 0.03, 0.05, 0.08, 0.12, 0.20]
METHODS_OF_INTEREST = ["surrogate_qubo_sa", "cross_entropy", "genetic", "true_sa", "qaoa_statevector", "random"]
FUEL_TARGET = 0.091667


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enumerate_cardinality_schedules(n_segments: int, active_windows: int) -> list[np.ndarray]:
    """Enumerate binary schedules by inactive-window combinations in deterministic order."""
    n_segments = int(n_segments)
    active_windows = int(active_windows)
    if active_windows < 0 or active_windows > n_segments:
        raise ValueError("active_windows must be between 0 and n_segments")
    schedules: list[np.ndarray] = []
    for inactive in combinations(range(n_segments), n_segments - active_windows):
        schedule = np.ones(n_segments, dtype=int)
        schedule[list(inactive)] = 0
        schedules.append(schedule)
    return schedules


def schedule_active_count(bits: str) -> int:
    return sum(char == "1" for char in str(bits).strip())


def normalize_schedule_bits(value, n_segments: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    bits = str(value).strip()
    if bits.endswith(".0") and bits[:-2].isdigit():
        bits = bits[:-2]
    if not bits:
        return None
    if any(char not in {"0", "1"} for char in bits):
        return None
    if len(bits) > n_segments:
        return None
    return bits.zfill(n_segments)


def coast_windows(schedule: np.ndarray) -> str:
    inactive = np.flatnonzero(np.asarray(schedule, dtype=int) == 0)
    return ",".join(str(int(i)) for i in inactive) if inactive.size else "-"


def validate_requested_setup(config: dict, cfg) -> None:
    benchmark = config["benchmark"]
    thresholds = config["objective"]["thresholds"]
    expected_active = 11.0 / 12.0
    checks = [
        (str(benchmark.get("target_mode")) == "catalog_halo_phase_shift", "benchmark.target_mode must be catalog_halo_phase_shift"),
        (int(benchmark["segments"]) == 12, "benchmark.segments must be 12"),
        (cfg.n_segments == 12, "objective config must have 12 segments"),
        (abs(float(config["objective"].get("target_active_fraction")) - expected_active) < 1e-12, "target_active_fraction must be 11/12"),
        (abs(float(thresholds["nominal_success"]) - 0.09) < 1e-12, "nominal_success threshold must be 0.09"),
        (abs(float(thresholds["robust_success"]) - 0.17) < 1e-12, "robust_success threshold must be 0.17"),
        (str(config["refinement"].get("mode")) == "branch_recovery", "refinement.mode must be branch_recovery"),
    ]
    failures = [message for ok, message in checks if not ok]
    if failures:
        raise ValueError("phase-shift cardinality ablation requires the existing non-teacher setup: " + "; ".join(failures))


def finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.median(finite)) if finite.size else float("nan")


def refine_row(
    *,
    schedule: np.ndarray,
    group: str,
    states,
    cfg,
    masks: np.ndarray,
    evaluator: Evaluator,
    base_refine_cfg: dict,
    thresholds: dict,
    fuel_residual_weight: float | None = None,
    max_nfev: int | None = None,
) -> dict:
    refine_cfg = copy.deepcopy(base_refine_cfg)
    if fuel_residual_weight is not None:
        refine_cfg["fuel_residual_weight"] = float(fuel_residual_weight)
    if max_nfev is not None:
        refine_cfg["max_nfev"] = int(max_nfev)

    schedule = np.asarray(schedule, dtype=int)
    base = evaluator.evaluate(schedule)
    refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, refine_cfg, thresholds)
    nominal_error = float(refined.get("nominal_error", np.nan))
    selected_worst = float(refined.get("selected_worst_error", refined.get("worst_error", np.nan)))
    all_mask_worst = float(refined.get("all_mask_worst_error", refined.get("worst_error", np.nan)))
    success = bool(nominal_error <= thresholds["nominal_success"] and selected_worst <= thresholds["robust_success"])
    all_mask_success = bool(nominal_error <= thresholds["nominal_success"] and all_mask_worst <= thresholds["robust_success"])
    return {
        "group": group,
        "k_active": int(np.count_nonzero(schedule)),
        "active_fraction": float(np.mean(schedule)),
        "schedule": schedule_to_string(schedule),
        "coast_windows_zero_based": coast_windows(schedule),
        "fuel_residual_weight": float(refine_cfg.get("fuel_residual_weight", np.nan)),
        "max_nfev": int(refine_cfg.get("max_nfev", 0)),
        "true_objective": float(base["objective"]),
        "feedback_nominal_error": float(base["nominal_error"]),
        "feedback_worst_error": float(base["worst_error"]),
        "refinement_success": success,
        "all_mask_success": all_mask_success,
        "refined_nominal_error": nominal_error,
        "refined_selected_worst_error": selected_worst,
        "refined_all_mask_worst_error": all_mask_worst,
        "refined_nominal_fuel": float(refined.get("nominal_fuel", np.nan)),
        "refined_recovery_fuel_mean": float(refined.get("recovery_fuel_mean", np.nan)),
        "refined_recovery_fuel_max": float(refined.get("recovery_fuel_max", np.nan)),
        "refinement_nfev": int(refined.get("nfev", 0) or 0),
        "refinement_runtime_seconds": float(refined.get("runtime_seconds", 0.0) or 0.0),
        "best_initial_guess": refined.get("best_initial_guess"),
        "selected_outage_indices": json.dumps(refined.get("selected_outage_indices", [])),
        "selected_outage_errors": json.dumps(refined.get("selected_outage_errors", [])),
        "all_outage_errors": json.dumps(refined.get("all_outage_errors", [])),
    }


def mark_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "refined_nominal_fuel",
    y_col: str = "refined_selected_worst_error",
    feasible_col: str = "refinement_success",
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    candidate_mask = df[feasible_col].fillna(False).astype(bool)
    if not candidate_mask.any():
        candidate_mask = pd.Series(True, index=df.index)
    frontier = pd.Series(False, index=df.index)
    candidates = df.loc[candidate_mask, [x_col, y_col]].astype(float)
    eps = 1e-12
    for idx, row in candidates.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        dominated = (
            (candidates[x_col] <= x + eps)
            & (candidates[y_col] <= y + eps)
            & ((candidates[x_col] < x - eps) | (candidates[y_col] < y - eps))
        ).any()
        frontier.loc[idx] = not bool(dominated)
    return frontier


def summarize_cardinality(cardinality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k_active, group in cardinality.groupby("k_active", sort=True):
        feasible = group[group["refinement_success"]]
        source = feasible if len(feasible) else group
        best_selected = source.sort_values(["refined_selected_worst_error", "refined_nominal_error", "refined_nominal_fuel"]).iloc[0]
        min_fuel = source.sort_values(["refined_nominal_fuel", "refined_selected_worst_error", "refined_nominal_error"]).iloc[0]
        rows.append(
            {
                "k_active": int(k_active),
                "schedules_refined": int(len(group)),
                "success_count": int(group["refinement_success"].sum()),
                "all_mask_success_count": int(group["all_mask_success"].sum()),
                "best_nominal_error": float(group["refined_nominal_error"].min()),
                "median_nominal_error": finite_median(group["refined_nominal_error"]),
                "best_selected_worst_error": float(group["refined_selected_worst_error"].min()),
                "median_selected_worst_error": finite_median(group["refined_selected_worst_error"]),
                "best_all_mask_worst_error": float(group["refined_all_mask_worst_error"].min()),
                "median_all_mask_worst_error": finite_median(group["refined_all_mask_worst_error"]),
                "best_nominal_fuel": float(group["refined_nominal_fuel"].min()),
                "median_nominal_fuel": finite_median(group["refined_nominal_fuel"]),
                "best_schedule_by_selected_worst": str(best_selected["schedule"]),
                "min_fuel_feasible_schedule": str(min_fuel["schedule"]),
            }
        )
    return pd.DataFrame(rows)


def load_method_results(path: Path, n_segments: int) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in [
        "refined_nominal_error",
        "refined_selected_worst_error",
        "refined_all_mask_worst_error",
        "refined_nominal_fuel",
    ]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    schedule_col = "refinement_candidate_schedule" if "refinement_candidate_schedule" in raw.columns else "schedule"
    raw["normalized_schedule"] = raw[schedule_col].map(lambda value: normalize_schedule_bits(value, n_segments))
    raw["normalized_active_count"] = raw["normalized_schedule"].map(lambda bits: schedule_active_count(bits) if bits else np.nan)
    return raw


def compare_methods_to_frontier(
    method_results: pd.DataFrame,
    one_coast: pd.DataFrame,
    near_selected_tolerance: float,
) -> pd.DataFrame:
    one_coast_by_schedule = one_coast.set_index("schedule")
    feasible = one_coast[one_coast["refinement_success"]]
    frontier = one_coast[one_coast["one_coast_pareto_frontier"]]
    best_selected = float(feasible["refined_selected_worst_error"].min()) if len(feasible) else float(one_coast["refined_selected_worst_error"].min())
    frontier_schedules = set(frontier["schedule"])
    rows = []
    for method in METHODS_OF_INTEREST:
        group = method_results[method_results["method"] == method].copy()
        if group.empty:
            continue
        one_coast_group = group[group["normalized_active_count"] == 11].copy()
        mapped_rows = []
        for _, method_row in one_coast_group.iterrows():
            schedule = method_row["normalized_schedule"]
            if schedule in one_coast_by_schedule.index:
                mapped = one_coast_by_schedule.loc[schedule]
                mapped_rows.append(
                    {
                        "schedule": schedule,
                        "exhaustive_success": bool(mapped["refinement_success"]),
                        "on_one_coast_pareto_frontier": schedule in frontier_schedules,
                        "near_best_selected": float(mapped["refined_selected_worst_error"]) <= best_selected + near_selected_tolerance,
                        "exhaustive_selected_worst_error": float(mapped["refined_selected_worst_error"]),
                        "exhaustive_nominal_fuel": float(mapped["refined_nominal_fuel"]),
                    }
                )
        mapped_df = pd.DataFrame(mapped_rows)
        counts = one_coast_group["normalized_schedule"].value_counts()
        dominant = "; ".join(f"{schedule}:{count}" for schedule, count in counts.items()) if len(counts) else "-"
        rows.append(
            {
                "method": method,
                "runs": int(len(group)),
                "one_coast_runs": int(len(one_coast_group)),
                "one_coast_success_runs": int(mapped_df["exhaustive_success"].sum()) if len(mapped_df) else 0,
                "one_coast_frontier_runs": int(mapped_df["on_one_coast_pareto_frontier"].sum()) if len(mapped_df) else 0,
                "near_best_selected_runs": int(mapped_df["near_best_selected"].sum()) if len(mapped_df) else 0,
                "median_method_selected_worst_error": finite_median(one_coast_group["refined_selected_worst_error"]) if len(one_coast_group) else float("nan"),
                "median_method_nominal_fuel": finite_median(one_coast_group["refined_nominal_fuel"]) if len(one_coast_group) else float("nan"),
                "median_exhaustive_selected_worst_error": finite_median(mapped_df["exhaustive_selected_worst_error"]) if len(mapped_df) else float("nan"),
                "median_exhaustive_nominal_fuel": finite_median(mapped_df["exhaustive_nominal_fuel"]) if len(mapped_df) else float("nan"),
                "median_selected_gap_to_best_one_coast": finite_median(mapped_df["exhaustive_selected_worst_error"] - best_selected) if len(mapped_df) else float("nan"),
                "dominant_one_coast_schedules": dominant,
            }
        )
    return pd.DataFrame(rows)


def write_latex_outputs(
    cardinality: pd.DataFrame,
    cardinality_summary: pd.DataFrame,
    fuel_sweep: pd.DataFrame,
    method_vs_frontier: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    one_coast_cols = [
        "schedule",
        "coast_windows_zero_based",
        "refinement_success",
        "one_coast_pareto_frontier",
        "refined_nominal_error",
        "refined_selected_worst_error",
        "refined_all_mask_worst_error",
        "refined_nominal_fuel",
        "refinement_nfev",
    ]
    one_coast = cardinality[cardinality["k_active"] == 11].copy()
    one_coast[one_coast_cols].to_latex(
        tables_dir / "one_coast_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )
    cardinality_summary.to_latex(
        tables_dir / "high_duty_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )
    fuel_cols = [
        "fuel_residual_weight",
        "refinement_success",
        "all_mask_success",
        "refined_nominal_error",
        "refined_selected_worst_error",
        "refined_all_mask_worst_error",
        "refined_nominal_fuel",
        "refinement_nfev",
    ]
    fuel_sweep[fuel_cols].to_latex(
        tables_dir / "fuel_pareto_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )
    if not method_vs_frontier.empty:
        method_vs_frontier.to_latex(
            tables_dir / "method_vs_frontier_table.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )


def plot_pareto(
    cardinality: pd.DataFrame,
    fuel_sweep: pd.DataFrame,
    method_vs_frontier: pd.DataFrame,
    thresholds: dict,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.4, 5.2))
    for k_active, group in cardinality.groupby("k_active", sort=True):
        plt.scatter(
            group["refined_nominal_fuel"],
            group["refined_selected_worst_error"],
            s=46,
            alpha=0.72,
            label=f"k={int(k_active)} schedules",
        )
    plt.plot(
        fuel_sweep["refined_nominal_fuel"],
        fuel_sweep["refined_selected_worst_error"],
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=5,
        label="all-windows fuel sweep",
    )
    if not method_vs_frontier.empty:
        plt.scatter(
            method_vs_frontier["median_method_nominal_fuel"],
            method_vs_frontier["median_method_selected_worst_error"],
            marker="x",
            s=80,
            linewidths=1.8,
            color="black",
            label="method medians",
        )
    plt.axhline(float(thresholds["robust_success"]), color="0.25", linestyle=":", linewidth=1.2, label="robust threshold")
    plt.axvline(FUEL_TARGET, color="0.45", linestyle="-.", linewidth=1.0, label="0.091667 fuel")
    plt.xlabel("Refined nominal fuel")
    plt.ylabel("Selected recovery worst error")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "fuel_error_pareto.png", dpi=220)
    plt.savefig(figures_dir / "fuel_error_pareto.pdf")
    plt.close()


def run_ablation(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    states = load_configured_states(ROOT, config)
    cfg = make_objective_config(config, states.mu)
    validate_requested_setup(config, cfg)

    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else ROOT / args.figures_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else ROOT / args.tables_dir
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    thresholds = config["objective"]["thresholds"]
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    evaluator = Evaluator(states.initial, states.target, cfg)
    base_refine_cfg = copy.deepcopy(config["refinement"])
    cardinality_rows = []

    k10_schedules = enumerate_cardinality_schedules(cfg.n_segments, 10)
    k10_group = "k10_full"
    if args.k10_mode == "skip":
        k10_schedules = []
        k10_group = "k10_skipped"
    elif args.k10_mode == "top":
        ranked = sorted(k10_schedules, key=lambda schedule: evaluator.evaluate(schedule)["objective"])
        k10_schedules = ranked[: int(args.k10_top_count)]
        k10_group = f"k10_top_{len(k10_schedules)}_by_true_objective"

    cardinality_plan = [
        (k10_group, k10_schedules, args.k10_max_nfev),
        ("one_coast_k11_full", enumerate_cardinality_schedules(cfg.n_segments, 11), args.max_nfev),
        ("all_windows_k12", enumerate_cardinality_schedules(cfg.n_segments, 12), args.max_nfev),
    ]
    total_cardinality = sum(len(schedules) for _, schedules, _ in cardinality_plan)
    completed = 0
    for group, schedules, max_nfev in cardinality_plan:
        for schedule in schedules:
            completed += 1
            print(
                f"[cardinality {completed}/{total_cardinality}] {group} {schedule_to_string(schedule)}",
                flush=True,
            )
            cardinality_rows.append(
                refine_row(
                    schedule=schedule,
                    group=group,
                    states=states,
                    cfg=cfg,
                    masks=masks,
                    evaluator=evaluator,
                    base_refine_cfg=base_refine_cfg,
                    thresholds=thresholds,
                    max_nfev=max_nfev,
                )
            )

    fuel_rows = []
    all_windows = np.ones(cfg.n_segments, dtype=int)
    for index, weight in enumerate(args.fuel_weights, start=1):
        print(f"[fuel sweep {index}/{len(args.fuel_weights)}] weight={weight:g}", flush=True)
        fuel_rows.append(
            refine_row(
                schedule=all_windows,
                group="all_windows_fuel_sweep",
                states=states,
                cfg=cfg,
                masks=masks,
                evaluator=evaluator,
                base_refine_cfg=base_refine_cfg,
                thresholds=thresholds,
                fuel_residual_weight=float(weight),
                max_nfev=args.max_nfev,
            )
        )

    cardinality = pd.DataFrame(cardinality_rows)
    fuel_sweep = pd.DataFrame(fuel_rows)
    one_coast_mask = cardinality["k_active"] == 11
    cardinality["one_coast_pareto_frontier"] = False
    cardinality.loc[one_coast_mask, "one_coast_pareto_frontier"] = mark_pareto_frontier(cardinality.loc[one_coast_mask]).to_numpy()
    cardinality_summary = summarize_cardinality(cardinality)

    method_path = args.method_results if args.method_results.is_absolute() else ROOT / args.method_results
    if method_path.exists():
        method_results = load_method_results(method_path, cfg.n_segments)
        method_vs_frontier = compare_methods_to_frontier(
            method_results,
            cardinality.loc[one_coast_mask].copy(),
            near_selected_tolerance=float(args.near_selected_tolerance),
        )
    else:
        method_vs_frontier = pd.DataFrame()

    cardinality.to_csv(results_dir / "cardinality_refinements.csv", index=False)
    cardinality_summary.to_csv(results_dir / "cardinality_summary.csv", index=False)
    fuel_sweep.to_csv(results_dir / "fuel_sweep.csv", index=False)
    method_vs_frontier.to_csv(results_dir / "method_vs_frontier.csv", index=False)
    write_latex_outputs(cardinality, cardinality_summary, fuel_sweep, method_vs_frontier, tables_dir)
    if not args.no_figure:
        plot_pareto(cardinality, fuel_sweep, method_vs_frontier, thresholds, figures_dir)

    feasible_fuel_sweep = fuel_sweep[
        (fuel_sweep["refinement_success"])
        & (fuel_sweep["refined_nominal_fuel"] <= FUEL_TARGET)
    ]
    runtime_seconds = float(time.perf_counter() - start)
    metadata = {
        "command": " ".join(sys.argv),
        "python": sys.version,
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas", "pyyaml"]),
        **revision_metadata(),
        "config_path": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "config_sha256": config_hash(config_path),
        "target_mode": state_target_mode(states),
        "thresholds": thresholds,
        "n_segments": int(cfg.n_segments),
        "target_active_fraction": float(config["objective"]["target_active_fraction"]),
        "refinement_mode": str(config["refinement"].get("mode")),
        "base_max_nfev": int(config["refinement"]["max_nfev"]),
        "k10_mode": args.k10_mode,
        "k10_schedule_count": int(len(k10_schedules)),
        "k10_total_possible": int(len(enumerate_cardinality_schedules(cfg.n_segments, 10))),
        "k10_runtime_compromise": None
        if args.k10_mode == "full" and args.k10_max_nfev is None
        else {
            "mode": args.k10_mode,
            "top_count": int(args.k10_top_count),
            "max_nfev": args.k10_max_nfev,
            "label": k10_group,
        },
        "one_coast_schedule_count": int(one_coast_mask.sum()),
        "all_windows_schedule_count": 1,
        "fuel_sweep_weights": [float(weight) for weight in args.fuel_weights],
        "near_frontier_definition": f"one-coast selected worst error within {args.near_selected_tolerance:g} of the best feasible one-coast selected worst error",
        "fuel_target": FUEL_TARGET,
        "all_windows_fuel_sweep_meets_fuel_target_and_thresholds": bool(len(feasible_fuel_sweep) > 0),
        "runtime_seconds": runtime_seconds,
        "interpretation_limits": [
            "Branch-recovery success uses the selected outage subset; all-mask worst error is reported as a diagnostic.",
            "The high-duty enumeration changes only binary availability; continuous controls are re-optimized from the same configured refinement model.",
            "The method-vs-frontier comparison maps existing method schedules to the exhaustive one-coast refinement table and does not rerun the discrete optimizers.",
        ],
    }
    write_json(results_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-shift cardinality ablation and all-windows fuel Pareto sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/phase_shift_cardinality_ablation"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/phase_shift_cardinality_ablation"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables/phase_shift_cardinality_ablation"))
    parser.add_argument("--method-results", type=Path, default=Path("data/results/phase_shift_cardinality/raw_results.csv"))
    parser.add_argument("--fuel-weights", type=float, nargs="+", default=DEFAULT_FUEL_WEIGHTS)
    parser.add_argument("--max-nfev", type=int, default=None, help="Override max_nfev for k=11, k=12, and fuel sweep.")
    parser.add_argument("--k10-mode", choices=["full", "top", "skip"], default="full")
    parser.add_argument("--k10-top-count", type=int, default=12)
    parser.add_argument("--k10-max-nfev", type=int, default=None, help="Optional k=10-only max_nfev compromise.")
    parser.add_argument("--near-selected-tolerance", type=float, default=0.005)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run_ablation(parse_args())
    print(f"completed phase-shift cardinality ablation in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
