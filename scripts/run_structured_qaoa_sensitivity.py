from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.reporting import package_versions, revision_metadata, write_json
from qlt.structured_qaoa import adjacency_mixer, cost_biased_initial_probabilities, optimize_subspace_qaoa


DEFAULT_CARDINALITY = Path("data/results/phase_shift_cardinality_ablation/cardinality_refinements.csv")
DEFAULT_GENERIC_QAOA_SUMMARY = Path("data/results/phase_shift_cardinality_30seed/summary.csv")
DEFAULT_RESULTS_DIR = Path("data/results/structured_qaoa_sensitivity")
DEFAULT_FIGURES_DIR = Path("figures/structured_qaoa_sensitivity")
DEFAULT_TABLES_DIR = Path("tables/structured_qaoa_sensitivity")
DEFAULT_GROUP = "one_coast_k11_full"
DEFAULT_DEPTHS = [1, 2, 3, 4]
DEFAULT_RESTART_GRID = [1, 4, 8]
DEFAULT_TOPOLOGIES = ["path", "cycle", "complete"]
DEFAULT_WARM_START_MODES = ["inverse", "exponential"]


def _matched_uniform_method(mixer_topology: str, depth: int, restarts: int) -> str:
    return f"uniform_match_{mixer_topology}_p{int(depth)}_r{int(restarts)}"


MATCHED_COMPARATOR = {
    "method": _matched_uniform_method("complete", 4, 8),
    "mixer_topology": "complete",
    "warm_start_mode": "uniform",
    "depth": 4,
    "restarts": 8,
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _command_string() -> str:
    argv = getattr(sys, "orig_argv", sys.argv)
    return subprocess.list2cmdline([str(arg) for arg in argv])


def _python_launcher() -> str:
    return f"py -{sys.version_info.major}.{sys.version_info.minor}"


def _load_cardinality_group(path: Path, group: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"schedule": str})
    subspace = data[data["group"] == str(group)].copy()
    if subspace.empty:
        available = ", ".join(sorted(data["group"].dropna().astype(str).unique().tolist()))
        raise RuntimeError(f"cardinality group {group!r} was not found in {path}; available groups: {available}")
    subspace["schedule"] = subspace["schedule"].astype(str).str.zfill(12)
    if str(group) == "one_coast_k11_full":
        subspace["_sort_key"] = pd.to_numeric(subspace["coast_windows_zero_based"], errors="raise")
    else:
        subspace["_sort_key"] = subspace["schedule"]
    subspace = subspace.sort_values("_sort_key", kind="mergesort").drop(columns="_sort_key").reset_index(drop=True)
    subspace["state_index"] = np.arange(len(subspace), dtype=int)
    subspace["state_label"] = subspace["coast_windows_zero_based"].astype(str)
    return subspace


def _pareto_frontier_schedules(subspace: pd.DataFrame) -> set[str]:
    if "one_coast_pareto_frontier" in subspace.columns and subspace["one_coast_pareto_frontier"].astype(bool).any():
        return set(subspace.loc[subspace["one_coast_pareto_frontier"].astype(bool), "schedule"].astype(str))
    costs = subspace[["refined_selected_worst_error", "refined_nominal_error"]].to_numpy(dtype=float)
    frontier: set[str] = set()
    schedules = subspace["schedule"].astype(str).to_list()
    for index, candidate in enumerate(costs):
        dominated = False
        for other_index, other in enumerate(costs):
            if other_index == index:
                continue
            if np.all(other <= candidate) and np.any(other < candidate):
                dominated = True
                break
        if not dominated:
            frontier.add(schedules[index])
    return frontier


def _normalise_cost(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0:
        raise ValueError("structured-QAOA costs must not be constant")
    return (values - low) / span, low, span


def _clean_outputs(results_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    patterns = {
        results_dir: [
            "summary.csv",
            "summary.json",
            "restart_grid.csv",
            "restart_grid.json",
            "matched_uniform_restart_grid.csv",
            "matched_uniform_restart_grid.json",
            "structured_vs_matched_uniform.csv",
            "structured_vs_matched_uniform.json",
            "structured_vs_matched_uniform_family.csv",
            "structured_vs_matched_uniform_family.json",
            "distribution.csv",
            "distribution.json",
            "generic_qaoa_summary.csv",
            "sensitivity_table.csv",
            "sensitivity_table.json",
            "metadata.json",
            "structured_vs_generic_qaoa.csv",
            "structured_vs_generic_qaoa.json",
        ],
        figures_dir: [
            "structured_qaoa_sensitivity_heatmap.png",
            "structured_qaoa_sensitivity_heatmap.pdf",
        ],
        tables_dir: [
            "structured_qaoa_sensitivity_table.tex",
        ],
    }
    for directory, names in patterns.items():
        for name in names:
            for path in directory.glob(name):
                if path.is_file():
                    path.unlink()


def _as_list(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in np.asarray(values, dtype=float)])


def _summary_base(
    *,
    method: str,
    probabilities: np.ndarray,
    selected_worst: np.ndarray,
    nominal_error: np.ndarray,
    schedules: list[str],
    frontier_schedules: set[str],
) -> dict:
    best_index = int(np.argmin(selected_worst))
    order = np.argsort(-probabilities, kind="mergesort")
    top_index = int(order[0])
    frontier_probability = float(sum(probabilities[index] for index, schedule in enumerate(schedules) if schedule in frontier_schedules))
    top3 = order[:3]
    return {
        "method": method,
        "states_in_support": int(probabilities.size),
        "feasible_support_probability": 1.0,
        "expected_selected_worst_error": float(probabilities @ selected_worst),
        "expected_nominal_error": float(probabilities @ nominal_error),
        "probability_best_one_coast": float(probabilities[best_index]),
        "probability_best_state": float(probabilities[best_index]),
        "probability_pareto_frontier": frontier_probability,
        "top_schedule": schedules[top_index],
        "top_schedule_probability": float(probabilities[top_index]),
        "top_schedule_selected_worst_error": float(selected_worst[top_index]),
        "best_one_coast_schedule": schedules[best_index],
        "best_schedule": schedules[best_index],
        "best_one_coast_selected_worst_error": float(selected_worst[best_index]),
        "best_selected_worst_error": float(selected_worst[best_index]),
        "top3_schedules": "; ".join(f"{schedules[int(index)]}:{probabilities[int(index)]:.3f}" for index in top3),
    }


def _distribution_rows(
    *,
    method: str,
    probabilities: np.ndarray,
    selected_worst: np.ndarray,
    nominal_error: np.ndarray,
    subspace: pd.DataFrame,
    schedules: list[str],
    frontier_schedules: set[str],
) -> list[dict]:
    best_index = int(np.argmin(selected_worst))
    rows = []
    for index, probability in enumerate(np.asarray(probabilities, dtype=float)):
        rows.append(
            {
                "method": method,
                "state_index": int(subspace.loc[index, "state_index"]),
                "state_label": str(subspace.loc[index, "state_label"]),
                "coast_windows": str(subspace.loc[index, "coast_windows_zero_based"]),
                "schedule": schedules[index],
                "probability": float(probability),
                "refined_selected_worst_error": float(selected_worst[index]),
                "refined_nominal_error": float(nominal_error[index]),
                "is_best_state": bool(index == best_index),
                "is_best_one_coast": bool(index == best_index),
                "is_pareto_frontier": schedules[index] in frontier_schedules,
            }
        )
    return rows


def _baseline_row(
    *,
    method: str,
    probabilities: np.ndarray,
    selected_worst: np.ndarray,
    nominal_error: np.ndarray,
    subspace: pd.DataFrame,
    schedules: list[str],
    frontier_schedules: set[str],
    warm_start_mode: str,
    notes: str,
) -> tuple[dict, list[dict]]:
    summary = _summary_base(
        method=method,
        probabilities=probabilities,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
    )
    summary.update(
        {
            "mixer_topology": "none",
            "warm_start_mode": warm_start_mode,
            "depth": 0,
            "restarts": 0,
            "restart_grid": "[]",
            "best_restart_count": 0,
            "best_seed": 0,
            "restart_grid_size": 0,
            "restart_grid_expected_selected_worst_error_min": np.nan,
            "restart_grid_expected_selected_worst_error_median": np.nan,
            "restart_grid_expected_selected_worst_error_max": np.nan,
            "restart_grid_expected_selected_worst_error_spread": np.nan,
            "restart_grid_probability_best_one_coast_min": np.nan,
            "restart_grid_probability_best_one_coast_median": np.nan,
            "restart_grid_probability_best_one_coast_max": np.nan,
            "angles": "[]",
            "initial_state": warm_start_mode,
            "notes": notes,
        }
    )
    return summary, _distribution_rows(
        method=method,
        probabilities=probabilities,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        subspace=subspace,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
    )


def _run_configuration(
    *,
    method: str,
    mixer_topology: str,
    warm_start_mode: str,
    depth: int,
    restarts: int,
    normalized_costs: np.ndarray,
    selected_worst: np.ndarray,
    nominal_error: np.ndarray,
    subspace: pd.DataFrame,
    schedules: list[str],
    frontier_schedules: set[str],
    seed: int,
    warm_start_epsilon: float,
    warm_start_beta: float,
) -> tuple[dict, dict, list[dict]]:
    if warm_start_mode == "uniform":
        initial_probabilities = np.full(len(normalized_costs), 1.0 / len(normalized_costs), dtype=float)
    else:
        initial_probabilities = cost_biased_initial_probabilities(
            normalized_costs,
            mode=warm_start_mode,
            epsilon=float(warm_start_epsilon),
            beta=float(warm_start_beta),
        )
    mixer = adjacency_mixer(len(subspace), mixer_topology)
    result = optimize_subspace_qaoa(
        normalized_costs,
        mixer,
        depth=int(depth),
        restarts=int(restarts),
        maxiter=350,
        seed=int(seed),
        initial_probabilities=initial_probabilities,
    )
    summary = _summary_base(
        method=method,
        probabilities=result.probabilities,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
    )
    summary.update(
        {
            "mixer_topology": mixer_topology,
            "warm_start_mode": warm_start_mode,
            "depth": int(depth),
            "restarts": int(restarts),
            "restart_grid": f"[{int(restarts)}]",
            "best_restart_count": int(restarts),
            "best_seed": int(seed),
            "restart_grid_size": 1,
            "expected_normalized_cost": float(result.expected_cost),
            "restart_grid_expected_selected_worst_error_min": float(summary["expected_selected_worst_error"]),
            "restart_grid_expected_selected_worst_error_median": float(summary["expected_selected_worst_error"]),
            "restart_grid_expected_selected_worst_error_max": float(summary["expected_selected_worst_error"]),
            "restart_grid_expected_selected_worst_error_spread": 0.0,
            "restart_grid_probability_best_one_coast_min": float(summary["probability_best_one_coast"]),
            "restart_grid_probability_best_one_coast_median": float(summary["probability_best_one_coast"]),
            "restart_grid_probability_best_one_coast_max": float(summary["probability_best_one_coast"]),
            "angles": _as_list(result.angles),
            "initial_state": warm_start_mode,
            "notes": "matched uniform feasible-subspace comparator" if warm_start_mode == "uniform" else "feasible-subspace structured QAOA",
        }
    )
    expected_selected_worst_error = float(summary["expected_selected_worst_error"])
    return summary, {
        "method": method,
        "mixer_topology": mixer_topology,
        "warm_start_mode": warm_start_mode,
        "depth": int(depth),
        "restarts": int(restarts),
        "seed": int(seed),
        "expected_selected_worst_error": expected_selected_worst_error,
        "expected_normalized_cost": float(result.expected_cost),
        "expected_nominal_error": float(result.probabilities @ nominal_error),
        "probability_best_one_coast": float(summary["probability_best_one_coast"]),
        "probability_pareto_frontier": float(summary["probability_pareto_frontier"]),
        "top_schedule": summary["top_schedule"],
        "top_schedule_probability": float(summary["top_schedule_probability"]),
        "top_schedule_selected_worst_error": float(summary["top_schedule_selected_worst_error"]),
        "best_one_coast_schedule": summary["best_one_coast_schedule"],
        "best_one_coast_selected_worst_error": float(summary["best_one_coast_selected_worst_error"]),
        "angles": _as_list(result.angles),
    }, _distribution_rows(
        method=method,
        probabilities=result.probabilities,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        subspace=subspace,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
    )


def _comparison_row(*, structured_row: dict, matched_row: dict) -> dict:
    structured_error = float(structured_row["expected_selected_worst_error"])
    matched_error = float(matched_row["expected_selected_worst_error"])
    restart_spread = structured_row.get("restart_grid_expected_selected_worst_error_spread", np.nan)
    probability_min = structured_row.get("restart_grid_probability_best_one_coast_min", np.nan)
    probability_max = structured_row.get("restart_grid_probability_best_one_coast_max", np.nan)
    return {
        "structured_method": str(structured_row["method"]),
        "matched_uniform_method": str(matched_row["method"]),
        "mixer_topology": str(structured_row["mixer_topology"]),
        "warm_start_mode": str(structured_row["warm_start_mode"]),
        "depth": int(structured_row["depth"]),
        "restarts": int(structured_row["best_restart_count"]),
        "restart_grid": structured_row.get("restart_grid", "[]"),
        "structured_expected_selected_worst_error": structured_error,
        "matched_uniform_expected_selected_worst_error": matched_error,
        "structured_minus_matched_uniform_selected_worst_error": structured_error - matched_error,
        "absolute_selected_worst_error_difference": abs(structured_error - matched_error),
        "structured_probability_best_one_coast": float(structured_row["probability_best_one_coast"]),
        "matched_uniform_probability_best_one_coast": float(matched_row["probability_best_one_coast"]),
        "structured_probability_pareto_frontier": float(structured_row["probability_pareto_frontier"]),
        "matched_uniform_probability_pareto_frontier": float(matched_row["probability_pareto_frontier"]),
        "structured_restart_grid_expected_selected_worst_error_min": float(structured_row["restart_grid_expected_selected_worst_error_min"]),
        "structured_restart_grid_expected_selected_worst_error_max": float(structured_row["restart_grid_expected_selected_worst_error_max"]),
        "structured_restart_grid_expected_selected_worst_error_spread": float(restart_spread) if pd.notna(restart_spread) else np.nan,
        "structured_restart_grid_probability_best_one_coast_min": float(probability_min) if pd.notna(probability_min) else np.nan,
        "structured_restart_grid_probability_best_one_coast_max": float(probability_max) if pd.notna(probability_max) else np.nan,
        "comparison_basis": "same feasible subspace, same mixer topology, same depth, same restart count",
    }


def _build_report_rows(
    summary_rows: list[dict],
    restart_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(summary_rows)
    restart_grid = pd.DataFrame(restart_rows)
    report_rows: list[dict] = []
    baseline_methods = [
        "uniform_feasible",
        "warm_start_only_inverse",
        "warm_start_only_exponential",
        MATCHED_COMPARATOR["method"],
    ]
    for method in baseline_methods:
        row = summary[summary["method"] == method]
        if len(row):
            report_rows.append(row.iloc[0].to_dict())

    family_rows = restart_grid[restart_grid["method"].astype(str).str.startswith("structured_")].copy()
    if not family_rows.empty:
        for (topology, warm_start_mode, depth), group in family_rows.groupby(
            ["mixer_topology", "warm_start_mode", "depth"], sort=False
        ):
            group = group.sort_values(
                ["expected_selected_worst_error", "restarts", "method"],
                kind="mergesort",
            )
            best_method = str(group.iloc[0]["method"])
            best_summary = summary[summary["method"] == best_method]
            if best_summary.empty:
                continue
            aggregated = best_summary.iloc[0].to_dict()
            aggregated["restart_grid"] = json.dumps([int(value) for value in sorted(group["restarts"].astype(int).unique().tolist())])
            aggregated["restart_grid_size"] = int(group["restarts"].nunique())
            aggregated["restart_grid_expected_selected_worst_error_min"] = float(group["expected_selected_worst_error"].min())
            aggregated["restart_grid_expected_selected_worst_error_median"] = float(group["expected_selected_worst_error"].median())
            aggregated["restart_grid_expected_selected_worst_error_max"] = float(group["expected_selected_worst_error"].max())
            aggregated["restart_grid_expected_selected_worst_error_spread"] = float(
                group["expected_selected_worst_error"].max() - group["expected_selected_worst_error"].min()
            )
            aggregated["restart_grid_probability_best_one_coast_min"] = float(group["probability_best_one_coast"].min())
            aggregated["restart_grid_probability_best_one_coast_median"] = float(group["probability_best_one_coast"].median())
            aggregated["restart_grid_probability_best_one_coast_max"] = float(group["probability_best_one_coast"].max())
            report_rows.append(aggregated)
    table = pd.DataFrame(report_rows)
    table = table.sort_values(
        ["expected_selected_worst_error", "method"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Ensure baseline rows retain their natural ordering at the top of the manuscript table.
    rank = {method: index for index, method in enumerate(baseline_methods)}
    table["_sort_rank"] = table["method"].map(lambda value: rank.get(str(value), len(rank)))
    table = table.sort_values(["_sort_rank", "expected_selected_worst_error", "method"], kind="mergesort")
    table = table.drop(columns="_sort_rank").reset_index(drop=True)

    return table, restart_grid


def _write_table(table: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "mixer_topology",
        "warm_start_mode",
        "depth",
        "best_restart_count",
        "restart_grid_expected_selected_worst_error_min",
        "restart_grid_expected_selected_worst_error_max",
        "expected_selected_worst_error",
        "probability_best_state",
        "probability_pareto_frontier",
        "top_schedule",
    ]
    available = [column for column in columns if column in table.columns]
    table[available].to_latex(
        tables_dir / "structured_qaoa_sensitivity_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )


def _plot_heatmap(restart_grid: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0), sharex=True, sharey=True, constrained_layout=True)
    pivot_min = float(restart_grid["expected_selected_worst_error"].min())
    pivot_max = float(restart_grid["expected_selected_worst_error"].max())
    topology_order = ["path", "cycle", "complete"]
    mode_order = ["inverse", "exponential"]
    restart_order = sorted({int(value) for value in restart_grid["restarts"].tolist()})
    depth_order = sorted({int(value) for value in restart_grid["depth"].tolist()})
    for row_index, warm_start_mode in enumerate(mode_order):
        for col_index, topology in enumerate(topology_order):
            ax = axes[row_index, col_index]
            subset = restart_grid[
                (restart_grid["warm_start_mode"] == warm_start_mode)
                & (restart_grid["mixer_topology"] == topology)
            ].copy()
            pivot = subset.pivot(index="restarts", columns="depth", values="expected_selected_worst_error")
            pivot = pivot.reindex(index=restart_order, columns=depth_order)
            image = ax.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=pivot_min,
                vmax=pivot_max,
                cmap="viridis",
            )
            for y_index, restarts in enumerate(restart_order):
                for x_index, depth in enumerate(depth_order):
                    value = pivot.loc[restarts, depth]
                    if np.isfinite(value):
                        ax.text(
                            x_index,
                            y_index,
                            f"{float(value):.4f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if float(value) > (pivot_min + pivot_max) / 2.0 else "black",
                        )
            ax.set_title(f"{topology} | {warm_start_mode}")
            ax.set_xticks(np.arange(len(depth_order)))
            ax.set_xticklabels([f"p={depth}" for depth in depth_order])
            ax.set_yticks(np.arange(len(restart_order)))
            ax.set_yticklabels([str(restarts) for restarts in restart_order])
            ax.set_xlabel("Depth")
            ax.set_ylabel("Restarts")
            ax.grid(False)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("Expected selected-worst error")
    fig.savefig(figures_dir / "structured_qaoa_sensitivity_heatmap.png", dpi=220)
    fig.savefig(figures_dir / "structured_qaoa_sensitivity_heatmap.pdf")
    plt.close(fig)


def _load_generic_qaoa_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, dtype={"method": str})
    qaoa = data[data["method"] == "qaoa_statevector"].copy()
    if qaoa.empty:
        return pd.DataFrame()
    keep = [
        "method",
        "runs",
        "refinement_success_rate",
        "refined_selected_worst_error_median",
        "refined_selected_worst_error_iqr",
        "refined_nominal_error_median",
        "solver_true_evaluations_median",
        "shared_qubo_training_evaluations_median",
        "total_true_evaluations_including_training_median",
    ]
    return qaoa[[col for col in keep if col in qaoa.columns]].copy()


def _write_structured_vs_generic(summary: pd.DataFrame, generic: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    structured = summary[summary["method"].astype(str).str.startswith("structured_")].copy()
    if structured.empty or generic.empty:
        comparison = pd.DataFrame()
    else:
        best = structured.sort_values("expected_selected_worst_error", kind="mergesort").iloc[0]
        generic_row = generic.iloc[0]
        generic_error = float(generic_row["refined_selected_worst_error_median"])
        comparison = pd.DataFrame(
            [
                {
                    "structured_method": str(best["method"]),
                    "generic_method": "qaoa_statevector",
                    "comparison_scope": "descriptive_not_like_for_like",
                    "comparison_basis": "structured expected selected-worst error versus generic 30-seed median selected-worst error",
                    "structured_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
                    "generic_median_selected_worst_error": generic_error,
                    "structured_minus_generic_selected_worst_error": float(best["expected_selected_worst_error"]) - generic_error,
                    "structured_probability_best_one_coast": float(best["probability_best_one_coast"]),
                    "structured_probability_pareto_frontier": float(best["probability_pareto_frontier"]),
                    "structured_top_schedule": str(best["top_schedule"]),
                    "structured_top_schedule_probability": float(best["top_schedule_probability"]),
                    "generic_refinement_success_rate": float(generic_row["refinement_success_rate"]),
                    "generic_runs": int(generic_row["runs"]),
                }
            ]
        )
    comparison.to_csv(results_dir / "structured_vs_generic_qaoa.csv", index=False)
    write_json(results_dir / "structured_vs_generic_qaoa.json", comparison.to_dict(orient="records"))
    return comparison


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    cardinality_path = _resolve(args.cardinality)
    generic_path = _resolve(args.generic_qaoa_summary)
    results_dir = _resolve(args.results_dir)
    figures_dir = _resolve(args.figures_dir)
    tables_dir = _resolve(args.tables_dir)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clean_outputs(results_dir, figures_dir, tables_dir)

    subspace = _load_cardinality_group(cardinality_path, args.group)
    selected_worst = subspace["refined_selected_worst_error"].to_numpy(dtype=float)
    nominal_error = subspace["refined_nominal_error"].to_numpy(dtype=float)
    normalized_costs, cost_offset, cost_scale = _normalise_cost(selected_worst)
    schedules = subspace["schedule"].astype(str).to_list()
    frontier_schedules = _pareto_frontier_schedules(subspace)
    best_index = int(np.argmin(selected_worst))

    uniform = np.full(len(subspace), 1.0 / len(subspace), dtype=float)
    inverse_start = cost_biased_initial_probabilities(
        normalized_costs,
        mode="inverse",
        epsilon=float(args.warm_start_epsilon),
        beta=float(args.warm_start_beta),
    )
    exponential_start = cost_biased_initial_probabilities(
        normalized_costs,
        mode="exponential",
        epsilon=float(args.warm_start_epsilon),
        beta=float(args.warm_start_beta),
    )

    summary_rows: list[dict] = []
    restart_rows: list[dict] = []
    distribution_rows: list[dict] = []
    matched_uniform_rows: list[dict] = []
    matched_uniform_restart_rows: list[dict] = []
    matched_uniform_comparison_rows: list[dict] = []
    matched_uniform_cache: dict[tuple[str, int, int], dict[str, dict]] = {}

    def ensure_matched_uniform(topology_index: int, topology: str, depth: int, restarts: int) -> dict:
        key = (str(topology), int(depth), int(restarts))
        cached = matched_uniform_cache.get(key)
        if cached is not None:
            return cached
        method = _matched_uniform_method(topology, depth, restarts)
        summary_row, restart_row, _ = _run_configuration(
            method=method,
            mixer_topology=topology,
            warm_start_mode="uniform",
            depth=int(depth),
            restarts=int(restarts),
            normalized_costs=normalized_costs,
            selected_worst=selected_worst,
            nominal_error=nominal_error,
            subspace=subspace,
            schedules=schedules,
            frontier_schedules=frontier_schedules,
            seed=int(args.seed) + 9000 + 1000 * topology_index + 10 * int(depth) + int(restarts),
            warm_start_epsilon=float(args.warm_start_epsilon),
            warm_start_beta=float(args.warm_start_beta),
        )
        matched_uniform_cache[key] = {"summary": summary_row, "restart": restart_row}
        matched_uniform_rows.append(summary_row)
        matched_uniform_restart_rows.append(restart_row)
        return matched_uniform_cache[key]

    baseline_specs = [
        (
            "uniform_feasible",
            uniform,
            "uniform",
            "feasible-subspace baseline with no mixer dynamics",
        ),
        (
            "warm_start_only_inverse",
            inverse_start,
            "inverse",
            "feasible-subspace baseline with inverse warm-start only",
        ),
        (
            "warm_start_only_exponential",
            exponential_start,
            "exponential",
            "feasible-subspace baseline with exponential warm-start only",
        ),
    ]
    for method, probabilities, warm_start_mode, notes in baseline_specs:
        row, dist_rows = _baseline_row(
            method=method,
            probabilities=probabilities,
            selected_worst=selected_worst,
            nominal_error=nominal_error,
            subspace=subspace,
            schedules=schedules,
            frontier_schedules=frontier_schedules,
            warm_start_mode=warm_start_mode,
            notes=notes,
        )
        summary_rows.append(row)
        distribution_rows.extend(dist_rows)

    matched_summary, matched_restart, matched_distribution = _run_configuration(
        method=MATCHED_COMPARATOR["method"],
        mixer_topology=MATCHED_COMPARATOR["mixer_topology"],
        warm_start_mode=MATCHED_COMPARATOR["warm_start_mode"],
        depth=MATCHED_COMPARATOR["depth"],
        restarts=MATCHED_COMPARATOR["restarts"],
        normalized_costs=normalized_costs,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        subspace=subspace,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
        seed=int(args.seed) + 9000,
        warm_start_epsilon=float(args.warm_start_epsilon),
        warm_start_beta=float(args.warm_start_beta),
    )
    summary_rows.append(matched_summary)
    restart_rows.append(matched_restart)
    distribution_rows.extend(matched_distribution)
    matched_uniform_cache[(MATCHED_COMPARATOR["mixer_topology"], int(MATCHED_COMPARATOR["depth"]), int(MATCHED_COMPARATOR["restarts"]))] = {
        "summary": matched_summary,
        "restart": matched_restart,
    }
    matched_uniform_rows.append(matched_summary)
    matched_uniform_restart_rows.append(matched_restart)

    for topology_index, topology in enumerate(args.topologies):
        for warm_start_index, warm_start_mode in enumerate(args.warm_start_modes):
            for depth in args.depths:
                for restarts in args.restart_grid:
                    method = f"structured_{warm_start_mode}_{topology}_p{int(depth)}_r{int(restarts)}"
                    seed = int(args.seed) + 1000 * topology_index + 100 * warm_start_index + 10 * int(depth) + int(restarts)
                    row, restart_row, dist_rows = _run_configuration(
                        method=method,
                        mixer_topology=topology,
                        warm_start_mode=warm_start_mode,
                        depth=int(depth),
                        restarts=int(restarts),
                        normalized_costs=normalized_costs,
                        selected_worst=selected_worst,
                        nominal_error=nominal_error,
                        subspace=subspace,
                        schedules=schedules,
                        frontier_schedules=frontier_schedules,
                        seed=seed,
                        warm_start_epsilon=float(args.warm_start_epsilon),
                        warm_start_beta=float(args.warm_start_beta),
                    )
                    summary_rows.append(row)
                    restart_rows.append(restart_row)
                    distribution_rows.extend(dist_rows)
                    matched_uniform = ensure_matched_uniform(topology_index, topology, int(depth), int(restarts))
                    matched_uniform_comparison_rows.append(
                        _comparison_row(structured_row=row, matched_row=matched_uniform["summary"])
                    )

    summary, restart_grid = _build_report_rows(
        summary_rows,
        restart_rows,
    )
    generic = _load_generic_qaoa_summary(generic_path)
    comparison = _write_structured_vs_generic(summary, generic, results_dir)

    matched_uniform_restart_grid = pd.DataFrame(matched_uniform_restart_rows)
    matched_uniform_restart_grid = matched_uniform_restart_grid.sort_values(
        ["mixer_topology", "depth", "restarts", "method"],
        kind="mergesort",
    ).reset_index(drop=True)
    matched_uniform_restart_grid.to_csv(results_dir / "matched_uniform_restart_grid.csv", index=False)
    matched_uniform_restart_grid.to_json(results_dir / "matched_uniform_restart_grid.json", orient="records", indent=2)

    structured_vs_matched_uniform = pd.DataFrame(matched_uniform_comparison_rows)
    structured_vs_matched_uniform = structured_vs_matched_uniform.sort_values(
        ["mixer_topology", "warm_start_mode", "depth", "restarts", "structured_method"],
        kind="mergesort",
    ).reset_index(drop=True)
    structured_vs_matched_uniform.to_csv(results_dir / "structured_vs_matched_uniform.csv", index=False)
    structured_vs_matched_uniform.to_json(results_dir / "structured_vs_matched_uniform.json", orient="records", indent=2)

    structured_family = summary[summary["method"].astype(str).str.startswith("structured_")].copy()
    family_comparisons: list[dict] = []
    if not structured_family.empty:
        for _, structured_row in structured_family.iterrows():
            key = (str(structured_row["mixer_topology"]), int(structured_row["depth"]), int(structured_row["best_restart_count"]))
            matched_row = matched_uniform_cache.get(key)
            if matched_row is None:
                continue
            family_comparisons.append(
                _comparison_row(structured_row=structured_row.to_dict(), matched_row=matched_row["summary"])
            )
    structured_vs_matched_family = pd.DataFrame(family_comparisons)
    if not structured_vs_matched_family.empty:
        structured_vs_matched_family = structured_vs_matched_family.sort_values(
            ["structured_expected_selected_worst_error", "structured_method"],
            kind="mergesort",
        ).reset_index(drop=True)
    structured_vs_matched_family.to_csv(results_dir / "structured_vs_matched_uniform_family.csv", index=False)
    structured_vs_matched_family.to_json(results_dir / "structured_vs_matched_uniform_family.json", orient="records", indent=2)

    if structured_vs_matched_family.empty:
        matched_uniform_effect_summary = {
            "family_row_count": 0,
            "structured_better_count": 0,
            "structured_minus_matched_uniform_selected_worst_error_min": np.nan,
            "structured_minus_matched_uniform_selected_worst_error_median": np.nan,
            "structured_minus_matched_uniform_selected_worst_error_max": np.nan,
            "absolute_selected_worst_error_difference_min": np.nan,
            "absolute_selected_worst_error_difference_median": np.nan,
            "absolute_selected_worst_error_difference_max": np.nan,
            "structured_restart_grid_expected_selected_worst_error_spread_min": np.nan,
            "structured_restart_grid_expected_selected_worst_error_spread_median": np.nan,
            "structured_restart_grid_expected_selected_worst_error_spread_max": np.nan,
        }
    else:
        deltas = structured_vs_matched_family["structured_minus_matched_uniform_selected_worst_error"].astype(float)
        abs_deltas = structured_vs_matched_family["absolute_selected_worst_error_difference"].astype(float)
        spreads = structured_vs_matched_family["structured_restart_grid_expected_selected_worst_error_spread"].astype(float)
        matched_uniform_effect_summary = {
            "family_row_count": int(len(structured_vs_matched_family)),
            "structured_better_count": int((deltas < 0).sum()),
            "structured_minus_matched_uniform_selected_worst_error_min": float(deltas.min()),
            "structured_minus_matched_uniform_selected_worst_error_median": float(deltas.median()),
            "structured_minus_matched_uniform_selected_worst_error_max": float(deltas.max()),
            "absolute_selected_worst_error_difference_min": float(abs_deltas.min()),
            "absolute_selected_worst_error_difference_median": float(abs_deltas.median()),
            "absolute_selected_worst_error_difference_max": float(abs_deltas.max()),
            "structured_restart_grid_expected_selected_worst_error_spread_min": float(spreads.min()),
            "structured_restart_grid_expected_selected_worst_error_spread_median": float(spreads.median()),
            "structured_restart_grid_expected_selected_worst_error_spread_max": float(spreads.max()),
        }

    summary.to_csv(results_dir / "summary.csv", index=False)
    summary.to_json(results_dir / "summary.json", orient="records", indent=2)
    restart_grid.to_csv(results_dir / "restart_grid.csv", index=False)
    restart_grid.to_json(results_dir / "restart_grid.json", orient="records", indent=2)
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(results_dir / "distribution.csv", index=False)
    distribution.to_json(results_dir / "distribution.json", orient="records", indent=2)
    generic.to_csv(results_dir / "generic_qaoa_summary.csv", index=False)
    _write_table(summary, tables_dir)
    _plot_heatmap(restart_grid, figures_dir)

    best_row = summary[summary["method"].astype(str).str.startswith("structured_")].sort_values(
        "expected_selected_worst_error",
        kind="mergesort",
    ).iloc[0]
    strongest_restart_grid = restart_grid[
        (restart_grid["mixer_topology"] == best_row["mixer_topology"])
        & (restart_grid["warm_start_mode"] == best_row["warm_start_mode"])
        & (restart_grid["depth"] == best_row["depth"])
    ].sort_values("expected_selected_worst_error", kind="mergesort")

    metadata = {
        "command": _command_string(),
        "python": _python_launcher(),
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas"]),
        **revision_metadata(),
        "cardinality_source": str(cardinality_path.relative_to(ROOT) if cardinality_path.is_relative_to(ROOT) else cardinality_path),
        "cardinality_group": str(args.group),
        "states_in_subspace": int(len(subspace)),
        "generic_qaoa_summary_source": str(generic_path.relative_to(ROOT) if generic_path.is_relative_to(ROOT) else generic_path),
        "feasible_subspace": f"the {len(subspace)} schedules in cardinality group {args.group} from the phase-shift cardinality benchmark",
        "constraint_preservation": "all probability mass remains on the selected feasible cardinality subspace; no penalty terms or infeasible bitstrings are used",
        "cost_column": "refined_selected_worst_error",
        "cost_normalization": {
            "offset": cost_offset,
            "scale": cost_scale,
            "formula": "(selected_worst_error - offset) / scale",
        },
        "restart_grid": [int(value) for value in args.restart_grid],
        "depths": [int(value) for value in args.depths],
        "topologies": [str(value) for value in args.topologies],
        "warm_start_modes": [str(value) for value in args.warm_start_modes],
        "baseline_rows": [spec[0] for spec in baseline_specs] + [MATCHED_COMPARATOR["method"]],
        "strongest_configuration": {
            "method": str(best_row["method"]),
            "mixer_topology": str(best_row["mixer_topology"]),
            "warm_start_mode": str(best_row["warm_start_mode"]),
            "depth": int(best_row["depth"]),
            "best_restart_count": int(best_row["best_restart_count"]),
            "expected_selected_worst_error": float(best_row["expected_selected_worst_error"]),
            "probability_best_state": float(best_row["probability_best_state"]),
            "probability_best_one_coast": float(best_row["probability_best_one_coast"]),
            "probability_pareto_frontier": float(best_row["probability_pareto_frontier"]),
            "top_schedule": str(best_row["top_schedule"]),
            "best_schedule": str(best_row["best_schedule"]),
        },
        "restart_grid_spread_for_strongest_configuration": {
            "method": str(best_row["method"]),
            "restart_expected_selected_worst_error_min": float(strongest_restart_grid["expected_selected_worst_error"].min()),
            "restart_expected_selected_worst_error_max": float(strongest_restart_grid["expected_selected_worst_error"].max()),
            "restart_probability_best_one_coast_min": float(strongest_restart_grid["probability_best_one_coast"].min()),
            "restart_probability_best_one_coast_max": float(strongest_restart_grid["probability_best_one_coast"].max()),
        },
        "matched_uniform_effect_summary": matched_uniform_effect_summary,
        "structured_vs_generic_qaoa": comparison.to_dict(orient="records"),
        "runtime_seconds": float(time.perf_counter() - start),
        "interpretation_limits": [
            "This is a statevector simulation over the selected feasible cardinality subspace, not hardware evidence.",
            "The cost oracle uses already persisted branch-refined selected-worst errors from the cardinality ablation.",
            "The all-windows continuous row is outside the selected feasible subspace and remains the lower-error dense-control baseline where applicable.",
            "The restart grid shows sensitivity to optimizer initialization count, not a guarantee of global optimality.",
        ],
        "artifact_paths": {
            "summary_csv": str((results_dir / "summary.csv").relative_to(ROOT)),
            "summary_json": str((results_dir / "summary.json").relative_to(ROOT)),
            "restart_grid_csv": str((results_dir / "restart_grid.csv").relative_to(ROOT)),
            "restart_grid_json": str((results_dir / "restart_grid.json").relative_to(ROOT)),
            "distribution_csv": str((results_dir / "distribution.csv").relative_to(ROOT)),
            "distribution_json": str((results_dir / "distribution.json").relative_to(ROOT)),
            "matched_uniform_restart_grid_csv": str((results_dir / "matched_uniform_restart_grid.csv").relative_to(ROOT)),
            "matched_uniform_restart_grid_json": str((results_dir / "matched_uniform_restart_grid.json").relative_to(ROOT)),
            "structured_vs_matched_uniform_csv": str((results_dir / "structured_vs_matched_uniform.csv").relative_to(ROOT)),
            "structured_vs_matched_uniform_json": str((results_dir / "structured_vs_matched_uniform.json").relative_to(ROOT)),
            "structured_vs_matched_uniform_family_csv": str((results_dir / "structured_vs_matched_uniform_family.csv").relative_to(ROOT)),
            "structured_vs_matched_uniform_family_json": str((results_dir / "structured_vs_matched_uniform_family.json").relative_to(ROOT)),
            "figure_pdf": str((figures_dir / "structured_qaoa_sensitivity_heatmap.pdf").relative_to(ROOT)),
            "figure_png": str((figures_dir / "structured_qaoa_sensitivity_heatmap.png").relative_to(ROOT)),
            "table_tex": str((tables_dir / "structured_qaoa_sensitivity_table.tex").relative_to(ROOT)),
        },
    }
    write_json(results_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured-QAOA sensitivity analysis on a cardinality-refinement group.")
    parser.add_argument("--cardinality", type=Path, default=DEFAULT_CARDINALITY)
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP)
    parser.add_argument("--generic-qaoa-summary", type=Path, default=DEFAULT_GENERIC_QAOA_SUMMARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--restart-grid", type=int, nargs="+", default=DEFAULT_RESTART_GRID)
    parser.add_argument("--topologies", type=str, nargs="+", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--warm-start-modes", type=str, nargs="+", default=DEFAULT_WARM_START_MODES)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--warm-start-epsilon", type=float, default=0.02)
    parser.add_argument("--warm-start-beta", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(
        f"completed structured-QAOA sensitivity run in {metadata['runtime_seconds']:.1f} seconds",
        flush=True,
    )


if __name__ == "__main__":
    main()
