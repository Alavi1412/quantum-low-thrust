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
DEFAULT_RESULTS_DIR = Path("data/results/structured_qaoa_one_coast")
DEFAULT_FIGURES_DIR = Path("figures/structured_qaoa_one_coast")
DEFAULT_TABLES_DIR = Path("tables/structured_qaoa_one_coast")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _command_string() -> str:
    argv = getattr(sys, "orig_argv", sys.argv)
    return subprocess.list2cmdline([str(arg) for arg in argv])


def _python_launcher() -> str:
    return f"py -{sys.version_info.major}.{sys.version_info.minor}"


def _load_one_coast(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"schedule": str})
    one_coast = data[data["group"] == "one_coast_k11_full"].copy()
    if len(one_coast) != 12:
        raise RuntimeError(f"expected 12 one-coast schedules, found {len(one_coast)} in {path}")
    one_coast["coast_windows_zero_based"] = pd.to_numeric(one_coast["coast_windows_zero_based"], errors="raise").astype(int)
    one_coast["schedule"] = one_coast["schedule"].astype(str).str.zfill(12)
    one_coast = one_coast.sort_values("coast_windows_zero_based").reset_index(drop=True)
    return one_coast


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
            "distribution.csv",
            "generic_qaoa_summary.csv",
            "structured_vs_generic_qaoa.csv",
            "structured_vs_generic_qaoa.json",
            "metadata.json",
            "generic_method_frontier_comparison.csv",
        ],
        figures_dir: [
            "structured_qaoa_distribution.png",
            "structured_qaoa_distribution.pdf",
        ],
        tables_dir: [
            "structured_qaoa_one_coast_table.tex",
            "structured_vs_generic_qaoa_table.tex",
        ],
    }
    for directory, names in patterns.items():
        for name in names:
            path = directory / name
            if path.is_file():
                path.unlink()


def _summarize_distribution(
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
        "probability_pareto_frontier": frontier_probability,
        "top_schedule": schedules[top_index],
        "top_schedule_probability": float(probabilities[top_index]),
        "top_schedule_selected_worst_error": float(selected_worst[top_index]),
        "best_one_coast_schedule": schedules[best_index],
        "best_one_coast_selected_worst_error": float(selected_worst[best_index]),
        "top3_schedules": "; ".join(f"{schedules[int(index)]}:{probabilities[int(index)]:.3f}" for index in top3),
    }


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
    structured = summary[summary["method"].astype(str).str.startswith("warm_start_xy_qaoa")].copy()
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


def _write_table(summary: pd.DataFrame, comparison: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "expected_selected_worst_error",
        "probability_best_one_coast",
        "probability_pareto_frontier",
        "top_schedule",
        "top_schedule_selected_worst_error",
    ]
    summary[columns].to_latex(
        tables_dir / "structured_qaoa_one_coast_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )
    if not comparison.empty:
        comparison.to_latex(
            tables_dir / "structured_vs_generic_qaoa_table.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )


def _plot_distribution(distribution: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pivot = distribution.pivot(index="coast_window", columns="method", values="probability")
    cost = distribution.drop_duplicates("coast_window").sort_values("coast_window")
    x = np.arange(len(pivot))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    width = 0.22
    methods = list(pivot.columns)
    offsets = np.linspace(-width, width, len(methods))
    for offset, method in zip(offsets, methods):
        ax1.bar(x + offset, pivot[method].to_numpy(dtype=float), width=width, label=method)
    ax1.set_xlabel("Coast window index")
    ax1.set_ylabel("Sampling probability")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(value)) for value in pivot.index])
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, cost["refined_selected_worst_error"], color="black", marker="o", linewidth=1.2, label="selected worst error")
    ax2.set_ylabel("Selected worst error")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "structured_qaoa_distribution.png", dpi=220)
    fig.savefig(figures_dir / "structured_qaoa_distribution.pdf")
    plt.close(fig)


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

    one_coast = _load_one_coast(cardinality_path)
    selected_worst = one_coast["refined_selected_worst_error"].to_numpy(dtype=float)
    nominal_error = one_coast["refined_nominal_error"].to_numpy(dtype=float)
    normalized_costs, cost_offset, cost_scale = _normalise_cost(selected_worst)
    schedules = one_coast["schedule"].astype(str).to_list()
    frontier_schedules = set(one_coast.loc[one_coast["one_coast_pareto_frontier"].astype(bool), "schedule"].astype(str))
    best_index = int(np.argmin(selected_worst))
    warm_start_name = "inverse_cost" if args.warm_start_mode == "inverse" else "exponential_cost"

    mixer = adjacency_mixer(len(one_coast), args.mixer)
    distribution_rows = []
    summary_rows = []
    uniform = np.full(len(one_coast), 1.0 / len(one_coast), dtype=float)
    warm_start = cost_biased_initial_probabilities(
        normalized_costs,
        mode=args.warm_start_mode,
        epsilon=float(args.warm_start_epsilon),
        beta=float(args.warm_start_beta),
    )
    summary_rows.append(
        _summarize_distribution(
            method="uniform_feasible",
            probabilities=uniform,
            selected_worst=selected_worst,
            nominal_error=nominal_error,
            schedules=schedules,
            frontier_schedules=frontier_schedules,
        )
    )
    warm_summary = _summarize_distribution(
        method="warm_start_xy_qaoa",
        probabilities=warm_start,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
    )
    warm_summary.update(
        {
            "mixer_topology": args.mixer,
            "depth": 0,
            "angles": "[]",
            "expected_normalized_cost": float(warm_start @ normalized_costs),
            "cost_offset": cost_offset,
            "cost_scale": cost_scale,
            "initial_state": warm_start_name,
            "warm_start_epsilon": float(args.warm_start_epsilon) if args.warm_start_mode == "inverse" else np.nan,
            "warm_start_beta": float(args.warm_start_beta) if args.warm_start_mode == "exponential" else np.nan,
        }
    )
    summary_rows.append(warm_summary)

    for initial_name, initial_probabilities, label in [
        ("uniform", uniform, "structured_xy_qaoa"),
        ("inverse_cost", warm_start, "warm_start_xy_qaoa"),
    ]:
        for depth in args.depths:
            result = optimize_subspace_qaoa(
                normalized_costs,
                mixer,
                depth=int(depth),
                restarts=int(args.restarts),
                maxiter=int(args.maxiter),
                seed=int(args.seed) + int(depth) + (1000 if initial_name == "inverse_cost" else 0),
                initial_probabilities=initial_probabilities,
            )
            method = f"{label}_p{int(depth)}"
            summary = _summarize_distribution(
                method=method,
                probabilities=result.probabilities,
                selected_worst=selected_worst,
                nominal_error=nominal_error,
                schedules=schedules,
                frontier_schedules=frontier_schedules,
            )
            summary.update(
                {
                    "mixer_topology": args.mixer,
                    "depth": int(depth),
                    "angles": json.dumps([float(value) for value in result.angles]),
                    "expected_normalized_cost": float(result.expected_cost),
                    "cost_offset": cost_offset,
                    "cost_scale": cost_scale,
                    "initial_state": warm_start_name if initial_name != "uniform" else "uniform",
                    "warm_start_epsilon": float(args.warm_start_epsilon)
                    if initial_name == "inverse_cost" and args.warm_start_mode == "inverse"
                    else np.nan,
                    "warm_start_beta": float(args.warm_start_beta)
                    if initial_name == "inverse_cost" and args.warm_start_mode == "exponential"
                    else np.nan,
                }
            )
            summary_rows.append(summary)
            for index, probability in enumerate(result.probabilities):
                distribution_rows.append(
                    {
                        "method": method,
                        "coast_window": int(one_coast.loc[index, "coast_windows_zero_based"]),
                        "schedule": schedules[index],
                        "probability": float(probability),
                        "refined_selected_worst_error": float(selected_worst[index]),
                        "refined_nominal_error": float(nominal_error[index]),
                        "is_best_one_coast": bool(index == best_index),
                        "is_pareto_frontier": schedules[index] in frontier_schedules,
                    }
                )
    for index, probability in enumerate(uniform):
        distribution_rows.append(
            {
                "method": "uniform_feasible",
                "coast_window": int(one_coast.loc[index, "coast_windows_zero_based"]),
                "schedule": schedules[index],
                "probability": float(probability),
                "refined_selected_worst_error": float(selected_worst[index]),
                "refined_nominal_error": float(nominal_error[index]),
                "is_best_one_coast": bool(index == best_index),
                "is_pareto_frontier": schedules[index] in frontier_schedules,
            }
        )
    for index, probability in enumerate(warm_start):
        distribution_rows.append(
            {
                "method": "warm_start_xy_qaoa",
                "coast_window": int(one_coast.loc[index, "coast_windows_zero_based"]),
                "schedule": schedules[index],
                "probability": float(probability),
                "refined_selected_worst_error": float(selected_worst[index]),
                "refined_nominal_error": float(nominal_error[index]),
                "is_best_one_coast": bool(index == best_index),
                "is_pareto_frontier": schedules[index] in frontier_schedules,
            }
        )

    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    generic = _load_generic_qaoa_summary(generic_path)
    comparison = _write_structured_vs_generic(summary, generic, results_dir)
    summary.to_csv(results_dir / "summary.csv", index=False)
    distribution.to_csv(results_dir / "distribution.csv", index=False)
    generic.to_csv(results_dir / "generic_qaoa_summary.csv", index=False)
    write_json(results_dir / "summary.json", summary.to_dict(orient="records"))
    _write_table(summary, comparison, tables_dir)
    if not args.no_figure:
        _plot_distribution(distribution, figures_dir)

    metadata = {
        "command": _command_string(),
        "python": _python_launcher(),
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas"]),
        **revision_metadata(),
        "cardinality_source": str(cardinality_path.relative_to(ROOT) if cardinality_path.is_relative_to(ROOT) else cardinality_path),
        "generic_qaoa_summary_source": str(generic_path.relative_to(ROOT) if generic_path.is_relative_to(ROOT) else generic_path),
        "mixer_topology": args.mixer,
        "depths": [int(depth) for depth in args.depths],
        "cost_column": "refined_selected_worst_error",
        "cost_normalization": {
            "offset": cost_offset,
            "scale": cost_scale,
            "formula": "(selected_worst_error - offset) / scale",
        },
        "feasible_subspace": "the 12 Hamming-weight-11 one-coast schedules from the phase-shift cardinality benchmark",
        "constraint_preservation": "all probability mass remains on the one-coast feasible subspace; no penalty terms or infeasible bitstrings are used",
        "warm_start": {
            "initial_state": warm_start_name,
            "epsilon": float(args.warm_start_epsilon),
            "beta": float(args.warm_start_beta),
            "formula": (
                "initial probability proportional to 1 / (normalized selected-worst error + epsilon)"
                if args.warm_start_mode == "inverse"
                else "initial probability proportional to exp(-beta * shifted normalized selected-worst error)"
            ),
        },
        "runtime_seconds": float(time.perf_counter() - start),
        "interpretation_limits": [
            "This is a statevector simulation over the one-coast feasible subspace, not hardware evidence.",
            "The cost oracle uses already persisted branch-refined selected-worst errors from the cardinality ablation.",
            "The all-windows continuous row is outside the one-coast feasible subspace and remains the lower-error dense-control baseline.",
        ],
    }
    write_json(results_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured constraint-preserving QAOA on the one-coast cardinality benchmark.")
    parser.add_argument("--cardinality", type=Path, default=DEFAULT_CARDINALITY)
    parser.add_argument("--generic-qaoa-summary", type=Path, default=DEFAULT_GENERIC_QAOA_SUMMARY)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--mixer", choices=["path", "cycle", "complete"], default="complete")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--warm-start-epsilon", type=float, default=0.02)
    parser.add_argument("--warm-start-mode", choices=["inverse", "exponential"], default="inverse")
    parser.add_argument("--warm-start-beta", type=float, default=3.0)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(f"completed structured one-coast QAOA in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()


