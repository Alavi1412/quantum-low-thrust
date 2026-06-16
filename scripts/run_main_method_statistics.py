"""30-seed statistical postprocessor for the main non-teacher N=12 phase-shift
duty-cycle-prior benchmark.

Reads experiment outputs already produced by::

    py -3.11 scripts/run_experiment.py --config configs/q1_phase_shift_cardinality_30seed.yaml

Statistical outputs written to data/results/<output_subdir> and
tables/<output_subdir>:

  - success_intervals.csv / .json    : Wilson 95% success-rate CI per method.
  - paired_success_deltas.csv        : per-seed method minus baseline success flags.
  - paired_comparisons.csv / .json   : bootstrap CI and sign-test vs each baseline.
  - main_method_statistics_table.tex : LaTeX table for paper inclusion.
  - main_method_statistics_metadata.json : conservative run record (no advantage claim).

A summary bar chart is written under figures/<output_subdir> when --no-figure is
not passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import output_directories, run as run_experiment
from qlt.reporting import package_versions, revision_metadata, write_json

# ---
# Constants
# ---

DEFAULT_CONFIG = Path("configs/q1_phase_shift_cardinality_30seed.yaml")
EXPECTED_MAIN_METHODS = [
    "random",
    "cross_entropy",
    "genetic",
    "true_sa",
    "surrogate_qubo_sa",
    "qaoa_statevector",
    "all_windows_continuous",
]
WILSON_Z_95 = 1.959963984540054
STATISTICS_BOOTSTRAP_SAMPLES = 20_000
STATISTICS_RNG_SEED = 20260615
LOWER_ERROR_IS_BETTER_METRIC = "refined_selected_worst_error"

# Baselines used for paired comparisons.  Comparisons are generated for each
# baseline that is present in the raw results.
COMPARISON_BASELINES = ["all_windows_continuous", "surrogate_qubo_sa"]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative_or_absolute(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


# ---
# Statistical helpers (self-contained; importable by tests)
# ---


def _success_as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    return values.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})


def wilson_interval(successes: int, runs: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if runs <= 0:
        return (float("nan"), float("nan"))
    phat = successes / runs
    denominator = 1.0 + z ** 2 / runs
    center = (phat + z ** 2 / (2.0 * runs)) / denominator
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) / runs) + (z ** 2 / (4.0 * runs ** 2)))
        / denominator
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def exact_sign_test_pvalue(
    differences: np.ndarray, tolerance: float = 1e-12
) -> tuple[int, int, int, float]:
    """Two-sided exact sign test; ties (|diff| <= tolerance) are excluded defensibly."""
    finite = np.asarray(differences, dtype=float)
    finite = finite[np.isfinite(finite)]
    wins = int(np.sum(finite < -tolerance))
    losses = int(np.sum(finite > tolerance))
    ties = int(finite.size - wins - losses)
    trials = wins + losses
    if trials == 0:
        return wins, losses, ties, 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(trials, k) for k in range(smaller + 1)) / (2.0 ** trials)
    return wins, losses, ties, float(min(1.0, 2.0 * tail))


def format_pvalue(value: float) -> str:
    """Format exact p-values without rounding very small values to literal zero."""
    try:
        pvalue = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(pvalue):
        return ""
    if pvalue == 0.0:
        return "<1e-300"
    if abs(pvalue) < 1e-4:
        return f"{pvalue:.2e}"
    return f"{pvalue:.4f}"


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str,
    samples: int = STATISTICS_BOOTSTRAP_SAMPLES,
    seed: int = STATISTICS_RNG_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the requested statistic of *values*."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    if finite.size == 1:
        value = float(finite[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(int(samples), finite.size))
    resampled = finite[indices]
    if statistic == "mean":
        estimates = np.mean(resampled, axis=1)
    elif statistic == "median":
        estimates = np.median(resampled, axis=1)
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return (float(lower), float(upper))


# ---
# Statistical artifact generation
# ---


def success_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    """Wilson 95% CI on per-method binomial refinement success rate."""
    rows = []
    for method, group in raw.groupby("method", sort=False):
        successes = int(_success_as_bool(group["refinement_success"]).sum())
        runs = int(group["seed"].nunique())
        lower, upper = wilson_interval(successes, runs)
        rows.append(
            {
                "method": method,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else float("nan"),
                "success_rate_wilson95_lower": lower,
                "success_rate_wilson95_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def paired_statistics(
    raw: pd.DataFrame, baselines: list[str] = COMPARISON_BASELINES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired deltas and sign-test / bootstrap comparisons versus each baseline."""
    rows = raw.copy()
    rows["refinement_success_bool"] = _success_as_bool(rows["refinement_success"])
    available_methods = list(dict.fromkeys(rows["method"].astype(str).to_list()))

    delta_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for baseline in baselines:
        if baseline not in available_methods:
            continue
        baseline_rows = (
            rows[rows["method"] == baseline][
                ["seed", "refinement_success_bool", LOWER_ERROR_IS_BETTER_METRIC]
            ]
            .rename(
                columns={
                    "refinement_success_bool": "baseline_success",
                    LOWER_ERROR_IS_BETTER_METRIC: "baseline_selected_worst_error",
                }
            )
        )
        compared_methods = [m for m in available_methods if m != baseline]
        for method in compared_methods:
            method_rows = (
                rows[rows["method"] == method][
                    ["seed", "refinement_success_bool", LOWER_ERROR_IS_BETTER_METRIC]
                ]
                .rename(
                    columns={
                        "refinement_success_bool": "method_success",
                        LOWER_ERROR_IS_BETTER_METRIC: "method_selected_worst_error",
                    }
                )
            )
            paired = method_rows.merge(baseline_rows, on="seed", how="inner").sort_values(
                "seed"
            )
            if paired.empty:
                continue
            paired["success_delta"] = (
                paired["method_success"].astype(int) - paired["baseline_success"].astype(int)
            )
            paired["selected_worst_error_diff"] = (
                paired["method_selected_worst_error"].astype(float)
                - paired["baseline_selected_worst_error"].astype(float)
            )
            for r in paired.itertuples(index=False):
                delta_rows.append(
                    {
                        "baseline_method": baseline,
                        "method": method,
                        "seed": int(r.seed),
                        "baseline_success": bool(r.baseline_success),
                        "method_success": bool(r.method_success),
                        "success_delta": int(r.success_delta),
                        "baseline_selected_worst_error": float(r.baseline_selected_worst_error),
                        "method_selected_worst_error": float(r.method_selected_worst_error),
                        "selected_worst_error_diff": float(r.selected_worst_error_diff),
                    }
                )
            diffs = paired["selected_worst_error_diff"].to_numpy(dtype=float)
            wins, losses, ties, pvalue = exact_sign_test_pvalue(diffs)
            mean_lower, mean_upper = bootstrap_ci(diffs, statistic="mean")
            median_lower, median_upper = bootstrap_ci(diffs, statistic="median")
            comparison_rows.append(
                {
                    "baseline_method": baseline,
                    "method": method,
                    "paired_seeds": int(paired["seed"].nunique()),
                    "baseline_successes": int(paired["baseline_success"].sum()),
                    "method_successes": int(paired["method_success"].sum()),
                    "paired_success_delta_mean": float(paired["success_delta"].mean()),
                    "paired_success_delta_sum": int(paired["success_delta"].sum()),
                    "selected_worst_error_diff_mean": float(np.mean(diffs)),
                    "selected_worst_error_diff_mean_bootstrap95_lower": mean_lower,
                    "selected_worst_error_diff_mean_bootstrap95_upper": mean_upper,
                    "selected_worst_error_diff_median": float(np.median(diffs)),
                    "selected_worst_error_diff_median_bootstrap95_lower": median_lower,
                    "selected_worst_error_diff_median_bootstrap95_upper": median_upper,
                    "selected_worst_error_diff_negative_favors_method": True,
                    "selected_worst_error_method_wins": wins,
                    "selected_worst_error_method_losses": losses,
                    "selected_worst_error_ties": ties,
                    "selected_worst_error_sign_test_p_two_sided": pvalue,
                }
            )

    return pd.DataFrame(delta_rows), pd.DataFrame(comparison_rows)


def write_statistics_latex(
    success: pd.DataFrame,
    comparisons: pd.DataFrame,
    tables_dir: Path,
    baselines: list[str] = COMPARISON_BASELINES,
) -> None:
    """Write a LaTeX table with success CIs and paired comparison columns per baseline."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in success.itertuples(index=False):
        method = str(s.method)
        row: dict = {
            "method": method,
            "success": f"{int(s.successes)}/{int(s.runs)}",
            "wilson_95_ci": (
                f"[{s.success_rate_wilson95_lower:.2f}, "
                f"{s.success_rate_wilson95_upper:.2f}]"
            ),
        }
        for baseline in baselines:
            prefix = f"vs_{baseline}"
            mask = (
                (comparisons["baseline_method"] == baseline)
                & (comparisons["method"] == method)
            )
            cmp = comparisons[mask]
            if len(cmp):
                c = cmp.iloc[0]
                row[f"delta_success_{prefix}"] = f"{c['paired_success_delta_mean']:+.2f}"
                row[f"mean_error_delta_{prefix}"] = (
                    f"{c['selected_worst_error_diff_mean']:+.4f} "
                    f"[{c['selected_worst_error_diff_mean_bootstrap95_lower']:+.4f}, "
                    f"{c['selected_worst_error_diff_mean_bootstrap95_upper']:+.4f}]"
                )
                row[f"sign_p_{prefix}"] = format_pvalue(
                    c["selected_worst_error_sign_test_p_two_sided"]
                )
            else:
                row[f"delta_success_{prefix}"] = ""
                row[f"mean_error_delta_{prefix}"] = ""
                row[f"sign_p_{prefix}"] = ""
        rows.append(row)
    pd.DataFrame(rows).to_latex(
        tables_dir / "main_method_statistics_table.tex",
        index=False,
        escape=True,
    )


def write_statistical_artifacts(
    raw: pd.DataFrame,
    results_dir: Path,
    tables_dir: Path,
    baselines: list[str] = COMPARISON_BASELINES,
) -> dict:
    """Compute and write all statistical artifacts; return dict of DataFrames."""
    success = success_intervals(raw)
    deltas, comparisons = paired_statistics(raw, baselines)
    results_dir.mkdir(parents=True, exist_ok=True)
    success.to_csv(results_dir / "success_intervals.csv", index=False)
    deltas.to_csv(results_dir / "paired_success_deltas.csv", index=False)
    comparisons.to_csv(results_dir / "paired_comparisons.csv", index=False)
    write_json(results_dir / "success_intervals.json", success.to_dict(orient="records"))
    write_json(results_dir / "paired_comparisons.json", comparisons.to_dict(orient="records"))
    write_statistics_latex(success, comparisons, tables_dir, baselines)
    return {
        "success_intervals": success,
        "paired_success_deltas": deltas,
        "paired_comparisons": comparisons,
    }


def validate_main_method_results(raw: pd.DataFrame, config: dict) -> dict:
    """Validate that raw results cover the configured seeds and main methods."""
    required_columns = {"seed", "method", "refinement_success", LOWER_ERROR_IS_BETTER_METRIC}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise RuntimeError(f"raw_results.csv is missing columns: {sorted(missing_columns)}")

    expected_seeds = [int(seed) for seed in config["experiments"]["seeds"]]
    seeds_in_data = sorted(int(seed) for seed in raw["seed"].unique())
    methods_in_data = list(dict.fromkeys(raw["method"].astype(str).to_list()))

    missing_seeds = sorted(set(expected_seeds).difference(seeds_in_data))
    extra_seeds = sorted(set(seeds_in_data).difference(expected_seeds))
    missing_methods = [method for method in EXPECTED_MAIN_METHODS if method not in methods_in_data]

    duplicate_pairs = (
        raw.groupby(["seed", "method"], dropna=False)
        .size()
        .reset_index(name="rows")
        .query("rows != 1")
    )

    problems = []
    if missing_seeds:
        problems.append(f"missing seeds: {missing_seeds}")
    if extra_seeds:
        problems.append(f"unexpected seeds: {extra_seeds}")
    if missing_methods:
        problems.append(f"missing methods: {missing_methods}")
    if not duplicate_pairs.empty:
        problems.append(
            "seed/method pairs with row count != 1: "
            f"{duplicate_pairs.to_dict(orient='records')}"
        )
    if problems:
        raise RuntimeError("raw_results.csv validation failed: " + "; ".join(problems))

    return {
        "expected_seed_count": len(expected_seeds),
        "seed_count": len(seeds_in_data),
        "seeds_in_data": seeds_in_data,
        "expected_methods": EXPECTED_MAIN_METHODS,
        "methods_covered": methods_in_data,
        "row_count": int(len(raw)),
    }


# ---
# Summary figure
# ---


def plot_summary(
    raw: pd.DataFrame,
    figures_dir: Path,
    success: pd.DataFrame | None = None,
) -> None:
    """Bar chart of success rate and median selected-worst error per method."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    methods = list(dict.fromkeys(raw["method"].astype(str).to_list()))
    x = np.arange(len(methods))

    success_rates = []
    error_medians = []
    for method in methods:
        group = raw[raw["method"] == method]
        success_col = _success_as_bool(group["refinement_success"])
        success_rates.append(float(success_col.mean()))
        if LOWER_ERROR_IS_BETTER_METRIC in group.columns:
            error_medians.append(float(group[LOWER_ERROR_IS_BETTER_METRIC].median()))
        else:
            error_medians.append(float("nan"))

    yerr = None
    if success is not None and len(success):
        lookup = success.set_index("method")
        lower_err, upper_err = [], []
        for method, rate in zip(methods, success_rates):
            if method in lookup.index:
                iv = lookup.loc[method]
                lower_err.append(max(0.0, rate - float(iv["success_rate_wilson95_lower"])))
                upper_err.append(max(0.0, float(iv["success_rate_wilson95_upper"]) - rate))
            else:
                lower_err.append(0.0)
                upper_err.append(0.0)
        yerr = np.vstack([lower_err, upper_err])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    axes[0].bar(x, success_rates, yerr=yerr, capsize=3)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Refinement success rate (Wilson 95% CI)")

    axes[1].bar(x, error_medians)
    axes[1].set_ylabel("Median selected worst error")
    axes[1].set_title("Median refined selected-worst error")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(figures_dir / "main_method_statistics_summary.png", dpi=220)
    fig.savefig(figures_dir / "main_method_statistics_summary.pdf")
    plt.close(fig)


# ---
# Runner and postprocessor entry points
# ---


def run_benchmark(config_path: Path) -> None:
    """Run the shared experiment pipeline from the repository root."""
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        run_experiment(config_path)
    finally:
        os.chdir(previous_cwd)


def run_statistics(args: argparse.Namespace) -> dict:
    """Read raw_results.csv from a completed experiment and write statistical artifacts."""
    wall_start = time.perf_counter()

    config_path = _resolve(Path(args.config))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    results_dir, figures_dir, tables_dir = output_directories(ROOT, config)

    raw_csv = results_dir / "raw_results.csv"
    if not raw_csv.is_file():
        print(
            f"ERROR: raw_results.csv not found at {raw_csv}\n"
            "Run the experiment first:\n"
            f"  py -3.11 scripts/run_experiment.py --config {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = pd.read_csv(raw_csv)
    if raw.empty:
        print("ERROR: raw_results.csv is empty.", file=sys.stderr)
        sys.exit(1)

    validation = validate_main_method_results(raw, config)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    statistics = write_statistical_artifacts(raw, results_dir, tables_dir)

    if not args.no_figure:
        plot_summary(raw, figures_dir, statistics["success_intervals"])

    # --- metadata (conservative; no quantum-advantage claim) ---
    comparisons_df = statistics["paired_comparisons"]
    comparisons_meta: dict[str, dict] = {}
    for baseline in COMPARISON_BASELINES:
        baseline_cmp = comparisons_df[comparisons_df["baseline_method"] == baseline]
        for _, cmp_row in baseline_cmp.iterrows():
            method = str(cmp_row["method"])
            comparisons_meta.setdefault(baseline, {})[method] = {
                "paired_seeds": int(cmp_row["paired_seeds"]),
                "paired_success_delta_mean": float(cmp_row["paired_success_delta_mean"]),
                "selected_worst_error_diff_mean": float(
                    cmp_row["selected_worst_error_diff_mean"]
                ),
                "selected_worst_error_diff_mean_bootstrap95": [
                    float(cmp_row["selected_worst_error_diff_mean_bootstrap95_lower"]),
                    float(cmp_row["selected_worst_error_diff_mean_bootstrap95_upper"]),
                ],
                "selected_worst_error_sign_test_p_two_sided": float(
                    cmp_row["selected_worst_error_sign_test_p_two_sided"]
                ),
            }

    metadata: dict = {
        "command": " ".join(sys.argv),
        "python": sys.version,
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas", "pyyaml"]),
        **revision_metadata(),
        "config": _relative_or_absolute(config_path),
        "raw_results_csv": _relative_or_absolute(raw_csv),
        "raw_results_validation": validation,
        "seeds_in_data": validation["seeds_in_data"],
        "seed_count": validation["seed_count"],
        "methods_covered": validation["methods_covered"],
        "comparison_baselines": COMPARISON_BASELINES,
        "comparisons": comparisons_meta,
        "formal_statistics": {
            "success_interval": (
                "Wilson 95% confidence interval for the binomial refinement "
                "success rate per method."
            ),
            "paired_success_delta": (
                "Per-seed method success minus baseline success; +1 favors the "
                "method, -1 favors the baseline."
            ),
            "paired_selected_worst_error_diff": (
                "Per-seed method refined_selected_worst_error minus baseline "
                "refined_selected_worst_error; negative values favor the method."
            ),
            "comparison_baselines": COMPARISON_BASELINES,
            "selected_worst_error_sign_test": (
                "Two-sided exact sign test after excluding exact ties "
                "(|diff| <= 1e-12)."
            ),
            "bootstrap_ci": {
                "samples": STATISTICS_BOOTSTRAP_SAMPLES,
                "seed": STATISTICS_RNG_SEED,
                "confidence": 0.95,
            },
        },
        "statistical_artifacts": {
            "success_intervals_csv": _relative_or_absolute(
                results_dir / "success_intervals.csv"
            ),
            "success_intervals_json": _relative_or_absolute(
                results_dir / "success_intervals.json"
            ),
            "paired_success_deltas_csv": _relative_or_absolute(
                results_dir / "paired_success_deltas.csv"
            ),
            "paired_comparisons_csv": _relative_or_absolute(
                results_dir / "paired_comparisons.csv"
            ),
            "paired_comparisons_json": _relative_or_absolute(
                results_dir / "paired_comparisons.json"
            ),
            "statistics_table_tex": _relative_or_absolute(
                tables_dir / "main_method_statistics_table.tex"
            ),
        },
        "runtime_seconds": float(time.perf_counter() - wall_start),
        "interpretation_limits": [
            "This is a controlled CR3BP benchmark; not flight-ready trajectory optimization.",
            "QUBO/QAOA methods share 96 training evaluations; classical methods receive "
            "equal total true-objective budget.",
            "Paired error differences are method minus baseline; negative values favor the method.",
            "This package does not claim quantum advantage.",
            "Statistical significance is assessed by the exact two-sided sign test "
            "without parametric distributional assumptions.",
        ],
    }

    metadata_path = results_dir / "main_method_statistics_metadata.json"
    write_json(metadata_path, metadata)

    return metadata


def run_benchmark_and_statistics(args: argparse.Namespace) -> dict:
    config_path = _resolve(Path(args.config))
    if getattr(args, "run_experiment", False):
        run_benchmark(config_path)
    return run_statistics(args)


# ---
# CLI
# ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statistical postprocessor for the 30-seed phase-shift cardinality benchmark.\n"
            "Reads raw_results.csv produced by run_experiment.py and writes\n"
            "formal statistical artifacts without re-running any evaluations.\n\n"
            "To also run the experiment before computing statistics, pass --run-experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Path to the benchmark YAML config "
            "(default: configs/q1_phase_shift_cardinality_30seed.yaml)."
        ),
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--run-experiment",
        action="store_true",
        help=(
            "Run the benchmark experiment via run_experiment.py logic first, "
            "then compute statistics. By default only statistics are computed "
            "from an existing raw_results.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    metadata = run_benchmark_and_statistics(parse_args())
    print(
        f"Completed main-method statistics in {metadata['runtime_seconds']:.1f} s "
        f"over {metadata['seed_count']} seeds.",
        flush=True,
    )


if __name__ == "__main__":
    main()
