from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import (
    load_configured_states,
    make_objective_config,
    refinement_row,
    schedule_repair_variants,
    solver_configs_for_budget,
    state_target_mode,
    train_qubo,
)
from qlt.objective import Evaluator, outage_masks
from qlt.refinement import refine_schedule
from qlt.reporting import package_versions, revision_metadata, summarize, write_json
from qlt.solvers import SolverResult, solve_qaoa, solve_surrogate_sa
from qlt.surrogate import QuboModel, all_bitstrings


DEFAULT_CONFIG = Path("configs/qaoa_depth_ablation.yaml")
QUBO_TRAINING_EVALUATIONS = 96
SOLVER_TRUE_EVALUATIONS = 24
WILSON_Z_95 = 1.959963984540054
STATISTICS_BOOTSTRAP_SAMPLES = 20000
STATISTICS_RNG_SEED = 20260615
LOWER_ERROR_IS_BETTER_METRIC = "refined_selected_worst_error"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def validate_base_setup(config: dict, cfg) -> None:
    benchmark = config["benchmark"]
    failures = []
    if str(benchmark.get("target_mode")) != "catalog_halo_phase_shift":
        failures.append("benchmark.target_mode must be catalog_halo_phase_shift")
    if int(benchmark["segments"]) != 12 or int(cfg.n_segments) != 12:
        failures.append("benchmark.segments must be 12")
    if int(config["experiments"]["qubo_train_samples"]) != QUBO_TRAINING_EVALUATIONS:
        failures.append("experiments.qubo_train_samples must be 96")
    if int(config["experiments"]["solver_true_budget"]) != SOLVER_TRUE_EVALUATIONS:
        failures.append("experiments.solver_true_budget must be 24")
    expected_active = 11.0 / 12.0
    if abs(float(config["objective"].get("target_active_fraction", np.nan)) - expected_active) > 1e-12:
        failures.append("objective.target_active_fraction must be 11/12")
    if failures:
        raise ValueError("QAOA depth ablation requires the existing non-teacher cardinality-prior setup: " + "; ".join(failures))


def clean_outputs(results_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    patterns = {
        results_dir: [
            "raw_results.csv",
            "summary.csv",
            "summary.json",
            "qubo_diagnostics.csv",
            "metadata.json",
            "success_intervals.csv",
            "success_intervals.json",
            "paired_success_deltas.csv",
            "paired_comparisons.csv",
            "paired_comparisons.json",
            "qubo_coefficients_seed*.json",
            "qubo_fit_seed*.csv",
        ],
        figures_dir: [
            "qaoa_depth_ablation_summary.png",
            "qaoa_depth_ablation_summary.pdf",
        ],
        tables_dir: [
            "qaoa_depth_ablation_table.tex",
            "qaoa_depth_ablation_statistics_table.tex",
        ],
    }
    for directory, names in patterns.items():
        for name in names:
            for path in directory.glob(name):
                if path.is_file():
                    path.unlink()


def exact_qubo_top_candidates(evaluator: Evaluator, qubo: QuboModel, candidates_to_evaluate: int) -> SolverResult:
    start_count = evaluator.count
    start_time = time.perf_counter()
    bits = all_bitstrings(qubo.n)
    energies = qubo.energy(bits)
    order = np.argsort(energies, kind="mergesort")[: int(candidates_to_evaluate)]
    rows = [(bits[int(i)].astype(int), evaluator.evaluate(bits[int(i)])) for i in order]
    best_schedule, best_metrics = min(rows, key=lambda item: item[1]["objective"])
    return SolverResult(
        method="exact_qubo_top24",
        best_schedule=best_schedule.copy(),
        best_metrics=dict(best_metrics),
        evaluated=rows,
        true_evaluations=evaluator.count - start_count,
        runtime_seconds=time.perf_counter() - start_time,
    )


def refine_solver_result(
    *,
    result: SolverResult,
    seed: int,
    states,
    cfg,
    config: dict,
    masks: np.ndarray,
    benchmark_label: str,
    target_mode: str,
) -> dict:
    thresholds = config["objective"]["thresholds"]
    ranked = sorted(result.evaluated, key=lambda item: item[1]["objective"])
    refine_candidates = ranked[: int(config["experiments"]["top_refine_per_method"])]
    best_refinement = None
    best_refined_schedule = None
    best_schedule_repair = None
    best_refined_schedule_source = None
    best_refinement_candidate_schedule = None
    for schedule, _ in refine_candidates:
        for repair_name, repaired_schedule, repair_description in schedule_repair_variants(schedule, config):
            refined = refine_schedule(
                repaired_schedule,
                states.initial,
                states.target,
                cfg,
                masks,
                config["refinement"],
                thresholds,
            )
            if best_refinement is None or refined.get("worst_error", np.inf) < best_refinement.get("worst_error", np.inf):
                best_refinement = refined
                best_refined_schedule = repaired_schedule.copy()
                best_schedule_repair = repair_name
                best_refined_schedule_source = repair_description
                best_refinement_candidate_schedule = schedule.copy()
    assert best_refinement is not None
    assert best_refined_schedule is not None
    assert best_schedule_repair is not None
    assert best_refined_schedule_source is not None
    assert best_refinement_candidate_schedule is not None
    row = refinement_row(
        seed=seed,
        method=result.method,
        schedule=result.best_schedule,
        refined_schedule=best_refined_schedule,
        base=result.best_metrics,
        solver_true_evaluations=result.true_evaluations,
        shared_training=QUBO_TRAINING_EVALUATIONS,
        runtime_seconds=result.runtime_seconds,
        best_refinement=best_refinement,
        target_mode=target_mode,
        benchmark_label=benchmark_label,
        comparison_group="qaoa_depth_ablation",
        schedule_repair=best_schedule_repair,
        refined_schedule_source=best_refined_schedule_source,
        refinement_candidate_schedule=best_refinement_candidate_schedule,
    )
    for key, value in result.best_metrics.items():
        if key.startswith("qaoa_"):
            row[key] = value
    return row


def variant_qaoa_config(base_qaoa_cfg: dict, variant_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_qaoa_cfg)
    cfg.update(copy.deepcopy(variant_cfg))
    if cfg.get("optimizer") in {"random_trials", "random"}:
        cfg.pop("optimizer", None)
    return cfg


def run_seed(
    *,
    seed: int,
    states,
    cfg,
    base_config: dict,
    ablation_config: dict,
    results_dir: Path,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    evaluator = Evaluator(states.initial, states.target, cfg)
    qubo, fit_path, qubo_training = train_qubo(evaluator, rng, base_config, seed, results_dir)
    if qubo_training != QUBO_TRAINING_EVALUATIONS:
        raise RuntimeError(f"expected 96 QUBO training evaluations, got {qubo_training}")
    solver_cfgs = solver_configs_for_budget(base_config, qubo_training)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    benchmark_label = str(base_config.get("run", {}).get("label"))
    target_mode = state_target_mode(states)

    method_results: list[SolverResult] = [
        solve_surrogate_sa(evaluator, rng, cfg.n_segments, solver_cfgs["surrogate_sa"], qubo),
        exact_qubo_top_candidates(evaluator, qubo, SOLVER_TRUE_EVALUATIONS),
    ]
    for name, variant_cfg in ablation_config["qaoa_variants"].items():
        result = solve_qaoa(
            evaluator,
            rng,
            variant_qaoa_config(solver_cfgs["qaoa"], variant_cfg),
            qubo,
        )
        result.method = f"qaoa_{name}"
        method_results.append(result)

    rows = [
        refine_solver_result(
            result=result,
            seed=seed,
            states=states,
            cfg=cfg,
            config=base_config,
            masks=masks,
            benchmark_label=benchmark_label,
            target_mode=target_mode,
        )
        for result in method_results
    ]
    for row in rows:
        if int(row["solver_true_evaluations"]) != SOLVER_TRUE_EVALUATIONS:
            raise RuntimeError(f"{row['method']} used {row['solver_true_evaluations']} true candidate evaluations, expected 24")
        if int(row["shared_qubo_training_evaluations"]) != QUBO_TRAINING_EVALUATIONS:
            raise RuntimeError(f"{row['method']} has incorrect QUBO training charge")
        row["qubo_fit_path"] = str(fit_path.relative_to(ROOT) if fit_path.is_relative_to(ROOT) else fit_path)

    qubo_diag = {
        "seed": seed,
        **qubo.diagnostics,
        "shared_qubo_training_evaluations": qubo_training,
        "qubo_training_true_evaluations": qubo_training,
    }
    return rows, qubo_diag


def write_latex(summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "method",
        "runs",
        "refinement_success_rate",
        "refined_nominal_error_median",
        "refined_selected_worst_error_median",
        "refined_nominal_fuel_median",
        "solver_true_evaluations_median",
        "shared_qubo_training_evaluations_median",
        "runtime_seconds_median",
    ]
    available = [col for col in cols if col in summary.columns]
    summary[available].to_latex(
        tables_dir / "qaoa_depth_ablation_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )


def _success_as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    return values.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def wilson_interval(successes: int, runs: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if runs <= 0:
        return (float("nan"), float("nan"))
    phat = successes / runs
    denominator = 1.0 + z**2 / runs
    center = (phat + z**2 / (2.0 * runs)) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) / runs) + (z**2 / (4.0 * runs**2))) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def exact_sign_test_pvalue(differences: np.ndarray, tolerance: float = 1e-12) -> tuple[int, int, int, float]:
    finite = np.asarray(differences, dtype=float)
    finite = finite[np.isfinite(finite)]
    wins = int(np.sum(finite < -tolerance))
    losses = int(np.sum(finite > tolerance))
    ties = int(finite.size - wins - losses)
    trials = wins + losses
    if trials == 0:
        return wins, losses, ties, 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(trials, index) for index in range(smaller + 1)) / (2.0**trials)
    return wins, losses, ties, float(min(1.0, 2.0 * tail))


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str,
    samples: int = STATISTICS_BOOTSTRAP_SAMPLES,
    seed: int = STATISTICS_RNG_SEED,
) -> tuple[float, float]:
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


def success_intervals(raw: pd.DataFrame) -> pd.DataFrame:
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


def paired_statistics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = raw.copy()
    rows["refinement_success_bool"] = _success_as_bool(rows["refinement_success"])
    baselines = ["surrogate_qubo_sa", "qaoa_random_p1"]
    delta_rows: list[dict] = []
    comparison_rows: list[dict] = []
    available_methods = list(dict.fromkeys(rows["method"].astype(str).to_list()))
    for baseline in baselines:
        if baseline not in available_methods:
            continue
        if baseline == "qaoa_random_p1":
            methods = [method for method in available_methods if method.startswith("qaoa_optimized")]
        else:
            methods = [method for method in available_methods if method != baseline]
        baseline_rows = rows[rows["method"] == baseline][
            ["seed", "refinement_success_bool", LOWER_ERROR_IS_BETTER_METRIC]
        ].rename(
            columns={
                "refinement_success_bool": "baseline_success",
                LOWER_ERROR_IS_BETTER_METRIC: "baseline_selected_worst_error",
            }
        )
        for method in methods:
            method_rows = rows[rows["method"] == method][
                ["seed", "refinement_success_bool", LOWER_ERROR_IS_BETTER_METRIC]
            ].rename(
                columns={
                    "refinement_success_bool": "method_success",
                    LOWER_ERROR_IS_BETTER_METRIC: "method_selected_worst_error",
                }
            )
            paired = method_rows.merge(baseline_rows, on="seed", how="inner").sort_values("seed")
            if paired.empty:
                continue
            paired["success_delta"] = paired["method_success"].astype(int) - paired["baseline_success"].astype(int)
            paired["selected_worst_error_diff"] = (
                paired["method_selected_worst_error"].astype(float) - paired["baseline_selected_worst_error"].astype(float)
            )
            for row in paired.itertuples(index=False):
                delta_rows.append(
                    {
                        "baseline_method": baseline,
                        "method": method,
                        "seed": int(row.seed),
                        "baseline_success": bool(row.baseline_success),
                        "method_success": bool(row.method_success),
                        "success_delta": int(row.success_delta),
                        "baseline_selected_worst_error": float(row.baseline_selected_worst_error),
                        "method_selected_worst_error": float(row.method_selected_worst_error),
                        "selected_worst_error_diff": float(row.selected_worst_error_diff),
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


def write_statistics_latex(success: pd.DataFrame, comparisons: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    surrogate = comparisons[comparisons["baseline_method"] == "surrogate_qubo_sa"]
    qaoa_random = comparisons[comparisons["baseline_method"] == "qaoa_random_p1"]
    for row in success.itertuples(index=False):
        method = str(row.method)
        vs_sa = surrogate[surrogate["method"] == method]
        vs_random = qaoa_random[qaoa_random["method"] == method]
        table_row = {
            "method": method,
            "success": f"{int(row.successes)}/{int(row.runs)}",
            "wilson_95_ci": f"[{row.success_rate_wilson95_lower:.2f}, {row.success_rate_wilson95_upper:.2f}]",
            "delta_success_vs_sa": "",
            "mean_error_delta_vs_sa": "",
            "sign_p_vs_sa": "",
            "delta_success_vs_random_qaoa": "",
            "mean_error_delta_vs_random_qaoa": "",
            "sign_p_vs_random_qaoa": "",
        }
        if len(vs_sa):
            cmp_row = vs_sa.iloc[0]
            table_row["delta_success_vs_sa"] = f"{cmp_row['paired_success_delta_mean']:+.2f}"
            table_row["mean_error_delta_vs_sa"] = (
                f"{cmp_row['selected_worst_error_diff_mean']:+.4f} "
                f"[{cmp_row['selected_worst_error_diff_mean_bootstrap95_lower']:+.4f}, "
                f"{cmp_row['selected_worst_error_diff_mean_bootstrap95_upper']:+.4f}]"
            )
            table_row["sign_p_vs_sa"] = f"{cmp_row['selected_worst_error_sign_test_p_two_sided']:.4f}"
        if len(vs_random):
            cmp_row = vs_random.iloc[0]
            table_row["delta_success_vs_random_qaoa"] = f"{cmp_row['paired_success_delta_mean']:+.2f}"
            table_row["mean_error_delta_vs_random_qaoa"] = (
                f"{cmp_row['selected_worst_error_diff_mean']:+.4f} "
                f"[{cmp_row['selected_worst_error_diff_mean_bootstrap95_lower']:+.4f}, "
                f"{cmp_row['selected_worst_error_diff_mean_bootstrap95_upper']:+.4f}]"
            )
            table_row["sign_p_vs_random_qaoa"] = f"{cmp_row['selected_worst_error_sign_test_p_two_sided']:.4f}"
        rows.append(table_row)
    pd.DataFrame(rows).to_latex(
        tables_dir / "qaoa_depth_ablation_statistics_table.tex",
        index=False,
        escape=True,
    )


def write_statistical_artifacts(raw: pd.DataFrame, results_dir: Path, tables_dir: Path) -> dict:
    success = success_intervals(raw)
    deltas, comparisons = paired_statistics(raw)
    success.to_csv(results_dir / "success_intervals.csv", index=False)
    deltas.to_csv(results_dir / "paired_success_deltas.csv", index=False)
    comparisons.to_csv(results_dir / "paired_comparisons.csv", index=False)
    write_json(results_dir / "success_intervals.json", success.to_dict(orient="records"))
    write_json(results_dir / "paired_comparisons.json", comparisons.to_dict(orient="records"))
    write_statistics_latex(success, comparisons, tables_dir)
    return {
        "success_intervals": success,
        "paired_success_deltas": deltas,
        "paired_comparisons": comparisons,
    }


def plot_summary(summary: pd.DataFrame, figures_dir: Path, success: pd.DataFrame | None = None) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["method"].astype(str).to_list()
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    success_rates = summary["refinement_success_rate"].to_numpy(dtype=float)
    yerr = None
    if success is not None and len(success):
        interval_lookup = success.set_index("method")
        lower = []
        upper = []
        for method, rate in zip(labels, success_rates):
            if method in interval_lookup.index:
                interval = interval_lookup.loc[method]
                lower.append(max(0.0, rate - float(interval["success_rate_wilson95_lower"])))
                upper.append(max(0.0, float(interval["success_rate_wilson95_upper"]) - rate))
            else:
                lower.append(0.0)
                upper.append(0.0)
        yerr = np.vstack([lower, upper])
    axes[0].bar(x, success_rates, yerr=yerr, capsize=3)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Success rate")
    axes[1].bar(x, summary["refined_selected_worst_error_median"])
    axes[1].set_ylabel("Median selected worst error")
    axes[2].bar(x, summary["refined_nominal_fuel_median"])
    axes[2].set_ylabel("Median nominal fuel")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "qaoa_depth_ablation_summary.png", dpi=220)
    fig.savefig(figures_dir / "qaoa_depth_ablation_summary.pdf")
    plt.close(fig)


def run_ablation(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    ablation_path = _resolve(args.config)
    ablation_config = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    base_path = _resolve(Path(ablation_config["base_config"]))
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if args.seeds:
        ablation_config["seeds"] = [int(seed) for seed in args.seeds]
    if args.angle_restarts is not None:
        for variant in ablation_config["qaoa_variants"].values():
            if variant.get("optimizer") == "scipy_minimize":
                variant["angle_restarts"] = int(args.angle_restarts)
    if args.maxiter is not None:
        for variant in ablation_config["qaoa_variants"].values():
            if variant.get("optimizer") == "scipy_minimize":
                variant["maxiter"] = int(args.maxiter)

    results_dir = _resolve(Path(ablation_config["results_dir"]))
    figures_dir = _resolve(Path(ablation_config["figures_dir"]))
    tables_dir = _resolve(Path(ablation_config["tables_dir"]))
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    clean_outputs(results_dir, figures_dir, tables_dir)

    states = load_configured_states(ROOT, base_config)
    cfg = make_objective_config(base_config, states.mu)
    validate_base_setup(base_config, cfg)

    raw_rows: list[dict] = []
    qubo_rows: list[dict] = []
    seeds = [int(seed) for seed in ablation_config["seeds"]]
    for index, seed in enumerate(seeds, start=1):
        print(f"[qaoa depth ablation {index}/{len(seeds)}] seed={seed}", flush=True)
        rows, qubo_diag = run_seed(
            seed=seed,
            states=states,
            cfg=cfg,
            base_config=base_config,
            ablation_config=ablation_config,
            results_dir=results_dir,
        )
        raw_rows.extend(rows)
        qubo_rows.append(qubo_diag)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    qubo_diag = pd.DataFrame(qubo_rows)
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    qubo_diag.to_csv(results_dir / "qubo_diagnostics.csv", index=False)
    write_json(results_dir / "summary.json", summary.to_dict(orient="records"))
    write_latex(summary, tables_dir)
    statistics = write_statistical_artifacts(raw, results_dir, tables_dir)
    if not args.no_figure:
        plot_summary(summary, figures_dir, statistics["success_intervals"])

    random_row = summary[summary["method"] == "qaoa_random_p1"]
    comparisons = {}
    if len(random_row):
        random_success = float(random_row.iloc[0]["refinement_success_rate"])
        random_error = float(random_row.iloc[0]["refined_selected_worst_error_median"])
        for method in ["qaoa_optimized_p1", "qaoa_optimized_p2"]:
            row = summary[summary["method"] == method]
            if len(row):
                comparisons[method] = {
                    "success_rate_delta_vs_qaoa_random_p1": float(row.iloc[0]["refinement_success_rate"]) - random_success,
                    "selected_worst_error_delta_vs_qaoa_random_p1": float(row.iloc[0]["refined_selected_worst_error_median"]) - random_error,
                }

    metadata = {
        "command": " ".join(sys.argv),
        "python": sys.version,
        "packages": package_versions(["numpy", "scipy", "matplotlib", "pandas", "pyyaml"]),
        **revision_metadata(),
        "ablation_config": str(ablation_path.relative_to(ROOT) if ablation_path.is_relative_to(ROOT) else ablation_path),
        "base_config": str(base_path.relative_to(ROOT) if base_path.is_relative_to(ROOT) else base_path),
        "target_mode": state_target_mode(states),
        "seeds": seeds,
        "pilot": len(seeds) < 10,
        "qubo_training_evaluations_per_seed": QUBO_TRAINING_EVALUATIONS,
        "solver_true_candidate_evaluations_per_method_seed": SOLVER_TRUE_EVALUATIONS,
        "qaoa_variants": ablation_config["qaoa_variants"],
        "comparisons_vs_qaoa_random_p1": comparisons,
        "formal_statistics": {
            "success_interval": "Wilson 95% confidence interval for the binomial refinement success rate per method.",
            "paired_success_delta": "Per-seed method success minus baseline success; +1 favors the method, -1 favors the baseline.",
            "paired_selected_worst_error_diff": (
                "Per-seed method refined_selected_worst_error minus baseline refined_selected_worst_error; "
                "negative values favor the method."
            ),
            "paired_comparison_baselines": ["surrogate_qubo_sa", "qaoa_random_p1"],
            "selected_worst_error_sign_test": "Two-sided exact sign test after excluding exact ties.",
            "bootstrap_ci": {
                "samples": STATISTICS_BOOTSTRAP_SAMPLES,
                "seed": STATISTICS_RNG_SEED,
                "confidence": 0.95,
            },
        },
        "statistical_artifacts": {
            "success_intervals_csv": str((results_dir / "success_intervals.csv").relative_to(ROOT)),
            "success_intervals_json": str((results_dir / "success_intervals.json").relative_to(ROOT)),
            "paired_success_deltas_csv": str((results_dir / "paired_success_deltas.csv").relative_to(ROOT)),
            "paired_comparisons_csv": str((results_dir / "paired_comparisons.csv").relative_to(ROOT)),
            "paired_comparisons_json": str((results_dir / "paired_comparisons.json").relative_to(ROOT)),
            "statistics_table_tex": str((tables_dir / "qaoa_depth_ablation_statistics_table.tex").relative_to(ROOT)),
        },
        "runtime_seconds": float(time.perf_counter() - start),
        "interpretation_limits": [
            "The optimized-angle QAOA variants optimize expected surrogate QUBO energy only.",
            "All QUBO-derived methods receive the same 96 shared QUBO training evaluations and 24 post-QUBO true candidate evaluations per seed.",
            "This ablation does not claim quantum advantage.",
            "Paired error differences are method minus baseline; negative values favor the method.",
        ],
    }
    write_json(results_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimized-angle QAOA depth ablation on the phase-shift cardinality benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional deterministic seed subset.")
    parser.add_argument("--angle-restarts", type=int, default=None, help="Override optimized QAOA angle restarts.")
    parser.add_argument("--maxiter", type=int, default=None, help="Override optimized QAOA scipy maxiter.")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run_ablation(parse_args())
    print(f"completed QAOA depth ablation in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
