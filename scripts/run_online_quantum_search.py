from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.cr3bp import propagate_feedback_batch
from qlt.experiment import load_configured_states, make_objective_config
from qlt.objective import outage_masks, schedule_to_string, state_error
from qlt.online_quantum_search import (
    cross_entropy_baseline,
    genetic_baseline,
    online_threshold_search,
    ranked_baseline,
    simulated_annealing_baseline,
    uniform_random_baseline,
)
from qlt.reporting import package_versions, revision_metadata, write_json


DEFAULT_CARDINALITY = Path("data/results/phase_shift_cardinality_ablation/cardinality_refinements.csv")
DEFAULT_CONFIG = Path("configs/q1_phase_shift_cardinality.yaml")
RECORDED_GROUPS = ["one_coast_k11_full", "k10_full"]
SCREENING_CASES = [(16, 12), (20, 16)]
OPTIONAL_SCREENING_CASES = [(24, 20)]
WILSON_Z_95 = 1.959963984540054
STATISTICS_BOOTSTRAP_SAMPLES = 20_000
STATISTICS_RNG_SEED = 20260618
ONLINE_METHOD = "online_threshold_amplitude_amplification"
DEPLOYABLE_BASELINES = [
    "uniform_random_without_replacement",
    "simulated_annealing_baseline",
    "cross_entropy_baseline",
    "genetic_baseline",
]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _command_string() -> str:
    argv = getattr(sys, "orig_argv", sys.argv)
    return subprocess.list2cmdline([str(arg) for arg in argv])


def _enumerate_cardinality(n_segments: int, active_windows: int) -> np.ndarray:
    schedules = []
    for inactive in combinations(range(n_segments), n_segments - active_windows):
        schedule = np.ones(n_segments, dtype=int)
        schedule[list(inactive)] = 0
        schedules.append(schedule)
    return np.asarray(schedules, dtype=int)


def _budget_grid(m: int) -> list[tuple[str, int]]:
    root = int(np.ceil(np.sqrt(m)))
    values = [
        ("sqrtM", root),
        ("2sqrtM", 2 * root),
        ("4sqrtM", 4 * root),
        ("M", m),
    ]
    dedup: list[tuple[str, int]] = []
    seen = set()
    for label, value in values:
        value = max(1, min(int(value), int(m)))
        if (label, value) not in seen:
            dedup.append((label, value))
            seen.add((label, value))
    return dedup


def _schedule_bits(schedules: list[str]) -> np.ndarray:
    return np.asarray([[int(ch) for ch in str(schedule)] for schedule in schedules], dtype=int)


def _recorded_subspace(cardinality_path: Path, group: str) -> dict:
    data = pd.read_csv(cardinality_path, dtype={"schedule": str})
    subspace = data[data["group"].astype(str) == group].copy()
    if subspace.empty:
        raise RuntimeError(f"missing recorded cardinality group: {group}")
    subspace["schedule"] = subspace["schedule"].astype(str).str.zfill(12)
    if group == "one_coast_k11_full":
        subspace["_sort_key"] = pd.to_numeric(subspace["coast_windows_zero_based"], errors="raise")
    else:
        subspace["_sort_key"] = subspace["schedule"]
    subspace = subspace.sort_values("_sort_key", kind="mergesort").drop(columns="_sort_key").reset_index(drop=True)
    costs = subspace["refined_selected_worst_error"].to_numpy(dtype=float)
    score_col = "feedback_worst_error" if "feedback_worst_error" in subspace.columns else "true_objective"
    scores = subspace[score_col].to_numpy(dtype=float) if score_col in subspace.columns else costs
    return {
        "case_id": group,
        "evidence_tier": "recorded_branch_refined",
        "n_segments": 12,
        "active_windows": int(subspace["k_active"].iloc[0]),
        "schedules": subspace["schedule"].astype(str).to_list(),
        "costs": costs,
        "surrogate_scores": scores,
        "cost_column": "refined_selected_worst_error",
        "score_column": score_col,
        "source": str(cardinality_path.relative_to(ROOT)),
        "state_metrics": subspace,
    }


def _screening_subspace(config_path: Path, n_segments: int, active_windows: int) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["benchmark"]["segments"] = int(n_segments)
    config["objective"]["target_active_fraction"] = float(active_windows) / float(n_segments)
    states = load_configured_states(ROOT, config)
    cfg = make_objective_config(config, states.mu)
    schedules = _enumerate_cardinality(n_segments, active_windows)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    all_schedules = np.concatenate(
        [schedules[:, None, :], schedules[:, None, :] * masks[None, :, :].astype(int)],
        axis=1,
    )
    flat_schedules = all_schedules.reshape(-1, cfg.n_segments)
    finals, _ = propagate_feedback_batch(
        states.initial,
        flat_schedules,
        states.target,
        cfg.mu,
        cfg.tf,
        cfg.amax,
        cfg.kr,
        cfg.kv,
        cfg.substeps,
    )
    errors = state_error(finals, states.target, cfg.position_scale, cfg.velocity_scale).reshape(schedules.shape[0], -1)
    nominal = errors[:, 0]
    outage = errors[:, 1:]
    worst = np.max(outage, axis=1) if outage.size else nominal
    degradation = np.maximum(0.0, worst - nominal)
    active = np.mean(schedules, axis=1)
    smooth = np.mean(np.abs(np.diff(schedules, axis=1)), axis=1)
    target_fraction = cfg.target_active_fraction
    active_target_penalty = np.zeros(schedules.shape[0], dtype=float)
    if target_fraction is not None and cfg.target_active_weight != 0.0:
        active_target_penalty = float(cfg.target_active_weight) * (active - float(target_fraction)) ** 2
    weights = cfg.weights
    objective = (
        weights["nominal"] * nominal
        + weights["robust_worst"] * worst
        + weights["robust_degradation"] * degradation
        + weights["active_fraction"] * active
        + weights["smoothness"] * smooth
        + active_target_penalty
    )
    state_metrics = pd.DataFrame(
        {
            "schedule": [schedule_to_string(row) for row in schedules],
            "feedback_objective": objective,
            "feedback_nominal_error": nominal,
            "feedback_worst_error": worst,
            "feedback_robust_degradation": degradation,
            "active_fraction": active,
            "smoothness": smooth,
        }
    )
    case_id = f"feedback_N{n_segments}_k{active_windows}"
    return {
        "case_id": case_id,
        "evidence_tier": "feedback_screening",
        "n_segments": int(n_segments),
        "active_windows": int(active_windows),
        "schedules": state_metrics["schedule"].to_list(),
        "costs": objective.astype(float),
        "surrogate_scores": worst.astype(float),
        "cost_column": "feedback_objective",
        "score_column": "feedback_worst_error",
        "source": str(config_path.relative_to(ROOT)),
        "state_metrics": state_metrics,
    }


def _run_case(case: dict, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    costs = np.asarray(case["costs"], dtype=float)
    scores = np.asarray(case["surrogate_scores"], dtype=float)
    schedules = _schedule_bits(case["schedules"])
    rows = []
    for budget_label, budget in _budget_grid(costs.size):
        for seed in seeds:
            runs = [
                online_threshold_search(costs, budget=budget, seed=seed),
                uniform_random_baseline(costs, budget=budget, seed=seed),
                simulated_annealing_baseline(costs, budget=budget, seed=seed, schedules=schedules),
                cross_entropy_baseline(costs, budget=budget, seed=seed, schedules=schedules),
                genetic_baseline(costs, budget=budget, seed=seed, schedules=schedules),
                ranked_baseline(costs, budget=budget, seed=seed, scores=scores),
            ]
            for result in runs:
                row = asdict(result)
                row.update(
                    {
                        "case_id": case["case_id"],
                        "evidence_tier": case["evidence_tier"],
                        "n_segments": int(case["n_segments"]),
                        "active_windows": int(case["active_windows"]),
                        "M": int(costs.size),
                        "budget_label": budget_label,
                        "cost_column": case["cost_column"],
                        "score_column": case["score_column"],
                    }
                )
                rows.append(row)
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(
            [
                "evidence_tier",
                "case_id",
                "n_segments",
                "active_windows",
                "M",
                "budget_label",
                "budget",
                "method",
            ],
            sort=False,
        )
        .agg(
            runs=("seed", "count"),
            successes=("success_global_optimum", "sum"),
            success_rate=("success_global_optimum", "mean"),
            top5_rate=("hit_top5", "mean"),
            top10_percent_rate=("hit_top10_percent", "mean"),
            median_best_cost=("best_cost", "median"),
            mean_best_cost=("best_cost", "mean"),
            median_cost_evaluations=("cost_evaluations", "median"),
            median_threshold_oracle_calls=("threshold_oracle_calls", "median"),
            median_total_oracle_calls=("total_oracle_calls", "median"),
            median_accepted_updates=("accepted_updates", "median"),
        )
        .reset_index()
    )
    intervals = [
        wilson_interval(int(row.successes), int(row.runs))
        for row in summary.itertuples(index=False)
    ]
    summary["success_rate_wilson95_lower"] = [float(lower) for lower, _ in intervals]
    summary["success_rate_wilson95_upper"] = [float(upper) for _, upper in intervals]
    return raw, summary


def wilson_interval(successes: int, runs: int, z: float = WILSON_Z_95) -> tuple[float, float]:
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
    tail = sum(math.comb(trials, k) for k in range(smaller + 1)) / (2.0 ** trials)
    return wins, losses, ties, float(min(1.0, 2.0 * tail))


def bootstrap_ci(
    values: np.ndarray,
    *,
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
    estimates = np.mean(finite[indices], axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return (float(lower), float(upper))


def format_pvalue(value: float) -> str:
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


def paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["evidence_tier", "case_id", "n_segments", "active_windows", "M", "budget_label", "budget"]
    for keys, group in raw.groupby(group_columns, sort=False):
        key_data = dict(zip(group_columns, keys))
        online = group[group["method"] == ONLINE_METHOD][["seed", "success_global_optimum", "best_cost"]].rename(
            columns={"success_global_optimum": "online_success", "best_cost": "online_best_cost"}
        )
        for baseline in DEPLOYABLE_BASELINES:
            baseline_rows = group[group["method"] == baseline][["seed", "success_global_optimum", "best_cost"]].rename(
                columns={"success_global_optimum": "baseline_success", "best_cost": "baseline_best_cost"}
            )
            paired = online.merge(baseline_rows, on="seed", how="inner").sort_values("seed")
            if paired.empty:
                continue
            paired["success_delta_online_minus_baseline"] = (
                paired["online_success"].astype(int) - paired["baseline_success"].astype(int)
            )
            paired["best_cost_diff_online_minus_baseline"] = (
                paired["online_best_cost"].astype(float) - paired["baseline_best_cost"].astype(float)
            )
            diffs = paired["best_cost_diff_online_minus_baseline"].to_numpy(dtype=float)
            wins, losses, ties, pvalue = exact_sign_test_pvalue(diffs)
            lower, upper = bootstrap_ci(diffs)
            rows.append(
                {
                    **key_data,
                    "online_method": ONLINE_METHOD,
                    "baseline_method": baseline,
                    "paired_seeds": int(paired["seed"].nunique()),
                    "online_successes": int(paired["online_success"].sum()),
                    "baseline_successes": int(paired["baseline_success"].sum()),
                    "paired_success_delta_mean": float(paired["success_delta_online_minus_baseline"].mean()),
                    "paired_success_delta_sum": int(paired["success_delta_online_minus_baseline"].sum()),
                    "best_cost_diff_mean_online_minus_baseline": float(np.mean(diffs)),
                    "best_cost_diff_mean_bootstrap95_lower": lower,
                    "best_cost_diff_mean_bootstrap95_upper": upper,
                    "best_cost_diff_median_online_minus_baseline": float(np.median(diffs)),
                    "best_cost_diff_negative_favors_online": True,
                    "best_cost_online_wins": wins,
                    "best_cost_online_losses": losses,
                    "best_cost_ties": ties,
                    "best_cost_sign_test_p_two_sided": pvalue,
                }
            )
    return pd.DataFrame(rows)


def _write_tables(summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    compact = summary[
        [
            "evidence_tier",
            "case_id",
            "M",
            "budget_label",
            "budget",
            "method",
            "successes",
            "runs",
            "success_rate",
            "success_rate_wilson95_lower",
            "success_rate_wilson95_upper",
            "top10_percent_rate",
            "median_best_cost",
            "median_total_oracle_calls",
        ]
    ].copy()
    compact["method"] = compact["method"].replace(
        {
            "online_threshold_amplitude_amplification": "online_threshold_AA",
            "uniform_random_without_replacement": "uniform_random",
            "simulated_annealing_baseline": "simulated_annealing",
            "cross_entropy_baseline": "cross_entropy",
            "genetic_baseline": "genetic",
            "surrogate_ranked_baseline": "surrogate_ranked",
        }
    )
    compact.to_latex(tables_dir / "online_quantum_search_table.tex", index=False, float_format="%.4f", escape=True)


def _write_statistics_table(comparisons: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    compact = comparisons[
        [
            "case_id",
            "budget_label",
            "budget",
            "baseline_method",
            "online_successes",
            "baseline_successes",
            "paired_success_delta_mean",
            "best_cost_diff_mean_online_minus_baseline",
            "best_cost_diff_mean_bootstrap95_lower",
            "best_cost_diff_mean_bootstrap95_upper",
            "best_cost_sign_test_p_two_sided",
        ]
    ].copy()
    compact["baseline_method"] = compact["baseline_method"].replace(
        {
            "uniform_random_without_replacement": "uniform",
            "simulated_annealing_baseline": "SA",
            "cross_entropy_baseline": "CEM",
            "genetic_baseline": "GA",
        }
    )
    compact["success_online_vs_baseline"] = (
        compact["online_successes"].astype(int).astype(str) + "/" + compact["baseline_successes"].astype(int).astype(str)
    )
    compact["mean_cost_diff_bootstrap95"] = compact.apply(
        lambda row: (
            f"{row['best_cost_diff_mean_online_minus_baseline']:+.4f} "
            f"[{row['best_cost_diff_mean_bootstrap95_lower']:+.4f}, "
            f"{row['best_cost_diff_mean_bootstrap95_upper']:+.4f}]"
        ),
        axis=1,
    )
    compact["sign_p"] = compact["best_cost_sign_test_p_two_sided"].map(format_pvalue)
    out = compact[
        [
            "case_id",
            "budget_label",
            "baseline_method",
            "success_online_vs_baseline",
            "paired_success_delta_mean",
            "mean_cost_diff_bootstrap95",
            "sign_p",
        ]
    ]
    out.to_latex(tables_dir / "online_quantum_search_statistics_table.tex", index=False, escape=True)


def _plot(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        "online_threshold_amplitude_amplification",
        "uniform_random_without_replacement",
        "simulated_annealing_baseline",
        "cross_entropy_baseline",
        "genetic_baseline",
        "surrogate_ranked_baseline",
    ]
    method_labels = {
        "online_threshold_amplitude_amplification": "online threshold AA",
        "uniform_random_without_replacement": "uniform random",
        "simulated_annealing_baseline": "SA",
        "cross_entropy_baseline": "CEM",
        "genetic_baseline": "GA",
        "surrogate_ranked_baseline": "surrogate ranked",
    }
    cases = summary["case_id"].drop_duplicates().to_list()
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.0, max(3.0, 2.2 * len(cases))), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, case_id in zip(axes, cases):
        case_rows = summary[summary["case_id"] == case_id]
        x_labels = case_rows["budget_label"].drop_duplicates().to_list()
        x = np.arange(len(x_labels))
        for method in methods:
            y = []
            for label in x_labels:
                row = case_rows[(case_rows["method"] == method) & (case_rows["budget_label"] == label)]
                y.append(float(row["success_rate"].iloc[0]) if len(row) else np.nan)
            ax.plot(x, y, marker="o", linewidth=1.3, label=method_labels[method])
        ax.set_title(case_id)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("global optimum hit rate")
        ax.set_xticks(x, x_labels)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(figures_dir / "online_quantum_search_success.png", dpi=220)
    fig.savefig(figures_dir / "online_quantum_search_success.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online threshold quantum-search simulation.")
    parser.add_argument("--cardinality", type=Path, default=DEFAULT_CARDINALITY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--include-largest", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/online_quantum_search"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/online_quantum_search"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables/online_quantum_search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    results_dir = _resolve(args.results_dir)
    figures_dir = _resolve(args.figures_dir)
    tables_dir = _resolve(args.tables_dir)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cardinality = _resolve(args.cardinality)
    config_path = _resolve(args.config)
    seeds = list(range(int(args.seeds)))
    cases = [_recorded_subspace(cardinality, group) for group in RECORDED_GROUPS]
    skipped = []
    for n_segments, active_windows in SCREENING_CASES + (OPTIONAL_SCREENING_CASES if args.include_largest else []):
        try:
            cases.append(_screening_subspace(config_path, n_segments, active_windows))
        except Exception as exc:
            skipped.append(
                {
                    "case_id": f"feedback_N{n_segments}_k{active_windows}",
                    "evidence_tier": "feedback_screening",
                    "reason": str(exc),
                }
            )
    if not args.include_largest:
        for n_segments, active_windows in OPTIONAL_SCREENING_CASES:
            skipped.append(
                {
                    "case_id": f"feedback_N{n_segments}_k{active_windows}",
                    "evidence_tier": "feedback_screening",
                    "reason": "not requested in bounded default run; pass --include-largest to attempt it",
                }
            )

    raw_frames = []
    summary_frames = []
    state_metric_paths = {}
    for case in cases:
        raw, summary = _run_case(case, seeds)
        raw_frames.append(raw)
        summary_frames.append(summary)
        state_metrics = case["state_metrics"]
        path = results_dir / f"{case['case_id']}_subspace_costs.csv"
        state_metrics.to_csv(path, index=False)
        state_metric_paths[case["case_id"]] = str(path.relative_to(ROOT))
        print(f"completed {case['case_id']} M={len(case['costs'])}", flush=True)

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    raw_all.to_csv(results_dir / "raw_results.csv", index=False)
    summary_all.to_csv(results_dir / "summary.csv", index=False)
    write_json(results_dir / "summary.json", summary_all.to_dict(orient="records"))
    comparisons = paired_comparisons(raw_all)
    comparisons.to_csv(results_dir / "paired_comparisons.csv", index=False)
    write_json(results_dir / "paired_comparisons.json", comparisons.to_dict(orient="records"))
    _write_tables(summary_all, tables_dir)
    _write_statistics_table(comparisons, tables_dir)
    _plot(summary_all, figures_dir)

    metadata = {
        "command": _command_string(),
        "python": sys.version,
        "packages": package_versions(["numpy", "matplotlib", "pandas", "pyyaml"]),
        **revision_metadata(),
        "seeds": {"count": len(seeds), "first": seeds[0] if seeds else None, "last": seeds[-1] if seeds else None},
        "budget_policy": "ceil(sqrt(M)), 2ceil(sqrt(M)), 4ceil(sqrt(M)), and M total oracle calls",
        "query_accounting": {
            "online_quantum": "total_oracle_calls = incumbent/candidate cost evaluations + threshold-oracle calls used by Grover iterations",
            "classical_baselines": "deployable classical baselines receive the same numeric budget as direct finite-subspace schedule cost evaluations; no trajectory regeneration occurs beyond the subspace cost vector construction",
        },
        "baseline_semantics": {
            "deployable": {
                "uniform_random_without_replacement": "uniform finite-subspace sampling without replacement",
                "simulated_annealing_baseline": "seeded finite-subspace simulated annealing using schedule Hamming-swap neighbors when schedules are available and index-neighbor proposals otherwise",
                "cross_entropy_baseline": "seeded finite-subspace cross-entropy method using repaired Bernoulli schedule models when schedules are available and smoothed finite-index categorical models otherwise",
                "genetic_baseline": "seeded finite-subspace genetic search using schedule crossover/mutation/repair when schedules are available and index-level genetic operators otherwise",
            },
            "diagnostic": {
                "surrogate_ranked_baseline": "optimistic ranked screening ceiling; retained in raw/summary but excluded from deployable paired-comparison claims",
            },
        },
        "statistics": {
            "success_interval": "Wilson 95% confidence interval for global-optimum hit rate per case/budget/method",
            "paired_comparisons": "online threshold AA versus each deployable classical baseline at matched case/budget/seed",
            "paired_success_delta": "online success flag minus baseline success flag; positive favors online",
            "best_cost_diff": "online best_cost minus baseline best_cost; negative favors online",
            "best_cost_sign_test": "two-sided exact sign test after excluding exact ties (|diff| <= 1e-12)",
            "bootstrap_ci": {
                "statistic": "mean best_cost_diff_online_minus_baseline",
                "samples": STATISTICS_BOOTSTRAP_SAMPLES,
                "seed": STATISTICS_RNG_SEED,
                "confidence": 0.95,
            },
        },
        "evidence_tiers": {
            "recorded_branch_refined": "uses recorded branch-refined selected-worst costs for M=12 and M=66",
            "feedback_screening": "uses CR3BP feedback propagation objective/terminal errors only; no continuous branch refinement",
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "evidence_tier": case["evidence_tier"],
                "n_segments": case["n_segments"],
                "active_windows": case["active_windows"],
                "M": int(len(case["costs"])),
                "cost_column": case["cost_column"],
                "score_column": case["score_column"],
                "source": case["source"],
            }
            for case in cases
        ],
        "skipped_cases": skipped,
        "runtime_seconds": float(time.perf_counter() - start),
        "artifact_paths": {
            "raw_results": str((results_dir / "raw_results.csv").relative_to(ROOT)),
            "summary_csv": str((results_dir / "summary.csv").relative_to(ROOT)),
            "summary_json": str((results_dir / "summary.json").relative_to(ROOT)),
            "paired_comparisons_csv": str((results_dir / "paired_comparisons.csv").relative_to(ROOT)),
            "paired_comparisons_json": str((results_dir / "paired_comparisons.json").relative_to(ROOT)),
            "table_tex": str((tables_dir / "online_quantum_search_table.tex").relative_to(ROOT)),
            "statistics_table_tex": str((tables_dir / "online_quantum_search_statistics_table.tex").relative_to(ROOT)),
            "figure_pdf": str((figures_dir / "online_quantum_search_success.pdf").relative_to(ROOT)),
            "subspace_costs": state_metric_paths,
        },
        "claim_limits": [
            "The online protocol does not know a top-k set; it uses only current-incumbent threshold comparisons.",
            "Classical deployable baselines are matched-budget finite-subspace searches over the same precomputed constrained cost vectors, not continuous trajectory regeneration.",
            "The surrogate-ranked row is a diagnostic ceiling and is excluded from deployable paired-comparison claims.",
            "The feedback-screening cases are trajectory-backed screening costs, not branch-refined trajectory evidence.",
            "No hardware evidence, quantum advantage, high-fidelity validation, or operational flight-design claim is made.",
        ],
    }
    write_json(results_dir / "metadata.json", metadata)
    print(f"completed online quantum search in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
