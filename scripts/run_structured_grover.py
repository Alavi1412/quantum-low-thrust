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
from qlt.structured_grover import amplify_subspace, grover_sweep, run_dual_candidate_warm_amplification
from qlt.structured_qaoa import cost_biased_initial_probabilities


DEFAULT_CARDINALITY = Path("data/results/phase_shift_cardinality_ablation/cardinality_refinements.csv")
DEFAULT_GENERIC_QAOA_SUMMARY = Path("data/results/phase_shift_cardinality_30seed/summary.csv")
GROUP_OUTPUTS = {
    "one_coast_k11_full": "one_coast",
    "k10_full": "k10_full",
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _command_string() -> str:
    argv = getattr(sys, "orig_argv", sys.argv)
    return subprocess.list2cmdline([str(arg) for arg in argv])


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


def _normalise_cost(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0:
        raise ValueError("costs must not be constant")
    return (values - low) / span, low, span


def _thresholds(costs: np.ndarray, quantiles: list[float]) -> list[float]:
    costs = np.asarray(costs, dtype=float)
    values = []
    for quantile in quantiles:
        value = float(np.quantile(costs, float(quantile), method="nearest"))
        if value not in values:
            values.append(value)
    return values


def _frontier_schedules(subspace: pd.DataFrame) -> set[str]:
    if "one_coast_pareto_frontier" in subspace.columns and subspace["one_coast_pareto_frontier"].astype(bool).any():
        return set(subspace.loc[subspace["one_coast_pareto_frontier"].astype(bool), "schedule"].astype(str))
    costs = subspace[["refined_selected_worst_error", "refined_nominal_error"]].to_numpy(dtype=float)
    schedules = subspace["schedule"].astype(str).to_list()
    frontier: set[str] = set()
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


def _summarize(
    *,
    method: str,
    oracle: str,
    iterations: int,
    probabilities: np.ndarray,
    initial_probabilities: np.ndarray,
    selected_worst: np.ndarray,
    nominal_error: np.ndarray,
    schedules: list[str],
    frontier_schedules: set[str],
    initial_state: str,
    good_probability: float,
    analysis_stage: str = "exploratory_sweep",
) -> dict:
    best_index = int(np.argmin(selected_worst))
    order = np.argsort(-probabilities, kind="mergesort")
    top_index = int(order[0])
    frontier_probability = float(sum(probabilities[index] for index, schedule in enumerate(schedules) if schedule in frontier_schedules))
    return {
        "method": method,
        "analysis_stage": analysis_stage,
        "oracle": oracle,
        "iterations": int(iterations),
        "initial_state": initial_state,
        "states_in_support": int(probabilities.size),
        "feasible_support_probability": 1.0,
        "expected_selected_worst_error": float(probabilities @ selected_worst),
        "initial_expected_selected_worst_error": float(initial_probabilities @ selected_worst),
        "expected_nominal_error": float(probabilities @ nominal_error),
        "probability_best_state": float(probabilities[best_index]),
        "initial_probability_best_state": float(initial_probabilities[best_index]),
        "probability_good_set": float(good_probability),
        "probability_pareto_frontier": frontier_probability,
        "top_schedule": schedules[top_index],
        "top_schedule_probability": float(probabilities[top_index]),
        "top_schedule_selected_worst_error": float(selected_worst[top_index]),
        "best_schedule": schedules[best_index],
        "best_selected_worst_error": float(selected_worst[best_index]),
        "top3_schedules": "; ".join(f"{schedules[int(index)]}:{probabilities[int(index)]:.3f}" for index in order[:3]),
    }


def _distribution_rows(
    *,
    method: str,
    oracle: str,
    iterations: int,
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
                "oracle": oracle,
                "iterations": int(iterations),
                "state_index": int(subspace.loc[index, "state_index"]),
                "state_label": str(subspace.loc[index, "state_label"]),
                "coast_windows": str(subspace.loc[index, "coast_windows_zero_based"]),
                "schedule": schedules[index],
                "probability": float(probability),
                "refined_selected_worst_error": float(selected_worst[index]),
                "refined_nominal_error": float(nominal_error[index]),
                "is_best_state": bool(index == best_index),
                "is_pareto_frontier": schedules[index] in frontier_schedules,
            }
        )
    return rows


def _load_generic_qaoa_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, dtype={"method": str})
    return data[data["method"] == "qaoa_statevector"].copy()


def _load_structured_qaoa_best(group_slug: str) -> dict | None:
    path = ROOT / "data" / "results" / f"structured_qaoa_{group_slug}" / "summary.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    structured = data[data["method"].astype(str).str.startswith("structured_") | data["method"].astype(str).str.startswith("warm_start_xy_qaoa_p")]
    if structured.empty:
        return None
    row = structured.sort_values("expected_selected_worst_error", kind="mergesort").iloc[0]
    return {
        "method": str(row["method"]),
        "expected_selected_worst_error": float(row["expected_selected_worst_error"]),
        "probability_best_state": float(row.get("probability_best_state", row.get("probability_best_one_coast", np.nan))),
    }


def _write_tables(summary: pd.DataFrame, comparison: pd.DataFrame, audit: pd.DataFrame, query: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "analysis_stage",
        "oracle",
        "iterations",
        "expected_selected_worst_error",
        "probability_best_state",
        "probability_good_set",
        "top_schedule",
    ]
    sorted_summary = summary.sort_values(["expected_selected_worst_error", "method"], kind="mergesort")
    predeclared = sorted_summary[sorted_summary["analysis_stage"] == "predeclared_protocol"]
    exploratory = sorted_summary[sorted_summary["analysis_stage"] != "predeclared_protocol"]
    best = pd.concat([predeclared, exploratory.head(max(0, 12 - len(predeclared)))], ignore_index=True)
    best[columns].to_latex(
        tables_dir / "structured_grover_table.tex",
        index=False,
        float_format="%.4f",
        escape=True,
    )
    if not comparison.empty:
        comparison.to_latex(
            tables_dir / "structured_grover_comparison_table.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )
    if not audit.empty:
        audit_columns = [
            "row_type",
            "method",
            "expected_selected_worst_error",
            "delta_vs_predeclared",
            "probability_best_state",
            "predeclared_equals_best_warm_sweep",
        ]
        audit[audit_columns].to_latex(
            tables_dir / "structured_grover_selection_bias_audit_table.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )
    if not query.empty:
        query_columns = [
            "protocol",
            "M",
            "marked_count",
            "initial_good_probability",
            "selected_iterations",
            "oracle_calls_per_sample",
            "probability_best_state",
            "expected_selected_worst_error",
            "classical_exhaustive_M",
            "warm_start_random_samples_to_match_p_best",
        ]
        query[query_columns].to_latex(
            tables_dir / "structured_grover_query_scaling_table.tex",
            index=False,
            float_format="%.4f",
            escape=True,
        )


def _plot_sweep(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    grover = summary[summary["method"].astype(str).str.startswith("grover_warm_")].copy()
    if grover.empty:
        return
    best_oracles = (
        grover.groupby("oracle", sort=False)["expected_selected_worst_error"]
        .min()
        .sort_values(kind="mergesort")
        .head(5)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for oracle in best_oracles:
        rows = grover[grover["oracle"] == oracle].sort_values("iterations", kind="mergesort")
        ax.plot(
            rows["iterations"],
            rows["expected_selected_worst_error"],
            marker="o",
            linewidth=1.4,
            label=oracle,
        )
    uniform = summary[summary["method"] == "uniform_feasible"]
    warm = summary[summary["method"] == "warm_start_only"]
    if not uniform.empty:
        ax.axhline(float(uniform.iloc[0]["expected_selected_worst_error"]), color="0.35", linestyle="--", linewidth=1.0, label="uniform feasible")
    if not warm.empty:
        ax.axhline(float(warm.iloc[0]["expected_selected_worst_error"]), color="0.15", linestyle=":", linewidth=1.2, label="warm start")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Expected selected-worst error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "structured_grover_sweep.png", dpi=220)
    fig.savefig(figures_dir / "structured_grover_sweep.pdf")
    plt.close(fig)


def _comparison_rows(summary: pd.DataFrame, group_slug: str, generic: pd.DataFrame) -> pd.DataFrame:
    warm_rows = summary[summary["method"].astype(str).str.startswith("grover_warm_")].copy()
    rows: list[dict] = []
    if warm_rows.empty:
        return pd.DataFrame()
    best = warm_rows.sort_values("expected_selected_worst_error", kind="mergesort").iloc[0]
    uniform = summary[summary["method"] == "uniform_feasible"].iloc[0]
    warm = summary[summary["method"] == "warm_start_only"].iloc[0]
    matched_name = str(best["method"]).replace("grover_warm_", "grover_uniform_")
    matched = summary[summary["method"] == matched_name]
    matched_row = matched.iloc[0] if not matched.empty else uniform
    rows.append(
        {
            "comparison": "best_grover_warm_minus_uniform_feasible",
            "grover_method": str(best["method"]),
            "grover_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
            "baseline_method": "uniform_feasible",
            "baseline_expected_selected_worst_error": float(uniform["expected_selected_worst_error"]),
            "delta_selected_worst_error": float(best["expected_selected_worst_error"]) - float(uniform["expected_selected_worst_error"]),
            "grover_probability_best_state": float(best["probability_best_state"]),
            "baseline_probability_best_state": float(uniform["probability_best_state"]),
        }
    )
    rows.append(
        {
            "comparison": "best_grover_warm_minus_warm_start_only",
            "grover_method": str(best["method"]),
            "grover_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
            "baseline_method": "warm_start_only",
            "baseline_expected_selected_worst_error": float(warm["expected_selected_worst_error"]),
            "delta_selected_worst_error": float(best["expected_selected_worst_error"]) - float(warm["expected_selected_worst_error"]),
            "grover_probability_best_state": float(best["probability_best_state"]),
            "baseline_probability_best_state": float(warm["probability_best_state"]),
        }
    )
    rows.append(
        {
            "comparison": "best_grover_warm_minus_matched_uniform_grover",
            "grover_method": str(best["method"]),
            "grover_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
            "baseline_method": str(matched_row["method"]),
            "baseline_expected_selected_worst_error": float(matched_row["expected_selected_worst_error"]),
            "delta_selected_worst_error": float(best["expected_selected_worst_error"]) - float(matched_row["expected_selected_worst_error"]),
            "grover_probability_best_state": float(best["probability_best_state"]),
            "baseline_probability_best_state": float(matched_row["probability_best_state"]),
        }
    )
    qaoa = _load_structured_qaoa_best(group_slug)
    if qaoa is not None:
        rows.append(
            {
                "comparison": "best_grover_warm_minus_best_structured_qaoa",
                "grover_method": str(best["method"]),
                "grover_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
                "baseline_method": qaoa["method"],
                "baseline_expected_selected_worst_error": qaoa["expected_selected_worst_error"],
                "delta_selected_worst_error": float(best["expected_selected_worst_error"]) - qaoa["expected_selected_worst_error"],
                "grover_probability_best_state": float(best["probability_best_state"]),
                "baseline_probability_best_state": qaoa["probability_best_state"],
            }
        )
    if not generic.empty and "refined_selected_worst_error_median" in generic.columns:
        generic_row = generic.iloc[0]
        rows.append(
            {
                "comparison": "best_grover_warm_minus_generic_qaoa_median",
                "grover_method": str(best["method"]),
                "grover_expected_selected_worst_error": float(best["expected_selected_worst_error"]),
                "baseline_method": "qaoa_statevector_30seed_median",
                "baseline_expected_selected_worst_error": float(generic_row["refined_selected_worst_error_median"]),
                "delta_selected_worst_error": float(best["expected_selected_worst_error"]) - float(generic_row["refined_selected_worst_error_median"]),
                "grover_probability_best_state": float(best["probability_best_state"]),
                "baseline_probability_best_state": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _best_structured_qaoa_summary(group_slug: str) -> dict:
    qaoa = _load_structured_qaoa_best(group_slug)
    if qaoa is None:
        return {
            "method": "best_structured_qaoa_unavailable",
            "expected_selected_worst_error": np.nan,
            "probability_best_state": np.nan,
        }
    return qaoa


def _generic_qaoa_median_summary(generic: pd.DataFrame) -> dict:
    if generic.empty or "refined_selected_worst_error_median" not in generic.columns:
        return {
            "method": "qaoa_statevector_30seed_median_unavailable",
            "expected_selected_worst_error": np.nan,
            "probability_best_state": np.nan,
        }
    row = generic.iloc[0]
    return {
        "method": "qaoa_statevector_30seed_median_descriptive_only",
        "expected_selected_worst_error": float(row["refined_selected_worst_error_median"]),
        "probability_best_state": np.nan,
    }


def _selection_bias_audit(summary: pd.DataFrame, group_slug: str, generic: pd.DataFrame) -> pd.DataFrame:
    predeclared = summary[summary["analysis_stage"] == "predeclared_protocol"].iloc[0]
    warm_rows = summary[summary["method"].astype(str).str.startswith("grover_warm_")].copy()
    best_warm = warm_rows.sort_values("expected_selected_worst_error", kind="mergesort").iloc[0]
    uniform = summary[summary["method"] == "uniform_feasible"].iloc[0]
    warm = summary[summary["method"] == "warm_start_only"].iloc[0]
    matched_name = f"grover_uniform_{predeclared['oracle']}_r{int(predeclared['iterations'])}".replace("<=", "le").replace(".", "p")
    matched = summary[summary["method"] == matched_name]
    matched_row = matched.iloc[0] if not matched.empty else uniform
    qaoa = _best_structured_qaoa_summary(group_slug)
    generic_row = _generic_qaoa_median_summary(generic)
    pre_error = float(predeclared["expected_selected_worst_error"])
    same_as_best = bool(
        str(predeclared["method"]) == str(best_warm["method"])
        or (
            int(predeclared["iterations"]) == int(best_warm["iterations"])
            and abs(pre_error - float(best_warm["expected_selected_worst_error"])) <= 1e-14
            and abs(float(predeclared["probability_best_state"]) - float(best_warm["probability_best_state"])) <= 1e-14
        )
    )
    rows = [
        ("predeclared_protocol", predeclared),
        ("best_warm_start_sweep_row", best_warm),
        ("uniform_feasible", uniform),
        ("warm_start_only", warm),
        ("matched_uniform_grover_same_oracle_r", matched_row),
        ("best_structured_qaoa", qaoa),
        ("generic_qaoa_median_descriptive_only", generic_row),
    ]
    audit_rows = []
    for row_type, row in rows:
        expected = float(row["expected_selected_worst_error"])
        audit_rows.append(
            {
                "row_type": row_type,
                "method": str(row["method"]),
                "oracle": str(row.get("oracle", "not_applicable")),
                "iterations": int(row.get("iterations", 0)) if pd.notna(row.get("iterations", 0)) else 0,
                "expected_selected_worst_error": expected,
                "delta_vs_predeclared": expected - pre_error if np.isfinite(expected) else np.nan,
                "probability_best_state": float(row.get("probability_best_state", np.nan)),
                "predeclared_equals_best_warm_sweep": same_as_best,
                "comparison_scope": "like_for_like" if row_type != "generic_qaoa_median_descriptive_only" else "descriptive_not_like_for_like",
            }
        )
    return pd.DataFrame(audit_rows)


def _warm_samples_to_match(target_probability: float, single_sample_probability: float) -> float:
    target = float(target_probability)
    p = float(single_sample_probability)
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return float("inf")
    if p <= 0.0:
        return float("inf")
    if p >= target:
        return 1.0
    return float(np.ceil(np.log1p(-target) / np.log1p(-p)))


def _query_scaling_row(
    *,
    protocol_name: str,
    selected_worst: np.ndarray,
    warm_start: np.ndarray,
    protocol,
    predeclared_row: pd.Series,
) -> pd.DataFrame:
    best_index = int(np.argmin(selected_worst))
    probability_best = float(predeclared_row["probability_best_state"])
    return pd.DataFrame(
        [
            {
                "protocol": protocol_name,
                "M": int(selected_worst.size),
                "marked_count": int(protocol.result.good_mask.sum()),
                "initial_good_probability": float(protocol.initial_good_probability),
                "selected_iterations": int(protocol.selected_iterations),
                "oracle_calls_per_sample": int(protocol.selected_iterations),
                "state_preparation_count": int(2 * protocol.selected_iterations + 1),
                "exact_dense_statevector_dimension": int(selected_worst.size),
                "probability_best_state": probability_best,
                "expected_selected_worst_error": float(predeclared_row["expected_selected_worst_error"]),
                "classical_exhaustive_M": int(selected_worst.size),
                "warm_start_random_samples_to_match_p_best": _warm_samples_to_match(probability_best, float(warm_start[best_index])),
                "amplitude_amplification_scaling_note": "O(sqrt(M/g)) query scaling for known marked set; this row is exact simulation",
                "minimum_finding_scaling_note": "O(sqrt(M)) theoretical minimum-finding scaling; not implemented here",
            }
        ]
    )


def run_group(args: argparse.Namespace, group: str) -> dict:
    start = time.perf_counter()
    group_slug = GROUP_OUTPUTS.get(group, group.replace("_k11_full", "").replace("_", "-"))
    results_dir = ROOT / "data" / "results" / f"structured_grover_{group_slug}"
    figures_dir = ROOT / "figures" / f"structured_grover_{group_slug}"
    tables_dir = ROOT / "tables" / f"structured_grover_{group_slug}"
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    subspace = _load_cardinality_group(_resolve(args.cardinality), group)
    selected_worst = subspace["refined_selected_worst_error"].to_numpy(dtype=float)
    nominal_error = subspace["refined_nominal_error"].to_numpy(dtype=float)
    normalized_costs, cost_offset, cost_scale = _normalise_cost(selected_worst)
    schedules = subspace["schedule"].astype(str).to_list()
    frontier_schedules = _frontier_schedules(subspace)
    top_ks = [int(value) for value in args.top_ks if int(value) <= len(subspace)]
    thresholds = _thresholds(selected_worst, [float(value) for value in args.threshold_quantiles])
    iterations = [int(value) for value in args.iterations]

    uniform = np.full(len(subspace), 1.0 / len(subspace), dtype=float)
    warm_start = cost_biased_initial_probabilities(
        normalized_costs,
        mode=args.warm_start_mode,
        epsilon=float(args.warm_start_epsilon),
        beta=float(args.warm_start_beta),
    )

    summary_rows: list[dict] = []
    distribution_rows: list[dict] = []
    for method, probabilities, initial_state in [
        ("uniform_feasible", uniform, "uniform"),
        ("warm_start_only", warm_start, args.warm_start_mode),
    ]:
        row = _summarize(
            method=method,
            oracle="none",
            iterations=0,
            probabilities=probabilities,
            initial_probabilities=probabilities,
            selected_worst=selected_worst,
            nominal_error=nominal_error,
            schedules=schedules,
            frontier_schedules=frontier_schedules,
            initial_state=initial_state,
            good_probability=float("nan"),
        )
        summary_rows.append(row)
        distribution_rows.extend(
            _distribution_rows(
                method=method,
                oracle="none",
                iterations=0,
                probabilities=probabilities,
                selected_worst=selected_worst,
                nominal_error=nominal_error,
                subspace=subspace,
                schedules=schedules,
                frontier_schedules=frontier_schedules,
            )
        )

    protocol = run_dual_candidate_warm_amplification(
        selected_worst,
        warm_start,
        iteration_cap=int(args.protocol_iteration_cap),
    )
    protocol_method = f"predeclared_protocol_{protocol.name}"
    protocol_row = _summarize(
        method=protocol_method,
        oracle=protocol.result.oracle,
        iterations=protocol.result.iterations,
        probabilities=protocol.result.probabilities,
        initial_probabilities=protocol.result.initial_probabilities,
        selected_worst=selected_worst,
        nominal_error=nominal_error,
        schedules=schedules,
        frontier_schedules=frontier_schedules,
        initial_state=args.warm_start_mode,
        good_probability=protocol.result.good_probability,
        analysis_stage="predeclared_protocol",
    )
    protocol_row.update(
        {
            "predeclared_protocol_name": protocol.name,
            "predeclared_good_rule": protocol.good_rule,
            "predeclared_iteration_rule": protocol.iteration_rule,
            "predeclared_iteration_cap": int(protocol.iteration_cap),
            "initial_good_probability_predeclared": float(protocol.initial_good_probability),
            "predicted_good_probability_predeclared": float(protocol.predicted_good_probability),
        }
    )
    summary_rows.append(protocol_row)
    distribution_rows.extend(
        _distribution_rows(
            method=protocol_method,
            oracle=protocol.result.oracle,
            iterations=protocol.result.iterations,
            probabilities=protocol.result.probabilities,
            selected_worst=selected_worst,
            nominal_error=nominal_error,
            subspace=subspace,
            schedules=schedules,
            frontier_schedules=frontier_schedules,
        )
    )

    for prefix, initial_probabilities, initial_state in [
        ("grover_warm", warm_start, args.warm_start_mode),
        ("grover_uniform", uniform, "uniform"),
    ]:
        for result in grover_sweep(
            selected_worst,
            initial_probabilities,
            thresholds=thresholds,
            top_ks=top_ks,
            iterations=iterations,
        ):
            method = f"{prefix}_{result.oracle}_r{result.iterations}".replace("<=", "le").replace(".", "p")
            row = _summarize(
                method=method,
                oracle=result.oracle,
                iterations=result.iterations,
                probabilities=result.probabilities,
                initial_probabilities=result.initial_probabilities,
                selected_worst=selected_worst,
                nominal_error=nominal_error,
                schedules=schedules,
                frontier_schedules=frontier_schedules,
                initial_state=initial_state,
                good_probability=result.good_probability,
            )
            summary_rows.append(row)
            distribution_rows.extend(
                _distribution_rows(
                    method=method,
                    oracle=result.oracle,
                    iterations=result.iterations,
                    probabilities=result.probabilities,
                    selected_worst=selected_worst,
                    nominal_error=nominal_error,
                    subspace=subspace,
                    schedules=schedules,
                    frontier_schedules=frontier_schedules,
                )
            )

    summary = pd.DataFrame(summary_rows).sort_values(["expected_selected_worst_error", "method"], kind="mergesort").reset_index(drop=True)
    distribution = pd.DataFrame(distribution_rows)
    generic = _load_generic_qaoa_summary(_resolve(args.generic_qaoa_summary))
    comparison = _comparison_rows(summary, group_slug, generic)
    audit = _selection_bias_audit(summary, group_slug, generic)
    predeclared_summary_row = summary[summary["analysis_stage"] == "predeclared_protocol"].iloc[0]
    query = _query_scaling_row(
        protocol_name=protocol.name,
        selected_worst=selected_worst,
        warm_start=warm_start,
        protocol=protocol,
        predeclared_row=predeclared_summary_row,
    )

    summary.to_csv(results_dir / "summary.csv", index=False)
    summary.to_json(results_dir / "summary.json", orient="records", indent=2)
    distribution.to_csv(results_dir / "distribution.csv", index=False)
    distribution.to_json(results_dir / "distribution.json", orient="records", indent=2)
    comparison.to_csv(results_dir / "comparisons.csv", index=False)
    comparison.to_json(results_dir / "comparisons.json", orient="records", indent=2)
    audit.to_csv(results_dir / "selection_bias_audit.csv", index=False)
    audit.to_json(results_dir / "selection_bias_audit.json", orient="records", indent=2)
    query.to_csv(results_dir / "query_scaling.csv", index=False)
    query.to_json(results_dir / "query_scaling.json", orient="records", indent=2)
    generic.to_csv(results_dir / "generic_qaoa_summary.csv", index=False)
    _write_tables(summary, comparison, audit, query, tables_dir)
    _plot_sweep(summary, figures_dir)

    best = summary[summary["method"].astype(str).str.startswith("grover_warm_")].iloc[0]
    metadata = {
        "command": _command_string(),
        "python": f"py -{sys.version_info.major}.{sys.version_info.minor}",
        "packages": package_versions(["numpy", "matplotlib", "pandas"]),
        **revision_metadata(),
        "cardinality_source": str(_resolve(args.cardinality).relative_to(ROOT)),
        "cardinality_group": group,
        "states_in_subspace": int(len(subspace)),
        "cost_column": "refined_selected_worst_error",
        "cost_normalization": {
            "offset": cost_offset,
            "scale": cost_scale,
            "formula": "(selected_worst_error - offset) / scale",
        },
        "warm_start": {
            "mode": args.warm_start_mode,
            "epsilon": float(args.warm_start_epsilon),
            "beta": float(args.warm_start_beta),
            "probability_best_state": float(warm_start[int(np.argmin(selected_worst))]),
            "expected_selected_worst_error": float(warm_start @ selected_worst),
        },
        "oracle_sweep": {
            "analysis_stage": "exploratory_sweep",
            "threshold_quantiles": [float(value) for value in args.threshold_quantiles],
            "threshold_values": thresholds,
            "top_ks": top_ks,
            "iterations": iterations,
        },
        "predeclared_protocol": {
            "analysis_stage": "predeclared_protocol",
            "name": protocol.name,
            "good_rule": protocol.good_rule,
            "iteration_rule": protocol.iteration_rule,
            "iteration_cap": int(protocol.iteration_cap),
            "oracle": protocol.result.oracle,
            "marked_count": int(protocol.result.good_mask.sum()),
            "initial_good_probability": float(protocol.initial_good_probability),
            "selected_iterations": int(protocol.selected_iterations),
            "predicted_good_probability": float(protocol.predicted_good_probability),
            "simulated_good_probability": float(protocol.result.good_probability),
            "expected_selected_worst_error": float(predeclared_summary_row["expected_selected_worst_error"]),
            "probability_best_state": float(predeclared_summary_row["probability_best_state"]),
            "best_schedule": str(predeclared_summary_row["best_schedule"]),
        },
        "strongest_configuration": {
            "analysis_stage": "exploratory_sweep",
            "method": str(best["method"]),
            "oracle": str(best["oracle"]),
            "iterations": int(best["iterations"]),
            "expected_selected_worst_error": float(best["expected_selected_worst_error"]),
            "probability_best_state": float(best["probability_best_state"]),
            "probability_good_set": float(best["probability_good_set"]),
            "top_schedule": str(best["top_schedule"]),
            "best_schedule": str(best["best_schedule"]),
        },
        "comparisons": comparison.to_dict(orient="records"),
        "selection_bias_audit": audit.to_dict(orient="records"),
        "query_scaling": query.to_dict(orient="records")[0],
        "runtime_seconds": float(time.perf_counter() - start),
        "interpretation_limits": [
            "This is an exact feasible-subspace statevector simulation, not hardware evidence.",
            "The phase oracle marks states using already recorded branch-refined selected-worst errors.",
            "The matched uniform comparator uses the same feasible subspace, oracle, and Grover iteration count.",
            "The sweep is finite and descriptive; it does not claim asymptotic advantage or universal dominance.",
        ],
        "artifact_paths": {
            "summary_csv": str((results_dir / "summary.csv").relative_to(ROOT)),
            "summary_json": str((results_dir / "summary.json").relative_to(ROOT)),
            "distribution_csv": str((results_dir / "distribution.csv").relative_to(ROOT)),
            "distribution_json": str((results_dir / "distribution.json").relative_to(ROOT)),
            "comparisons_csv": str((results_dir / "comparisons.csv").relative_to(ROOT)),
            "comparisons_json": str((results_dir / "comparisons.json").relative_to(ROOT)),
            "selection_bias_audit_csv": str((results_dir / "selection_bias_audit.csv").relative_to(ROOT)),
            "selection_bias_audit_json": str((results_dir / "selection_bias_audit.json").relative_to(ROOT)),
            "query_scaling_csv": str((results_dir / "query_scaling.csv").relative_to(ROOT)),
            "query_scaling_json": str((results_dir / "query_scaling.json").relative_to(ROOT)),
            "figure_pdf": str((figures_dir / "structured_grover_sweep.pdf").relative_to(ROOT)),
            "figure_png": str((figures_dir / "structured_grover_sweep.png").relative_to(ROOT)),
            "table_tex": str((tables_dir / "structured_grover_table.tex").relative_to(ROOT)),
            "comparison_table_tex": str((tables_dir / "structured_grover_comparison_table.tex").relative_to(ROOT)),
            "selection_bias_audit_table_tex": str((tables_dir / "structured_grover_selection_bias_audit_table.tex").relative_to(ROOT)),
            "query_scaling_table_tex": str((tables_dir / "structured_grover_query_scaling_table.tex").relative_to(ROOT)),
        },
    }
    write_json(results_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact feasible-subspace Grover amplification benchmarks.")
    parser.add_argument("--cardinality", type=Path, default=DEFAULT_CARDINALITY)
    parser.add_argument("--generic-qaoa-summary", type=Path, default=DEFAULT_GENERIC_QAOA_SUMMARY)
    parser.add_argument("--groups", type=str, nargs="+", default=["one_coast_k11_full", "k10_full"])
    parser.add_argument("--iterations", type=int, nargs="+", default=list(range(0, 13)))
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--threshold-quantiles", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.50])
    parser.add_argument("--warm-start-mode", choices=["inverse", "exponential"], default="inverse")
    parser.add_argument("--warm-start-epsilon", type=float, default=0.02)
    parser.add_argument("--warm-start-beta", type=float, default=3.0)
    parser.add_argument("--protocol-iteration-cap", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for group in args.groups:
        metadata = run_group(args, group)
        best = metadata["strongest_configuration"]
        print(
            f"completed {group}: best {best['method']} expected selected-worst "
            f"{best['expected_selected_worst_error']:.6f}, p(best)={best['probability_best_state']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
