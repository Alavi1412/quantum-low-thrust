from __future__ import annotations

import argparse
import copy
import json
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
            "qubo_coefficients_seed*.json",
            "qubo_fit_seed*.csv",
        ],
        figures_dir: [
            "qaoa_depth_ablation_summary.png",
            "qaoa_depth_ablation_summary.pdf",
        ],
        tables_dir: [
            "qaoa_depth_ablation_table.tex",
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


def plot_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["method"].astype(str).to_list()
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    axes[0].bar(x, summary["refinement_success_rate"])
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
    if not args.no_figure:
        plot_summary(summary, figures_dir)

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
        "runtime_seconds": float(time.perf_counter() - start),
        "interpretation_limits": [
            "The optimized-angle QAOA variants optimize expected surrogate QUBO energy only.",
            "All QUBO-derived methods receive the same 96 shared QUBO training evaluations and 24 post-QUBO true candidate evaluations per seed.",
            "This ablation does not claim quantum advantage.",
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
