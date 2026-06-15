from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .cr3bp import load_source_states
from .objective import Evaluator, ObjectiveConfig, outage_masks, schedule_to_string
from .refinement import refine_schedule
from .reporting import (
    plot_method_comparison,
    plot_qubo_fit,
    plot_recovery_fuel,
    plot_refinement_success,
    plot_trajectory_example,
    summarize,
    write_latex_tables,
    write_metadata,
)
from .solvers import solve_cem, solve_genetic, solve_qaoa, solve_random, solve_surrogate_sa, solve_true_sa
from .surrogate import fit_qubo


QUBO_METHODS = {"surrogate_qubo_sa", "qaoa_statevector"}
SCHEDULE_REPAIR_DESCRIPTIONS = {
    "identity": "original solver candidate schedule",
    "prefix_to_last_active": "heuristic dense-availability envelope: set every window through the last active solver window to available, preserving later zeros",
}


def state_target_mode(states) -> str:
    return str(getattr(states, "target_mode", "catalog_dro_phase"))


def active_target_prior_config(config: dict) -> dict:
    objective_cfg = config.get("objective", {}) or {}
    weights = objective_cfg.get("weights", {}) or {}
    target_active_fraction = objective_cfg.get("target_active_fraction")
    target_active_weight = objective_cfg.get("target_active_weight", weights.get("active_target", 0.0))
    target_active_weight = float(target_active_weight or 0.0)
    if target_active_fraction is not None:
        target_active_fraction = float(target_active_fraction)
    return {
        "target_active_fraction": target_active_fraction,
        "target_active_weight": target_active_weight,
        "active_target_enabled": target_active_fraction is not None and target_active_weight != 0.0,
    }


def make_objective_config(config: dict, mu: float) -> ObjectiveConfig:
    b = config["benchmark"]
    active_target = active_target_prior_config(config)
    return ObjectiveConfig(
        mu=mu,
        tf=float(b["transfer_time"]),
        n_segments=int(b["segments"]),
        substeps=int(b["substeps_per_segment"]),
        amax=float(b["amax"]),
        kr=float(b["steering"]["kr"]),
        kv=float(b["steering"]["kv"]),
        position_scale=float(config["objective"]["position_scale"]),
        velocity_scale=float(config["objective"]["velocity_scale"]),
        weights={k: float(v) for k, v in config["objective"]["weights"].items()},
        outage_lengths=tuple(int(v) for v in config["outages"]["block_lengths"]),
        target_active_fraction=active_target["target_active_fraction"],
        target_active_weight=active_target["target_active_weight"],
    )


def train_qubo(evaluator: Evaluator, rng: np.random.Generator, config: dict, seed: int, out_dir: Path):
    start_count = evaluator.count
    n = evaluator.cfg.n_segments
    n_train = int(config["experiments"]["qubo_train_samples"])
    schedules = (rng.random((n_train, n)) < rng.uniform(0.35, 0.75, size=(n_train, 1))).astype(int)
    # Include edge cases to stabilize active-fraction terms.
    schedules[:2] = np.array([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    objectives = np.asarray([evaluator.evaluate(s)["objective"] for s in schedules], dtype=float)
    model, fit_rows = fit_qubo(
        schedules,
        objectives,
        ridge=1e-4,
        validation_fraction=float(config["experiments"]["qubo_validation_fraction"]),
        seed=seed,
    )
    model.save(out_dir / f"qubo_coefficients_seed{seed}.json")
    rows = []
    for split, t_key, p_key in [
        ("train", "train_true", "train_pred"),
        ("val", "val_true", "val_pred"),
    ]:
        rows.extend({"seed": seed, "split": split, "true": t, "predicted": p} for t, p in zip(fit_rows[t_key], fit_rows[p_key]))
    fit_df = pd.DataFrame(rows)
    fit_path = out_dir / f"qubo_fit_seed{seed}.csv"
    fit_df.to_csv(fit_path, index=False)
    return model, fit_path, evaluator.count - start_count


def clean_generated_outputs(results_dir: Path, figures_dir: Path, tables_dir: Path) -> None:
    patterns = {
        results_dir: [
            "raw_results.csv",
            "summary.csv",
            "summary.json",
            "qubo_diagnostics.csv",
            "run_metadata.json",
            "qubo_coefficients_seed*.json",
            "qubo_fit_seed*.csv",
        ],
        figures_dir: [
            "method_comparison.png",
            "method_comparison.pdf",
            "qubo_fit.png",
            "qubo_fit.pdf",
            "refinement_success.png",
            "refinement_success.pdf",
            "recovery_fuel.png",
            "recovery_fuel.pdf",
            "trajectory_example.png",
            "trajectory_example.pdf",
        ],
        tables_dir: [
            "results_table.tex",
            "ablation_table.tex",
            "recovery_table.tex",
        ],
    }
    for directory, directory_patterns in patterns.items():
        for pattern in directory_patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()


def output_directories(root: Path, config: dict) -> tuple[Path, Path, Path]:
    run_cfg = dict(config.get("run", {}) or {})
    output_subdir = run_cfg.get("output_subdir")
    if output_subdir:
        safe = Path(str(output_subdir))
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError("run.output_subdir must be a relative path without '..'")
        return root / "data" / "results" / safe, root / "figures" / safe, root / "tables" / safe
    return root / "data" / "results", root / "figures", root / "tables"


def load_configured_states(root: Path, config: dict, source_states: Path | None = None):
    b = config["benchmark"]
    source_path = Path(source_states) if source_states is not None else root / "data" / "source_states.json"
    if not source_path.is_absolute():
        source_path = root / source_path
    return load_source_states(
        source_path,
        float(b["transfer_time"]),
        substeps=int(b["substeps_per_segment"]),
        target_mode=str(b.get("target_mode", "catalog_dro_phase")),
        segments=int(b["segments"]),
        amax=float(b["amax"]),
        teacher=b.get("teacher"),
        phase_time=b.get("target_phase_time", b.get("phase_time")),
    )


def run_metadata_extra(config: dict, states) -> dict:
    target_metadata = getattr(states, "target_metadata", {}) or {}
    baselines = config.get("baselines", {}) or {}
    if baselines.get("teacher_seeded", {}).get("enabled", False) or baselines.get("teacher_controls_oracle", {}).get("enabled", False):
        diagnostic_baselines = "teacher_controls_oracle_diagnostic is an oracle-only diagnostic initialized from the hidden teacher continuous controls and is excluded from normal method-comparison plots."
    else:
        diagnostic_baselines = "No teacher-seeded schedule or teacher-controls oracle baseline is enabled for this run."
    return {
        "benchmark_label": config.get("run", {}).get("label"),
        "target_mode": state_target_mode(states),
        "target_state": states.target.tolist(),
        "target_state_generation": target_metadata.get("target_state_generation"),
        "target_metadata": target_metadata,
        "refinement_mode": config["refinement"].get("mode", "single_sequence"),
        "refinement_description": "single_sequence optimizes one open-loop control sequence; branch_recovery optimizes nominal controls plus selected-outage-specific post-outage recovery controls",
        "control_bound": "refined piecewise-constant acceleration vectors are projected to the Euclidean ball ||u_i|| <= amax before propagation and persistence; component-wise optimizer bounds remain only as finite-variable guards",
        "true_evaluation_accounting": "true_evaluations equals total_true_evaluations_including_training; solver_true_evaluations excludes shared QUBO training; shared_qubo_training_evaluations is charged to surrogate_qubo_sa and qaoa_statevector; all_windows_continuous reports optimizer nfev as solver_true_evaluations because it has no discrete schedule-search stage",
        "schedule_repairs": {
            **schedule_repair_config(config),
            "descriptions": SCHEDULE_REPAIR_DESCRIPTIONS,
            "scientific_framing": "Configured repairs are deterministic heuristic dense-availability envelopes applied before continuous refinement; they are not oracle continuous controls and do not by themselves establish quantum advantage.",
        },
        "objective_prior": active_target_prior_config(config),
        "diagnostic_baselines": diagnostic_baselines,
        "claims_limit": "controlled normalized CR3BP initialization benchmark; not flight-ready trajectory optimization",
    }


def solver_configs_for_budget(config: dict, qubo_training_evaluations: int) -> dict:
    solvers = copy.deepcopy(config["solvers"])
    experiments = config["experiments"]
    if not experiments.get("equal_total_budget", False):
        return solvers
    target = int(experiments["solver_true_budget"]) + int(qubo_training_evaluations)
    solvers["random"]["samples"] = target
    solvers["true_sa"]["steps"] = target
    if "cem" in solvers and "batch_size" in solvers["cem"]:
        batch = max(1, int(solvers["cem"]["batch_size"]))
        solvers["cem"]["iterations"] = max(1, int(np.ceil(target / batch)))
    if "genetic" in solvers and {"population", "elite"}.issubset(solvers["genetic"]):
        population = max(1, int(solvers["genetic"]["population"]))
        elite = min(population - 1, max(0, int(solvers["genetic"]["elite"])))
        per_generation = max(1, population - elite)
        solvers["genetic"]["generations"] = max(0, int(np.ceil(max(0, target - population) / per_generation)))
        solvers["genetic"]["max_evaluations"] = target
    return solvers


def schedule_repair_config(config: dict) -> dict:
    repair_cfg = config.get("experiments", {}).get("schedule_repairs")
    if repair_cfg is None:
        repair_cfg = config.get("refinement", {}).get("schedule_repairs")
    if not repair_cfg:
        return {"enabled": False, "variants": ["identity"]}
    if isinstance(repair_cfg, list):
        return {"enabled": True, "variants": repair_cfg}
    if isinstance(repair_cfg, dict):
        enabled = bool(repair_cfg.get("enabled", False))
        variants = repair_cfg.get("variants", repair_cfg.get("methods", ["identity"]))
        if isinstance(variants, str):
            variants = [variants]
        return {"enabled": enabled, "variants": list(variants)}
    raise ValueError("schedule_repairs must be a mapping or list")


def repair_schedule_variant(schedule: np.ndarray, variant: str) -> np.ndarray:
    repaired = np.asarray(schedule, dtype=int).copy()
    if variant == "identity":
        return repaired
    if variant == "prefix_to_last_active":
        active = np.flatnonzero(repaired)
        if active.size:
            repaired[: int(active[-1]) + 1] = 1
        return repaired
    raise ValueError(f"unknown schedule repair variant: {variant}")


def schedule_repair_variants(schedule: np.ndarray, config: dict) -> list[tuple[str, np.ndarray, str]]:
    repair_cfg = schedule_repair_config(config)
    variants = repair_cfg["variants"] if repair_cfg["enabled"] else ["identity"]
    if not variants:
        variants = ["identity"]
    rows: list[tuple[str, np.ndarray, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_variant in variants:
        variant = str(raw_variant)
        repaired = repair_schedule_variant(schedule, variant)
        bits = schedule_to_string(repaired)
        key = (variant, bits)
        if key in seen:
            continue
        seen.add(key)
        rows.append((variant, repaired, SCHEDULE_REPAIR_DESCRIPTIONS.get(variant, variant)))
    return rows


def refinement_row(
    *,
    seed: int,
    method: str,
    schedule: np.ndarray,
    refined_schedule: np.ndarray,
    base: dict,
    solver_true_evaluations: int,
    shared_training: int,
    runtime_seconds: float,
    best_refinement: dict,
    target_mode: str,
    benchmark_label: str | None,
    comparison_group: str = "method_comparison",
    diagnostic_notes: str | None = None,
    schedule_repair: str = "identity",
    refined_schedule_source: str | None = None,
    refinement_candidate_schedule: np.ndarray | None = None,
) -> dict:
    total_true_evaluations = int(solver_true_evaluations) + int(shared_training)
    candidate_schedule = refinement_candidate_schedule if refinement_candidate_schedule is not None else schedule
    row = {
        "seed": seed,
        "benchmark_label": benchmark_label,
        "target_mode": target_mode,
        "method": method,
        "comparison_group": comparison_group,
        "diagnostic_notes": diagnostic_notes,
        "schedule": schedule_to_string(schedule),
        "refined_schedule": schedule_to_string(refined_schedule),
        "refinement_candidate_schedule": schedule_to_string(candidate_schedule),
        "schedule_repair": schedule_repair,
        "refined_schedule_source": refined_schedule_source or SCHEDULE_REPAIR_DESCRIPTIONS.get(schedule_repair, schedule_repair),
        "objective": base["objective"],
        "nominal_error": base["nominal_error"],
        "worst_error": base["worst_error"],
        "robust_degradation": base["robust_degradation"],
        "active_fraction": base["active_fraction"],
        "refinement_candidate_active_fraction": float(np.mean(candidate_schedule)),
        "refined_active_fraction": float(np.mean(refined_schedule)),
        "smoothness": base["smoothness"],
        "solver_true_evaluations": int(solver_true_evaluations),
        "shared_qubo_training_evaluations": int(shared_training),
        "total_true_evaluations_including_training": total_true_evaluations,
        "true_evaluations": total_true_evaluations,
        "runtime_seconds": runtime_seconds,
        "refinement_mode": best_refinement.get("mode"),
        "refinement_success": bool(best_refinement.get("success", False)),
        "refined_nominal_error": best_refinement.get("nominal_error"),
        "refined_worst_error": best_refinement.get("worst_error"),
        "refined_selected_worst_error": best_refinement.get("selected_worst_error", best_refinement.get("worst_error")),
        "refined_all_mask_worst_error": best_refinement.get("all_mask_worst_error", best_refinement.get("worst_error")),
        "refinement_nfev": best_refinement.get("nfev"),
        "refinement_runtime_seconds": best_refinement.get("runtime_seconds"),
        "refined_nominal_fuel": best_refinement.get("nominal_fuel"),
        "refined_recovery_fuel_mean": best_refinement.get("recovery_fuel_mean"),
        "refined_recovery_fuel_max": best_refinement.get("recovery_fuel_max"),
        "best_initial_guess": best_refinement.get("best_initial_guess"),
        "selected_outage_indices": best_refinement.get("selected_outage_indices"),
        "selected_outage_errors": best_refinement.get("selected_outage_errors"),
        "all_outage_errors": best_refinement.get("all_outage_errors"),
        "recovery_indices": best_refinement.get("recovery_indices"),
        "refined_controls": best_refinement.get("controls").tolist() if best_refinement.get("controls") is not None else None,
        "refined_recovery_controls": best_refinement.get("recovery_controls"),
    }
    for key in [
        "active_target_penalty",
        "active_target_deviation",
        "target_active_fraction",
        "target_active_weight",
    ]:
        if key in base:
            row[key] = base[key]
    return row


def _diagnostic_comparison_mask(raw: pd.DataFrame) -> pd.Series:
    if "comparison_group" not in raw.columns:
        return pd.Series(False, index=raw.index)
    return raw["comparison_group"].fillna("").astype(str).str.contains("diagnostic", case=False, regex=False)


def select_trajectory_example_row(raw: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Select the best non-diagnostic trajectory row, falling back only when necessary."""
    if raw.empty:
        raise ValueError("cannot select a trajectory example from empty results")
    sort_cols = ["refined_worst_error", "worst_error"]
    diagnostic = _diagnostic_comparison_mask(raw)
    candidates = raw.loc[~diagnostic]
    used_diagnostic_fallback = bool(candidates.empty)
    if used_diagnostic_fallback:
        candidates = raw
    row = candidates.sort_values(sort_cols).iloc[0]
    metadata = {
        "selection_policy": "best non-diagnostic row by refined_worst_error,worst_error; diagnostic rows are used only when no non-diagnostic rows exist",
        "used_diagnostic_fallback": used_diagnostic_fallback,
        "method": str(row.get("method", "")),
        "comparison_group": None if pd.isna(row.get("comparison_group")) else str(row.get("comparison_group", "")),
    }
    if used_diagnostic_fallback:
        metadata["fallback_reason"] = "no non-diagnostic trajectory rows were available"
    return row, metadata


def _teacher_schedule_from_metadata(states, cfg: ObjectiveConfig) -> np.ndarray:
    teacher = (getattr(states, "target_metadata", {}) or {}).get("teacher")
    if not teacher or "active_schedule" not in teacher:
        raise RuntimeError("teacher baseline requires benchmark.target_mode=teacher_controlled")
    schedule = np.asarray([int(c) for c in teacher["active_schedule"]], dtype=int)
    if schedule.shape != (cfg.n_segments,):
        raise RuntimeError("teacher active schedule length does not match benchmark segments")
    return schedule


def _teacher_controls_from_states(states, cfg: ObjectiveConfig) -> np.ndarray:
    controls = getattr(states, "teacher_controls", None)
    if controls is None:
        teacher = (getattr(states, "target_metadata", {}) or {}).get("teacher", {})
        controls = teacher.get("controls")
    if controls is None:
        raise RuntimeError("teacher_controls_oracle baseline requires stored teacher controls")
    controls = np.asarray(controls, dtype=float)
    expected_shape = (cfg.n_segments, 3)
    if controls.shape != expected_shape:
        raise RuntimeError(f"teacher controls have shape {controls.shape}, expected {expected_shape}")
    return controls


def run_seed(seed: int, states, cfg: ObjectiveConfig, config: dict, out_dir: Path) -> tuple[list[dict], dict, Path]:
    rng = np.random.default_rng(seed)
    evaluator = Evaluator(states.initial, states.target, cfg)
    qubo, fit_path, qubo_training_evaluations = train_qubo(evaluator, rng, config, seed, out_dir)
    solver_cfgs = solver_configs_for_budget(config, qubo_training_evaluations)
    qubo_diag = {
        "seed": seed,
        **qubo.diagnostics,
        "shared_qubo_training_evaluations": qubo_training_evaluations,
        "qubo_training_true_evaluations": qubo_training_evaluations,
        "equal_total_budget": bool(config["experiments"].get("equal_total_budget", False)),
    }
    methods = [
        solve_random(evaluator, rng, cfg.n_segments, int(solver_cfgs["random"]["samples"])),
        solve_cem(evaluator, rng, cfg.n_segments, solver_cfgs["cem"]),
        solve_genetic(evaluator, rng, cfg.n_segments, solver_cfgs["genetic"]),
        solve_true_sa(evaluator, rng, cfg.n_segments, solver_cfgs["true_sa"]),
        solve_surrogate_sa(evaluator, rng, cfg.n_segments, solver_cfgs["surrogate_sa"], qubo),
        solve_qaoa(evaluator, rng, solver_cfgs["qaoa"], qubo),
    ]

    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    rows = []
    thresholds = config["objective"]["thresholds"]
    benchmark_label = config.get("run", {}).get("label")
    for result in methods:
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
        base = result.best_metrics
        shared_training = qubo_training_evaluations if result.method in QUBO_METHODS else 0
        rows.append(
            refinement_row(
                seed=seed,
                method=result.method,
                schedule=result.best_schedule,
                refined_schedule=best_refined_schedule,
                base=base,
                solver_true_evaluations=result.true_evaluations,
                shared_training=shared_training,
                runtime_seconds=result.runtime_seconds,
                best_refinement=best_refinement,
                target_mode=state_target_mode(states),
                benchmark_label=benchmark_label,
                schedule_repair=best_schedule_repair,
                refined_schedule_source=best_refined_schedule_source,
                refinement_candidate_schedule=best_refinement_candidate_schedule,
            )
        )

    if config.get("baselines", {}).get("all_windows_continuous", {}).get("enabled", False):
        schedule = np.ones(cfg.n_segments, dtype=int)
        base = evaluator.evaluate(schedule)
        refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, config["refinement"], thresholds)
        rows.append(
            refinement_row(
                seed=seed,
                method="all_windows_continuous",
                schedule=schedule,
                refined_schedule=schedule,
                base=base,
                solver_true_evaluations=int(refined.get("nfev", 0) or 0),
                shared_training=0,
                runtime_seconds=float(refined.get("runtime_seconds", 0.0) or 0.0),
                best_refinement=refined,
                target_mode=state_target_mode(states),
                benchmark_label=benchmark_label,
            )
        )

    if config.get("baselines", {}).get("teacher_seeded", {}).get("enabled", False):
        schedule = _teacher_schedule_from_metadata(states, cfg)
        base = evaluator.evaluate(schedule)
        refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, config["refinement"], thresholds)
        rows.append(
            refinement_row(
                seed=seed,
                method="teacher_seeded_schedule",
                schedule=schedule,
                refined_schedule=schedule,
                base=base,
                solver_true_evaluations=int(refined.get("nfev", 0) or 0),
                shared_training=0,
                runtime_seconds=float(refined.get("runtime_seconds", 0.0) or 0.0),
                best_refinement=refined,
                target_mode=state_target_mode(states),
                benchmark_label=benchmark_label,
            )
        )
    if config.get("baselines", {}).get("teacher_controls_oracle", {}).get("enabled", False):
        schedule = _teacher_schedule_from_metadata(states, cfg)
        teacher_controls = _teacher_controls_from_states(states, cfg)
        oracle_refine_cfg = copy.deepcopy(config["refinement"])
        oracle_refine_cfg["initial_control_guesses"] = [
            {"name": "teacher_controls_oracle", "controls": teacher_controls}
        ]
        oracle_refine_cfg["use_only_initial_control_guesses"] = True
        base = evaluator.evaluate(schedule)
        refined = refine_schedule(schedule, states.initial, states.target, cfg, masks, oracle_refine_cfg, thresholds)
        rows.append(
            refinement_row(
                seed=seed,
                method="teacher_controls_oracle_diagnostic",
                schedule=schedule,
                refined_schedule=schedule,
                base=base,
                solver_true_evaluations=int(refined.get("nfev", 0) or 0),
                shared_training=0,
                runtime_seconds=float(refined.get("runtime_seconds", 0.0) or 0.0),
                best_refinement=refined,
                target_mode=state_target_mode(states),
                benchmark_label=benchmark_label,
                comparison_group="oracle_diagnostic",
                diagnostic_notes="oracle continuous-control initialization from states.teacher_controls; not a valid competing schedule initializer",
            )
        )
    return rows, qubo_diag, fit_path


def run(config_path: Path) -> None:
    root = Path.cwd()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    for path in (results_dir, figures_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(results_dir, figures_dir, tables_dir)

    states = load_configured_states(root, config)
    cfg = make_objective_config(config, states.mu)

    raw_rows: list[dict] = []
    qubo_rows: list[dict] = []
    fit_paths: list[Path] = []
    for seed in config["experiments"]["seeds"]:
        rows, diag, fit_path = run_seed(int(seed), states, cfg, config, results_dir)
        raw_rows.extend(rows)
        qubo_rows.append(diag)
        fit_paths.append(fit_path)

    raw = pd.DataFrame(raw_rows)
    required_budget_cols = {
        "solver_true_evaluations",
        "shared_qubo_training_evaluations",
        "total_true_evaluations_including_training",
        "true_evaluations",
    }
    missing_budget_cols = required_budget_cols.difference(raw.columns)
    if missing_budget_cols:
        raise RuntimeError(f"missing budget accounting columns: {sorted(missing_budget_cols)}")
    if not raw["true_evaluations"].equals(raw["total_true_evaluations_including_training"]):
        raise RuntimeError("true_evaluations must equal total_true_evaluations_including_training")
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    qubo_diag = pd.DataFrame(qubo_rows)
    qubo_diag.to_csv(results_dir / "qubo_diagnostics.csv", index=False)
    write_latex_tables(summary, qubo_diag, tables_dir)

    best_row, trajectory_selection = select_trajectory_example_row(raw)
    controls = np.asarray(json.loads(best_row["refined_controls"]) if isinstance(best_row["refined_controls"], str) else best_row["refined_controls"], dtype=float)
    schedule_bits = best_row.get("refined_schedule", best_row["schedule"])
    schedule = np.asarray([int(c) for c in schedule_bits], dtype=int)
    plot_trajectory_example(states.initial, states.target, cfg, schedule, controls, figures_dir)
    plot_method_comparison(summary, figures_dir)
    plot_qubo_fit(fit_paths, figures_dir)
    plot_refinement_success(summary, figures_dir)
    plot_recovery_fuel(summary, figures_dir)

    write_metadata(
        results_dir / "run_metadata.json",
        " ".join(sys.argv),
        config,
        {**run_metadata_extra(config, states), "trajectory_example_selection": trajectory_selection},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
