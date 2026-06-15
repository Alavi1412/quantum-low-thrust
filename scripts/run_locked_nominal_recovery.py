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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.direct_collocation import config_hash, file_identity, settings_fingerprint
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.locked_recovery import (
    BranchRecoveryWeights,
    normalize_selected_outage_policy,
    run_locked_nominal_recovery_baseline,
    selected_outage_count_for_policy,
)
from qlt.objective import outage_masks
from qlt.reporting import write_metadata


LOCKED_RECOVERY_COLUMNS = [
    "suite_case_id",
    "purpose",
    "target_mode",
    "target_generation",
    "transfer_time",
    "amax",
    "segments",
    "substeps_per_segment",
    "outage_lengths",
    "outage_count",
    "selected_outage_policy",
    "selected_outages",
    "selected_outage_count",
    "nominal_max_nfev",
    "branch_max_nfev",
    "min_recovery_segments",
    "node_initialization",
    "node_initialization_blend",
    "terminal_weight",
    "control_weight",
    "smooth_weight",
    "continuity_weight",
    "xtol",
    "ftol",
    "gtol",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "mode",
    "method_type",
    "nominal_error",
    "nominal_baseline_error",
    "nominal_lock_error_delta",
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
    "backend_success",
    "optimizer_success",
    "nominal_optimizer_success",
    "nominal_backend_success",
    "nominal_fuel",
    "recovery_fuel_mean",
    "recovery_fuel_max",
    "control_max_norm",
    "control_bound_violation",
    "nominal_nfev",
    "total_branch_nfev",
    "nfev",
    "nominal_runtime_seconds",
    "total_branch_runtime_seconds",
    "runtime_seconds",
    "cost",
    "optimality",
    "nominal_cost",
    "nominal_optimality",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "nominal_masked_outage_errors",
    "branch_nfev",
    "branch_runtime_seconds",
    "branch_optimizer_success",
    "branch_recovery_starts",
    "branch_recovery_segments",
    "branch_results",
    "nominal_accepted_candidate",
    "backend_semantics",
    "selection_semantics",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "nominal_lock_semantics",
    "worst_error_semantics",
    "message",
    "nominal_message",
]


def _json_list(value) -> str:
    return json.dumps(value, sort_keys=False)


def _as_list(value, cast) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [cast(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _outage_count(segments: int, lengths: list[int]) -> int:
    return int(sum(max(0, int(segments) - int(length) + 1) for length in lengths))


def _locked_config(config: dict) -> dict:
    return dict(config.get("locked_recovery", {}) or {})


def _nominal_settings(config: dict) -> dict:
    return dict(_locked_config(config).get("nominal", {}) or {})


def _branch_settings(config: dict) -> dict:
    return dict(_locked_config(config).get("branch", {}) or {})


def _nominal_residual_weights(config: dict, case: dict | None = None) -> dict:
    nominal = _nominal_settings(config)
    raw = dict(nominal.get("residual_weights", {}) or {})
    if case is not None:
        raw.update(dict(case.get("nominal_residual_weights", {}) or {}))
    return {str(key): float(value) for key, value in raw.items()}


def _branch_weights(config: dict, case: dict | None = None) -> BranchRecoveryWeights:
    raw = _branch_settings(config)
    if case is not None:
        merged = copy.deepcopy(raw)
        merged.update(copy.deepcopy(case.get("branch", {}) or {}))
        for key in ("terminal_weight", "control_weight", "smooth_weight", "continuity_weight"):
            if key in case:
                merged[key] = case[key]
        raw = merged
    return BranchRecoveryWeights.from_config(raw)


def _tolerances(config: dict, case: dict | None = None) -> dict[str, float]:
    branch = _branch_settings(config)
    out = {
        "xtol": float(branch.get("xtol", 1e-5)),
        "ftol": float(branch.get("ftol", 1e-5)),
        "gtol": float(branch.get("gtol", 1e-5)),
    }
    if case is not None:
        for key in out:
            if key in case:
                out[key] = float(case[key])
            elif key in (case.get("branch", {}) or {}):
                out[key] = float(case["branch"][key])
    return out


def _node_initialization(config: dict, case: dict | None = None) -> tuple[str, float]:
    nominal = _nominal_settings(config)
    mode = str(nominal.get("node_initialization", "linear"))
    blend = float(nominal.get("node_initialization_blend", 0.5))
    if case is not None:
        mode = str(case.get("node_initialization", mode))
        blend = float(case.get("node_initialization_blend", blend))
    return mode, blend


def _case_config(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    config.setdefault("outages", {})["block_lengths"] = [int(value) for value in case["outage_lengths"]]
    return config


def _suite_cases(config: dict) -> list[dict]:
    locked = _locked_config(config)
    raw_cases = list(locked.get("cases", (config.get("suite", {}) or {}).get("cases", [])) or [])
    benchmark = config.get("benchmark", {}) or {}
    default_lengths = [int(value) for value in config.get("outages", {}).get("block_lengths", [1])]
    nominal = _nominal_settings(config)
    branch = _branch_settings(config)

    cases: list[dict] = []
    for index, raw in enumerate(raw_cases):
        if not bool(raw.get("enabled", True)):
            continue
        case = dict(raw)
        lengths = [int(value) for value in case.get("outage_lengths", case.get("block_lengths", default_lengths))]
        segments = int(case.get("segments", benchmark.get("segments")))
        masks = outage_masks(segments, tuple(lengths))
        policy = normalize_selected_outage_policy(case.get("selected_outages", 0))
        selected_count = selected_outage_count_for_policy(policy, masks)
        outage_count = _outage_count(segments, lengths)
        if selected_count > outage_count:
            raise ValueError(f"selected_outages exceeds outage_count for {case.get('case_id', index)}")
        cases.append(
            {
                "suite_case_id": str(case.get("case_id") or f"locked_case_{index:03d}"),
                "purpose": str(case.get("purpose", "locked nominal branch recovery baseline case")),
                "transfer_time": float(case.get("transfer_time", benchmark.get("transfer_time"))),
                "amax": float(case.get("amax", benchmark.get("amax"))),
                "segments": segments,
                "outage_lengths": lengths,
                "outage_count": outage_count,
                "selected_outage_policy": policy.label,
                "selected_outages_raw": case.get("selected_outages", 0),
                "selected_outage_count": selected_count,
                "nominal_max_nfev": int(case.get("nominal_max_nfev", nominal.get("max_nfev", 140))),
                "branch_max_nfev": int(case.get("branch_max_nfev", branch.get("max_nfev", 120))),
                "min_recovery_segments": int(
                    case.get(
                        "min_recovery_segments",
                        locked.get("min_recovery_segments", nominal.get("min_recovery_segments", 0)),
                    )
                ),
                "node_initialization": str(case.get("node_initialization", nominal.get("node_initialization", "linear"))),
                "node_initialization_blend": float(
                    case.get("node_initialization_blend", nominal.get("node_initialization_blend", 0.5))
                ),
                "case_order": index,
                "case_raw": case,
            }
        )
    seen = set()
    for case in cases:
        if case["suite_case_id"] in seen:
            raise ValueError(f"duplicate locked recovery case_id: {case['suite_case_id']}")
        seen.add(case["suite_case_id"])
    return cases


def _case_payload(case: dict) -> dict:
    return {
        "suite_case_id": str(case["suite_case_id"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "outage_lengths": [int(value) for value in case["outage_lengths"]],
        "outage_count": int(case["outage_count"]),
        "selected_outage_policy": str(case["selected_outage_policy"]),
        "selected_outages_raw": str(case["selected_outages_raw"]),
        "selected_outage_count": int(case["selected_outage_count"]),
        "nominal_max_nfev": int(case["nominal_max_nfev"]),
        "branch_max_nfev": int(case["branch_max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "node_initialization": str(case["node_initialization"]),
        "node_initialization_blend": float(case["node_initialization_blend"]),
    }


def _effective_settings(config: dict, args, case: dict) -> dict:
    case_config = _case_config(config, case)
    weights = _branch_weights(config, case["case_raw"])
    tolerances = _tolerances(config, case["case_raw"])
    return {
        "suite": "locked_nominal_recovery",
        "case": _case_payload(case),
        "nominal_residual_weights": _nominal_residual_weights(config, case["case_raw"]),
        "branch_weights": weights.as_dict(),
        "tolerances": tolerances,
        "thresholds": copy.deepcopy(case_config["objective"]["thresholds"]),
        "benchmark": {
            "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
            "substeps_per_segment": int(case_config["benchmark"]["substeps_per_segment"]),
        },
        "config_hash": config_hash(case_config),
        "source_states_id": file_identity(args.source_states),
    }


def _expected_index(config: dict, args, cases: list[dict]) -> dict[str, dict]:
    expected = {}
    for case in cases:
        settings = _effective_settings(config, args, case)
        expected[str(case["suite_case_id"])] = {
            "case": case,
            "settings_fingerprint": settings_fingerprint(settings),
            "config_hash": settings["config_hash"],
            "source_states_id": settings["source_states_id"],
        }
    return expected


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=LOCKED_RECOVERY_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in LOCKED_RECOVERY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[LOCKED_RECOVERY_COLUMNS]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _missing_or_different(value, expected: str) -> bool:
    return _is_missing(value) or str(value) != str(expected)


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return pd.DataFrame(columns=LOCKED_RECOVERY_COLUMNS), []
    kept = []
    rejected = []
    seen = set()
    for row in df.to_dict(orient="records"):
        case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": case_id, "reason": "not in current requested case set"})
            continue
        if _missing_or_different(row.get("settings_fingerprint"), expected_row["settings_fingerprint"]):
            rejected.append(
                {
                    "suite_case_id": case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": None if _is_missing(row.get("settings_fingerprint")) else str(row.get("settings_fingerprint")),
                }
            )
            continue
        mismatched = [
            key
            for key in ("config_hash", "source_states_id")
            if _missing_or_different(row.get(key), expected_row[key])
        ]
        if mismatched:
            rejected.append({"suite_case_id": case_id, "reason": "provenance field mismatch", "mismatched_fields": mismatched})
            continue
        fingerprint = str(row["settings_fingerprint"])
        if fingerprint in seen:
            rejected.append({"suite_case_id": case_id, "reason": "duplicate compatible settings_fingerprint"})
            continue
        kept.append({column: row.get(column) for column in LOCKED_RECOVERY_COLUMNS})
        seen.add(fingerprint)
    return pd.DataFrame(kept, columns=LOCKED_RECOVERY_COLUMNS), rejected


def _row_from_result(case: dict, case_config: dict, states, cfg, result: dict, expected: dict, config: dict) -> dict:
    weights = _branch_weights(config, case["case_raw"])
    tolerances = _tolerances(config, case["case_raw"])
    row = {
        **_case_payload(case),
        "purpose": str(case["purpose"]),
        "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
        "target_generation": str((getattr(states, "target_metadata", {}) or {}).get("target_state_generation", "catalog target generation")),
        "substeps_per_segment": int(cfg.substeps),
        "outage_lengths": _json_list(case["outage_lengths"]),
        "selected_outages": str(case["selected_outages_raw"]),
        "selected_outage_count": int(case["selected_outage_count"]),
        "terminal_weight": weights.terminal,
        "control_weight": weights.control,
        "smooth_weight": weights.smooth,
        "continuity_weight": weights.continuity,
        **tolerances,
        "settings_fingerprint": expected["settings_fingerprint"],
        "config_hash": expected["config_hash"],
        "source_states_id": expected["source_states_id"],
        "mode": str(result["mode"]),
        "method_type": str(result["method_type"]),
        "nominal_error": float(result["nominal_error"]),
        "nominal_baseline_error": float(result["nominal_baseline_error"]),
        "nominal_lock_error_delta": float(result["nominal_lock_error_delta"]),
        "selected_recovery_worst_error": float(result["selected_recovery_worst_error"]),
        "selected_worst_error": float(result["selected_worst_error"]),
        "all_outage_worst_error": float(result["all_outage_worst_error"]),
        "all_mask_worst_error": float(result["all_mask_worst_error"]),
        "nominal_threshold": float(result["nominal_threshold"]),
        "selected_recovery_threshold": float(result["selected_recovery_threshold"]),
        "selected_worst_threshold": float(result["selected_worst_threshold"]),
        "meets_nominal_threshold": bool(result["meets_nominal_threshold"]),
        "meets_selected_recovery_threshold": bool(result["meets_selected_recovery_threshold"]),
        "meets_selected_worst_threshold": bool(result["meets_selected_worst_threshold"]),
        "meets_thresholds": bool(result["meets_thresholds"]),
        "backend_success": bool(result["backend_success"]),
        "optimizer_success": bool(result["optimizer_success"]),
        "nominal_optimizer_success": bool(result["nominal_optimizer_success"]),
        "nominal_backend_success": bool(result["nominal_backend_success"]),
        "nominal_fuel": float(result["nominal_fuel"]),
        "recovery_fuel_mean": float(result["recovery_fuel_mean"]),
        "recovery_fuel_max": float(result["recovery_fuel_max"]),
        "control_max_norm": float(result["control_max_norm"]),
        "control_bound_violation": float(result["control_bound_violation"]),
        "nominal_nfev": int(result["nominal_nfev"]),
        "total_branch_nfev": int(result["total_branch_nfev"]),
        "nfev": int(result["nfev"]),
        "nominal_runtime_seconds": float(result["nominal_runtime_seconds"]),
        "total_branch_runtime_seconds": float(result["total_branch_runtime_seconds"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "cost": float(result["cost"]),
        "optimality": float(result["optimality"]),
        "nominal_cost": float(result["nominal_cost"]),
        "nominal_optimality": float(result["nominal_optimality"]),
        "selected_outage_indices": _json_list(result["selected_outage_indices"]),
        "selected_outage_errors": _json_list(result["selected_outage_errors"]),
        "all_outage_errors": _json_list(result["all_outage_errors"]),
        "nominal_masked_outage_errors": _json_list(result["nominal_masked_outage_errors"]),
        "branch_nfev": _json_list(result["branch_nfev"]),
        "branch_runtime_seconds": _json_list(result["branch_runtime_seconds"]),
        "branch_optimizer_success": _json_list(result["branch_optimizer_success"]),
        "branch_recovery_starts": _json_list(result["branch_recovery_starts"]),
        "branch_recovery_segments": _json_list(result["branch_recovery_segments"]),
        "branch_results": _json_list(result["branch_results"]),
        "nominal_accepted_candidate": str(result["nominal_accepted_candidate"]),
        "backend_semantics": str(result["backend_semantics"]),
        "selection_semantics": str(result["selection_semantics"]),
        "selected_branch_semantics": str(result["selected_branch_semantics"]),
        "all_mask_diagnostic_semantics": str(result["all_mask_diagnostic_semantics"]),
        "control_bound_semantics": str(result["control_bound_semantics"]),
        "nominal_lock_semantics": str(result["nominal_lock_semantics"]),
        "worst_error_semantics": str(result["worst_error_semantics"]),
        "message": str(result["message"]),
        "nominal_message": str(result["nominal_message"]),
    }
    return {column: row.get(column) for column in LOCKED_RECOVERY_COLUMNS}


def run_case(config: dict, args, case: dict, expected: dict) -> dict:
    case_config = _case_config(config, case)
    states = load_configured_states(Path.cwd(), case_config, args.source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    tolerances = _tolerances(config, case["case_raw"])
    result = run_locked_nominal_recovery_baseline(
        state0=states.initial,
        target=states.target,
        cfg=cfg,
        masks=masks,
        thresholds=case_config["objective"]["thresholds"],
        selected_outages=case["selected_outages_raw"],
        nominal_max_nfev=int(case["nominal_max_nfev"]),
        branch_max_nfev=int(case["branch_max_nfev"]),
        min_recovery_segments=int(case["min_recovery_segments"]),
        nominal_residual_weights=_nominal_residual_weights(config, case["case_raw"]),
        branch_weights=_branch_weights(config, case["case_raw"]),
        node_initialization=str(case["node_initialization"]),
        node_initialization_blend=float(case["node_initialization_blend"]),
        **tolerances,
    )
    return _row_from_result(case, case_config, states, cfg, result, expected, config)


def write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    path = tables_dir / "locked_nominal_recovery_table.tex"
    if df.empty:
        path.write_text("% No locked nominal recovery rows.\n", encoding="utf-8")
        return
    table = df.sort_values("suite_case_id")[
        [
            "suite_case_id",
            "selected_outage_policy",
            "selected_outage_count",
            "nominal_error",
            "selected_worst_error",
            "all_mask_worst_error",
            "control_max_norm",
            "control_bound_violation",
            "nfev",
            "runtime_seconds",
            "meets_thresholds",
        ]
    ].copy()
    table.columns = [
        "Case",
        "Policy",
        "Selected masks",
        "Nominal error",
        "Selected worst error",
        "All-mask diagnostic worst",
        "Max ||u||",
        "Bound violation",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
    ]
    path.write_text(table.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8")


def write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if df.empty:
        ax.text(0.5, 0.5, "No locked nominal recovery rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        colors = np.where(df["meets_thresholds"].astype(bool).to_numpy(), "tab:green", "tab:red")
        ax.scatter(df["nominal_error"], df["selected_worst_error"], c=colors, s=90, edgecolors="black", linewidths=0.6)
        offsets = [(6, 5), (6, -10), (-8, 7), (-8, -12)]
        for index, (_, row) in enumerate(df.iterrows()):
            ax.annotate(
                str(row["suite_case_id"]).replace("locked_hard_", ""),
                (float(row["nominal_error"]), float(row["selected_worst_error"])),
                textcoords="offset points",
                xytext=offsets[index % len(offsets)],
                ha="left" if offsets[index % len(offsets)][0] >= 0 else "right",
                fontsize=8,
            )
        ax.axvline(float(df["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(float(df["selected_worst_threshold"].iloc[0]), color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Locked nominal error")
        ax.set_ylabel("Selected-worst recovery error")
        ax.set_title("Locked-nominal hard-catalog recovery")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"locked_nominal_recovery{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def regenerate(
    df: pd.DataFrame,
    *,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    cases: list[dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> None:
    df = pd.DataFrame(df, columns=LOCKED_RECOVERY_COLUMNS)
    df.to_csv(results_dir / "locked_nominal_recovery.csv", index=False)
    write_table(df, tables_dir)
    write_plot(df, figures_dir)
    feasible_count = int(df["meets_thresholds"].astype(bool).sum()) if not df.empty else 0
    extra = {
        "row_count": int(len(df)),
        "feasible_row_count": feasible_count,
        "expected_case_count": int(len(cases)),
        "completed_case_count": int(len(df)),
        "resume_rejected_rows": resume_rejected_rows,
        "skipped_cases": skipped_cases,
        "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_worst_error <= robust_success",
        "semantics": {
            "backend": (
                "Locked-nominal independent branch recovery is a diagnostic continuous backend baseline, not quantum evidence. "
                "It first solves a nominal all-windows multiple-shooting trajectory, then freezes that nominal control sequence."
            ),
            "branch_recovery": (
                "Each selected missed-thrust mask is optimized independently. Pre-recovery controls are fixed to locked nominal "
                "controls with the missed segment(s) zeroed by the mask; only post-outage recovery controls are variables."
            ),
            "selection": (
                "Integer selected_outages policies choose hardest masks by fixed nominal masked-control terminal error. "
                "all_single selects all one-segment masks; all/all_configured selects every configured mask."
            ),
            "all_mask": (
                "all_mask_worst_error is diagnostic across all configured masks; selected masks use optimized branches and "
                "unselected masks use masked locked nominal controls only."
            ),
            "resume": "resume reuses only rows whose suite_case_id, settings_fingerprint, config_hash, and source_states_id match current settings",
        },
        "limitations": [
            "This baseline is not integrated into the manuscript by this script.",
            "It does not prove universal all-mask robustness unless all configured masks are selected.",
            "Optimizer failure flags are retained even when threshold metrics are met.",
        ],
        "expected_cases": [_case_payload(case) for case in cases],
        "rows": df.to_dict(orient="records"),
    }
    write_metadata(results_dir / "locked_nominal_recovery_metadata.json", command, config, extra)


def _runtime_budget(args, config: dict) -> float | None:
    if args.runtime_budget_seconds is not None:
        return float(args.runtime_budget_seconds)
    value = _locked_config(config).get("runtime_budget_seconds")
    if value is None:
        value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    return None if value is None else float(value)


def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "locked_nominal_recovery.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    expected = _expected_index(config, args, cases)

    if args.resume:
        df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
    else:
        df = pd.DataFrame(columns=LOCKED_RECOVERY_COLUMNS)
        resume_rejected_rows = []
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())
    command = " ".join(sys.argv)
    skipped_cases: list[dict] = []
    budget = _runtime_budget(args, config)
    started = time.perf_counter()

    for case in cases:
        expected_row = expected[str(case["suite_case_id"])]
        if expected_row["settings_fingerprint"] in completed:
            continue
        elapsed = time.perf_counter() - started
        if budget is not None and elapsed >= budget:
            skipped_cases.append(
                {
                    "suite_case_id": str(case["suite_case_id"]),
                    "reason": "runtime budget reached before launching case",
                    "elapsed_seconds": elapsed,
                    "runtime_budget_seconds": budget,
                }
            )
            continue
        row = run_case(config, args, case, expected_row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        regenerate(
            df,
            results_dir=results_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            config=config,
            command=command,
            cases=cases,
            resume_rejected_rows=resume_rejected_rows,
            skipped_cases=skipped_cases,
        )
        print(
            f"case {case['suite_case_id']} policy={row['selected_outage_policy']} "
            f"selected={row['selected_outage_count']}/{row['outage_count']}: "
            f"nominal={row['nominal_error']:.6f}, selected={row['selected_worst_error']:.6f}, "
            f"all={row['all_mask_worst_error']:.6f}, met={row['meets_thresholds']}, "
            f"branch_opt={row['optimizer_success']}, nfev={row['nfev']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(df, columns=LOCKED_RECOVERY_COLUMNS)
    if not df.empty:
        order = {str(case["suite_case_id"]): int(case["case_order"]) for case in cases}
        df["_case_order"] = df["suite_case_id"].map(order)
        df = df.sort_values("_case_order").drop(columns=["_case_order"]).reset_index(drop=True)
    regenerate(
        df,
        results_dir=results_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        config=config,
        command=command,
        cases=cases,
        resume_rejected_rows=resume_rejected_rows,
        skipped_cases=skipped_cases,
    )
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked-nominal independent branch-recovery baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/hard_catalog_locked_nominal_recovery.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
