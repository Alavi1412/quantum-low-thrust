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

from qlt.delayed_recovery import (
    normalize_branch_weight_variants,
    run_delayed_locked_recovery_baseline,
)
from qlt.direct_collocation import config_hash, file_identity, settings_fingerprint
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.locked_recovery import (
    BranchRecoveryWeights,
    normalize_selected_outage_policy,
    selected_outage_count_for_policy,
)
from qlt.objective import outage_masks
from qlt.reporting import write_metadata


DELAYED_RECOVERY_COLUMNS = [
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
    "recovery_horizon_segments",
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
    "nominal_original_error",
    "nominal_baseline_error",
    "nominal_lock_error_delta",
    "nominal_delayed_coast_error",
    "selected_recovery_worst_error",
    "selected_worst_error",
    "all_outage_worst_error",
    "all_mask_worst_error",
    "nominal_dt",
    "delayed_target_time",
    "branch_total_duration",
    "branch_control_count",
    "original_target_state",
    "delayed_target_state",
    "nominal_threshold",
    "selected_recovery_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_recovery_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "backend_success",
    "optimizer_success",
    "optimizer_success_semantics",
    "nominal_optimizer_success",
    "nominal_backend_success",
    "branch_optimizer_success",
    "branch_optimizer_all_success",
    "branch_optimizer_ran",
    "branch_portfolio_enabled",
    "branch_portfolio_variant_count",
    "branch_portfolio_variant_labels",
    "branch_portfolio_all_success",
    "portfolio_acceptance_rule",
    "branch_portfolio_converged_threshold_feasible_candidate_counts",
    "branch_weight_variants",
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
    "branch_optimizer_success_by_branch",
    "branch_optimizer_ran_by_branch",
    "branch_accepted_weight_variant_labels",
    "branch_accepted_weight_variant_indices",
    "branch_accepted_weights",
    "branch_accepted_variant_nfev",
    "branch_accepted_variant_runtime_seconds",
    "branch_recovery_starts",
    "branch_recovery_segments",
    "branch_control_counts",
    "branch_results",
    "nominal_accepted_candidate",
    "backend_semantics",
    "selection_semantics",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "nominal_lock_semantics",
    "delayed_target_semantics",
    "worst_error_semantics",
    "message",
    "nominal_message",
]


def _json_list(value) -> str:
    return json.dumps(value, sort_keys=False)


def _outage_count(segments: int, lengths: list[int]) -> int:
    return int(sum(max(0, int(segments) - int(length) + 1) for length in lengths))


def _implementation_identities() -> dict[str, str]:
    return {
        "delayed_recovery_module": file_identity(ROOT / "src" / "qlt" / "delayed_recovery.py"),
        "delayed_recovery_runner": file_identity(Path(__file__)),
    }


def _delayed_config(config: dict) -> dict:
    return dict(config.get("delayed_recovery", {}) or {})


def _nominal_settings(config: dict) -> dict:
    return dict(_delayed_config(config).get("nominal", {}) or {})


def _branch_settings(config: dict) -> dict:
    return dict(_delayed_config(config).get("branch", {}) or {})


def _nominal_residual_weights(config: dict, case: dict | None = None) -> dict:
    nominal = _nominal_settings(config)
    raw = dict(nominal.get("residual_weights", {}) or {})
    if case is not None:
        raw.update(dict(case.get("nominal_residual_weights", {}) or {}))
    return {str(key): float(value) for key, value in raw.items()}


def _branch_config(config: dict, case: dict | None = None) -> dict:
    raw = copy.deepcopy(_branch_settings(config))
    if case is not None:
        merged = copy.deepcopy(raw)
        merged.update(copy.deepcopy(case.get("branch", {}) or {}))
        for key in ("terminal_weight", "control_weight", "smooth_weight", "continuity_weight"):
            if key in case:
                merged[key] = case[key]
        raw = merged
    return raw


def _branch_weights(config: dict, case: dict | None = None) -> BranchRecoveryWeights:
    raw = _branch_config(config, case)
    return BranchRecoveryWeights.from_config(raw)


def _branch_weight_variants(config: dict, case: dict | None = None) -> list[dict]:
    raw = _branch_config(config, case)
    variants = normalize_branch_weight_variants(
        branch_weights=_branch_weights(config, case),
        branch_weight_variants=raw.get("weight_variants"),
    )
    return [
        {
            "label": str(variant["label"]),
            "index": int(variant["index"]),
            "weights": variant["weights"].as_dict(),
        }
        for variant in variants
    ]


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


def _case_config(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    config.setdefault("outages", {})["block_lengths"] = [int(value) for value in case["outage_lengths"]]
    return config


def _suite_cases(config: dict) -> list[dict]:
    delayed = _delayed_config(config)
    raw_cases = list(delayed.get("cases", (config.get("suite", {}) or {}).get("cases", [])) or [])
    benchmark = config.get("benchmark", {}) or {}
    default_lengths = [int(value) for value in config.get("outages", {}).get("block_lengths", [1])]
    default_horizon = int(delayed.get("recovery_horizon_segments", 0))
    nominal = _nominal_settings(config)
    branch = _branch_settings(config)

    cases: list[dict] = []
    for index, raw in enumerate(raw_cases):
        if not bool(raw.get("enabled", True)):
            continue
        case = dict(raw)
        lengths = [int(value) for value in case.get("outage_lengths", case.get("block_lengths", default_lengths))]
        segments = int(case.get("segments", benchmark.get("segments")))
        horizon = int(case.get("recovery_horizon_segments", default_horizon))
        if horizon < 0:
            raise ValueError(f"recovery_horizon_segments must be non-negative for {case.get('case_id', index)}")
        masks = outage_masks(segments, tuple(lengths))
        policy = normalize_selected_outage_policy(case.get("selected_outages", 0))
        selected_count = selected_outage_count_for_policy(policy, masks)
        outage_count = _outage_count(segments, lengths)
        if selected_count > outage_count:
            raise ValueError(f"selected_outages exceeds outage_count for {case.get('case_id', index)}")
        cases.append(
            {
                "suite_case_id": str(case.get("case_id") or f"delayed_case_{index:03d}"),
                "purpose": str(case.get("purpose", "delayed-arrival locked nominal branch recovery baseline case")),
                "transfer_time": float(case.get("transfer_time", benchmark.get("transfer_time"))),
                "amax": float(case.get("amax", benchmark.get("amax"))),
                "segments": segments,
                "outage_lengths": lengths,
                "outage_count": outage_count,
                "selected_outage_policy": policy.label,
                "selected_outages_raw": case.get("selected_outages", 0),
                "selected_outage_count": selected_count,
                "recovery_horizon_segments": horizon,
                "nominal_max_nfev": int(case.get("nominal_max_nfev", nominal.get("max_nfev", 140))),
                "branch_max_nfev": int(case.get("branch_max_nfev", branch.get("max_nfev", 120))),
                "min_recovery_segments": int(
                    case.get(
                        "min_recovery_segments",
                        delayed.get("min_recovery_segments", nominal.get("min_recovery_segments", 0)),
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
            raise ValueError(f"duplicate delayed recovery case_id: {case['suite_case_id']}")
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
        "recovery_horizon_segments": int(case["recovery_horizon_segments"]),
        "nominal_max_nfev": int(case["nominal_max_nfev"]),
        "branch_max_nfev": int(case["branch_max_nfev"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "node_initialization": str(case["node_initialization"]),
        "node_initialization_blend": float(case["node_initialization_blend"]),
    }


def _effective_settings(config: dict, args, case: dict) -> dict:
    case_config = _case_config(config, case)
    weights = _branch_weights(config, case["case_raw"])
    variants = _branch_weight_variants(config, case["case_raw"])
    tolerances = _tolerances(config, case["case_raw"])
    return {
        "suite": "delayed_locked_recovery",
        "case": _case_payload(case),
        "nominal_residual_weights": _nominal_residual_weights(config, case["case_raw"]),
        "branch_weights": weights.as_dict(),
        "branch_weight_variants": _branch_weight_variants(config, case["case_raw"]),
        "tolerances": tolerances,
        "thresholds": copy.deepcopy(case_config["objective"]["thresholds"]),
        "benchmark": {
            "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
            "substeps_per_segment": int(case_config["benchmark"]["substeps_per_segment"]),
        },
        "config_hash": config_hash(case_config),
        "source_states_id": file_identity(args.source_states),
        "implementation_identities": _implementation_identities(),
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
        return pd.DataFrame(columns=DELAYED_RECOVERY_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in DELAYED_RECOVERY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[DELAYED_RECOVERY_COLUMNS]


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
        return pd.DataFrame(columns=DELAYED_RECOVERY_COLUMNS), []
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
        kept.append({column: row.get(column) for column in DELAYED_RECOVERY_COLUMNS})
        seen.add(fingerprint)
    return pd.DataFrame(kept, columns=DELAYED_RECOVERY_COLUMNS), rejected


def _row_from_result(case: dict, case_config: dict, states, cfg, result: dict, expected: dict, config: dict) -> dict:
    weights = _branch_weights(config, case["case_raw"])
    variants = _branch_weight_variants(config, case["case_raw"])
    tolerances = _tolerances(config, case["case_raw"])
    row = {
        **_case_payload(case),
        "purpose": str(case["purpose"]),
        "target_mode": str(case_config["benchmark"].get("target_mode", "catalog_dro_phase")),
        "target_generation": str((getattr(states, "target_metadata", {}) or {}).get("target_state_generation", "catalog target generation")),
        "substeps_per_segment": int(cfg.substeps),
        "outage_lengths": _json_list(case["outage_lengths"]),
        "selected_outages": str(case["selected_outages_raw"]),
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
        "nominal_original_error": float(result["nominal_original_error"]),
        "nominal_baseline_error": float(result["nominal_baseline_error"]),
        "nominal_lock_error_delta": float(result["nominal_lock_error_delta"]),
        "nominal_delayed_coast_error": float(result["nominal_delayed_coast_error"]),
        "selected_recovery_worst_error": float(result["selected_recovery_worst_error"]),
        "selected_worst_error": float(result["selected_worst_error"]),
        "all_outage_worst_error": float(result["all_outage_worst_error"]),
        "all_mask_worst_error": float(result["all_mask_worst_error"]),
        "nominal_dt": float(result["nominal_dt"]),
        "delayed_target_time": float(result["delayed_target_time"]),
        "branch_total_duration": float(result["branch_total_duration"]),
        "branch_control_count": int(result["branch_control_count"]),
        "original_target_state": _json_list(np.asarray(result["original_target_state"], dtype=float).tolist()),
        "delayed_target_state": _json_list(np.asarray(result["delayed_target_state"], dtype=float).tolist()),
        "nominal_threshold": float(result["nominal_threshold"]),
        "selected_recovery_threshold": float(result["selected_recovery_threshold"]),
        "selected_worst_threshold": float(result["selected_worst_threshold"]),
        "meets_nominal_threshold": bool(result["meets_nominal_threshold"]),
        "meets_selected_recovery_threshold": bool(result["meets_selected_recovery_threshold"]),
        "meets_selected_worst_threshold": bool(result["meets_selected_worst_threshold"]),
        "meets_thresholds": bool(result["meets_thresholds"]),
        "backend_success": bool(result["backend_success"]),
        "optimizer_success": bool(result["optimizer_success"]),
        "optimizer_success_semantics": str(result["optimizer_success_semantics"]),
        "nominal_optimizer_success": bool(result["nominal_optimizer_success"]),
        "nominal_backend_success": bool(result["nominal_backend_success"]),
        "branch_optimizer_success": bool(result["branch_optimizer_success"]),
        "branch_optimizer_all_success": bool(result["branch_optimizer_all_success"]),
        "branch_optimizer_ran": bool(result["branch_optimizer_ran"]),
        "branch_portfolio_enabled": bool(result["branch_portfolio_enabled"]),
        "branch_portfolio_variant_count": int(result["branch_portfolio_variant_count"]),
        "branch_portfolio_variant_labels": _json_list(result["branch_portfolio_variant_labels"]),
        "branch_portfolio_all_success": bool(result["branch_portfolio_all_success"]),
        "portfolio_acceptance_rule": str(result["portfolio_acceptance_rule"]),
        "branch_portfolio_converged_threshold_feasible_candidate_counts": _json_list(
            result["branch_portfolio_converged_threshold_feasible_candidate_counts"]
        ),
        "branch_weight_variants": _json_list(variants),
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
        "branch_optimizer_success_by_branch": _json_list(result["branch_optimizer_success_by_branch"]),
        "branch_optimizer_ran_by_branch": _json_list(result["branch_optimizer_ran_by_branch"]),
        "branch_accepted_weight_variant_labels": _json_list(result["branch_accepted_weight_variant_labels"]),
        "branch_accepted_weight_variant_indices": _json_list(result["branch_accepted_weight_variant_indices"]),
        "branch_accepted_weights": _json_list(result["branch_accepted_weights"]),
        "branch_accepted_variant_nfev": _json_list(result["branch_accepted_variant_nfev"]),
        "branch_accepted_variant_runtime_seconds": _json_list(result["branch_accepted_variant_runtime_seconds"]),
        "branch_recovery_starts": _json_list(result["branch_recovery_starts"]),
        "branch_recovery_segments": _json_list(result["branch_recovery_segments"]),
        "branch_control_counts": _json_list(result["branch_control_counts"]),
        "branch_results": _json_list(result["branch_results"]),
        "nominal_accepted_candidate": str(result["nominal_accepted_candidate"]),
        "backend_semantics": str(result["backend_semantics"]),
        "selection_semantics": str(result["selection_semantics"]),
        "selected_branch_semantics": str(result["selected_branch_semantics"]),
        "all_mask_diagnostic_semantics": str(result["all_mask_diagnostic_semantics"]),
        "control_bound_semantics": str(result["control_bound_semantics"]),
        "nominal_lock_semantics": str(result["nominal_lock_semantics"]),
        "delayed_target_semantics": str(result["delayed_target_semantics"]),
        "worst_error_semantics": str(result["worst_error_semantics"]),
        "message": str(result["message"]),
        "nominal_message": str(result["nominal_message"]),
    }
    return {column: row.get(column) for column in DELAYED_RECOVERY_COLUMNS}


def run_case(config: dict, args, case: dict, expected: dict) -> dict:
    case_config = _case_config(config, case)
    states = load_configured_states(Path.cwd(), case_config, args.source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    tolerances = _tolerances(config, case["case_raw"])
    result = run_delayed_locked_recovery_baseline(
        state0=states.initial,
        target=states.target,
        cfg=cfg,
        masks=masks,
        thresholds=case_config["objective"]["thresholds"],
        recovery_horizon_segments=int(case["recovery_horizon_segments"]),
        selected_outages=case["selected_outages_raw"],
        nominal_max_nfev=int(case["nominal_max_nfev"]),
        branch_max_nfev=int(case["branch_max_nfev"]),
        min_recovery_segments=int(case["min_recovery_segments"]),
        nominal_residual_weights=_nominal_residual_weights(config, case["case_raw"]),
        branch_weights=_branch_weights(config, case["case_raw"]),
        branch_weight_variants=_branch_weight_variants(config, case["case_raw"]),
        node_initialization=str(case["node_initialization"]),
        node_initialization_blend=float(case["node_initialization_blend"]),
        **tolerances,
    )
    return _row_from_result(case, case_config, states, cfg, result, expected, config)


def write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    path = tables_dir / "delayed_locked_recovery_table.tex"
    if df.empty:
        path.write_text("% No delayed locked recovery rows.\n", encoding="utf-8")
        return
    table = df.sort_values("suite_case_id")[
        [
            "suite_case_id",
            "selected_outage_policy",
            "selected_outage_count",
            "recovery_horizon_segments",
            "branch_portfolio_enabled",
            "branch_portfolio_variant_labels",
            "nominal_error",
            "nominal_delayed_coast_error",
            "selected_worst_error",
            "all_mask_worst_error",
            "terminal_weight",
            "control_weight",
            "smooth_weight",
            "continuity_weight",
            "control_max_norm",
            "control_bound_violation",
            "nfev",
            "runtime_seconds",
            "meets_thresholds",
            "nominal_optimizer_success",
            "branch_optimizer_ran",
            "branch_optimizer_all_success",
        ]
    ].copy()
    table.columns = [
        "Case",
        "Policy",
        "Selected masks",
        "Horizon segments",
        "Portfolio",
        "Variants",
        "Nominal original error",
        "Nominal delayed coast error",
        "Selected delayed worst error",
        "All-mask delayed diagnostic worst",
        "Terminal weight",
        "Control weight",
        "Smooth weight",
        "Continuity weight",
        "Max ||u||",
        "Bound violation",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
        "Nominal optimizer success",
        "Branch optimizer ran",
        "Branch optimizer all success",
    ]
    path.write_text(table.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8")


def write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if df.empty:
        ax.text(0.5, 0.5, "No delayed locked recovery rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        colors = np.where(df["meets_thresholds"].astype(bool).to_numpy(), "tab:green", "tab:red")
        ax.scatter(df["nominal_error"], df["selected_worst_error"], c=colors, s=90, edgecolors="black", linewidths=0.6)
        offsets = [(6, 5), (6, -10), (-8, 7), (-8, -12)]
        for index, (_, row) in enumerate(df.iterrows()):
            label = f"{row['suite_case_id']} h={int(row['recovery_horizon_segments'])}"
            ax.annotate(
                label.replace("delayed_hard_", ""),
                (float(row["nominal_error"]), float(row["selected_worst_error"])),
                textcoords="offset points",
                xytext=offsets[index % len(offsets)],
                ha="left" if offsets[index % len(offsets)][0] >= 0 else "right",
                fontsize=8,
            )
        ax.axvline(float(df["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(float(df["selected_worst_threshold"].iloc[0]), color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Nominal original-target error")
        ax.set_ylabel("Selected delayed-target worst error")
        ax.set_title("Delayed-arrival locked-nominal recovery")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"delayed_locked_recovery{suffix}", dpi=220 if suffix == ".png" else None)
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
    df = pd.DataFrame(df, columns=DELAYED_RECOVERY_COLUMNS)
    df.to_csv(results_dir / "delayed_locked_recovery.csv", index=False)
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
        "implementation_identities": _implementation_identities(),
        "threshold_rule": (
            "meets_thresholds requires nominal original-target error <= nominal_success and selected delayed-target "
            "worst error <= robust_success"
        ),
        "semantics": {
            "backend": (
                "Delayed-arrival locked-nominal branch recovery is a diagnostic continuous backend baseline, not "
                "quantum evidence and not fixed-final-time robustness."
            ),
            "nominal": (
                "The nominal all-windows controls are solved once to the original target at transfer_time and then "
                "locked for every branch."
            ),
            "branch_recovery": (
                "Each selected missed-thrust mask is optimized independently. Pre-recovery controls are locked nominal "
                "controls with missed segment(s) zeroed; post-outage controls through the added horizon are variables."
            ),
            "delayed_target": (
                "The branch target is the original target propagated ballistically by recovery_horizon_segments * "
                "nominal_dt, and branch_total_duration is (N + recovery_horizon_segments) * nominal_dt."
            ),
            "selection": (
                "Integer selected_outages policies choose hardest masks by delayed-target terminal error under locked "
                "nominal masked controls and zero acceleration over the added horizon. all_single selects all one-segment "
                "masks; all/all_configured selects every configured mask."
            ),
            "resume": "resume reuses only rows whose suite_case_id, settings_fingerprint, config_hash, and source_states_id match current settings",
            "optimizer_success": (
                "optimizer_success is an overall convergence flag requiring nominal_optimizer_success and "
                "branch_optimizer_all_success. Nominal-only provenance rows and skipped branch optimizers report "
                "branch_optimizer_ran=False and branch_optimizer_all_success=False."
            ),
            "branch_weights": (
                "terminal_weight, control_weight, smooth_weight, and continuity_weight are recorded per row in "
                "the CSV and compact table so terminal-only feasibility diagnostics are distinguishable from "
                "regularized branch-recovery cases."
            ),
            "branch_weight_portfolio": (
                "When branch.weight_variants is configured, every listed branch residual variant is optimized for "
                "each selected outage mask. Branch nfev/runtime charge all variants, not only the accepted one."
            ),
            "portfolio_acceptance_rule": (
                "Portfolio acceptance first considers optimizer_success=True candidates whose terminal_error is at "
                "or below robust_success, selecting among them by terminal error then fuel. If none exist, it falls "
                "back to terminal error then fuel across all variants."
            ),
        },
        "limitations": [
            "This is delayed-arrival recovery horizon evidence, not a fixed-final-time robustness claim.",
            "Optimizer failure flags are retained even when threshold metrics are met.",
            "It does not prove universal all-mask robustness unless all configured masks are selected.",
        ],
        "expected_cases": [_case_payload(case) for case in cases],
        "branch_weight_variants_by_case": {
            str(case["suite_case_id"]): _branch_weight_variants(config, case["case_raw"])
            for case in cases
        },
        "rows": df.to_dict(orient="records"),
    }
    write_metadata(results_dir / "delayed_locked_recovery_metadata.json", command, config, extra)


def _runtime_budget(args, config: dict) -> float | None:
    if args.runtime_budget_seconds is not None:
        return float(args.runtime_budget_seconds)
    value = _delayed_config(config).get("runtime_budget_seconds")
    if value is None:
        value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    return None if value is None else float(value)


def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "delayed_locked_recovery.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    expected = _expected_index(config, args, cases)

    if args.resume:
        df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
    else:
        df = pd.DataFrame(columns=DELAYED_RECOVERY_COLUMNS)
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
            f"case {case['suite_case_id']} horizon={row['recovery_horizon_segments']} "
            f"policy={row['selected_outage_policy']} selected={row['selected_outage_count']}/{row['outage_count']}: "
            f"nominal={row['nominal_error']:.6f}, selected_delayed={row['selected_worst_error']:.6f}, "
            f"all_delayed={row['all_mask_worst_error']:.6f}, met={row['meets_thresholds']}, "
            f"portfolio_variants={row['branch_portfolio_variant_count']}, "
            f"branch_opt_ran={row['branch_optimizer_ran']}, "
            f"branch_opt_all_success={row['branch_optimizer_all_success']}, "
            f"nfev={row['nfev']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(df, columns=DELAYED_RECOVERY_COLUMNS)
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
    parser = argparse.ArgumentParser(description="Delayed-arrival locked-nominal independent branch-recovery baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/hard_catalog_delayed_recovery.yaml"))
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
