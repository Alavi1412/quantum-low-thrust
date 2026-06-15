from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.direct_collocation import (
    collocation_method_type,
    config_hash,
    file_identity,
    normalize_collocation_method,
    run_direct_collocation_baseline,
    settings_fingerprint,
)
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.multiple_shooting import run_multiple_shooting_baseline
from qlt.objective import outage_masks
from qlt.reporting import write_metadata


CATALOG_FEASIBILITY_ENVELOPE_COLUMNS = [
    "suite_case_id",
    "case_type",
    "purpose",
    "target_mode",
    "target_generation",
    "backend",
    "method",
    "method_type",
    "collocation_method",
    "transfer_time",
    "amax",
    "segments",
    "substeps_per_segment",
    "max_nfev",
    "selected_outages",
    "min_recovery_segments",
    "node_initialization",
    "node_initialization_blend",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "nominal_threshold",
    "selected_worst_threshold",
    "thresholds",
    "meets_nominal_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "control_max_norm",
    "control_bound_violation",
    "nominal_fuel",
    "recovery_fuel_mean",
    "recovery_fuel_max",
    "cost",
    "optimality",
    "nfev",
    "runtime_seconds",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "optimizer_success",
    "backend_success",
    "accepted_candidate",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "nominal_only_semantics",
    "message",
]


def _normalize_backend(value: str | None) -> str:
    backend = str(value or "multiple_shooting").strip().lower().replace("-", "_")
    if backend not in {"multiple_shooting", "direct_collocation"}:
        raise ValueError("catalog feasibility envelope backend must be multiple_shooting or direct_collocation")
    return backend


def _method_for_case(case: dict) -> str:
    backend = _normalize_backend(case.get("backend"))
    if backend == "multiple_shooting":
        return "multiple_shooting"
    method = normalize_collocation_method(case.get("collocation_method", case.get("method", "trapezoidal")))
    return f"direct_collocation_{method}"


def _suite_cases(config: dict) -> list[dict]:
    raw_cases = list(((config.get("envelope", {}) or {}).get("cases", []) or []))
    cases = []
    for index, raw in enumerate(raw_cases):
        if not bool(raw.get("enabled", True)):
            continue
        case = dict(raw)
        case["suite_case_id"] = str(case.get("case_id") or f"case_{index:03d}")
        case["backend"] = _normalize_backend(case.get("backend"))
        if case["backend"] == "direct_collocation":
            case["collocation_method"] = normalize_collocation_method(
                case.get("collocation_method", case.get("method", (config.get("direct_collocation", {}) or {}).get("method", "trapezoidal")))
            )
        else:
            case["collocation_method"] = ""
        case["method"] = _method_for_case(case)
        case["transfer_time"] = float(case.get("transfer_time", config["benchmark"]["transfer_time"]))
        case["amax"] = float(case.get("amax", config["benchmark"]["amax"]))
        case["segments"] = int(case.get("segments", config["benchmark"]["segments"]))
        case["max_nfev"] = int(case.get("max_nfev", 120))
        case["selected_outages"] = int(case.get("selected_outages", 0))
        case["min_recovery_segments"] = int(case.get("min_recovery_segments", 4))
        case["node_initialization"] = str(case.get("node_initialization", "linear"))
        case["node_initialization_blend"] = float(case.get("node_initialization_blend", 0.5))
        case["purpose"] = str(case.get("purpose", "catalog feasibility envelope case"))
        cases.append(case)
    seen = set()
    for case in cases:
        if case["suite_case_id"] in seen:
            raise ValueError(f"duplicate catalog feasibility envelope case_id: {case['suite_case_id']}")
        seen.add(case["suite_case_id"])
    return cases


def _case_payload(case: dict) -> dict:
    return {
        "suite_case_id": str(case["suite_case_id"]),
        "backend": str(case["backend"]),
        "method": str(case["method"]),
        "collocation_method": str(case.get("collocation_method", "")),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "max_nfev": int(case["max_nfev"]),
        "selected_outages": int(case["selected_outages"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "node_initialization": str(case["node_initialization"]),
        "node_initialization_blend": float(case["node_initialization_blend"]),
    }


def _config_for_case(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    benchmark = config.setdefault("benchmark", {})
    benchmark["target_mode"] = "catalog_dro_phase"
    benchmark["transfer_time"] = float(case["transfer_time"])
    benchmark["amax"] = float(case["amax"])
    benchmark["segments"] = int(case["segments"])
    return config


def _direct_config_for_case(case_config: dict, case: dict) -> dict:
    direct = copy.deepcopy(case_config.get("direct_collocation", {}) or {})
    direct.update(copy.deepcopy(case.get("direct_collocation", {}) or {}))
    direct["method"] = normalize_collocation_method(case.get("collocation_method", direct.get("method", "trapezoidal")))
    direct["node_initialization"] = str(case["node_initialization"])
    direct["node_initialization_blend"] = float(case["node_initialization_blend"])
    return direct


def _multiple_shooting_weights(case_config: dict, case: dict) -> dict:
    configured = copy.deepcopy(((case_config.get("multiple_shooting", {}) or {}).get("residual_weights", {}) or {}))
    configured.update(copy.deepcopy(case.get("residual_weights", {}) or {}))
    return {key: float(value) for key, value in configured.items()}


def _effective_settings(config: dict, args, case: dict) -> dict:
    case_config = _config_for_case(config, case)
    backend_settings = (
        _direct_config_for_case(case_config, case)
        if case["backend"] == "direct_collocation"
        else {"residual_weights": _multiple_shooting_weights(case_config, case)}
    )
    return {
        "suite": "catalog_feasibility_envelope",
        "case": _case_payload(case),
        "backend_settings": backend_settings,
        "thresholds": copy.deepcopy(case_config["objective"]["thresholds"]),
        "benchmark": {
            "target_mode": "catalog_dro_phase",
            "substeps_per_segment": int(case_config["benchmark"]["substeps_per_segment"]),
        },
        "outages": copy.deepcopy(case_config.get("outages", {})),
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
        return pd.DataFrame(columns=CATALOG_FEASIBILITY_ENVELOPE_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in CATALOG_FEASIBILITY_ENVELOPE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[CATALOG_FEASIBILITY_ENVELOPE_COLUMNS]


def _missing_or_different(value, expected: str) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value) != str(expected)


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return pd.DataFrame(columns=CATALOG_FEASIBILITY_ENVELOPE_COLUMNS), []
    kept = []
    rejected = []
    seen = set()
    for row in df.to_dict(orient="records"):
        suite_case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(suite_case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": suite_case_id, "reason": "not in current requested case set"})
            continue
        if _missing_or_different(row.get("settings_fingerprint"), expected_row["settings_fingerprint"]):
            rejected.append(
                {
                    "suite_case_id": suite_case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": None if row.get("settings_fingerprint") is None else str(row.get("settings_fingerprint")),
                }
            )
            continue
        mismatched = [
            key
            for key in ("config_hash", "source_states_id")
            if _missing_or_different(row.get(key), expected_row[key])
        ]
        if mismatched:
            rejected.append(
                {
                    "suite_case_id": suite_case_id,
                    "reason": "provenance field mismatch",
                    "mismatched_fields": mismatched,
                }
            )
            continue
        fingerprint = str(row["settings_fingerprint"])
        if fingerprint in seen:
            rejected.append({"suite_case_id": suite_case_id, "reason": "duplicate compatible settings_fingerprint"})
            continue
        kept.append(row)
        seen.add(fingerprint)
    return pd.DataFrame(kept, columns=CATALOG_FEASIBILITY_ENVELOPE_COLUMNS), rejected


def _case_type(case: dict) -> str:
    return "nominal_only" if int(case["selected_outages"]) <= 0 else "selected_branch"


def _nominal_only_semantics(case: dict) -> str:
    if int(case["selected_outages"]) <= 0:
        return (
            "selected_outages=0 rows optimize only the nominal trajectory; selected_worst_error equals nominal_error, "
            "and all_mask_worst_error is only a diagnostic under masked nominal controls, not missed-thrust robustness evidence"
        )
    return "selected outage recovery branches are optimized for the selected outage masks"


def _multiple_shooting_selected_branch_semantics(case: dict) -> str:
    if int(case["selected_outages"]) > 0:
        return "direct multiple-shooting nominal and selected outage recovery branches are optimized together"
    return "direct multiple shooting optimizes the nominal trajectory only because selected_outages=0"


def _multiple_shooting_all_mask_semantics(case: dict) -> str:
    if int(case["selected_outages"]) > 0:
        return (
            "all outage masks are evaluated after optimization; selected masks use optimized recovery branches, "
            "while unselected masks use masked nominal controls only"
        )
    return "all outage masks are evaluated after optimization; with selected_outages=0 every mask uses masked nominal controls only"


def _multiple_shooting_control_bound_semantics() -> str:
    return "controls are projected to the Euclidean acceleration ball during multiple-shooting residual and evaluation paths"


def _has_selected_recovery_branches(case: dict, row: pd.Series | None = None) -> bool:
    if row is not None:
        raw = row.get("selected_outage_indices")
        if raw is not None and not pd.isna(raw):
            try:
                return len(json.loads(str(raw))) > 0
            except (TypeError, json.JSONDecodeError):
                pass
    return int(case["selected_outages"]) > 0


def _direct_collocation_selected_branch_semantics(case: dict, row: pd.Series | None = None) -> str:
    if _has_selected_recovery_branches(case, row):
        return (
            "nominal trajectory and selected outage branches are optimized in one least-squares direct transcription; "
            "branch starts are fixed by RK4 propagation through the missed segment(s), then branch controls are re-optimized after the outage"
        )
    return (
        "selected_outages=0/no selected outage branches: only the nominal trajectory is optimized; "
        "selected_worst_error equals nominal_error; all outage masks are diagnostic under masked nominal controls"
    )


def _direct_collocation_all_mask_semantics(case: dict, row: pd.Series | None = None) -> str:
    if _has_selected_recovery_branches(case, row):
        return (
            "all outage masks are evaluated after optimization; selected masks use optimized branch recovery controls "
            "and unselected masks use masked nominal controls only"
        )
    return "all outage masks are diagnostic under masked nominal controls only; no selected outage branch recovery controls are optimized"


def _direct_collocation_control_bound_semantics() -> str:
    return "all nominal and branch controls are Euclidean projected to ||u_i|| <= amax inside residual evaluation, RK4 reporting propagation, fuel computation, and output diagnostics; scalar optimizer bounds are finite guards, not the scientific bound"


def _refresh_reused_row_semantics(df: pd.DataFrame, cases: Sequence[dict]) -> pd.DataFrame:
    if df.empty:
        return df
    refreshed = df.copy()
    cases_by_id = {str(case["suite_case_id"]): case for case in cases}
    for index, row in refreshed.iterrows():
        case = cases_by_id.get(str(row.get("suite_case_id")))
        if case is None:
            continue
        refreshed.at[index, "case_type"] = _case_type(case)
        refreshed.at[index, "nominal_only_semantics"] = _nominal_only_semantics(case)
        if str(case.get("backend")) == "multiple_shooting":
            refreshed.at[index, "selected_branch_semantics"] = _multiple_shooting_selected_branch_semantics(case)
            refreshed.at[index, "all_mask_diagnostic_semantics"] = _multiple_shooting_all_mask_semantics(case)
            refreshed.at[index, "control_bound_semantics"] = _multiple_shooting_control_bound_semantics()
        elif str(case.get("backend")) == "direct_collocation":
            refreshed.at[index, "selected_branch_semantics"] = _direct_collocation_selected_branch_semantics(case, row)
            refreshed.at[index, "all_mask_diagnostic_semantics"] = _direct_collocation_all_mask_semantics(case, row)
            refreshed.at[index, "control_bound_semantics"] = _direct_collocation_control_bound_semantics()
    return refreshed


def _row_from_result(case: dict, case_config: dict, states, cfg, result: dict, expected: dict) -> dict:
    thresholds = {
        "nominal_success": float(case_config["objective"]["thresholds"]["nominal_success"]),
        "selected_recovery_success": float(case_config["objective"]["thresholds"]["robust_success"]),
    }
    nominal_error = float(result["nominal_error"])
    selected_worst = float(result["selected_worst_error"])
    all_mask_worst = float(result["all_mask_worst_error"])
    backend = str(case["backend"])
    collocation_method = str(case.get("collocation_method", ""))
    if backend == "multiple_shooting":
        method_type = str(result.get("solver_mode", "bounded_projected_multiple_shooting"))
        selected_branch_semantics = _multiple_shooting_selected_branch_semantics(case)
        all_mask_semantics = _multiple_shooting_all_mask_semantics(case)
        control_bound_semantics = _multiple_shooting_control_bound_semantics()
    else:
        method_type = str(result["method_type"])
        selected_branch_semantics = str(result.get("selected_branch_semantics", ""))
        all_mask_semantics = str(result.get("all_mask_diagnostic_semantics", ""))
        control_bound_semantics = str(result.get("control_bound_semantics", ""))

    row = {
        **_case_payload(case),
        "case_type": _case_type(case),
        "purpose": str(case["purpose"]),
        "target_mode": "catalog_dro_phase",
        "target_generation": str((getattr(states, "target_metadata", {}) or {}).get("target_state_generation", "catalog DRO target generation")),
        "substeps_per_segment": int(cfg.substeps),
        "settings_fingerprint": expected["settings_fingerprint"],
        "config_hash": expected["config_hash"],
        "source_states_id": expected["source_states_id"],
        "backend": backend,
        "method": str(case["method"]),
        "method_type": method_type,
        "collocation_method": collocation_method,
        "nominal_error": nominal_error,
        "selected_worst_error": selected_worst,
        "all_mask_worst_error": all_mask_worst,
        "nominal_threshold": thresholds["nominal_success"],
        "selected_worst_threshold": thresholds["selected_recovery_success"],
        "thresholds": json.dumps(thresholds, sort_keys=True),
        "meets_nominal_threshold": bool(nominal_error <= thresholds["nominal_success"]),
        "meets_selected_worst_threshold": bool(selected_worst <= thresholds["selected_recovery_success"]),
        "meets_thresholds": bool(nominal_error <= thresholds["nominal_success"] and selected_worst <= thresholds["selected_recovery_success"]),
        "control_max_norm": float(result["control_max_norm"]),
        "control_bound_violation": float(result["control_bound_violation"]),
        "nominal_fuel": float(result["nominal_fuel"]),
        "recovery_fuel_mean": float(result["recovery_fuel_mean"]),
        "recovery_fuel_max": float(result["recovery_fuel_max"]),
        "cost": float(result["cost"]),
        "optimality": float(result["optimality"]),
        "nfev": int(result["nfev"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "selected_outage_indices": json.dumps(result.get("selected_outage_indices", [])),
        "selected_outage_errors": json.dumps(result.get("selected_outage_errors", [])),
        "all_outage_errors": json.dumps(result.get("all_outage_errors", [])),
        "optimizer_success": bool(result["optimizer_success"]),
        "backend_success": bool(result["success"]),
        "accepted_candidate": str(result.get("accepted_candidate", "optimizer")),
        "selected_branch_semantics": selected_branch_semantics,
        "all_mask_diagnostic_semantics": all_mask_semantics,
        "control_bound_semantics": control_bound_semantics,
        "nominal_only_semantics": _nominal_only_semantics(case),
        "message": str(result["message"]),
    }
    return {column: row.get(column) for column in CATALOG_FEASIBILITY_ENVELOPE_COLUMNS}


def run_case(config: dict, args, case: dict, expected: dict) -> dict:
    case_config = _config_for_case(config, case)
    states = load_configured_states(Path.cwd(), case_config, args.source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = case_config["objective"]["thresholds"]
    if case["backend"] == "multiple_shooting":
        result = run_multiple_shooting_baseline(
            state0=states.initial,
            target=states.target,
            cfg=cfg,
            masks=masks,
            thresholds=thresholds,
            selected_outages=int(case["selected_outages"]),
            max_nfev=int(case["max_nfev"]),
            min_recovery_segments=int(case["min_recovery_segments"]),
            residual_weights=_multiple_shooting_weights(case_config, case),
            nominal_control_guess=None,
            selected_branch_control_guesses=None,
            node_initialization=str(case["node_initialization"]),
            node_initialization_blend=float(case["node_initialization_blend"]),
            warm_start_info={"suite": "catalog_feasibility_envelope", "case_id": str(case["suite_case_id"])},
        )
    else:
        direct_cfg = _direct_config_for_case(case_config, case)
        result = run_direct_collocation_baseline(
            state0=states.initial,
            target=states.target,
            cfg=cfg,
            masks=masks,
            thresholds=thresholds,
            selected_outages=int(case["selected_outages"]),
            max_nfev=int(case["max_nfev"]),
            min_recovery_segments=int(case["min_recovery_segments"]),
            collocation_config=direct_cfg,
        )
    return _row_from_result(case, case_config, states, cfg, result, expected)


def write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    path = tables_dir / "catalog_feasibility_envelope_table.tex"
    if df.empty:
        path.write_text("% No catalog feasibility envelope rows.\n", encoding="utf-8")
        return
    table = df.sort_values(["case_type", "method", "amax"])[
        [
            "case_type",
            "method",
            "transfer_time",
            "amax",
            "segments",
            "selected_outages",
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
        "Method",
        "Transfer time",
        "amax",
        "Segments",
        "Selected outages",
        "Nominal error",
        "Selected worst error",
        "All-mask diagnostic worst error",
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
        ax.text(0.5, 0.5, "No catalog feasibility envelope rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        colors = np.where(df["meets_thresholds"].to_numpy(dtype=bool), "tab:green", "tab:red")
        ax.scatter(df["nominal_error"], df["all_mask_worst_error"], c=colors, s=90, edgecolors="black", linewidths=0.6)
        for _, row in df.iterrows():
            ax.annotate(
                f"{row['method']}, a={row['amax']:.2f}",
                (row["nominal_error"], row["all_mask_worst_error"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
        ax.axvline(float(df["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(float(df["selected_worst_threshold"].iloc[0]), color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Nominal error")
        ax.set_ylabel("All-mask diagnostic worst error")
        ax.set_title("Catalog-DRO feasibility envelope")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"catalog_feasibility_envelope{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def regenerate(
    df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    resume_rejected_rows: list[dict],
) -> None:
    df = df[CATALOG_FEASIBILITY_ENVELOPE_COLUMNS] if not df.empty else pd.DataFrame(columns=CATALOG_FEASIBILITY_ENVELOPE_COLUMNS)
    df.to_csv(results_dir / "catalog_feasibility_envelope.csv", index=False)
    write_table(df, tables_dir)
    write_plot(df, figures_dir)
    feasible = df[df["meets_thresholds"].astype(bool)] if not df.empty else df
    extra = {
        "row_count": int(len(df)),
        "feasible_row_count": int(len(feasible)),
        "target_semantics": "catalog_dro_phase target generated from the configured source catalog, not a teacher-controlled target",
        "envelope_semantics": "small hard-catalog continuous-backend feasibility envelope; rows retain optimizer status and threshold flags even when not feasible",
        "method_types": sorted(str(value) for value in df["method_type"].dropna().unique().tolist()) if not df.empty else [],
        "control_bound_semantics": "reported control norms and bound violations are taken from bounded projected backend evaluations",
        "resume_semantics": "resume reuses only rows whose suite_case_id, settings_fingerprint, config_hash, and source_states_id match the current effective settings",
        "resume_rejected_rows": resume_rejected_rows,
        "limitations": [
            "Nominal-only rows with selected_outages=0 do not demonstrate missed-thrust robustness.",
            "For selected_outages=0, all_mask_worst_error is diagnostic under masked nominal controls and can be much worse than the nominal terminal error.",
            "Constant-control Hermite-Simpson rows hold controls fixed over each segment and do not optimize independent midpoint controls.",
            "This envelope is intentionally small and is not integrated into the manuscript until inspected by the main session.",
        ],
        "rows": df.to_dict(orient="records"),
    }
    write_metadata(results_dir / "catalog_feasibility_envelope_metadata.json", command, config, extra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hard catalog-DRO continuous-backend feasibility envelope.")
    parser.add_argument("--config", type=Path, default=Path("configs/catalog_feasibility_envelope.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results_dir, figures_dir, tables_dir = output_directories(Path.cwd(), config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "catalog_feasibility_envelope.csv"
    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    expected = _expected_index(config, args, cases)

    if args.resume:
        df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
        df = _refresh_reused_row_semantics(df, cases)
    else:
        df = pd.DataFrame(columns=CATALOG_FEASIBILITY_ENVELOPE_COLUMNS)
        resume_rejected_rows = []
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist()) if "settings_fingerprint" in df else set()
    command = " ".join(sys.argv)

    for case in cases:
        expected_row = expected[str(case["suite_case_id"])]
        if expected_row["settings_fingerprint"] in completed:
            continue
        row = run_case(config, args, case, expected_row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        regenerate(df, results_dir, figures_dir, tables_dir, config, command, resume_rejected_rows)
        print(
            f"case {case['suite_case_id']} method={row['method']} amax={row['amax']}: "
            f"nominal={row['nominal_error']:.6f}, selected={row['selected_worst_error']:.6f}, "
            f"all_mask={row['all_mask_worst_error']:.6f}, met={row['meets_thresholds']}, "
            f"runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    regenerate(df, results_dir, figures_dir, tables_dir, config, command, resume_rejected_rows)
    if not df.empty:
        best = df.sort_values(["meets_thresholds", "nominal_error", "all_mask_worst_error"], ascending=[False, True, True]).iloc[0]
        print(
            "best catalog feasibility envelope case: "
            f"case={best['suite_case_id']}, method={best['method']}, nominal={best['nominal_error']:.6f}, "
            f"selected={best['selected_worst_error']:.6f}, all_mask={best['all_mask_worst_error']:.6f}, "
            f"met={bool(best['meets_thresholds'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
