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
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.direct_collocation import config_hash, file_identity, run_direct_collocation_baseline, settings_fingerprint
from qlt.experiment import load_configured_states, make_objective_config, output_directories
from qlt.objective import outage_masks
from qlt.reporting import write_metadata


DIRECT_COLLOCATION_COLUMNS = [
    "suite_case_id",
    "case_type",
    "target_mode",
    "target_generation",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "max_nfev",
    "selected_outages",
    "min_recovery_segments",
    "settings_fingerprint",
    "config_hash",
    "source_states_id",
    "method_type",
    "collocation_method",
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
    "direct_collocation_success",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "control_bound_semantics",
    "message",
]


def _as_list(value, cast):
    if value is None:
        return []
    if isinstance(value, str):
        return [cast(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _case_id(payload: dict) -> str:
    return settings_fingerprint(payload)[:16]


def _outage_count(segments: int, block_lengths: list[int]) -> int:
    total = 0
    for length in block_lengths:
        if int(length) <= int(segments):
            total += int(segments) - int(length) + 1
    return total


def _case_payload(case: dict) -> dict:
    return {
        "case_type": str(case["case_type"]),
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "max_nfev": int(case["max_nfev"]),
        "selected_outages": int(case["selected_outages"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
    }


def _suite_cases(config: dict) -> list[dict]:
    suite = config.get("suite", {}) or {}
    benchmark = config["benchmark"]
    direct = config.get("direct_collocation", {}) or {}
    phases = _as_list(suite.get("phase_times", benchmark.get("phase_time")), float)
    transfer_times = _as_list(suite.get("transfer_times", benchmark.get("transfer_time")), float)
    amax_values = _as_list(suite.get("amax", benchmark.get("amax")), float)
    segment_values = _as_list(suite.get("segments", benchmark.get("segments")), int)
    max_nfev_values = _as_list(suite.get("max_nfev", direct.get("max_nfev", 30)), int)
    selected_values = _as_list(suite.get("selected_outages", direct.get("selected_outages", 1)), int)
    min_recovery = int(suite.get("min_recovery_segments", direct.get("min_recovery_segments", 1)))
    cases = []
    for phase_time in phases:
        for transfer_time in transfer_times:
            for amax in amax_values:
                for segments in segment_values:
                    for max_nfev in max_nfev_values:
                        for selected_outages in selected_values:
                            cases.append(
                                {
                                    "case_type": "selected_branch",
                                    "phase_time": phase_time,
                                    "transfer_time": transfer_time,
                                    "amax": amax,
                                    "segments": segments,
                                    "max_nfev": max_nfev,
                                    "selected_outages": selected_outages,
                                    "min_recovery_segments": min_recovery,
                                }
                            )
    return cases


def _diagnostic_cases(config: dict, elapsed: float) -> list[dict]:
    suite = config.get("suite", {}) or {}
    diagnostics = suite.get("diagnostics", {}) or {}
    if not bool(diagnostics.get("include_all_outage_selected", False)):
        return []
    budget = float(suite.get("runtime_budget_seconds", 600.0))
    elapsed_fraction_limit = float(diagnostics.get("run_if_elapsed_fraction_below", 0.5))
    if budget > 0 and elapsed > budget * elapsed_fraction_limit:
        return []
    benchmark = config["benchmark"]
    direct = config.get("direct_collocation", {}) or {}
    phases = _as_list(diagnostics.get("phase_times", suite.get("phase_times", benchmark.get("phase_time"))), float)
    transfer_times = _as_list(diagnostics.get("transfer_times", suite.get("transfer_times", benchmark.get("transfer_time"))), float)
    amax_values = _as_list(diagnostics.get("amax", suite.get("amax", benchmark.get("amax"))), float)
    segment_values = _as_list(diagnostics.get("segments", suite.get("segments", benchmark.get("segments"))), int)
    max_nfev_values = _as_list(diagnostics.get("max_nfev", suite.get("max_nfev", direct.get("max_nfev", 30))), int)
    min_recovery = int(suite.get("min_recovery_segments", direct.get("min_recovery_segments", 1)))
    block_lengths = [int(v) for v in config.get("outages", {}).get("block_lengths", [1])]
    for phase_time in phases[:1]:
        for transfer_time in transfer_times[:1]:
            for amax in amax_values[:1]:
                for segments in segment_values[:1]:
                    for max_nfev in max_nfev_values[:1]:
                        return [
                            {
                                "case_type": "all_outage_selected_diagnostic",
                                "phase_time": phase_time,
                                "transfer_time": transfer_time,
                                "amax": amax,
                                "segments": segments,
                                "max_nfev": max_nfev,
                                "selected_outages": _outage_count(segments, block_lengths),
                                "min_recovery_segments": min_recovery,
                            }
                        ]
    return []


def _config_for_case(base_config: dict, case: dict) -> dict:
    config = copy.deepcopy(base_config)
    b = config.setdefault("benchmark", {})
    b["target_mode"] = "catalog_halo_phase_shift"
    b["phase_time"] = float(case["phase_time"])
    b["transfer_time"] = float(case["transfer_time"])
    b["amax"] = float(case["amax"])
    b["segments"] = int(case["segments"])
    return config


def _effective_settings(config: dict, args, case: dict) -> dict:
    case_config = _config_for_case(config, case)
    direct = copy.deepcopy(case_config.get("direct_collocation", {}) or {})
    direct["max_nfev"] = int(case["max_nfev"])
    direct["selected_outages"] = int(case["selected_outages"])
    direct["min_recovery_segments"] = int(case["min_recovery_segments"])
    return {
        "suite": "direct_collocation_baseline",
        **_case_payload(case),
        "method": str(direct.get("method", "trapezoidal")),
        "direct_collocation": direct,
        "thresholds": copy.deepcopy(case_config["objective"]["thresholds"]),
        "benchmark": {
            "target_mode": "catalog_halo_phase_shift",
            "substeps_per_segment": int(case_config["benchmark"]["substeps_per_segment"]),
        },
        "outages": copy.deepcopy(case_config.get("outages", {})),
        "config_hash": config_hash(case_config),
        "source_states_id": file_identity(args.source_states),
    }


def _expected_index(config: dict, args, cases: list[dict]) -> dict[str, dict]:
    expected = {}
    for case in cases:
        payload = _case_payload(case)
        settings = _effective_settings(config, args, case)
        expected[_case_id(payload)] = {
            "case": case,
            "settings_fingerprint": settings_fingerprint(settings),
            "config_hash": settings["config_hash"],
            "source_states_id": settings["source_states_id"],
        }
    return expected


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=DIRECT_COLLOCATION_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in DIRECT_COLLOCATION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[DIRECT_COLLOCATION_COLUMNS]


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return pd.DataFrame(columns=DIRECT_COLLOCATION_COLUMNS), []
    kept = []
    rejected = []
    seen = set()
    for row in df.to_dict(orient="records"):
        suite_case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(suite_case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": suite_case_id, "reason": "not in current requested case set"})
            continue
        row_fp = row.get("settings_fingerprint")
        if pd.isna(row_fp) or str(row_fp) != expected_row["settings_fingerprint"]:
            rejected.append(
                {
                    "suite_case_id": suite_case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": None if pd.isna(row_fp) else str(row_fp),
                }
            )
            continue
        mismatched = [
            key
            for key in ("config_hash", "source_states_id")
            if pd.isna(row.get(key)) or str(row.get(key)) != expected_row[key]
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
        if str(row_fp) in seen:
            rejected.append({"suite_case_id": suite_case_id, "reason": "duplicate compatible settings_fingerprint"})
            continue
        kept.append(row)
        seen.add(str(row_fp))
    return pd.DataFrame(kept, columns=DIRECT_COLLOCATION_COLUMNS), rejected


def _observed_primary_runtime(df: pd.DataFrame) -> float:
    if df.empty or "runtime_seconds" not in df.columns or "case_type" not in df.columns:
        return 0.0
    primary = df[df["case_type"] == "selected_branch"]
    if primary.empty:
        return 0.0
    return float(pd.to_numeric(primary["runtime_seconds"], errors="coerce").fillna(0.0).sum())


def _row_from_result(case: dict, result: dict, expected: dict) -> dict:
    thresholds = {
        "nominal_success": float(result["nominal_threshold"]),
        "selected_recovery_success": float(result["selected_worst_threshold"]),
    }
    row = {
        **_case_payload(case),
        "suite_case_id": _case_id(_case_payload(case)),
        "settings_fingerprint": expected["settings_fingerprint"],
        "config_hash": expected["config_hash"],
        "source_states_id": expected["source_states_id"],
        "target_mode": "catalog_halo_phase_shift",
        "target_generation": "JPL initial_nrho_like_l2_southern_halo propagated ballistically in CR3BP for phase_time",
        "method_type": result["method_type"],
        "collocation_method": result["collocation_method"],
        "nominal_error": float(result["nominal_error"]),
        "selected_worst_error": float(result["selected_worst_error"]),
        "all_mask_worst_error": float(result["all_mask_worst_error"]),
        "nominal_threshold": thresholds["nominal_success"],
        "selected_worst_threshold": thresholds["selected_recovery_success"],
        "thresholds": json.dumps(thresholds, sort_keys=True),
        "meets_nominal_threshold": bool(result["nominal_error"] <= thresholds["nominal_success"]),
        "meets_selected_worst_threshold": bool(result["selected_worst_error"] <= thresholds["selected_recovery_success"]),
        "meets_thresholds": bool(
            result["nominal_error"] <= thresholds["nominal_success"]
            and result["selected_worst_error"] <= thresholds["selected_recovery_success"]
        ),
        "control_max_norm": float(result["control_max_norm"]),
        "control_bound_violation": float(result["control_bound_violation"]),
        "nominal_fuel": float(result["nominal_fuel"]),
        "recovery_fuel_mean": float(result["recovery_fuel_mean"]),
        "recovery_fuel_max": float(result["recovery_fuel_max"]),
        "cost": float(result["cost"]),
        "optimality": float(result["optimality"]),
        "nfev": int(result["nfev"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "selected_outage_indices": json.dumps(result["selected_outage_indices"]),
        "selected_outage_errors": json.dumps(result["selected_outage_errors"]),
        "all_outage_errors": json.dumps(result["all_outage_errors"]),
        "optimizer_success": bool(result["optimizer_success"]),
        "direct_collocation_success": bool(result["success"]),
        "selected_branch_semantics": result["selected_branch_semantics"],
        "all_mask_diagnostic_semantics": result["all_mask_diagnostic_semantics"],
        "control_bound_semantics": result["control_bound_semantics"],
        "message": str(result["message"]),
    }
    return {column: row.get(column) for column in DIRECT_COLLOCATION_COLUMNS}


def run_case(config: dict, args, case: dict, expected: dict) -> dict:
    case_config = _config_for_case(config, case)
    states = load_configured_states(Path.cwd(), case_config, args.source_states)
    cfg = make_objective_config(case_config, states.mu)
    masks = outage_masks(cfg.n_segments, cfg.outage_lengths)
    thresholds = case_config["objective"]["thresholds"]
    direct_cfg = copy.deepcopy(case_config.get("direct_collocation", {}) or {})
    direct_cfg["max_nfev"] = int(case["max_nfev"])
    direct_cfg["selected_outages"] = int(case["selected_outages"])
    direct_cfg["min_recovery_segments"] = int(case["min_recovery_segments"])
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
    result["nominal_threshold"] = float(thresholds["nominal_success"])
    result["selected_worst_threshold"] = float(thresholds["robust_success"])
    return _row_from_result(case, result, expected)


def write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    if df.empty:
        return
    table = df.sort_values(["case_type", "phase_time"])[
        [
            "case_type",
            "phase_time",
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
        "Phase time",
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
    (tables_dir / "direct_collocation_baseline_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    primary = df[df["case_type"] == "selected_branch"].copy()
    if primary.empty:
        return
    primary = primary.sort_values("phase_time")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(primary["phase_time"], primary["nominal_error"], marker="o", label="nominal")
    ax.plot(primary["phase_time"], primary["selected_worst_error"], marker="s", label="selected branch")
    ax.plot(primary["phase_time"], primary["all_mask_worst_error"], marker="^", label="all-mask diagnostic")
    ax.axhline(float(primary["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0, label="nominal threshold")
    ax.axhline(float(primary["selected_worst_threshold"].iloc[0]), color="0.25", linestyle=":", linewidth=1.0, label="selected threshold")
    ax.set_xlabel("Phase time")
    ax.set_ylabel("Normalized terminal error")
    ax.set_title("Trapezoidal direct-collocation baseline")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"direct_collocation_baseline{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def regenerate(
    df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    resume_rejected_rows: list[dict],
    skipped_diagnostic_reason: str | None,
) -> None:
    df = df[DIRECT_COLLOCATION_COLUMNS] if not df.empty else pd.DataFrame(columns=DIRECT_COLLOCATION_COLUMNS)
    df.to_csv(results_dir / "direct_collocation_baseline.csv", index=False)
    write_table(df, tables_dir)
    write_plot(df, figures_dir)
    feasible = df[df["meets_thresholds"].astype(bool)] if not df.empty else df
    extra = {
        "row_count": int(len(df)),
        "feasible_row_count": int(len(feasible)),
        "method_type": "bounded_projected_trapezoidal_direct_collocation",
        "target_semantics": "non-teacher catalog_halo_phase_shift target generated by zero-thrust CR3BP phase propagation of the JPL halo source state",
        "selected_branch_vs_all_mask_semantics": "selected_branch rows optimize nominal plus selected outage recovery branches; all_mask_worst_error is a diagnostic over every outage mask, with unselected masks evaluated using masked nominal controls",
        "control_bound_semantics": "reported and evaluation controls are projected to the Euclidean ball ||u_i|| <= amax; control_bound_violation is computed after projection",
        "resume_semantics": "resume reuses only rows whose suite_case_id, settings_fingerprint, config_hash, and source_states_id match the current effective settings",
        "resume_rejected_rows": resume_rejected_rows,
        "skipped_diagnostic_reason": skipped_diagnostic_reason,
        "limitations": [
            "This is a compact trapezoidal direct-collocation baseline, not a high-fidelity flight trajectory design.",
            "Hermite-Simpson was not used in this compact run to keep the default phase suite short and stable.",
            "Only selected outage branches are optimized; all-mask results are diagnostic unless case_type is all_outage_selected_diagnostic.",
            "The optimizer may fail or stop at max_nfev; such rows are retained with optimizer_success and direct_collocation_success flags.",
        ],
        "rows": df.to_dict(orient="records"),
    }
    write_metadata(results_dir / "direct_collocation_baseline_metadata.json", command, config, extra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact bounded direct-collocation baseline for the non-teacher phase-shift suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/direct_collocation_baseline.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results_dir, figures_dir, tables_dir = output_directories(Path.cwd(), config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "direct_collocation_baseline.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    possible_cases = cases + ([] if args.skip_diagnostic else _diagnostic_cases(config, elapsed=0.0))
    expected = _expected_index(config, args, possible_cases)

    if args.resume:
        df, resume_rejected_rows = _compatible_existing_rows(_load_existing(csv_path), expected)
    else:
        df = pd.DataFrame(columns=DIRECT_COLLOCATION_COLUMNS)
        resume_rejected_rows = []
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())
    command = " ".join(sys.argv)
    skipped_diagnostic_reason = None
    start = time.perf_counter()

    for case in cases:
        case_id = _case_id(_case_payload(case))
        expected_row = expected[case_id]
        if expected_row["settings_fingerprint"] in completed:
            continue
        row = run_case(config, args, case, expected_row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        regenerate(df, results_dir, figures_dir, tables_dir, config, command, resume_rejected_rows, skipped_diagnostic_reason)
        print(
            f"case phase={case['phase_time']} tf={case['transfer_time']} selected={case['selected_outages']}: "
            f"nominal={row['nominal_error']:.4f}, selected={row['selected_worst_error']:.4f}, "
            f"all={row['all_mask_worst_error']:.4f}, met={row['meets_thresholds']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    diagnostic_elapsed = _observed_primary_runtime(df) + float(time.perf_counter() - start)
    diagnostic_cases = [] if args.skip_diagnostic else _diagnostic_cases(config, diagnostic_elapsed)
    if args.skip_diagnostic:
        skipped_diagnostic_reason = "disabled by --skip-diagnostic"
    elif not diagnostic_cases and bool((config.get("suite", {}) or {}).get("diagnostics", {}).get("include_all_outage_selected", False)):
        skipped_diagnostic_reason = "elapsed runtime exceeded configured diagnostic gate"

    for case in diagnostic_cases:
        case_id = _case_id(_case_payload(case))
        expected_row = expected[case_id]
        if expected_row["settings_fingerprint"] in completed:
            continue
        row = run_case(config, args, case, expected_row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        regenerate(df, results_dir, figures_dir, tables_dir, config, command, resume_rejected_rows, skipped_diagnostic_reason)
        print(
            f"diagnostic phase={case['phase_time']} selected={case['selected_outages']}: "
            f"nominal={row['nominal_error']:.4f}, selected={row['selected_worst_error']:.4f}, "
            f"all={row['all_mask_worst_error']:.4f}, met={row['meets_thresholds']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    regenerate(df, results_dir, figures_dir, tables_dir, config, command, resume_rejected_rows, skipped_diagnostic_reason)
    if not df.empty:
        best = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True]).iloc[0]
        print(
            "best direct-collocation case: "
            f"phase={best['phase_time']}, selected={best['selected_outages']}, "
            f"nominal={best['nominal_error']:.6f}, selected_worst={best['selected_worst_error']:.6f}, "
            f"all_mask={best['all_mask_worst_error']:.6f}, met={bool(best['meets_thresholds'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
