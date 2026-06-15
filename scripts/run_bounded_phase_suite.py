from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import output_directories
from qlt.multiple_shooting import _settings_fingerprint, run_case, settings_values_for_case


PHASE_SUITE_COLUMNS = [
    "suite_case_id",
    "case_type",
    "target_mode",
    "target_generation",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "selected_outages",
    "min_recovery_segments",
    "max_nfev",
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
    "nfev",
    "runtime_seconds",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "selected_branch_semantics",
    "all_mask_diagnostic_semantics",
    "optimizer_success",
    "multiple_shooting_success",
    "accepted_candidate",
    "node_initialization",
    "solver_mode",
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _outage_count(segments: int, block_lengths: list[int]) -> int:
    total = 0
    for length in block_lengths:
        if int(length) <= int(segments):
            total += int(segments) - int(length) + 1
    return total


def _suite_cases(config: dict) -> list[dict]:
    suite = config.get("suite", {}) or {}
    benchmark = config["benchmark"]
    refinement = config.get("refinement", {}) or {}
    phases = _as_list(suite.get("phase_times", benchmark.get("phase_time")), float)
    transfer_times = _as_list(suite.get("transfer_times", benchmark.get("transfer_time")), float)
    amax_values = _as_list(suite.get("amax", benchmark.get("amax")), float)
    segment_values = _as_list(suite.get("segments", benchmark.get("segments")), int)
    max_nfev_values = _as_list(suite.get("max_nfev", refinement.get("max_nfev", 40)), int)
    selected_values = _as_list(suite.get("selected_outages", refinement.get("selected_outages", 1)), int)
    min_recovery = int(suite.get("min_recovery_segments", refinement.get("outage_selection_min_recovery_segments", 1)))

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
    budget = float(suite.get("runtime_budget_seconds", 600))
    elapsed_fraction_limit = float(diagnostics.get("run_if_elapsed_fraction_below", 0.5))
    if budget > 0 and elapsed > budget * elapsed_fraction_limit:
        return []

    benchmark = config["benchmark"]
    refinement = config.get("refinement", {}) or {}
    phases = _as_list(diagnostics.get("phase_times", benchmark.get("phase_time")), float)
    transfer_times = _as_list(diagnostics.get("transfer_times", suite.get("transfer_times", benchmark.get("transfer_time"))), float)
    amax_values = _as_list(diagnostics.get("amax", suite.get("amax", benchmark.get("amax"))), float)
    segment_values = _as_list(diagnostics.get("segments", suite.get("segments", benchmark.get("segments"))), int)
    max_nfev_values = _as_list(diagnostics.get("max_nfev", suite.get("max_nfev", refinement.get("max_nfev", 40))), int)
    min_recovery = int(suite.get("min_recovery_segments", refinement.get("outage_selection_min_recovery_segments", 1)))
    block_lengths = [int(v) for v in config.get("outages", {}).get("block_lengths", [1])]

    cases = []
    for phase_time in phases[:1]:
        for transfer_time in transfer_times[:1]:
            for amax in amax_values[:1]:
                for segments in segment_values[:1]:
                    for max_nfev in max_nfev_values[:1]:
                        cases.append(
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
                        )
                        return cases
    return cases


def _args_for_case(args, case: dict) -> SimpleNamespace:
    return SimpleNamespace(
        config=args.config,
        source_states=args.source_states,
        selected_outages=int(case["selected_outages"]),
        min_recovery_segments=int(case["min_recovery_segments"]),
        node_initialization=args.node_initialization,
        node_initialization_blend=args.node_initialization_blend,
        nominal_first_homotopy=args.nominal_first_homotopy,
        nominal_first_nfev=args.nominal_first_nfev,
        initial_weight=args.initial_weight,
        defect_weight=args.defect_weight,
        terminal_weight=args.terminal_weight,
        branch_terminal_weight=args.branch_terminal_weight,
        branch_start_weight=args.branch_start_weight,
        control_weight=args.control_weight,
        smooth_weight=args.smooth_weight,
        solver_mode=args.solver_mode,
        warm_start_refinement=args.warm_start_refinement,
        warm_start_nfev=args.warm_start_nfev,
    )


def _config_for_phase(base_config: dict, phase_time: float) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("benchmark", {})["target_mode"] = "catalog_halo_phase_shift"
    config["benchmark"]["phase_time"] = float(phase_time)
    return config


def _case_payload(case: dict) -> dict:
    return {
        "case_type": case["case_type"],
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "selected_outages": int(case["selected_outages"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
        "max_nfev": int(case["max_nfev"]),
    }


def _expected_case_provenance(config: dict, args, case: dict) -> dict:
    case_config = _config_for_phase(config, float(case["phase_time"]))
    case_args = _args_for_case(args, case)
    settings = settings_values_for_case(
        case_config,
        float(case["transfer_time"]),
        float(case["amax"]),
        int(case["segments"]),
        int(case["max_nfev"]),
        case_args,
    )
    return {
        "settings_fingerprint": _settings_fingerprint(settings),
        "config_hash": str(settings["config_hash"]),
        "source_states_id": str(settings["source_states_id"]),
    }


def _expected_cases(config: dict, args, elapsed: float = 0.0) -> list[dict]:
    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    if not args.skip_diagnostic:
        cases.extend(_diagnostic_cases(config, elapsed))
    return cases


def _expected_index(config: dict, args, cases: list[dict]) -> dict[str, dict]:
    expected = {}
    for case in cases:
        payload = _case_payload(case)
        expected[_case_id(payload)] = {
            "case": case,
            "payload": payload,
            **_expected_case_provenance(config, args, case),
        }
    return expected


def _row_from_backend(case: dict, backend_row: dict, expected_provenance: dict) -> dict:
    thresholds = {
        "nominal_success": float(backend_row["nominal_threshold"]),
        "selected_recovery_success": float(backend_row["selected_worst_threshold"]),
    }
    payload = _case_payload(case)
    row = {
        **payload,
        "suite_case_id": _case_id(payload),
        "settings_fingerprint": str(backend_row.get("settings_fingerprint") or expected_provenance["settings_fingerprint"]),
        "config_hash": str(backend_row.get("config_hash") or expected_provenance["config_hash"]),
        "source_states_id": str(backend_row.get("source_states_id") or expected_provenance["source_states_id"]),
        "target_mode": "catalog_halo_phase_shift",
        "target_generation": "JPL initial_nrho_like_l2_southern_halo propagated ballistically in CR3BP for phase_time",
        "nominal_error": float(backend_row["nominal_error"]),
        "selected_worst_error": float(backend_row["selected_worst_error"]),
        "all_mask_worst_error": float(backend_row["all_mask_worst_error"]),
        "nominal_threshold": thresholds["nominal_success"],
        "selected_worst_threshold": thresholds["selected_recovery_success"],
        "thresholds": json.dumps(thresholds, sort_keys=True),
        "meets_nominal_threshold": bool(backend_row["meets_nominal_threshold"]),
        "meets_selected_worst_threshold": bool(backend_row["meets_selected_worst_threshold"]),
        "meets_thresholds": bool(backend_row["meets_thresholds"]),
        "control_max_norm": float(backend_row["control_max_norm"]),
        "control_bound_violation": float(backend_row["control_bound_violation"]),
        "nfev": int(backend_row["nfev"]),
        "runtime_seconds": float(backend_row["runtime_seconds"]),
        "selected_outage_indices": backend_row["selected_outage_indices"],
        "selected_outage_errors": backend_row["selected_outage_errors"],
        "all_outage_errors": backend_row["all_outage_errors"],
        "selected_branch_semantics": "selected outage masks are optimized with branch-recovery controls",
        "all_mask_diagnostic_semantics": "all masks are evaluated diagnostically; unselected masks use masked nominal controls, not optimized recovery branches",
        "optimizer_success": bool(backend_row["optimizer_success"]),
        "multiple_shooting_success": bool(backend_row["multiple_shooting_success"]),
        "accepted_candidate": str(backend_row["accepted_candidate"]),
        "node_initialization": str(backend_row["node_initialization"]),
        "solver_mode": str(backend_row["solver_mode"]),
        "message": str(backend_row["message"]),
    }
    return {column: row.get(column) for column in PHASE_SUITE_COLUMNS}


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=PHASE_SUITE_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in PHASE_SUITE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[PHASE_SUITE_COLUMNS]


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return df, []

    kept_rows: list[dict] = []
    rejected: list[dict] = []
    seen_fingerprints: set[str] = set()
    for row in df.to_dict(orient="records"):
        suite_case_id = str(row.get("suite_case_id", ""))
        expected_row = expected.get(suite_case_id)
        if expected_row is None:
            rejected.append({"suite_case_id": suite_case_id, "reason": "not in current requested case set"})
            continue

        row_fingerprint = row.get("settings_fingerprint")
        if pd.isna(row_fingerprint) or str(row_fingerprint) != expected_row["settings_fingerprint"]:
            rejected.append(
                {
                    "suite_case_id": suite_case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": None if pd.isna(row_fingerprint) else str(row_fingerprint),
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
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": str(row_fingerprint),
                }
            )
            continue

        if str(row_fingerprint) in seen_fingerprints:
            rejected.append(
                {
                    "suite_case_id": suite_case_id,
                    "reason": "duplicate compatible settings_fingerprint",
                    "found_settings_fingerprint": str(row_fingerprint),
                }
            )
            continue

        kept_rows.append(row)
        seen_fingerprints.add(str(row_fingerprint))

    return pd.DataFrame(kept_rows, columns=PHASE_SUITE_COLUMNS), rejected


def _write_table(df: pd.DataFrame, tables_dir: Path) -> None:
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
    (tables_dir / "bounded_phase_suite_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def _write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    primary = df[df["case_type"] == "selected_branch"].copy()
    if primary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    primary = primary.sort_values("phase_time")
    ax.plot(primary["phase_time"], primary["nominal_error"], marker="o", label="nominal")
    ax.plot(primary["phase_time"], primary["selected_worst_error"], marker="s", label="selected worst")
    ax.plot(primary["phase_time"], primary["all_mask_worst_error"], marker="^", label="all-mask diagnostic")
    ax.axhline(float(primary["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0, label="nominal threshold")
    ax.axhline(float(primary["selected_worst_threshold"].iloc[0]), color="0.25", linestyle=":", linewidth=1.0, label="selected threshold")
    ax.set_xlabel("Phase time")
    ax.set_ylabel("Normalized terminal error")
    ax.set_title("Bounded phase-shift suite")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"bounded_phase_suite{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def _write_metadata(
    df: pd.DataFrame,
    results_dir: Path,
    config: dict,
    command: str,
    skipped_diagnostic_reason: str | None,
    resume_rejected_rows: list[dict],
) -> None:
    feasible = df[df["meets_thresholds"].astype(bool)] if not df.empty else df
    metadata = {
        "command": command,
        "row_count": int(len(df)),
        "feasible_row_count": int(len(feasible)),
        "config": config,
        "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_worst_error <= selected_recovery_success",
        "target_semantics": "non-teacher catalog_halo_phase_shift; target is generated by ballistic CR3BP propagation of the JPL halo source state for each phase_time",
        "mask_semantics": "selected_worst_error uses optimized branch recovery only for selected outage masks; all_mask_worst_error evaluates every outage mask diagnostically, with unselected masks using masked nominal controls rather than optimized recovery branches",
        "resume_semantics": "resume reuses only rows whose settings_fingerprint, config_hash, and source_states_id match the current effective case settings; missing or mismatched provenance rows are rejected and recomputed before metadata is rewritten",
        "resume_rejected_rows": resume_rejected_rows,
        "skipped_diagnostic_reason": skipped_diagnostic_reason,
        "rows": df.to_dict(orient="records"),
    }
    (results_dir / "bounded_phase_suite_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _regenerate(
    df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    skipped_diagnostic_reason: str | None,
    resume_rejected_rows: list[dict],
) -> None:
    df.to_csv(results_dir / "bounded_phase_suite.csv", index=False)
    _write_table(df, tables_dir)
    _write_plot(df, figures_dir)
    _write_metadata(df, results_dir, config, command, skipped_diagnostic_reason, resume_rejected_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded non-teacher halo phase-shift multiple-shooting suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/bounded_phase_suite.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--node-initialization", choices=["rollout", "linear", "blend"], default=None)
    parser.add_argument("--node-initialization-blend", "--node-blend", dest="node_initialization_blend", type=float, default=None)
    parser.add_argument("--nominal-first-homotopy", "--nominal-first", dest="nominal_first_homotopy", action="store_true", default=None)
    parser.add_argument("--nominal-first-nfev", type=int, default=None)
    parser.add_argument("--initial-weight", type=float, default=10.0)
    parser.add_argument("--defect-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-start-weight", type=float, default=5.0)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--smooth-weight", type=float, default=0.012)
    parser.add_argument("--solver-mode", default=None)
    parser.add_argument("--warm-start-refinement", action="store_true")
    parser.add_argument("--warm-start-nfev", type=int, default=30)
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
    csv_path = results_dir / "bounded_phase_suite.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    possible_expected = _expected_index(config, args, _expected_cases(config, args, elapsed=0.0))

    resume_rejected_rows: list[dict] = []
    if args.resume:
        loaded = _load_existing(csv_path)
        df, resume_rejected_rows = _compatible_existing_rows(loaded, possible_expected)
    else:
        df = pd.DataFrame(columns=PHASE_SUITE_COLUMNS)
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())

    start = time.perf_counter()
    skipped_diagnostic_reason = None
    command = " ".join(sys.argv)

    for case in cases:
        expected_provenance = _expected_case_provenance(config, args, case)
        if expected_provenance["settings_fingerprint"] in completed:
            continue
        case_config = _config_for_phase(config, float(case["phase_time"]))
        backend_row = run_case(
            case_config,
            float(case["transfer_time"]),
            float(case["amax"]),
            int(case["segments"]),
            int(case["max_nfev"]),
            _args_for_case(args, case),
        )
        row = _row_from_backend(case, backend_row, expected_provenance)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        _regenerate(df, results_dir, figures_dir, tables_dir, config, command, skipped_diagnostic_reason, resume_rejected_rows)
        print(
            f"case phase={case['phase_time']} tf={case['transfer_time']} selected={case['selected_outages']}: "
            f"nominal={row['nominal_error']:.4f}, selected={row['selected_worst_error']:.4f}, "
            f"all={row['all_mask_worst_error']:.4f}, met={row['meets_thresholds']}, "
            f"runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    diagnostic_cases = [] if args.skip_diagnostic else _diagnostic_cases(config, time.perf_counter() - start)
    if args.skip_diagnostic:
        skipped_diagnostic_reason = "disabled by --skip-diagnostic"
    elif not diagnostic_cases and bool((config.get("suite", {}) or {}).get("diagnostics", {}).get("include_all_outage_selected", False)):
        has_compatible_diagnostic = bool((not df.empty) and (df["case_type"] == "all_outage_selected_diagnostic").any())
        skipped_diagnostic_reason = (
            "elapsed runtime exceeded configured diagnostic gate; compatible existing diagnostic rows retained"
            if has_compatible_diagnostic
            else "elapsed runtime exceeded configured diagnostic gate"
        )

    for case in diagnostic_cases:
        expected_provenance = _expected_case_provenance(config, args, case)
        if expected_provenance["settings_fingerprint"] in completed:
            continue
        case_config = _config_for_phase(config, float(case["phase_time"]))
        backend_row = run_case(
            case_config,
            float(case["transfer_time"]),
            float(case["amax"]),
            int(case["segments"]),
            int(case["max_nfev"]),
            _args_for_case(args, case),
        )
        row = _row_from_backend(case, backend_row, expected_provenance)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        _regenerate(df, results_dir, figures_dir, tables_dir, config, command, skipped_diagnostic_reason, resume_rejected_rows)
        print(
            f"diagnostic phase={case['phase_time']} selected={case['selected_outages']}: "
            f"nominal={row['nominal_error']:.4f}, selected={row['selected_worst_error']:.4f}, "
            f"all={row['all_mask_worst_error']:.4f}, met={row['meets_thresholds']}, "
            f"runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    _regenerate(df, results_dir, figures_dir, tables_dir, config, command, skipped_diagnostic_reason, resume_rejected_rows)
    if not df.empty:
        best = df.sort_values(["meets_thresholds", "selected_worst_error", "nominal_error"], ascending=[False, True, True]).iloc[0]
        print(
            "best phase-suite case: "
            f"phase={best['phase_time']}, selected={best['selected_outages']}, "
            f"nominal={best['nominal_error']:.6f}, selected_worst={best['selected_worst_error']:.6f}, "
            f"all_mask={best['all_mask_worst_error']:.6f}, met={bool(best['meets_thresholds'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
