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
import qlt.multiple_shooting as multiple_shooting
from qlt.reporting import write_json


ROBUST_MARGIN_COLUMNS = [
    "case_id",
    "case_group",
    "group_purpose",
    "target_mode",
    "target_generation",
    "phase_time",
    "transfer_time",
    "amax",
    "segments",
    "max_nfev",
    "selected_outages",
    "outage_count",
    "selected_all_outages",
    "min_recovery_segments",
    "node_initialization",
    "node_initialization_blend",
    "nominal_first_homotopy",
    "nominal_first_max_nfev",
    "initial_weight",
    "defect_weight",
    "terminal_weight",
    "branch_terminal_weight",
    "branch_start_weight",
    "control_weight",
    "smooth_weight",
    "solver_mode",
    "warm_start_refinement",
    "warm_start_max_nfev",
    "nominal_error",
    "selected_worst_error",
    "all_mask_worst_error",
    "thresholds",
    "nominal_threshold",
    "selected_worst_threshold",
    "meets_nominal_threshold",
    "meets_selected_worst_threshold",
    "meets_thresholds",
    "nfev",
    "runtime_seconds",
    "optimizer_success",
    "multiple_shooting_success",
    "accepted_candidate",
    "control_max_norm",
    "control_bound_violation",
    "selected_outage_indices",
    "selected_outage_errors",
    "all_outage_errors",
    "settings_fingerprint",
    "backend_settings_fingerprint",
    "config_hash",
    "source_states_id",
    "message",
]


TARGET_GENERATION = (
    "non-teacher catalog_halo_phase_shift target from the JPL initial_nrho_like_l2_southern_halo "
    "source state propagated ballistically in normalized CR3BP for the configured phase_time"
)


def _as_list(value, cast) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [cast(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _case_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _settings_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _outage_count(segments: int, block_lengths: list[int]) -> int:
    count = 0
    for length in block_lengths:
        if int(length) <= int(segments):
            count += int(segments) - int(length) + 1
    return count


def _selected_outage_values(raw_value, segments: int, block_lengths: list[int]) -> list[int]:
    if isinstance(raw_value, str) and raw_value.strip().lower() in {
        "all_single_outage_masks",
        "all_single_outages",
        "all",
    }:
        if sorted(int(length) for length in block_lengths) != [1]:
            raise ValueError("all_single_outage_masks requires outages.block_lengths: [1]")
        return [_outage_count(segments, block_lengths)]
    return _as_list(raw_value, int)


def _suite_cases(config: dict) -> list[dict]:
    suite = config.get("suite", {}) or {}
    groups = suite.get("groups", {}) or {}
    benchmark = config["benchmark"]
    refinement = config.get("refinement", {}) or {}
    block_lengths = [int(value) for value in config.get("outages", {}).get("block_lengths", [1])]

    cases: list[dict] = []
    for group_name, group in groups.items():
        phase_times = _as_list(group.get("phase_times", benchmark.get("phase_time")), float)
        transfer_times = _as_list(group.get("transfer_times", benchmark.get("transfer_time")), float)
        amax_values = _as_list(group.get("amax", benchmark.get("amax")), float)
        segment_values = _as_list(group.get("segments", benchmark.get("segments")), int)
        max_nfev_values = _as_list(group.get("max_nfev", refinement.get("max_nfev", 80)), int)
        min_recovery = int(group.get("min_recovery_segments", refinement.get("outage_selection_min_recovery_segments", 1)))
        purpose = str(group.get("purpose", ""))

        for phase_time in phase_times:
            for transfer_time in transfer_times:
                for amax in amax_values:
                    for segments in segment_values:
                        outage_total = _outage_count(segments, block_lengths)
                        selected_values = _selected_outage_values(
                            group.get("selected_outages", refinement.get("selected_outages", 1)),
                            segments,
                            block_lengths,
                        )
                        for max_nfev in max_nfev_values:
                            for selected_outages in selected_values:
                                cases.append(
                                    {
                                        "case_group": str(group_name),
                                        "group_purpose": purpose,
                                        "phase_time": float(phase_time),
                                        "transfer_time": float(transfer_time),
                                        "amax": float(amax),
                                        "segments": int(segments),
                                        "max_nfev": int(max_nfev),
                                        "selected_outages": int(selected_outages),
                                        "outage_count": int(outage_total),
                                        "selected_all_outages": int(selected_outages) == int(outage_total),
                                        "min_recovery_segments": int(min_recovery),
                                    }
                                )
    return cases


def _case_config(config: dict, case: dict) -> dict:
    case_config = copy.deepcopy(config)
    case_config.setdefault("benchmark", {})["target_mode"] = "catalog_halo_phase_shift"
    case_config["benchmark"]["phase_time"] = float(case["phase_time"])
    return case_config


def _resolved_residual_weights(config: dict, args) -> dict[str, float]:
    configured = ((config.get("suite", {}) or {}).get("residual_weights", {}) or {})
    weights = {}
    for key, default in multiple_shooting.RESIDUAL_WEIGHT_DEFAULTS.items():
        value = getattr(args, key, None)
        if value is None:
            value = configured.get(key, default)
        weights[key] = float(value)
    return weights


def _args_for_case(config: dict, args, case: dict) -> SimpleNamespace:
    weights = _resolved_residual_weights(config, args)
    return SimpleNamespace(
        config=args.config,
        source_states=args.source_states,
        selected_outages=int(case["selected_outages"]),
        min_recovery_segments=int(case["min_recovery_segments"]),
        node_initialization=args.node_initialization,
        node_initialization_blend=args.node_initialization_blend,
        nominal_first_homotopy=args.nominal_first_homotopy,
        nominal_first=args.nominal_first_homotopy,
        nominal_first_nfev=args.nominal_first_nfev,
        **weights,
        solver_mode=args.solver_mode,
        warm_start_refinement=bool(args.warm_start_refinement),
        warm_start_nfev=int(args.warm_start_nfev),
    )


def _case_payload(case: dict) -> dict:
    return {
        "case_group": str(case["case_group"]),
        "phase_time": float(case["phase_time"]),
        "transfer_time": float(case["transfer_time"]),
        "amax": float(case["amax"]),
        "segments": int(case["segments"]),
        "max_nfev": int(case["max_nfev"]),
        "selected_outages": int(case["selected_outages"]),
        "outage_count": int(case["outage_count"]),
        "selected_all_outages": bool(case["selected_all_outages"]),
        "min_recovery_segments": int(case["min_recovery_segments"]),
    }


def _expected_case(config: dict, args, case: dict) -> dict:
    case_config = _case_config(config, case)
    case_args = _args_for_case(config, args, case)
    backend_settings = multiple_shooting.settings_values_for_case(
        case_config,
        float(case["transfer_time"]),
        float(case["amax"]),
        int(case["segments"]),
        int(case["max_nfev"]),
        case_args,
    )
    thresholds = {
        "nominal_success": float(config["objective"]["thresholds"]["nominal_success"]),
        "robust_success": float(config["objective"]["thresholds"]["robust_success"]),
    }
    payload = {
        **_case_payload(case),
        "target_mode": "catalog_halo_phase_shift",
        "thresholds": thresholds,
        "node_initialization": backend_settings["node_initialization"],
        "node_initialization_blend": float(backend_settings["node_initialization_blend"]),
        "nominal_first_homotopy": bool(backend_settings["nominal_first_homotopy"]),
        "nominal_first_max_nfev": int(backend_settings["nominal_first_max_nfev"]),
        "residual_weights": {
            key: float(backend_settings[key])
            for key in multiple_shooting.RESIDUAL_WEIGHT_DEFAULTS
        },
        "solver_mode": str(backend_settings["solver_mode"]),
        "warm_start_refinement": bool(backend_settings["warm_start_refinement"]),
        "warm_start_max_nfev": int(backend_settings["warm_start_max_nfev"]),
        "config_hash": str(backend_settings["config_hash"]),
        "source_states_id": str(backend_settings["source_states_id"]),
    }
    return {
        "case": case,
        "case_id": _case_hash(_case_payload(case)),
        "settings_fingerprint": _settings_hash(payload),
        "backend_settings_fingerprint": multiple_shooting._settings_fingerprint(backend_settings),
        "config_hash": str(backend_settings["config_hash"]),
        "source_states_id": str(backend_settings["source_states_id"]),
        "payload": payload,
    }


def _expected_index(config: dict, args, cases: list[dict]) -> dict[str, dict]:
    return {
        expected["case_id"]: expected
        for expected in (_expected_case(config, args, case) for case in cases)
    }


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=ROBUST_MARGIN_COLUMNS)
    df = pd.read_csv(csv_path)
    for column in ROBUST_MARGIN_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[ROBUST_MARGIN_COLUMNS]


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _compatible_existing_rows(df: pd.DataFrame, expected: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return df, []

    kept_rows: list[dict] = []
    rejected: list[dict] = []
    seen_fingerprints: set[str] = set()
    for row in df.to_dict(orient="records"):
        case_id = str(row.get("case_id", ""))
        expected_row = expected.get(case_id)
        if expected_row is None:
            rejected.append({"case_id": case_id, "reason": "case is not in the current requested suite"})
            continue

        row_fingerprint = row.get("settings_fingerprint")
        if _is_missing(row_fingerprint) or str(row_fingerprint) != expected_row["settings_fingerprint"]:
            rejected.append(
                {
                    "case_id": case_id,
                    "reason": "settings_fingerprint missing or mismatched",
                    "expected_settings_fingerprint": expected_row["settings_fingerprint"],
                    "found_settings_fingerprint": None if _is_missing(row_fingerprint) else str(row_fingerprint),
                }
            )
            continue

        mismatched = [
            key
            for key in ("config_hash", "source_states_id")
            if _is_missing(row.get(key)) or str(row.get(key)) != expected_row[key]
        ]
        if mismatched:
            rejected.append(
                {
                    "case_id": case_id,
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
                    "case_id": case_id,
                    "reason": "duplicate compatible settings_fingerprint",
                    "found_settings_fingerprint": str(row_fingerprint),
                }
            )
            continue

        kept_rows.append(row)
        seen_fingerprints.add(str(row_fingerprint))

    return pd.DataFrame(kept_rows, columns=ROBUST_MARGIN_COLUMNS), rejected


def _row_from_backend(case: dict, backend_row: dict, expected: dict) -> dict:
    thresholds = {
        "nominal_success": float(backend_row["nominal_threshold"]),
        "robust_success": float(backend_row["selected_worst_threshold"]),
    }
    weights = {
        key: float(backend_row[key])
        for key in multiple_shooting.RESIDUAL_WEIGHT_DEFAULTS
    }
    row = {
        **_case_payload(case),
        "case_id": expected["case_id"],
        "group_purpose": str(case["group_purpose"]),
        "target_mode": "catalog_halo_phase_shift",
        "target_generation": TARGET_GENERATION,
        "node_initialization": str(backend_row["node_initialization"]),
        "node_initialization_blend": float(backend_row["node_initialization_blend"]),
        "nominal_first_homotopy": bool(backend_row["nominal_first_homotopy"]),
        "nominal_first_max_nfev": int(backend_row["nominal_first_max_nfev"]),
        **weights,
        "solver_mode": str(backend_row["solver_mode"]),
        "warm_start_refinement": bool(backend_row["warm_start_refinement"]),
        "warm_start_max_nfev": int(backend_row["warm_start_max_nfev"]),
        "nominal_error": float(backend_row["nominal_error"]),
        "selected_worst_error": float(backend_row["selected_worst_error"]),
        "all_mask_worst_error": float(backend_row["all_mask_worst_error"]),
        "thresholds": json.dumps(thresholds, sort_keys=True),
        "nominal_threshold": thresholds["nominal_success"],
        "selected_worst_threshold": thresholds["robust_success"],
        "meets_nominal_threshold": bool(backend_row["meets_nominal_threshold"]),
        "meets_selected_worst_threshold": bool(backend_row["meets_selected_worst_threshold"]),
        "meets_thresholds": bool(backend_row["meets_thresholds"]),
        "nfev": int(backend_row["nfev"]),
        "runtime_seconds": float(backend_row["runtime_seconds"]),
        "optimizer_success": bool(backend_row["optimizer_success"]),
        "multiple_shooting_success": bool(backend_row["multiple_shooting_success"]),
        "accepted_candidate": str(backend_row["accepted_candidate"]),
        "control_max_norm": float(backend_row["control_max_norm"]),
        "control_bound_violation": float(backend_row["control_bound_violation"]),
        "selected_outage_indices": backend_row["selected_outage_indices"],
        "selected_outage_errors": backend_row["selected_outage_errors"],
        "all_outage_errors": backend_row["all_outage_errors"],
        "settings_fingerprint": expected["settings_fingerprint"],
        "backend_settings_fingerprint": str(
            backend_row.get("settings_fingerprint") or expected["backend_settings_fingerprint"]
        ),
        "config_hash": str(backend_row.get("config_hash") or expected["config_hash"]),
        "source_states_id": str(backend_row.get("source_states_id") or expected["source_states_id"]),
        "message": str(backend_row["message"]),
    }
    return {column: row.get(column) for column in ROBUST_MARGIN_COLUMNS}


def _table_group_label(value: str) -> str:
    labels = {
        "thrust_margin_selected": "Thrust margin",
        "all_single_outage_margin": "All single outages",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _write_table(df: pd.DataFrame, tables_dir: Path) -> None:
    if df.empty:
        return
    table = df.sort_values(["case_group", "phase_time", "amax"])[
        [
            "case_group",
            "phase_time",
            "transfer_time",
            "amax",
            "segments",
            "max_nfev",
            "selected_outages",
            "outage_count",
            "selected_all_outages",
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
    table["case_group"] = table["case_group"].map(_table_group_label)
    table["selected_all_outages"] = table["selected_all_outages"].map(lambda value: "yes" if bool(value) else "no")
    table.columns = [
        "Group",
        "Phase time",
        "Transfer time",
        "amax",
        "Segments",
        "Max nfev",
        "Selected outages",
        "Outage masks",
        "All selected?",
        "Nominal error",
        "Selected worst error",
        "All-mask worst error",
        "Max ||u||",
        "Bound violation",
        "nfev",
        "Runtime (s)",
        "Meets thresholds",
    ]
    (tables_dir / "robust_margin_suite_table.tex").write_text(
        table.to_latex(index=False, float_format="%.4f", escape=True),
        encoding="utf-8",
    )


def _plot_group(ax, df: pd.DataFrame, x_column: str, title: str, xlabel: str) -> None:
    if df.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return
    ordered = df.sort_values(x_column)
    ax.plot(ordered[x_column], ordered["nominal_error"], marker="o", label="nominal")
    ax.plot(ordered[x_column], ordered["selected_worst_error"], marker="s", label="selected worst")
    ax.plot(ordered[x_column], ordered["all_mask_worst_error"], marker="^", label="all-mask worst")
    ax.axhline(float(ordered["nominal_threshold"].iloc[0]), color="0.45", linestyle="--", linewidth=1.0)
    ax.axhline(float(ordered["selected_worst_threshold"].iloc[0]), color="0.25", linestyle=":", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized terminal error")
    ax.grid(alpha=0.25)


def _write_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    thrust = df[df["case_group"] == "thrust_margin_selected"].copy()
    all_single = df[df["case_group"] == "all_single_outage_margin"].copy()
    _plot_group(axes[0], thrust, "amax", "Phase 0.2 thrust margin", "amax")
    _plot_group(axes[1], all_single, "phase_time", "All one-segment outages", "Phase time")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        fig.savefig(figures_dir / f"robust_margin_suite{suffix}", dpi=220 if suffix == ".png" else None)
    plt.close(fig)


def _metadata(
    df: pd.DataFrame,
    config: dict,
    command: str,
    expected: dict[str, dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> dict:
    return {
        "command": command,
        "row_count": int(len(df)),
        "feasible_row_count": int(df["meets_thresholds"].astype(bool).sum()) if not df.empty else 0,
        "config": config,
        "expected_case_count": int(len(expected)),
        "completed_case_count": int(len(df)),
        "skipped_cases": skipped_cases,
        "resume_rejected_rows": resume_rejected_rows,
        "threshold_rule": "meets_thresholds requires nominal_error <= nominal_success and selected_worst_error <= robust_success",
        "selected_all_outages_rule": "selected_all_outages is true when selected_outages equals the count of configured one-segment outage masks for the row",
        "semantics": {
            "selected": (
                "selected_worst_error and selected_outage_errors evaluate only the outage masks chosen for optimized "
                "branch-recovery controls by qlt.multiple_shooting.run_case."
            ),
            "all_mask": (
                "all_mask_worst_error and all_outage_errors evaluate every configured outage mask. For rows where "
                "selected_all_outages is false, unselected masks are diagnostic evaluations rather than optimized "
                "recovery branches."
            ),
            "all_single_outage_margin": (
                "For this suite, all-outage optimization means all one-segment outage masks only; multi-segment outage "
                "families are not included."
            ),
        },
        "target_generation": TARGET_GENERATION,
        "limitations": [
            "Normalized Earth-Moon CR3BP benchmark in nondimensional units.",
            "No flight-ready trajectory optimization or mission-design claim is made.",
            "The all-outage group covers one-segment outage masks only in this suite.",
            "Rows are optimizer outcomes for reproducible evidence; they are not fabricated or extrapolated across missing cases.",
        ],
        "expected_cases": list(expected.values()),
        "rows": df.to_dict(orient="records"),
    }


def _regenerate(
    df: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    config: dict,
    command: str,
    expected: dict[str, dict],
    resume_rejected_rows: list[dict],
    skipped_cases: list[dict],
) -> None:
    df = pd.DataFrame(df, columns=ROBUST_MARGIN_COLUMNS)
    df.to_csv(results_dir / "robust_margin_suite.csv", index=False)
    _write_table(df, tables_dir)
    _write_plot(df, figures_dir)
    write_json(
        results_dir / "robust_margin_suite_metadata.json",
        _metadata(df, config, command, expected, resume_rejected_rows, skipped_cases),
    )


def _runtime_budget(args, config: dict) -> float | None:
    if args.runtime_budget_seconds is not None:
        return float(args.runtime_budget_seconds)
    suite_value = (config.get("suite", {}) or {}).get("runtime_budget_seconds")
    if suite_value is None:
        return None
    return float(suite_value)


def run(args) -> pd.DataFrame:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path.cwd()
    results_dir, figures_dir, tables_dir = output_directories(root, config)
    for directory in (results_dir, figures_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "robust_margin_suite.csv"

    cases = _suite_cases(config)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    expected = _expected_index(config, args, cases)

    resume_rejected_rows: list[dict] = []
    if args.resume:
        loaded = _load_existing(csv_path)
        df, resume_rejected_rows = _compatible_existing_rows(loaded, expected)
    else:
        df = pd.DataFrame(columns=ROBUST_MARGIN_COLUMNS)
    completed = set(str(value) for value in df["settings_fingerprint"].dropna().tolist())

    command = " ".join(sys.argv)
    skipped_cases: list[dict] = []
    start = time.perf_counter()
    budget = _runtime_budget(args, config)

    for case in cases:
        expected_row = expected[_case_hash(_case_payload(case))]
        if expected_row["settings_fingerprint"] in completed:
            continue
        elapsed = time.perf_counter() - start
        if budget is not None and elapsed >= budget:
            skipped_cases.append(
                {
                    "case_id": expected_row["case_id"],
                    "case": _case_payload(case),
                    "reason": "runtime budget reached before launching case",
                    "elapsed_seconds": elapsed,
                    "runtime_budget_seconds": budget,
                }
            )
            continue

        case_config = _case_config(config, case)
        backend_row = multiple_shooting.run_case(
            case_config,
            float(case["transfer_time"]),
            float(case["amax"]),
            int(case["segments"]),
            int(case["max_nfev"]),
            _args_for_case(config, args, case),
        )
        row = _row_from_backend(case, backend_row, expected_row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        completed.add(str(row["settings_fingerprint"]))
        _regenerate(df, results_dir, figures_dir, tables_dir, config, command, expected, resume_rejected_rows, skipped_cases)
        print(
            f"{case['case_group']} phase={case['phase_time']} tf={case['transfer_time']} "
            f"amax={case['amax']} N={case['segments']} selected={case['selected_outages']}/{case['outage_count']}: "
            f"nominal={row['nominal_error']:.6f}, selected={row['selected_worst_error']:.6f}, "
            f"all={row['all_mask_worst_error']:.6f}, met={row['meets_thresholds']}, "
            f"nfev={row['nfev']}, runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

    _regenerate(df, results_dir, figures_dir, tables_dir, config, command, expected, resume_rejected_rows, skipped_cases)
    return pd.DataFrame(df, columns=ROBUST_MARGIN_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust-margin multiple-shooting evidence suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/robust_margin_suite.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--node-initialization", choices=["rollout", "linear", "blend"], default=None)
    parser.add_argument("--node-initialization-blend", "--node-blend", dest="node_initialization_blend", type=float, default=None)
    parser.add_argument("--nominal-first-homotopy", "--nominal-first", dest="nominal_first_homotopy", action="store_true", default=None)
    parser.add_argument("--nominal-first-nfev", type=int, default=None)
    for key in multiple_shooting.RESIDUAL_WEIGHT_DEFAULTS:
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=float, default=None)
    parser.add_argument("--solver-mode", default=None)
    parser.add_argument("--warm-start-refinement", action="store_true")
    parser.add_argument("--warm-start-nfev", type=int, default=90)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--runtime-budget-seconds", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
