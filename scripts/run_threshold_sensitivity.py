"""Threshold-sensitivity postprocessor for the 30-seed phase-shift package.

This script reads the recorded 30-seed raw_results.csv and recomputes success
counts under alternate nominal and selected-worst terminal-error thresholds. It
does not launch trajectory optimization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_CSV = ROOT / "data" / "results" / "phase_shift_cardinality_30seed" / "raw_results.csv"
DEFAULT_RESULTS_DIR = ROOT / "data" / "results" / "phase_shift_cardinality_30seed"
DEFAULT_TABLES_DIR = ROOT / "tables" / "phase_shift_cardinality_30seed"

REQUIRED_COLUMNS = {
    "seed",
    "method",
    "refined_nominal_error",
    "refined_selected_worst_error",
}

THRESHOLD_GRID = [
    {
        "threshold_id": "configured_0p09_0p17",
        "label": "Configured",
        "nominal_threshold": 0.09,
        "selected_worst_threshold": 0.17,
    },
    {
        "threshold_id": "stricter_0p075_0p12",
        "label": "Stricter",
        "nominal_threshold": 0.075,
        "selected_worst_threshold": 0.12,
    },
    {
        "threshold_id": "stringent_0p065_0p10",
        "label": "Stringent",
        "nominal_threshold": 0.065,
        "selected_worst_threshold": 0.10,
    },
    {
        "threshold_id": "continuous_dominance_0p05_0p09",
        "label": "Tight",
        "nominal_threshold": 0.05,
        "selected_worst_threshold": 0.09,
    },
]

METHOD_ORDER = [
    "random",
    "cross_entropy",
    "genetic",
    "true_sa",
    "surrogate_qubo_sa",
    "qaoa_statevector",
    "all_windows_continuous",
]

METHOD_LABELS = {
    "random": "Random",
    "cross_entropy": "CEM",
    "genetic": "Genetic",
    "true_sa": "True SA",
    "surrogate_qubo_sa": "QUBO SA",
    "qaoa_statevector": "QAOA",
    "all_windows_continuous": "All-windows",
}


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _method_order(methods: list[str]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    ordered.extend(sorted(method for method in methods if method not in ordered))
    return ordered


def validate_raw_results(raw: pd.DataFrame) -> dict:
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise RuntimeError(f"raw results missing required columns: {sorted(missing)}")
    if raw.empty:
        raise RuntimeError("raw results are empty")

    duplicate_pairs = (
        raw.groupby(["seed", "method"], dropna=False)
        .size()
        .reset_index(name="rows")
        .query("rows != 1")
    )
    if not duplicate_pairs.empty:
        raise RuntimeError(
            "raw results contain duplicate seed/method rows: "
            f"{duplicate_pairs.to_dict(orient='records')}"
        )

    methods = _method_order(list(dict.fromkeys(raw["method"].astype(str).to_list())))
    return {
        "row_count": int(len(raw)),
        "seed_count": int(raw["seed"].nunique()),
        "methods": methods,
    }


def threshold_sensitivity(
    raw: pd.DataFrame,
    threshold_grid: list[dict] | None = None,
) -> pd.DataFrame:
    grid = threshold_grid or THRESHOLD_GRID
    validation = validate_raw_results(raw)
    methods = validation["methods"]

    rows: list[dict] = []
    for threshold in grid:
        nominal_threshold = float(threshold["nominal_threshold"])
        selected_threshold = float(threshold["selected_worst_threshold"])
        for method in methods:
            group = raw[raw["method"].astype(str) == method]
            nominal = group["refined_nominal_error"].astype(float)
            selected = group["refined_selected_worst_error"].astype(float)
            passed = (nominal <= nominal_threshold) & (selected <= selected_threshold)
            runs = int(len(group))
            successes = int(passed.sum())
            rows.append(
                {
                    "threshold_id": str(threshold["threshold_id"]),
                    "threshold_label": str(threshold["label"]),
                    "nominal_threshold": nominal_threshold,
                    "selected_worst_threshold": selected_threshold,
                    "method": method,
                    "runs": runs,
                    "successes": successes,
                    "success_count": f"{successes}/{runs}",
                    "success_rate": successes / runs if runs else float("nan"),
                    "refined_nominal_error_median": float(nominal.median()),
                    "refined_selected_worst_error_median": float(selected.median()),
                }
            )
    return pd.DataFrame(rows)


def write_threshold_table(sensitivity: pd.DataFrame, tables_dir: Path) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "threshold_sensitivity_table.tex"

    threshold_rows = []
    methods = _method_order(list(dict.fromkeys(sensitivity["method"].astype(str).to_list())))
    for threshold_id, group in sensitivity.groupby("threshold_id", sort=False):
        first = group.iloc[0]
        row: dict[str, str] = {
            "Threshold": str(first["threshold_label"]),
            "$(e_n,e_s)$": (
                f"({float(first['nominal_threshold']):.3f}, "
                f"{float(first['selected_worst_threshold']):.3f})"
            ),
        }
        by_method = group.set_index("method")
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            row[label] = str(by_method.loc[method, "success_count"]) if method in by_method.index else ""
        threshold_rows.append(row)

    pd.DataFrame(threshold_rows).to_latex(path, index=False, escape=False)
    return path


def write_artifacts(
    raw: pd.DataFrame,
    *,
    raw_csv: Path,
    results_dir: Path,
    tables_dir: Path,
    command: str,
) -> dict:
    validation = validate_raw_results(raw)
    sensitivity = threshold_sensitivity(raw)

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "threshold_sensitivity.csv"
    metadata_path = results_dir / "threshold_sensitivity_metadata.json"
    table_path = write_threshold_table(sensitivity, tables_dir)
    sensitivity.to_csv(csv_path, index=False)

    metadata = {
        "command": command,
        "raw_results_csv": _relative_or_absolute(raw_csv),
        "raw_results_validation": validation,
        "threshold_grid": [
            {
                "threshold_id": str(item["threshold_id"]),
                "label": str(item["label"]),
                "nominal_threshold": float(item["nominal_threshold"]),
                "selected_worst_threshold": float(item["selected_worst_threshold"]),
            }
            for item in THRESHOLD_GRID
        ],
        "success_rule": (
            "A method-seed row passes when refined_nominal_error <= nominal_threshold "
            "and refined_selected_worst_error <= selected_worst_threshold."
        ),
        "source_columns": [
            "refined_nominal_error",
            "refined_selected_worst_error",
        ],
        "artifacts": {
            "threshold_sensitivity_csv": _relative_or_absolute(csv_path),
            "threshold_sensitivity_table_tex": _relative_or_absolute(table_path),
            "threshold_sensitivity_metadata_json": _relative_or_absolute(metadata_path),
        },
        "interpretation_limits": [
            "Derived only from recorded raw_results.csv; no trajectory optimization is rerun.",
            "Thresholds are normalized screening tolerances, not flight targeting tolerances.",
            "Counts test threshold sensitivity of recorded refinements, not a new optimizer.",
        ],
        "determinism_note": (
            "Runtime is intentionally omitted so rerunning this deterministic "
            "postprocessor leaves metadata unchanged when inputs and command are unchanged."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "threshold_sensitivity": sensitivity,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate threshold-sensitivity success counts from the recorded "
            "30-seed phase-shift cardinality raw_results.csv."
        )
    )
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_csv = args.raw_csv if args.raw_csv.is_absolute() else ROOT / args.raw_csv
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else ROOT / args.tables_dir

    if not raw_csv.is_file():
        print(f"ERROR: raw results not found: {raw_csv}", file=sys.stderr)
        return 1
    raw = pd.read_csv(raw_csv)
    artifacts = write_artifacts(
        raw,
        raw_csv=raw_csv,
        results_dir=results_dir,
        tables_dir=tables_dir,
        command=" ".join(sys.argv),
    )
    metadata = artifacts["metadata"]
    print(
        "Completed threshold sensitivity "
        f"for {metadata['raw_results_validation']['seed_count']} seeds and "
        f"{len(metadata['raw_results_validation']['methods'])} methods.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
