from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import output_directories
from qlt.reporting import write_latex_tables


def configured_output_paths(config_path: Path, results_subdir: str | None = None) -> tuple[Path, Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if results_subdir is not None:
        config = dict(config)
        config["run"] = dict(config.get("run", {}) or {})
        config["run"]["output_subdir"] = results_subdir
    return output_directories(ROOT, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate LaTeX tables from saved experiment CSVs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--results-subdir", default=None, help="Override run.output_subdir for legacy or ad hoc outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir, _, tables_dir = configured_output_paths(args.config, args.results_subdir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_dir / "summary.csv")
    qubo = pd.read_csv(results_dir / "qubo_diagnostics.csv")
    write_latex_tables(summary, qubo, tables_dir)


if __name__ == "__main__":
    main()
