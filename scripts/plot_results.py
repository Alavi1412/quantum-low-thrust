from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.experiment import output_directories
from qlt.reporting import plot_method_comparison, plot_qubo_fit, plot_recovery_fuel, plot_refinement_success


def configured_output_paths(config_path: Path, results_subdir: str | None = None) -> tuple[Path, Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if results_subdir is not None:
        config = dict(config)
        config["run"] = dict(config.get("run", {}) or {})
        config["run"]["output_subdir"] = results_subdir
    return output_directories(ROOT, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate result plots from saved experiment CSVs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--results-subdir", default=None, help="Override run.output_subdir for legacy or ad hoc outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir, figures_dir, _ = configured_output_paths(args.config, args.results_subdir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_dir / "summary.csv")
    fit_paths = sorted(results_dir.glob("qubo_fit_seed*.csv"))
    plot_method_comparison(summary, figures_dir)
    plot_qubo_fit(fit_paths, figures_dir)
    plot_refinement_success(summary, figures_dir)
    plot_recovery_fuel(summary, figures_dir)


if __name__ == "__main__":
    main()
