from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.multiple_shooting import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact bounded projected direct multiple-shooting benchmark for catalog-derived retargeting."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/bounded_collocation_phase_or_catalog.yaml"))
    parser.add_argument("--source-states", type=Path, default=Path("data/source_states.json"))
    parser.add_argument("--transfer-times", default="0.5")
    parser.add_argument("--amax", default="0.2")
    parser.add_argument("--segments", default="12")
    parser.add_argument("--max-nfev", default="80")
    parser.add_argument("--selected-outages", type=int, default=12)
    parser.add_argument("--min-recovery-segments", type=int, default=0)
    parser.add_argument("--node-initialization", choices=["rollout", "linear", "blend"], default="linear")
    parser.add_argument("--node-initialization-blend", "--node-blend", dest="node_initialization_blend", type=float, default=0.5)
    parser.add_argument("--nominal-first-homotopy", "--nominal-first", dest="nominal_first_homotopy", action="store_true", default=False)
    parser.add_argument("--nominal-first-nfev", type=int, default=40)
    parser.add_argument("--initial-weight", type=float, default=10.0)
    parser.add_argument("--defect-weight", type=float, default=1.0)
    parser.add_argument("--terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-terminal-weight", type=float, default=4.0)
    parser.add_argument("--branch-start-weight", type=float, default=5.0)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--smooth-weight", type=float, default=0.012)
    parser.add_argument("--solver-mode", default="bounded_projected_multiple_shooting")
    parser.add_argument("--warm-start-refinement", action="store_true")
    parser.add_argument("--warm-start-nfev", type=int, default=30)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = run(SimpleNamespace(**vars(args)))
    if not df.empty:
        best = df.sort_values(
            ["meets_thresholds", "selected_worst_error", "nominal_error"],
            ascending=[False, True, True],
        ).iloc[0]
        print(
            "best bounded case: "
            f"nominal={best['nominal_error']:.6f}, "
            f"selected={best['selected_worst_error']:.6f}, "
            f"all={best['all_mask_worst_error']:.6f}, "
            f"max_norm={best['control_max_norm']:.6f}, "
            f"violation={best['control_bound_violation']:.3e}, "
            f"met={bool(best['meets_thresholds'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
