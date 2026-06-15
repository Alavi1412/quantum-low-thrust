from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlt.multiple_shooting import build_parser, run


def main() -> None:
    parser = build_parser()
    parser.set_defaults(
        config=ROOT / "configs" / "catalog_collocation_feasibility.yaml",
        source_states=ROOT / "data" / "source_states.json",
        transfer_times="4.0,5.0,6.0",
        amax="0.3,0.5",
        segments="14,18",
        max_nfev="250,400",
        selected_outages=3,
        min_recovery_segments=4,
        node_initialization="linear",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
