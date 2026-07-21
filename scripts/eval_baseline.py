"""Evaluate a saved policy checkpoint against the Isaac Sim SO-101 environment.

IMPORTANT: Requires Isaac Sim + LeIsaac installed.

Usage:
    uv run scripts/eval_baseline.py --checkpoint-dir outputs/checkpoints/pi05_so101 --num-episodes 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy checkpoint in sim.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    print("Eval: requires Isaac Sim + LeIsaac (see record_sim.py)")
    print(f"Would evaluate checkpoint {args.checkpoint_dir}")
    print(f"  episodes: {args.num_episodes}, headless: {args.headless}")

    # Placeholder — real eval in Phase 8
    results = {"checkpoint": args.checkpoint_dir, "episodes": args.num_episodes, "success_rate": 0.0}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
