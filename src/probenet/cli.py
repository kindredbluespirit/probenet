"""ProbeNet CLI entrypoint — dispatches to trainer, rollout, or eval modes.

Usage:
    uv run python -m probenet.cli --mode trainer --config configs/trainer.yaml
    uv run python -m probenet.cli --mode rollout --config configs/rollout.yaml
    uv run python -m probenet.cli --mode eval --config configs/eval.yaml
"""

import argparse
import logging
import sys


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ProbeNet — interactive physical perception")
    parser.add_argument("--mode", choices=["trainer", "rollout", "eval"], required=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--policy", choices=["pi05", "gr00t"], default="pi05")
    args = parser.parse_args()

    if args.mode == "trainer":
        logging.info("Trainer mode — not yet implemented (Phase 4-5)")
        sys.exit(0)
    elif args.mode == "rollout":
        logging.info("Rollout mode — not yet implemented (Phase 2)")
        sys.exit(0)
    elif args.mode == "eval":
        logging.info("Eval mode — not yet implemented (Phase 10)")
        sys.exit(0)


if __name__ == "__main__":
    main()
