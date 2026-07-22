"""ProbeNet CLI entrypoint — dispatches to trainer, rollout, or eval modes.

Usage:
    uv run python -m probenet.cli --mode trainer --config configs/trainer.yaml
    uv run python -m probenet.cli --mode rollout --config configs/rollout.yaml
    uv run python -m probenet.cli --mode eval --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from probenet.training.config import TrainingConfig
from probenet.training.trainer import Trainer
from probenet.rollout.worker import RolloutConfig, RolloutWorker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ProbeNet — interactive physical perception")
    parser.add_argument("--mode", choices=["trainer", "rollout", "eval"], required=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--policy", choices=["pi05", "gr00t"], default="pi05")
    parser.add_argument("--exp-name", type=str, default="probenet")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--probenet", action="store_true", help="Enable ProbeNet physical property conditioning")
    args = parser.parse_args()

    if args.mode == "trainer":
        _run_trainer(args)
    elif args.mode == "rollout":
        _run_rollout(args)
    elif args.mode == "eval":
        _run_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_trainer(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        exp_name=args.exp_name,
        policy=args.policy,
        probenet_enabled=args.probenet,
    )
    trainer = Trainer(config)
    trainer.run()


def _run_rollout(args: argparse.Namespace) -> None:
    config = RolloutConfig(
        num_episodes=args.num_episodes,
        headless=args.headless,
    )
    worker = RolloutWorker(config)
    worker.run()


def _run_eval(args: argparse.Namespace) -> None:
    logging.info("Eval mode — not yet implemented (Phase 8)")
    logging.info("Would evaluate: policy=%s, episodes=%d", args.policy, args.num_episodes)
    sys.exit(0)


if __name__ == "__main__":
    main()
