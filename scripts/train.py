"""Training orchestrator — backend-agnostic.

Loads a registered config and dispatches to the appropriate training backend
(openpi or GR00T).

Usage:
    uv run python scripts/train.py \
        --config-name pi05_so101 \
        --exp-name baseline_v1 \
        --dataset data/lerobot/so101_pick_place
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ProbeNet training orchestrator")
    parser.add_argument("--policy", choices=["openpi", "gr00t"], default="openpi")
    parser.add_argument("--config-name", type=str, required=True, help="Registered config name")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name for checkpoint dir")
    parser.add_argument("--dataset", type=str, default="data/lerobot/so101_pick_place")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true", default=True)
    args = parser.parse_args()

    if args.policy == "openpi":
        _run_openpi_training(args)
    elif args.policy == "gr00t":
        _run_gr00t_training(args)


def _run_openpi_training(args):
    """Shell out to openpi's training script with the correct config."""
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)
    logger.info("Starting openpi training: config=%s, exp=%s", args.config_name, args.exp_name)

    ckpt_dir = Path(args.output_dir) / args.config_name / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "openpi.scripts.train",
        args.config_name,
        f"exp_name={args.exp_name}",
        f"checkpoint_dir={ckpt_dir}",
    ]
    if args.resume:
        cmd.append("resume=true")
    if args.overwrite:
        cmd.append("overwrite=true")

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_gr00t_training(args):
    """Shell out to GR00T's launch_finetune script."""
    logger.info("GR00T training not yet implemented")
    raise NotImplementedError("GR00T training — coming in Phase E")


if __name__ == "__main__":
    main()
