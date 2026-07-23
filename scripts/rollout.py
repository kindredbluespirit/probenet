"""Rollout orchestrator — connects an inference server to a data generator.

Drives the episode loop:
    reset data_gen → get obs → infer policy → step data_gen → repeat

Usage:
    uv run python scripts/rollout.py \
        --policy openpi --policy-url ws://localhost:8000 \
        --data-gen sim --data-gen-url ws://localhost:8226 \
        --num-episodes 50 \
        --output-dir outputs/eval_runs/baseline_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from probenet.clients import create_policy_client, create_data_gen_client

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ProbeNet rollout orchestrator")
    parser.add_argument("--policy", choices=["openpi", "gr00t"], default="openpi")
    parser.add_argument("--policy-url", type=str, default=None)
    parser.add_argument("--data-gen", choices=["sim", "so101"], default="sim")
    parser.add_argument("--data-gen-url", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/eval_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    policy = create_policy_client(args.policy, args.policy_url)
    data_gen = create_data_gen_client(args.data_gen, args.data_gen_url)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for ep_id in range(args.num_episodes):
        logger.info("Episode %d / %d", ep_id + 1, args.num_episodes)

        result = data_gen.reset(seed=int(rng.integers(0, 2**31)))
        obs = result["obs"] if isinstance(result, dict) else result

        episode_data = []
        for step in range(args.max_steps_per_episode):
            inference_result = policy.infer(obs)
            actions = np.asarray(inference_result["actions"])

            obs, reward, done = data_gen.step(actions)
            episode_data.append({
                "step": step,
                "obs": obs,
                "action": actions.tolist(),
                "reward": reward,
            })

            if done:
                logger.info("  Episode done at step %d", step)
                break

        ep_path = output_dir / f"episode_{ep_id:04d}.json"
        ep_path.write_text(json.dumps(episode_data, indent=2))
        logger.info("  Saved to %s", ep_path)

    policy.close()
    data_gen.close()
    logger.info("Rollout complete: %d episodes → %s", args.num_episodes, output_dir)


if __name__ == "__main__":
    main()
