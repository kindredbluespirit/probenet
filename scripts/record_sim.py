"""Generate pick-and-place demonstrations in MuJoCo and save as LeRobotDataset.

Usage:
    uv run scripts/record_sim.py --num-episodes 100 --output-dir data/lerobot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm

from probenet.env import SO101Env
from probenet.env.lerobot_adapter import LerobotAdapter

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
    },
    "observation.images.cam_primary": {
        "dtype": "video",
        "shape": (3, 224, 224),
    },
}


def record_episode(
    adapter: LerobotAdapter,
    dataset: LeRobotDataset,
    trajectory: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> None:
    """Run one pick-and-place episode and add frames to the dataset."""
    for target_ctrl in trajectory:
        noisy = target_ctrl.copy()
        noisy[:5] += rng.normal(0, noise_scale, size=5)
        adapter.send_action(noisy)

        obs = adapter.get_observation()
        joint_state = np.array(
            [
                obs["shoulder_pan.pos"],
                obs["shoulder_lift.pos"],
                obs["elbow_flex.pos"],
                obs["wrist_flex.pos"],
                obs["wrist_roll.pos"],
                obs["gripper.pos"],
            ],
            dtype=np.float32,
        )
        image = obs["cam_primary"]  # HWC uint8

        dataset.add_frame(
            {
                "observation.state": joint_state,
                "action": target_ctrl.astype(np.float32),
                "observation.images.cam_primary": image,
                "task": "pick up the shell and place it on the table",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sim demos as LeRobot dataset.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/lerobot")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", type=str, default="probenet/so101_pick_place")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=30,
        robot_type="so101_follower",
        features=FEATURES,
    )

    num_each = args.num_episodes // 2
    image_size = (args.image_size, args.image_size)

    for ep_id in tqdm(range(num_each * 2), desc="Recording episodes"):
        object_type = "shell_a" if ep_id % 2 == 0 else "shell_b"
        env = SO101Env(object_type=object_type, image_size=image_size)
        env.reset(seed=int(rng.integers(0, 2**31)))
        adapter = LerobotAdapter(env)
        trajectory = env.pick_place_trajectory

        record_episode(adapter, dataset, trajectory, args.noise_scale, rng)
        dataset.save_episode()
        env.close()

    dataset.finalize()
    print(f"Saved {num_each * 2} episodes to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
