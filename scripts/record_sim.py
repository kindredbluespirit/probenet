"""Generate pick-and-place demonstrations in Isaac Sim and save as LeRobotDataset.

IMPORTANT: Requires Isaac Sim + LeIsaac installed. Run inside the rollout
Docker image or a local environment with ``isaacsim`` pip package.

Usage:
    uv run scripts/record_sim.py --num-episodes 100 --output-dir data/lerobot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset
from tqdm import tqdm

from probenet.env import IsaacSO101Env, LerobotAdapter, ObjectSpec
from probenet.probe.probe_runner import JOINT_ORDER

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_ORDER,
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_ORDER,
    },
    "observation.images.cam_primary": {
        "dtype": "video",
        "shape": (3, 224, 224),
    },
}


def make_default_object_specs() -> list[ObjectSpec]:
    return [
        ObjectSpec(
            name="shell",
            usd_path="assets/objects/shell.usd",
            mass=0.3,
            friction=(0.5, 0.5),
            spawn_pos=(0.25, 0.0, 0.035),
        ),
    ]


def build_leisaac_task(headless: bool = True) -> Any:
    """Create a LeIsaac pick-and-place task env for SO-101.

    Requires Isaac Sim + LeIsaac to be installed and a SimulationApp
    to be running.
    """
    from isaacsim.simulation_app import SimulationApp  # noqa: F811

    SimulationApp({"headless": headless})

    from leisaac.tasks.template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg

    raise NotImplementedError(
        "LeIsaac task creation not yet implemented — "
        "define a custom task env extending SingleArmTaskDirectEnv. "
        "See leisaac/tasks/lift_cube/direct/lift_cube_env.py for example."
    )


def record_episode(
    adapter: LerobotAdapter,
    dataset: LeRobotDataset,
    trajectory: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> None:
    for target in trajectory:
        noisy = target.copy()
        noisy[:5] += rng.normal(0, noise_scale, size=5)
        adapter.send_action(noisy)

        obs = adapter.get_observation()
        joint_state = np.array(
            [obs.get(f"{j}.pos", 0.0) for j in JOINT_ORDER],
            dtype=np.float32,
        )
        image = obs.get("cam_primary", np.zeros((224, 224, 3), dtype=np.uint8))

        dataset.add_frame({
            "observation.state": joint_state,
            "action": target.astype(np.float32),
            "observation.images.cam_primary": image,
            "task": "pick up the object and place it",
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sim demos as LeRobot dataset.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data/lerobot")
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", type=str, default="probenet/so101_pick_place")
    parser.add_argument("--headless", action="store_true", default=True)
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

    leisaac_task = build_leisaac_task(headless=args.headless)
    object_specs = make_default_object_specs()
    env = IsaacSO101Env(leisaac_task, object_specs=object_specs, headless=args.headless)
    adapter = LerobotAdapter(env)

    for _ep_id in tqdm(range(args.num_episodes), desc="Recording episodes"):
        env.reset()
        # TODO: replace with actual pick-and-place trajectory from state machine
        trajectory = np.tile(np.zeros(6, dtype=np.float32), (100, 1))
        record_episode(adapter, dataset, trajectory, args.noise_scale, rng)
        dataset.save_episode()

    dataset.finalize()
    env.close()
    print(f"Saved {args.num_episodes} episodes to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
