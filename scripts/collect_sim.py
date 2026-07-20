"""Generate probe + pick-and-place demonstrations in MuJoCo simulation.

Usage:
    python scripts/collect_sim.py --num-episodes 100 --output-dir data/sim

This saves episodes in the format expected by ``ProbeNetDataset``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from probenet.dataset import save_episode
from probenet.env import SO101Env
from probenet.probe import ProbeRunner, default_probe_config, extract_probe_features


def ctrl_to_action(ctrl: np.ndarray, ctrl_range: np.ndarray) -> np.ndarray:
    """Convert absolute actuator positions to the env's normalized [-1, 1] action."""
    low, high = ctrl_range[:, 0], ctrl_range[:, 1]
    return 2.0 * (ctrl - low) / (high - low) - 1.0


# Keyframe trajectory for pick-and-place (joint targets).
#   joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
KEYFRAMES = [
    # 0: home (arm above object area, gripper open)
    np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32),
    # 1: approach (lean in toward object)
    np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32),
    # 2: grasp (close gripper)
    np.array([0.0, -0.5, 0.8, 0.8, 1.58, 1.5], dtype=np.float32),
    # 3: lift (raise elbow/wrist)
    np.array([0.0, -0.7, 0.8, 1.0, 1.58, 1.5], dtype=np.float32),
    # 4: carry (rotate shoulder to move object sideways)
    np.array([0.3, -0.6, 0.8, 0.9, 1.58, 1.5], dtype=np.float32),
    # 5: lower to place
    np.array([0.3, -0.5, 0.8, 0.8, 1.58, 1.5], dtype=np.float32),
    # 6: release
    np.array([0.3, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32),
    # 7: return to home
    np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32),
]

# Steps between consecutive keyframes.
STEPS_BETWEEN = [50, 80, 80, 120, 80, 60, 80, 80]


def interpolate_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Interpolate linearly between keyframes.

    Returns:
        actions: (T, 6) joint target commands.
        labels: (T,) phase labels (0-7).
    """
    actions: list[np.ndarray] = []
    labels: list[int] = []
    for i, (kf0, kf1) in enumerate(zip(KEYFRAMES[:-1], KEYFRAMES[1:], strict=True)):
        steps = STEPS_BETWEEN[i]
        for t in range(steps):
            alpha = t / steps
            act = kf0 + alpha * (kf1 - kf0)
            actions.append(act)
            labels.append(i)
    return np.stack(actions), np.array(labels, dtype=np.int32)


MAIN_TRAJECTORY, _ = interpolate_trajectory()


def collect_episode(
    env: SO101Env,
    object_type: str,
    probe_cfg: default_probe_config.__class__,
    ctrl_range: np.ndarray,
    noise_scale: float = 0.01,
) -> dict:
    """Run one probe + pick-and-place episode and return recorded data.

    Args:
        env: A reset environment.
        object_type: ``"shell_a"`` or ``"shell_b"``.
        probe_cfg: Probe configuration.
        ctrl_range: (nu, 2) actuator ctrl range from the env model.
        noise_scale: Std of Gaussian noise added to actions for variety.

    Returns:
        Dict with keys ``rgb``, ``state``, ``action``, ``probe_signal``, ``metadata``.
    """
    # Run probe first.
    runner = ProbeRunner(env, probe_cfg)
    probe_signal = runner.run()
    probe_features = extract_probe_features(probe_signal)

    # Record episode data.
    rgbs: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    # Normalize all keyframes to [-1, 1] action space.
    norm_trajectory = np.stack([ctrl_to_action(kf, ctrl_range) for kf in MAIN_TRAJECTORY])

    # Reset to home after probe.
    env.reset(seed=None, options={"object_type": object_type})

    for target_norm in norm_trajectory:
        # Add noise for variety.
        noisy = target_norm.copy()
        noisy[:6] += env.np_random.normal(0, noise_scale, size=6)
        obs, _, _, _, _ = env.step(noisy)
        rgbs.append(obs["image"])
        states.append(obs["state"])
        actions.append(target_norm)

    # Physical params (ground truth from sim).
    if object_type == "shell_a":
        physical_params = {"mass": 0.05, "compliance": 1.0, "friction": 1.2}
    else:
        physical_params = {"mass": 0.3, "compliance": 0.5, "friction": 0.3}

    # Visual params — identical for both shells (vision cannot distinguish them).
    visual_params = {"shape": 1.0, "size": 0.035, "gloss": 0.5, "material": 0.0}

    return {
        "rgb": np.stack(rgbs),
        "state": np.stack(states),
        "action": np.stack(actions),
        "probe_signal": probe_signal,
        "metadata": {
            "object_type": object_type,
            "visual_params": visual_params,
            "physical_params": physical_params,
            "probe_features": probe_features,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sim demonstrations.")
    parser.add_argument("--num-episodes", type=int, default=100, help="Total episodes.")
    parser.add_argument("--output-dir", type=str, default="data/sim", help="Output directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Render size (square).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_cfg = default_probe_config()
    num_each = args.num_episodes // 2
    rng = np.random.RandomState(42)

    # Get ctrl range from a fresh env.
    tmp_env = SO101Env(object_type="shell_a", image_size=(args.image_size, args.image_size))
    ctrl_range = tmp_env.model.actuator_ctrlrange.copy()
    tmp_env.close()

    for ep_id in tqdm(range(num_each * 2), desc="Collecting episodes"):
        object_type = "shell_a" if ep_id % 2 == 0 else "shell_b"
        env = SO101Env(object_type=object_type, image_size=(args.image_size, args.image_size))
        env.reset(seed=int(rng.randint(0, 2**31)))

        ep_data = collect_episode(env, object_type, probe_cfg, ctrl_range, noise_scale=0.02)
        save_episode(output_dir, ep_id, ep_data["rgb"], ep_data["state"], ep_data["action"], ep_data["metadata"])
        env.close()

    # Write a dataset manifest.
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "num_episodes": num_each * 2,
                "num_shell_a": num_each,
                "num_shell_b": num_each,
                "image_size": args.image_size,
                "action_dim": 6,
                "state_dim": 25,
            },
            f,
            indent=2,
        )

    print(f"Done. Episodes saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
