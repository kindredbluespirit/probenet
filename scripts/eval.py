"""Evaluate a trained policy in simulation.

Usage:
    python scripts/eval.py --checkpoint outputs/models/baseline_best.pt --variant baseline --num-trials 20
    python scripts/eval.py --checkpoint outputs/models/probenet_best.pt --variant probenet --num-trials 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from probenet.conditioning import MLPConditioner
from probenet.env import SO101Env
from probenet.policies import BCPolicy, ProbeNetPolicy


def ctrl_to_action(ctrl: np.ndarray, ctrl_range: np.ndarray) -> np.ndarray:
    """Convert absolute actuator positions to normalized [-1, 1] action."""
    low, high = ctrl_range[:, 0], ctrl_range[:, 1]
    return 2.0 * (ctrl - low) / (high - low) - 1.0


@torch.no_grad()
def rollout(
    policy: BCPolicy | ProbeNetPolicy,
    env: SO101Env,
    object_type: str,
    device: torch.device,
    start_pose: np.ndarray | None = None,
) -> dict:
    """Run one episode with the policy and return results."""
    obs, _ = env.reset(seed=None, options={"object_type": object_type})

    # If a start_pose is given (in ctrl space), move arm there first.
    if start_pose is not None:
        ctrl_range = env.model.actuator_ctrlrange
        action = ctrl_to_action(start_pose.copy(), ctrl_range)
        env.step(action)

    steps = 0

    for _ in range(500):
        rgb = torch.from_numpy(obs["image"]).float().to(device) / 255.0
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)
        state = torch.from_numpy(obs["state"]).float().to(device).unsqueeze(0)

        if isinstance(policy, ProbeNetPolicy):
            vis = torch.tensor([[1.0, 0.035, 0.5, 0.0]], device=device)
            if object_type == "shell_a":
                phys = torch.tensor([[0.05, 1.0, 1.2]], device=device)
            else:
                phys = torch.tensor([[0.3, 0.5, 0.3]], device=device)
            action_pred = policy(rgb, state, vis, phys)
        else:
            action_pred = policy(rgb, state)

        action = action_pred.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    return {"steps": steps, "object_type": object_type}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "probenet"])
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model and load weights.
    if args.variant == "baseline":
        model = BCPolicy(action_dim=6).to(device)
    else:
        conditioner = MLPConditioner(visual_dim=4, physical_dim=3, output_dim=32)
        model = ProbeNetPolicy(action_dim=6, conditioner=conditioner).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} (val_loss={ckpt['val_loss']:.6f})")

    # Starting pose for evaluation (home).
    start_pose = np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32)

    results: dict[str, list] = {"shell_a": [], "shell_b": []}
    for trial in range(args.num_trials):
        object_type = "shell_a" if trial % 2 == 0 else "shell_b"
        env = SO101Env(object_type=object_type, image_size=(args.image_size, args.image_size))

        outcome = rollout(model, env, object_type, device, start_pose)
        results[object_type].append(outcome)

        env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation results — {args.variant} ({args.num_trials} trials)")
    print(f"{'='*50}")
    total_steps = {"shell_a": 0, "shell_b": 0}
    for shell in ["shell_a", "shell_b"]:
        n = len(results[shell])
        steps = [r["steps"] for r in results[shell]]
        avg_steps = np.mean(steps)
        total_steps[shell] = avg_steps
        print(f"  {shell}: {n} trials, avg steps = {avg_steps:.1f}")

    print(f"\nShell A avg steps: {total_steps['shell_a']:.1f}")
    print(f"Shell B avg steps: {total_steps['shell_b']:.1f}")
    asymmetry = abs(total_steps["shell_a"] - total_steps["shell_b"])
    print(f"Asymmetry: {asymmetry:.1f} steps")


if __name__ == "__main__":
    main()
