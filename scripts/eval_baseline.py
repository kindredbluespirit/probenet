"""Evaluate a fine-tuned π₀.₅ policy on the SO-101 MuJoCo environment.

Connects to an openpi policy server via websocket and runs rollouts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from probenet.env import SO101Env
from probenet.env.lerobot_adapter import LerobotAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate openpi policy on SO-101.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to fine-tuned checkpoint.")
    parser.add_argument("--config-name", type=str, default="pi05_so101", help="openpi config name.")
    parser.add_argument("--num-episodes", type=int, default=20, help="Rollouts per shell.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    openpi_dir = checkpoint_dir.parents[3]

    server_cmd = [
        "uv", "run", "scripts/serve_policy.py", "policy:checkpoint",
        f"--policy.config={args.config_name}",
        f"--policy.dir={checkpoint_dir}",
        f"--policy.host={args.policy_host}",
        f"--policy.port={args.policy_port}",
    ]

    proc = subprocess.Popen(server_cmd, cwd=str(openpi_dir))
    time.sleep(30)

    try:
        results: dict[str, list[int]] = {"shell_a": [], "shell_b": []}
        image_size = (args.image_size, args.image_size)

        for object_type in ("shell_a", "shell_b"):
            for ep in range(args.num_episodes):
                env = SO101Env(object_type=object_type, image_size=image_size)
                env.reset(seed=ep)
                adapter = LerobotAdapter(env)

                steps = 0
                for _ in range(args.max_steps):
                    obs = adapter.get_observation()
                    payload = {
                        "observation": {
                            "state": [
                                obs["shoulder_pan.pos"],
                                obs["shoulder_lift.pos"],
                                obs["elbow_flex.pos"],
                                obs["wrist_flex.pos"],
                                obs["wrist_roll.pos"],
                                obs["gripper.pos"],
                            ],
                            "image": obs["cam_primary"].tolist(),
                        },
                        "prompt": "pick up the shell and place it on the table",
                    }

                    req = Request(
                        f"http://{args.policy_host}:{args.policy_port}/act",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    resp = json.loads(urlopen(req).read())
                    action = np.array(resp["action"], dtype=np.float64)
                    adapter.send_action(action)
                    steps += 1

                results[object_type].append(steps)
                env.close()

        print("\n=== Evaluation results ===")
        for shell in ("shell_a", "shell_b"):
            ep_steps = results[shell]
            print(f"{shell}: {np.mean(ep_steps):.1f} ± {np.std(ep_steps):.1f} steps (n={len(ep_steps)})")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
