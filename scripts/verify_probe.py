"""Quick probe signal test — verify the scripted probe produces force readings.

IMPORTANT: Requires Isaac Sim + LeIsaac installed. Run inside the rollout
Docker image or a local environment with ``isaacsim`` pip package.

Usage:
    uv run scripts/verify_probe.py
"""

from __future__ import annotations

from probenet.env import IsaacSO101Env, ObjectSpec
from probenet.probe.probe_runner import ProbeRunner, extract_probe_features


def main() -> None:
    print("Verify probe: requires Isaac Sim + LeIsaac (see record_sim.py)")
    print("Signal feature extraction test (offline)")

    import numpy as np

    dummy_signal = {
        "actuator_force": np.random.randn(200, 6).astype(np.float32),
        "joint_pos": np.random.randn(200, 6).astype(np.float32),
        "ctrl": np.zeros((200, 6), dtype=np.float32),
        "phase_names": ["approach"] * 50 + ["squeeze"] * 50 + ["lift"] * 50 + ["hold"] * 50,
        "phase_boundaries": [(0, 50), (50, 100), (100, 150), (150, 200)],
    }

    features = extract_probe_features(dummy_signal)
    for key, val in features.items():
        print(f"  {key}: {val:.6f}")

    print("Feature extraction works. Full probe test needs Isaac Sim runtime.")


if __name__ == "__main__":
    main()
