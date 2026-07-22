"""LeRobot-compatible adapter for the Isaac Sim SO-101 environment.

Wraps ``IsaacSO101Env`` to produce observations and accept actions in the
LeRobot SO101Follower convention (flat dict with ``<joint>.pos`` entries
and camera images).
"""

from __future__ import annotations

import numpy as np

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class LerobotAdapter:
    """Wraps IsaacSO101Env for LeRobot-compatible observation/action I/O.

    Args:
        env: A constructed and reset ``IsaacSO101Env`` instance.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def get_observation(self) -> dict[str, float | np.ndarray]:
        """Return LeRobot flat dict observation."""
        return self._env.get_observation()

    def send_action(self, action: dict[str, float] | np.ndarray) -> None:
        """Step the environment with a joint-target action.

        Accepts either a LeRobot-style dict or a numpy array of absolute
        joint targets (radians, 6D).
        """
        if isinstance(action, dict):
            arr = np.array(
                [action.get(f"{j}.pos", 0.0) for j in JOINT_ORDER],
                dtype=np.float64,
            )
        else:
            arr = np.asarray(action, dtype=np.float64)

        self._env.step(arr)

    @property
    def env(self) -> Any:
        """Access the underlying IsaacSO101Env."""
        return self._env
