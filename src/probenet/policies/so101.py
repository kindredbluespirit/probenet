"""Shared observation and action specification for SO-101.

This defines the canonical joint order, camera layout, and feature shapes
used by all policy backends (π₀.₅, GR00T). Every backend wraps this spec
with its own TransformFn or EmbodimentTag.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

STATE_DIM = 6
ACTION_DIM = 6

# Camera keys as they appear in LeRobot datasets.
CAMERA_KEYS = ["cam_primary", "cam_wrist"]

# openpi expects images in a dict of named views.
OPENPI_IMAGE_KEYS = {
    "cam_primary": "base_0_rgb",
    "cam_wrist": "left_wrist_0_rgb",
}

# LeRobot feature spec for recording / training.
LEROBOT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": JOINT_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": JOINT_NAMES,
    },
    "observation.images.cam_primary": {
        "dtype": "video",
        "shape": (3, 224, 224),
    },
}


def joint_pos_to_array(obs: dict[str, float | np.ndarray]) -> np.ndarray:
    """Extract joint positions from a LeRobot flat obs dict."""
    return np.array([float(obs.get(f"{j}.pos", 0.0)) for j in JOINT_NAMES], dtype=np.float32)


def joint_vel_to_array(obs: dict[str, float | np.ndarray]) -> np.ndarray:
    """Extract joint velocities from a LeRobot flat obs dict."""
    return np.array([float(obs.get(f"{j}.vel", 0.0)) for j in JOINT_NAMES], dtype=np.float32)


def dict_to_action(action: dict[str, float] | np.ndarray) -> np.ndarray:
    """Normalise action dict to a flat ``(6,)`` float32 array."""
    if isinstance(action, dict):
        return joint_pos_to_array(action)
    return np.asarray(action, dtype=np.float32).ravel()[:ACTION_DIM]
