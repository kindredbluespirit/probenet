"""Real hardware runner — LeRobot SO-101 follower integration.

Wraps LeRobot's native ``SO101Follower`` class for teleoperation recording
and policy deployment on the physical arm.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from probenet.policies.so101 import JOINT_NAMES, joint_pos_to_array

logger = logging.getLogger(__name__)


class RealSO101Runner:
    """Interface to the physical SO-101 arm via LeRobot.

    Args:
        follower_port: USB serial port for the follower arm (e.g. /dev/ttyACM0).
        leader_port: USB serial port for the leader/teleop arm (optional).
        camera_ids: Dict ``{name: device_path}`` for V4L cameras.
    """

    def __init__(
        self,
        follower_port: str = "/dev/ttyACM0",
        leader_port: str | None = None,
        camera_ids: dict[str, str] | None = None,
    ) -> None:
        self._follower_port = follower_port
        self._leader_port = leader_port
        self._camera_ids = camera_ids or {}
        self._robot = None
        self._cameras: dict[str, Any] = {}
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect to the physical SO-101 arm and cameras."""
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

            config = SO101FollowerConfig(
                port=self._follower_port,
                leader_port=self._leader_port,
            )
            self._robot = SO101Follower(config=config)
            self._robot.connect()
            self._connected = True
            logger.info("Connected to SO-101 follower at %s", self._follower_port)
        except ImportError:
            logger.warning("LeRobot not installed — real hardware unavailable")
        except Exception:
            logger.exception("Failed to connect to SO-101")

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
        self._connected = False

    def get_observation(self) -> dict[str, float | np.ndarray]:
        """Return LeRobot-format observation from the physical arm."""
        if not self._connected or self._robot is None:
            return {}
        robot_obs = self._robot.get_observation()
        obs: dict[str, float | np.ndarray] = {}
        for joint, val in zip(JOINT_NAMES, robot_obs.get("joint_positions", [0] * 6)):
            obs[f"{joint}.pos"] = float(val)
        for cam_name, cam in self._cameras.items():
            obs[f"cam_{cam_name}"] = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float] | np.ndarray) -> None:
        """Send a joint-target action to the physical arm."""
        if not self._connected or self._robot is None:
            return
        arr = joint_pos_to_array(action) if isinstance(action, dict) else np.asarray(action)
        self._robot.send_action(arr.reshape(1, -1))

    def teleop_record(
        self,
        dataset_repo_id: str,
        num_episodes: int = 50,
    ) -> None:
        """Record teleoperation demonstrations via LeRobot recorder.

        Requires the leader arm to be connected.
        """
        logger.info("Teleop recording: %d episodes → %s", num_episodes, dataset_repo_id)
        # Actual recording handled by LeRobot's lerobot-record CLI
        logger.info("Run: lerobot-record --robot.type=so101_follower ...")


class DummyRobotRunner:
    """Fake hardware runner for testing without a physical arm."""

    def __init__(self):
        self._joints = np.zeros(6, dtype=np.float32)
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> dict[str, float | np.ndarray]:
        return {f"{j}.pos": float(v) for j, v in zip(JOINT_NAMES, self._joints)}

    def send_action(self, action: np.ndarray) -> None:
        self._joints = np.asarray(action, dtype=np.float32).flatten()[:6]

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass
