"""Isaac Sim SO-101 environment with probe support.

Provides a gym-style interface wrapping Isaac Sim via the LeIsaac framework.
The scene includes the SO-101 arm, a table, and one or more probe objects with
randomized physical properties (mass, friction, compliance).

Joint actuator forces are logged during probing and exposed alongside
observations, mimicking the probe signal that would come from real servo
current readings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ObjectSpec:
    """Physical properties and appearance for a probe object."""

    name: str
    usd_path: str
    mass: float
    friction: tuple[float, float] = (0.5, 0.5)
    compliance: float = 0.0
    color: tuple[float, float, float, float] = (0.6, 0.35, 0.25, 1.0)
    spawn_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PropertyRange:
    """Range for randomizing a physical property."""

    name: str
    low: float
    high: float


DEFAULT_PROPERTY_RANGES: list[PropertyRange] = [
    PropertyRange("mass", 0.01, 5.0),
    PropertyRange("friction", 0.0, 2.0),
    PropertyRange("compliance", 0.0, 1.0),
]

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ── Observation helpers ──────────────────────────────────────────────────────


def _pack_joint_dict(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Convert joint array to LeRobot flat dict."""
    return {f"{joint}.{prefix}": float(v) for joint, v in zip(JOINT_NAMES, values)}


def _unpack_joint_dict(action: dict[str, float]) -> np.ndarray:
    """Convert LeRobot flat dict to joint array."""
    return np.array(
        [action.get(f"{joint}.pos", 0.0) for joint in JOINT_NAMES],
        dtype=np.float32,
    )


# ── Environment ───────────────────────────────────────────────────────────────


class IsaacSO101Env:
    """Isaac Sim SO-101 environment with probe-object support.

    This class wraps an Isaac Sim simulation instance using the LeIsaac
    task infrastructure. It expects a fully constructed LeIsaac task env
    (typically ``SingleArmTaskDirectEnv`` subclass) passed in externally.

    The caller is responsible for creating and tearing down the
    ``SimulationApp`` and the underlying Isaac Lab env. This wrapper adds
    property randomization, probe signal access, and LeRobot-compatible
    observation output.

    Args:
        leisaac_env: A pre-built LeIsaac direct task env.
        object_specs: One or more objects to spawn in the scene.
        property_ranges: Ranges for randomizing physical properties.
        headless: If ``True``, disable the viewer.
    """

    def __init__(
        self,
        leisaac_env: Any,
        object_specs: list[ObjectSpec] | None = None,
        property_ranges: list[PropertyRange] | None = None,
        headless: bool = True,
    ) -> None:
        self._env = leisaac_env
        self._object_specs = object_specs or []
        self._property_ranges = property_ranges or DEFAULT_PROPERTY_RANGES
        self._headless = headless
        self._device = getattr(leisaac_env, "device", torch.device("cuda:0"))
        self._num_envs = getattr(leisaac_env, "num_envs", 1)
        self._episode_step = 0
        self._current_properties: dict[str, dict[str, float]] = {}

    # ── Gym interface ──────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self._randomize_properties()
        self._episode_step = 0
        obs_dict, _ = self._env.reset()
        return self._pack_observation(obs_dict)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Step the environment with a joint-target action.

        Args:
            action: Joint position targets ``(6,)`` in radians for
                ``[pan, lift, elbow, wrist_flex, wrist_roll, gripper]``.

        Returns:
            ``(observation, reward, done, info)`` tuple.
        """
        action_tensor = torch.tensor(action, device=self._device).unsqueeze(0)
        obs_dict, reward, terminated, truncated, info = self._env.step(action_tensor)
        self._episode_step += 1
        done = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)
        obs = self._pack_observation(obs_dict)
        rew = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
        return obs, rew, done, info

    def get_observation(self) -> dict[str, float | np.ndarray]:
        """Return LeRobot-compatible flat observation dict."""
        return self._pack_observation(self._env._get_observations())

    def get_probe_signal(self) -> dict[str, np.ndarray]:
        """Return per-joint actuator forces for the current step.

        Returns a dict with keys:
        - ``actuator_force``: ``(6,)`` joint force/torque array.
        """
        robot = self._env.scene["robot"]
        forces = robot.data.applied_joint_effort
        return {"actuator_force": forces.squeeze(0).cpu().numpy()}

    def get_properties(self) -> dict[str, float]:
        """Return the current object's physical properties."""
        result: dict[str, float] = {}
        for obj_name, props in self._current_properties.items():
            for key, val in props.items():
                result[f"{obj_name}/{key}"] = val
        return result

    def close(self) -> None:
        self._env.close()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    # ── Internal ───────────────────────────────────────────────────────────

    def _randomize_properties(self) -> None:
        """Sample physical properties and apply them to USD prims."""
        self._current_properties = {}
        for spec in self._object_specs:
            props: dict[str, float] = {}
            for pr in self._property_ranges:
                if pr.name == "mass":
                    props["mass"] = np.random.uniform(pr.low, pr.high)
                elif pr.name == "friction":
                    props["friction"] = np.random.uniform(pr.low, pr.high)
                elif pr.name == "compliance":
                    props["compliance"] = np.random.uniform(pr.low, pr.high)
                else:
                    props[pr.name] = np.random.uniform(pr.low, pr.high)
            self._current_properties[spec.name] = props

            stage = getattr(self._env, "scene", None)
            if stage is None:
                continue
            stage_obj = getattr(stage, "stage", None)
            if stage_obj is None:
                continue

            prim = stage_obj.GetPrimAtPath(spec.usd_path)
            if not prim or not prim.IsValid():
                continue

            if "mass" in props:
                from pxr import UsdPhysics
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.GetMassAttr().Set(float(props["mass"]))

    def _pack_observation(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """Pack Isaac Lab observations into a flat LeRobot-style dict."""
        result: dict[str, Any] = {}
        policy_obs = obs_dict.get("policy", {})

        joint_pos = policy_obs.get("joint_pos")
        if joint_pos is not None:
            jp = joint_pos.squeeze(0).cpu().numpy() if isinstance(joint_pos, torch.Tensor) else np.asarray(joint_pos)
            result.update(_pack_joint_dict("pos", jp.flatten()))

        joint_vel = policy_obs.get("joint_vel")
        if joint_vel is not None:
            jv = joint_vel.squeeze(0).cpu().numpy() if isinstance(joint_vel, torch.Tensor) else np.asarray(joint_vel)
            result.update(_pack_joint_dict("vel", jv.flatten()))

        for cam_name in ("front", "wrist"):
            cam_data = policy_obs.get(cam_name)
            if cam_data is not None:
                img = cam_data.squeeze(0).cpu().numpy() if isinstance(cam_data, torch.Tensor) else np.asarray(cam_data)
                cam_key = "cam_primary" if cam_name == "front" else f"cam_{cam_name}"
                result[cam_key] = img

        result["properties"] = self.get_properties()

        return result
