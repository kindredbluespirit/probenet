"""Scripted probe and signal processing for the SO-101 arm via Isaac Sim.

The probe executes a fixed squeeze + lift-hold sequence on the object and logs
joint actuator forces, which serve as the sim analog of servo current readings
on the physical robot.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class ProbePhase:
    """A single phase of the scripted probe.

    Args:
        name: Human-readable phase label.
        duration_steps: Number of steps to hold this phase.
        target_joints: Absolute joint position targets ``(6,)`` rad.
        gripper: Absolute gripper position. ``None`` leaves gripper unchanged.
    """

    name: str
    duration_steps: int
    target_joints: np.ndarray
    gripper: float | None


@dataclass
class ProbeConfig:
    """Scripted probe sequence for the SO-101 arm.

    The sequence is intentionally simple and open-loop. It can be
    replaced by a learned or adaptive strategy later.
    """

    approach_joints: np.ndarray
    squeeze_joints: np.ndarray
    lift_joints: np.ndarray
    approach_steps: int = 200
    squeeze_steps: int = 100
    lift_steps: int = 100
    hold_steps: int = 100

    def build_phases(self) -> list[ProbePhase]:
        return [
            ProbePhase("approach", self.approach_steps, self.approach_joints, 0.0),
            ProbePhase("squeeze", self.squeeze_steps, self.squeeze_joints, 0.8),
            ProbePhase("lift", self.lift_steps, self.lift_joints, 0.8),
            ProbePhase("hold", self.hold_steps, self.lift_joints, 0.8),
            ProbePhase("release", self.squeeze_steps, self.lift_joints, 0.0),
        ]


def default_probe_config() -> ProbeConfig:
    """Return a conservative probe config for the SO-101.

    Joint targets are chosen so the gripper is positioned above the object
    at the default reset pose.
    """
    approach = np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32)
    squeeze = np.array([0.0, -0.5, 0.8, 0.8, 1.58, 1.5], dtype=np.float32)
    lift = np.array([0.0, -0.7, 0.8, 0.9, 1.58, 1.5], dtype=np.float32)
    return ProbeConfig(
        approach_joints=approach,
        squeeze_joints=squeeze,
        lift_joints=lift,
    )


class ProbeRunner:
    """Run a scripted probe sequence in an Isaac Sim env and log signals.

    Args:
        env: An ``IsaacSO101Env`` instance.
        config: ``ProbeConfig`` describing the probe sequence.
    """

    def __init__(self, env: Any, config: ProbeConfig | None = None) -> None:
        self.env = env
        self.config = config or default_probe_config()
        self.phases = self.config.build_phases()

    def run(self) -> dict[str, Any]:
        """Execute the probe sequence and return the logged signal.

        Returns:
            Dict with keys:
              - ``phase_names``: list of phase labels per timestep.
              - ``actuator_force``: ``(T, 6)`` array of joint forces.
              - ``joint_pos``: ``(T, 6)`` array of joint positions.
              - ``ctrl``: ``(T, 6)`` array of commanded joint targets.
              - ``phase_boundaries``: ``(start, end)`` step indices.
        """
        force_log: list[np.ndarray] = []
        pos_log: list[np.ndarray] = []
        ctrl_log: list[np.ndarray] = []
        phase_names: list[str] = []
        phase_boundaries: list[tuple[int, int]] = []
        step = 0

        for phase in self.phases:
            start = step
            target = phase.target_joints.copy()
            if phase.gripper is not None:
                target[-1] = phase.gripper

            for _ in range(phase.duration_steps):
                _, _, _, _ = self.env.step(target)

                signal = self.env.get_probe_signal()
                force_log.append(signal["actuator_force"])
                ctrl_log.append(target.copy())

                obs = self.env.get_observation()
                pos = np.array(
                    [obs.get(f"{j}.pos", 0.0) for j in JOINT_ORDER],
                    dtype=np.float32,
                )
                pos_log.append(pos)

                phase_names.append(phase.name)
                step += 1

            phase_boundaries.append((start, step))

        return {
            "phase_names": phase_names,
            "actuator_force": np.stack(force_log),
            "joint_pos": np.stack(pos_log),
            "ctrl": np.stack(ctrl_log),
            "phase_boundaries": phase_boundaries,
        }


def extract_probe_features(signal: dict[str, Any]) -> dict[str, float]:
    """Extract simple statistics from a probe signal.

    These features feed a small physical-parameter estimator MLP.
    """
    af = signal["actuator_force"]
    return {
        "af_mean": float(np.mean(af)),
        "af_max": float(np.max(af)),
        "af_std": float(np.std(af)),
    }
