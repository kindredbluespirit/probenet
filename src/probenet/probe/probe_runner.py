"""Scripted probe and signal processing for the SO-101 arm.

The probe executes a fixed squeeze + lift-hold sequence on the object and logs
actuator forces / joint torques, which serve as the MuJoCo analog of servo
current readings on the physical robot.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProbePhase:
    """A single phase of the scripted probe.

    Args:
        name: Human-readable phase label.
        duration_steps: Number of simulation steps to hold this phase.
        target_ctrl: Absolute actuator position targets for the arm (size 6).
        gripper: Absolute gripper position target. If ``None``, the gripper is
            left unchanged.
    """

    name: str
    duration_steps: int
    target_ctrl: np.ndarray
    gripper: float | None


@dataclass
class ProbeConfig:
    """Default probe sequence for the SO-101 arm.

    The sequence is intentionally simple and open-loop. It can be replaced by
    a learned or adaptive strategy later.
    """

    approach_ctrl: np.ndarray
    squeeze_ctrl: np.ndarray
    lift_ctrl: np.ndarray
    hold_steps: int = 100
    squeeze_steps: int = 100
    lift_steps: int = 100
    approach_steps: int = 200

    def build_phases(self) -> list[ProbePhase]:
        """Return the default probe phase sequence."""
        return [
            ProbePhase(
                "approach",
                self.approach_steps,
                self.approach_ctrl,
                0.0,
            ),
            ProbePhase(
                "squeeze",
                self.squeeze_steps,
                self.squeeze_ctrl,
                0.8,
            ),
            ProbePhase(
                "lift",
                self.lift_steps,
                self.lift_ctrl,
                0.8,
            ),
            ProbePhase(
                "hold",
                self.hold_steps,
                self.lift_ctrl,
                0.8,
            ),
            ProbePhase(
                "release",
                self.squeeze_steps,
                self.lift_ctrl,
                0.0,
            ),
        ]


def default_probe_config() -> ProbeConfig:
    """Return a conservative probe config for the SO-101.

    The joint targets are chosen so the gripper is positioned above the object
    at the default reset pose (object at x=0.25, z=0.035).
    """
    # Pre-grasp: above object, gripper open.
    approach_ctrl = np.array([0.0, -0.5, 0.8, 0.8, 1.58, 0.0], dtype=np.float32)
    # Squeeze: same pose, gripper closed.
    squeeze_ctrl = np.array([0.0, -0.5, 0.8, 0.8, 1.58, 1.5], dtype=np.float32)
    # Lift: raise the arm while keeping the gripper over the object.
    lift_ctrl = np.array([0.0, -0.7, 0.8, 0.9, 1.58, 1.5], dtype=np.float32)
    return ProbeConfig(
        approach_ctrl=approach_ctrl,
        squeeze_ctrl=squeeze_ctrl,
        lift_ctrl=lift_ctrl,
    )


class ProbeRunner:
    """Run a scripted probe sequence in an environment and log the signal.

    Args:
        env: A ``SO101Env`` instance.
        config: ``ProbeConfig`` describing the probe sequence.
    """

    def __init__(self, env: Any, config: ProbeConfig | None = None) -> None:
        self.env = env
        self.config = config or default_probe_config()
        self.phases = self.config.build_phases()

    def _ctrl_to_action(self, ctrl: np.ndarray) -> np.ndarray:
        """Convert absolute actuator positions to the env's normalized action."""
        ctrl_range = self.env.model.actuator_ctrlrange
        low, high = ctrl_range[:, 0], ctrl_range[:, 1]
        return 2.0 * (ctrl - low) / (high - low) - 1.0

    def run(self) -> dict[str, Any]:
        """Execute the probe sequence and return the logged signal.

        Returns:
            A dict with keys:
              - ``phase_names``: list of phase names per timestep.
              - ``actuator_force``: (T, nu) array of actuator forces.
              - ``qfrc_actuator``: (T, nv) array of joint torques.
              - ``ctrl``: (T, nu) array of commanded controls.
              - ``timestamps``: (T,) array of simulation times.
              - ``phase_boundaries``: list of (start, end) step indices.
        """
        actuator_force_log: list[np.ndarray] = []
        qfrc_actuator_log: list[np.ndarray] = []
        ctrl_log: list[np.ndarray] = []
        timestamps: list[float] = []
        phase_names: list[str] = []
        phase_boundaries: list[tuple[int, int]] = []
        step = 0

        for phase in self.phases:
            start = step
            target = phase.target_ctrl.copy()
            if phase.gripper is not None:
                target[-1] = phase.gripper
            action = self._ctrl_to_action(target)

            for _ in range(phase.duration_steps):
                _, _, _, _, _ = self.env.step(action)
                signal = self.env.get_probe_signal()
                actuator_force_log.append(signal["actuator_force"])
                qfrc_actuator_log.append(signal["qfrc_actuator"])
                ctrl_log.append(np.array(self.env.data.ctrl, dtype=np.float32))
                timestamps.append(float(self.env.data.time))
                phase_names.append(phase.name)
                step += 1

            phase_boundaries.append((start, step))

        return {
            "phase_names": phase_names,
            "actuator_force": np.stack(actuator_force_log),
            "qfrc_actuator": np.stack(qfrc_actuator_log),
            "ctrl": np.stack(ctrl_log),
            "timestamps": np.array(timestamps, dtype=np.float32),
            "phase_boundaries": phase_boundaries,
        }


def extract_probe_features(signal: dict[str, Any]) -> dict[str, float]:
    """Extract simple statistics from a probe signal.

    These features are meant to be interpretable and can feed a small MLP
    physical-parameter estimator.
    """
    actuator_force = signal["actuator_force"]
    qfrc_actuator = signal["qfrc_actuator"]
    return {
        "af_mean": float(np.mean(actuator_force)),
        "af_max": float(np.max(actuator_force)),
        "af_std": float(np.std(actuator_force)),
        "qfa_mean": float(np.mean(qfrc_actuator)),
        "qfa_max": float(np.max(qfrc_actuator)),
        "qfa_std": float(np.std(qfrc_actuator)),
    }
