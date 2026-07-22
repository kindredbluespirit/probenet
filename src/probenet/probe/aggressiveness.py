"""Aggressiveness steering — continuous risk-tolerance modulation.

Maps a user-specified aggressiveness value ∈ [0, 1] to behavior parameters
for both Stage 1 (probing) and Stage 2 (manipulation).

Behavior modulation:
    low agg (0.0–0.2): gentle taps, slow squeezes, wide safety margins
    med agg (0.3–0.6): moderate pressure, standard sequence
    high agg (0.7–1.0): firm squeezes, shake test, tight grip, fast motions

The aggressiveness token is injected into the VLA prompt alongside property
tokens. The model learns to interpret it as a continuous conditioning signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AggressivenessParams:
    """Per-phase behavior parameters modulated by aggressiveness."""

    # --- Probing ---
    probe_force_max: float = 1.0  # max force allowed during probing
    probe_speed: float = 1.0  # speed multiplier for probing actions
    probe_tap_force: float = 0.1  # force for exploration taps
    probe_hold_duration: float = 100  # hold steps during squeeze

    # --- Manipulation ---
    grip_force: float = 0.5  # gripper closing force
    approach_speed: float = 1.0  # speed multiplier for approach
    safety_margin: float = 0.05  # collision/clearance margin (meters)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_aggressiveness_params(aggressiveness: float) -> AggressivenessParams:
    """Map continuous aggressiveness to behavior parameters.

    The mapping is intentionally simple — as the model trains with
    aggressiveness tokens, it learns to interpret the value contextually.
    These params serve as guardrails for safety during early training.
    """
    aggressiveness = max(0.0, min(1.0, aggressiveness))

    return AggressivenessParams(
        probe_force_max=_lerp(0.3, 5.0, aggressiveness),
        probe_speed=_lerp(0.3, 2.0, aggressiveness),
        probe_tap_force=_lerp(0.05, 1.0, aggressiveness),
        probe_hold_duration=_lerp(50, 200, aggressiveness),
        grip_force=_lerp(0.2, 1.0, aggressiveness),
        approach_speed=_lerp(0.3, 2.0, aggressiveness),
        safety_margin=_lerp(0.10, 0.005, aggressiveness),
    )


# ── Aggressiveness token helpers ──────────────────────────────────────────


AGGRESSIVENESS_LEVELS = {
    (0.0, 0.2): "very low",
    (0.2, 0.4): "low",
    (0.4, 0.6): "medium",
    (0.6, 0.8): "high",
    (0.8, 1.0): "very high",
}


def aggressiveness_to_level(value: float) -> str:
    for (lo, hi), label in AGGRESSIVENESS_LEVELS.items():
        if lo <= value <= hi or (value >= 1.0 and label == "very high"):
            return label
    return "medium"


# ── Action modulation ─────────────────────────────────────────────────────


class AggressivenessModulator:
    """Modulates policy actions based on aggressiveness.

    Args:
        aggressiveness: Risk tolerance [0, 1].
    """

    def __init__(self, aggressiveness: float = 0.3) -> None:
        self._agg = max(0.0, min(1.0, aggressiveness))

    @property
    def aggressiveness(self) -> float:
        return self._agg

    @aggressiveness.setter
    def aggressiveness(self, value: float) -> None:
        self._agg = max(0.0, min(1.0, value))

    def modulate_action(
        self,
        action: float | list[float] | Any,
        mode: str = "manipulation",
    ) -> Any:
        """Modulate action based on current aggressiveness.

        During probing, high aggressiveness increases exploration force.
        During manipulation, high aggressiveness tightens grip and speeds up.
        """
        import numpy as np

        arr = np.asarray(action, dtype=np.float32)
        params = get_aggressiveness_params(self._agg)

        if mode == "probing":
            # scale force components (all but gripper)
            arr[:-1] = np.clip(arr[:-1] * params.probe_force_max, -1.0, 1.0)
        else:
            # scale gripper (last element)
            arr[-1] = np.clip(arr[-1] * params.grip_force, 0.0, 1.0)

        return arr

    def modulate_trajectory(
        self,
        trajectory: list[Any],
        mode: str = "manipulation",
    ) -> list[Any]:
        """Modulate a full trajectory chunk."""
        return [self.modulate_action(a, mode) for a in trajectory]
