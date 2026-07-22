"""Learned probing pipeline — RECAP-style advantage conditioning.

Replaces scripted oracle probing with a learned probing policy. The probe
policy is conditioned on a "probe advantage" token that predicts whether the
current probing strategy will lead to downstream manipulation success.

Implementation follows the π₀.₆ RECAP pattern:
    1. Collect probing trajectories from the current policy.
    2. Run downstream manipulation with the estimated properties.
    3. Label each probe episode as "positive probe" or "negative probe".
    4. Add probe_advantage token to the prompt.
    5. Fine-tune π₀.₅ with advantage-conditioned data.
    6. Repeat (iterative improvement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probenet.probe.properties import PropertyState, build_probenet_prompt


@dataclass
class ProbeEpisode:
    """One probing episode with its downstream outcome."""

    episode_id: int
    object_type: str
    properties_gt: dict[str, float]
    properties_estimated: dict[str, float]
    probe_signal: dict[str, np.ndarray]
    downstream_success: bool
    probe_advantage: float = 0.0  # +1 for success, -1 for failure
    prompt_used: str = ""

    def __post_init__(self) -> None:
        self.probe_advantage = 1.0 if self.downstream_success else -1.0


@dataclass
class ProbeBuffer:
    """Replay buffer of probe episodes for iterative improvement."""

    episodes: list[ProbeEpisode] = field(default_factory=list)
    max_size: int = 10_000

    def add(self, episode: ProbeEpisode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.max_size:
            self.episodes = self.episodes[-self.max_size:]

    def sample_positive(self, n: int = 16) -> list[ProbeEpisode]:
        pos = [e for e in self.episodes if e.downstream_success]
        if len(pos) < n:
            return pos
        return list(np.random.choice(pos, size=n, replace=False))

    def sample_negative(self, n: int = 16) -> list[ProbeEpisode]:
        neg = [e for e in self.episodes if not e.downstream_success]
        if len(neg) < n:
            return neg
        return list(np.random.choice(neg, size=n, replace=False))

    def balanced_sample(self, n: int = 32) -> list[ProbeEpisode]:
        return self.sample_positive(n // 2) + self.sample_negative(n // 2)

    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.downstream_success) / len(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass
class LearnedProbingConfig:
    """Configuration for the learned probing pipeline."""

    max_probe_steps: int = 300
    min_probe_steps: int = 50
    confidence_threshold: float = 0.9

    advantage_positive: float = 1.0
    advantage_negative: float = -1.0
    advantage_unknown: float = 0.0

    probe_buffer_size: int = 10_000
    improvement_iterations: int = 10
    episodes_per_iteration: int = 100

    use_probe_encoder: bool = True  # Phase 12: learned encoder
    use_scripted_fallback: bool = True  # Fallback for early iterations


class LearnedProbeRunner:
    """Runs the learned probing loop — iterate: probe → task → evaluate → retrain.

    Args:
        env_factory: Callable that returns a fresh env.
        policy: Policy with ``infer(obs)`` method (probe-mode conditioned).
        probe_encoder: ``ProbeEncoder`` for estimating properties from signal.
        config: Pipeline configuration.
    """

    def __init__(
        self,
        env_factory: Any,
        policy: Any,
        probe_encoder: Any = None,
        config: LearnedProbingConfig | None = None,
    ) -> None:
        self._env_factory = env_factory
        self._policy = policy
        self._probe_encoder = probe_encoder
        self._config = config or LearnedProbingConfig()
        self._buffer = ProbeBuffer(max_size=self._config.probe_buffer_size)

    def run_iteration(self, num_episodes: int | None = None) -> ProbeBuffer:
        """Run one iteration of the learned probing improvement loop.

        Returns:
            Updated probe buffer.
        """
        n = num_episodes or self._config.episodes_per_iteration
        for ep_id in range(n):
            episode = self._run_single_probe_episode(ep_id)
            self._buffer.add(episode)
        return self._buffer

    def _run_single_probe_episode(self, ep_id: int) -> ProbeEpisode:
        env = self._env_factory()

        # Phase 1: probing
        obs = env.reset()
        history: dict[str, list[np.ndarray]] = {
            "signals": [], "images": [], "actions": [], "states": [],
        }

        for step in range(self._config.max_probe_steps):
            prompt = build_probenet_prompt(
                "Explore this object to understand its physical properties",
                PropertyState.empty(),
                probe_mode="probing",
                aggressiveness=0.3,
            )
            obs["prompt"] = prompt
            action = self._policy.infer(obs)
            obs, _, done, info = env.step(action)

            history["signals"].append(env.get_probe_signal().get("actuator_force", np.zeros(6)))
            history["actions"].append(action)
            history["states"].append(
                np.array([obs.get(f"{j}.pos", 0.0) for j in [
                    "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll", "gripper",
                ]])
            )

            if step >= self._config.min_probe_steps and self._probe_encoder is not None:
                signal_t = np.stack(history["signals"])
                props = self._probe_encoder.predict(
                    torch.tensor(signal_t).unsqueeze(0),
                )
                if all(v["confidence"] > self._config.confidence_threshold for v in props.values()):
                    break

        # Phase 2: estimate properties
        properties_estimated = getattr(env, "get_properties", lambda: {})()

        # Phase 3: downstream manipulation (scripted for now)
        success = self._run_downstream_task(env, properties_estimated)

        env.close()

        return ProbeEpisode(
            episode_id=ep_id,
            object_type=getattr(env, "_current_object_type", "unknown"),
            properties_gt=getattr(env, "get_properties", lambda: {})(),
            properties_estimated=properties_estimated,
            probe_signal={
                "actuator_force": np.stack(history["signals"]),
                "joint_pos": np.stack(history["states"]),
            },
            downstream_success=success,
        )

    def _run_downstream_task(self, env: Any, properties: dict[str, float]) -> bool:
        """Run the downstream manipulation task with estimated properties.

        Returns ``True`` on success.
        """
        obs = env.reset()
        props_state = PropertyState(values=properties, confidences={k: 0.8 for k in properties})

        for _step in range(500):
            prompt = build_probenet_prompt(
                "Pick up the object and place it on the shelf",
                props_state,
                probe_mode="manipulation",
                aggressiveness=0.3,
            )
            obs["prompt"] = prompt
            action = self._policy.infer(obs)
            obs, _, done, info = env.step(action)
            if done:
                return True
        return False


# Torch import deferred to avoid CI requirement
try:
    import torch  # noqa: F811
except ImportError:
    torch = None  # type: ignore[assignment]
