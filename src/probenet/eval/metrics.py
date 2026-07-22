"""Evaluation pipeline — per-object metrics, success detection, aggregation.

Tracks success/failure modes, episode statistics, and per-property-group
breakdowns for the ablation studies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    episode_id: int
    success: bool
    length: int
    failure_mode: str = ""  # "drop", "not_reached", "collision", "timeout", ""
    object_type: str = ""
    properties: dict[str, float] = field(default_factory=dict)
    episode_reward: float = 0.0


@dataclass
class ObjectStats:
    """Aggregated statistics for one object type / property group."""

    object_type: str = ""
    total: int = 0
    successes: int = 0
    failures: int = 0
    failure_modes: dict[str, int] = field(default_factory=dict)
    avg_length: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0


@dataclass
class EvalStats:
    """Aggregated evaluation statistics across all episodes."""

    total_episodes: int = 0
    total_successes: int = 0
    avg_length: float = 0.0
    by_object: dict[str, ObjectStats] = field(default_factory=dict)
    results: list[EpisodeResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.total_successes / self.total_episodes if self.total_episodes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_successes": self.total_successes,
            "success_rate": self.success_rate,
            "avg_length": self.avg_length,
            "by_object": {
                k: {
                    "successes": v.successes,
                    "failures": v.failures,
                    "success_rate": v.success_rate,
                    "avg_length": v.avg_length,
                    "failure_modes": v.failure_modes,
                }
                for k, v in self.by_object.items()
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


class EvalRunner:
    """Run policy evaluation episodes and aggregate results.

    Args:
        env_factory: Callable that returns a fresh env instance per episode.
        policy: A policy object with an ``infer(obs)`` method.
        success_detector: Callable ``(obs, info) -> bool``.
    """

    def __init__(self, env_factory: Any, policy: Any, success_detector: Any = None) -> None:
        self._env_factory = env_factory
        self._policy = policy
        self._success_detector = success_detector or (lambda o, i: False)

    def run(self, num_episodes: int = 10, max_steps: int = 500) -> EvalStats:
        stats = EvalStats()
        for ep_id in range(num_episodes):
            result = self._run_episode(ep_id, max_steps)
            stats.results.append(result)
            stats.total_episodes += 1
            if result.success:
                stats.total_successes += 1
            if result.object_type not in stats.by_object:
                stats.by_object[result.object_type] = ObjectStats(object_type=result.object_type)
            ostats = stats.by_object[result.object_type]
            ostats.total += 1
            if result.success:
                ostats.successes += 1
            else:
                ostats.failures += 1
                ostats.failure_modes[result.failure_mode] = ostats.failure_modes.get(result.failure_mode, 0) + 1
            ostats.avg_length = (ostats.avg_length * (ostats.total - 1) + result.length) / ostats.total

        stats.avg_length = sum(r.length for r in stats.results) / max(1, stats.total_episodes)
        return stats

    def _run_episode(self, ep_id: int, max_steps: int) -> EpisodeResult:
        env = self._env_factory()
        obs = env.reset()
        for step in range(max_steps):
            action = self._policy.infer(obs)
            obs, _, done, info = env.step(action)
            if done:
                success = self._success_detector(obs, info) if not isinstance(done, bool) or not done else False
                return EpisodeResult(
                    episode_id=ep_id,
                    success=success,
                    length=step + 1,
                    failure_mode="" if success else "timeout" if step >= max_steps - 1 else "drop",
                    object_type=getattr(env, "_current_object_type", "unknown"),
                )
            if step >= max_steps - 1:
                return EpisodeResult(
                    episode_id=ep_id,
                    success=False,
                    length=step + 1,
                    failure_mode="timeout",
                    object_type=getattr(env, "_current_object_type", "unknown"),
                )
        return EpisodeResult(episode_id=ep_id, success=False, length=max_steps, failure_mode="timeout")


# ── Success detectors ────────────────────────────────────────────────────────


def pick_place_success(
    obs: dict[str, Any],
    target_height: float = 0.15,
    table_height: float = 0.03,
) -> bool:
    """Success detector: object lifted above table by target_height."""
    z = obs.get("gripper.pos", 0.0)
    return z > table_height + target_height


def proximity_success(
    obs: dict[str, Any],
    target_position: tuple[float, float, float] = (0.3, 0.0, 0.03),
    threshold: float = 0.05,
) -> bool:
    """Success detector: object within threshold distance of target."""
    ox = obs.get("object_x", 0.0)
    oy = obs.get("object_y", 0.0)
    oz = obs.get("object_z", 0.0)
    dist = ((ox - target_position[0]) ** 2 + (oy - target_position[1]) ** 2 + (oz - target_position[2]) ** 2) ** 0.5
    return dist < threshold
