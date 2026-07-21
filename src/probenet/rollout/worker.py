"""Rollout worker — polls HF Hub for checkpoints and collects Isaac Sim episodes.

Runs on the Lambda A10 instance. The worker:
1. Polls HF Hub for the latest model checkpoint.
2. Downloads the checkpoint and creates a LeIsaac task environment.
3. Uses the scripted oracle (state machine) to collect episodes.
4. Exports each episode as a LeRobotDataset and uploads to HF Hub.

Phase 1 (BC): scripted oracle only. Phase 2+ (RL): policy rollout + exploration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from probenet.sync import SyncDaemon, make_rollout_id

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """Configuration for rollout worker."""

    hf_model_repo: str = ""
    hf_dataset_repo: str = ""

    local_ckpt_dir: str = "outputs/checkpoints"
    local_data_dir: str = "outputs/eval_runs"

    strategy: str = "scripted_oracle"
    num_episodes: int = 100
    worker_id: str = ""
    headless: bool = True

    poll_interval: float = 30.0
    max_cycles: int = 0  # 0 = run forever

    dataset_repo_id: str = "probenet/so101_sim"

    def __post_init__(self) -> None:
        Path(self.local_ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.local_data_dir).mkdir(parents=True, exist_ok=True)


class RolloutWorker:
    """Rollout collection loop with sync daemon.

    Args:
        config: Rollout configuration.
    """

    def __init__(self, config: RolloutConfig) -> None:
        self._config = config
        self._daemon: SyncDaemon | None = None
        self._pending_checkpoints: list[tuple[str, int]] = []
        self._current_model_step: int | None = None
        self._cycle = 0

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the rollout collection loop."""
        logger.info("Rollout worker starting (strategy=%s)", self._config.strategy)

        self._start_sync_daemon()

        while True:
            if self._process_next_checkpoint():
                continue

            self._cycle += 1
            if self._config.max_cycles and self._cycle >= self._config.max_cycles:
                break

            import time

            time.sleep(self._config.poll_interval)

        if self._daemon is not None:
            self._daemon.stop()

    # ── Collection ────────────────────────────────────────────────────────

    def _process_next_checkpoint(self) -> bool:
        """If a new checkpoint is queued, collect episodes."""
        if not self._pending_checkpoints:
            return False

        ckpt_dir, model_step = self._pending_checkpoints.pop(0)
        logger.info("Processing checkpoint step %d: %s", model_step, ckpt_dir)
        self._collect_episodes(model_step)
        return True

    def _collect_episodes(self, model_step: int) -> None:
        """Run scripted oracle data collection for one model step."""
        rollout_id = make_rollout_id(model_step, self._config.strategy, self._config.worker_id or None)
        run_dir = str(Path(self._config.local_data_dir) / rollout_id / "eval_dataset")

        logger.info(
            "Collecting %d episodes (sim requires Isaac Sim + LeIsaac) → %s",
            self._config.num_episodes,
            run_dir,
        )

        # TODO: wire to actual Isaac Sim + LeIsaac env
        # 1. Create LeIsaac task + IsaacSO101Env
        # 2. Run scripted oracle (state machine) per episode
        # 3. Export to LeRobotDataset
        # 4. Call self._daemon.upload_rollout(run_dir, rollout_id)

        logger.info("Rollout upload ready: %s", rollout_id)

    # ── Sync daemon ───────────────────────────────────────────────────────

    def _start_sync_daemon(self) -> None:
        self._daemon = SyncDaemon(
            hf_model_repo=self._config.hf_model_repo,
            hf_dataset_repo=self._config.hf_dataset_repo,
            local_ckpt_dir=self._config.local_ckpt_dir,
            local_data_dir=self._config.local_data_dir,
            mode="rollout",
            on_new_checkpoint=self._on_new_checkpoint,
            poll_interval=self._config.poll_interval,
        )
        self._daemon.start()

    def _on_new_checkpoint(self, ckpt_dir: str, step: int) -> None:
        """Callback when a new checkpoint is downloaded."""
        if self._current_model_step is not None and step <= self._current_model_step:
            return
        self._current_model_step = step
        self._pending_checkpoints.append((ckpt_dir, step))
