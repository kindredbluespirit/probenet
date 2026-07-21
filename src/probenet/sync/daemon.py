"""Background sync daemon — polls HF Hub for new checkpoints and datasets.

Runs as a background thread inside the trainer / rollout worker process.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from probenet.sync import hub

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL = 30  # seconds


class SyncDaemon:
    """Background daemon for HF Hub coordination.

    Args:
        hf_model_repo: HuggingFace repo ID for model checkpoints.
        hf_dataset_repo: HuggingFace repo ID for rollout datasets.
        local_ckpt_dir: Local directory for downloaded checkpoints.
        local_data_dir: Local directory for downloaded datasets.
        mode: ``"trainer"`` or ``"rollout"`` — controls which side of the
            sync loop this daemon handles.
        on_new_checkpoint: Callback invoked when a new checkpoint is
            available. Receives ``(checkpoint_dir, step)``.
        on_new_rollout: Callback invoked when new rollout data is available.
            Receives ``(rollout_id, local_path)``.
        poll_interval: Seconds between HF Hub polls.
    """

    def __init__(
        self,
        hf_model_repo: str,
        hf_dataset_repo: str,
        local_ckpt_dir: str,
        local_data_dir: str,
        mode: str,
        on_new_checkpoint: Callable[[str, int], None] | None = None,
        on_new_rollout: Callable[[str, str], None] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._hf_model_repo = hf_model_repo
        self._hf_dataset_repo = hf_dataset_repo
        self._local_ckpt_dir = local_ckpt_dir
        self._local_data_dir = local_data_dir
        self._mode = mode
        self._on_new_checkpoint = on_new_checkpoint
        self._on_new_rollout = on_new_rollout
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_rollout_ids: set[str] = set()
        self._last_model_step: int | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the daemon as a background thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="sync-daemon")
        self._thread.start()
        logger.info("Sync daemon started (mode=%s, poll=%ds)", self._mode, self._poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the daemon to stop and wait for the thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Sync daemon stopped")

    def upload_checkpoint(self, checkpoint_dir: str, step: int) -> bool:
        """Upload a checkpoint (called from the training loop)."""
        return hub.upload_checkpoint_step(checkpoint_dir, step, self._hf_model_repo)

    def upload_rollout(self, eval_run_dir: str, rollout_id: str) -> bool:
        """Upload a rollout dataset (called after collection)."""
        return hub.upload_rollout_dataset(eval_run_dir, rollout_id, self._hf_dataset_repo)

    # ── Internal loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._mode == "trainer":
                    self._poll_datasets()
                elif self._mode == "rollout":
                    self._poll_checkpoints()
            except Exception:
                logger.exception("Sync daemon poll failed")
            self._stop_event.wait(self._poll_interval)

    def _poll_checkpoints(self) -> None:
        step = hub.get_latest_model_step(self._hf_model_repo)
        if step is None or step == self._last_model_step:
            return
        result = hub.download_latest_checkpoint(self._hf_model_repo, self._local_ckpt_dir)
        if result is not None and self._on_new_checkpoint is not None:
            ckpt_dir, step = result
            self._last_model_step = step
            self._on_new_checkpoint(ckpt_dir, step)

    def _poll_datasets(self) -> None:
        rollouts = hub.list_available_rollouts(self._hf_dataset_repo)
        for info in rollouts:
            rid = info["rollout_id"]
            if rid in self._seen_rollout_ids:
                continue
            local_path = hub.download_rollout_dataset(self._hf_dataset_repo, rid, self._local_data_dir)
            if local_path is not None:
                self._seen_rollout_ids.add(rid)
                if self._on_new_rollout is not None:
                    self._on_new_rollout(rid, local_path)
