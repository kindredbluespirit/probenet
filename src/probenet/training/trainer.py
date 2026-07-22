"""Training orchestrator — manages BC/RL training loop with sync daemon.

The trainer runs on the Lambda A100 instance. It:
1. Downloads initial BC dataset and computes norm stats.
2. Runs supervised fine-tuning (BC warmup).
3. In RL mode: polls HF Hub for rollout datasets, recomputes advantages,
   performs RL training, uploads new checkpoints.

When ProbeNet conditioning is enabled, the trainer sets up the
``ProbeNetConditioner`` and augments observations with physical property
tokens before feeding them to the policy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from probenet.policies.probenet import ProbeNetConditioner, ProbeNetConfig
from probenet.sync.daemon import SyncDaemon
from probenet.training.config import PolicyName, TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator with sync daemon and ProbeNet conditioning support.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._daemon: SyncDaemon | None = None
        self._conditioner: ProbeNetConditioner | None = None
        self._current_step = 0
        self._rollout_queue: list[tuple[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full training pipeline (Phase 1: BC only for now)."""
        logger.info("Trainer starting (policy=%s, probenet=%s)", self._config.policy, self._config.probenet_enabled)

        if self._config.probenet_enabled:
            self._setup_conditioner()

        self._start_sync_daemon()
        self._run_bc_warmup()

        if self._daemon is not None:
            self._daemon.stop()

    def _run_bc_warmup(self) -> None:
        """Supervised BC fine-tuning on initial dataset."""
        logger.info("Running BC warmup (%d steps)", self._config.num_train_steps)

        if self._config.policy == "pi05":
            self._run_pi05_training()
        elif self._config.policy == "gr00t":
            self._run_gr00t_training()

    # ── ProbeNet conditioning ─────────────────────────────────────────────

    def _setup_conditioner(self) -> None:
        """Set up the ProbeNet conditioning module."""
        self._conditioner = ProbeNetConditioner(
            ProbeNetConfig(
                enabled=True,
                probe_mode_dropout=self._config.probe_mode_dropout,
                property_dropout=self._config.property_dropout,
                aggressiveness_dropout=self._config.aggressiveness_dropout,
                language_dropout=self._config.language_dropout,
            )
        )
        logger.info(
            "ProbeNet conditioning enabled (prop_drop=%.2f, probe_drop=%.2f, agg_drop=%.2f, lang_drop=%.2f)",
            self._config.property_dropout,
            self._config.probe_mode_dropout,
            self._config.aggressiveness_dropout,
            self._config.language_dropout,
        )

    # ── Policy-specific training ──────────────────────────────────────────

    def _run_pi05_training(self) -> None:
        """Kick off π₀.₅ fine-tuning via openpi's train script."""
        ckpt_dir = Path(self._config.output_dir) / self._config.exp_name
        self._current_step = self._config.num_train_steps

        mode = "ProbeNet" if self._config.probenet_enabled else "baseline"
        logger.info("π₀.₅ %s training placeholder — run via openpi CLI:", mode)
        logger.info(
            "  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "
            "pi05_so101 --exp-name=%s --overwrite",
            self._config.exp_name,
        )
        if self._config.probenet_enabled:
            logger.info("  → Data augmentation via ProbeNetConditioner active")
        logger.info("Checkpoint dir: %s", ckpt_dir)

        if self._daemon is not None:
            ckpt_path = ckpt_dir / str(self._config.num_train_steps)
            if ckpt_path.exists():
                self._daemon.upload_checkpoint(str(ckpt_path), self._current_step)

    def _run_gr00t_training(self) -> None:
        """Kick off GR00T fine-tuning via its launch_finetune script."""
        ckpt_dir = Path(self._config.output_dir) / self._config.exp_name
        self._current_step = self._config.num_train_steps

        mode = "ProbeNet" if self._config.probenet_enabled else "baseline"
        logger.info("GR00T %s training placeholder — run via GR00T CLI:", mode)
        logger.info(
            "  uv run torchrun --nproc_per_node=1 backends/isaac-gr00t/gr00t/experiment/launch_finetune.py "
            "--base-model-path=nvidia/GR00T-N1.7-3B "
            "--dataset-path=%s "
            "--embodiment-tag=SO101 "
            "--max-steps=%d",
            self._config.dataset_root,
            self._config.num_train_steps,
        )

        if self._daemon is not None:
            ckpt_path = ckpt_dir / str(self._config.num_train_steps)
            if ckpt_path.exists():
                self._daemon.upload_checkpoint(str(ckpt_path), self._current_step)

    # ── Sync daemon ───────────────────────────────────────────────────────

    def _start_sync_daemon(self) -> None:
        self._daemon = SyncDaemon(
            hf_model_repo=self._config.hf_model_repo,
            hf_dataset_repo=self._config.hf_dataset_repo,
            local_ckpt_dir=str(Path("outputs").resolve()),
            local_data_dir=str(Path("data").resolve()),
            mode="trainer",
            on_new_rollout=self._on_new_rollout,
        )
        self._daemon.start()

    def _on_new_rollout(self, rollout_id: str, local_path: str) -> None:
        """Callback when a new rollout dataset is downloaded."""
        logger.info("New rollout available: %s → %s", rollout_id, local_path)
        self._rollout_queue.append((rollout_id, local_path))
