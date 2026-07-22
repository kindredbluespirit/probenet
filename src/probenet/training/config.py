"""Training configuration for both policy backends.

Defines data configs, learning rate schedules, and training loops shared
across π₀.₅ and GR00T backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PolicyName = Literal["pi05", "gr00t"]


@dataclass
class TrainingConfig:
    """Common training configuration shared by both policy backends."""

    # HF Hub
    hf_model_repo: str = ""
    hf_dataset_repo: str = ""

    # Data
    dataset_root: str = "data/lerobot"
    bc_dataset_root: str = "data/lerobot"
    batch_size: int = 64
    num_workers: int = 4

    # Training
    policy: PolicyName = "pi05"
    num_train_steps: int = 100_000
    save_interval: int = 5_000
    keep_period: int = 10_000
    log_interval: int = 100

    # Optimizer
    lr_peak: float = 5e-5
    lr_warmup_steps: int = 10_000
    lr_decay_steps: int = 1_000_000
    lr_decay_lr: float = 5e-5
    clip_gradient_norm: float = 1.0

    # Checkpoint
    init_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base"
    output_dir: str = "outputs/checkpoints"
    resume: bool = False
    overwrite: bool = False

    # ProbeNet conditioning
    probenet_enabled: bool = False
    probe_mode_dropout: float = 0.1
    property_dropout: float = 0.25
    aggressiveness_dropout: float = 0.15
    language_dropout: float = 0.05

    # Logging
    exp_name: str = "probenet_baseline"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        self.output_dir = str(Path(self.output_dir).resolve())


@dataclass
class NormStatsConfig:
    """Configuration for computing dataset normalization statistics."""

    config_name: str = ""
    repo_id: str = ""
    extra_roots: list[str] = field(default_factory=list)
    output_dir: str = "outputs/norm_stats"


def create_fast_tokenizer_config() -> dict[str, Any]:
    """Return a placeholder config for the FAST tokenizer pipeline."""
    return {
        "config_name": "pi05_so101",
        "repo_id": "",
        "extra_roots": [],
    }
