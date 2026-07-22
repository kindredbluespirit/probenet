"""GR00T N1.7 SO-101 policy — embodiment tag and training config.

Registers the SO-101 arm as a ``NEW_EMBODIMENT`` in the GR00T fine-tuning
pipeline and exposes trainer configuration.

Reference: NVIDIA/Isaac-GR00T ``getting_started/finetune_new_embodiment.md``
    and ``examples/SO100/so100_config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

SO101_MODALITY_CONFIG = {
    "action": {
        "left_arm": {
            "type": "joint",
            "dim": 6,
            "horizon": 50,
        },
    },
    "observation": {
        "images": {
            "cam_primary": {"type": "rgb", "height": 224, "width": 224},
            "cam_wrist": {"type": "rgb", "height": 224, "width": 224},
        },
        "state": {
            "joint_pos": 6,
            "joint_vel": 6,
        },
    },
}


@dataclass
class Gr00tTrainConfig:
    """GR00T fine-tuning configuration for the SO-101 arm.

    Maps to the CLI arguments of ``launch_finetune.py``.
    """

    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    dataset_path: str = ""
    embodiment_tag: str = "SO101"
    modality_config: dict = field(default_factory=lambda: SO101_MODALITY_CONFIG)
    num_gpus: int = 1
    output_dir: str = "/tmp/gr00t_finetune"
    max_steps: int = 5_000
    global_batch_size: int = 32

    def as_args(self) -> list[str]:
        """Return CLI argument list for ``launch_finetune.py``."""
        return [
            f"--base-model-path={self.base_model_path}",
            f"--dataset-path={self.dataset_path}",
            f"--embodiment-tag={self.embodiment_tag}",
            f"--modality-config-path=<generated>",
            f"--num-gpus={self.num_gpus}",
            f"--output-dir={self.output_dir}",
            f"--max-steps={self.max_steps}",
            f"--global-batch-size={self.global_batch_size}",
        ]


def write_modality_config(output_path: str) -> None:
    """Write the SO-101 modality config as a Python module for GR00T."""
    import json

    with open(output_path, "w") as f:
        f.write("modality = ")
        json.dump(SO101_MODALITY_CONFIG, f, indent=2)
        f.write("\n")


# ── ProbeNet GR00T configuration ──────────────────────────────────────────────


@dataclass
class Gr00tProbeNetConfig:
    """ProbeNet-specific GR00T fine-tuning configuration.

    Adds physical property conditioning to GR00T training pipeline.
    Properties are injected as text tokens into the task prompt.
    """

    name: str = "gr00t_so101_probenet"

    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    dataset_path: str = ""
    embodiment_tag: str = "SO101"

    num_gpus: int = 1
    output_dir: str = "/tmp/gr00t_finetune"
    max_steps: int = 5_000
    global_batch_size: int = 32

    # ProbeNet conditioning
    probenet_enabled: bool = True
    probe_mode_dropout: float = 0.1
    property_dropout: float = 0.25
    aggressiveness_dropout: float = 0.15
    language_dropout: float = 0.05

    wandb_enabled: bool = True
