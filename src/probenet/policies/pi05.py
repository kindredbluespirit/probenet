"""π₀.₅ SO-101 policy — data transforms, training configs, ProbeNet variant.

Adapts LeRobot SO-101 observations and actions to the π₀.₅ model input / output
space via ``openpi.transforms.DataTransformFn`` subclasses.

Reference: Maelic/openpi-SO100 ``so100_policy.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from probenet.policies.so101 import ACTION_DIM, STATE_DIM


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class So101Inputs(transforms.DataTransformFn):
    """Map LeRobot SO-101 observations → openpi model inputs.

    Args:
        model_type: PI0 or PI0_FAST (affects state padding behaviour).
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation.state"], self.model_type.defaults.action_dim)
        if self.model_type == _model.ModelType.PI0:
            state = transforms.pad_to_dim(data["observation.state"], self.model_type.defaults.action_dim)
        else:
            state = np.asarray(data["observation.state"], dtype=np.float32)

        base_image = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist_image = np.zeros((224, 224, 3), dtype=np.uint8)

        if "observation.images.cam_primary" in data:
            base_image = _parse_image(data["observation.images.cam_primary"])
        if "observation.images.cam_wrist" in data:
            wrist_image = _parse_image(data["observation.images.cam_wrist"])

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(wrist_image),
        }
        image_masks = {k: np.True_ for k in images}

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.model_type.defaults.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "task" in data and "prompt" not in inputs:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class So101Outputs(transforms.DataTransformFn):
    """Map openpi model outputs → SO-101 joint actions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, :ACTION_DIM]}


# ── ProbeNet π₀.₅ configuration ──────────────────────────────────────────────


@dataclass
class Pi05ProbeNetConfig:
    """ProbeNet-specific π₀.₅ fine-tuning configuration.

    This maps to openpi's ``TrainConfig`` entries but adds ProbeNet
    conditioning parameters. The trainer uses this to construct the
    openpi training pipeline and condition the data loader with
    property tokens.
    """

    name: str = "pi05_so101_probenet"

    # base π₀.₅ model
    init_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base"
    pi05: bool = True
    action_horizon: int = 50
    action_dim: int = 32  # π₀.₅ padded dim

    # data
    repo_id: str = ""
    asset_id: str = "so101"
    prompt_from_task: bool = True

    # training
    batch_size: int = 64
    lr_peak: float = 5e-5
    lr_warmup_steps: int = 10_000
    lr_decay_steps: int = 1_000_000
    lr_decay_lr: float = 5e-5
    clip_gradient_norm: float = 1.0
    num_train_steps: int = 100_000
    ema_decay: float | None = 0.999

    # LoRA fine-tuning
    use_lora: bool = True

    # ProbeNet conditioning
    probenet_enabled: bool = True
    probe_mode_dropout: float = 0.1
    property_dropout: float = 0.25
    aggressiveness_dropout: float = 0.15
    language_dropout: float = 0.05

    # FSDP (multi-GPU)
    fsdp_devices: int = 0

    wandb_enabled: bool = True
