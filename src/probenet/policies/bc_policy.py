"""BC policy with optional ProbeNet conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Small CNN for encoding RGB images."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),   # 224 → 110
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),  # 110 → 53
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),  # 53 → 25
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),  # 25 → 12
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x)


class StateEncoder(nn.Module):
    """MLP for encoding the proprioceptive state."""

    def __init__(self, state_dim: int = 25, hidden_dim: int = 64, output_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCPolicy(nn.Module):
    """Simple behavioral cloning policy.

    Encodes RGB image + state, outputs actions. Optionally accepts a
    conditioning embedding from a ProbeNet conditioniner.

    Args:
        action_dim: Dimensionality of the action space.
        conditioning_dim: Conditioning embedding dimension (0 to disable).
    """

    def __init__(
        self,
        action_dim: int = 6,
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=64)
        self.state_encoder = StateEncoder(output_dim=16)

        fusion_dim = 64 + 16 + conditioning_dim
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vis = self.image_encoder(rgb)
        s = self.state_encoder(state)
        features = [vis, s]
        if conditioning is not None:
            features.append(conditioning)
        x = torch.cat(features, dim=-1)
        return self.action_head(x)


class ProbeNetPolicy(nn.Module):
    """BC policy conditioned on a ProbeNet conditioniner.

    Args:
        action_dim: Action dimension.
        conditioner: A ``nn.Module`` that takes ``(visual_params, physical_params)``
            and returns a conditioning embedding.
        conditioning_dim: Output dimension of the conditioner.
    """

    def __init__(
        self,
        action_dim: int = 6,
        conditioner: nn.Module | None = None,
        conditioning_dim: int = 32,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.policy = BCPolicy(
            action_dim=action_dim,
            conditioning_dim=conditioning_dim if conditioner else 0,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        visual_params: torch.Tensor | None = None,
        physical_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioning = None
        if self.conditioner is not None and visual_params is not None and physical_params is not None:
            conditioning = self.conditioner(visual_params, physical_params)
        return self.policy(rgb, state, conditioning)
