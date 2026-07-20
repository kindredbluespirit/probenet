"""Conditioning modules for fusing visual and physical parameters."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPConditioner(nn.Module):
    """Concatenate visual and physical params, project to embedding.

    Args:
        visual_dim: Dimension of visual parameter vector.
        physical_dim: Dimension of physical parameter vector.
        hidden_dim: Hidden layer size.
        output_dim: Output embedding dimension.
    """

    def __init__(
        self,
        visual_dim: int = 4,
        physical_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + physical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        visual_params: torch.Tensor,
        physical_params: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([visual_params, physical_params], dim=-1)
        return self.net(x)


class GNNConditioner(nn.Module):
    """Simple two-node GNN: visual node <-> physical node.

    Each parameter type is one node with a learned embedding.
    Two rounds of message passing, then concatenate node embeddings.
    """

    def __init__(
        self,
        visual_dim: int = 4,
        physical_dim: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.vis_embed = nn.Linear(visual_dim, hidden_dim)
        self.phys_embed = nn.Linear(physical_dim, hidden_dim)

        self.msg_vis = nn.Linear(hidden_dim, hidden_dim)
        self.msg_phys = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        visual_params: torch.Tensor,
        physical_params: torch.Tensor,
    ) -> torch.Tensor:
        v = self.vis_embed(visual_params)
        p = self.phys_embed(physical_params)
        b, _ = v.shape

        for _ in range(2):
            v_agg = v + self.msg_vis(p)
            p_agg = p + self.msg_phys(v)
            v = v_agg
            p = p_agg

        return self.out(torch.cat([v, p], dim=-1))


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation: physical params produce scale/shift.

    The physical parameter embedding is used to modulate visual features
    through learned scale and shift (FiLM).
    """

    def __init__(
        self,
        visual_dim: int = 4,
        physical_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.film = nn.Sequential(
            nn.Linear(physical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        visual_params: torch.Tensor,
        physical_params: torch.Tensor,
    ) -> torch.Tensor:
        v = self.visual_proj(visual_params)
        gamma_beta = self.film(physical_params)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        v = gamma * v + beta
        return self.out(v)


def build_conditioner(
    name: str,
    visual_dim: int = 4,
    physical_dim: int = 3,
    output_dim: int = 32,
) -> nn.Module:
    """Factory: return a conditioniner by name."""
    factory = {
        "mlp": MLPConditioner,
        "gnn": GNNConditioner,
        "film": FiLMConditioner,
    }
    if name not in factory:
        raise ValueError(f"Unknown conditioner {name!r}. Choose from {list(factory)}")
    return factory[name](visual_dim=visual_dim, physical_dim=physical_dim, output_dim=output_dim)
