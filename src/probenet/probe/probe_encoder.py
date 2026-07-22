"""Probe Encoder — CNN + Transformer over raw force signal with visual cross-attention.

Processes the interaction history (actuator forces, joint positions, visual
features) to estimate physical properties of the object being probed.

Architecture:
    1D CNN over (T, signal_dim) force signal → temporal features.
    Transformer encoder with learned position embeddings → context.
    Visual cross-attention (from frozen π₀.₅ visual encoder output).
    Prediction heads per property (value + confidence).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1D CNN over raw force signal ──────────────────────────────────────────────

class ForceEncoder(nn.Module):
    """1D CNN that compresses a raw force signal (T, signal_dim) to fixed features.

    Args:
        signal_dim: Number of force channels (6 for SO-101 joints).
        hidden_dim: Output feature dimension.
        kernel_size: Conv1d kernel size.
        stride: Conv1d stride.
    """

    def __init__(
        self,
        signal_dim: int = 6,
        hidden_dim: int = 128,
        kernel_size: int = 7,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(signal_dim, 64, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, T, signal_dim) → (B, T', hidden_dim)"""
        x = signal.permute(0, 2, 1)  # (B, signal_dim, T)
        x = self.conv(x)  # (B, hidden_dim, T')
        return x.permute(0, 2, 1)  # (B, T', hidden_dim)


# ── Transformer encoder with visual cross-attention ───────────────────────────

class TransformerBlock(nn.Module):
    """Self-attention + cross-attention to visual features."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, ffn_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        if visual_features is not None:
            x = x + self.dropout(self.cross_attn(self.norm2(x), visual_features, visual_features)[0])
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class SignalTransformer(nn.Module):
    """Stacked transformer blocks over compressed force features.

    Args:
        num_layers: Number of transformer blocks.
        hidden_dim: Feature dimension.
        num_heads: Multi-head attention heads.
    """

    def __init__(self, num_layers: int = 2, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, T, hidden_dim) → (B, hidden_dim) after global pooling."""
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x, visual_features)
        return x.mean(dim=1)  # global average pooling


# ── Property prediction heads ─────────────────────────────────────────────────

class PropertyHeads(nn.Module):
    """Predict physical properties and confidences from encoded features.

    Args:
        hidden_dim: Input feature dimension.
        property_keys: List of property names to predict.
    """

    def __init__(self, hidden_dim: int = 128, property_keys: list[str] | None = None) -> None:
        super().__init__()
        self.property_keys = property_keys or ["mass", "friction", "compliance"]
        self.value_heads = nn.ModuleDict({
            key: nn.Linear(hidden_dim, 1) for key in self.property_keys
        })
        self.confidence_heads = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            for key in self.property_keys
        })

    def forward(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return ``{key: (value, confidence)}`` per property."""
        return {
            key: (
                self.value_heads[key](x),
                self.confidence_heads[key](x),
            )
            for key in self.property_keys
        }


# ── Full ProbeEncoder ─────────────────────────────────────────────────────────

class ProbeEncoder(nn.Module):
    """Full probe encoder: CNN → Transformer → prediction heads.

    Args:
        signal_dim: Number of force channels.
        hidden_dim: Internal representation dimension.
        property_keys: Properties to predict.
        visual_dim: Dimension of visual features from frozen encoder.
    """

    def __init__(
        self,
        signal_dim: int = 6,
        hidden_dim: int = 128,
        property_keys: list[str] | None = None,
        visual_dim: int = 768,
    ) -> None:
        super().__init__()
        self.property_keys = property_keys or ["mass", "friction", "compliance"]
        self.hidden_dim = hidden_dim

        self.force_encoder = ForceEncoder(signal_dim=signal_dim, hidden_dim=hidden_dim)
        self.transformer = SignalTransformer(hidden_dim=hidden_dim)

        if visual_dim != hidden_dim:
            self.visual_projector = nn.Linear(visual_dim, hidden_dim)
        else:
            self.visual_projector = nn.Identity()

        self.heads = PropertyHeads(hidden_dim, self.property_keys)

    def forward(
        self,
        signal: torch.Tensor,
        visual_features: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Predict physical properties from probe signal + visual context.

        Args:
            signal: ``(B, T, signal_dim)`` raw force signal.
            visual_features: ``(B, visual_dim)`` from frozen π₀.₅ visual encoder.

        Returns:
            ``{property_key: (value_tensor, confidence_tensor)}`` per property.
        """
        x = self.force_encoder(signal)  # (B, T', hidden_dim)

        vis = None
        if visual_features is not None:
            vis = self.visual_projector(visual_features).unsqueeze(1)  # (B, 1, hidden_dim)

        x = self.transformer(x, vis)  # (B, hidden_dim)
        return self.heads(x)

    def predict(self, signal: torch.Tensor, visual_features: torch.Tensor | None = None) -> dict[str, dict[str, float]]:
        """Predict as Python dict of ``{key: {"value": float, "confidence": float}}``."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(signal, visual_features)
        return {
            key: {
                "value": float(preds[key][0].item()),
                "confidence": float(preds[key][1].item()),
            }
            for key in preds
        }


# ── Training utilities ────────────────────────────────────────────────────────


def param_estimation_loss(
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ground_truth: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute supervised MSE + confidence NLL loss.

    Returns:
        ``(total_loss, mse_loss, conf_loss)`` tuple.
    """
    mse_total = torch.tensor(0.0)
    conf_total = torch.tensor(0.0)
    count = 0

    for key, (value, confidence) in predictions.items():
        if key not in ground_truth:
            continue
        target = torch.tensor(ground_truth[key], device=value.device)
        # value loss
        mse_total += F.mse_loss(value.squeeze(-1), target.expand_as(value.squeeze(-1)))
        # confidence loss: encourage high confidence when prediction is close, low when far
        error = (value.squeeze(-1) - target.expand_as(value.squeeze(-1))).abs()
        conf_target = torch.exp(-error).detach()
        conf_total += F.binary_cross_entropy(confidence.squeeze(-1), conf_target)
        count += 1

    if count == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    mse_loss = mse_total / count
    conf_loss = conf_total / count
    return mse_loss + 0.1 * conf_loss, mse_loss, conf_loss
