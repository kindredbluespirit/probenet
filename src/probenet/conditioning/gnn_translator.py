"""Property Graph Translator — bidirectional GNN for interpretable ↔ latent mapping.

Maps between interpretable physical parameters (mass, friction, compliance, ...)
and dense latent codes for π₀.₅ conditioning. Operations:

Forward:  probe → properties → GNN message passing → global latent node → π₀.₅
Reverse:  latent code → GNN message passing → per-param nodes → interpretable values
Straight-through: probe → latent (bypasses interpretable params for speed)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Graph layers ──────────────────────────────────────────────────────────────


class GATLayer(nn.Module):
    """Graph attention layer with per-node propagation."""

    def __init__(self, node_dim: int, hidden_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.node_dim = node_dim
        self.out_dim = hidden_dim // heads
        self.attn = nn.Linear(node_dim * 2, heads)
        self.msg = nn.Linear(node_dim, node_dim)
        self.out = nn.Linear(node_dim * heads, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, node_dim) → (B, N, hidden_dim).

        If ``adj`` is None, full attention between all node pairs.
        """
        B, N, D = nodes.shape
        messages = torch.zeros(B, N, self.heads, self.out_dim, device=nodes.device)

        for i in range(N):
            for j in range(N):
                if adj is not None and adj[i, j] == 0:
                    continue
                cat = torch.cat([nodes[:, i], nodes[:, j]], dim=-1)  # (B, 2*D)
                a = self.attn(cat)  # (B, heads)
                a = F.softmax(a, dim=-1)
                msg = self.msg(nodes[:, j])  # (B, D)
                weighted = a.unsqueeze(-1) * msg.unsqueeze(1)  # (B, heads, D)
                messages[:, i] += weighted.view(B, self.heads, D)[:, :, : self.out_dim]

        out = messages.flatten(2)  # (B, N, heads * out_dim)
        out = self.out(out)  # (B, N, hidden_dim)
        out = self.dropout(F.relu(out))
        return self.norm(out + nodes[:, :, :out.shape[-1]])


class GraphTransformer(nn.Module):
    """Stacked GAT layers + global readout node.

    A special global latent node aggregates information from all
    property nodes through message passing.
    """

    def __init__(
        self,
        num_nodes: int,
        node_dim: int = 32,
        latent_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.latent_dim = latent_dim

        # per-property node embeddings + one global latent node
        self.node_embed = nn.Parameter(torch.randn(num_nodes + 1, node_dim) * 0.02)

        # value → node embedding
        self.value_proj = nn.Linear(2, node_dim)  # (value, confidence)

        self.layers = nn.ModuleList([
            GATLayer(node_dim, node_dim * 2, heads) for _ in range(num_layers)
        ])

        # readout
        self.latent_head = nn.Linear(node_dim, latent_dim)
        self.value_heads = nn.Linear(node_dim, 1)
        self.confidence_heads = nn.Sequential(nn.Linear(node_dim, 1), nn.Sigmoid())

    # ── Forward path: property values → latent code ─────────────────────

    def forward(self, values: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """(B, N, 2) → (B, latent_dim).

        Args:
            values: ``(B, N)`` property values.
            confidences: ``(B, N)`` property confidences.

        Returns:
            Global latent code ``(B, latent_dim)``.
        """
        B = values.shape[0]
        # convert values+confidences to node embeddings
        vc = torch.stack([values, confidences], dim=-1)  # (B, N, 2)
        prop_nodes = self.value_proj(vc)  # (B, N, node_dim)

        # add global latent node
        global_node = self.node_embed[self.num_nodes].unsqueeze(0).expand(B, -1, -1)
        all_nodes = torch.cat([prop_nodes, global_node], dim=1)  # (B, N+1, node_dim)
        all_nodes = all_nodes + self.node_embed.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            all_nodes = layer(all_nodes)

        latent = self.latent_head(all_nodes[:, -1])  # (B, latent_dim)
        return F.normalize(latent, dim=-1)

    # ── Reverse path: latent code → property values ─────────────────────

    def reverse(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress latent code to per-property values + confidences.

        Args:
            latent: ``(B, latent_dim)``.

        Returns:
            ``(values, confidences)`` each ``(B, N)``.
        """
        B = latent.shape[0]
        # broadcast latent as initial node states
        init = latent.unsqueeze(1).expand(B, self.num_nodes + 1, -1)

        # project to node_dim
        proj = nn.Linear(self.latent_dim, self.node_dim, device=latent.device)
        nodes = proj(init) + self.node_embed.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            nodes = layer(nodes)

        prop_nodes = nodes[:, : self.num_nodes]
        values = self.value_heads(prop_nodes).squeeze(-1)  # (B, N)
        confs = self.confidence_heads(prop_nodes).squeeze(-1)  # (B, N)
        return values, confs


# ── Full GNN Translator ──────────────────────────────────────────────────────


class PropertyTranslator(nn.Module):
    """Bidirectional graph-based translator between properties and latent codes.

    Args:
        property_keys: Ordered list of property names.
        node_dim: Per-node embedding dimension.
        latent_dim: Global latent code dimension.
    """

    def __init__(
        self,
        property_keys: list[str] | None = None,
        node_dim: int = 32,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.property_keys = property_keys or ["mass", "friction", "compliance", "fragility", "slipperiness"]
        self.num_props = len(self.property_keys)
        self.graph = GraphTransformer(
            self.num_props, node_dim=node_dim, latent_dim=latent_dim
        )

    def forward(self, values: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
        """Encode properties → latent code."""
        return self.graph.forward(values, confidences)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent code → property values + confidences."""
        return self.graph.reverse(latent)

    def encode_dict(self, prop_dict: dict[str, dict[str, float]]) -> torch.Tensor:
        """Encode from Python dict format → latent code (single item)."""
        import torch

        values = torch.tensor([prop_dict[k]["value"] for k in self.property_keys], dtype=torch.float32)
        confs = torch.tensor([prop_dict[k]["confidence"] for k in self.property_keys], dtype=torch.float32)
        return self.forward(values.unsqueeze(0), confs.unsqueeze(0))


# ── Training utilities ────────────────────────────────────────────────────────


def graph_reconstruction_loss(
    translator: PropertyTranslator,
    values: torch.Tensor,
    confidences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cycle consistency: params → latent → reconstructed_params.

    Returns:
        ``(total_loss, recon_loss, contrastive_loss)``
    """
    latent = translator.forward(values, confidences)
    recon_values, recon_confs = translator.decode(latent)

    recon_loss = F.mse_loss(recon_values, values)
    conf_loss = F.binary_cross_entropy(recon_confs, confidences)

    # contrastive: similar params → close latents, different params → far
    latent2 = translator.forward(values + torch.randn_like(values) * 0.05, confidences)
    contrastive = -F.cosine_similarity(latent, latent2, dim=-1).mean()

    return recon_loss + 0.1 * conf_loss + 0.1 * contrastive, recon_loss, conf_loss
