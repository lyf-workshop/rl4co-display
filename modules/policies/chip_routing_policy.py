"""Attention-based policy for ordering chip-routing nets."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from modules.envs.chip_routing_env import ChipRoutingInstance


CHIP_ROUTING_FEATURE_DIM = 9


def build_net_features(instance: ChipRoutingInstance) -> np.ndarray:
    """Build normalized geometric and blockage features for every net."""
    width_scale = max(1, instance.width - 1)
    height_scale = max(1, instance.height - 1)
    distance_scale = max(1, width_scale + height_scale)
    features = []

    for sx, sy, tx, ty in instance.nets:
        sx, sy, tx, ty = int(sx), int(sy), int(tx), int(ty)
        min_x, max_x = sorted((sx, tx))
        min_y, max_y = sorted((sy, ty))
        box = instance.obstacles[min_y : max_y + 1, min_x : max_x + 1]
        blockage_density = float(box.mean()) if box.size else 0.0
        dx = abs(tx - sx)
        dy = abs(ty - sy)
        features.append(
            [
                sx / width_scale,
                sy / height_scale,
                tx / width_scale,
                ty / height_scale,
                dx / width_scale,
                dy / height_scale,
                (dx + dy) / distance_scale,
                blockage_density,
                float(sx == tx or sy == ty),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def build_feature_batch(
    instances: Sequence[ChipRoutingInstance],
    device: torch.device,
) -> torch.Tensor:
    arrays = [build_net_features(instance) for instance in instances]
    return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)


class NetOrderingPolicy(nn.Module):
    """Autoregressive pointer policy that outputs a permutation of net IDs."""

    def __init__(
        self,
        feature_dim: int = CHIP_ROUTING_FEATURE_DIM,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.feature_dim = int(feature_dim)
        self.embed_dim = int(embed_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.decoder_cell = nn.GRUCell(embed_dim, embed_dim)
        self.start_context = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.start_context, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        decode_type: str = "sampling",
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, num_nets, feature_dim]")
        if decode_type not in {"sampling", "greedy"}:
            raise ValueError("decode_type must be 'sampling' or 'greedy'")

        batch_size, num_nets, _ = features.shape
        encoded = self.encoder(self.input_projection(features))
        keys = self.key_projection(encoded)
        context = self.start_context.unsqueeze(0).expand(batch_size, -1)
        selected_mask = torch.zeros(
            batch_size,
            num_nets,
            dtype=torch.bool,
            device=features.device,
        )
        orders = []
        log_probs = []
        entropies = []
        batch_indices = torch.arange(batch_size, device=features.device)

        for _ in range(num_nets):
            query = self.query_projection(context)
            logits = torch.einsum("bnd,bd->bn", keys, query)
            logits = logits / math.sqrt(self.embed_dim)
            logits = logits / max(float(temperature), 1e-6)
            logits = logits.masked_fill(selected_mask, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)

            if decode_type == "sampling":
                action = distribution.sample()
            else:
                action = logits.argmax(dim=-1)

            orders.append(action)
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            selected_mask = selected_mask.clone()
            selected_mask[batch_indices, action] = True
            selected_embedding = encoded[batch_indices, action]
            context = self.decoder_cell(selected_embedding, context)

        return {
            "order": torch.stack(orders, dim=1),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
            "entropy": torch.stack(entropies, dim=1).mean(dim=1),
        }

