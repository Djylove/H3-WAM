"""Small action-only flow transformer used as the non-H3 control experiment."""

from __future__ import annotations

import math

import torch
from torch import nn


class SmallActionFlowTransformer(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        context_dim: int = 5376,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_horizon: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.time_dim = int(time_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, max_horizon, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def _time_embedding(self, video_sigma: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=video_sigma.device, dtype=torch.float32)
            / half
        )
        angles = video_sigma.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        *,
        state: torch.Tensor,
        context: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, action_dim = noisy_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim {self.action_dim}, got {action_dim}")
        if horizon > self.position.shape[1]:
            raise ValueError(f"horizon {horizon} exceeds maximum {self.position.shape[1]}")
        if tuple(state.shape) != (batch, self.state_dim):
            raise ValueError(
                f"state must have shape {(batch, self.state_dim)}, got {tuple(state.shape)}"
            )
        condition = (
            self.state_projection(state.float())
            + self.context_projection(context.float().mean(dim=1))
            + self.time_projection(self._time_embedding(video_sigma))
        )
        hidden = self.action_projection(noisy_actions.float())
        hidden = hidden + self.position[:, :horizon] + condition.unsqueeze(1)
        return self.output(self.output_norm(self.transformer(hidden)))
