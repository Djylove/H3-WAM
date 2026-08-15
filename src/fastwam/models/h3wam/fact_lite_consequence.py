"""Small, isolated FACT-style future-proprio consequence model.

This module deliberately does not contain an action generator.  Candidate
actions cross a detached, read-only boundary before entering the consequence
expert, so a future-state loss cannot update whichever policy produced those
actions.  That is the structural no-leakage equivalent of FACT's causal mask
for the first F0/F1 canary.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FutureProprioConsequenceModel(nn.Module):
    """Predict normalized future proprioception from observation and actions."""

    def __init__(
        self,
        *,
        state_dim: int = 8,
        action_dim: int = 7,
        action_horizon: int = 32,
        h3_feature_dim: int = 5376,
        hidden_dim: int = 256,
        feature_input_scale: float = 0.009606920816877307,
    ) -> None:
        super().__init__()
        if min(
            state_dim,
            action_dim,
            action_horizon,
            h3_feature_dim,
            hidden_dim,
        ) <= 0:
            raise ValueError("all consequence-model dimensions must be positive")
        if feature_input_scale <= 0:
            raise ValueError("feature_input_scale must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.h3_feature_dim = int(h3_feature_dim)
        self.feature_input_scale = float(feature_input_scale)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.h3_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_horizon * self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def forward(
        self,
        current_proprio: torch.Tensor,
        h3_features: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> torch.Tensor:
        if current_proprio.ndim != 2 or current_proprio.shape[-1] != self.state_dim:
            raise ValueError(
                f"current_proprio must be [B,{self.state_dim}], got "
                f"{tuple(current_proprio.shape)}"
            )
        if h3_features.ndim == 4 and h3_features.shape[1] == 1:
            h3_features = h3_features[:, 0]
        if (
            h3_features.ndim != 3
            or h3_features.shape[0] != current_proprio.shape[0]
            or h3_features.shape[-1] != self.h3_feature_dim
        ):
            raise ValueError(
                "h3_features must be [B,T,F] (or [B,1,T,F]) with matching "
                f"batch and F={self.h3_feature_dim}, got {tuple(h3_features.shape)}"
            )
        expected_action_shape = (
            current_proprio.shape[0],
            self.action_horizon,
            self.action_dim,
        )
        if tuple(candidate_actions.shape) != expected_action_shape:
            raise ValueError(
                f"candidate_actions must be {expected_action_shape}, got "
                f"{tuple(candidate_actions.shape)}"
            )

        # This is the critical F0/F1 boundary: consequence supervision may
        # update this expert's action encoder, but never an upstream policy.
        read_only_actions = candidate_actions.detach()
        state = self.state_encoder(current_proprio.float())
        visual = self.visual_encoder(
            h3_features.float().mean(dim=1) * self.feature_input_scale
        )
        action = self.action_encoder(read_only_actions.float().flatten(1))
        delta = self.predictor(torch.cat((state, visual, action), dim=-1))
        return current_proprio.float() + delta


def future_proprio_mse(
    model: FutureProprioConsequenceModel,
    *,
    current_proprio: torch.Tensor,
    h3_features: torch.Tensor,
    candidate_actions: torch.Tensor,
    future_proprio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prediction and MSE without exposing the target to ``forward``."""

    if future_proprio.shape != current_proprio.shape:
        raise ValueError("future_proprio must match current_proprio shape")
    prediction = model(current_proprio, h3_features, candidate_actions)
    return prediction, F.mse_loss(prediction.float(), future_proprio.detach().float())


def deranged_batch_indices(batch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a deterministic no-self-map cyclic permutation."""

    if batch_size < 2:
        raise ValueError("a shuffled-action control requires batch_size >= 2")
    indices = torch.arange(batch_size, device=device)
    return indices.roll(1)


def actions_for_arm(actions: torch.Tensor, arm: str) -> torch.Tensor:
    """Construct the sole action-input difference for the three F1 arms."""

    if actions.ndim != 3:
        raise ValueError("actions must be [B,T,D]")
    if arm == "conditioned":
        return actions
    if arm == "shuffled":
        return actions.index_select(
            0, deranged_batch_indices(actions.shape[0], device=actions.device)
        )
    if arm == "independent":
        return torch.zeros_like(actions)
    raise ValueError(f"unknown consequence arm: {arm!r}")


__all__ = [
    "FutureProprioConsequenceModel",
    "actions_for_arm",
    "deranged_batch_indices",
    "future_proprio_mse",
]
