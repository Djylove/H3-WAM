"""FACT-style causal auxiliary training for the frozen-H3 ActionDiT carrier.

This is a labelled H3 backbone port, not an official FACT reproduction.  FACT
uses clean executed actions to condition future-state/value/video tokens while
preventing predicted action tokens from seeing those future targets.  Here the
deployed D0 action forward is kept byte-for-byte unchanged.  A training-only
second pass encodes the clean executed action through the *same* ActionDiT
blocks, and small heads predict train-standardized future H3 representation, proprioception and
progress value.  Targets are loss arguments only and can never enter the action
forward.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy


FACT_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
FACT_ACTION_WEIGHT = 10.0
FACT_FUTURE_H3_WEIGHT = 1.0
FACT_FUTURE_STATE_WEIGHT = 0.4
FACT_VALUE_WEIGHT = 0.4


class H3FactJointAuxPolicy(nn.Module):
    """Train-only causal consequence branch sharing the deployed action blocks."""

    def __init__(
        self,
        carrier: H3DreamWAMKVCarrierPolicy,
        *,
        hidden_dim: int,
        future_h3_dim: int = 256,
        future_state_dim: int = 8,
    ) -> None:
        super().__init__()
        if not carrier.enabled:
            raise ValueError("FACT joint auxiliary policy requires an enabled carrier")
        if min(hidden_dim, future_h3_dim, future_state_dim) <= 0:
            raise ValueError("joint auxiliary dimensions must be positive")
        self.carrier = carrier
        self.hidden_dim = int(hidden_dim)
        self.future_h3_dim = int(future_h3_dim)
        self.future_state_dim = int(future_state_dim)
        self.aux_norm = nn.LayerNorm(self.hidden_dim)
        self.future_h3_decoder = nn.Linear(self.hidden_dim, self.future_h3_dim)
        self.future_state_decoder = nn.Linear(self.hidden_dim, self.future_state_dim)
        self.value_decoder = nn.Linear(self.hidden_dim, 1)

    def forward_action(self, *args, **kwargs) -> torch.Tensor:
        """Exact deployment path; auxiliary modules are not evaluated."""

        return self.carrier(*args, **kwargs)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        clean_executed_actions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """DDP-safe dispatch for the unchanged action or joint training path."""

        if clean_executed_actions is None:
            return self.forward_action(noisy_actions, timestep, **kwargs)
        return self.forward_joint(
            noisy_actions,
            timestep,
            clean_executed_actions=clean_executed_actions,
            **kwargs,
        )

    def forward_joint(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        clean_executed_actions: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
        executed_action_history: torch.Tensor | None = None,
        executed_action_history_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if clean_executed_actions.shape != noisy_actions.shape:
            raise ValueError("clean_executed_actions must match noisy_actions")
        common = {
            "text_context": text_context,
            "proprio": proprio,
            "video_kv_cache": video_kv_cache,
            "text_mask": text_mask,
            "executed_action_history": executed_action_history,
            "executed_action_history_valid": executed_action_history_valid,
        }
        action_prediction, _ = self.carrier.forward_hidden(
            noisy_actions, timestep, **common
        )
        # FACT's gt_action condition is clean (t=0) and cannot attend the noisy
        # predicted-action track.  The isolated second pass is the equivalent
        # no-leakage boundary for this frozen-H3 ActionDiT port.
        clean_timestep = torch.zeros_like(timestep)
        _, clean_action_hidden = self.carrier.forward_hidden(
            clean_executed_actions, clean_timestep, **common
        )
        latent = self.aux_norm(clean_action_hidden.float().mean(dim=1))
        return {
            "action": action_prediction,
            "future_h3": self.future_h3_decoder(latent),
            "future_state": self.future_state_decoder(latent),
            "value": self.value_decoder(latent).squeeze(-1),
        }


def fact_joint_auxiliary_loss(
    predictions: Mapping[str, torch.Tensor],
    *,
    action_loss: torch.Tensor,
    future_h3_target: torch.Tensor,
    future_state_target: torch.Tensor,
    value_target: torch.Tensor,
    normalize_by_action_weight: bool = True,
) -> dict[str, torch.Tensor]:
    """Return FACT-relative losses without changing the parent action LR scale."""

    future_h3_loss = F.mse_loss(
        predictions["future_h3"].float(), future_h3_target.detach().float()
    )
    future_state_loss = F.mse_loss(
        predictions["future_state"].float(), future_state_target.detach().float()
    )
    value_loss = F.mse_loss(
        predictions["value"].float(), value_target.detach().float()
    )
    total = (
        FACT_ACTION_WEIGHT * action_loss
        + FACT_FUTURE_H3_WEIGHT * future_h3_loss
        + FACT_FUTURE_STATE_WEIGHT * future_state_loss
        + FACT_VALUE_WEIGHT * value_loss
    )
    if normalize_by_action_weight:
        total = total / FACT_ACTION_WEIGHT
    return {
        "loss": total,
        "action_loss": action_loss,
        "future_h3_loss": future_h3_loss,
        "future_state_loss": future_state_loss,
        "value_loss": value_loss,
    }


__all__ = [
    "FACT_ACTION_WEIGHT",
    "FACT_COMMIT",
    "FACT_FUTURE_H3_WEIGHT",
    "FACT_FUTURE_STATE_WEIGHT",
    "FACT_VALUE_WEIGHT",
    "H3FactJointAuxPolicy",
    "fact_joint_auxiliary_loss",
]
