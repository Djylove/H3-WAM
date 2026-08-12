"""Minimal frozen-H3 action-flow training step for the feasibility stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .bridge import H3ActionBridge, H3ActionBridgeOutput
from .scheduler import H3ActionFlowScheduler


@dataclass
class H3WAMFlowBatch:
    noisy_video_latents: torch.Tensor
    noisy_actions: torch.Tensor
    video_sigma: torch.Tensor
    timestep: torch.Tensor
    video_target: torch.Tensor
    action_target: torch.Tensor


@dataclass
class H3WAMLoss:
    total: torch.Tensor
    action: torch.Tensor
    video: torch.Tensor


def prepare_h3wam_flow_batch(
    *,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    scheduler: H3ActionFlowScheduler,
    video_sigma: torch.Tensor | None = None,
    video_noise: torch.Tensor | None = None,
    action_noise: torch.Tensor | None = None,
) -> H3WAMFlowBatch:
    """Noise future-video latents and actions on H3's coupled schedules."""

    if video_latents.shape[0] != actions.shape[0]:
        raise ValueError(
            "video and action batch sizes must match, "
            f"got {video_latents.shape[0]} and {actions.shape[0]}"
        )
    batch_size = actions.shape[0]
    if video_sigma is None:
        video_sigma, _ = scheduler.sample_training_sigmas(
            batch_size,
            device=actions.device,
            dtype=torch.float32,
        )
    if video_noise is None:
        video_noise = torch.randn_like(video_latents)
    if action_noise is None:
        action_noise = torch.randn_like(actions)

    return H3WAMFlowBatch(
        noisy_video_latents=scheduler.add_video_noise(video_latents, video_noise, video_sigma),
        noisy_actions=scheduler.add_action_noise(actions, action_noise, video_sigma),
        video_sigma=video_sigma,
        timestep=scheduler.timestep(video_sigma),
        video_target=scheduler.video_training_target(video_latents, video_noise),
        action_target=scheduler.training_target(actions, action_noise, video_sigma),
    )


def h3wam_action_training_step(
    bridge: H3ActionBridge,
    *,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    scheduler: H3ActionFlowScheduler,
    state: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
    minimax_payload: dict[str, Any] | None = None,
    transformer_options: dict[str, Any] | None = None,
    video_sigma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, H3ActionBridgeOutput, H3WAMFlowBatch]:
    """Run one action-supervised flow-matching step with a frozen H3 core."""

    flow_batch = prepare_h3wam_flow_batch(
        video_latents=video_latents,
        actions=actions,
        scheduler=scheduler,
        video_sigma=video_sigma,
    )
    output = bridge(
        video_latents=flow_batch.noisy_video_latents,
        noisy_actions=flow_batch.noisy_actions,
        timestep=flow_batch.timestep,
        context=context,
        state=state,
        minimax_payload=minimax_payload,
        transformer_options=transformer_options,
    )
    per_element = (output.action_velocity.float() - flow_batch.action_target.float()).square()
    if action_is_pad is None:
        loss = per_element.mean()
    else:
        expected_shape = actions.shape[:2]
        if action_is_pad.shape != expected_shape:
            raise ValueError(
                f"action_is_pad must have shape {expected_shape}, got {tuple(action_is_pad.shape)}"
            )
        valid = (~action_is_pad).to(device=per_element.device, dtype=per_element.dtype)
        valid = valid.unsqueeze(-1).expand_as(per_element)
        denominator = valid.sum().clamp_min(1.0)
        loss = (per_element * valid).sum() / denominator
    return loss, output, flow_batch


def h3wam_joint_training_step(
    bridge: H3ActionBridge,
    *,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    context: torch.Tensor,
    scheduler: H3ActionFlowScheduler,
    video_loss_weight: float = 0.2,
    state: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
    minimax_payload: dict[str, Any] | None = None,
    transformer_options: dict[str, Any] | None = None,
    video_sigma: torch.Tensor | None = None,
) -> tuple[H3WAMLoss, H3ActionBridgeOutput, H3WAMFlowBatch]:
    """Jointly supervise action and future-video velocities in one H3 pass."""

    if video_loss_weight < 0:
        raise ValueError("video_loss_weight must be non-negative")
    flow_batch = prepare_h3wam_flow_batch(
        video_latents=video_latents,
        actions=actions,
        scheduler=scheduler,
        video_sigma=video_sigma,
    )
    output = bridge(
        video_latents=flow_batch.noisy_video_latents,
        noisy_actions=flow_batch.noisy_actions,
        timestep=flow_batch.timestep,
        context=context,
        state=state,
        minimax_payload=minimax_payload,
        transformer_options=transformer_options,
    )
    action_elements = (
        output.action_velocity.float() - flow_batch.action_target.float()
    ).square()
    if action_is_pad is None:
        action_loss = action_elements.mean()
    else:
        expected_shape = actions.shape[:2]
        if action_is_pad.shape != expected_shape:
            raise ValueError(
                f"action_is_pad must have shape {expected_shape}, got {tuple(action_is_pad.shape)}"
            )
        valid = (~action_is_pad).to(device=action_elements.device, dtype=action_elements.dtype)
        valid = valid.unsqueeze(-1).expand_as(action_elements)
        action_loss = (action_elements * valid).sum() / valid.sum().clamp_min(1.0)
    video_loss = (
        output.video_velocity.float() - flow_batch.video_target.float()
    ).square().mean()
    losses = H3WAMLoss(
        total=action_loss + float(video_loss_weight) * video_loss,
        action=action_loss,
        video=video_loss,
    )
    return losses, output, flow_batch
