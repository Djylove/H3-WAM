"""A thin bridge between robot actions and an H3 audio-video denoiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .action_adapter import H3ActionAdapter


def make_first_frame_payload(
    first_frame_latents: torch.Tensor,
    *,
    frame_count: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the FL2VA payload expected by ComfyUI's MiniMax H3 model."""

    if first_frame_latents.ndim != 5:
        raise ValueError(
            "first-frame latents must have shape [batch, 24, time, height, width], "
            f"got {tuple(first_frame_latents.shape)}"
        )
    if first_frame_latents.shape[0] != 1:
        raise ValueError("MiniMax H3 currently supports one sample per packed sequence")
    if first_frame_latents.shape[1] != 24:
        raise ValueError(
            f"MiniMax H3 first-frame latents require 24 channels, got {first_frame_latents.shape[1]}"
        )
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError(f"frame_count must follow H3's 17n+5 grid, got {frame_count}")

    keyframe = {
        "resolved_frame_index": 0,
        "latent": first_frame_latents,
    }
    return {
        "keyframes": [keyframe],
        "frame_count": int(frame_count),
        "cond_video_latents": [first_frame_latents],
        "seed": int(seed),
    }


@dataclass
class H3ActionBridgeOutput:
    """Joint video/action velocity returned by :class:`H3ActionBridge`."""

    video_velocity: torch.Tensor
    action_velocity: torch.Tensor
    action_latent_velocity: torch.Tensor


class H3ActionBridge(nn.Module):
    """Run robot action chunks through H3's existing audio denoising path.

    The wrapped H3 module is expected to follow the ComfyUI MiniMax H3 call
    contract and return ``[video_velocity, audio_velocity]``.  Keeping that
    contract here avoids a fork of the H3 transformer during the feasibility
    stage.
    """

    def __init__(
        self,
        h3_model: nn.Module,
        action_adapter: H3ActionAdapter,
        *,
        freeze_h3: bool = True,
    ) -> None:
        super().__init__()
        self.h3_model = h3_model
        self.action_adapter = action_adapter
        self.freeze_h3 = bool(freeze_h3)
        if self.freeze_h3:
            self.h3_model.requires_grad_(False)
            self.h3_model.eval()

    def train(self, mode: bool = True) -> "H3ActionBridge":
        super().train(mode)
        if self.freeze_h3:
            self.h3_model.eval()
        return self

    def forward(
        self,
        *,
        video_latents: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor | None = None,
        minimax_payload: dict[str, Any] | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> H3ActionBridgeOutput:
        action_latents = self.action_adapter.encode_actions(noisy_actions, state)
        h3_output = self.h3_model(
            [video_latents, action_latents],
            timestep,
            context,
            transformer_options=transformer_options or {},
            minimax_payload=minimax_payload,
        )
        if not isinstance(h3_output, (list, tuple)) or len(h3_output) != 2:
            raise TypeError(
                "H3 model must return [video_velocity, audio_velocity], "
                f"got {type(h3_output).__name__}"
            )
        video_velocity, action_latent_velocity = h3_output
        action_velocity = self.action_adapter.decode_velocity(
            action_latent_velocity,
            state=state,
            context=context,
        )
        return H3ActionBridgeOutput(
            video_velocity=video_velocity,
            action_velocity=action_velocity,
            action_latent_velocity=action_latent_velocity,
        )
