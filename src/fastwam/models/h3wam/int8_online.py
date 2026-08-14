"""Standalone INT8 H3 feature provider for live robot observations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import nn


AUDIO_CHANNELS = 2
AUDIO_LATENT_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)


@dataclass(frozen=True)
class H3Int8LayoutFunctions:
    build_packed_sequence: Callable
    build_row_timesteps: Callable
    patchify_video_latents: Callable


def _official_layout_functions() -> H3Int8LayoutFunctions:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )

    return H3Int8LayoutFunctions(
        MiniMaxH3PrepareLayoutStep.build_packed_sequence,
        MiniMaxH3SetTimestepsStep.build_row_timesteps,
        patchify_video_latents,
    )


@dataclass(frozen=True)
class H3Int8OnlineFeatureContract:
    layers: tuple[int, ...] = (9, 19, 29, 39, 49)
    action_horizon: int = 8
    target_latent_frames: int = 12
    video_timestep: float = 0.0
    condition_video_timestep: float = 0.999
    capture_compatibility: str = "comfy_alias_v1"

    def __post_init__(self) -> None:
        if not self.layers or min(self.layers) < 0 or max(self.layers) >= 50:
            raise ValueError("layers must select H3 blocks in [0,49]")
        if self.action_horizon <= 0 or self.target_latent_frames <= 0:
            raise ValueError("action horizon and target latent frames must be positive")
        if not 0.0 <= self.video_timestep <= 1.0:
            raise ValueError("video timestep must be in [0,1]")
        if not 0.0 <= self.condition_video_timestep <= 1.0:
            raise ValueError("condition-video timestep must be in [0,1]")
        if self.capture_compatibility not in ("none", "comfy_alias_v1"):
            raise ValueError("unsupported capture compatibility mode")


def apply_online_capture_compatibility(
    features: torch.Tensor, mode: str
) -> torch.Tensor:
    """Apply an explicit historical live-feature capture convention."""

    if features.ndim != 3:
        raise ValueError("features must be [layers,tokens,hidden]")
    if mode == "none":
        return features
    if mode == "comfy_alias_v1":
        return features[-1:].expand_as(features).clone()
    raise ValueError(f"unsupported capture compatibility mode {mode!r}")


def encode_h3_vae_condition_standalone(
    vae: nn.Module,
    pixels: torch.Tensor,
    pixel_mean: tuple[float, float, float],
    pixel_std: tuple[float, float, float],
    encode_seed: int = 42,
) -> torch.Tensor:
    """Exact released H3 keyframe VAE recipe without text-stack imports."""

    if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
        raise ValueError("VAE condition pixels must be [1,3,frames,height,width]")
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)
    normalized = (pixels.to(torch.float32).div(255.0) - mean) / std
    posterior = vae.encode(normalized, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


class H3Int8OnlineFeatureProvider(nn.Module):
    """Turn one live H3 first-frame latent into historical action-head features.

    The provider deliberately starts after the VAE. Deployment owns live camera
    preprocessing and VAE encoding; this module owns the exact packed-sequence,
    timestep and capture contracts shared by offline parity and rollout.
    """

    def __init__(
        self,
        backbone: nn.Module,
        contract: H3Int8OnlineFeatureContract,
        *,
        layout_functions: H3Int8LayoutFunctions | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.contract = contract
        self.layout_functions = layout_functions

    @torch.inference_mode()
    def forward(
        self,
        first_frame_latents: torch.Tensor,
        encoder_context: torch.Tensor,
        text_token_tags: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(first_frame_latents.shape[:3]) != (1, 24, 1):
            raise ValueError("first-frame latents must be [1,24,1,H,W]")
        if encoder_context.ndim != 3 or encoder_context.shape[0] != 1:
            raise ValueError("encoder context must be [1,tokens,hidden]")
        if encoder_context.shape[-1] not in (5120, 5376):
            raise ValueError("H3 encoder context width must be raw 5120 or refined 5376")
        if text_token_tags.ndim != 1 or text_token_tags.numel() != encoder_context.shape[1]:
            raise ValueError("text token tags must cover every encoder context row")

        layout_functions = self.layout_functions or _official_layout_functions()
        device = first_frame_latents.device
        _, channels, _, latent_height, latent_width = first_frame_latents.shape
        packed = layout_functions.build_packed_sequence(
            text_token_tags=text_token_tags.to(device="cpu", dtype=torch.long),
            num_latent_frames=self.contract.target_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=self.contract.action_horizon,
            patch_size=PATCH_SIZE,
            audio_channels=AUDIO_CHANNELS,
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=("first",),
        )
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = packed
        unique_timesteps, timestep_indices = layout_functions.build_row_timesteps(
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_condition_video_rows=num_condition_video_rows,
            num_condition_audio_rows=num_condition_audio_rows,
            num_text_tokens=text_indices.numel(),
            video_timestep=self.contract.video_timestep,
            audio_timestep=0.0,
            condition_video_timestep=self.contract.condition_video_timestep,
            condition_audio_timestep=1.0,
        )

        target = torch.zeros(
            (1, channels, self.contract.target_latent_frames, latent_height, latent_width),
            device=device,
            dtype=torch.float32,
        )
        row_width = channels * 4
        first_rows = layout_functions.patchify_video_latents(
            first_frame_latents.float(), PATCH_SIZE
        ).reshape(1, -1, row_width)
        target_rows = layout_functions.patchify_video_latents(target, PATCH_SIZE).reshape(
            1, -1, row_width
        )
        video_rows = torch.cat((first_rows, target_rows), dim=1)
        audio_rows = torch.zeros(
            (1, self.contract.action_horizon * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            device=device,
            dtype=torch.float32,
        )
        condition_indices = video_indices[:num_condition_video_rows].to(device)
        result = self.backbone(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            # Match the standalone offline cache path exactly: the stored
            # refined values are BF16, then promoted to FP32 before packing.
            # Raw 5120-wide Qwen values are accepted too; the backbone applies
            # its pinned condition projection and token refiner exactly once.
            encoder_hidden_states=encoder_context.to(
                device=device, dtype=torch.float32
            ),
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=token_tags.to(device),
            position_ids=position_ids.to(device),
            video_indices=video_indices.to(device),
            audio_indices=audio_indices.to(device),
            text_indices=text_indices.to(device),
            capture_layers=self.contract.layers,
            capture_indices=condition_indices,
        )
        features = torch.stack(
            [result.captured_features[layer][0] for layer in self.contract.layers],
            dim=0,
        )
        features = apply_online_capture_compatibility(
            features, self.contract.capture_compatibility
        )
        return features.unsqueeze(0)
