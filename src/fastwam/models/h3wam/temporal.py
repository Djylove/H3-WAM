"""Temporal alignment helpers for robot trajectories and H3 video clips."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class H3WindowPlan:
    action_horizon: int
    source_fps: float
    target_fps: float
    source_frame_count: int
    h3_frame_count: int
    h3_latent_frames: int

    @property
    def action_duration_seconds(self) -> float:
        return self.action_horizon / self.source_fps


def align_h3_frame_count(frame_count: int) -> int:
    """Round upward to H3's valid ``17n+5`` pixel-frame grid."""

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    return max(5, frame_count + (5 - frame_count) % 17)


def h3_video_latent_frames(frame_count: int) -> int:
    if frame_count < 5 or frame_count % 17 != 5:
        raise ValueError(f"H3 frame_count must follow 17n+5, got {frame_count}")
    return 2 if frame_count == 5 else ((frame_count - 5) // 17) * 5 + 2


def plan_h3_window(
    *,
    action_horizon: int,
    source_fps: float,
    target_fps: float = 24.0,
) -> H3WindowPlan:
    """Plan a clip covering the same wall time as a robot action chunk."""

    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    source_frame_count = action_horizon + 1
    nominal_target_frames = math.ceil(action_horizon / source_fps * target_fps)
    frame_count = align_h3_frame_count(nominal_target_frames)
    return H3WindowPlan(
        action_horizon=int(action_horizon),
        source_fps=float(source_fps),
        target_fps=float(target_fps),
        source_frame_count=source_frame_count,
        h3_frame_count=frame_count,
        h3_latent_frames=h3_video_latent_frames(frame_count),
    )


def resample_video_nearest(video: torch.Tensor, target_frame_count: int) -> torch.Tensor:
    """Nearest-neighbor temporal resampling with fixed first/last endpoints."""

    if video.ndim < 1 or video.shape[0] <= 0:
        raise ValueError("video must have a non-empty leading time dimension")
    if target_frame_count <= 0:
        raise ValueError(f"target_frame_count must be positive, got {target_frame_count}")
    indices = torch.linspace(
        0,
        video.shape[0] - 1,
        target_frame_count,
        device=video.device,
        dtype=torch.float32,
    ).round().to(dtype=torch.long)
    return video.index_select(0, indices)


def h3_latent_is_pad(
    pixel_is_pad: torch.Tensor,
    *,
    clip_length: int = 17,
    temporal_ratio: int = 4,
    token_drop: int = 3,
) -> torch.Tensor:
    """Map H3 pixel padding to its chunked causal VAE latent timeline.

    Each 17-frame encoder chunk uses three causal pre-padding frames, so its
    five latent steps cover pixel groups ``[0]``, ``[1:5]``, ..., ``[13:17]``.
    The released VAE then drops three trailing latent steps globally.
    """

    if pixel_is_pad.ndim != 1 or pixel_is_pad.numel() <= 1:
        raise ValueError("pixel_is_pad must be a 1D multi-frame mask")
    if clip_length <= 0 or temporal_ratio <= 0 or token_drop < 0:
        raise ValueError("invalid H3 temporal geometry")
    mask = pixel_is_pad.bool()
    pad = (-mask.numel()) % clip_length
    if pad:
        mask = torch.cat((mask, torch.ones(pad, dtype=torch.bool, device=mask.device)))
    latent = []
    for offset in range(0, mask.numel(), clip_length):
        chunk = mask[offset : offset + clip_length]
        latent.append(chunk[:1].all())
        for start in range(1, clip_length, temporal_ratio):
            latent.append(chunk[start : start + temporal_ratio].all())
    result = torch.stack(latent)
    if token_drop:
        if result.numel() <= token_drop:
            raise ValueError("token_drop removes every H3 latent step")
        result = result[:-token_drop]
    return result
