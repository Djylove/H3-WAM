"""Joint Euler sampling utilities for the H3-DreamWAM RGB/action ODE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class H3DreamInferenceSchedule:
    video_clean_times: torch.Tensor
    video_clean_deltas: torch.Tensor
    action_sigmas: torch.Tensor
    action_sigma_deltas: torch.Tensor


@dataclass(frozen=True)
class H3DreamJointSample:
    video_rows: torch.Tensor
    actions: torch.Tensor


def _shift(value: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("flow shift must be positive")
    return shift * value / (1.0 + (shift - 1.0) * value)


@lru_cache(maxsize=None)
def _training_weight_stats(num_train_timesteps: int, shift: float) -> tuple[float, float]:
    if num_train_timesteps <= 0 or shift <= 0:
        raise ValueError("training timesteps and flow shift must be positive")
    grid = torch.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=torch.float64)[:-1]
    sigma = _shift(grid, shift)
    weight = torch.exp(-2.0 * (sigma - 0.5).square())
    minimum = float(weight.min())
    return minimum, float((weight - minimum).mean())


def h3dream_flow_training_weight(
    timestep: torch.Tensor,
    *,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,
    eps: float = 1.0e-10,
) -> torch.Tensor:
    """DreamWAM's normalized mid-timestep flow-matching weight."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    minimum, mean = _training_weight_stats(num_train_timesteps, float(shift))
    sigma = timestep.float() / float(num_train_timesteps)
    weight = torch.exp(-2.0 * (sigma - 0.5).square())
    return (weight - minimum) / (mean + eps)


def build_h3dream_inference_schedule(
    num_steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    video_shift: float = 12.0,
    action_shift: float = 5.0,
) -> H3DreamInferenceSchedule:
    """Build H3 clean-time and ActionDiT noise-sigma schedules."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    base = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    video_clean = 1.0 - _shift(base, video_shift)
    action_sigma = _shift(base, action_shift)
    return H3DreamInferenceSchedule(
        video_clean_times=video_clean[:-1].to(dtype),
        video_clean_deltas=(video_clean[1:] - video_clean[:-1]).to(dtype),
        action_sigmas=action_sigma[:-1].to(dtype),
        action_sigma_deltas=(action_sigma[1:] - action_sigma[:-1]).to(dtype),
    )


@torch.inference_mode()
def sample_h3dream_joint_rows(
    predict_velocity: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    *,
    initial_video_rows: torch.Tensor,
    condition_video_rows: int,
    initial_actions: torch.Tensor,
    schedule: H3DreamInferenceSchedule,
) -> H3DreamJointSample:
    """Euler-integrate future H3 rows and an ActionDiT chunk in lockstep."""

    if initial_video_rows.ndim != 3:
        raise ValueError("initial_video_rows must be [B,rows,width]")
    if initial_actions.ndim != 3:
        raise ValueError("initial_actions must be [B,horizon,action_dim]")
    if initial_video_rows.shape[0] != initial_actions.shape[0]:
        raise ValueError("video/action batch sizes must match")
    if not 0 < condition_video_rows < initial_video_rows.shape[1]:
        raise ValueError("condition_video_rows must leave at least one future row")
    lengths = {
        schedule.video_clean_times.numel(),
        schedule.video_clean_deltas.numel(),
        schedule.action_sigmas.numel(),
        schedule.action_sigma_deltas.numel(),
    }
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("inference schedule tensors must have one positive length")

    video_rows = initial_video_rows.clone()
    condition = video_rows[:, :condition_video_rows].clone()
    actions = initial_actions.clone()
    for video_time, video_delta, action_sigma, action_delta in zip(
        schedule.video_clean_times,
        schedule.video_clean_deltas,
        schedule.action_sigmas,
        schedule.action_sigma_deltas,
        strict=True,
    ):
        video_velocity, action_velocity = predict_velocity(
            video_rows, actions, video_time, action_sigma
        )
        if video_velocity.shape != video_rows.shape:
            raise ValueError("video velocity shape does not match video rows")
        if action_velocity.shape != actions.shape:
            raise ValueError("action velocity shape does not match actions")
        video_rows[:, condition_video_rows:] += (
            video_velocity[:, condition_video_rows:]
            * video_delta.to(video_rows.dtype)
        )
        video_rows[:, :condition_video_rows] = condition
        actions += action_velocity * action_delta.to(actions.dtype)
    return H3DreamJointSample(video_rows=video_rows, actions=actions)
