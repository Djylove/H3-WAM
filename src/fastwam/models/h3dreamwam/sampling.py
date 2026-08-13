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


@dataclass(frozen=True)
class H3LingBotCausalSample:
    """Generated streams plus the clean history exposed to later chunks."""

    video_rows: torch.Tensor
    actions: torch.Tensor
    clean_video_rows: torch.Tensor
    clean_actions: torch.Tensor


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


@torch.inference_mode()
def sample_h3_lingbot_chunk_causal(
    predict_velocity: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[torch.Tensor, torch.Tensor],
    ],
    *,
    initial_video_rows: torch.Tensor,
    observed_video_mask: torch.Tensor,
    video_chunk_ids: torch.Tensor,
    initial_actions: torch.Tensor,
    action_chunk_ids: torch.Tensor,
    video_schedule: H3DreamInferenceSchedule,
    action_schedule: H3DreamInferenceSchedule,
    observed_action_mask: torch.Tensor | None = None,
    ignored_action_mask: torch.Tensor | None = None,
) -> H3LingBotCausalSample:
    """Generate interleaved video/action chunks without clean-future leakage.

    LingBot training contains noisy and clean copies of both modalities. At
    inference, the clean stream may contain only observed or already generated
    rows. For each chunk this routine first denoises video, commits it to clean
    history, then denoises action and commits that action. Unreached clean rows
    stay zero and are invisible through the block-causal attention mask.
    """

    if initial_video_rows.ndim != 3 or initial_actions.ndim != 3:
        raise ValueError("video/actions must be [B,length,width]")
    if initial_video_rows.shape[0] != initial_actions.shape[0]:
        raise ValueError("video/action batch sizes must match")
    video_length = initial_video_rows.shape[1]
    action_length = initial_actions.shape[1]
    observed_video_mask = observed_video_mask.reshape(-1).bool()
    video_chunk_ids = video_chunk_ids.reshape(-1).long()
    action_chunk_ids = action_chunk_ids.reshape(-1).long()
    if observed_video_mask.shape != (video_length,):
        raise ValueError("observed video mask must cover every video row")
    if video_chunk_ids.shape != (video_length,):
        raise ValueError("video chunk ids must cover every video row")
    if action_chunk_ids.shape != (action_length,):
        raise ValueError("action chunk ids must cover every action row")
    if observed_action_mask is None:
        observed_action_mask = torch.zeros_like(action_chunk_ids, dtype=torch.bool)
    else:
        observed_action_mask = observed_action_mask.reshape(-1).bool()
    if observed_action_mask.shape != (action_length,):
        raise ValueError("observed action mask must cover every action row")
    if ignored_action_mask is None:
        ignored_action_mask = torch.zeros_like(action_chunk_ids, dtype=torch.bool)
    else:
        ignored_action_mask = ignored_action_mask.reshape(-1).bool()
    if ignored_action_mask.shape != (action_length,):
        raise ValueError("ignored action mask must cover every action row")
    if bool((observed_action_mask & ignored_action_mask).any()):
        raise ValueError("an action row cannot be both observed and ignored")
    if not bool(observed_video_mask.any()):
        raise ValueError("at least one observed video row is required")
    if int(video_chunk_ids.min()) < 0 or int(action_chunk_ids.min()) < 0:
        raise ValueError("chunk ids must be non-negative")

    video_rows = initial_video_rows.clone()
    actions = initial_actions.clone()
    clean_video = torch.zeros_like(video_rows)
    clean_video[:, observed_video_mask] = video_rows[:, observed_video_mask]
    clean_actions = torch.zeros_like(actions)
    clean_actions[:, observed_action_mask] = actions[:, observed_action_mask]
    clean_video_valid = observed_video_mask.clone()
    clean_action_valid = observed_action_mask.clone()
    chunks = torch.unique(
        torch.cat((video_chunk_ids, action_chunk_ids)), sorted=True
    ).tolist()
    one = torch.ones((), device=video_rows.device, dtype=torch.float32)

    for chunk in chunks:
        video_selection = (video_chunk_ids == int(chunk)) & ~observed_video_mask
        if bool(video_selection.any()):
            for video_time, video_delta in zip(
                video_schedule.video_clean_times,
                video_schedule.video_clean_deltas,
                strict=True,
            ):
                video_velocity, _ = predict_velocity(
                    video_rows,
                    clean_video,
                    actions,
                    clean_actions,
                    video_time,
                    one,
                    clean_video_valid,
                    clean_action_valid,
                )
                if video_velocity.shape != video_rows.shape:
                    raise ValueError("video velocity shape does not match video rows")
                video_rows[:, video_selection] += (
                    video_velocity[:, video_selection]
                    * video_delta.to(video_rows.dtype)
                )
                video_rows[:, observed_video_mask] = clean_video[
                    :, observed_video_mask
                ]
            clean_video[:, video_selection] = video_rows[:, video_selection]
            clean_video_valid[video_selection] = True

        action_selection = (
            (action_chunk_ids == int(chunk))
            & ~observed_action_mask
            & ~ignored_action_mask
        )
        if bool(action_selection.any()):
            for action_sigma, action_delta in zip(
                action_schedule.action_sigmas,
                action_schedule.action_sigma_deltas,
                strict=True,
            ):
                _, action_velocity = predict_velocity(
                    video_rows,
                    clean_video,
                    actions,
                    clean_actions,
                    one,
                    action_sigma,
                    clean_video_valid,
                    clean_action_valid,
                )
                if action_velocity.shape != actions.shape:
                    raise ValueError("action velocity shape does not match actions")
                actions[:, action_selection] += (
                    action_velocity[:, action_selection]
                    * action_delta.to(actions.dtype)
                )
            clean_actions[:, action_selection] = actions[:, action_selection]
            clean_action_valid[action_selection] = True

    return H3LingBotCausalSample(
        video_rows=video_rows,
        actions=actions,
        clean_video_rows=clean_video,
        clean_actions=clean_actions,
    )
