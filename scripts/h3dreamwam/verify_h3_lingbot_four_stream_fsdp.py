#!/usr/bin/env python3
"""Full-50-layer FSDP smoke for the H3 LingBot four-stream model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--last-trainable-layers", type=int, default=2)
    parser.add_argument("--video-frames", type=int, default=2)
    parser.add_argument("--tokens-per-frame", type=int, default=98)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument(
        "--executed-action-history-steps",
        type=int,
        default=0,
        help=(
            "Number of real actions immediately preceding each window to expose "
            "as fixed clean history. These tokens are excluded from action loss."
        ),
    )
    parser.add_argument(
        "--executed-action-history-root",
        type=Path,
        help="Episode action sidecars produced by build_executed_action_history.py.",
    )
    parser.add_argument(
        "--allow-history-bootstrap",
        action="store_true",
        help="Allow a history-conditioned run to initialize from a legacy history=0 stage.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument(
        "--action-train-shift",
        type=float,
        default=0.05,
        help=(
            "FlowMatch shift for sampling action training noise and weighting "
            "its loss. LingBot LIBERO uses 0.05; other released WAM recipes "
            "use 1.0 or 5.0."
        ),
    )
    parser.add_argument(
        "--action-infer-shift",
        type=float,
        default=0.05,
        help=(
            "FlowMatch shift for the action sampling schedule. Kept separate "
            "from action-train-shift so training-support and solver effects "
            "can be evaluated independently."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-stage", type=Path)
    parser.add_argument("--load-stage", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--base-completed-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rotate-windows", action="store_true")
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--random-timesteps", action="store_true")
    parser.add_argument("--freeze-shared-blocks", action="store_true")
    parser.add_argument("--mask-clean-future", action="store_true")
    parser.add_argument("--sample-eval", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--video-sample-steps", type=int, default=0)
    parser.add_argument("--action-sample-steps", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument(
        "--action-normalization",
        choices=("minmax", "quantile"),
        default="minmax",
        help="Action normalization contract. Quantile follows LingBot-VA.",
    )
    parser.add_argument(
        "--action-stats-json",
        type=Path,
        help="JSON containing q01/q99 arrays; required for quantile normalization.",
    )
    parser.add_argument(
        "--per-chunk-action-timesteps",
        action="store_true",
        help="Sample one LingBot-style training timestep per action chunk.",
    )
    parser.add_argument(
        "--noisy-clean-video-prob",
        type=float,
        default=0.0,
        help="Probability of LingBot-style noise on the clean video stream.",
    )
    parser.add_argument(
        "--detached-generated-video-conditioning",
        action="store_true",
        help="Train action on a detached one-step video x0 prediction.",
    )
    parser.add_argument(
        "--h3-physical-time-alignment",
        action="store_true",
        help="Align H3 latent rows using the non-uniform H3 VAE clock.",
    )
    parser.add_argument(
        "--flow-match-loss-weighting",
        action="store_true",
        help="Apply the LingBot/DreamWAM timestep-dependent training weights.",
    )
    parser.add_argument(
        "--upstream-initial-action-anchor",
        action="store_true",
        help="Prepend one zero action chunk as LingBot's initial history frame.",
    )
    parser.add_argument(
        "--shared-backbone",
        action="store_true",
        help="Use LingBot's code-aligned shared H3 block stack instead of ActionDiT.",
    )
    return parser.parse_args()


def is_checkpoint_milestone(
    step: int,
    *,
    base_completed_steps: int,
    checkpoint_every: int,
    total_steps: int,
) -> bool:
    """Return whether this update lands on a cumulative checkpoint boundary."""

    if checkpoint_every <= 0 or step >= total_steps:
        return False
    return (base_completed_steps + step) % checkpoint_every == 0


def global_grad_norm(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    marker: str,
    device: torch.device,
) -> float:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, parameter in named_parameters:
        if marker in name and parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    dist.all_reduce(total)
    return float(total.sqrt())


def shifted_noise_sigma(uniform: torch.Tensor, shift: float) -> torch.Tensor:
    """FlowMatch shifted noise sigma used by H3/LingBot schedulers."""

    return float(shift) * uniform / (1.0 + (float(shift) - 1.0) * uniform)


def flow_match_training_weight(
    noise_sigma: torch.Tensor,
    *,
    shift: float,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Reproduce LingBot/DreamWAM's normalized bell-shaped loss weight.

    ``noise_sigma`` is already shifted, matching the scheduler timestep divided
    by ``num_train_timesteps``. The normalization is computed over the same
    discrete uniform pre-shift training grid used by the released schedulers.
    """

    if shift <= 0 or num_train_timesteps <= 0:
        raise ValueError("shift and num_train_timesteps must be positive")
    sigma = noise_sigma.float()
    raw_grid = torch.linspace(
        1.0,
        0.0,
        num_train_timesteps + 1,
        device=sigma.device,
        dtype=torch.float64,
    )[:-1]
    shifted_grid = shifted_noise_sigma(raw_grid, shift)
    grid_weight = torch.exp(-2.0 * (shifted_grid - 0.5).square())
    weight_min = grid_weight.min()
    weight_mean = (grid_weight - weight_min).mean().clamp_min(1.0e-10)
    weight = torch.exp(-2.0 * (sigma.double() - 0.5).square())
    return ((weight - weight_min) / weight_mean).to(sigma.dtype)


def weighted_video_action_losses(
    *,
    video_prediction: torch.Tensor,
    video_target: torch.Tensor,
    future: torch.Tensor,
    noisy_video_timesteps: torch.Tensor,
    noisy_video_timestep_indices: torch.Tensor,
    action_prediction: torch.Tensor,
    action_target: torch.Tensor,
    action_time: torch.Tensor,
    action_timestep_indices: torch.Tensor,
    action_loss_mask: torch.Tensor | None = None,
    action_shift: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame/token-wise official flow weighting for the two training losses."""

    video_per_row = F.mse_loss(
        video_prediction[:, future].float(),
        video_target[:, future].float(),
        reduction="none",
    ).mean(dim=-1)
    video_clean_time = noisy_video_timesteps.index_select(
        0, noisy_video_timestep_indices[future]
    )
    video_weight = flow_match_training_weight(
        1.0 - video_clean_time, shift=12.0
    )
    action_per_token = F.mse_loss(
        action_prediction.float(), action_target.float(), reduction="none"
    ).mean(dim=-1)
    action_clean_time = action_time.index_select(0, action_timestep_indices)
    action_weight = flow_match_training_weight(
        1.0 - action_clean_time, shift=action_shift
    )
    if action_loss_mask is not None:
        action_loss_mask = action_loss_mask.reshape(-1).bool()
        if action_loss_mask.shape != action_timestep_indices.shape:
            raise ValueError("action loss mask must cover every action token")
        if not bool(action_loss_mask.any()):
            raise ValueError("action loss mask must select at least one token")
        action_per_token = action_per_token[:, action_loss_mask]
        action_weight = action_weight[action_loss_mask]
    return (
        (video_per_row * video_weight[None]).mean(),
        (action_per_token * action_weight[None]).mean(),
    )


def video_clean_from_velocity(
    noisy_video: torch.Tensor,
    clean_time_per_token: torch.Tensor,
    clean_minus_noise_velocity: torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from H3's clean-time, clean-minus-noise convention."""

    sigma = 1.0 - clean_time_per_token
    return noisy_video + sigma[None, :, None] * clean_minus_noise_velocity


def normalize_action(
    action: torch.Tensor,
    *,
    mode: str,
    stats: dict,
    quantile_stats: dict | None,
) -> torch.Tensor:
    if mode == "minmax":
        low = stats["action_min"].float()
        high = stats["action_max"].float()
        clip = 1.0
    elif mode == "quantile":
        if quantile_stats is None:
            raise ValueError("quantile action normalization requires action stats")
        low = torch.as_tensor(quantile_stats["q01"], dtype=torch.float32)
        high = torch.as_tensor(quantile_stats["q99"], dtype=torch.float32)
        clip = 1.5
    else:
        raise ValueError(f"unsupported action normalization: {mode}")
    scale = (high - low).clamp_min(1e-6)
    return ((action.float() - low) / scale * 2.0 - 1.0).clamp(-clip, clip)


def prepend_initial_action_history(
    action: torch.Tensor,
    *,
    history_steps: int,
    horizon: int,
) -> torch.Tensor:
    """Match LingBot's raw-space zero action frame before normalization."""

    if action.ndim != 2 or action.shape[-1] != 7:
        raise ValueError(f"expected [T,7] actions, got {action.shape}")
    if history_steps <= 0 or horizon <= history_steps:
        raise ValueError("history steps must be positive and shorter than horizon")
    return torch.cat(
        (
            torch.zeros(
                history_steps,
                action.shape[-1],
                dtype=action.dtype,
                device=action.device,
            ),
            action,
        ),
        dim=0,
    )[:horizon]


def load_executed_action_history(
    row: dict,
    *,
    history_root: Path,
    history_steps: int,
) -> torch.Tensor:
    """Load the real actions immediately preceding a dense training window."""

    if history_steps <= 0:
        return torch.empty(0, 7, dtype=torch.float32)
    suite = str(row["suite"])
    episode = int(row["episode"])
    start = int(row["start"])
    path = history_root.resolve() / "actions" / f"{suite}_ep{episode:06d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actions = payload["actions"].float()
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"invalid action history sidecar shape: {actions.shape}")
    if start < 0 or start > len(actions):
        raise ValueError(f"window start {start} is outside episode length {len(actions)}")
    history = actions[max(0, start - history_steps) : start]
    if len(history) < history_steps:
        history = torch.cat(
            (torch.zeros(history_steps - len(history), 7), history), dim=0
        )
    return history


def prepare_real_batch(
    *,
    args: argparse.Namespace,
    row: dict,
    row_index: int,
    device: torch.device,
    generator: torch.Generator,
    model_root: torch.nn.Module,
    stats: dict,
    quantile_stats: dict | None,
    patchify_video_latents,
    layout_builder,
    align_chunk_ids,
    action_time_builder,
    deterministic_noise: bool,
) -> dict:
    sample_id = str(row["id"])
    context_id = str(row.get("context_id", sample_id))
    data_root = args.data_root.resolve()
    window = torch.load(
        data_root / "windows" / f"{sample_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    conditioning = torch.load(
        data_root / "contexts" / f"{context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    sample_generator = generator
    if deterministic_noise:
        sample_generator = torch.Generator(device=device).manual_seed(
            args.seed + 1000003 * int(row_index)
        )
    future_latents = window["video_latents"].to(
        device=device, dtype=torch.float32
    )
    first_latent = window["first_frame_latents"].to(
        device=device, dtype=torch.float32
    )
    _, _, latent_frames, latent_height, latent_width = future_latents.shape
    text_tags = torch.cat(
        (conditioning["token_tags"].long(), torch.ones(1, dtype=torch.long))
    )
    layout = layout_builder.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=10,
        patch_size=tuple(model_root.h3.config.patch_size),
        audio_channels=8,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=("first",),
    )
    position_ids, _, video_indices, _, text_indices, condition_rows, _ = layout
    video_position_ids = position_ids.index_select(0, video_indices).to(device)
    context_position_ids = position_ids.index_select(0, text_indices).to(device)
    first_rows = patchify_video_latents(
        first_latent, tuple(model_root.h3.config.patch_size)
    )[None]
    future_rows = patchify_video_latents(
        future_latents, tuple(model_root.h3.config.patch_size)
    )[None]
    clean_video = torch.cat((first_rows, future_rows), dim=1)
    video_noise = torch.randn(
        clean_video.shape, generator=sample_generator, device=device
    )
    action_timestep_indices = torch.zeros(
        args.action_horizon, device=device, dtype=torch.long
    )
    if args.random_timesteps and not deterministic_noise:
        video_uniform = torch.rand(1, generator=sample_generator, device=device)
        action_uniform = torch.rand(
            (
                math.ceil(args.action_horizon / args.actions_per_chunk)
                if args.per_chunk_action_timesteps
                else 1
            ),
            generator=sample_generator,
            device=device,
        )
        video_noise_sigma = shifted_noise_sigma(video_uniform, 12.0)
        action_noise_sigma = shifted_noise_sigma(
            action_uniform, args.action_train_shift
        )
        if args.per_chunk_action_timesteps:
            action_timestep_indices = torch.div(
                torch.arange(args.action_horizon, device=device),
                args.actions_per_chunk,
                rounding_mode="floor",
            )
    else:
        video_noise_sigma = torch.tensor([0.5], device=device)
        action_noise_sigma = torch.tensor([0.5], device=device)
    video_time = 1.0 - video_noise_sigma
    action_time = 1.0 - action_noise_sigma
    noisy_future = video_time[:, None, None] * future_rows
    noisy_future += (1.0 - video_time[:, None, None]) * video_noise[:, condition_rows:]
    noisy_video = torch.cat((first_rows, noisy_future), dim=1)
    initial_video = torch.cat((first_rows, video_noise[:, condition_rows:]), dim=1)
    clean_video_input = clean_video
    clean_video_timesteps = torch.ones(1, device=device)
    clean_video_timestep_indices = torch.zeros(
        clean_video.shape[1], device=device, dtype=torch.long
    )
    if not deterministic_noise and args.noisy_clean_video_prob > 0.0:
        condition_generator = torch.Generator(device=device).manual_seed(
            args.seed + 2000003 * int(row_index)
        )
        corrupt = torch.rand(
            (), generator=condition_generator, device=device
        ) < args.noisy_clean_video_prob
        if bool(corrupt):
            _, clean_video_timestep_indices = torch.unique(
                video_position_ids[:, 0], sorted=True, return_inverse=True
            )
            condition_uniform = torch.rand(
                int(clean_video_timestep_indices.max()) + 1,
                generator=condition_generator,
                device=device,
            )
            condition_uniform = condition_uniform * 0.5 + 0.5
            condition_sigma = shifted_noise_sigma(condition_uniform, 12.0)
            clean_video_timesteps = 1.0 - condition_sigma
            condition_sigma_rows = condition_sigma.index_select(
                0, clean_video_timestep_indices
            )[None, :, None]
            condition_noise = torch.randn(
                clean_video.shape,
                generator=condition_generator,
                device=device,
            )
            clean_video_input = (
                (1.0 - condition_sigma_rows) * clean_video
                + condition_sigma_rows * condition_noise
            )

    target_raw_action = window["actions"][: args.action_horizon]
    history_steps = int(args.executed_action_history_steps)
    if history_steps:
        if args.executed_action_history_root is None:
            raise ValueError(
                "--executed-action-history-root is required when history is enabled"
            )
        history_raw_action = load_executed_action_history(
            row,
            history_root=args.executed_action_history_root,
            history_steps=history_steps,
        )
        raw_action = torch.cat((history_raw_action, target_raw_action), dim=0)
    else:
        raw_action = target_raw_action
    if args.upstream_initial_action_anchor:
        if history_steps:
            raise ValueError(
                "executed action history and initial zero anchor are separate contracts"
            )
        raw_action = prepend_initial_action_history(
            raw_action,
            history_steps=args.actions_per_chunk,
            horizon=args.action_horizon,
        )
    normalized_action = normalize_action(
        raw_action,
        mode=args.action_normalization,
        stats=stats,
        quantile_stats=quantile_stats,
    ).to(device)
    clean_action = normalized_action[None]
    if history_steps:
        action_timestep_indices = torch.cat(
            (
                torch.zeros(history_steps, device=device, dtype=torch.long),
                action_timestep_indices,
            )
        )
    action_noise = torch.randn(
        clean_action.shape, generator=sample_generator, device=device
    )
    initial_action = action_noise.clone()
    action_loss_mask = torch.ones(
        clean_action.shape[1], device=device, dtype=torch.bool
    )
    if args.upstream_initial_action_anchor:
        initial_action[:, : args.actions_per_chunk] = 0.0
    action_sigma_per_token = action_noise_sigma.index_select(
        0, action_timestep_indices
    )[None, :, None]
    noisy_action = (1.0 - action_sigma_per_token) * clean_action
    noisy_action += action_sigma_per_token * action_noise
    if history_steps:
        noisy_action[:, :history_steps] = clean_action[:, :history_steps]
        initial_action[:, :history_steps] = clean_action[:, :history_steps]
        action_loss_mask[:history_steps] = False
    context = conditioning["context"].to(device=device, dtype=torch.bfloat16)
    state_scale = (
        stats["state_max"].float() - stats["state_min"].float()
    ).clamp_min(1e-6)
    state = (
        (window["state"].float() - stats["state_min"].float())
        / state_scale
        * 2.0
        - 1.0
    ).clamp(-1.0, 1.0)[None].to(device)
    context_mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
    alignment_kwargs = {}
    if args.h3_physical_time_alignment:
        alignment_kwargs["h3_frame_count"] = int(window["h3_frame_count"])
    video_chunks, target_action_chunks = align_chunk_ids(
        video_frame_ids=video_position_ids[:, 0],
        action_horizon=args.action_horizon,
        actions_per_chunk=args.actions_per_chunk,
        **alignment_kwargs,
    )
    history_chunks = history_steps // args.actions_per_chunk
    if history_steps:
        if history_steps % args.actions_per_chunk:
            raise ValueError("history steps must be divisible by actions-per-chunk")
        video_chunks = video_chunks + history_chunks
        action_chunks = torch.cat(
            (
                torch.div(
                    torch.arange(history_steps, device=device),
                    args.actions_per_chunk,
                    rounding_mode="floor",
                ),
                target_action_chunks + history_chunks,
            )
        )
    else:
        action_chunks = target_action_chunks
    video_tokens = clean_video.shape[1]
    noisy_video_timestep_indices = torch.ones(
        video_tokens, device=device, dtype=torch.long
    )
    noisy_video_timestep_indices[:condition_rows] = 0
    future = torch.ones(video_tokens, device=device, dtype=torch.bool)
    future[:condition_rows] = False
    target_action_temporal_positions = (
        action_time_builder(
            video_frame_ids=video_position_ids[:, 0],
            action_horizon=args.action_horizon,
            actions_per_chunk=args.actions_per_chunk,
            h3_frame_count=int(window["h3_frame_count"]),
        )
        if args.h3_physical_time_alignment
        else torch.arange(args.action_horizon, device=device).float()
        / args.actions_per_chunk
    )
    if history_steps:
        history_temporal_positions = (
            torch.arange(-history_steps, 0, device=device).float()
            / args.actions_per_chunk
        )
        action_temporal_positions = torch.cat(
            (history_temporal_positions, target_action_temporal_positions)
        )
    else:
        action_temporal_positions = target_action_temporal_positions
    action_position_ids = torch.stack(
        (
            action_temporal_positions,
            torch.full_like(action_temporal_positions, -1.0),
            torch.full_like(action_temporal_positions, -1.0),
        ),
        dim=-1,
    )
    return {
        "sample_id": sample_id,
        "video_tokens": video_tokens,
        "noisy_video": noisy_video,
        "initial_video": initial_video,
        "clean_video": clean_video,
        "clean_video_input": clean_video_input,
        "video_position_ids": video_position_ids,
        "video_chunks": video_chunks,
        "noisy_video_timesteps": torch.cat(
            (torch.tensor([0.9], device=device), video_time)
        ),
        "clean_video_timesteps": clean_video_timesteps,
        "noisy_video_timestep_indices": noisy_video_timestep_indices,
        "clean_video_timestep_indices": clean_video_timestep_indices,
        "noisy_action": noisy_action,
        "initial_action": initial_action,
        "clean_action": clean_action,
        "action_position_ids": action_position_ids,
        "action_chunks": action_chunks,
        "action_time": action_time,
        "action_timestep_indices": action_timestep_indices,
        "action_loss_mask": action_loss_mask,
        "observed_action_mask": ~action_loss_mask,
        "context": context,
        "context_position_ids": context_position_ids,
        "state": state,
        "context_mask": context_mask,
        "video_target": clean_video - video_noise,
        "action_target": action_noise - clean_action,
        "future": future,
    }


def main() -> None:
    args = parse_args()
    if args.mask_clean_future and not args.eval_only:
        raise ValueError("mask-clean-future is an evaluation-only intervention")
    if not 0.0 <= args.noisy_clean_video_prob <= 1.0:
        raise ValueError("noisy-clean-video-prob must be in [0,1]")
    if args.executed_action_history_steps < 0:
        raise ValueError("executed action history steps cannot be negative")
    if args.executed_action_history_steps and args.data_root is None:
        raise ValueError("executed action history requires a real cached dataset")
    if (
        args.executed_action_history_steps
        and args.executed_action_history_steps % args.actions_per_chunk
    ):
        raise ValueError("history steps must be divisible by actions-per-chunk")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if args.action_train_shift <= 0.0 or args.action_infer_shift <= 0.0:
        raise ValueError("action train/infer shifts must be positive")
    if args.checkpoint_every < 0 or args.base_completed_steps < 0:
        raise ValueError("checkpoint cadence and base completed steps must be non-negative")
    if args.checkpoint_every and args.save_stage is None:
        raise ValueError("checkpoint-every requires save-stage")
    if args.noisy_clean_video_prob > 0.0 and args.eval_only:
        raise ValueError("noisy clean video is a training-only intervention")
    if args.sample_eval and (not args.eval_only or not args.shared_backbone):
        raise ValueError("sample-eval requires eval-only and shared-backbone")
    if args.h3_physical_time_alignment and args.data_root is None:
        raise ValueError("H3 physical-time alignment requires a real cached window")
    if (
        args.sample_steps <= 0
        or args.video_sample_steps < 0
        or args.action_sample_steps < 0
        or args.eval_limit < 0
    ):
        raise ValueError("sample step counts must be positive/default-zero and eval-limit non-negative")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or not torch.cuda.is_available():
        raise RuntimeError("this smoke requires multi-GPU CUDA torchrun")
    if not 1 <= args.last_trainable_layers <= 50:
        raise ValueError("last-trainable-layers must be in [1,50]")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        patchify_video_latents,
    )
    from fastwam.models.h3dreamwam import (
        H3DreamActionExpert,
        H3LingBotPairedLayer,
        H3LingBotSharedLayer,
        H3LingBotSharedWAM,
        H3LingBotWAM,
        align_h3_action_chunk_ids,
        build_h3dream_inference_schedule,
        h3_action_temporal_positions,
        initialize_action_expert_from_h3,
        sample_h3_lingbot_chunk_causal,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    if (args.data_root is None) != (args.manifest is None):
        raise ValueError("data-root and manifest must be supplied together")

    started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        if args.shared_backbone:
            action_expert = None
            initialization = {
                "type": "lingbot_shared_h3_blocks",
                "action_modality_id": 2,
            }
        else:
            action_expert = H3DreamActionExpert(
                action_dim=7,
                state_dim=8,
                text_dim=5120,
                hidden_dim=1024,
                ffn_dim=4096,
                num_heads=56,
                head_dim=128,
                num_layers=50,
                frequency_dim=256,
                full_width_rmsnorm=True,
            )
            initialization = initialize_action_expert_from_h3(
                action_expert, h3, alpha_scaling=True
            ).__dict__
    finally:
        torch.set_default_dtype(previous_dtype)
    h3.requires_grad_(False)
    for block in h3.transformer_blocks[-args.last_trainable_layers :]:
        block.requires_grad_(True)
    h3.proj_out.requires_grad_(True)
    if args.shared_backbone:
        model = H3LingBotSharedWAM(
            h3,
            action_dim=7,
            state_dim=8,
            text_dim=5120,
            use_gradient_checkpointing=True,
            compute_dtype=torch.bfloat16,
        )
        model.action_adapters.requires_grad_(True)
        ignored_modules = [*model.h3.children(), *model.action_adapters.children()]
        wrap_class = H3LingBotSharedLayer
    else:
        action_expert.requires_grad_(False)
        for block in action_expert.blocks[-args.last_trainable_layers :]:
            block.requires_grad_(True)
        action_expert.output.requires_grad_(True)
        model = H3LingBotWAM(
            h3,
            action_expert,
            use_gradient_checkpointing=True,
            compute_dtype=torch.bfloat16,
        )
        ignored_modules = [*model.h3.children(), *model.action_expert.children()]
        wrap_class = H3LingBotPairedLayer
    if args.freeze_shared_blocks:
        if not args.shared_backbone:
            raise ValueError("freeze-shared-blocks requires shared-backbone")
        for layer in model.shared_layers:
            layer.requires_grad_(False)
        model.h3.proj_out.requires_grad_(False)
    ignored_modules = [
        module for module in ignored_modules if len(list(module.parameters())) > 0
    ]
    for module in ignored_modules:
        module.to(device)
    model.train()
    model = FSDP(
        model,
        # The paired unit must be the only nested FSDP boundary. Wrapping its
        # H3 block again leaves AdaLN parameters as 1-D shards during the outer
        # forward (observed as ``mat2 must be a matrix``).
        auto_wrap_policy=ModuleWrapPolicy({wrap_class}),
        ignored_modules=ignored_modules,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
    )
    # Stable low-LR updates require FP32 optimizer storage after FSDP has
    # sharded the 33B/ActionDiT blocks. Forward compute remains BF16 through
    # MixedPrecision, matching the proven H3-DreamWAM training path.
    model.float()
    if args.sample_eval:
        model.eval()
    # FSDP establishes root/non-root ownership lazily. Checkpoint restore may
    # summon nested layer parameters before the first real forward, so force
    # hierarchy initialization from the true root after the final storage
    # dtype has been set. Otherwise a child may mark itself as root, or its
    # all-gather buffer may retain the pre-conversion BF16 dtype.
    model.check_is_root()
    if args.load_stage is not None:
        payload = torch.load(
            args.load_stage.resolve(), map_location="cpu", weights_only=True
        )
        expected_format = (
            "h3_lingbot_shared_four_stream_tail_v1"
            if args.shared_backbone
            else "h3_lingbot_four_stream_tail_v1"
        )
        if payload.get("format") != expected_format:
            raise ValueError("four-stream stage checkpoint format mismatch")
        if payload.get("last_trainable_layers") != args.last_trainable_layers:
            raise ValueError("four-stream stage layer count mismatch")
        checkpoint_normalization = payload.get("action_normalization", "minmax")
        if checkpoint_normalization != args.action_normalization:
            raise ValueError(
                "stage action normalization mismatch: "
                f"{checkpoint_normalization} != {args.action_normalization}"
            )
        checkpoint_per_chunk = payload.get("per_chunk_action_timesteps", False)
        if checkpoint_per_chunk != args.per_chunk_action_timesteps:
            raise ValueError(
                "stage action timestep contract mismatch: "
                f"{checkpoint_per_chunk} != {args.per_chunk_action_timesteps}"
            )
        checkpoint_generated_video = payload.get(
            "detached_generated_video_conditioning", False
        )
        if (
            checkpoint_generated_video
            != args.detached_generated_video_conditioning
        ):
            raise ValueError(
                "stage generated-video conditioning contract mismatch: "
                f"{checkpoint_generated_video} != "
                f"{args.detached_generated_video_conditioning}"
            )
        checkpoint_action_anchor = payload.get(
            "upstream_initial_action_anchor", False
        )
        if checkpoint_action_anchor != args.upstream_initial_action_anchor:
            raise ValueError(
                "stage initial action anchor contract mismatch: "
                f"{checkpoint_action_anchor} != "
                f"{args.upstream_initial_action_anchor}"
            )
        checkpoint_history_steps = int(
            payload.get("executed_action_history_steps", 0)
        )
        if checkpoint_history_steps != args.executed_action_history_steps:
            bootstrap_ok = (
                args.allow_history_bootstrap
                and checkpoint_history_steps == 0
                and args.executed_action_history_steps > 0
            )
            if not bootstrap_ok:
                raise ValueError(
                    "stage executed-action history contract mismatch: "
                    f"{checkpoint_history_steps} != "
                    f"{args.executed_action_history_steps}"
                )
        checkpoint_physical_time = payload.get(
            "h3_physical_time_alignment", False
        )
        if checkpoint_physical_time != args.h3_physical_time_alignment:
            raise ValueError(
                "stage H3 temporal-alignment contract mismatch: "
                f"{checkpoint_physical_time} != "
                f"{args.h3_physical_time_alignment}"
            )
        checkpoint_loss_weighting = payload.get(
            "flow_match_loss_weighting", False
        )
        if checkpoint_loss_weighting != args.flow_match_loss_weighting:
            raise ValueError(
                "stage flow-match loss-weighting contract mismatch: "
                f"{checkpoint_loss_weighting} != "
                f"{args.flow_match_loss_weighting}"
            )
        checkpoint_action_train_shift = payload.get(
            "action_train_shift", 0.05
        )
        if not math.isclose(
            float(checkpoint_action_train_shift),
            args.action_train_shift,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "stage action training-shift contract mismatch: "
                f"{checkpoint_action_train_shift} != {args.action_train_shift}"
            )
        checkpoint_weight_decay = payload.get("weight_decay", 1.0e-2)
        if checkpoint_weight_decay != args.weight_decay:
            raise ValueError(
                "stage weight-decay contract mismatch: "
                f"{checkpoint_weight_decay} != {args.weight_decay}"
            )
        for index_text, layer_state in payload["layers"].items():
            layer = (
                model.module.shared_layers[int(index_text)]
                if args.shared_backbone
                else model.module.paired_layers[int(index_text)]
            )
            with FSDP.summon_full_params(layer, recurse=False, writeback=True):
                layer.module.load_state_dict(layer_state, strict=True)
        with FSDP.summon_full_params(model, recurse=False, writeback=True):
            model.module.h3.proj_out.load_state_dict(
                payload["h3_proj_out"], strict=True
            )
            if args.shared_backbone:
                model.module.action_adapters.load_state_dict(
                    payload["action_adapters"], strict=True
                )
            else:
                model.module.action_expert.output.load_state_dict(
                    payload["action_output"], strict=True
                )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            float(step) / float(max(1, args.warmup_steps))
            if args.warmup_steps > 0 and step < args.warmup_steps
            else 1.0
        ),
    )
    load_seconds = time.perf_counter() - started

    sigma = torch.tensor([0.5], device=device)
    sample_id = "synthetic"
    rows = []
    stats = None
    quantile_stats = None
    if args.data_root is None:
        video_tokens = args.video_frames * args.tokens_per_frame
        input_width = model.module.h3.proj_in.in_features
        frame_ids = torch.arange(args.video_frames, device=device).repeat_interleave(
            args.tokens_per_frame
        )
        video_chunks, action_chunks = align_h3_action_chunk_ids(
            video_frame_ids=frame_ids,
            action_horizon=args.action_horizon,
            actions_per_chunk=args.actions_per_chunk,
        )
        side = max(1, int(math.sqrt(args.tokens_per_frame)))
        spatial = torch.arange(args.tokens_per_frame, device=device)
        positions = []
        for frame in range(args.video_frames):
            positions.append(
                torch.stack(
                    (
                        torch.full_like(spatial, frame),
                        torch.div(spatial, side, rounding_mode="floor"),
                        spatial % side,
                    ),
                    dim=-1,
                ).float()
            )
        video_position_ids = torch.cat(positions, dim=0)
        context_position_ids = torch.zeros(3, 3, device=device)
        clean_video = torch.randn(
            1, video_tokens, input_width, generator=generator, device=device
        )
        clean_video_input = clean_video
        video_noise = torch.randn(
            clean_video.shape, generator=generator, device=device
        )
        noisy_video = (1.0 - sigma[:, None, None]) * clean_video
        noisy_video += sigma[:, None, None] * video_noise
        clean_action = torch.randn(
            1, args.action_horizon, 7, generator=generator, device=device
        )
        action_timestep_indices = torch.zeros(
            args.action_horizon, device=device, dtype=torch.long
        )
        action_noise = torch.randn(
            clean_action.shape, generator=generator, device=device
        )
        context = torch.randn(
            1, 2, 5120, generator=generator, device=device, dtype=torch.bfloat16
        )
        state = torch.randn(1, 8, generator=generator, device=device)
        context_mask = torch.ones(1, 2, device=device, dtype=torch.bool)
        noisy_video_timesteps = sigma
        noisy_video_timestep_indices = torch.zeros(
            video_tokens, device=device, dtype=torch.long
        )
        clean_video_timesteps = torch.ones(1, device=device)
        clean_video_timestep_indices = torch.zeros(
            video_tokens, device=device, dtype=torch.long
        )
        future = video_chunks > video_chunks.min()
    else:
        rows = [
            json.loads(line)
            for line in args.manifest.resolve().read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("manifest contains no rows")
        stats = torch.load(
            args.data_root.resolve() / "stats.pt",
            map_location="cpu",
            weights_only=False,
        )
        if args.action_normalization == "quantile":
            if args.action_stats_json is None:
                raise ValueError(
                    "--action-stats-json is required with quantile normalization"
                )
            quantile_stats = json.loads(
                args.action_stats_json.resolve().read_text()
            )
            if len(quantile_stats.get("q01", [])) != 7 or len(
                quantile_stats.get("q99", [])
            ) != 7:
                raise ValueError("action q01/q99 must each contain seven values")
        batch = prepare_real_batch(
            args=args,
            row=rows[(args.sample_offset * world_size + rank) % len(rows)],
            row_index=(args.sample_offset * world_size + rank) % len(rows),
            device=device,
            generator=generator,
            model_root=model.module,
            stats=stats,
            quantile_stats=quantile_stats,
            patchify_video_latents=patchify_video_latents,
            layout_builder=MiniMaxH3PrepareLayoutStep,
            align_chunk_ids=align_h3_action_chunk_ids,
            action_time_builder=h3_action_temporal_positions,
            deterministic_noise=args.eval_only,
        )

    if args.data_root is None:
        noisy_action = (1.0 - sigma[:, None, None]) * clean_action
        noisy_action += sigma[:, None, None] * action_noise
        video_target = clean_video - video_noise
        action_target = action_noise - clean_action
        action_position_ids = torch.stack(
            (
                torch.arange(args.action_horizon, device=device).float()
                / args.actions_per_chunk,
                torch.full((args.action_horizon,), -1.0, device=device),
                torch.full((args.action_horizon,), -1.0, device=device),
            ),
            dim=-1,
        )

    torch.cuda.reset_peak_memory_stats(device)
    history = []

    def save_training_stage(path: Path, completed_steps: int) -> None:
        stage = {
            "format": (
                "h3_lingbot_shared_four_stream_tail_v1"
                if args.shared_backbone
                else "h3_lingbot_four_stream_tail_v1"
            ),
            "last_trainable_layers": args.last_trainable_layers,
            "warmup_steps": args.warmup_steps,
            "action_normalization": args.action_normalization,
            "action_quantile_stats": quantile_stats,
            "per_chunk_action_timesteps": args.per_chunk_action_timesteps,
            "noisy_clean_video_prob": args.noisy_clean_video_prob,
            "detached_generated_video_conditioning": (
                args.detached_generated_video_conditioning
            ),
            "h3_physical_time_alignment": args.h3_physical_time_alignment,
            "flow_match_loss_weighting": args.flow_match_loss_weighting,
            "action_train_shift": args.action_train_shift,
            "upstream_initial_action_anchor": args.upstream_initial_action_anchor,
            "executed_action_history_steps": args.executed_action_history_steps,
            "weight_decay": args.weight_decay,
            "completed_steps": int(completed_steps),
            "sample_offset": int(args.sample_offset),
            "layers": {},
        }
        for index in range(50 - args.last_trainable_layers, 50):
            layer = (
                model.module.shared_layers[index]
                if args.shared_backbone
                else model.module.paired_layers[index]
            )
            with FSDP.summon_full_params(layer, recurse=False, writeback=False):
                if rank == 0:
                    stage["layers"][str(index)] = {
                        key: value.detach().cpu().clone()
                        for key, value in layer.module.state_dict().items()
                    }
        with FSDP.summon_full_params(model, recurse=False, writeback=False):
            if rank == 0:
                stage["h3_proj_out"] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.module.h3.proj_out.state_dict().items()
                }
                if args.shared_backbone:
                    stage["action_adapters"] = {
                        key: value.detach().cpu().clone()
                        for key, value in model.module.action_adapters.state_dict().items()
                    }
                else:
                    stage["action_output"] = {
                        key: value.detach().cpu().clone()
                        for key, value in model.module.action_expert.output.state_dict().items()
                    }
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
            torch.save(stage, temporary)
            os.replace(temporary, path)
    iterations = (
        math.ceil(
            (min(len(rows), args.eval_limit) if args.eval_limit else len(rows))
            / world_size
        )
        if args.eval_only and args.eval_all and rows
        else args.steps
    )
    evaluated_samples = 0
    loss_sums = torch.zeros(3, device=device, dtype=torch.float64)
    for step in range(1, iterations + 1):
        if args.data_root is not None and (
            args.rotate_windows or args.eval_all or step > 1
        ):
            row_index = (
                args.sample_offset * world_size
                + (step - 1) * world_size
                + rank
            )
            eval_size = (
                min(len(rows), args.eval_limit) if args.eval_limit else len(rows)
            )
            has_sample = row_index < eval_size if args.eval_all else True
            batch = prepare_real_batch(
                args=args,
                row=rows[row_index % len(rows)],
                row_index=row_index % len(rows),
                device=device,
                generator=generator,
                model_root=model.module,
                stats=stats,
                quantile_stats=quantile_stats,
                patchify_video_latents=patchify_video_latents,
                layout_builder=MiniMaxH3PrepareLayoutStep,
                align_chunk_ids=align_h3_action_chunk_ids,
                action_time_builder=h3_action_temporal_positions,
                deterministic_noise=args.eval_only,
            )
        else:
            has_sample = True
        if args.data_root is not None:
            sample_id = batch["sample_id"]
            video_tokens = batch["video_tokens"]
            noisy_video = batch["noisy_video"]
            clean_video = batch["clean_video"]
            clean_video_input = batch["clean_video_input"]
            video_position_ids = batch["video_position_ids"]
            video_chunks = batch["video_chunks"]
            noisy_video_timesteps = batch["noisy_video_timesteps"]
            clean_video_timesteps = batch["clean_video_timesteps"]
            noisy_video_timestep_indices = batch[
                "noisy_video_timestep_indices"
            ]
            clean_video_timestep_indices = batch[
                "clean_video_timestep_indices"
            ]
            noisy_action = batch["noisy_action"]
            clean_action = batch["clean_action"]
            action_position_ids = batch["action_position_ids"]
            action_chunks = batch["action_chunks"]
            action_time = batch["action_time"]
            action_timestep_indices = batch["action_timestep_indices"]
            action_loss_mask = batch["action_loss_mask"]
            observed_action_mask = batch["observed_action_mask"]
            context = batch["context"]
            context_position_ids = batch["context_position_ids"]
            state = batch["state"]
            context_mask = batch["context_mask"]
            video_target = batch["video_target"]
            action_target = batch["action_target"]
            future = batch["future"]
            if args.mask_clean_future:
                # Cold-start deployment proxy: only the observed keyframe may
                # enter the clean video stream and no ground-truth action may
                # enter the clean action stream. Targets stay unchanged.
                clean_video = clean_video.clone()
                clean_video[:, future] = 0.0
                clean_video_input = clean_video
                clean_action = clean_action.clone()
                clean_action[:, action_loss_mask] = 0.0
        else:
            action_time = sigma
            action_loss_mask = torch.ones(
                clean_action.shape[1], device=device, dtype=torch.bool
            )
            observed_action_mask = ~action_loss_mask
        optimizer.zero_grad(set_to_none=True)
        forward_arguments = dict(
            noisy_video_rows=noisy_video,
            clean_video_rows=clean_video_input,
            video_position_ids=video_position_ids,
            video_chunk_ids=video_chunks,
            noisy_video_timestep=noisy_video_timesteps,
            clean_video_timestep=clean_video_timesteps,
            noisy_video_timestep_indices=noisy_video_timestep_indices,
            clean_video_timestep_indices=clean_video_timestep_indices,
            noisy_actions=noisy_action,
            clean_actions=clean_action,
            action_chunk_ids=action_chunks,
            noisy_action_timestep=(
                action_time
                if args.shared_backbone
                else (1.0 - action_time) * 1000.0
            ),
            context=context,
            context_position_ids=context_position_ids,
            state=state,
            context_mask=context_mask,
        )
        if args.shared_backbone:
            forward_arguments.update(
                action_position_ids=action_position_ids,
                clean_action_timestep=torch.ones_like(sigma),
                noisy_action_timestep_indices=action_timestep_indices,
                clean_action_timestep_indices=torch.zeros_like(
                    action_timestep_indices
                ),
            )
        split_backward = False
        if args.detached_generated_video_conditioning and not args.eval_only:
            if not args.shared_backbone:
                raise ValueError(
                    "generated-video conditioning requires shared-backbone"
                )
            first_output = model(**forward_arguments)
            video_loss = F.mse_loss(
                first_output.video_velocity_rows[:, future].float(),
                video_target[:, future].float(),
            )
            # x_t = x_0 + sigma * (noise - x_0), while this H3 branch predicts
            # clean-minus-noise velocity, hence x_0 = x_t + sigma * velocity.
            # The first observed keyframe is committed exactly, matching the
            # chunk-causal sampler.
            video_time_per_token = noisy_video_timesteps.index_select(
                0, noisy_video_timestep_indices
            )
            generated_clean_video = video_clean_from_velocity(
                noisy_video,
                video_time_per_token,
                first_output.video_velocity_rows,
            ).detach()
            generated_clean_video[:, ~future] = clean_video[:, ~future]
            video_loss.backward()
            del first_output
            second_output = model(
                **{
                    **forward_arguments,
                    "clean_video_rows": generated_clean_video,
                    "clean_video_timestep": torch.ones(1, device=device),
                    "clean_video_timestep_indices": torch.zeros_like(
                        clean_video_timestep_indices
                    ),
                }
            )
            action_loss = F.mse_loss(
                second_output.action_velocity[:, action_loss_mask].float(),
                action_target[:, action_loss_mask].float(),
            )
            action_loss.backward()
            split_backward = True
        elif args.sample_eval:
            video_schedule = build_h3dream_inference_schedule(
                args.video_sample_steps or args.sample_steps,
                device=device,
                video_shift=12.0,
                action_shift=0.05,
            )
            action_schedule = build_h3dream_inference_schedule(
                args.action_sample_steps or args.sample_steps,
                device=device,
                video_shift=12.0,
                action_shift=args.action_infer_shift,
            )

            def predict_velocity(
                sampled_video,
                clean_video_history,
                sampled_actions,
                clean_action_history,
                video_time,
                action_sigma,
                clean_video_valid,
                clean_action_valid,
            ):
                sampled_video_times = torch.stack(
                    (
                        torch.tensor(0.9, device=device),
                        video_time.to(device),
                    )
                )
                sampled_output = model(
                    **{
                        **forward_arguments,
                        "noisy_video_rows": sampled_video,
                        "clean_video_rows": clean_video_history,
                        "noisy_video_timestep": sampled_video_times,
                        "clean_video_timestep": torch.ones(1, device=device),
                        "noisy_actions": sampled_actions,
                        "clean_actions": clean_action_history,
                        "noisy_action_timestep": (1.0 - action_sigma).reshape(1),
                        "clean_action_timestep": torch.ones(1, device=device),
                        "clean_video_valid": clean_video_valid,
                        "clean_action_valid": clean_action_valid,
                    }
                )
                return (
                    sampled_output.video_velocity_rows.float(),
                    sampled_output.action_velocity.float(),
                )

            sampled = sample_h3_lingbot_chunk_causal(
                predict_velocity,
                initial_video_rows=batch["initial_video"],
                observed_video_mask=~future,
                video_chunk_ids=video_chunks,
                initial_actions=batch["initial_action"],
                action_chunk_ids=action_chunks,
                video_schedule=video_schedule,
                action_schedule=action_schedule,
                observed_action_mask=observed_action_mask,
            )
            video_loss = F.mse_loss(
                sampled.video_rows[:, future].float(),
                clean_video[:, future].float(),
            )
            action_loss = F.mse_loss(
                sampled.actions[:, action_loss_mask].float(),
                clean_action[:, action_loss_mask].float(),
            )
        else:
            output = model(**forward_arguments)
            if args.flow_match_loss_weighting and not args.eval_only:
                video_loss, action_loss = weighted_video_action_losses(
                    video_prediction=output.video_velocity_rows,
                    video_target=video_target,
                    future=future,
                    noisy_video_timesteps=noisy_video_timesteps,
                    noisy_video_timestep_indices=noisy_video_timestep_indices,
                    action_prediction=output.action_velocity,
                    action_target=action_target,
                    action_time=action_time,
                    action_timestep_indices=action_timestep_indices,
                    action_loss_mask=action_loss_mask,
                    action_shift=args.action_train_shift,
                )
            else:
                video_loss = F.mse_loss(
                    output.video_velocity_rows[:, future].float(),
                    video_target[:, future].float(),
                )
                action_loss = F.mse_loss(
                    output.action_velocity[:, action_loss_mask].float(),
                    action_target[:, action_loss_mask].float(),
                )
        loss = video_loss + action_loss
        valid_weight = float(has_sample)
        loss_sums += torch.tensor(
            [float(loss.detach()), float(video_loss.detach()), float(action_loss.detach())],
            device=device,
            dtype=torch.float64,
        ) * valid_weight
        evaluated_samples += int(has_sample)
        if not args.eval_only and not split_backward:
            loss.backward()
        named = list(model.named_parameters())
        h3_gradient = 0.0 if args.eval_only else global_grad_norm(
            named, ".h3_block.", device
        )
        action_gradient = 0.0 if args.eval_only else global_grad_norm(
            named,
            ".action_adapters." if args.shared_backbone else ".action_block.",
            device,
        )
        required_gradients = (action_gradient,)
        if not args.freeze_shared_blocks:
            required_gradients = (h3_gradient, action_gradient)
        if not math.isfinite(float(loss.detach())) or (
            not args.eval_only
            and not all(
            math.isfinite(value) and value > 0
            for value in required_gradients
            )
        ):
            raise RuntimeError("full FSDP smoke produced non-finite/zero signal")
        # Clip the two experts independently. A large freshly interpolated
        # ActionDiT norm must not suppress the pretrained H3 update (or the
        # reverse), and FSDP owns the global norm calculation for its shards.
        all_with_grad = [
            parameter for parameter in model.parameters() if parameter.grad is not None
        ]
        expert_clip_norms = {}
        expert_markers = (
            {
                "h3": (".h3_block.", ".h3.proj_out."),
                "action": (".action_adapters.",),
            }
            if args.shared_backbone
            else {
                "h3": (".h3_block.", ".h3.proj_out."),
                "action": (".action_block.", ".action_expert.output."),
            }
        )
        for expert, markers in expert_markers.items():
            active_ids = {
                id(parameter)
                for name, parameter in named
                if any(marker in name for marker in markers)
                and parameter.grad is not None
            }
            hidden = [
                (parameter, parameter.grad)
                for parameter in all_with_grad
                if id(parameter) not in active_ids
            ]
            for parameter, _ in hidden:
                parameter.grad = None
            expert_clip_norms[expert] = (
                0.0 if args.eval_only else float(model.clip_grad_norm_(1.0))
            )
            for parameter, gradient in hidden:
                parameter.grad = gradient
        if not args.eval_only:
            optimizer.step()
            scheduler.step()
        item = {
            "step": step,
            "loss": float(loss.detach()),
            "video_loss": float(video_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "h3_gradient_norm": h3_gradient,
            "action_gradient_norm": action_gradient,
            "expert_clip_norms": expert_clip_norms,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "sample": sample_id,
        }
        history.append(item)
        if rank == 0:
            print(json.dumps(item), flush=True)
        if is_checkpoint_milestone(
            step,
            base_completed_steps=args.base_completed_steps,
            checkpoint_every=args.checkpoint_every,
            total_steps=args.steps,
        ):
            cumulative_step = args.base_completed_steps + step
            milestone = args.save_stage.with_name(
                f"{args.save_stage.stem}_step{cumulative_step:06d}"
                f"{args.save_stage.suffix}"
            )
            save_training_stage(milestone, cumulative_step)

    dist.all_reduce(loss_sums)
    evaluated_tensor = torch.tensor(evaluated_samples, device=device, dtype=torch.long)
    dist.all_reduce(evaluated_tensor)
    mean_losses = (
        (loss_sums / evaluated_tensor.clamp_min(1)).tolist()
        if int(evaluated_tensor) > 0
        else [float("nan")] * 3
    )

    if args.save_stage is not None and not args.eval_only:
        save_training_stage(
            args.save_stage,
            args.base_completed_steps + args.steps,
        )

    if rank == 0:
        report = {
            "event": "h3_lingbot_four_stream_full_fsdp_smoke",
            "world_size": world_size,
            "steps": args.steps,
            "layers": 50,
            "last_trainable_layers": args.last_trainable_layers,
            "video_tokens": video_tokens,
            "sample": sample_id,
            "real_data": args.data_root is not None,
            "eval_only": args.eval_only,
            "loaded_stage": None if args.load_stage is None else str(args.load_stage),
            "saved_stage": None if args.save_stage is None else str(args.save_stage),
            "action_horizon": args.action_horizon,
            "action_normalization": args.action_normalization,
            "action_stats_json": (
                None
                if args.action_stats_json is None
                else str(args.action_stats_json.resolve())
            ),
            "per_chunk_action_timesteps": args.per_chunk_action_timesteps,
            "noisy_clean_video_prob": args.noisy_clean_video_prob,
            "detached_generated_video_conditioning": (
                args.detached_generated_video_conditioning
            ),
            "h3_physical_time_alignment": args.h3_physical_time_alignment,
            "flow_match_loss_weighting": args.flow_match_loss_weighting,
            "action_train_shift": args.action_train_shift,
            "action_infer_shift": args.action_infer_shift,
            "upstream_initial_action_anchor": args.upstream_initial_action_anchor,
            "executed_action_history_steps": args.executed_action_history_steps,
            "weight_decay": args.weight_decay,
            "base_completed_steps": args.base_completed_steps,
            "completed_steps": args.base_completed_steps + args.steps,
            "checkpoint_every": args.checkpoint_every,
            "history": history,
            "evaluated_samples": int(evaluated_tensor),
            "mean_loss": mean_losses[0],
            "mean_video_loss": mean_losses[1],
            "mean_action_loss": mean_losses[2],
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "load_seconds": load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "shared_backbone": args.shared_backbone,
            "freeze_shared_blocks": args.freeze_shared_blocks,
            "mask_clean_future": args.mask_clean_future,
            "sample_eval": args.sample_eval,
            "sample_steps": args.sample_steps,
            "video_sample_steps": args.video_sample_steps or args.sample_steps,
            "action_sample_steps": args.action_sample_steps or args.sample_steps,
            "eval_limit": args.eval_limit,
            "initialization": initialization,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
