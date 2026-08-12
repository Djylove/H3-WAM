#!/usr/bin/env python3
"""Adapt MiniMax H3 to robot videos without backpropagating an action loss.

This stage follows the video-first WAM recipe: train only H3 LoRA branches on
future-video flow matching, while an optional feature anchor keeps the
first-frame representation close to the frozen base H3 used by the successful
action policy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--adaptation-mode",
        choices=("lora", "partial"),
        default="lora",
        help="Train injected LoRA weights or directly unfreeze the final H3 blocks.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-last-blocks", type=int, default=10)
    parser.add_argument(
        "--partial-last-blocks",
        type=int,
        default=4,
        help="Number of final BF16 H3 blocks to unfreeze in partial mode.",
    )
    parser.add_argument(
        "--include-mlp-lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--feature-anchor-weight",
        type=float,
        default=0.05,
        help="Cosine-distance weight for the final first-frame H3 tokens.",
    )
    parser.add_argument("--anchor-layer", type=int, default=49)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-windows", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_by_episode(
    items: list[dict], val_episodes_per_task: int
) -> tuple[list[dict], list[dict]]:
    episodes_by_task: dict[int, set[int]] = defaultdict(set)
    for item in items:
        episodes_by_task[int(item["task_group"])].add(int(item["episode"]))
    validation_episodes: set[int] = set()
    for episodes in episodes_by_task.values():
        ordered = sorted(episodes)
        if len(ordered) <= val_episodes_per_task:
            raise ValueError("not enough episodes for a disjoint validation split")
        validation_episodes.update(ordered[-val_episodes_per_task:])
    train = [item for item in items if int(item["episode"]) not in validation_episodes]
    validation = [item for item in items if int(item["episode"]) in validation_episodes]
    return train, validation


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.validation_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("steps and checkpoint intervals must be positive")
    if args.lora_rank <= 0 or args.lora_last_blocks <= 0:
        raise ValueError("LoRA rank and selected block count must be positive")
    if args.partial_last_blocks <= 0:
        raise ValueError("partial-last-blocks must be positive")
    if args.feature_anchor_weight < 0:
        raise ValueError("feature-anchor-weight must be non-negative")
    if args.adaptation_mode == "partial" and args.feature_anchor_weight > 0:
        raise ValueError(
            "partial mode requires --feature-anchor-weight 0 because a frozen "
            "20B reference model cannot be co-resident"
        )

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(args.comfy_root.resolve()))

    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3ActionFlowScheduler,
        H3BlockFeatureCapture,
        enable_comfy_h3_autograd,
        h3_lora_disabled,
        h3_lora_parameters,
        h3_lora_state_dict,
        inject_h3_attention_lora,
        make_first_frame_payload,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_management.in_training = True
    enable_comfy_h3_autograd(checkpoint_blocks=True)
    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required, got {device}")

    items = read_manifest(args.manifest.resolve())
    train_items, validation_items = split_by_episode(
        items, args.val_episodes_per_task
    )
    if len(validation_items) > args.validation_windows:
        indices = torch.linspace(
            0, len(validation_items) - 1, args.validation_windows
        ).round().long()
        validation_items = [validation_items[int(index)] for index in indices]

    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    h3_model = patcher.model.diffusion_model
    h3_model.requires_grad_(False).eval()
    lora_alpha = float(args.lora_rank if args.lora_alpha is None else args.lora_alpha)
    lora_report = None
    partial_parameter_names: list[str] = []
    if args.adaptation_mode == "lora":
        lora_report = inject_h3_attention_lora(
            h3_model,
            rank=args.lora_rank,
            alpha=lora_alpha,
            last_n_blocks=args.lora_last_blocks,
            include_mlp=args.include_mlp_lora,
        )
        trainable_parameters = h3_lora_parameters(h3_model)
    else:
        if args.partial_last_blocks > len(h3_model.blocks):
            raise ValueError(
                f"partial-last-blocks={args.partial_last_blocks} exceeds "
                f"H3 depth {len(h3_model.blocks)}"
            )
        first_block = len(h3_model.blocks) - args.partial_last_blocks
        for name, parameter in h3_model.named_parameters():
            if not name.startswith("blocks."):
                continue
            parts = name.split(".")
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            if int(parts[1]) < first_block:
                continue
            if not parameter.is_floating_point():
                raise TypeError(
                    "partial H3 unfreeze requires an unquantized BF16/FP16 "
                    f"checkpoint; {name} has dtype {parameter.dtype}. The local "
                    "INT8 ConvRot checkpoint is valid only for LoRA mode."
                )
            parameter.requires_grad_(True)
            partial_parameter_names.append(name)
        if not partial_parameter_names:
            raise RuntimeError("partial mode selected no trainable H3 parameters")
        named_parameters = dict(h3_model.named_parameters())
        trainable_parameters = [
            named_parameters[name] for name in partial_parameter_names
        ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = H3ActionFlowScheduler(
        video_shift=float(h3_model.sigma_shift_video),
        action_shift=float(h3_model.sigma_shift_audio),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def load_item(item: dict) -> tuple[dict, dict]:
        window = torch.load(
            args.cache_root / "windows" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        conditioning = torch.load(
            args.cache_root / "refined_contexts" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        return window, conditioning

    def forward_item(
        item: dict, *, validation_seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, conditioning = load_item(item)
        video = window["video_latents"].to(device=device, dtype=torch.bfloat16)
        first_frame = window["first_frame_latents"].to(
            device=device, dtype=torch.bfloat16
        )
        context = conditioning["context"].to(device=device, dtype=torch.bfloat16)
        token_tags = conditioning["token_tags"].to(device)

        local_generator = generator
        if validation_seed is not None:
            local_generator = torch.Generator(device=device).manual_seed(validation_seed)
        base_sigma = torch.rand(1, generator=local_generator, device=device)
        video_sigma = scheduler.shift(base_sigma, scheduler.video_shift)
        noise = torch.randn(
            video.shape,
            generator=local_generator,
            device=device,
            dtype=video.dtype,
        )
        noisy_video = scheduler.add_video_noise(video, noise, video_sigma)
        target = scheduler.video_training_target(video, noise)
        timestep = scheduler.timestep(video_sigma)
        audio = torch.zeros((1, 32, 2, 1), device=device, dtype=torch.float32)
        payload = make_first_frame_payload(
            first_frame,
            frame_count=int(window["h3_frame_count"]),
            seed=args.seed,
        )
        payload["text_token_tags"] = token_tags

        frame_rows = int(
            first_frame.shape[2]
            * (first_frame.shape[3] // 2)
            * (first_frame.shape[4] // 2)
        )
        token_start = int(context.shape[1])
        baseline_feature = None
        if args.feature_anchor_weight > 0:
            baseline_capture = H3BlockFeatureCapture(
                (args.anchor_layer,),
                token_start,
                token_start + frame_rows,
                detach=True,
            )
            with torch.no_grad(), h3_lora_disabled(h3_model):
                h3_model(
                    [noisy_video, audio],
                    timestep,
                    context,
                    transformer_options=baseline_capture.transformer_options(),
                    minimax_payload=payload,
                )
            baseline_feature = baseline_capture.stacked()

        adapted_capture = H3BlockFeatureCapture(
            (args.anchor_layer,),
            token_start,
            token_start + frame_rows,
            detach=False,
        )
        output = h3_model(
            [noisy_video, audio],
            timestep,
            context,
            transformer_options=adapted_capture.transformer_options(),
            minimax_payload=payload,
        )
        if not isinstance(output, (list, tuple)) or len(output) != 2:
            raise TypeError("H3 must return [video_velocity, audio_velocity]")
        video_velocity = output[0]
        video_loss = (video_velocity.float() - target.float()).square().mean()

        anchor_loss = video_loss.new_zeros(())
        if baseline_feature is not None:
            adapted_feature = adapted_capture.stacked().float()
            anchor_loss = (
                1.0
                - F.cosine_similarity(
                    adapted_feature,
                    baseline_feature.float(),
                    dim=-1,
                    eps=1e-6,
                )
            ).mean().clamp_min(0.0)
        total = video_loss + args.feature_anchor_weight * anchor_loss
        return total, video_loss, anchor_loss

    @torch.no_grad()
    def validation_metrics() -> dict[str, float]:
        totals, videos, anchors = [], [], []
        for index, item in enumerate(validation_items):
            total, video, anchor = forward_item(
                item, validation_seed=args.seed + index + 1
            )
            totals.append(float(total.item()))
            videos.append(float(video.item()))
            anchors.append(float(anchor.item()))
        return {
            "total": sum(totals) / len(totals),
            "video": sum(videos) / len(videos),
            "anchor": sum(anchors) / len(anchors),
        }

    def checkpoint(step: int, metrics: dict[str, float]) -> dict:
        artifact = {
            "policy_type": (
                "h3_video_lora"
                if args.adaptation_mode == "lora"
                else "h3_video_partial"
            ),
            "adaptation_mode": args.adaptation_mode,
            "base_h3_checkpoint": str(args.h3_checkpoint.resolve()),
            "feature_anchor_weight": args.feature_anchor_weight,
            "anchor_layer": args.anchor_layer,
            "training_objective": "future_video_flow",
            "training_tasks": sorted({str(item["task"]) for item in items}),
            "manifest": str(args.manifest.resolve()),
            "cache_root": str(args.cache_root.resolve()),
            "step": step,
            "validation_total": metrics["total"],
            "validation_video": metrics["video"],
            "validation_anchor": metrics["anchor"],
        }
        if args.adaptation_mode == "lora":
            assert lora_report is not None
            artifact.update(
                {
                    "h3_lora": h3_lora_state_dict(h3_model),
                    "h3_lora_rank": args.lora_rank,
                    "h3_lora_alpha": lora_alpha,
                    "h3_lora_last_blocks": args.lora_last_blocks,
                    "h3_lora_include_mlp": args.include_mlp_lora,
                    "h3_lora_trainable_parameters": lora_report.parameters,
                }
            )
        else:
            named_parameters = dict(h3_model.named_parameters())
            artifact.update(
                {
                    "h3_partial_last_blocks": args.partial_last_blocks,
                    "h3_partial_trainable_parameters": sum(
                        named_parameters[name].numel()
                        for name in partial_parameter_names
                    ),
                    "h3_partial_state": {
                        name: named_parameters[name].detach().cpu()
                        for name in partial_parameter_names
                    },
                }
            )
        return artifact

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = validation_metrics()
    best = baseline["total"]
    if args.adaptation_mode == "lora":
        step_zero = args.output.with_name(
            args.output.stem + "_step000000" + args.output.suffix
        )
        torch.save(checkpoint(0, baseline), step_zero)
    print(
        json.dumps(
            {
                "step": 0,
                "validation": baseline,
                "adaptation_mode": args.adaptation_mode,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "trainable_modules": (
                    lora_report.modules
                    if lora_report is not None
                    else args.partial_last_blocks
                ),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        ),
        flush=True,
    )

    started = time.perf_counter()
    last_metrics = baseline
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, video_loss, anchor_loss = forward_item(random.choice(train_items))
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.gradient_clip
        )
        optimizer.step()

        if step % args.validation_every == 0 or step == args.steps:
            last_metrics = validation_metrics()
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_total": float(total.detach().item()),
                        "train_video": float(video_loss.detach().item()),
                        "train_anchor": float(anchor_loss.detach().item()),
                        "validation": last_metrics,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": elapsed,
                        "seconds_per_step": elapsed / step,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / 2**30,
                    }
                ),
                flush=True,
            )
            if last_metrics["total"] < best:
                best = last_metrics["total"]
                best_path = args.output.with_name(
                    args.output.stem + "_best" + args.output.suffix
                )
                torch.save(checkpoint(step, last_metrics), best_path)

        if step % args.checkpoint_every == 0 and step != args.steps:
            numbered = args.output.with_name(
                args.output.stem + f"_step{step:06d}" + args.output.suffix
            )
            torch.save(checkpoint(step, last_metrics), numbered)

    torch.save(checkpoint(args.steps, last_metrics), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_windows": len(train_items),
                "validation_windows": len(validation_items),
                "baseline_validation": baseline,
                "best_validation_total": best,
                "total_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
