#!/usr/bin/env python3
"""Evaluate 2–4 step H3-WAM action sampling on held-out LIBERO episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("adapter_checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--model-evaluations", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-last-blocks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-ensemble-size", type=int, default=1)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--conditioning-ablation",
        choices=("none", "shuffled_context", "shuffled_first_frame", "shuffled_state"),
        default="none",
    )
    parser.add_argument("--task-groups", type=int, nargs="+")
    return parser.parse_args()


def load_training_helpers():
    script = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def choose_shuffled_item(item: dict, source_pool: list[dict]) -> dict:
    """Prefer another task, but support ablations on single-task manifests."""
    different_task = [
        candidate
        for candidate in source_pool
        if int(candidate["task_group"]) != int(item["task_group"])
    ]
    candidates = different_task or [
        candidate for candidate in source_pool if candidate["id"] != item["id"]
    ]
    if not candidates:
        raise ValueError(
            "conditioning ablation requires at least two source windows"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    if args.sample_ensemble_size <= 0:
        raise ValueError("sample-ensemble-size must be positive")
    sys.path.insert(0, str(args.comfy_root.resolve()))
    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3ActionAdapter,
        H3ActionBridge,
        H3ActionFlowScheduler,
        inject_h3_attention_lora,
        make_first_frame_payload,
        prepare_h3wam_flow_batch,
        sample_h3wam_actions,
    )

    helpers = load_training_helpers()
    items = helpers.read_manifest(args.manifest.resolve())
    if args.task_groups is not None:
        selected_groups = set(args.task_groups)
        items = [
            item for item in items if int(item["task_group"]) in selected_groups
        ]
    train_items, held_out_items = helpers.split_by_episode(
        items, args.val_episodes_per_task
    )
    validation_items = (
        train_items if args.split == "train" else held_out_items
    )[: args.max_windows]
    source_pool = train_items if args.split == "train" else held_out_items

    def shuffled_item(item: dict) -> dict:
        return choose_shuffled_item(item, source_pool)
    checkpoint = torch.load(
        args.adapter_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    stats = checkpoint["normalization"]

    model_management.in_training = False
    device = model_management.get_torch_device()
    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    h3_model = patcher.model.diffusion_model
    adapter = H3ActionAdapter(
        action_dim=int(checkpoint["action_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        direct_conditioning=bool(
            checkpoint.get("direct_action_conditioning", False)
        ),
    ).to(device=device, dtype=torch.float32)
    bridge = H3ActionBridge(h3_model, adapter, freeze_h3=True)
    if args.lora_rank > 0:
        inject_h3_attention_lora(
            h3_model,
            rank=args.lora_rank,
            last_n_blocks=args.lora_last_blocks,
        )
        from fastwam.models.h3wam import load_h3_lora_state_dict

        load_h3_lora_state_dict(h3_model, checkpoint["h3_lora"])
    adapter.load_state_dict(checkpoint["adapter"])
    bridge.eval()
    scheduler = H3ActionFlowScheduler(
        video_shift=float(h3_model.sigma_shift_video),
        action_shift=float(h3_model.sigma_shift_audio),
    )

    flow_action_errors = []
    flow_video_errors = []
    with torch.inference_mode():
        for index, item in enumerate(validation_items):
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
            ablation_item = (
                item
                if args.conditioning_ablation == "none"
                else shuffled_item(item)
            )
            ablation_window = None
            if args.conditioning_ablation in ("shuffled_first_frame", "shuffled_state"):
                ablation_window = torch.load(
                    args.cache_root / "windows" / f"{ablation_item['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            if args.conditioning_ablation == "shuffled_context":
                conditioning = torch.load(
                    args.cache_root
                    / "refined_contexts"
                    / f"{ablation_item['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            target = helpers.minmax_normalize(
                window["actions"].float(),
                stats["action_min"],
                stats["action_max"],
            ).unsqueeze(0).to(device)
            state = helpers.minmax_normalize(
                (
                    ablation_window["state"].float()
                    if args.conditioning_ablation == "shuffled_state"
                    else window["state"].float()
                ),
                stats["state_min"],
                stats["state_max"],
            ).unsqueeze(0).to(device)
            video = window["video_latents"].to(device, dtype=torch.bfloat16)
            context = conditioning["context"].to(device, dtype=torch.bfloat16)
            first_frame = (
                ablation_window["first_frame_latents"]
                if args.conditioning_ablation == "shuffled_first_frame"
                else window["first_frame_latents"]
            ).to(
                device, dtype=torch.bfloat16
            )
            generator = torch.Generator(device=device).manual_seed(args.seed + index + 1)
            base = torch.rand(1, generator=generator, device=device)
            video_sigma = scheduler.shift(base, scheduler.video_shift)
            flow = prepare_h3wam_flow_batch(
                video_latents=video,
                actions=target,
                scheduler=scheduler,
                video_sigma=video_sigma,
                video_noise=torch.randn(
                    video.shape,
                    generator=generator,
                    device=device,
                    dtype=video.dtype,
                ),
                action_noise=torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=target.dtype,
                ),
            )
            payload = make_first_frame_payload(
                first_frame,
                frame_count=int(window["h3_frame_count"]),
                seed=args.seed,
            )
            payload["text_token_tags"] = conditioning["token_tags"]
            output = bridge(
                video_latents=flow.noisy_video_latents,
                noisy_actions=flow.noisy_actions,
                timestep=flow.timestep,
                context=context,
                state=state,
                minimax_payload=payload,
            )
            flow_action_errors.append(
                float(
                    (output.action_velocity.float() - flow.action_target.float())
                    .square()
                    .mean()
                    .item()
                )
            )
            flow_video_errors.append(
                float(
                    (output.video_velocity.float() - flow.video_target.float())
                    .square()
                    .mean()
                    .item()
                )
            )

    results = []
    for evaluations in args.model_evaluations:
        squared_errors = []
        absolute_errors = []
        timestep_squared_errors = []
        dimension_squared_errors = []
        durations = []
        torch.cuda.reset_peak_memory_stats(device)
        for index, item in enumerate(validation_items):
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
            ablation_item = (
                item
                if args.conditioning_ablation == "none"
                else shuffled_item(item)
            )
            ablation_window = None
            if args.conditioning_ablation in ("shuffled_first_frame", "shuffled_state"):
                ablation_window = torch.load(
                    args.cache_root / "windows" / f"{ablation_item['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            if args.conditioning_ablation == "shuffled_context":
                conditioning = torch.load(
                    args.cache_root
                    / "refined_contexts"
                    / f"{ablation_item['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            target = helpers.minmax_normalize(
                window["actions"].float(),
                stats["action_min"],
                stats["action_max"],
            ).unsqueeze(0).to(device)
            state = helpers.minmax_normalize(
                (
                    ablation_window["state"].float()
                    if args.conditioning_ablation == "shuffled_state"
                    else window["state"].float()
                ),
                stats["state_min"],
                stats["state_max"],
            ).unsqueeze(0).to(device)
            context = conditioning["context"].to(device, dtype=torch.bfloat16)
            first_frame = (
                ablation_window["first_frame_latents"]
                if args.conditioning_ablation == "shuffled_first_frame"
                else window["first_frame_latents"]
            ).to(device, dtype=torch.bfloat16)
            payload = make_first_frame_payload(
                first_frame,
                frame_count=int(window["h3_frame_count"]),
                seed=args.seed + index,
            )
            payload["text_token_tags"] = conditioning["token_tags"]
            started = time.perf_counter()
            action_samples = []
            for sample_index in range(args.sample_ensemble_size):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + index + sample_index
                )
                sample = sample_h3wam_actions(
                    bridge,
                    context=context,
                    state=state,
                    scheduler=scheduler,
                    action_shape=tuple(target.shape),
                    video_shape=tuple(window["video_latents"].shape),
                    model_evaluations=evaluations,
                    minimax_payload=payload,
                    generator=generator,
                )
                action_samples.append(sample.actions)
            sampled_actions = torch.stack(action_samples).mean(dim=0)
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
            error = sampled_actions.float() - target.float()
            squared_errors.append(float(error.square().mean().item()))
            absolute_errors.append(float(error.abs().mean().item()))
            timestep_squared_errors.append(error.square().mean(dim=(0, 2)).cpu())
            dimension_squared_errors.append(error.square().mean(dim=(0, 1)).cpu())
        steady_durations = durations[1:] if len(durations) > 1 else durations
        per_timestep = torch.stack(timestep_squared_errors).mean(dim=0)
        per_dimension = torch.stack(dimension_squared_errors).mean(dim=0)
        results.append(
            {
                "model_evaluations": evaluations,
                "seed": args.seed,
                "sample_ensemble_size": args.sample_ensemble_size,
                "windows": len(validation_items),
                "normalized_action_mse": sum(squared_errors) / len(squared_errors),
                "normalized_action_mae": sum(absolute_errors) / len(absolute_errors),
                "first_10_action_mse": float(per_timestep[:10].mean().item()),
                "remaining_action_mse": float(per_timestep[10:].mean().item()),
                "per_action_dimension_mse": per_dimension.tolist(),
                "per_timestep_mse": per_timestep.tolist(),
                "cold_start_latency_seconds": durations[0],
                "mean_steady_latency_seconds": sum(steady_durations) / len(steady_durations),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
    print(
        json.dumps(
            {
                "flow_validation": {
                    "split": args.split,
                    "conditioning_ablation": args.conditioning_ablation,
                    "windows": len(validation_items),
                    "action_velocity_mse": sum(flow_action_errors)
                    / len(flow_action_errors),
                    "video_velocity_mse": sum(flow_video_errors)
                    / len(flow_video_errors),
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
