#!/usr/bin/env python3
"""Serve the shared-H3 LingBot port for a bounded LIBERO rollout canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "h3wam"))

from precompute_libero_official_h3 import PIXEL_MEAN, PIXEL_STD  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--actions-per-chunk", type=int, default=4)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--video-sample-steps", type=int, default=0)
    parser.add_argument("--action-sample-steps", type=int, default=0)
    parser.add_argument("--last-trainable-layers", type=int, default=2)
    parser.add_argument("--binarize-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-normalized-actions", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def broadcast_object(value, rank: int):
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def task_contexts(manifest: Path) -> dict[str, str]:
    result = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result.setdefault(str(row["task"]), str(row.get("context_id", row["id"])))
    if not result:
        raise ValueError("manifest contains no task contexts")
    return result


def main() -> None:
    args = parse_args()
    if min(args.port, args.action_horizon, args.actions_per_chunk, args.sample_steps) <= 0:
        raise ValueError("positive server arguments are required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from fastwam.models.h3dreamwam import (
        H3LingBotSharedLayer,
        H3LingBotSharedWAM,
        align_h3_action_chunk_ids,
        build_h3dream_inference_schedule,
        sample_h3_lingbot_chunk_causal,
    )
    from fastwam.models.h3wam import (
        action_denormalization_bounds,
        libero_environment_actions,
        libero_observation_state,
        minmax_normalize,
        preprocess_libero_cameras,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(), subfolder="transformer", dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    h3.requires_grad_(False)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = H3LingBotSharedWAM(
            h3, action_dim=7, state_dim=8, text_dim=5120,
            use_gradient_checkpointing=False, compute_dtype=torch.bfloat16,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    model.requires_grad_(False)
    ignored = [*model.h3.children(), *model.action_adapters.children()]
    ignored = [module for module in ignored if list(module.parameters())]
    for module in ignored:
        module.to(device)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({H3LingBotSharedLayer}),
        ignored_modules=ignored,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16, cast_forward_inputs=True,
        ),
    )
    model.float()
    model.check_is_root()
    payload = torch.load(args.stage.resolve(), map_location="cpu", weights_only=True)
    if payload.get("format") != "h3_lingbot_shared_four_stream_tail_v1":
        raise ValueError("incompatible shared-H3 stage")
    if payload.get("last_trainable_layers") != args.last_trainable_layers:
        raise ValueError("stage layer count mismatch")
    action_normalization = payload.get("action_normalization", "minmax")
    action_quantile_stats = payload.get("action_quantile_stats")
    upstream_initial_action_anchor = payload.get(
        "upstream_initial_action_anchor", False
    )
    for index_text, state in payload["layers"].items():
        layer = model.module.shared_layers[int(index_text)]
        with FSDP.summon_full_params(layer, recurse=False, writeback=True):
            layer.module.load_state_dict(state, strict=True)
    with FSDP.summon_full_params(model, recurse=False, writeback=True):
        model.module.h3.proj_out.load_state_dict(payload["h3_proj_out"], strict=True)
        model.module.action_adapters.load_state_dict(payload["action_adapters"], strict=True)
    del payload
    model.eval()

    stats = torch.load(args.cache_root.resolve() / "stats.pt", map_location="cpu", weights_only=False)
    action_low, action_high = action_denormalization_bounds(
        action_normalization, stats, action_quantile_stats
    )
    normalized_action_clip = 1.5 if action_normalization == "quantile" else 1.0
    language_to_context = task_contexts(args.manifest.resolve())
    context_cache = {}
    patch_size = tuple(model.module.h3.config.patch_size)
    video_schedule = build_h3dream_inference_schedule(
        args.video_sample_steps or args.sample_steps,
        device=device, video_shift=12.0, action_shift=0.05
    )
    action_schedule = build_h3dream_inference_schedule(
        args.action_sample_steps or args.sample_steps,
        device=device, video_shift=12.0, action_shift=0.05
    )
    vae = None
    if rank == 0:
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.model.resolve(), subfolder="vae", torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        vae.eval()
    dist.barrier()
    load_seconds = time.perf_counter() - started

    def load_context(task: str):
        if task not in language_to_context:
            raise KeyError(f"task language absent from manifest: {task!r}")
        context_id = language_to_context[task]
        if context_id not in context_cache:
            item = torch.load(
                args.cache_root.resolve() / "contexts" / f"{context_id}.pt",
                map_location="cpu", weights_only=False,
            )
            context_cache[context_id] = {
                "context": item["context"].to(device=device, dtype=torch.bfloat16),
                "token_tags": item["token_tags"].long(),
            }
        return context_id, context_cache[context_id]

    def encode_observation(request):
        assert rank == 0 and vae is not None
        encode_started = time.perf_counter()
        agent = np.frombuffer(request["agentview_bytes"], dtype=np.uint8).reshape(request["agentview_shape"])
        wrist = np.frombuffer(request["wristview_bytes"], dtype=np.uint8).reshape(request["wristview_shape"])
        pixels = preprocess_libero_cameras(agent, wrist)
        video = pixels.mul(255).round().to(torch.uint8).permute(0, 3, 1, 2).unsqueeze(2).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            first = encode_vae_condition(vae, video, PIXEL_MEAN, PIXEL_STD)
        state = minmax_normalize(
            libero_observation_state({
                "eef_pos": request["eef_pos"], "eef_quat": request["eef_quat"],
                "gripper_qpos": request["gripper_qpos"],
            }),
            stats["state_min"], stats["state_max"],
        ).clamp(-1, 1)
        return (
            first.to(device=device, dtype=torch.float32),
            state.to(device=device, dtype=torch.float32),
            time.perf_counter() - encode_started,
        )

    def policy_forward(task: str, first: torch.Tensor, state: torch.Tensor, seed: int):
        context_id, conditioning = load_context(task)
        _, channels, _, height, width = first.shape
        text_tags = torch.cat((conditioning["token_tags"], torch.ones(1, dtype=torch.long)))
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_tags, num_latent_frames=args.target_latent_frames,
            latent_height=height, latent_width=width, num_audio_latents=10,
            patch_size=patch_size, audio_channels=8, audio_tag=2, video_tag=0,
            keyframe_anchors=("first",),
        )
        position_ids, _, video_indices, _, text_indices, condition_rows, _ = layout
        video_position_ids = position_ids.index_select(0, video_indices).to(device)
        context_position_ids = position_ids.index_select(0, text_indices).to(device)
        first_rows = patchify_video_latents(first, patch_size)[None]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        future_noise = torch.randn(
            (1, channels, args.target_latent_frames, height, width),
            generator=generator, device=device,
        )
        initial_video = torch.cat((first_rows, patchify_video_latents(future_noise, patch_size)[None]), dim=1)
        video_chunks, action_chunks = align_h3_action_chunk_ids(
            video_frame_ids=video_position_ids[:, 0], action_horizon=args.action_horizon,
            actions_per_chunk=args.actions_per_chunk,
        )
        action_positions = torch.stack((
            torch.arange(args.action_horizon, device=device).float() / args.actions_per_chunk,
            torch.full((args.action_horizon,), -1.0, device=device),
            torch.full((args.action_horizon,), -1.0, device=device),
        ), dim=-1)
        observed = torch.zeros(initial_video.shape[1], device=device, dtype=torch.bool)
        observed[:condition_rows] = True
        video_time_indices = torch.ones(initial_video.shape[1], device=device, dtype=torch.long)
        video_time_indices[:condition_rows] = 0
        context_mask = torch.ones(conditioning["context"].shape[:2], device=device, dtype=torch.bool)

        def predict(video, clean_video, actions, clean_actions, video_time, action_sigma, clean_video_valid, clean_action_valid):
            output = model(
                noisy_video_rows=video, clean_video_rows=clean_video,
                video_position_ids=video_position_ids, video_chunk_ids=video_chunks,
                noisy_video_timestep=torch.stack((torch.tensor(0.9, device=device), video_time)),
                clean_video_timestep=torch.ones(1, device=device),
                noisy_video_timestep_indices=video_time_indices,
                clean_video_timestep_indices=torch.zeros_like(video_time_indices),
                noisy_actions=actions, clean_actions=clean_actions,
                action_position_ids=action_positions, action_chunk_ids=action_chunks,
                noisy_action_timestep=(1.0 - action_sigma).reshape(1),
                clean_action_timestep=torch.ones(1, device=device),
                context=conditioning["context"], context_position_ids=context_position_ids,
                state=state.reshape(1, 8), context_mask=context_mask,
                clean_video_valid=clean_video_valid, clean_action_valid=clean_action_valid,
            )
            return output.video_velocity_rows.float(), output.action_velocity.float()

        inference_started = time.perf_counter()
        initial_actions = torch.randn(
            (1, args.action_horizon, 7), generator=generator, device=device
        )
        if upstream_initial_action_anchor:
            initial_actions[:, : args.actions_per_chunk] = 0.0
        sampled = sample_h3_lingbot_chunk_causal(
            predict_velocity=predict, initial_video_rows=initial_video,
            observed_video_mask=observed, video_chunk_ids=video_chunks,
            initial_actions=initial_actions,
            action_chunk_ids=action_chunks,
            video_schedule=video_schedule,
            action_schedule=action_schedule,
            observed_action_mask=(
                torch.arange(args.action_horizon, device=device)
                < args.actions_per_chunk
                if upstream_initial_action_anchor
                else None
            ),
        )
        actions = sampled.actions[0]
        if upstream_initial_action_anchor:
            actions = actions[args.actions_per_chunk :]
        torch.cuda.synchronize(device)
        return actions, context_id, time.perf_counter() - inference_started

    listener = connection = None
    try:
        if rank == 0:
            args.ready_file.resolve().parent.mkdir(parents=True, exist_ok=True)
            listener = Listener(("127.0.0.1", args.port), authkey=b"h3wam-local-rollout")
            args.ready_file.resolve().write_text(json.dumps({
                "ready": True, "policy": "h3_lingbot_shared_fsdp",
                "world_size": world_size, "sample_steps": args.sample_steps,
                "video_sample_steps": args.video_sample_steps or args.sample_steps,
                "action_sample_steps": args.action_sample_steps or args.sample_steps,
                "action_horizon": args.action_horizon, "load_seconds": load_seconds,
                "stage": str(args.stage.resolve()),
                "action_normalization": action_normalization,
                "normalized_action_clipping": args.clip_normalized_actions,
                "normalized_action_clip_bound": normalized_action_clip,
                "upstream_initial_action_anchor": upstream_initial_action_anchor,
            }, indent=2))
            connection = listener.accept()
        while True:
            request = connection.recv() if rank == 0 else None
            command = broadcast_object(request.get("command", "") if rank == 0 else None, rank)
            if command == "close":
                if rank == 0:
                    connection.send({"ok": True})
                break
            if command != "predict":
                if rank == 0:
                    connection.send({"ok": False, "error": f"unknown command {command!r}"})
                continue
            task = broadcast_object(request["task"] if rank == 0 else None, rank)
            seed = int(broadcast_object(request["seed"] if rank == 0 else None, rank))
            if rank == 0:
                first, state, vae_seconds = encode_observation(request)
                shapes = (tuple(first.shape), tuple(state.shape))
            else:
                shapes = None
                vae_seconds = 0.0
            first_shape, state_shape = broadcast_object(shapes, rank)
            if rank != 0:
                first = torch.empty(first_shape, device=device, dtype=torch.float32)
                state = torch.empty(state_shape, device=device, dtype=torch.float32)
            dist.broadcast(first, src=0)
            dist.broadcast(state, src=0)
            normalized, context_id, inference_seconds = policy_forward(task, first, state, seed)
            if rank == 0:
                raw = normalized.float()
                if args.clip_normalized_actions:
                    normalized = normalized.clamp(
                        -normalized_action_clip, normalized_action_clip
                    )
                actions = libero_environment_actions(
                    normalized, action_low, action_high,
                    binarize_gripper=args.binarize_gripper,
                )
                connection.send({"ok": True, "actions": actions.tolist(), "metadata": {
                    "context_id": context_id, "inference_seconds": inference_seconds,
                    "vae_encode_seconds": vae_seconds,
                    "normalized_action_abs_mean": float(raw.abs().mean()),
                    "normalized_action_saturation_fraction": float((raw.abs() >= 1).float().mean()),
                }})
    except Exception:
        if rank == 0 and connection is not None:
            try:
                connection.send({"ok": False, "error": traceback.format_exc()})
            except (BrokenPipeError, EOFError):
                pass
        raise
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        if rank == 0:
            args.ready_file.resolve().unlink(missing_ok=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
