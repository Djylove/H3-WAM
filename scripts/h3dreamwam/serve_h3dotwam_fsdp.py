#!/usr/bin/env python3
"""Serve the H3 Faster-WAM/DoT policy for closed-loop LIBERO rollout."""

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
from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    KEYFRAME_TIMESTEP,
)
from scripts.h3dreamwam.serve_h3dreamwam_fsdp import (  # noqa: E402
    _broadcast_object,
    _task_contexts,
)
from scripts.h3dreamwam.train_h3dotwam_fsdp import (  # noqa: E402
    load_joint_h3_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--action-stage", type=Path, required=True)
    parser.add_argument("--h3-joint-stage", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clip-normalized-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-text-only-context", action="store_true")
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.port,
        args.action_horizon,
        args.target_latent_frames,
        args.sample_steps,
        args.action_median_window,
    ) <= 0:
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
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from fastwam.models.h3dreamwam import (
        H3DoTActionHead,
        H3DoTHubLayer,
        H3DoTKVFusion,
        H3DoTWAM,
        build_h3dream_inference_schedule,
        expand_h3_rgb_flow_projections,
    )
    from fastwam.models.h3wam import (
        build_h3_observation_attention_mask,
        libero_environment_actions,
        libero_observation_state,
        minmax_normalize,
        preprocess_libero_cameras,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    load_started = time.perf_counter()
    model_path = args.model.resolve()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    h3.requires_grad_(False)
    projection = expand_h3_rgb_flow_projections(
        h3,
        flow_input_init_scale=0.0,
        flow_output_init_scale=0.0,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        action_head = H3DoTActionHead(
            action_dim=7,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            head_dim=128,
            num_layers=1,
            frequency_dim=256,
            full_width_rmsnorm=True,
        )
        kv_fusion = H3DoTKVFusion(
            video_layers=50,
            action_layers=1,
            video_num_heads=56,
            video_head_dim=128,
            action_num_heads=24,
            action_head_dim=128,
        )
        model = H3DoTWAM(
            h3,
            action_head,
            kv_fusion,
            state_dim=8,
            text_dim=5120,
            rgb_patch_width=projection.old_patch_width,
            use_gradient_checkpointing=False,
            compute_dtype=torch.bfloat16,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    stage = torch.load(
        args.action_stage.resolve(), map_location="cpu", weights_only=True
    )
    if stage.get("format") != "h3dotwam_stage_v2":
        raise ValueError("DoT stage checkpoint format mismatch")
    model.action_head.load_state_dict(stage["action_head"], strict=True)
    model.kv_fusion.load_state_dict(stage["kv_fusion"], strict=True)
    model.state_embedding.load_state_dict(stage["state_embedding"], strict=True)
    model.requires_grad_(False)
    ignored_modules = [
        module
        for module in model.h3.children()
        if len(list(module.parameters())) > 0
    ]
    for module in ignored_modules:
        module.to(device)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({H3DoTHubLayer}),
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
    if args.h3_joint_stage is not None:
        load_joint_h3_shard(
            stage_dir=args.h3_joint_stage,
            model=model,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    model.eval()
    patch_size = tuple(model.module.h3.config.patch_size)
    stats = torch.load(
        args.cache_root.resolve() / "stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    language_to_context = _task_contexts(args.manifest.resolve())
    context_cache: dict[str, dict[str, torch.Tensor]] = {}
    schedule = build_h3dream_inference_schedule(args.sample_steps, device=device)

    vae = None
    if rank == 0:
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        vae.eval()
    dist.barrier()
    load_seconds = time.perf_counter() - load_started

    def load_context(task: str) -> tuple[str, dict[str, torch.Tensor]]:
        if task not in language_to_context:
            raise KeyError(f"task language is absent from manifest: {task!r}")
        context_id = language_to_context[task]
        if context_id not in context_cache:
            payload = torch.load(
                args.cache_root.resolve() / "contexts" / f"{context_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if args.require_text_only_context:
                if payload.get("text_only") is not True:
                    raise ValueError(f"context {context_id!r} is not text-only")
                if torch.any(payload["token_tags"] != 1):
                    raise ValueError(f"context {context_id!r} has non-text tags")
            context_cache[context_id] = {
                "context": payload["context"].to(device=device, dtype=torch.float32),
                "token_tags": payload["token_tags"].to(
                    device=device, dtype=torch.long
                ),
            }
        return context_id, context_cache[context_id]

    def encode_observation(request: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        assert rank == 0 and vae is not None
        encode_started = time.perf_counter()
        agent = np.frombuffer(request["agentview_bytes"], dtype=np.uint8).reshape(
            request["agentview_shape"]
        )
        wrist = np.frombuffer(request["wristview_bytes"], dtype=np.uint8).reshape(
            request["wristview_shape"]
        )
        pixels = preprocess_libero_cameras(agent, wrist)
        video = (
            pixels.mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
            .to(device)
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            first = encode_vae_condition(vae, video, PIXEL_MEAN, PIXEL_STD)
        state = minmax_normalize(
            libero_observation_state(
                {
                    "eef_pos": request["eef_pos"],
                    "eef_quat": request["eef_quat"],
                    "gripper_qpos": request["gripper_qpos"],
                }
            ),
            stats["state_min"],
            stats["state_max"],
        ).clamp(-1.0, 1.0)
        return (
            first.to(device=device, dtype=torch.float32),
            state.to(device=device, dtype=torch.float32),
            time.perf_counter() - encode_started,
        )

    def policy_forward(
        task: str, first: torch.Tensor, state: torch.Tensor, seed: int
    ) -> tuple[torch.Tensor, str, float]:
        context_id, conditioning = load_context(task)
        context = conditioning["context"]
        text_tags = conditioning["token_tags"]
        _, _, _, latent_height, latent_width = first.shape
        # Faster-WAM inference runs the hub on the observed conditioning
        # frame only. Training target rows are causally masked from these
        # queries, so omitting future video/audio rows preserves the docking
        # representation while avoiding unnecessary world generation.
        num_audio_latents = 0
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.cat(
                (text_tags.cpu(), torch.ones(1, dtype=text_tags.dtype))
            ),
            num_latent_frames=0,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
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
        ) = layout
        position_ids = position_ids.to(device)
        token_tags = token_tags.to(device)
        video_indices = video_indices.to(device)
        audio_indices = audio_indices.to(device)
        text_indices = text_indices.to(device)
        condition_indices = video_indices[:num_condition_video_rows]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        condition_rows = patchify_video_latents(first, patch_size)[None]
        rgb_rows = condition_rows
        video_rows = torch.cat((rgb_rows, torch.zeros_like(rgb_rows)), dim=-1)
        audio_rows = torch.empty(
            (1, 0, AUDIO_LATENT_CHANNELS), device=device, dtype=torch.float32
        )
        unique_times, row_time_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=video_indices.cpu(),
            audio_indices=audio_indices.cpu(),
            num_condition_video_rows=num_condition_video_rows,
            num_condition_audio_rows=num_condition_audio_rows,
            num_text_tokens=text_indices.numel(),
            video_timestep=0.0,
            audio_timestep=0.0,
            condition_video_timestep=KEYFRAME_TIMESTEP,
            condition_audio_timestep=1.0,
        )
        h3_mask = build_h3_observation_attention_mask(
            sequence_length=position_ids.shape[0],
            text_indices=text_indices,
            condition_video_indices=condition_indices,
            device=device,
        )
        forward_kwargs = {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "context": context,
            "timestep": unique_times.to(device),
            "timestep_indices": row_time_indices.to(device),
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "condition_video_indices": condition_indices,
            "state": state.reshape(1, 8),
            "context_mask": torch.ones(
                context.shape[:2], device=device, dtype=torch.bool
            ),
            "h3_attention_mask": h3_mask,
        }
        actions = torch.randn(
            (1, args.action_horizon, 7),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        torch.cuda.reset_peak_memory_stats(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            docked_keys = None
            docked_values = None
            for sigma, delta in zip(
                schedule.action_sigmas,
                schedule.action_sigma_deltas,
                strict=True,
            ):
                output = model(
                    **forward_kwargs,
                    noisy_actions=actions,
                    action_timestep=sigma.reshape(1) * 1000.0,
                    cached_docked_keys=docked_keys,
                    cached_docked_values=docked_values,
                )
                docked_keys = output.docked_keys
                docked_values = output.docked_values
                if docked_keys is None or docked_values is None:
                    raise RuntimeError("DoT forward did not return its docking cache")
                actions += output.action_velocity.float() * delta
        torch.cuda.synchronize(device)
        return actions[0], context_id, time.perf_counter() - inference_started

    listener = None
    connection = None
    try:
        if rank == 0:
            args.ready_file.resolve().parent.mkdir(parents=True, exist_ok=True)
            listener = Listener(
                ("127.0.0.1", args.port), authkey=b"h3wam-local-rollout"
            )
            args.ready_file.resolve().write_text(
                json.dumps(
                    {
                        "ready": True,
                        "policy": "h3dotwam_fsdp",
                        "world_size": world_size,
                        "sample_steps": args.sample_steps,
                        "action_horizon": args.action_horizon,
                        "checkpoint_steps": stage.get("steps"),
                        "h3_joint_stage": (
                            None
                            if args.h3_joint_stage is None
                            else str(args.h3_joint_stage.resolve())
                        ),
                        "load_seconds": load_seconds,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            connection = listener.accept()
        while True:
            request = connection.recv() if rank == 0 else None
            command = _broadcast_object(
                request.get("command", "") if rank == 0 else None, rank
            )
            if command == "close":
                if rank == 0:
                    connection.send({"ok": True})
                break
            if command != "predict":
                if rank == 0:
                    connection.send({"ok": False, "error": f"unknown {command!r}"})
                continue
            task = _broadcast_object(request["task"] if rank == 0 else None, rank)
            seed = int(_broadcast_object(request["seed"] if rank == 0 else None, rank))
            vae_seconds = 0.0
            if rank == 0:
                first, state, vae_seconds = encode_observation(request)
                latent_shape = tuple(first.shape)
            else:
                latent_shape = None
            latent_shape = tuple(_broadcast_object(latent_shape, rank))
            if rank != 0:
                first = torch.empty(latent_shape, device=device, dtype=torch.float32)
                state = torch.empty((8,), device=device, dtype=torch.float32)
            dist.broadcast(first, src=0)
            dist.broadcast(state, src=0)
            normalized, context_id, inference_seconds = policy_forward(
                task, first, state, seed
            )
            if rank == 0:
                raw = normalized.float()
                if args.clip_normalized_actions:
                    normalized = normalized.clamp(-1.0, 1.0)
                environment_actions = libero_environment_actions(
                    normalized,
                    stats["action_min"],
                    stats["action_max"],
                    binarize_gripper=args.binarize_gripper,
                    temporal_median_window=args.action_median_window,
                )
                environment_actions[:, :6] = np.clip(
                    environment_actions[:, :6] * args.action_scale, -1.0, 1.0
                )
                connection.send(
                    {
                        "ok": True,
                        "actions": environment_actions.tolist(),
                        "metadata": {
                            "context_id": context_id,
                            "inference_seconds": inference_seconds,
                            "vae_encode_seconds": vae_seconds,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                            / 2**30,
                            "first_environment_action": environment_actions[0].tolist(),
                            "environment_action_chunk": environment_actions.tolist(),
                            "normalized_action_abs_mean": float(raw.abs().mean()),
                            "normalized_action_saturation_fraction": float(
                                (raw.abs() >= 1.0).float().mean()
                            ),
                            "normalized_actions_clipped": args.clip_normalized_actions,
                        },
                    }
                )
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
        args.ready_file.resolve().unlink(missing_ok=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
