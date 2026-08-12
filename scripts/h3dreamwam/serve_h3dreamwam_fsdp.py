#!/usr/bin/env python3
"""Serve the staged H3-DreamWAM ActionDiT for closed-loop LIBERO rollout."""

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
    audio_latent_count,
)
from fastwam.models.h3dreamwam import load_action_block_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--action-stage", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--fp32-model-storage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clip-normalized-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project generated actions back to the normalized training support.",
    )
    parser.add_argument(
        "--require-text-only-context",
        action="store_true",
        help="Reject image-conditioned cached prompts to prevent stale-image leakage.",
    )
    parser.add_argument("--dreamwam-exact-action-norm", action="store_true")
    parser.add_argument("--action-init-alpha-scaling", action="store_true")
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    return parser.parse_args()


def _broadcast_object(value, rank: int):
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _load_stage(
    action_expert: torch.nn.Module,
    path: Path,
) -> tuple[set[int], dict, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "h3dreamwam_action_stage_v1":
        raise ValueError("action stage checkpoint has an incompatible format")
    modules = {
        "action_embedding": action_expert.action_embedding,
        "state_embedding": action_expert.state_embedding,
        "context_embedding": action_expert.context_embedding,
        "time_embedding": action_expert.time_embedding,
        "time_projection": action_expert.time_projection,
        "output": action_expert.output,
    }
    for name, module in modules.items():
        module.load_state_dict(payload["io"][name], strict=True)
    layers = set()
    migrated = 0
    for index_text, state_dict in payload["blocks"].items():
        index = int(index_text)
        if not 0 <= index < len(action_expert.blocks):
            raise ValueError(f"checkpoint action layer {index} lies outside model")
        migrated += int(
            load_action_block_state(action_expert.blocks[index], state_dict)
        )
        layers.add(index)
    return layers, payload, migrated


def _task_contexts(manifest: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result.setdefault(
                str(row["task"]),
                str(row.get("context_id", row["id"])),
            )
    if not result:
        raise ValueError("manifest contains no task contexts")
    return result


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
    if args.action_horizon != 32:
        raise ValueError("the current staged ActionDiT checkpoint was trained at horizon 32")

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
        H3DreamActionExpert,
        H3DreamPairedLayer,
        H3DreamWAM,
        build_h3dream_inference_schedule,
        expand_h3_rgb_flow_projections,
        initialize_action_expert_from_h3,
        sample_h3dream_joint_rows,
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
    projection = expand_h3_rgb_flow_projections(h3)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
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
            full_width_rmsnorm=args.dreamwam_exact_action_norm,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    initialization = initialize_action_expert_from_h3(
        action_expert,
        h3,
        alpha_scaling=args.action_init_alpha_scaling,
    )
    loaded_layers, stage_payload, migrated_legacy_action_blocks = _load_stage(
        action_expert, args.action_stage.resolve()
    )
    requested_architecture = {
        "full_width_rmsnorm": args.dreamwam_exact_action_norm,
        "alpha_scaling": args.action_init_alpha_scaling,
        "video_residual_gate": True,
        "video_residual_adapter_rank": 16,
    }
    stage_architecture = stage_payload.get("architecture", {})
    for key, value in stage_architecture.items():
        if key not in requested_architecture or requested_architecture[key] != value:
            raise ValueError(
                "action stage architecture mismatch: "
                f"checkpoint={stage_architecture}, requested={requested_architecture}"
            )
    stage_metadata = {
        "steps": stage_payload.get("steps"),
        "task": stage_payload.get("task"),
        "action_layers": sorted(loaded_layers),
        "h3_layers": sorted(int(index) for index in stage_payload.get("h3_blocks", {})),
        "has_h3_io": stage_payload.get("h3_io") is not None,
        "trained_action_horizon": stage_payload.get("action_horizon", 32),
        "migrated_legacy_action_blocks": migrated_legacy_action_blocks,
    }
    stage_h3_io = stage_payload.get("h3_io")
    if stage_h3_io is not None:
        h3.proj_in.load_state_dict(stage_h3_io["proj_in"], strict=True)
        h3.proj_out.load_state_dict(stage_h3_io["proj_out"], strict=True)
    for index_text, state_dict in stage_payload.get("h3_blocks", {}).items():
        h3.transformer_blocks[int(index_text)].load_state_dict(
            state_dict,
            strict=True,
        )
    action_expert.requires_grad_(False)
    model = H3DreamWAM(
        h3,
        action_expert,
        rgb_patch_width=projection.old_patch_width,
        use_gradient_checkpointing=False,
        compute_dtype=torch.bfloat16,
    )
    ignored_modules = [*model.h3.children()]
    ignored_modules = [
        module
        for module in ignored_modules
        if module is not model.paired_layers and len(list(module.parameters())) > 0
    ]
    for module in ignored_modules:
        module.to(device)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({H3DreamPairedLayer}),
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
    if args.fp32_model_storage:
        model.float()
        # The stage was initially read into BF16 modules to keep construction
        # memory bounded. Restore every saved FP32 tensor after FSDP has
        # materialized sharded FP32 storage, including DreamWAM's H3 branch.
        with FSDP.summon_full_params(model, recurse=False, writeback=True):
            root_action = model.module.action_expert
            for name, module in {
                "action_embedding": root_action.action_embedding,
                "state_embedding": root_action.state_embedding,
                "context_embedding": root_action.context_embedding,
                "time_embedding": root_action.time_embedding,
                "time_projection": root_action.time_projection,
                "output": root_action.output,
            }.items():
                module.load_state_dict(stage_payload["io"][name], strict=True)
            h3_io = stage_payload.get("h3_io")
            if h3_io is not None:
                model.module.h3.proj_in.load_state_dict(h3_io["proj_in"], strict=True)
                model.module.h3.proj_out.load_state_dict(h3_io["proj_out"], strict=True)
        restored_layers = set(stage_payload["blocks"]) | set(
            stage_payload.get("h3_blocks", {})
        )
        for index_text in sorted(restored_layers, key=int):
            paired_fsdp = model.module.paired_layers[int(index_text)]
            with FSDP.summon_full_params(
                paired_fsdp,
                recurse=False,
                writeback=True,
            ):
                if index_text in stage_payload["blocks"]:
                    load_action_block_state(
                        paired_fsdp.module.action_block,
                        stage_payload["blocks"][index_text],
                    )
                if index_text in stage_payload.get("h3_blocks", {}):
                    paired_fsdp.module.h3_block.load_state_dict(
                        stage_payload["h3_blocks"][index_text],
                        strict=True,
                    )
    del stage_payload
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
                    raise ValueError(
                        f"context {context_id!r} is not marked text_only=True"
                    )
                if torch.any(payload["token_tags"] != 1):
                    raise ValueError(
                        f"text-only context {context_id!r} has non-text token tags"
                    )
            context_cache[context_id] = {
                "context": payload["context"].to(device=device, dtype=torch.float32),
                "token_tags": payload["token_tags"].to(device=device, dtype=torch.long),
            }
        return context_id, context_cache[context_id]

    def encode_observation(request: dict) -> tuple[torch.Tensor, torch.Tensor, float]:
        assert rank == 0 and vae is not None
        started = time.perf_counter()
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
            time.perf_counter() - started,
        )

    def policy_forward(
        task: str, first: torch.Tensor, state: torch.Tensor, seed: int
    ) -> tuple[torch.Tensor, str, float]:
        context_id, conditioning = load_context(task)
        context = conditioning["context"]
        text_tags = conditioning["token_tags"]
        _, channels, _, latent_height, latent_width = first.shape
        pixel_frames = 3 * args.target_latent_frames + 3
        num_audio_latents = audio_latent_count(pixel_frames)
        layout_text_tags = torch.cat(
            (text_tags.cpu(), torch.ones(1, dtype=text_tags.dtype)),
        )
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=layout_text_tags,
            num_latent_frames=args.target_latent_frames,
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
        condition_rows = patchify_video_latents(first, patch_size)[None]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        future_shape = (
            1,
            channels,
            args.target_latent_frames,
            latent_height,
            latent_width,
        )
        future_noise = torch.randn(
            future_shape, generator=generator, device=device, dtype=torch.float32
        )
        initial_rows = torch.cat(
            (condition_rows, patchify_video_latents(future_noise, patch_size)[None]),
            dim=1,
        )
        audio_rows = torch.randn(
            (1, num_audio_latents * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        h3_mask = build_h3_observation_attention_mask(
            sequence_length=position_ids.shape[0],
            text_indices=text_indices,
            condition_video_indices=video_indices[:num_condition_video_rows],
            device=device,
        )
        context_mask = torch.ones(
            context.shape[:2], device=device, dtype=torch.bool
        )

        def predict_velocity(rgb_rows, actions, video_time, action_sigma):
            video_rows = torch.cat((rgb_rows, torch.zeros_like(rgb_rows)), dim=-1)
            unique_times, row_time_indices = (
                MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    video_indices=video_indices.cpu(),
                    audio_indices=audio_indices.cpu(),
                    num_condition_video_rows=int(num_condition_video_rows),
                    num_condition_audio_rows=int(num_condition_audio_rows),
                    num_text_tokens=text_indices.numel(),
                    video_timestep=float(video_time),
                    audio_timestep=0.0,
                    condition_video_timestep=max(float(video_time), KEYFRAME_TIMESTEP),
                    condition_audio_timestep=1.0,
                )
            )
            output = model(
                video_rows=video_rows,
                audio_rows=audio_rows,
                context=context,
                timestep=unique_times.to(device),
                timestep_indices=row_time_indices.to(device),
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                noisy_actions=actions,
                action_timestep=action_sigma.reshape(1) * 1000.0,
                state=state.reshape(1, 8),
                context_mask=context_mask,
                action_video_indices=video_indices,
                h3_attention_mask=h3_mask,
            )
            return output.rgb_velocity_rows, output.action_velocity

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        sampled = sample_h3dream_joint_rows(
            predict_velocity,
            initial_video_rows=initial_rows,
            condition_video_rows=int(num_condition_video_rows),
            initial_actions=torch.randn(
                (1, args.action_horizon, 7),
                generator=generator,
                device=device,
                dtype=torch.float32,
            ),
            schedule=schedule,
        )
        torch.cuda.synchronize(device)
        return sampled.actions[0], context_id, time.perf_counter() - started

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
                        "policy": "h3dreamwam_fsdp",
                        "world_size": world_size,
                        "sample_steps": args.sample_steps,
                        "action_horizon": args.action_horizon,
                        "loaded_action_layers": sorted(loaded_layers),
                        "loaded_h3_layers": stage_metadata["h3_layers"],
                        "loaded_h3_io": stage_metadata["has_h3_io"],
                        "checkpoint_steps": stage_metadata["steps"],
                        "checkpoint_task": stage_metadata["task"],
                        "fp32_model_storage": args.fp32_model_storage,
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
                    connection.send(
                        {"ok": False, "error": f"unknown command {command!r}"}
                    )
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
                raw_normalized = normalized.float()
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
                            "normalized_action_abs_mean": float(
                                raw_normalized.abs().mean()
                            ),
                            "normalized_action_saturation_fraction": float(
                                (raw_normalized.abs() >= 1.0).float().mean()
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
