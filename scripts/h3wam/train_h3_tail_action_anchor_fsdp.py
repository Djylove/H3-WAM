#!/usr/bin/env python3
"""Conservatively adapt the official H3 tail through a frozen action expert.

Only the selected final H3 blocks are trainable.  The local regression action
head stays frozen and supplies the task loss, while cached features from the
released BF16 H3 model anchor the observation representation against drift.
Checkpoints use the same rank-local format consumed by the rollout server.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    replicated_non_block_modules,
    set_trainable_tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-subdir", default="h3_official_features_fixedctx")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument(
        "--adaptation", choices=("lora", "full_tail"), default="lora"
    )
    parser.add_argument("--last-blocks", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--keep-last-checkpoints", type=int, default=5)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument(
        "--max-initial-anchor-mse",
        type=float,
        default=1e-2,
        help="Abort before optimization if the released-backbone feature check fails.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--feature-video-timestep", type=float, default=1.0)
    parser.add_argument(
        "--fp32-master-weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--training-episode",
        type=int,
        help="Restrict optimization rows to one corrective episode; validation remains global.",
    )
    parser.add_argument(
        "--fixed-row-index",
        type=int,
        help="Diagnostic: train every rank/step on one manifest row.",
    )
    return parser.parse_args()


def cosine_factor(step: int, *, warmup: int, total: int, minimum: float) -> float:
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    progress = min(1.0, max(0.0, (step - warmup) / max(total - warmup, 1)))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def minmax_normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scale = (high.float() - low.float()).clamp_min(1e-6)
    return ((value.float() - low.float()) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def main() -> None:
    args = parse_args()
    if min(
        args.steps,
        args.last_blocks,
        args.validate_every,
        args.target_latent_frames,
    ) <= 0:
        raise ValueError("steps, last-blocks, validate-every and latent frames must be positive")
    if args.learning_rate <= 0 or args.anchor_weight < 0 or args.gradient_clip <= 0:
        raise ValueError("invalid optimization arguments")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup-steps must be in [0, steps)")
    if not 0.0 <= args.minimum_lr_ratio <= 1.0:
        raise ValueError("minimum-lr-ratio must be in [0,1]")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("this script requires multi-GPU FSDP")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from fastwam.models.h3wam import (
        H3FeatureActionTransformer,
        H3MixtureActionOutput,
        H3OfficialFeatureCapture,
        H3LoRALinear,
        h3_lora_parameters,
        h3_lora_state_dict,
        inject_official_h3_lora,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("manifest is empty")
    training_rows = (
        rows
        if args.training_episode is None
        else [row for row in rows if int(row["episode"]) == args.training_episode]
    )
    if not training_rows:
        raise ValueError("training-episode selected no manifest rows")
    tasks = {str(row["task"]) for row in rows}
    if len(tasks) != 1:
        raise ValueError("the conservative canary currently expects exactly one task")
    task = next(iter(tasks))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    action_artifact = torch.load(
        args.action_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if action_artifact.get("policy_type") != "h3_feature_action":
        raise ValueError("action checkpoint is not an H3 feature action policy")
    if action_artifact.get("objective", "regression") != "regression":
        raise ValueError("tail canary requires a regression action expert")
    action_horizon = int(action_artifact["action_horizon"])
    capture_layers = tuple(int(value) for value in action_artifact["feature_layers"])
    feature_shape = tuple(int(value) for value in action_artifact["feature_shape"])
    if feature_shape[0] != len(capture_layers):
        raise ValueError("action checkpoint feature contract is inconsistent")
    task_to_index = {
        str(key): int(value)
        for key, value in action_artifact.get("task_to_index", {}).items()
    }
    if task not in task_to_index:
        raise ValueError(f"task {task!r} is absent from action checkpoint")
    task_mode = task_to_index[task]
    stats = action_artifact["normalization"]

    action_head = H3FeatureActionTransformer(
        action_dim=int(action_artifact["action_dim"]),
        state_dim=int(action_artifact["state_dim"]),
        h3_feature_dim=feature_shape[-1],
        hidden_dim=int(action_artifact["hidden_dim"]),
        num_layers=int(action_artifact["num_layers"]),
        num_heads=int(action_artifact["num_heads"]),
        ffn_dim=int(action_artifact["ffn_dim"]),
        num_action_modes=int(action_artifact.get("num_action_modes", 1)),
    ).to(device)
    action_head.load_state_dict(action_artifact["model"], strict=True)
    action_head.requires_grad_(False).eval()

    load_started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.adaptation == "lora":
        h3.requires_grad_(False)
        lora_report = inject_official_h3_lora(
            h3,
            last_n_blocks=args.last_blocks,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        trainable_names = [
            name
            for name, parameter in h3.named_parameters()
            if parameter.requires_grad
        ]
    else:
        lora_report = None
        trainable_names = set_trainable_tail(h3, args.last_blocks)
    first_trainable_layer = len(h3.transformer_blocks) - args.last_blocks
    anchored_positions = tuple(
        index for index, layer in enumerate(capture_layers) if layer >= first_trainable_layer
    )
    if not anchored_positions:
        raise ValueError("no captured feature layer is affected by the selected H3 tail")
    if args.adaptation == "full_tail" and args.fp32_master_weights:
        for block in h3.transformer_blocks[-args.last_blocks :]:
            block.to(torch.float32)
    # Keep the backbone in the same inference mode used to create the feature
    # cache and in deployment. Gradients still flow through the selected tail;
    # eval mode does not imply no_grad. This also avoids train-mode MoE routing
    # differences from invalidating the frozen-backbone anchor at step zero.
    if hasattr(h3, "disable_gradient_checkpointing"):
        h3.disable_gradient_checkpointing()
    replicated_modules = replicated_non_block_modules(h3)
    if args.adaptation == "lora":
        replicated_modules.extend(
            child
            for module in h3.modules()
            if isinstance(module, H3LoRALinear)
            for child in (module.lora_a, module.lora_b)
        )
    for module in replicated_modules:
        module.to(device)
    h3 = FSDP(
        h3,
        auto_wrap_policy=ModuleWrapPolicy({MiniMaxH3TransformerBlock}),
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        ignored_modules=replicated_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    )
    h3.eval()
    patch_size = tuple(h3.module.config.patch_size)
    if feature_shape[-1] != int(h3.module.config.hidden_size):
        raise ValueError("feature hidden size does not match official H3")
    trainable = (
        h3_lora_parameters(h3.module)
        if args.adaptation == "lora"
        else [parameter for parameter in h3.parameters() if parameter.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_factor(
            step,
            warmup=args.warmup_steps,
            total=args.steps,
            minimum=args.minimum_lr_ratio,
        ),
    )

    context_item = torch.load(args.context.resolve(), map_location="cpu", weights_only=False)
    context = context_item["context"].to(device=device, dtype=torch.float32)
    text_tags = context_item["token_tags"].long()
    example = torch.load(
        data_root / "windows" / f"{rows[0]['id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    first_example = example["first_frame_latents"]
    _, channels, _, latent_height, latent_width = first_example.shape
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=args.target_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=action_horizon,
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
    video_indices_device = video_indices.to(device)
    audio_indices_device = audio_indices.to(device)
    text_indices_device = text_indices.to(device)
    unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=video_indices,
        audio_indices=audio_indices,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        num_text_tokens=text_indices.numel(),
        video_timestep=args.feature_video_timestep,
        audio_timestep=0.0,
        condition_video_timestep=1.0,
        condition_audio_timestep=1.0,
    )
    capture = H3OfficialFeatureCapture(
        h3.module.transformer_blocks,
        capture_layers,
        video_indices_device[:num_condition_video_rows],
    )
    zero_target = torch.zeros(
        (1, channels, args.target_latent_frames, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )
    zero_target_rows = patchify_video_latents(zero_target, patch_size)
    zero_audio = torch.zeros(
        (1, action_horizon * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    gripper_weights = torch.ones(7, device=device)
    gripper_weights[-1] = float(action_artifact.get("gripper_loss_weight", 1.0))

    def load_row(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_id = str(row["id"])
        window = torch.load(
            data_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        base = torch.load(
            data_root / args.feature_subdir / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )["features"]
        first = window["first_frame_latents"].to(device=device, dtype=torch.float32)
        actions = minmax_normalize(
            window["actions"].float(), stats["action_min"], stats["action_max"]
        )[:action_horizon].to(device)
        state_parts = []
        if bool(action_artifact.get("use_proprio", False)):
            proprio = minmax_normalize(
                window["state"].float(), stats["state_min"], stats["state_max"]
            )
            state_parts.append(proprio.reshape(1, 8).expand(action_horizon, -1).clone())
        if bool(action_artifact.get("use_previous_action", False)):
            raise NotImplementedError("previous-action conditioning is not supported")
        if bool(action_artifact.get("include_phase", True)):
            length = int(
                action_artifact.get("phase_lengths_by_task", {}).get(
                    task, action_artifact["phase_length"]
                )
            )
            phase_steps = torch.arange(action_horizon, dtype=torch.float32)
            phase_steps.add_(int(row["start"])).clamp_max_(length - 1)
            state_parts.append((2.0 * phase_steps / max(length - 1, 1) - 1.0)[:, None])
        if bool(action_artifact.get("task_conditioning", False)):
            one_hot = torch.zeros(action_horizon, len(task_to_index))
            one_hot[:, task_mode] = 1.0
            state_parts.append(one_hot)
        state = torch.cat(state_parts, dim=-1).to(device)
        if state.shape[-1] != int(action_artifact["state_dim"]):
            raise RuntimeError("constructed policy state has the wrong width")
        return first, actions, state, base.to(device=device, dtype=torch.float32)

    def losses_for(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, actions, policy_state, base_features = load_row(row)
        video_rows = torch.cat(
            (patchify_video_latents(first, patch_size), zero_target_rows), dim=0
        )[None]
        capture.clear()
        h3(
            hidden_states=video_rows,
            audio_hidden_states=zero_audio,
            encoder_hidden_states=context,
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices_device,
            audio_indices=audio_indices_device,
            text_indices=text_indices_device,
            return_dict=True,
        )
        features = capture.stacked()
        if tuple(features.shape[1:]) != feature_shape:
            raise RuntimeError(f"feature shape {tuple(features.shape[1:])} != {feature_shape}")
        output = action_head(
            torch.zeros((1, action_horizon, 7), device=device),
            state=policy_state.unsqueeze(0),
            h3_features=features,
            video_sigma=torch.zeros(1, device=device),
        )
        if isinstance(output, H3MixtureActionOutput):
            prediction = output.actions[:, task_mode]
        else:
            prediction = output
        per_dimension = (prediction.float() - actions.unsqueeze(0)).square()
        action_loss = (
            per_dimension * gripper_weights.reshape(1, 1, -1)
        ).sum(dim=-1).div(gripper_weights.sum()).mean()
        selected = torch.tensor(anchored_positions, device=device, dtype=torch.long)
        current_anchor = features.index_select(1, selected).float()
        base_anchor = base_features.index_select(0, selected).unsqueeze(0)
        # H3's deep raw residual stream has a very large, depth-dependent
        # scale. The frozen action expert consumes it only after this learned
        # LayerNorm + projection, so anchor the representation the policy can
        # actually observe instead of a misleading absolute raw-token MSE.
        current_policy_features = action_head.feature_projection(current_anchor)
        base_policy_features = action_head.feature_projection(base_anchor)
        anchor_loss = F.mse_loss(current_policy_features, base_policy_features)
        total = action_loss + args.anchor_weight * anchor_loss
        return total, action_loss, anchor_loss

    @torch.no_grad()
    def validate(step: int) -> dict:
        h3.eval()
        totals: dict[int, torch.Tensor] = {}
        counts: dict[int, torch.Tensor] = {}
        # Every FSDP rank must execute the same number of forward collectives.
        # Pad the final distributed validation batch, but exclude padding from
        # metric accumulation.
        validation_rounds = math.ceil(len(rows) / world_size)
        for validation_round in range(validation_rounds):
            index = validation_round * world_size + rank
            valid = index < len(rows)
            row = rows[index % len(rows)]
            _, action_loss, anchor_loss = losses_for(row)
            if not valid:
                continue
            episode = int(row["episode"])
            if episode not in totals:
                totals[episode] = torch.zeros(2, device=device, dtype=torch.float64)
                counts[episode] = torch.zeros(1, device=device, dtype=torch.float64)
            totals[episode] += torch.tensor(
                [float(action_loss), float(anchor_loss)], device=device, dtype=torch.float64
            )
            counts[episode] += 1
        episodes = sorted({int(row["episode"]) for row in rows})
        report = {"event": "validation", "step": step}
        for episode in episodes:
            value = totals.get(episode, torch.zeros(2, device=device, dtype=torch.float64))
            count = counts.get(episode, torch.zeros(1, device=device, dtype=torch.float64))
            dist.all_reduce(value)
            dist.all_reduce(count)
            value /= count.clamp_min(1)
            report[f"episode_{episode}_action"] = float(value[0])
            report[f"episode_{episode}_anchor"] = float(value[1])
        h3.train()
        return report

    def checkpoint(step: int) -> Path:
        final = output_dir / f"step{step:06d}"
        partial = output_dir / f"step{step:06d}.partial"
        if rank == 0:
            existing = sorted(
                (path for path in output_dir.glob("step[0-9][0-9][0-9][0-9][0-9][0-9]") if path.is_dir()),
                key=lambda path: path.name,
            )
            while len(existing) >= args.keep_last_checkpoints:
                shutil.rmtree(existing.pop(0))
            partial.mkdir()
        dist.barrier()
        if args.adaptation == "lora":
            if rank == 0:
                payload = {
                    "format": "h3wam-official-h3-lora-v1",
                    "step": step,
                    "last_blocks": args.last_blocks,
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "state": h3_lora_state_dict(h3.module),
                }
                temporary = partial / f".h3_lora.{os.getpid()}.tmp"
                torch.save(payload, temporary)
                os.replace(temporary, partial / "h3_lora.pt")
        else:
            parameters = {
                name: parameter.detach().to(device="cpu", dtype=torch.bfloat16)
                for name, parameter in h3.named_parameters()
                if parameter.requires_grad
            }
            payload = {
                "format": "h3wam-fsdp-local-bf16-v1",
                "step": step,
                "rank": rank,
                "world_size": world_size,
                "parameters": parameters,
            }
            temporary = partial / f".h3_rank{rank:05d}.{os.getpid()}.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, partial / f"h3_rank{rank:05d}.pt")
        dist.barrier()
        if rank == 0:
            manifest = {
                "format": (
                    "h3wam-official-h3-lora-v1"
                    if args.adaptation == "lora"
                    else "h3wam-fsdp-local-bf16-v1"
                ),
                "step": step,
                "world_size": world_size,
                "last_blocks": args.last_blocks,
                "action_checkpoint": str(args.action_checkpoint.resolve()),
                "anchor_weight": args.anchor_weight,
            }
            (partial / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(partial, final)
        dist.barrier()
        return final

    metadata = {
        "event": "ready",
        "host": socket.gethostname(),
        "world_size": world_size,
        "rows": len(rows),
        "training_rows": len(training_rows),
        "training_episode": args.training_episode,
        "task": task,
        "adaptation": args.adaptation,
        "lora_rank": None if lora_report is None else args.lora_rank,
        "lora_alpha": None if lora_report is None else args.lora_alpha,
        "lora_modules": None if lora_report is None else lora_report.modules,
        "last_blocks": args.last_blocks,
        "capture_layers": capture_layers,
        "anchored_positions": anchored_positions,
        "trainable_parameter_shards": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_names": len(trainable_names),
        "fp32_master_weights": args.fp32_master_weights,
        "load_seconds": time.perf_counter() - load_started,
    }
    if rank == 0:
        print(json.dumps(metadata, sort_keys=True), flush=True)
    initial = validate(0)
    if rank == 0:
        print(json.dumps(initial, sort_keys=True), flush=True)
    maximum_initial_anchor = max(
        value for key, value in initial.items() if key.endswith("_anchor")
    )
    if maximum_initial_anchor > args.max_initial_anchor_mse:
        raise RuntimeError(
            "step-0 H3 feature equivalence failed: maximum anchor MSE "
            f"{maximum_initial_anchor} exceeds {args.max_initial_anchor_mse}"
        )
    if args.preflight_only:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "preflight_complete",
                        "maximum_initial_anchor_mse": maximum_initial_anchor,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        capture.close()
        dist.barrier()
        dist.destroy_process_group()
        return

    # FSDP installs its backward bookkeeping in training mode. The H3 feature
    # contract itself is protected by the explicit step-0 projected anchor.
    h3.train()

    torch.manual_seed(args.seed + rank)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    last_checkpoint = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        row_index = (
            int(args.fixed_row_index) % len(training_rows)
            if args.fixed_row_index is not None
            else ((step - 1) * world_size + rank) % len(training_rows)
        )
        total, action_loss, anchor_loss = losses_for(training_rows[row_index])
        total.backward()
        if args.adaptation == "lora":
            for parameter in trainable:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                    parameter.grad.div_(world_size)
        local_gradient_count = sum(
            parameter.grad is not None for parameter in trainable
        )
        gradient_counts = torch.tensor(
            [local_gradient_count], device=device, dtype=torch.int64
        )
        minimum_gradient_counts = gradient_counts.clone()
        maximum_gradient_counts = gradient_counts.clone()
        dist.all_reduce(minimum_gradient_counts, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum_gradient_counts, op=dist.ReduceOp.MAX)
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            if args.adaptation == "lora"
            else h3.clip_grad_norm_(args.gradient_clip)
        )
        optimizer.step()
        scheduler.step()
        values = torch.tensor(
            [float(total.detach()), float(action_loss.detach()), float(anchor_loss.detach())],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(values)
        values /= world_size
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "total": float(values[0]),
                        "action": float(values[1]),
                        "anchor": float(values[2]),
                        "grad_norm": float(grad_norm),
                        "gradient_tensors_per_rank_min": int(minimum_gradient_counts),
                        "gradient_tensors_per_rank_max": int(maximum_gradient_counts),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.validate_every == 0 or step == args.steps:
            report = validate(step)
            if rank == 0:
                print(json.dumps(report, sort_keys=True), flush=True)
        if args.checkpoint_every and (step % args.checkpoint_every == 0 or step == args.steps):
            last_checkpoint = checkpoint(step)
            if rank == 0:
                print(json.dumps({"event": "checkpoint", "step": step, "path": str(last_checkpoint)}), flush=True)

    report = {
        **metadata,
        "event": "complete",
        "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib_per_rank": torch.cuda.max_memory_allocated(device) / 2**30,
        "last_checkpoint": None if last_checkpoint is None else str(last_checkpoint),
        "initial_validation": initial,
    }
    if rank == 0:
        (output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    capture.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
