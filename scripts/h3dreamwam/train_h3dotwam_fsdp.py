#!/usr/bin/env python3
"""Train the H3 Faster-WAM/DoT head on real cached LIBERO windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "h3wam"))

from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    KEYFRAME_TIMESTEP,
    audio_latent_count,
    resolve_sample,
    shifted_video_timestep,
    validate_cached_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--motion-root",
        type=Path,
        help="Directory of DreamWAM RAFT/H3-VAE flow-latent artifacts.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-stage", type=Path)
    parser.add_argument("--load-stage", type=Path)
    parser.add_argument("--save-joint-stage", type=Path)
    parser.add_argument("--load-joint-stage", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--joint-checkpoint-every",
        type=int,
        default=0,
        help="Save rank-sharded H3 plus action weights every N optimizer steps.",
    )
    parser.add_argument(
        "--keep-last-joint-checkpoints",
        type=int,
        default=2,
        help="Retain this many periodic joint checkpoints; zero retains all.",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument(
        "--action-layers",
        type=int,
        help=(
            "Depth of the DoT action carrier. Defaults to the loaded stage's "
            "self-described depth, or one for a new stage."
        ),
    )
    parser.add_argument(
        "--initialize-action-from-h3",
        action="store_true",
        help=(
            "Initialize a new DoT carrier from uniformly depth-sampled H3 "
            "blocks using FastWAM-style interpolation and alpha scaling."
        ),
    )
    parser.add_argument(
        "--action-init-output-scale",
        type=float,
        default=0.01,
        help="Residual output scale for H3-initialized DoT attention/FFN routes.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--h3-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--h3-io-learning-rate",
        type=float,
        help=(
            "Learning rate for the newly expanded H3 RGB/flow projections. "
            "Defaults to 1e-4 for motion training and h3-learning-rate otherwise."
        ),
    )
    parser.add_argument("--last-h3-blocks", type=int, default=0)
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--flow-loss-weight",
        type=float,
        default=0.5,
        help="DreamWAM motion flow-matching weight (paper setting: 0.5).",
    )
    parser.add_argument(
        "--train-h3-io",
        action="store_true",
        help="Train and checkpoint the expanded H3 RGB/flow projections.",
    )
    parser.add_argument(
        "--flow-channel-init-scale",
        type=float,
        help=(
            "Gaussian initialization scale relative to pretrained RGB projection "
            "std. Defaults to DreamWAM's 0.1 for motion training and zero otherwise."
        ),
    )
    parser.add_argument(
        "--dreamwam-world-weighting",
        action="store_true",
        help="Apply DreamWAM normalized timestep weighting to RGB/flow losses.",
    )
    parser.add_argument("--language-ranking-weight", type=float, default=0.0)
    parser.add_argument("--language-ranking-margin", type=float, default=0.05)
    parser.add_argument("--language-ranking-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-text-only-context", action="store_true")
    parser.add_argument("--context-override-id")
    parser.add_argument("--record-sampled-actions", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def select_negative_context(rows: list[dict], row_index: int) -> tuple[str, str]:
    """Choose a deterministic task-mismatched text context for one window."""

    if len(rows) < 2:
        raise ValueError("language ranking requires at least two manifest rows")
    anchor = rows[row_index % len(rows)]
    anchor_context = str(anchor.get("context_id", anchor["id"]))
    anchor_task = str(anchor.get("task", ""))
    for offset in range(1, len(rows)):
        candidate = rows[(row_index + offset) % len(rows)]
        candidate_context = str(candidate.get("context_id", candidate["id"]))
        candidate_task = str(candidate.get("task", ""))
        if candidate_context != anchor_context and candidate_task != anchor_task:
            return candidate_context, candidate_task
    raise ValueError("manifest has no task-mismatched context for language ranking")


def action_language_ranking_loss(
    correct_action_loss: torch.Tensor,
    wrong_action_loss: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require the wrong-language prediction to be worse by ``margin``."""

    return F.relu(correct_action_loss.detach() + margin - wrong_action_loss)


def masked_token_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean MSE over valid temporal/patch tokens only."""

    if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
        raise ValueError(
            f"masked MSE shape mismatch: {tuple(prediction.shape)}, "
            f"{tuple(target.shape)}, {tuple(valid.shape)}"
        )
    token_loss = F.mse_loss(
        prediction.float(), target.float(), reduction="none"
    ).mean(dim=-1)
    weight = valid.to(device=token_loss.device, dtype=token_loss.dtype)
    return (token_loss * weight).sum() / weight.sum().clamp_min(1.0)


def normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scale = (high.float() - low.float()).clamp_min(1.0e-6)
    return ((value.float() - low.float()) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def normalize_world_latents(value: torch.Tensor) -> torch.Tensor:
    """Apply DreamWAM's per-sample normalization to motion teacher latents."""

    if value.ndim != 5:
        raise ValueError("world latents must be [B,C,T,H,W]")
    dimensions = tuple(range(1, value.ndim))
    mean = value.mean(dim=dimensions, keepdim=True)
    std = value.std(dim=dimensions, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (value - mean) / std


def resolve_h3_io_hyperparameters(
    *,
    motion_enabled: bool,
    h3_learning_rate: float,
    h3_io_learning_rate: float | None,
    flow_channel_init_scale: float | None,
) -> tuple[float, float]:
    """Resolve DreamWAM-specific defaults without changing the RGB-only path."""

    io_learning_rate = (
        (1.0e-4 if motion_enabled else h3_learning_rate)
        if h3_io_learning_rate is None
        else h3_io_learning_rate
    )
    init_scale = (
        (0.1 if motion_enabled else 0.0)
        if flow_channel_init_scale is None
        else flow_channel_init_scale
    )
    if io_learning_rate <= 0:
        raise ValueError("H3 I/O learning rate must be positive")
    if init_scale < 0:
        raise ValueError("flow channel init scale must be non-negative")
    return float(io_learning_rate), float(init_scale)


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def resolve_action_stage(
    *,
    load_stage: Path | None,
    load_joint_stage: Path | None,
) -> Path | None:
    """Select action weights independently from the H3 joint checkpoint.

    A joint stage still supplies its bundled action weights by default.  An
    explicit ``--load-stage`` overrides only those action-side weights while
    ``--load-joint-stage`` continues to restore the H3 shards.  Serving already
    supports this composition; training/evaluation need the same contract for
    frozen-world action-head ablations.
    """

    if load_stage is not None:
        return load_stage.resolve()
    if load_joint_stage is not None:
        return load_joint_stage.resolve() / "action_stage.pt"
    return None


def _is_joint_h3_parameter(name: str) -> bool:
    return (
        ".hub_layers." in name and ".h3_block." in name
    ) or any(marker in name for marker in (".h3.proj_in.", ".h3.proj_out."))


def save_joint_h3_shards(
    *,
    stage_dir: Path,
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    steps: int,
    last_h3_blocks: int,
) -> None:
    """Save same-world-size BF16 H3 shards without duplicating Adam states."""

    stage_dir = stage_dir.resolve()
    if rank == 0:
        if not stage_dir.is_dir() or (stage_dir / "joint_stage.json").exists():
            raise FileExistsError(f"joint stage directory is not fresh: {stage_dir}")
    dist.barrier()
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if _is_joint_h3_parameter(name) and parameter.requires_grad
    }
    if not state:
        raise RuntimeError("joint stage has no trainable H3 block shards")
    artifact = {
        "format": "h3dotwam_joint_h3_same_world_size_v1",
        "rank": rank,
        "world_size": world_size,
        "steps": steps,
        "last_h3_blocks": last_h3_blocks,
        "state": state,
    }
    path = stage_dir / f"h3_rank{rank:05d}.pt"
    partial = path.with_suffix(".pt.partial")
    torch.save(artifact, partial)
    os.replace(partial, path)
    dist.barrier()
    if rank == 0:
        manifest = {
            "format": artifact["format"],
            "world_size": world_size,
            "steps": steps,
            "last_h3_blocks": last_h3_blocks,
            "train_h3_io": any(
                marker in name
                for name in state
                for marker in (".h3.proj_in.", ".h3.proj_out.")
            ),
            "rank_files": [f"h3_rank{index:05d}.pt" for index in range(world_size)],
            "action_stage": "action_stage.pt",
        }
        (stage_dir / "joint_stage.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    dist.barrier()


def load_joint_h3_shard(
    *,
    stage_dir: Path,
    model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict:
    stage_dir = stage_dir.resolve()
    manifest = json.loads((stage_dir / "joint_stage.json").read_text())
    if manifest.get("format") != "h3dotwam_joint_h3_same_world_size_v1":
        raise ValueError("joint H3 stage format mismatch")
    if int(manifest["world_size"]) != world_size:
        raise ValueError("joint H3 stage requires the same FSDP world size")
    artifact = torch.load(
        stage_dir / f"h3_rank{rank:05d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        artifact.get("format") != manifest["format"]
        or int(artifact["rank"]) != rank
        or int(artifact["world_size"]) != world_size
    ):
        raise ValueError("rank-local joint H3 shard metadata mismatch")
    named = dict(model.named_parameters())
    state = artifact["state"]
    missing = sorted(set(state) - set(named))
    if missing:
        raise ValueError(f"joint H3 shard contains unknown tensors: {missing[:3]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(
                    f"joint H3 shard shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=device, dtype=parameter.dtype))
    return manifest


def main() -> None:
    args = parse_args()
    args.load_stage = resolve_action_stage(
        load_stage=args.load_stage,
        load_joint_stage=args.load_joint_stage,
    )
    if args.steps <= 0 or not 1 <= args.action_horizon <= 32:
        raise ValueError("steps must be positive and action horizon must be in [1,32]")
    if args.action_layers is not None and args.action_layers <= 0:
        raise ValueError("action-layers must be positive")
    if not 0.0 < args.action_init_output_scale <= 1.0:
        raise ValueError("action-init-output-scale must be in (0,1]")
    if args.initialize_action_from_h3 and args.load_stage is not None:
        raise ValueError("H3 action initialization requires a new action stage")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    if (
        args.sample_offset < 0
        or args.checkpoint_every < 0
        or args.joint_checkpoint_every < 0
        or args.keep_last_joint_checkpoints < 0
    ):
        raise ValueError("sample offset and checkpoint settings cannot be negative")
    if args.learning_rate <= 0 or args.h3_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    h3_io_learning_rate, flow_channel_init_scale = resolve_h3_io_hyperparameters(
        motion_enabled=args.motion_root is not None,
        h3_learning_rate=args.h3_learning_rate,
        h3_io_learning_rate=args.h3_io_learning_rate,
        flow_channel_init_scale=args.flow_channel_init_scale,
    )
    if args.video_loss_weight < 0 or args.flow_loss_weight < 0:
        raise ValueError("video/flow loss weights cannot be negative")
    if args.motion_root is not None and not args.train_h3_io:
        raise ValueError("--motion-root requires --train-h3-io")
    if args.train_h3_io and args.motion_root is None:
        raise ValueError("--train-h3-io requires --motion-root")
    if args.language_ranking_weight < 0 or args.language_ranking_margin <= 0:
        raise ValueError("language ranking weight/margin must be nonnegative/positive")
    if args.language_ranking_every <= 0:
        raise ValueError("language-ranking-every must be positive")
    if args.eval_only and args.language_ranking_weight:
        raise ValueError("eval-only does not use language ranking")
    if args.sample_steps <= 0:
        raise ValueError("sample-steps must be positive")
    if args.eval_only and args.load_stage is None:
        raise ValueError("eval-only requires --load-stage")
    if args.eval_only and args.save_stage is not None:
        raise ValueError("eval-only cannot save a training stage")
    if args.eval_only and args.save_joint_stage is not None:
        raise ValueError("eval-only cannot save a joint stage")
    if args.save_joint_stage is not None and args.last_h3_blocks == 0:
        raise ValueError("saving a joint stage requires trainable H3 blocks")
    if args.joint_checkpoint_every and args.save_joint_stage is None:
        raise ValueError("periodic joint checkpoints require --save-joint-stage")
    if args.eval_only and args.gradient_accumulation_steps != 1:
        raise ValueError("eval-only does not use gradient accumulation")
    if not 0 <= args.last_h3_blocks <= 50:
        raise ValueError("last-h3-blocks must be in [0,50]")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("real H3 DoT training requires multi-GPU FSDP")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from fastwam.models.h3dreamwam import (
        H3DoTActionHead,
        H3DoTHubLayer,
        H3DoTKVFusion,
        H3DoTWAM,
        expand_h3_rgb_flow_projections,
        build_h3dream_inference_schedule,
        h3dream_flow_training_weight,
        initialize_dot_action_head_from_h3,
    )
    from fastwam.models.h3wam import build_h3_observation_attention_mask
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    h3.requires_grad_(False)
    if args.last_h3_blocks:
        for block in h3.transformer_blocks[-args.last_h3_blocks :]:
            block.requires_grad_(True)
    projection_report = expand_h3_rgb_flow_projections(
        h3,
        flow_input_init_scale=flow_channel_init_scale,
        flow_output_init_scale=flow_channel_init_scale,
    )
    if args.train_h3_io:
        h3.proj_in.requires_grad_(True)
        h3.proj_out.requires_grad_(True)

    stage_payload = None
    if args.load_stage is not None:
        stage_payload = torch.load(
            args.load_stage.resolve(), map_location="cpu", weights_only=True
        )
        if stage_payload.get("format") != "h3dotwam_stage_v2":
            raise ValueError("DoT stage checkpoint format mismatch")
        stage_action_layers = int(
            stage_payload.get("architecture", {}).get("action_layers", 1)
        )
        if args.action_layers is not None and args.action_layers != stage_action_layers:
            raise ValueError(
                "action-layers does not match loaded stage: "
                f"requested {args.action_layers}, checkpoint {stage_action_layers}"
            )
        action_layers = stage_action_layers
    else:
        action_layers = 1 if args.action_layers is None else args.action_layers

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    action_initialization = None
    try:
        action_head = H3DoTActionHead(
            action_dim=7,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            head_dim=128,
            num_layers=action_layers,
            frequency_dim=256,
            full_width_rmsnorm=True,
        )
        kv_fusion = H3DoTKVFusion(
            video_layers=50,
            action_layers=action_layers,
            video_num_heads=56,
            video_head_dim=128,
            action_num_heads=24,
            action_head_dim=128,
        )
        if args.initialize_action_from_h3:
            # H3DoTWAM transfers transformer_blocks into its FSDP hub wrappers,
            # so copy pretrained tensors while the source stack is still
            # present on the original H3 module.
            action_initialization = initialize_dot_action_head_from_h3(
                action_head,
                h3,
                residual_output_scale=args.action_init_output_scale,
                alpha_scaling=True,
            )
        model = H3DoTWAM(
            h3,
            action_head,
            kv_fusion,
            state_dim=8,
            text_dim=5120,
            rgb_patch_width=projection_report.old_patch_width,
            use_gradient_checkpointing=True,
            compute_dtype=torch.bfloat16,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    action_initialization_metadata = (
        None
        if action_initialization is None
        else {
            "type": "h3_depth_sampled_fastwam_interpolation",
            "h3_layers": action_initialization.h3_layers,
            "source_layer_indices": list(
                action_initialization.source_layer_indices
            ),
            "copied_tensors": action_initialization.copied_tensors,
            "resized_tensors": action_initialization.resized_tensors,
            "zeroed_biases": action_initialization.zeroed_biases,
            "residual_output_scale": (
                action_initialization.residual_output_scale
            ),
            "alpha_scaling": action_initialization.alpha_scaling,
        }
    )

    if stage_payload is not None:
        action_initialization_metadata = stage_payload.get(
            "action_initialization"
        )
        model.action_head.load_state_dict(stage_payload["action_head"], strict=True)
        model.kv_fusion.load_state_dict(stage_payload["kv_fusion"], strict=True)
        model.state_embedding.load_state_dict(
            stage_payload["state_embedding"], strict=True
        )
        del stage_payload

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    action_count = sum(parameter.numel() for parameter in model.action_head.parameters())
    fusion_count = sum(parameter.numel() for parameter in model.kv_fusion.parameters())
    # Match the proven H3-DreamWAM loading layout: move the comparatively
    # small, unwrapped H3 I/O modules to their rank-local GPU before FSDP
    # materializes the 50 large transformer blocks. Without this, eight ranks
    # retain redundant CPU copies long enough to exceed the pod memory cgroup.
    # Keep the small H3 I/O modules replicated, matching serving. When they are
    # trainable their gradients are explicitly averaged below; sharding these
    # tensors would make the same-world-size checkpoint incompatible with the
    # inference layout, where H3 I/O is intentionally ignored by FSDP.
    ignored_modules = [
        module
        for module in model.h3.children()
        if len(list(module.parameters())) > 0
    ]
    for module in ignored_modules:
        module.to(device)
    model.train(not args.eval_only)
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

    if args.load_joint_stage is not None:
        load_joint_h3_shard(
            stage_dir=args.load_joint_stage,
            model=model,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    head_parameters: list[torch.nn.Parameter] = []
    h3_parameters: list[torch.nn.Parameter] = []
    h3_block_parameters: list[torch.nn.Parameter] = []
    replicated_h3_io_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_joint_h3_parameter(name):
            h3_parameters.append(parameter)
            if any(marker in name for marker in (".h3.proj_in.", ".h3.proj_out.")):
                replicated_h3_io_parameters.append(parameter)
            else:
                h3_block_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = None
    scheduler = None
    if not args.eval_only:
        parameter_groups: list[dict] = [
            {"params": head_parameters, "lr": args.learning_rate},
        ]
        if h3_block_parameters:
            parameter_groups.append(
                {"params": h3_block_parameters, "lr": args.h3_learning_rate}
            )
        if replicated_h3_io_parameters:
            parameter_groups.append(
                {
                    "params": replicated_h3_io_parameters,
                    "lr": h3_io_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            foreach=False,
        )
        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps, eta_min=0.0
            )

    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training manifest is empty")
    stats = torch.load(
        args.data_root.resolve() / "stats.pt",
        map_location="cpu",
        weights_only=False,
    )
    patch_size = tuple(h3.config.patch_size)

    def prepare_batch(
        row: dict,
        *,
        context_id_override: str | None = None,
        shared_noise_batch: dict | None = None,
    ) -> dict:
        sample_id = str(row["id"])
        context_id = str(
            context_id_override
            or args.context_override_id
            or row.get("context_id", sample_id)
        )
        window_path, context_path = resolve_sample(
            args.data_root.resolve(), sample_id, context_id
        )
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        conditioning = torch.load(
            context_path, map_location="cpu", weights_only=False
        )
        if args.require_text_only_context:
            if conditioning.get("text_only") is not True:
                raise ValueError(f"context {context_id} is not text-only")
            if torch.any(conditioning["token_tags"] != 1):
                raise ValueError(f"context {context_id} has non-text token tags")
        validate_cached_sample(window, conditioning)
        clean = window["video_latents"].to(device=device, dtype=torch.float32)
        first = window["first_frame_latents"].to(device=device, dtype=torch.float32)
        context = conditioning["context"].to(device=device, dtype=torch.float32)
        text_tags = conditioning["token_tags"].long()
        actions = normalize(
            window["actions"][: args.action_horizon],
            stats["action_min"],
            stats["action_max"],
        ).to(device)
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(args.action_horizon, dtype=torch.bool)
        )[: args.action_horizon].bool().to(device)
        state = normalize(
            window["state"], stats["state_min"], stats["state_max"]
        ).to(device)
        _, _, latent_frames, latent_height, latent_width = clean.shape
        latent_is_pad = window.get(
            "latent_is_pad", torch.zeros(latent_frames, dtype=torch.bool)
        ).bool()
        if tuple(latent_is_pad.shape) != (latent_frames,):
            raise ValueError(
                f"latent padding shape {tuple(latent_is_pad.shape)} != {(latent_frames,)}"
            )
        spatial_rows = (latent_height // patch_size[1]) * (
            latent_width // patch_size[2]
        )
        world_row_valid = (~latent_is_pad).repeat_interleave(spatial_rows).to(device)
        pixel_frames = int(window.get("h3_frame_count", latent_frames))
        num_audio_latents = audio_latent_count(pixel_frames)
        layout_text_tags = torch.cat(
            (text_tags, torch.ones(1, dtype=text_tags.dtype))
        )
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=layout_text_tags,
            num_latent_frames=latent_frames,
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
        if shared_noise_batch is None:
            video_time = shifted_video_timestep(generator=generator, device=device)
            video_noise = torch.randn(
                clean.shape, generator=generator, device=device, dtype=torch.float32
            )
            noisy_video = video_time * clean + (1.0 - video_time) * video_noise
            rgb_target = patchify_video_latents(clean - video_noise, patch_size)[None]
            flow_target = None
            noisy_flow = None
            if args.motion_root is not None:
                motion_path = args.motion_root.resolve() / f"{sample_id}.pt"
                motion = torch.load(
                    motion_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if motion.get("sample_id") not in (None, sample_id):
                    raise ValueError(f"motion sample mismatch for {sample_id}")
                flow_clean = motion["flow_latents"].to(
                    device=device,
                    dtype=torch.float32,
                )
                if flow_clean.shape != clean.shape:
                    raise ValueError(
                        f"flow/RGB shape mismatch for {sample_id}: "
                        f"{tuple(flow_clean.shape)} vs {tuple(clean.shape)}"
                    )
                flow_clean = normalize_world_latents(flow_clean)
                flow_noise = torch.randn(
                    flow_clean.shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                noisy_flow = video_time * flow_clean + (1.0 - video_time) * flow_noise
                flow_target = patchify_video_latents(
                    flow_clean - flow_noise,
                    patch_size,
                )[None]
            first_noise = torch.randn(
                first.shape, generator=generator, device=device, dtype=torch.float32
            )
            noisy_first = (
                KEYFRAME_TIMESTEP * first
                + (1.0 - KEYFRAME_TIMESTEP) * first_noise
            )
            rgb_rows = torch.cat(
                (
                    patchify_video_latents(noisy_first, patch_size),
                    patchify_video_latents(noisy_video, patch_size),
                ),
                dim=0,
            )[None]
            if noisy_flow is None:
                flow_rows = torch.zeros_like(rgb_rows)
            else:
                condition_flow_rows = torch.zeros_like(
                    patchify_video_latents(first, patch_size)
                )
                flow_rows = torch.cat(
                    (
                        condition_flow_rows,
                        patchify_video_latents(noisy_flow, patch_size),
                    ),
                    dim=0,
                )[None]
            video_rows = torch.cat((rgb_rows, flow_rows), dim=-1)
            audio_rows = torch.randn(
                (1, num_audio_latents * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            video_time = shared_noise_batch["video_time"]
            video_rows = shared_noise_batch["video_rows"]
            audio_rows = shared_noise_batch["audio_rows"]
            rgb_target = shared_noise_batch["rgb_target"]
            flow_target = shared_noise_batch["flow_target"]
        unique_times, row_time_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=video_indices.cpu(),
            audio_indices=audio_indices.cpu(),
            num_condition_video_rows=num_condition_video_rows,
            num_condition_audio_rows=num_condition_audio_rows,
            num_text_tokens=text_indices.numel(),
            video_timestep=float(video_time),
            audio_timestep=0.0,
            condition_video_timestep=max(float(video_time), KEYFRAME_TIMESTEP),
            condition_audio_timestep=1.0,
        )
        condition_video_indices = video_indices[:num_condition_video_rows]
        h3_mask = build_h3_observation_attention_mask(
            sequence_length=position_ids.shape[0],
            text_indices=text_indices,
            condition_video_indices=condition_video_indices,
            device=device,
        )
        if shared_noise_batch is None:
            uniform = torch.rand(1, generator=generator, device=device)
            action_sigma = 5.0 * uniform / (1.0 + 4.0 * uniform)
            action_noise = torch.randn(
                (1, *actions.shape),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            noisy_actions = (1.0 - action_sigma[:, None, None]) * actions[None]
            noisy_actions = noisy_actions + action_sigma[:, None, None] * action_noise
            action_target = action_noise - actions[None]
            clean_actions = actions[None]
        else:
            action_sigma = shared_noise_batch["action_timestep"] / 1000.0
            noisy_actions = shared_noise_batch["noisy_actions"]
            action_target = shared_noise_batch["action_target"]
            clean_actions = shared_noise_batch["clean_actions"]
        return {
            "sample": window_path.stem,
            "context_id": context_id,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "video_time": video_time,
            "video_timestep": video_time.reshape(1) * 1000.0,
            "context": context,
            "unique_times": unique_times.to(device),
            "row_time_indices": row_time_indices.to(device),
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "condition_video_indices": condition_video_indices,
            "noisy_actions": noisy_actions,
            "action_timestep": action_sigma * 1000.0,
            "state": state[None],
            "context_mask": torch.ones(
                context.shape[:2], device=device, dtype=torch.bool
            ),
            "h3_mask": h3_mask,
            "rgb_target": rgb_target,
            "flow_target": flow_target,
            "action_target": action_target,
            "action_valid": (~action_is_pad)[None],
            "world_row_valid": world_row_valid[None],
            "clean_actions": clean_actions,
        }

    def save_stage(path: Path, completed_steps: int) -> None:
        checkpoint_payload = None
        with FSDP.summon_full_params(
            model,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        ):
            if rank == 0:
                checkpoint_payload = {
                    "format": "h3dotwam_stage_v2",
                    "architecture": {
                        "video_layers": 50,
                        "action_layers": action_layers,
                        "hidden_dim": 1024,
                        "video_num_heads": 56,
                        "video_head_dim": 128,
                        "action_num_heads": 24,
                        "action_head_dim": 128,
                    },
                    "action_initialization": action_initialization_metadata,
                    "action_head": clone_state(model.module.action_head),
                    "kv_fusion": clone_state(model.module.kv_fusion),
                    "state_embedding": clone_state(model.module.state_embedding),
                    "steps": completed_steps,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "samples_seen": (
                        completed_steps
                        * args.gradient_accumulation_steps
                        * world_size
                    ),
                }
        if rank == 0:
            path.resolve().parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload, path.resolve())

    def save_joint_stage(path: Path, completed_steps: int) -> None:
        path = path.resolve()
        if rank == 0:
            path.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        save_stage(path / "action_stage.pt", completed_steps)
        save_joint_h3_shards(
            stage_dir=path,
            model=model,
            rank=rank,
            world_size=world_size,
            steps=completed_steps,
            last_h3_blocks=args.last_h3_blocks,
        )

    def prune_periodic_joint_stages() -> None:
        if rank != 0 or args.keep_last_joint_checkpoints == 0:
            return
        assert args.save_joint_stage is not None
        base = args.save_joint_stage.resolve()
        candidates = sorted(
            base.parent.glob(f"{base.name}_step*"),
            key=lambda path: path.name,
        )
        for expired in candidates[: -args.keep_last_joint_checkpoints]:
            if expired.is_dir() and (expired / "joint_stage.json").is_file():
                shutil.rmtree(expired)

    history: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)
    inference_schedule = (
        build_h3dream_inference_schedule(
            args.sample_steps,
            device=device,
        )
        if args.eval_only
        else None
    )
    for step in range(1, args.steps + 1):
        if args.eval_only:
            row_index = args.sample_offset + (step - 1) * world_size + rank
            row = rows[row_index % len(rows)]
            batch = prepare_batch(row)
            forward_kwargs = {
                "video_rows": batch["video_rows"],
                "audio_rows": batch["audio_rows"],
                "context": batch["context"],
                "timestep": batch["unique_times"],
                "timestep_indices": batch["row_time_indices"],
                "token_tags": batch["token_tags"],
                "position_ids": batch["position_ids"],
                "video_indices": batch["video_indices"],
                "audio_indices": batch["audio_indices"],
                "text_indices": batch["text_indices"],
                "condition_video_indices": batch["condition_video_indices"],
                "state": batch["state"],
                "context_mask": batch["context_mask"],
                "h3_attention_mask": batch["h3_mask"],
            }
            assert inference_schedule is not None
            sampled_actions = torch.randn(
                batch["clean_actions"].shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                docked_keys = None
                docked_values = None
                for sigma, delta in zip(
                    inference_schedule.action_sigmas,
                    inference_schedule.action_sigma_deltas,
                    strict=True,
                ):
                    output = model(
                        **forward_kwargs,
                        noisy_actions=sampled_actions,
                        action_timestep=sigma.reshape(1) * 1000.0,
                        cached_docked_keys=docked_keys,
                        cached_docked_values=docked_values,
                    )
                    docked_keys = output.docked_keys
                    docked_values = output.docked_values
                    if docked_keys is None or docked_values is None:
                        raise RuntimeError("DoT forward did not return its docking cache")
                    sampled_actions += output.action_velocity.float() * delta
            action_loss = masked_token_mse(
                sampled_actions,
                batch["clean_actions"].float(),
                batch["action_valid"],
            )
            video_loss = action_loss.new_zeros(())
            flow_loss = action_loss.new_zeros(())
            wrong_action_loss = action_loss.new_zeros(())
            language_ranking_loss = action_loss.new_zeros(())
            language_ranking_count = action_loss.new_zeros(())
            loss = action_loss
            grad_norm = action_loss.new_zeros(())
            metric_sums = torch.stack(
                (
                    loss.detach(),
                    action_loss.detach(),
                    video_loss.detach(),
                    flow_loss.detach(),
                    wrong_action_loss.detach(),
                    language_ranking_loss.detach(),
                    language_ranking_count.detach(),
                )
            ).double()
        else:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            metric_sums = torch.zeros(7, device=device, dtype=torch.float64)
            for micro_step in range(args.gradient_accumulation_steps):
                micro_index = (
                    (step - 1) * args.gradient_accumulation_steps + micro_step
                )
                row_index = args.sample_offset + micro_index * world_size + rank
                row = rows[row_index % len(rows)]
                batch = prepare_batch(row)
                forward_kwargs = {
                    "video_rows": batch["video_rows"],
                    "audio_rows": batch["audio_rows"],
                    "context": batch["context"],
                    "timestep": batch["unique_times"],
                    "timestep_indices": batch["row_time_indices"],
                    "token_tags": batch["token_tags"],
                    "position_ids": batch["position_ids"],
                    "video_indices": batch["video_indices"],
                    "audio_indices": batch["audio_indices"],
                    "text_indices": batch["text_indices"],
                    "condition_video_indices": batch["condition_video_indices"],
                    "state": batch["state"],
                    "context_mask": batch["context_mask"],
                    "h3_attention_mask": batch["h3_mask"],
                }
                # Synchronize every microbatch. FSDP no_sync retains full
                # unsharded gradients and becomes unsafe once the H3 hub is
                # unfrozen for the paper-matched joint stage.
                output = model(
                    **forward_kwargs,
                    noisy_actions=batch["noisy_actions"],
                    action_timestep=batch["action_timestep"],
                )
                video_loss = masked_token_mse(
                    output.rgb_velocity_rows[
                        :, batch["condition_video_indices"].numel() :
                    ].float(),
                    batch["rgb_target"].float(),
                    batch["world_row_valid"],
                )
                action_loss = masked_token_mse(
                    output.action_velocity.float(),
                    batch["action_target"].float(),
                    batch["action_valid"],
                )
                if batch["flow_target"] is None:
                    flow_loss = video_loss.new_zeros(())
                else:
                    flow_loss = masked_token_mse(
                        output.flow_velocity_rows[
                            :, batch["condition_video_indices"].numel() :
                        ].float(),
                        batch["flow_target"].float(),
                        batch["world_row_valid"],
                    )
                action_weight = h3dream_flow_training_weight(
                    batch["action_timestep"]
                ).mean()
                loss = action_loss * action_weight
                if h3_parameters:
                    world_weight = (
                        h3dream_flow_training_weight(
                            batch["video_timestep"],
                            shift=12.0,
                        ).mean()
                        if args.dreamwam_world_weighting
                        else video_loss.new_ones(())
                    )
                    loss = loss + args.video_loss_weight * world_weight * (
                        video_loss + args.flow_loss_weight * flow_loss
                    )
                (loss / args.gradient_accumulation_steps).backward()
                action_loss_value = action_loss.detach()
                video_loss_value = video_loss.detach()
                flow_loss_value = flow_loss.detach()
                base_loss_value = loss.detach()
                wrong_action_loss = action_loss_value.new_zeros(())
                language_ranking_loss = action_loss_value.new_zeros(())
                language_ranking_count = action_loss_value.new_zeros(())
                del output, action_loss, video_loss, flow_loss, loss

                if (
                    args.language_ranking_weight > 0
                    and micro_step % args.language_ranking_every == 0
                ):
                    negative_context_id, _ = select_negative_context(rows, row_index)
                    negative_batch = prepare_batch(
                        row,
                        context_id_override=negative_context_id,
                        shared_noise_batch=batch,
                    )
                    negative_forward_kwargs = {
                        "video_rows": negative_batch["video_rows"],
                        "audio_rows": negative_batch["audio_rows"],
                        "context": negative_batch["context"],
                        "timestep": negative_batch["unique_times"],
                        "timestep_indices": negative_batch["row_time_indices"],
                        "token_tags": negative_batch["token_tags"],
                        "position_ids": negative_batch["position_ids"],
                        "video_indices": negative_batch["video_indices"],
                        "audio_indices": negative_batch["audio_indices"],
                        "text_indices": negative_batch["text_indices"],
                        "condition_video_indices": negative_batch[
                            "condition_video_indices"
                        ],
                        "state": negative_batch["state"],
                        "context_mask": negative_batch["context_mask"],
                        "h3_attention_mask": negative_batch["h3_mask"],
                    }
                    negative_output = model(
                        **negative_forward_kwargs,
                        noisy_actions=negative_batch["noisy_actions"],
                        action_timestep=negative_batch["action_timestep"],
                    )
                    wrong_action_loss = masked_token_mse(
                        negative_output.action_velocity.float(),
                        negative_batch["action_target"].float(),
                        negative_batch["action_valid"],
                    )
                    language_ranking_loss = action_language_ranking_loss(
                        action_loss_value,
                        wrong_action_loss,
                        args.language_ranking_margin,
                    )
                    language_ranking_count = action_loss_value.new_ones(())
                    (
                        args.language_ranking_weight
                        * language_ranking_loss
                        / args.gradient_accumulation_steps
                    ).backward()
                    del negative_output, negative_batch

                combined_loss_value = (
                    base_loss_value
                    + args.language_ranking_weight * language_ranking_loss.detach()
                )
                metric_sums += torch.stack(
                    (
                        combined_loss_value,
                        action_loss_value,
                        video_loss_value,
                        flow_loss_value,
                        wrong_action_loss.detach(),
                        language_ranking_loss.detach(),
                        language_ranking_count.detach(),
                    )
                ).double()
            for parameter in replicated_h3_io_parameters:
                if parameter.grad is None:
                    raise RuntimeError("trainable replicated H3 I/O has no gradient")
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)
            grad_norm = model.clip_grad_norm_(1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            metric_sums /= args.gradient_accumulation_steps
        metrics = torch.cat(
            (metric_sums, grad_norm.detach().reshape(1).double())
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device) / 2**30,
                torch.cuda.max_memory_reserved(device) / 2**30,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        ranking_fraction = float(metrics[6])
        ranking_denominator = max(ranking_fraction, 1.0e-12)
        record = {
            "step": step,
            "loss": float(metrics[0]),
            "action_loss": float(metrics[1]),
            "video_loss": float(metrics[2]),
            "flow_loss": float(metrics[3]),
            "wrong_action_loss": float(metrics[4]) / ranking_denominator,
            "language_ranking_loss": float(metrics[5]) / ranking_denominator,
            "language_ranking_fraction": ranking_fraction,
            "gradient_norm": float(metrics[7]),
            "peak_allocated_gib": float(memory[0]),
            "peak_reserved_gib": float(memory[1]),
            "sample": batch["sample"],
            "learning_rate": (
                0.0 if optimizer is None else float(optimizer.param_groups[0]["lr"])
            ),
        }
        if args.eval_only and args.record_sampled_actions:
            record["sampled_actions"] = sampled_actions[0].float().cpu().tolist()
        history.append(record)
        if rank == 0 and step % args.log_every == 0:
            event = "h3dotwam_eval" if args.eval_only else "h3dotwam_step"
            print(json.dumps({"event": event, **record}), flush=True)
        if (
            args.save_stage is not None
            and args.checkpoint_every
            and step % args.checkpoint_every == 0
            and step != args.steps
        ):
            milestone = args.save_stage.with_name(
                f"{args.save_stage.stem}_step{step:06d}{args.save_stage.suffix}"
            )
            save_stage(milestone, step)
        if (
            args.save_joint_stage is not None
            and args.joint_checkpoint_every
            and step % args.joint_checkpoint_every == 0
            and step != args.steps
        ):
            milestone = args.save_joint_stage.with_name(
                f"{args.save_joint_stage.name}_step{step:06d}"
            )
            save_joint_stage(milestone, step)
            prune_periodic_joint_stages()
            dist.barrier()

    if args.save_stage is not None:
        save_stage(args.save_stage, args.steps)
    if args.save_joint_stage is not None:
        save_joint_stage(args.save_joint_stage, args.steps)

    if rank == 0:
        report = {
            "event": "h3dotwam_training",
            "model": str(args.model.resolve()),
            "manifest": str(args.manifest.resolve()),
            "world_size": world_size,
            "steps": args.steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": world_size * args.gradient_accumulation_steps,
            "samples_seen": (
                args.steps * args.gradient_accumulation_steps * world_size
            ),
            "sample_offset": args.sample_offset,
            "lr_schedule": args.lr_schedule,
            "eval_only": args.eval_only,
            "sample_steps": args.sample_steps,
            "action_horizon": args.action_horizon,
            "action_layers": action_layers,
            "action_initialization": action_initialization_metadata,
            "learning_rate": args.learning_rate,
            "h3_learning_rate": args.h3_learning_rate,
            "h3_io_learning_rate": h3_io_learning_rate,
            "flow_channel_init_scale": flow_channel_init_scale,
            "last_h3_blocks": args.last_h3_blocks,
            "video_loss_weight": args.video_loss_weight,
            "motion_root": (
                None if args.motion_root is None else str(args.motion_root.resolve())
            ),
            "flow_loss_weight": args.flow_loss_weight,
            "train_h3_io": args.train_h3_io,
            "dreamwam_world_weighting": args.dreamwam_world_weighting,
            "language_ranking_weight": args.language_ranking_weight,
            "language_ranking_margin": args.language_ranking_margin,
            "language_ranking_every": args.language_ranking_every,
            "loaded_joint_stage": (
                None
                if args.load_joint_stage is None
                else str(args.load_joint_stage.resolve())
            ),
            "saved_joint_stage": (
                None
                if args.save_joint_stage is None
                else str(args.save_joint_stage.resolve())
            ),
            "trainable_parameters": trainable_count,
            "action_head_parameters": action_count,
            "kv_fusion_parameters": fusion_count,
            "load_and_train_seconds": time.perf_counter() - started,
            "first": history[0],
            "last": history[-1],
            "mean_action_loss": sum(row["action_loss"] for row in history)
            / len(history),
            "mean_flow_loss": sum(row["flow_loss"] for row in history)
            / len(history),
            "history": history,
        }
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in report.items() if key != "history"}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
