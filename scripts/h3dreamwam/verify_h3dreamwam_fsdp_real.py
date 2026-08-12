#!/usr/bin/env python3
"""One-window, one-update FSDP smoke for the real 33B MiniMax-H3."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
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
        help="Optional directory of DreamWAM RAFT/H3-VAE flow-latent artifacts.",
    )
    parser.add_argument(
        "--flow-loss-weight",
        type=float,
        default=0.5,
        help="DreamWAM uses 0.5 for motion flow-matching loss.",
    )
    parser.add_argument(
        "--train-h3-io",
        action="store_true",
        help="Train expanded H3 RGB/flow input and output projections.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSONL manifest; each rank selects a different cached window.",
    )
    parser.add_argument("--task", help="Exact task string used to filter --manifest.")
    parser.add_argument(
        "--rotate-manifest",
        action="store_true",
        help="Select a new rank-disjoint manifest window on every optimizer step.",
    )
    parser.add_argument("--last-h3-blocks", type=int, default=2)
    parser.add_argument(
        "--action-train-stage",
        choices=("head", "io", "tail", "tail_sharded", "full"),
        default="full",
        help="Staged ActionDiT warmup: output head, input/output, or all tensors.",
    )
    parser.add_argument("--last-action-blocks", type=int, default=8)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=32,
        help="Train on the first N actions; use 8 -> 16 -> 32 for curriculum.",
    )
    parser.add_argument(
        "--train-video-residual-gates",
        action="store_true",
        help=(
            "Optimize zero-initialized per-head video residual gates in the "
            "selected ActionDiT tail, including during --freeze-action-body warmup."
        ),
    )
    parser.add_argument(
        "--train-video-residual-adapters",
        action="store_true",
        help="Train zero-output low-rank video adapters in the selected ActionDiT tail.",
    )
    parser.add_argument(
        "--train-cross-attention-output",
        action="store_true",
        help="Warm up zero-initialized language cross-attention output projections.",
    )
    parser.add_argument(
        "--freeze-action-body",
        action="store_true",
        help=(
            "With tail_sharded, keep Action blocks require-grad for efficient "
            "FSDP layout but discard their gradients before clipping/update."
        ),
    )
    parser.add_argument(
        "--freeze-shared-state",
        action="store_true",
        help=(
            "With --freeze-action-body, also keep the shared state embedding "
            "fixed. This makes a gate warmup update only the action output and "
            "the explicitly selected video residual gates."
        ),
    )
    parser.add_argument(
        "--freeze-action-output",
        action="store_true",
        help="With --freeze-action-body, keep the inherited action output head fixed.",
    )
    parser.add_argument(
        "--separate-expert-clipping",
        action="store_true",
        help=(
            "Clip H3 and ActionDiT optimizer groups independently so a large "
            "new ActionDiT gradient cannot suppress world-model updates."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--h3-learning-rate",
        type=float,
        help="Optional separate LR for H3 flow I/O and unfrozen H3 tail blocks.",
    )
    parser.add_argument(
        "--new-layer-lr-scale",
        type=float,
        default=1.0,
        help="LR multiplier for newly unfrozen ActionDiT layers absent from the loaded stage.",
    )
    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        help="Optional LR for MiniWorld-style video residual gates.",
    )
    parser.add_argument(
        "--adapter-learning-rate",
        type=float,
        help="Optional LR for zero-initialized low-rank video residual adapters.",
    )
    parser.add_argument(
        "--cross-attention-learning-rate",
        type=float,
        help="Optional LR for language cross-attention output projections.",
    )
    parser.add_argument(
        "--tail-learning-rate",
        type=float,
        help="Optional absolute LR for all selected ActionDiT tail tensors.",
    )
    parser.add_argument(
        "--no-fp32-master",
        action="store_true",
        help="Diagnostic only; direct BF16 Adam updates are known to be unstable.",
    )
    parser.add_argument(
        "--bf16-model-storage",
        action="store_true",
        help="Diagnostic only; stable training keeps FSDP parameters in FP32.",
    )
    parser.add_argument("--load-action-head", type=Path)
    parser.add_argument("--save-action-head", type=Path)
    parser.add_argument("--load-action-stage", type=Path)
    parser.add_argument(
        "--override-action-io",
        type=Path,
        help="Eval ablation: restore Action I/O from a second stage checkpoint.",
    )
    parser.add_argument(
        "--disable-video-residual-adapters",
        action="store_true",
        help="Eval ablation: zero all low-rank video adapter outputs after loading.",
    )
    parser.add_argument("--save-action-stage", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--joint-sample-steps", type=int, default=0)
    parser.add_argument(
        "--dreamwam-action-weighting",
        action="store_true",
        help="Use DreamWAM's normalized mid-timestep action loss weighting.",
    )
    parser.add_argument(
        "--dreamwam-world-weighting",
        action="store_true",
        help="Apply DreamWAM's normalized timestep weight to RGB and flow losses.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Repeat the fixed real window to verify short-run optimization stability.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--require-text-only-context",
        action="store_true",
        help=(
            "Reject cached conditioning that contains image tokens. DreamWAM keeps "
            "the observation image exclusively in the video branch."
        ),
    )
    parser.add_argument(
        "--dreamwam-exact-action-norm",
        action="store_true",
        help="Use FastWAM/DreamWAM full-attention-width Q/K RMSNorm.",
    )
    parser.add_argument(
        "--action-init-alpha-scaling",
        action="store_true",
        help="Apply FastWAM alpha=sqrt(source_width/target_width) after interpolation.",
    )
    return parser.parse_args()


def normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scale = (high.float() - low.float()).clamp_min(1.0e-6)
    return ((value.float() - low.float()) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def normalize_world_latents(value: torch.Tensor) -> torch.Tensor:
    """DreamWAM per-sample normalization for flow/geometry teacher latents."""

    if value.ndim != 5:
        raise ValueError("world latents must be [B,C,T,H,W]")
    dimensions = tuple(range(1, value.ndim))
    mean = value.mean(dim=dimensions, keepdim=True)
    std = value.std(dim=dimensions, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (value - mean) / std


def gradient_norm_groups(model: torch.nn.Module, device: torch.device) -> dict[str, float]:
    """Report unscaled FSDP gradient norms by parameter role on the first step."""

    names = (
        "h3",
        "video_gate",
        "modulation",
        "action_time",
        "action_output",
        "action_other",
        "other",
    )
    squared = torch.zeros(len(names), device=device, dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if ".h3." in name or ".h3_block." in name:
            group = "h3"
        elif name.endswith(".action_block.video_residual_gate"):
            group = "video_gate"
        elif ".modulation" in name:
            group = "modulation"
        elif any(
            marker in name
            for marker in (
                ".action_expert.time_embedding.",
                ".action_expert.time_projection.",
            )
        ):
            group = "action_time"
        elif ".action_expert.output" in name:
            group = "action_output"
        elif ".action_expert." in name or ".action_block." in name:
            group = "action_other"
        else:
            group = "other"
        squared[names.index(group)] += parameter.grad.detach().double().square().sum()
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return {name: float(value.sqrt()) for name, value in zip(names, squared)}


def global_parameter_count(
    parameters: list[torch.nn.Parameter], device: torch.device
) -> int:
    count = torch.tensor(
        sum(parameter.numel() for parameter in parameters),
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count)


def video_gate_summary(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Summarize sharded MiniWorld-style gates without gathering the model."""

    total_abs = torch.zeros((), device=device, dtype=torch.float64)
    maximum = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if not name.endswith(".action_block.video_residual_gate"):
            continue
        values = parameter.detach().double().abs()
        total_abs += values.sum()
        if values.numel():
            maximum = torch.maximum(maximum, values.max())
            count += values.numel()
    dist.all_reduce(total_abs, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return {
        "mean_abs": float(total_abs / count.clamp_min(1.0)),
        "max_abs": float(maximum),
        "count": int(count),
    }


def main() -> None:
    args = parse_args()
    # FSDP emits one multi-line warning per paired layer when an Action block is
    # trainable beside a frozen H3 block. The layout is intentional and the
    # parameter report below records the exact optimized subset.
    warnings.filterwarnings(
        "ignore",
        message=r".*has both parameters with requires_grad=True and False.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Mismatch dtype between input and weight.*",
    )
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.flow_loss_weight < 0:
        raise ValueError("flow-loss-weight must be non-negative")
    if args.h3_learning_rate is not None and args.h3_learning_rate <= 0:
        raise ValueError("h3-learning-rate must be positive")
    if args.gate_learning_rate is not None and args.gate_learning_rate <= 0:
        raise ValueError("gate-learning-rate must be positive")
    if args.adapter_learning_rate is not None and args.adapter_learning_rate <= 0:
        raise ValueError("adapter-learning-rate must be positive")
    if (
        args.cross_attention_learning_rate is not None
        and args.cross_attention_learning_rate <= 0
    ):
        raise ValueError("cross-attention-learning-rate must be positive")
    if args.tail_learning_rate is not None and args.tail_learning_rate <= 0:
        raise ValueError("tail-learning-rate must be positive")
    if not 1 <= args.action_horizon <= 32:
        raise ValueError("action-horizon must be in [1,32]")
    if args.freeze_action_body and args.action_train_stage != "tail_sharded":
        raise ValueError("freeze-action-body requires tail_sharded")
    if args.freeze_shared_state and not args.freeze_action_body:
        raise ValueError("freeze-shared-state requires freeze-action-body")
    if args.freeze_action_output and not args.freeze_action_body:
        raise ValueError("freeze-action-output requires freeze-action-body")
    if args.separate_expert_clipping and args.action_train_stage != "tail_sharded":
        raise ValueError("separate-expert-clipping requires tail_sharded")
    if (
        args.train_video_residual_gates
        and args.action_train_stage != "tail_sharded"
    ):
        raise ValueError("train-video-residual-gates requires tail_sharded")
    if (
        args.train_video_residual_adapters
        and args.action_train_stage != "tail_sharded"
    ):
        raise ValueError("train-video-residual-adapters requires tail_sharded")
    if (
        args.train_cross_attention_output
        and args.action_train_stage != "tail_sharded"
    ):
        raise ValueError("train-cross-attention-output requires tail_sharded")
    if args.motion_root is None and args.flow_loss_weight != 0.5:
        raise ValueError("flow-loss-weight has no effect without --motion-root")
    if args.eval_only and (args.save_action_head or args.save_action_stage):
        raise ValueError("eval-only cannot save training checkpoints")
    if (args.override_action_io or args.disable_video_residual_adapters) and not args.eval_only:
        raise ValueError("checkpoint ablations require eval-only")
    if args.override_action_io is not None and args.load_action_stage is None:
        raise ValueError("override-action-io requires load-action-stage")
    if args.joint_sample_steps < 0 or (
        args.joint_sample_steps and not args.eval_only
    ):
        raise ValueError("joint-sample-steps must be non-negative and requires eval-only")
    if not 0.0 < args.new_layer_lr_scale <= 1.0:
        raise ValueError("new-layer-lr-scale must be in (0,1]")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or not 0 <= args.last_h3_blocks <= 50:
        raise ValueError("use multi-GPU and last-h3-blocks in [0,50]")
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
        H3DreamActionExpert,
        H3DreamPairedLayer,
        H3DreamWAM,
        expand_h3_rgb_flow_projections,
        initialize_action_expert_from_h3,
        load_action_block_state,
        build_h3dream_inference_schedule,
        h3dream_flow_training_weight,
        sample_h3dream_joint_rows,
    )
    from fastwam.models.h3wam import build_h3_observation_attention_mask
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    load_started = time.perf_counter()
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
    projection_report = expand_h3_rgb_flow_projections(h3)
    if args.train_h3_io:
        h3.proj_in.requires_grad_(True)
        h3.proj_out.requires_grad_(True)
    model_storage_dtype = (
        torch.bfloat16 if args.bf16_model_storage else torch.float32
    )

    previous_dtype = torch.get_default_dtype()
    # Construct on CPU in BF16 so eight ranks do not each hold a 145 GB FP32
    # model. Stable FP32 storage is applied after FSDP has sharded parameters.
    action_storage_dtype = torch.bfloat16
    torch.set_default_dtype(action_storage_dtype)
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
    initialization_started = time.perf_counter()
    initialization_report = initialize_action_expert_from_h3(
        action_expert,
        h3,
        alpha_scaling=args.action_init_alpha_scaling,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    if args.load_action_head is not None:
        head_payload = torch.load(
            args.load_action_head.resolve(), map_location="cpu", weights_only=True
        )
        if head_payload.get("format") != "h3dreamwam_action_head_v1":
            raise ValueError("action head checkpoint has an incompatible format")
        action_expert.output.load_state_dict(head_payload["state_dict"], strict=True)
    loaded_stage_layers: set[int] = set()
    loaded_h3_layers: set[int] = set()
    migrated_legacy_action_blocks = 0
    stage_payload = None
    override_io_payload = None
    if args.load_action_stage is not None:
        stage_payload = torch.load(
            args.load_action_stage.resolve(), map_location="cpu", weights_only=True
        )
        if stage_payload.get("format") != "h3dreamwam_action_stage_v1":
            raise ValueError("action stage checkpoint has an incompatible format")
        stage_architecture = stage_payload.get("architecture", {})
        expected_architecture = {
            "full_width_rmsnorm": args.dreamwam_exact_action_norm,
            "alpha_scaling": args.action_init_alpha_scaling,
            "video_residual_gate": True,
            "video_residual_adapter_rank": 16,
        }
        for key, value in stage_architecture.items():
            if key not in expected_architecture or expected_architecture[key] != value:
                raise ValueError(
                    "action stage architecture mismatch: "
                    f"checkpoint={stage_architecture}, requested={expected_architecture}"
                )
        io_modules = {
            "action_embedding": action_expert.action_embedding,
            "state_embedding": action_expert.state_embedding,
            "context_embedding": action_expert.context_embedding,
            "time_embedding": action_expert.time_embedding,
            "time_projection": action_expert.time_projection,
            "output": action_expert.output,
        }
        for name, module in io_modules.items():
            module.load_state_dict(stage_payload["io"][name], strict=True)
        for index_text, state_dict in stage_payload["blocks"].items():
            layer_index = int(index_text)
            migrated_legacy_action_blocks += int(
                load_action_block_state(action_expert.blocks[layer_index], state_dict)
            )
            loaded_stage_layers.add(layer_index)
        loaded_h3_layers = {
            int(index) for index in stage_payload.get("h3_blocks", {})
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
    if args.override_action_io is not None:
        override_io_payload = torch.load(
            args.override_action_io.resolve(), map_location="cpu", weights_only=True
        )
        if override_io_payload.get("format") != "h3dreamwam_action_stage_v1":
            raise ValueError("override Action I/O checkpoint has an incompatible format")
        for name, module in io_modules.items():
            module.load_state_dict(override_io_payload["io"][name], strict=True)
    if args.disable_video_residual_adapters:
        for block in action_expert.blocks:
            block.video_residual_adapter[-1].weight.data.zero_()
    if not 1 <= args.last_action_blocks <= 50:
        raise ValueError("last-action-blocks must be in [1,50]")
    if args.action_train_stage not in ("full", "tail_sharded"):
        action_expert.requires_grad_(False)
        action_expert.output.requires_grad_(True)
        if args.action_train_stage in ("io", "tail"):
            action_expert.action_embedding.requires_grad_(True)
        if args.action_train_stage == "tail":
            action_expert.state_embedding.requires_grad_(True)
            action_expert.context_embedding.requires_grad_(True)
            action_expert.time_embedding.requires_grad_(True)
            for block in action_expert.blocks[-args.last_action_blocks :]:
                block.requires_grad_(True)
                # H3 and DreamWAM AdaLN statistics are not structurally
                # equivalent. Keep gates fixed during the first tail stage;
                # unfreeze them only after the residual tensors calibrate.
                block.modulation.requires_grad_(False)
    model = H3DreamWAM(
        h3,
        action_expert,
        rgb_patch_width=projection_report.old_patch_width,
        use_gradient_checkpointing=True,
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
    full_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.train()
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
    if model_storage_dtype == torch.float32:
        model.float()
    # The construction path is intentionally BF16 to stay within host memory.
    # Reload stage tensors after FSDP FP32 materialization so small low-LR
    # updates are not rounded away during BF16 checkpoint restore.
    if stage_payload is not None and model_storage_dtype == torch.float32:
        with FSDP.summon_full_params(model, recurse=False, writeback=True):
            root_action = model.module.action_expert
            restored_io = (
                override_io_payload["io"]
                if override_io_payload is not None
                else stage_payload["io"]
            )
            for name, module in {
                "action_embedding": root_action.action_embedding,
                "state_embedding": root_action.state_embedding,
                "context_embedding": root_action.context_embedding,
                "time_embedding": root_action.time_embedding,
                "time_projection": root_action.time_projection,
                "output": root_action.output,
            }.items():
                module.load_state_dict(restored_io[name], strict=True)
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
                    if args.disable_video_residual_adapters:
                        paired_fsdp.module.action_block.video_residual_adapter[
                            -1
                        ].weight.data.zero_()
                if index_text in stage_payload.get("h3_blocks", {}):
                    paired_fsdp.module.h3_block.load_state_dict(
                        stage_payload["h3_blocks"][index_text],
                        strict=True,
                    )
    if args.eval_only:
        model.eval()
    load_seconds = time.perf_counter() - load_started
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimized_parameters = trainable
    new_layer_parameters: list[torch.nn.Parameter] = []
    tail_parameters: list[torch.nn.Parameter] = []
    gate_parameters: list[torch.nn.Parameter] = []
    adapter_parameters: list[torch.nn.Parameter] = []
    cross_attention_parameters: list[torch.nn.Parameter] = []
    h3_parameters: list[torch.nn.Parameter] = []
    zero_lr_parameters: list[torch.nn.Parameter] = []
    if args.action_train_stage == "tail_sharded":
        first_action_layer = (
            50 - args.last_action_blocks
            if args.action_train_stage in ("tail", "tail_sharded")
            else 50
        )
        first_h3_layer = 50 - args.last_h3_blocks
        optimized_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_h3_io = args.train_h3_io and any(
                marker in name for marker in (".h3.proj_in.", ".h3.proj_out.")
            )
            is_h3_tail = any(
                f".paired_layers.{index}." in name and ".h3_block." in name
                for index in range(first_h3_layer, 50)
            )
            is_shared_state = (
                args.freeze_action_body
                and not args.freeze_shared_state
                and ".action_expert.state_embedding." in name
            )
            is_io = any(
                marker in name
                for marker in (
                    ".action_expert.output.",
                    ".action_expert.action_embedding.",
                    ".action_expert.state_embedding.",
                    ".action_expert.context_embedding.",
                    ".action_expert.time_embedding.",
                    ".action_expert.time_projection.",
                )
            )
            if args.freeze_action_body:
                is_io = (
                    ".action_expert.output." in name
                    and not args.freeze_action_output
                )
            tail_index = next(
                (
                    index
                    for index in range(first_action_layer, 50)
                    if f".paired_layers.{index}." in name
                    and ".action_block." in name
                ),
                None,
            )
            is_video_gate = name.endswith(".action_block.video_residual_gate")
            is_video_adapter = ".action_block.video_residual_adapter." in name
            is_cross_attention_output = (
                ".action_block.cross_attn.to_out." in name
            )
            if is_h3_io or is_h3_tail or is_shared_state:
                h3_parameters.append(parameter)
            elif (
                is_video_gate
                and tail_index is not None
                and (
                    args.train_video_residual_gates
                    or not args.freeze_action_body
                )
            ):
                gate_parameters.append(parameter)
            elif (
                is_video_adapter
                and tail_index is not None
                and args.train_video_residual_adapters
            ):
                adapter_parameters.append(parameter)
            elif is_video_adapter:
                zero_lr_parameters.append(parameter)
            elif (
                is_cross_attention_output
                and tail_index is not None
                and args.train_cross_attention_output
            ):
                cross_attention_parameters.append(parameter)
            elif (
                tail_index is not None
                and not args.freeze_action_body
                and args.tail_learning_rate is not None
            ):
                tail_parameters.append(parameter)
            elif is_io or (
                tail_index in loaded_stage_layers and not args.freeze_action_body
            ):
                optimized_parameters.append(parameter)
            elif (
                tail_index is not None
                and not args.freeze_action_body
            ):
                new_layer_parameters.append(parameter)
            else:
                zero_lr_parameters.append(parameter)
        if not any(
            (
                optimized_parameters,
                new_layer_parameters,
                tail_parameters,
                gate_parameters,
                adapter_parameters,
                cross_attention_parameters,
                h3_parameters,
            )
        ):
            raise RuntimeError("tail_sharded optimizer selected no parameters")
    else:
        optimized_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".h3." in name or ".h3_block." in name:
                h3_parameters.append(parameter)
            else:
                optimized_parameters.append(parameter)
    master_parameters = (
        None
        if args.no_fp32_master or model_storage_dtype == torch.float32
        else [
            torch.nn.Parameter(parameter.detach().float(), requires_grad=True)
            for parameter in optimized_parameters
        ]
    )
    optimizer_parameters = (
        optimized_parameters if master_parameters is None else master_parameters
    )
    parameter_groups: list[dict] = [{"params": optimizer_parameters}]
    if new_layer_parameters:
        parameter_groups.append(
            {
                "params": new_layer_parameters,
                "lr": args.learning_rate * args.new_layer_lr_scale,
            }
        )
    if tail_parameters:
        parameter_groups.append(
            {
                "params": tail_parameters,
                "lr": args.tail_learning_rate,
            }
        )
    if gate_parameters:
        parameter_groups.append(
            {
                "params": gate_parameters,
                "lr": (
                    args.gate_learning_rate
                    if args.gate_learning_rate is not None
                    else args.learning_rate
                ),
                "weight_decay": 0.0,
            }
        )
    if adapter_parameters:
        parameter_groups.append(
            {
                "params": adapter_parameters,
                "lr": (
                    args.adapter_learning_rate
                    if args.adapter_learning_rate is not None
                    else args.learning_rate
                ),
                "weight_decay": 0.0,
            }
        )
    if cross_attention_parameters:
        parameter_groups.append(
            {
                "params": cross_attention_parameters,
                "lr": (
                    args.cross_attention_learning_rate
                    if args.cross_attention_learning_rate is not None
                    else args.learning_rate
                ),
                "weight_decay": 0.0,
            }
        )
    if h3_parameters:
        parameter_groups.append(
            {
                "params": h3_parameters,
                "lr": (
                    args.h3_learning_rate
                    if args.h3_learning_rate is not None
                    else args.learning_rate
                ),
            }
        )
    if zero_lr_parameters:
        parameter_groups.append({"params": zero_lr_parameters, "lr": 0.0})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    global_optimized_parameter_count = global_parameter_count(
        optimized_parameters, device
    )
    global_new_layer_parameter_count = global_parameter_count(
        new_layer_parameters, device
    )
    global_tail_parameter_count = global_parameter_count(tail_parameters, device)
    global_gate_parameter_count = global_parameter_count(gate_parameters, device)
    global_adapter_parameter_count = global_parameter_count(
        adapter_parameters, device
    )
    global_cross_attention_parameter_count = global_parameter_count(
        cross_attention_parameters, device
    )
    global_h3_parameter_count = global_parameter_count(h3_parameters, device)

    data_root = args.data_root.resolve()
    selected_sample_id = args.sample_id
    selected_context_id = None
    selected_rows = None
    if args.manifest is not None:
        selected_rows = [
            json.loads(line)
            for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.task is not None:
            selected_rows = [row for row in selected_rows if row.get("task") == args.task]
        if not selected_rows:
            raise ValueError("manifest/task selection produced no windows")
        selected_row = selected_rows[rank % len(selected_rows)]
        selected_sample_id = str(selected_row["id"])
        selected_context_id = str(selected_row.get("context_id", selected_row["id"]))
    elif args.task is not None:
        raise ValueError("--task requires --manifest")
    stats = torch.load(data_root / "stats.pt", map_location="cpu", weights_only=False)
    patch_size = tuple(h3.config.patch_size)

    def prepare_training_batch(
        sample_ref: tuple[str | None, str | None],
    ) -> dict[str, torch.Tensor | int | str]:
        sample_id, context_id = sample_ref
        window_path, context_path = resolve_sample(data_root, sample_id, context_id)
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        conditioning = torch.load(context_path, map_location="cpu", weights_only=False)
        if args.require_text_only_context:
            if conditioning.get("text_only") is not True:
                raise ValueError(
                    f"context {context_path} is not marked text_only=True"
                )
            if torch.any(conditioning["token_tags"] != 1):
                raise ValueError(
                    f"text-only context {context_path} contains non-text token tags"
                )
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
        state = normalize(
            window["state"], stats["state_min"], stats["state_max"]
        ).to(device)
        _, _, latent_frames, latent_height, latent_width = clean.shape
        pixel_frames = int(window.get("h3_frame_count", latent_frames))
        num_audio_latents = audio_latent_count(pixel_frames)
        # H3 packs context into self-attention rather than using Wan-style
        # cross-attention. Reserve one real packed text row for the shared
        # proprio token that H3DreamWAM appends to `context`.
        layout_text_tags = torch.cat(
            (text_tags, torch.ones(1, dtype=text_tags.dtype)),
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
        video_time = shifted_video_timestep(generator=generator, device=device)
        video_noise = torch.randn(
            clean.shape, generator=generator, device=device, dtype=torch.float32
        )
        noisy_video = video_time * clean + (1.0 - video_time) * video_noise
        rgb_target = patchify_video_latents(clean - video_noise, patch_size)[None]
        flow_target = None
        noisy_flow = None
        if args.motion_root is not None:
            motion_path = args.motion_root.resolve() / f"{window_path.stem}.pt"
            motion = torch.load(
                motion_path,
                map_location="cpu",
                weights_only=False,
            )
            flow_clean = motion["flow_latents"].to(
                device=device,
                dtype=torch.float32,
            )
            if flow_clean.shape != clean.shape:
                raise ValueError(
                    f"flow/RGB shape mismatch for {window_path.stem}: "
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
        noisy_first = KEYFRAME_TIMESTEP * first + (1.0 - KEYFRAME_TIMESTEP) * first_noise
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
        h3_mask = build_h3_observation_attention_mask(
            sequence_length=position_ids.shape[0],
            text_indices=text_indices,
            condition_video_indices=video_indices[:num_condition_video_rows],
            device=device,
        )
        action_uniform = torch.rand(1, generator=generator, device=device)
        action_sigma = 5.0 * action_uniform / (1.0 + 4.0 * action_uniform)
        action_noise = torch.randn(
            (1, *actions.shape), generator=generator, device=device, dtype=torch.float32
        )
        noisy_actions = (1.0 - action_sigma[:, None, None]) * actions[None]
        noisy_actions = noisy_actions + action_sigma[:, None, None] * action_noise
        clean_first_rows = patchify_video_latents(first, patch_size)[None]
        clean_future_rows = patchify_video_latents(clean, patch_size)[None]
        sampling_future_noise = torch.randn(
            clean.shape, generator=generator, device=device, dtype=torch.float32
        )
        return {
            "sample": window_path.stem,
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "context": context,
            "unique_times": unique_times.to(device),
            "row_time_indices": row_time_indices.to(device),
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "noisy_actions": noisy_actions,
            "action_timestep": action_sigma * 1000.0,
            "video_timestep": video_time.reshape(1) * 1000.0,
            "state": state[None],
            "context_mask": torch.ones(
                context.shape[:2], device=device, dtype=torch.bool
            ),
            "h3_mask": h3_mask,
            "num_condition_video_rows": num_condition_video_rows,
            "rgb_target": rgb_target,
            "flow_target": flow_target,
            "action_target": action_noise - actions[None],
            "clean_actions": actions[None],
            "clean_rgb_rows": torch.cat(
                (clean_first_rows, clean_future_rows), dim=1
            ),
            "initial_rgb_rows": torch.cat(
                (
                    clean_first_rows,
                    patchify_video_latents(sampling_future_noise, patch_size)[None],
                ),
                dim=1,
            ),
            "num_condition_audio_rows": num_condition_audio_rows,
        }

    if args.rotate_manifest and selected_rows is None:
        raise ValueError("--rotate-manifest requires --manifest")
    if args.rotate_manifest:
        sample_schedule = []
        for step in range(1, args.steps + 1):
            row = selected_rows[
                ((step - 1) * world_size + rank) % len(selected_rows)
            ]
            sample_schedule.append(
                (str(row["id"]), str(row.get("context_id", row["id"])))
            )
    else:
        sample_schedule = [(selected_sample_id, selected_context_id)] * args.steps

    torch.cuda.reset_peak_memory_stats(device)
    step_started = time.perf_counter()
    history = []
    first_gradient_groups = None
    first_expert_clip_norms = None
    for step in range(1, args.steps + 1):
        batch = prepare_training_batch(sample_schedule[step - 1])
        if not args.eval_only:
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
        if args.joint_sample_steps:
            schedule = build_h3dream_inference_schedule(
                args.joint_sample_steps, device=device
            )

            def predict_velocity(rgb_rows, actions, video_time, action_sigma):
                video_rows = torch.cat((rgb_rows, torch.zeros_like(rgb_rows)), dim=-1)
                unique_times, row_time_indices = (
                    MiniMaxH3SetTimestepsStep.build_row_timesteps(
                        video_indices=batch["video_indices"].cpu(),
                        audio_indices=batch["audio_indices"].cpu(),
                        num_condition_video_rows=int(batch["num_condition_video_rows"]),
                        num_condition_audio_rows=int(batch["num_condition_audio_rows"]),
                        num_text_tokens=batch["text_indices"].numel(),
                        video_timestep=float(video_time),
                        audio_timestep=0.0,
                        condition_video_timestep=max(
                            float(video_time), KEYFRAME_TIMESTEP
                        ),
                        condition_audio_timestep=1.0,
                    )
                )
                output = model(
                    video_rows=video_rows,
                    audio_rows=batch["audio_rows"],
                    context=batch["context"],
                    timestep=unique_times.to(device),
                    timestep_indices=row_time_indices.to(device),
                    token_tags=batch["token_tags"],
                    position_ids=batch["position_ids"],
                    video_indices=batch["video_indices"],
                    audio_indices=batch["audio_indices"],
                    text_indices=batch["text_indices"],
                    noisy_actions=actions,
                    action_timestep=action_sigma.reshape(1) * 1000.0,
                    state=batch["state"],
                    context_mask=batch["context_mask"],
                    action_video_indices=batch["video_indices"],
                    h3_attention_mask=batch["h3_mask"],
                )
                return output.rgb_velocity_rows, output.action_velocity

            sampled = sample_h3dream_joint_rows(
                predict_velocity,
                initial_video_rows=batch["initial_rgb_rows"],
                condition_video_rows=int(batch["num_condition_video_rows"]),
                initial_actions=torch.randn(
                    batch["clean_actions"].shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                ),
                schedule=schedule,
            )
            video_loss = F.mse_loss(
                sampled.video_rows[:, int(batch["num_condition_video_rows"]):].float(),
                batch["clean_rgb_rows"][:, int(batch["num_condition_video_rows"]):].float(),
            )
            action_loss = F.mse_loss(
                sampled.actions.float(), batch["clean_actions"].float()
            )
        else:
            with torch.set_grad_enabled(not args.eval_only):
                output = model(
                    video_rows=batch["video_rows"],
                    audio_rows=batch["audio_rows"],
                    context=batch["context"],
                    timestep=batch["unique_times"],
                    timestep_indices=batch["row_time_indices"],
                    token_tags=batch["token_tags"],
                    position_ids=batch["position_ids"],
                    video_indices=batch["video_indices"],
                    audio_indices=batch["audio_indices"],
                    text_indices=batch["text_indices"],
                    noisy_actions=batch["noisy_actions"],
                    action_timestep=batch["action_timestep"],
                    state=batch["state"],
                    context_mask=batch["context_mask"],
                    action_video_indices=batch["video_indices"],
                    h3_attention_mask=batch["h3_mask"],
                )
            predicted_rgb = output.rgb_velocity_rows[
                :, int(batch["num_condition_video_rows"]):
            ]
            video_loss = F.mse_loss(
                predicted_rgb.float(), batch["rgb_target"].float()
            )
            action_loss = F.mse_loss(
                output.action_velocity.float(), batch["action_target"].float()
            )
            if batch["flow_target"] is None:
                flow_loss = video_loss.new_zeros(())
            else:
                predicted_flow = output.flow_velocity_rows[
                    :, int(batch["num_condition_video_rows"]):
                ]
                flow_loss = F.mse_loss(
                    predicted_flow.float(), batch["flow_target"].float()
                )
        if args.joint_sample_steps:
            flow_loss = video_loss.new_zeros(())
        action_weight = (
            h3dream_flow_training_weight(batch["action_timestep"]).mean()
            if args.dreamwam_action_weighting and not args.joint_sample_steps
            else action_loss.new_ones(())
        )
        world_weight = (
            h3dream_flow_training_weight(
                batch["video_timestep"],
                shift=12.0,
            ).mean()
            if args.dreamwam_world_weighting and not args.joint_sample_steps
            else video_loss.new_ones(())
        )
        objective_action_loss = action_loss * action_weight
        loss = (
            world_weight * (video_loss + args.flow_loss_weight * flow_loss)
            + objective_action_loss
        )
        if args.eval_only:
            grad_norm = loss.new_zeros(())
        else:
            loss.backward()
            if step == 1:
                first_gradient_groups = gradient_norm_groups(model, device)
            # These tensors stay require-grad only because FSDP shards the
            # paired frozen/trainable layout much more efficiently this way.
            # Remove them before global clipping so the untrained Action body
            # cannot suppress the H3 world-model update.
            for parameter in zero_lr_parameters:
                parameter.grad = None
            if args.separate_expert_clipping:
                expert_groups = {
                    "h3": h3_parameters,
                    # Keep the inherited I/O, newly introduced modulation,
                    # large ActionDiT core and scalar gate independent. A tail
                    # norm in the 1e5 range otherwise suppresses the I/O and
                    # gate updates that are needed to adapt the new route.
                    "action_io": optimized_parameters,
                    "action_new": new_layer_parameters,
                    "action_tail": tail_parameters,
                    "video_gate": gate_parameters,
                    "video_adapter": adapter_parameters,
                    "cross_attention": cross_attention_parameters,
                }
                expert_norms: dict[str, torch.Tensor] = {}
                all_with_grad = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                for group_name, group in expert_groups.items():
                    if not group:
                        expert_norms[group_name] = torch.zeros(
                            (), device=device, dtype=torch.float32
                        )
                        continue
                    active_ids = {id(parameter) for parameter in group}
                    hidden_gradients = [
                        (parameter, parameter.grad)
                        for parameter in all_with_grad
                        if id(parameter) not in active_ids
                        and parameter.grad is not None
                    ]
                    for parameter, _ in hidden_gradients:
                        parameter.grad = None
                    expert_norms[group_name] = model.clip_grad_norm_(1.0).float()
                    for parameter, gradient in hidden_gradients:
                        parameter.grad = gradient
                if step == 1:
                    first_expert_clip_norms = {
                        name: float(value) for name, value in expert_norms.items()
                    }
                grad_norm = torch.stack(list(expert_norms.values())).square().sum().sqrt()
            else:
                grad_norm = model.clip_grad_norm_(1.0)
            if master_parameters is not None:
                for parameter, master in zip(
                    optimized_parameters, master_parameters, strict=True
                ):
                    master.grad = (
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().float()
                    )
            optimizer.step()
            if master_parameters is not None:
                with torch.no_grad():
                    for parameter, master in zip(
                        optimized_parameters, master_parameters, strict=True
                    ):
                        parameter.copy_(master.to(dtype=parameter.dtype))
        metrics = torch.tensor(
            [
                loss.detach(),
                video_loss.detach(),
                flow_loss.detach(),
                action_loss.detach(),
                grad_norm.detach(),
                action_weight.detach(),
                world_weight.detach(),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        row = {
            "step": step,
            "loss": float(metrics[0]),
            "video_loss": float(metrics[1]),
            "flow_loss": float(metrics[2]),
            "action_loss": float(metrics[3]),
            "gradient_norm": float(metrics[4]),
            "action_weight": float(metrics[5]),
            "world_weight": float(metrics[6]),
            "sample": str(batch["sample"]),
        }
        history.append(row)
        if rank == 0:
            print(json.dumps({"event": "optimization_step", **row}), flush=True)
    torch.cuda.synchronize(device)
    final_video_gates = video_gate_summary(model, device)
    report = {
        "event": "h3dreamwam_real_fsdp_smoke",
        "eval_only": args.eval_only,
        "joint_sample_steps": args.joint_sample_steps,
        "dreamwam_action_weighting": args.dreamwam_action_weighting,
        "dreamwam_world_weighting": args.dreamwam_world_weighting,
        "motion_root": str(args.motion_root.resolve()) if args.motion_root else None,
        "flow_loss_weight": args.flow_loss_weight,
        "train_h3_io": args.train_h3_io,
        "require_text_only_context": args.require_text_only_context,
        "dreamwam_exact_action_norm": args.dreamwam_exact_action_norm,
        "action_init_alpha_scaling": args.action_init_alpha_scaling,
        "world_size": world_size,
        "loaded_action_head": (
            str(args.load_action_head.resolve()) if args.load_action_head else None
        ),
        "loaded_action_stage": (
            str(args.load_action_stage.resolve()) if args.load_action_stage else None
        ),
        "override_action_io": (
            str(args.override_action_io.resolve()) if args.override_action_io else None
        ),
        "disabled_video_residual_adapters": args.disable_video_residual_adapters,
        "sample": sample_schedule[0],
        "rotate_manifest": args.rotate_manifest,
        "global_unique_samples_seen": (
            min(len(selected_rows), args.steps * world_size)
            if args.rotate_manifest and selected_rows is not None
            else world_size
        ),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "manifest_windows": len(selected_rows) if selected_rows is not None else None,
        "task": args.task,
        "last_h3_blocks": args.last_h3_blocks,
        "action_horizon": args.action_horizon,
        "action_train_stage": args.action_train_stage,
        "freeze_action_body": args.freeze_action_body,
        "freeze_shared_state": args.freeze_shared_state,
        "freeze_action_output": args.freeze_action_output,
        "train_video_residual_gates": args.train_video_residual_gates,
        "train_video_residual_adapters": args.train_video_residual_adapters,
        "train_cross_attention_output": args.train_cross_attention_output,
        "separate_expert_clipping": args.separate_expert_clipping,
        "last_action_blocks": args.last_action_blocks,
        "trainable_parameters": full_trainable_parameters,
        "optimized_parameters": global_optimized_parameter_count,
        "new_layer_parameters": global_new_layer_parameter_count,
        "tail_parameters": global_tail_parameter_count,
        "video_gate_parameters": global_gate_parameter_count,
        "video_adapter_parameters": global_adapter_parameter_count,
        "cross_attention_parameters": global_cross_attention_parameter_count,
        "h3_optimized_parameters": global_h3_parameter_count,
        "h3_learning_rate": args.h3_learning_rate,
        "new_layer_lr_scale": args.new_layer_lr_scale,
        "gate_learning_rate": args.gate_learning_rate,
        "adapter_learning_rate": args.adapter_learning_rate,
        "cross_attention_learning_rate": args.cross_attention_learning_rate,
        "tail_learning_rate": args.tail_learning_rate,
        "action_construction_dtype": str(action_storage_dtype),
        "model_storage_dtype": str(model_storage_dtype),
        "fp32_optimizer_master": master_parameters is not None,
        "initialization": initialization_report.__dict__,
        "initialization_seconds": initialization_seconds,
        "migrated_legacy_action_blocks": migrated_legacy_action_blocks,
        "final_video_gates": final_video_gates,
        "steps": args.steps,
        "loss": history[-1]["loss"],
        "video_loss": history[-1]["video_loss"],
        "flow_loss": history[-1]["flow_loss"],
        "action_loss": history[-1]["action_loss"],
        "gradient_norm": history[-1]["gradient_norm"],
        "history": history,
        "mean_action_loss": sum(row["action_loss"] for row in history) / len(history),
        "mean_video_loss": sum(row["video_loss"] for row in history) / len(history),
        "mean_flow_loss": sum(row["flow_loss"] for row in history) / len(history),
        "first_gradient_norm_groups": first_gradient_groups,
        "first_expert_clip_norms": first_expert_clip_norms,
        "load_seconds": load_seconds,
        "step_seconds": time.perf_counter() - step_started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    if args.save_action_head is not None:
        # The output head is a root-FSDP parameter. Materialize only root-level
        # parameters, avoiding a 70+ GiB full model state dict.
        with FSDP.summon_full_params(
            model, recurse=False, writeback=False, rank0_only=True
        ):
            if rank == 0:
                head_path = args.save_action_head.resolve()
                head_path.parent.mkdir(parents=True, exist_ok=True)
                head = model.module.action_expert.output
                torch.save(
                    {
                        "format": "h3dreamwam_action_head_v1",
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in head.state_dict().items()
                        },
                        "steps": args.steps,
                        "task": args.task,
                    },
                    head_path,
                )
                report["saved_action_head"] = str(head_path)
    if args.save_action_stage is not None:
        stage_path = args.save_action_stage.resolve()
        stage_blocks: dict[str, dict[str, torch.Tensor]] = {}
        stage_h3_blocks: dict[str, dict[str, torch.Tensor]] = {}
        stage_h3_io = None
        with FSDP.summon_full_params(
            model, recurse=False, writeback=False, rank0_only=True
        ):
            if rank == 0:
                root_action = model.module.action_expert
                stage_io = {
                    name: {
                        key: value.detach().cpu().clone()
                        for key, value in module.state_dict().items()
                    }
                    for name, module in {
                        "action_embedding": root_action.action_embedding,
                        "state_embedding": root_action.state_embedding,
                        "context_embedding": root_action.context_embedding,
                        "time_embedding": root_action.time_embedding,
                        "time_projection": root_action.time_projection,
                        "output": root_action.output,
                    }.items()
                }
                if args.train_h3_io or (
                    stage_payload is not None
                    and stage_payload.get("h3_io") is not None
                ):
                    stage_h3_io = {
                        name: {
                            key: value.detach().cpu().clone()
                            for key, value in module.state_dict().items()
                        }
                        for name, module in {
                            "proj_in": model.module.h3.proj_in,
                            "proj_out": model.module.h3.proj_out,
                        }.items()
                    }
        trained_action_layers: set[int] = set()
        if args.action_train_stage == "full":
            trained_action_layers.update(range(50))
        elif args.action_train_stage in ("tail", "tail_sharded") and (
            not args.freeze_action_body
            or args.train_video_residual_gates
            or args.train_video_residual_adapters
            or args.train_cross_attention_output
        ):
            trained_action_layers.update(range(50 - args.last_action_blocks, 50))
        saved_action_layers = loaded_stage_layers | trained_action_layers
        first_h3_layer = 50 - args.last_h3_blocks
        trained_h3_layers = (
            set(range(first_h3_layer, 50)) if args.last_h3_blocks else set()
        )
        saved_h3_layers = loaded_h3_layers | trained_h3_layers
        saved_layers = saved_action_layers | saved_h3_layers
        for index in sorted(saved_layers):
            paired_fsdp = model.module.paired_layers[index]
            with FSDP.summon_full_params(
                paired_fsdp, recurse=False, writeback=False, rank0_only=True
            ):
                if rank == 0:
                    if index in saved_action_layers:
                        stage_blocks[str(index)] = {
                            key: value.detach().cpu().clone()
                            for key, value in paired_fsdp.module.action_block.state_dict().items()
                        }
                    if index in saved_h3_layers:
                        stage_h3_blocks[str(index)] = {
                            key: value.detach().cpu().clone()
                            for key, value in paired_fsdp.module.h3_block.state_dict().items()
                        }
        if rank == 0:
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format": "h3dreamwam_action_stage_v1",
                    "architecture": {
                        "full_width_rmsnorm": args.dreamwam_exact_action_norm,
                        "alpha_scaling": args.action_init_alpha_scaling,
                        "video_residual_gate": True,
                        "video_residual_adapter_rank": 16,
                    },
                    "io": stage_io,
                    "blocks": stage_blocks,
                    "h3_io": stage_h3_io,
                    "h3_blocks": stage_h3_blocks,
                    "steps": args.steps,
                    "task": args.task,
                    "action_horizon": args.action_horizon,
                    "parent_stage": (
                        str(args.load_action_stage.resolve())
                        if args.load_action_stage is not None
                        else None
                    ),
                },
                stage_path,
            )
            report["saved_action_stage"] = str(stage_path)
    if rank == 0:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
