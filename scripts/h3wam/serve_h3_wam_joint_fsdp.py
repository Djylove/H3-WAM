#!/usr/bin/env python3
"""Serve official BF16 H3 plus either a joint or local-style action expert.

The checkpoint stores FSDP-local shards, so this server intentionally restores
it with the same world size used for training.  Rank zero owns the VAE and the
local rollout socket; every rank participates in the H3/action forward pass.
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precompute_libero_official_h3 import PIXEL_MEAN, PIXEL_STD  # noqa: E402
from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    audio_latent_count,
    replicated_non_block_modules,
    set_trainable_tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional rank-sharded joint H3-WAM checkpoint.",
    )
    parser.add_argument(
        "--action-checkpoint",
        type=Path,
        help="Local-style frozen-H3 feature-action checkpoint.",
    )
    parser.add_argument(
        "--h3-tail-checkpoint",
        type=Path,
        help=(
            "Optional rank-sharded H3 tail checkpoint used with "
            "--action-checkpoint. It must use the same world size and "
            "--last-blocks value as training."
        ),
    )
    parser.add_argument(
        "--h3-lora-checkpoint",
        type=Path,
        help="Optional official-H3 LoRA checkpoint directory used with --action-checkpoint.",
    )
    parser.add_argument("--h3-lora-recovery-only", action="store_true")
    parser.add_argument(
        "--h3-lora-recovery-max-trigger-step",
        type=int,
        help="With recovery-only LoRA, enable only for gates triggered by this step.",
    )
    parser.add_argument(
        "--h3-lora-rerun-on-recovery-trigger",
        action="store_true",
        help=(
            "After the frozen-H3 recovery gate fires, rerun H3 with LoRA on the "
            "same observation so the first recovery action uses adapted features."
        ),
    )
    parser.add_argument("--action-recovery-checkpoint", type=Path)
    parser.add_argument("--action-recovery-task")
    parser.add_argument("--action-recovery-after-step", type=int, default=64)
    parser.add_argument("--action-recovery-gate-checkpoint", type=Path)
    parser.add_argument("--action-recovery-gate-threshold", type=float)
    parser.add_argument(
        "--action-ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Additional compatible local action heads averaged at inference.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-contexts", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--last-blocks", type=int, default=50)
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        default=(4, 9, 14, 19, 24, 29, 34, 39, 44, 49),
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-hidden-dim", type=int, default=1024)
    parser.add_argument("--action-layers", type=int, default=1)
    parser.add_argument("--action-heads", type=int, default=16)
    parser.add_argument("--action-ffn-dim", type=int, default=4096)
    parser.add_argument("--action-flow-shift", type=float, default=5.0)
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument("--feature-video-timestep", type=float, default=1.0)
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--history-adapter-scale",
        type=float,
        default=1.0,
        help="Inference-only multiplier for trained temporal adapter residuals.",
    )
    parser.add_argument(
        "--event-stage-routing",
        choices=("learned", "monotonic"),
        default="learned",
        help=(
            "Routing for event_stage mixture heads. Monotonic routing starts "
            "at stage zero and unlocks later stages only after demonstrated "
            "stage boundaries."
        ),
    )
    return parser.parse_args()


def _broadcast_object(value, rank: int):
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _load_local_checkpoint(model, checkpoint: Path, rank: int, world_size: int) -> int:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "h3wam-fsdp-local-bf16-v1":
        raise ValueError(f"unsupported checkpoint format: {manifest.get('format')}")
    if int(manifest["world_size"]) != world_size:
        raise ValueError(
            f"checkpoint world size {manifest['world_size']} != runtime {world_size}"
        )
    payload = torch.load(
        checkpoint / f"h3_rank{rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    saved = payload["parameters"]
    if current.keys() != saved.keys():
        missing = sorted(current.keys() - saved.keys())
        unexpected = sorted(saved.keys() - current.keys())
        raise ValueError(
            f"checkpoint parameter mismatch: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            value = saved[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"checkpoint shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    del payload, saved
    return int(manifest["step"])


def _task_lookup(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_id = json.loads(path.read_text(encoding="utf-8"))
    by_language = {str(language): str(context_id) for context_id, language in by_id.items()}
    if len(by_language) != len(by_id):
        raise ValueError("task-context map contains duplicate task languages")
    return by_id, by_language


def main() -> None:
    args = parse_args()
    if args.port <= 0 or args.action_horizon <= 0 or args.flow_steps <= 0:
        raise ValueError("port, action-horizon and flow-steps must be positive")
    if (args.checkpoint is None) == (args.action_checkpoint is None):
        raise ValueError("set exactly one of --checkpoint and --action-checkpoint")
    if args.h3_tail_checkpoint is not None and args.action_checkpoint is None:
        raise ValueError("--h3-tail-checkpoint requires --action-checkpoint")
    if args.h3_lora_checkpoint is not None and args.action_checkpoint is None:
        raise ValueError("--h3-lora-checkpoint requires --action-checkpoint")
    if args.h3_tail_checkpoint is not None and args.h3_lora_checkpoint is not None:
        raise ValueError("H3 tail and LoRA checkpoints are mutually exclusive")
    if args.h3_lora_recovery_only and args.h3_lora_checkpoint is None:
        raise ValueError("recovery-only LoRA requires --h3-lora-checkpoint")
    if args.h3_lora_recovery_only and args.action_recovery_checkpoint is None:
        raise ValueError("recovery-only LoRA requires a recovery action checkpoint")
    if args.h3_lora_rerun_on_recovery_trigger and not args.h3_lora_recovery_only:
        raise ValueError("trigger rerun requires --h3-lora-recovery-only")
    if args.action_ensemble_checkpoint and args.action_checkpoint is None:
        raise ValueError("action ensembles require --action-checkpoint")
    if (args.action_recovery_checkpoint is None) != (args.action_recovery_task is None):
        raise ValueError("recovery checkpoint and recovery task must be set together")
    if args.action_recovery_after_step < 0:
        raise ValueError("action-recovery-after-step cannot be negative")
    if (
        args.action_recovery_gate_checkpoint is not None
        and args.action_recovery_checkpoint is None
    ):
        raise ValueError("recovery gate requires a recovery action checkpoint")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
    )
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from fastwam.models.h3wam import (
        H3BlockAttentionMask,
        H3FeatureActionTransformer,
        H3FeatureSwitchGate,
        H3MixtureActionOutput,
        H3MultiLayerActionTransformer,
        H3OfficialFeatureCapture,
        H3LoRALinear,
        build_h3_observation_attention_mask,
        inject_official_h3_lora,
        libero_dataset_action,
        libero_environment_actions,
        libero_observation_state,
        minmax_normalize,
        load_h3_lora_state_dict,
        set_h3_lora_enabled,
        preprocess_libero_cameras,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    model_path = args.model.resolve()
    checkpoint = None if args.checkpoint is None else args.checkpoint.resolve()
    joint_checkpoint_manifest = (
        None
        if checkpoint is None
        else json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    )
    h3_tail_checkpoint = (
        None
        if args.h3_tail_checkpoint is None
        else args.h3_tail_checkpoint.resolve()
    )
    h3_lora_checkpoint = (
        None
        if args.h3_lora_checkpoint is None
        else torch.load(
            args.h3_lora_checkpoint.resolve() / "h3_lora.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    if (
        h3_lora_checkpoint is not None
        and h3_lora_checkpoint.get("format") != "h3wam-official-h3-lora-v1"
    ):
        raise ValueError("unsupported official H3 LoRA checkpoint format")
    local_action_checkpoint = (
        None
        if args.action_checkpoint is None
        else torch.load(
            args.action_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
    )
    local_action_members = (
        []
        if local_action_checkpoint is None
        else [local_action_checkpoint]
        + [
            torch.load(path.resolve(), map_location="cpu", weights_only=False)
            for path in args.action_ensemble_checkpoint
        ]
    )
    recovery_action_checkpoint = (
        None
        if args.action_recovery_checkpoint is None
        else torch.load(
            args.action_recovery_checkpoint.resolve(),
            map_location="cpu",
            weights_only=False,
        )
    )
    recovery_gate_checkpoint = (
        None
        if args.action_recovery_gate_checkpoint is None
        else torch.load(
            args.action_recovery_gate_checkpoint.resolve(),
            map_location="cpu",
            weights_only=False,
        )
    )
    if local_action_checkpoint is not None:
        if local_action_checkpoint.get("policy_type") != "h3_feature_action":
            raise ValueError("action-checkpoint is not an H3 feature-action policy")
        if str(local_action_checkpoint.get("objective", "regression")) != "regression":
            raise ValueError("the compatibility canary currently requires regression")
        trained_horizon = int(local_action_checkpoint["action_horizon"])
        if args.action_horizon > trained_horizon:
            raise ValueError(
                f"requested horizon {args.action_horizon} exceeds trained {trained_horizon}"
            )
        contract_keys = (
            "policy_type",
            "objective",
            "action_horizon",
            "action_dim",
            "state_dim",
            "feature_shape",
            "feature_layers",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "num_action_modes",
            "use_proprio",
            "use_previous_action",
            "include_phase",
            "action_mode_labeling",
            "task_conditioning",
            "task_to_index",
        )
        for member in local_action_members[1:]:
            for key in contract_keys:
                expected = local_action_checkpoint.get(key)
                actual = member.get(key)
                if key in ("feature_shape", "feature_layers"):
                    expected = tuple(expected)
                    actual = tuple(actual)
                if actual != expected:
                    raise ValueError(
                        f"action ensemble {key} mismatch: {actual!r} != {expected!r}"
                    )
            for key in ("action_min", "action_max", "state_min", "state_max"):
                if not torch.equal(
                    member["normalization"][key],
                    local_action_checkpoint["normalization"][key],
                ):
                    raise ValueError(f"action ensemble normalization mismatch for {key}")
        if recovery_action_checkpoint is not None:
            for key in contract_keys:
                expected = local_action_checkpoint.get(key)
                actual = recovery_action_checkpoint.get(key)
                if key in ("feature_shape", "feature_layers"):
                    expected = tuple(expected)
                    actual = tuple(actual)
                if actual != expected:
                    raise ValueError(
                        f"action recovery {key} mismatch: {actual!r} != {expected!r}"
                    )
            for key in ("action_min", "action_max", "state_min", "state_max"):
                if not torch.equal(
                    recovery_action_checkpoint["normalization"][key],
                    local_action_checkpoint["normalization"][key],
                ):
                    raise ValueError(f"action recovery normalization mismatch for {key}")
            primary_phase = int(
                local_action_checkpoint.get("phase_lengths_by_task", {}).get(
                    args.action_recovery_task, local_action_checkpoint["phase_length"]
                )
            )
            recovery_phase = int(
                recovery_action_checkpoint.get("phase_lengths_by_task", {}).get(
                    args.action_recovery_task,
                    recovery_action_checkpoint["phase_length"],
                )
            )
            if recovery_phase != primary_phase:
                raise ValueError(
                    "action recovery task phase mismatch: "
                    f"{recovery_phase} != {primary_phase}"
                )
    cache_root = args.cache_root.resolve()
    load_started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if h3_lora_checkpoint is not None:
        h3.requires_grad_(False)
        inject_official_h3_lora(
            h3,
            last_n_blocks=int(h3_lora_checkpoint["last_blocks"]),
            rank=int(h3_lora_checkpoint["rank"]),
            alpha=float(h3_lora_checkpoint["alpha"]),
        )
        load_h3_lora_state_dict(h3, h3_lora_checkpoint["state"])
    elif (
        h3_tail_checkpoint is not None
        or (
            checkpoint is not None
            and not bool(joint_checkpoint_manifest.get("backbone_frozen", False))
        )
    ):
        set_trainable_tail(h3, args.last_blocks)
    else:
        h3.requires_grad_(False)
    # The rank-local artifact is BF16.  Keeping BF16 parameter storage for
    # inference preserves the saved values and avoids rebuilding eight full
    # FP32 master copies before FSDP shards the model.
    if hasattr(h3, "disable_gradient_checkpointing"):
        h3.disable_gradient_checkpointing()
    replicated_modules = replicated_non_block_modules(h3)
    if h3_lora_checkpoint is not None:
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
    h3_checkpoint = checkpoint if checkpoint is not None else h3_tail_checkpoint
    checkpoint_step = (
        _load_local_checkpoint(h3, h3_checkpoint, rank, world_size)
        if h3_checkpoint is not None
        else 0
    )
    if h3_lora_checkpoint is not None:
        checkpoint_step = int(h3_lora_checkpoint["step"])
    capture_layers = tuple(
        sorted(
            set(
                local_action_checkpoint["feature_layers"]
                if local_action_checkpoint is not None
                else args.capture_layers
            )
        )
    )
    if not capture_layers or capture_layers[-1] >= len(h3.module.transformer_blocks):
        raise ValueError("capture layer lies outside H3")
    patch_size = tuple(h3.module.config.patch_size)
    if local_action_checkpoint is not None:
        feature_shape = tuple(local_action_checkpoint["feature_shape"])
        if feature_shape[0] != len(capture_layers) or feature_shape[-1] != int(
            h3.module.config.hidden_size
        ):
            raise ValueError(
                f"local feature contract {feature_shape} does not match official H3"
            )
        action_heads = torch.nn.ModuleList()
        head_members = local_action_members + (
            [] if recovery_action_checkpoint is None else [recovery_action_checkpoint]
        )
        for member in head_members:
            member_head = H3FeatureActionTransformer(
                action_dim=int(member["action_dim"]),
                state_dim=int(member["state_dim"]),
                h3_feature_dim=feature_shape[-1],
                hidden_dim=int(member["hidden_dim"]),
                num_layers=int(member["num_layers"]),
                num_heads=int(member["num_heads"]),
                ffn_dim=int(member["ffn_dim"]),
                num_action_modes=int(member.get("num_action_modes", 1)),
            ).to(device)
            member_head.load_state_dict(member["model"], strict=True)
            action_heads.append(member_head)
        recovery_action_head = (
            None if recovery_action_checkpoint is None else action_heads[-1]
        )
        primary_action_heads = action_heads[: len(local_action_members)]
        action_head = action_heads[0]
        recovery_gate = None
        recovery_gate_threshold = None
        if recovery_gate_checkpoint is not None:
            if recovery_gate_checkpoint.get("policy_type") != "h3_feature_switch_gate":
                raise ValueError("recovery gate checkpoint has the wrong policy type")
            if tuple(recovery_gate_checkpoint["feature_shape"]) != feature_shape:
                raise ValueError("recovery gate feature shape differs")
            if tuple(recovery_gate_checkpoint["feature_layers"]) != capture_layers:
                raise ValueError("recovery gate feature layers differ")
            recovery_gate = H3FeatureSwitchGate(
                h3_feature_dim=int(recovery_gate_checkpoint["h3_feature_dim"]),
                state_dim=int(recovery_gate_checkpoint["state_dim"]),
                hidden_dim=int(recovery_gate_checkpoint["hidden_dim"]),
            ).to(device)
            recovery_gate.load_state_dict(recovery_gate_checkpoint["model"])
            recovery_gate.eval()
            recovery_gate_threshold = float(
                recovery_gate_checkpoint.get("threshold", 0.5)
                if args.action_recovery_gate_threshold is None
                else args.action_recovery_gate_threshold
            )
            if not 0.0 < recovery_gate_threshold < 1.0:
                raise ValueError("recovery gate threshold must be in (0,1)")
        action_mode = "local_regression"
    else:
        action_heads = None
        primary_action_heads = None
        recovery_action_head = None
        recovery_gate = None
        recovery_gate_threshold = None
        action_head = H3MultiLayerActionTransformer(
            action_dim=7,
            state_dim=int(joint_checkpoint_manifest.get("action_state_dim", 8)),
            num_h3_layers=len(capture_layers),
            h3_feature_dim=int(h3.module.config.hidden_size),
            hidden_dim=int(
                joint_checkpoint_manifest.get(
                    "action_hidden_dim", args.action_hidden_dim
                )
            ),
            num_layers=int(
                joint_checkpoint_manifest.get("action_layers", args.action_layers)
            ),
            num_heads=int(
                joint_checkpoint_manifest.get("action_heads", args.action_heads)
            ),
            ffn_dim=int(
                joint_checkpoint_manifest.get("action_ffn_dim", args.action_ffn_dim)
            ),
            max_horizon=max(64, args.action_horizon),
            language_feature_dim=(
                int(joint_checkpoint_manifest["language_feature_dim"])
                if bool(
                    joint_checkpoint_manifest.get(
                        "explicit_language_conditioning", False
                    )
                )
                else None
            ),
            layer_mix_initialization=str(
                joint_checkpoint_manifest.get(
                    "layer_mix_initialization", "spaced"
                )
            ),
            history_conditioning=bool(
                joint_checkpoint_manifest.get("history_frame_conditioning", False)
            ),
            history_adapter_rank=int(
                joint_checkpoint_manifest.get("history_adapter_rank", 0)
            ),
        ).to(device)
        action_state = torch.load(
            checkpoint / "action_head.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        action_head.load_state_dict(action_state, strict=True)
        del action_state
        if not 0.0 <= args.history_adapter_scale <= 1.0:
            raise ValueError("history-adapter-scale must be in [0,1]")
        if action_head.history_up is None and args.history_adapter_scale != 1.0:
            raise ValueError("history-adapter-scale requires an adapter checkpoint")
        if action_head.history_up is not None and args.history_adapter_scale != 1.0:
            with torch.no_grad():
                for projection in action_head.history_up:
                    projection.weight.mul_(args.history_adapter_scale)
        joint_action_objective = str(
            joint_checkpoint_manifest.get("action_objective", "flow")
        )
        if joint_action_objective not in ("flow", "regression"):
            raise ValueError(
                f"unsupported joint action objective: {joint_action_objective!r}"
            )
        action_mode = f"joint_{joint_action_objective}"
    h3.eval()
    action_head.eval()
    if action_heads is not None:
        action_heads.eval()
    capture = H3OfficialFeatureCapture(
        h3.module.transformer_blocks, capture_layers, torch.tensor([0])
    )
    attention_hooks = H3BlockAttentionMask(h3.module.transformer_blocks)
    stats = (
        local_action_checkpoint["normalization"]
        if local_action_checkpoint is not None
        else torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)
    )
    _, language_to_context = _task_lookup(args.task_contexts.resolve())
    context_cache: dict[str, dict] = {}

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

    def load_context(task: str) -> tuple[str, dict]:
        if task not in language_to_context:
            raise KeyError(f"task language is absent from training contexts: {task!r}")
        context_id = language_to_context[task]
        if context_id not in context_cache:
            item = torch.load(
                cache_root / "contexts" / f"{context_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            context_cache[context_id] = {
                "context": item["context"].to(device=device, dtype=torch.float32),
                "token_tags": item["token_tags"].to(dtype=torch.long),
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
        # The shared deployment helper returns [0,1], while the released
        # official H3 encoder consumes uint8-like [0,255] pixels and performs
        # its own division/ImageNet normalization.
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
            first = encode_vae_condition(vae, video, PIXEL_MEAN, PIXEL_STD).to(device)
        observation = {
            "eef_pos": request["eef_pos"],
            "eef_quat": request["eef_quat"],
            "gripper_qpos": request["gripper_qpos"],
        }
        state = minmax_normalize(
            libero_observation_state(observation),
            stats["state_min"],
            stats["state_max"],
        ).clamp(-1.0, 1.0).to(device)
        return first.float(), state.float(), time.perf_counter() - started

    recovery_latched = False
    lora_recovery_latched = False
    event_stage_states = [0 for _ in local_action_members]

    def policy_forward(
        task: str,
        first: torch.Tensor,
        history_first: torch.Tensor | None,
        state: torch.Tensor,
        previous_action: torch.Tensor | None,
        seed: int,
        step: int,
    ) -> tuple[
        torch.Tensor,
        str,
        float,
        float | None,
        bool,
        int | None,
        str | None,
        list[float] | None,
    ]:
        nonlocal recovery_latched, lora_recovery_latched, event_stage_states
        if step == 0:
            recovery_latched = False
            lora_recovery_latched = False
            event_stage_states = [0 for _ in local_action_members]
        context_id, conditioning = load_context(task)
        gate_probability = None
        recovery_active = False
        selected_action_mode = None
        selected_action_mode_source = None
        action_mode_probabilities = None
        text_tags = conditioning["token_tags"]
        _, channels, _, latent_height, latent_width = first.shape
        pixel_frames = 3 * args.target_latent_frames + 3
        # The successful local Comfy feature contract used a zero audio tensor
        # shaped [B, 32, 2, action_horizon].  Its packed sequence therefore had
        # ``action_horizon`` audio positions, not the physical 40-Hz audio
        # length used by the native video-training path.
        num_audio_latents = (
            args.action_horizon
            if local_action_checkpoint is not None
            else audio_latent_count(pixel_frames)
        )
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_tags,
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
        video_indices_device = video_indices.to(device)
        audio_indices_device = audio_indices.to(device)
        text_indices_device = text_indices.to(device)
        target = torch.zeros(
            (1, channels, args.target_latent_frames, latent_height, latent_width),
            device=device,
            dtype=torch.float32,
        )
        video_rows = torch.cat(
            (patchify_video_latents(first, patch_size), patchify_video_latents(target, patch_size)),
            dim=0,
        )[None]
        audio_rows = torch.zeros(
            (1, num_audio_latents * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            device=device,
            dtype=torch.float32,
        )
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
        condition_indices = video_indices_device[:num_condition_video_rows]
        capture.set_condition_video_indices(condition_indices)
        if h3_lora_checkpoint is not None:
            set_h3_lora_enabled(
                h3.module,
                not args.h3_lora_recovery_only or lora_recovery_latched,
            )
        if local_action_checkpoint is None:
            attention_hooks.set(
                build_h3_observation_attention_mask(
                    sequence_length=int(position_ids.shape[0]),
                    text_indices=text_indices_device,
                    condition_video_indices=condition_indices,
                    device=device,
                )
            )
        else:
            # Match the successful local Comfy feature cache: standard H3
            # packed attention over text, keyframe and zero target rows.
            attention_hooks.set(None)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            history_features = None
            if bool(
                joint_checkpoint_manifest is not None
                and joint_checkpoint_manifest.get(
                    "history_frame_conditioning", False
                )
            ):
                if history_first is None or tuple(history_first.shape) != tuple(
                    first.shape
                ):
                    raise ValueError(
                        "history-conditioned checkpoint requires a matching history "
                        "latent"
                    )
                history_video_rows = torch.cat(
                    (
                        patchify_video_latents(history_first, patch_size),
                        patchify_video_latents(target, patch_size),
                    ),
                    dim=0,
                )[None]
                h3(
                    hidden_states=history_video_rows,
                    audio_hidden_states=audio_rows,
                    encoder_hidden_states=conditioning["context"],
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices_device,
                    audio_indices=audio_indices_device,
                    text_indices=text_indices_device,
                    return_dict=True,
                )
                history_features = capture.stacked()
            h3(
                hidden_states=video_rows,
                audio_hidden_states=audio_rows,
                encoder_hidden_states=conditioning["context"],
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
            if local_action_checkpoint is not None:
                expected_feature_shape = tuple(local_action_checkpoint["feature_shape"])
                if tuple(features.shape[1:]) != expected_feature_shape:
                    raise RuntimeError(
                        "official/local H3 feature shape mismatch: "
                        f"{tuple(features.shape[1:])} != {expected_feature_shape}"
                    )
                training_tasks = set(local_action_checkpoint.get("training_tasks", ()))
                if training_tasks and task not in training_tasks:
                    raise ValueError(
                        f"task {task!r} is outside action checkpoint {training_tasks}"
                    )
                state_parts = []
                if bool(local_action_checkpoint.get("use_proprio", False)):
                    state_parts.append(
                        state.reshape(1, 1, 8).expand(
                            1, args.action_horizon, 8
                        ).clone()
                    )
                if bool(local_action_checkpoint.get("use_previous_action", False)):
                    if previous_action is None:
                        raise ValueError("checkpoint requires a previous action")
                    state_parts.append(
                        previous_action.reshape(1, 1, 7).expand(
                            1, args.action_horizon, 7
                        ).clone()
                    )
                if bool(local_action_checkpoint.get("include_phase", True)):
                    phase_lengths_by_task = local_action_checkpoint.get(
                        "phase_lengths_by_task", {}
                    )
                    phase_length = int(
                        phase_lengths_by_task.get(
                            task, local_action_checkpoint["phase_length"]
                        )
                    )
                    phase_steps = torch.arange(
                        args.action_horizon, device=device, dtype=torch.float32
                    )
                    phase_steps.add_(step).clamp_max_(phase_length - 1)
                    phase = (
                        2.0 * phase_steps / max(phase_length - 1, 1) - 1.0
                    ).reshape(1, args.action_horizon, 1)
                    state_parts.append(phase)
                if bool(local_action_checkpoint.get("task_conditioning", False)):
                    task_to_index = local_action_checkpoint.get("task_to_index", {})
                    if task not in task_to_index:
                        raise ValueError(
                            f"task {task!r} is absent from explicit task conditioning"
                        )
                    task_one_hot = torch.zeros(
                        (1, args.action_horizon, len(task_to_index)),
                        device=device,
                        dtype=torch.float32,
                    )
                    task_one_hot[:, :, int(task_to_index[task])] = 1.0
                    state_parts.append(task_one_hot)
                if not state_parts:
                    state_parts.append(
                        torch.zeros(
                            (1, args.action_horizon, 1),
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                policy_state = torch.cat(state_parts, dim=-1)
                expected_state_dim = int(local_action_checkpoint["state_dim"])
                if policy_state.shape[-1] != expected_state_dim:
                    raise RuntimeError(
                        f"constructed state dim {policy_state.shape[-1]} does not "
                        f"match checkpoint {expected_state_dim}"
                    )
                assert primary_action_heads is not None
                model_input = torch.zeros(
                    (1, args.action_horizon, 7),
                    device=device,
                    dtype=torch.float32,
                )
                if (
                    recovery_gate is not None
                    and task == args.action_recovery_task
                    and not recovery_latched
                ):
                    assert recovery_gate_threshold is not None
                    gate_probability = float(
                        recovery_gate(features, state.reshape(1, 8))
                        .sigmoid()
                        .item()
                    )
                    recovery_latched = gate_probability >= recovery_gate_threshold
                recovery_active = (
                    recovery_action_checkpoint is not None
                    and task == args.action_recovery_task
                    and (
                        recovery_latched
                        if recovery_gate is not None
                        else step >= args.action_recovery_after_step
                    )
                )
                lora_just_triggered = False
                if (
                    args.h3_lora_recovery_only
                    and recovery_active
                    and not lora_recovery_latched
                    and (
                        args.h3_lora_recovery_max_trigger_step is None
                        or step <= args.h3_lora_recovery_max_trigger_step
                    )
                ):
                    lora_recovery_latched = True
                    lora_just_triggered = True
                if lora_just_triggered and args.h3_lora_rerun_on_recovery_trigger:
                    # Keep the gate decision on released-H3 features, then adapt
                    # the very same observation.  Corrective roll-ins start at
                    # the trigger state, so delaying LoRA by one action chunk
                    # would make its first input an out-of-distribution state.
                    set_h3_lora_enabled(h3.module, True)
                    h3(
                        hidden_states=video_rows,
                        audio_hidden_states=audio_rows,
                        encoder_hidden_states=conditioning["context"],
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
                active_members = (
                    [recovery_action_checkpoint]
                    if recovery_active
                    else local_action_members
                )
                active_heads = (
                    [recovery_action_head]
                    if recovery_active
                    else list(primary_action_heads)
                )
                member_outputs = [
                    member_head(
                        model_input,
                        state=policy_state,
                        h3_features=features,
                        video_sigma=torch.zeros(1, device=device),
                    )
                    for member_head in active_heads
                ]
                member_actions = []
                for member_index, (member, value) in enumerate(
                    zip(active_members, member_outputs)
                ):
                    if isinstance(value, H3MixtureActionOutput):
                        probabilities = value.mode_logits.softmax(dim=-1)
                        if member.get("action_mode_labeling") == "task":
                            member_task_to_index = member.get("task_to_index", {})
                            if task not in member_task_to_index:
                                raise ValueError(
                                    f"task {task!r} has no task-routed action expert"
                                )
                            selected_mode = int(member_task_to_index[task])
                            mode_source = "task"
                        elif (
                            member.get("action_mode_labeling") == "event_stage"
                            and args.event_stage_routing == "monotonic"
                        ):
                            learned_mode = int(value.mode_logits.argmax(dim=-1).item())
                            boundary_values = list(
                                member.get("event_stage_boundaries", {}).values()
                            )
                            if not boundary_values:
                                raise ValueError(
                                    "monotonic event-stage routing needs stored boundaries"
                                )
                            earliest_boundaries = np.asarray(
                                boundary_values, dtype=np.int64
                            ).min(axis=0)
                            unlocked_mode = int(
                                np.count_nonzero(step >= earliest_boundaries)
                            )
                            current_mode = event_stage_states[member_index]
                            if learned_mode > current_mode and unlocked_mode > current_mode:
                                current_mode += 1
                                event_stage_states[member_index] = current_mode
                            selected_mode = current_mode
                            mode_source = "event_stage_monotonic"
                        else:
                            selected_mode = int(
                                value.mode_logits.argmax(dim=-1).item()
                            )
                            mode_source = "learned"
                        member_actions.append(value.actions[:, selected_mode])
                        if member_index == 0:
                            selected_action_mode = selected_mode
                            selected_action_mode_source = mode_source
                            action_mode_probabilities = (
                                probabilities[0].detach().float().cpu().tolist()
                            )
                    elif isinstance(value, torch.Tensor):
                        member_actions.append(value)
                    else:
                        raise TypeError(f"unsupported action output {type(value)!r}")
                actions = torch.stack(member_actions).mean(dim=0)
            elif joint_action_objective == "regression":
                model_input = torch.zeros(
                    (1, args.action_horizon, 7),
                    device=device,
                    dtype=torch.float32,
                )
                state_parts = [
                    state.reshape(1, 1, 8).expand(
                        1, args.action_horizon, 8
                    )
                ]
                if bool(
                    joint_checkpoint_manifest.get(
                        "previous_action_conditioning", False
                    )
                ):
                    if previous_action is None:
                        raise ValueError("checkpoint requires a previous action")
                    state_parts.append(
                        previous_action.reshape(1, 1, 7).expand(
                            1, args.action_horizon, 7
                        )
                    )
                if bool(
                    joint_checkpoint_manifest.get("phase_conditioning", False)
                ):
                    phase_lengths = joint_checkpoint_manifest.get(
                        "phase_lengths_by_task", {}
                    )
                    if task not in phase_lengths:
                        raise ValueError(
                            f"task {task!r} has no stored phase length"
                        )
                    phase_steps = torch.arange(
                        args.action_horizon, device=device, dtype=torch.float32
                    )
                    phase_steps.add_(step).clamp_max_(
                        int(phase_lengths[task]) - 1
                    )
                    phase = (
                        2.0
                        * phase_steps
                        / max(int(phase_lengths[task]) - 1, 1)
                        - 1.0
                    ).reshape(1, args.action_horizon, 1)
                    state_parts.append(phase)
                action_state = torch.cat(state_parts, dim=-1)
                actions = action_head(
                    model_input,
                    state=action_state,
                    h3_features=features,
                    action_timestep=torch.zeros(1, device=device),
                    language_features=(
                        conditioning["context"]
                        if bool(
                            joint_checkpoint_manifest.get(
                                "explicit_language_conditioning", False
                            )
                        )
                        else None
                    ),
                    history_h3_features=history_features,
                )
            else:
                generator = torch.Generator(device=device).manual_seed(int(seed))
                actions = torch.randn(
                    (1, args.action_horizon, 7),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                state_parts = [
                    state.reshape(1, 1, 8).expand(
                        1, args.action_horizon, 8
                    )
                ]
                if bool(
                    joint_checkpoint_manifest.get(
                        "previous_action_conditioning", False
                    )
                ):
                    if previous_action is None:
                        raise ValueError("checkpoint requires a previous action")
                    state_parts.append(
                        previous_action.reshape(1, 1, 7).expand(
                            1, args.action_horizon, 7
                        )
                    )
                if bool(
                    joint_checkpoint_manifest.get("phase_conditioning", False)
                ):
                    phase_lengths = joint_checkpoint_manifest.get(
                        "phase_lengths_by_task", {}
                    )
                    if task not in phase_lengths:
                        raise ValueError(
                            f"task {task!r} has no stored phase length"
                        )
                    phase_steps = torch.arange(
                        args.action_horizon, device=device, dtype=torch.float32
                    )
                    phase_steps.add_(step).clamp_max_(
                        int(phase_lengths[task]) - 1
                    )
                    phase = (
                        2.0
                        * phase_steps
                        / max(int(phase_lengths[task]) - 1, 1)
                        - 1.0
                    ).reshape(1, args.action_horizon, 1)
                    state_parts.append(phase)
                action_state = torch.cat(state_parts, dim=-1)
                u = torch.linspace(1.0, 0.0, args.flow_steps + 1, device=device)
                sigma = args.action_flow_shift * u / (
                    1.0 + (args.action_flow_shift - 1.0) * u
                )
                for index in range(args.flow_steps):
                    action_timestep = (1.0 - sigma[index]).reshape(1)
                    velocity = action_head(
                        actions,
                        state=action_state,
                        h3_features=features,
                        action_timestep=action_timestep,
                        language_features=(
                            conditioning["context"]
                            if bool(
                                joint_checkpoint_manifest.get(
                                    "explicit_language_conditioning", False
                                )
                            )
                            else None
                        ),
                        history_h3_features=history_features,
                    )
                    actions = actions + velocity * (sigma[index] - sigma[index + 1])
        torch.cuda.synchronize(device)
        return (
            actions[0],
            context_id,
            time.perf_counter() - started,
            gate_probability,
            recovery_active,
            selected_action_mode,
            selected_action_mode_source,
            action_mode_probabilities,
        )

    listener = None
    connection = None
    previous_first = None
    previous_episode_key = None
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
                        "checkpoint_step": checkpoint_step,
                        "action_mode": action_mode,
                        "action_ensemble_size": len(local_action_members),
                        "world_size": world_size,
                        "load_seconds": load_seconds,
                    }
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "event": "ready",
                        "checkpoint_step": checkpoint_step,
                        "action_mode": action_mode,
                        "action_ensemble_size": len(local_action_members),
                        "world_size": world_size,
                        "load_seconds": load_seconds,
                    }
                ),
                flush=True,
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
                    connection.send({"ok": False, "error": f"unknown command {command!r}"})
                continue
            task = _broadcast_object(request["task"] if rank == 0 else None, rank)
            seed = int(_broadcast_object(request["seed"] if rank == 0 else None, rank))
            step = int(
                _broadcast_object(
                    request.get("step", 0) if rank == 0 else None, rank
                )
            )
            episode_key = _broadcast_object(
                request.get("episode_key", f"{task}:{seed}")
                if rank == 0
                else None,
                rank,
            )
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
            needs_history_frame = bool(
                joint_checkpoint_manifest is not None
                and joint_checkpoint_manifest.get(
                    "history_frame_conditioning", False
                )
            )
            history_first = None
            if needs_history_frame:
                history_first = (
                    first
                    if step == 0
                    or previous_first is None
                    or episode_key != previous_episode_key
                    else previous_first
                )
            needs_previous_action = bool(
                (
                    local_action_checkpoint is not None
                    and local_action_checkpoint.get("use_previous_action", False)
                )
                or (
                    joint_checkpoint_manifest is not None
                    and joint_checkpoint_manifest.get(
                        "previous_action_conditioning", False
                    )
                )
            )
            previous_action = None
            if needs_previous_action:
                previous_environment_action = _broadcast_object(
                    request.get("previous_environment_action") if rank == 0 else None,
                    rank,
                )
                if previous_environment_action is None:
                    raise ValueError("request has no previous_environment_action")
                previous_action = minmax_normalize(
                    libero_dataset_action(previous_environment_action),
                    stats["action_min"],
                    stats["action_max"],
                ).clamp(-1.0, 1.0).to(device)
            (
                normalized,
                context_id,
                inference_seconds,
                gate_probability,
                recovery_active,
                selected_action_mode,
                selected_action_mode_source,
                action_mode_probabilities,
            ) = policy_forward(
                task,
                first,
                history_first,
                state,
                previous_action,
                seed,
                step,
            )
            if needs_history_frame:
                previous_first = first.detach().clone()
                previous_episode_key = episode_key
            if rank == 0:
                normalized = normalized.clone()
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
                            "checkpoint_step": checkpoint_step,
                            "context_id": context_id,
                            "inference_seconds": inference_seconds,
                            "vae_encode_seconds": vae_seconds,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                            / 1024**3,
                            "first_environment_action": environment_actions[0].tolist(),
                            "environment_action_chunk": environment_actions.tolist(),
                            "action_head_switch_gate_probability": gate_probability,
                            "action_head_selected_index": int(recovery_active),
                            "selected_action_mode": selected_action_mode,
                            "selected_action_mode_source": (
                                selected_action_mode_source
                                if selected_action_mode_source is not None
                                else (
                                    "learned_recovery_gate"
                                    if recovery_gate is not None
                                    else None
                                )
                            ),
                            "action_mode_probabilities": action_mode_probabilities,
                        },
                    }
                )
    except BaseException:
        print(
            f"rank {rank} failed:\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if rank == 0:
            if connection is not None:
                connection.close()
            if listener is not None:
                listener.close()
            args.ready_file.resolve().unlink(missing_ok=True)
        attention_hooks.close()
        capture.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
