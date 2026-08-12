#!/usr/bin/env python3
"""Jointly tune official BF16 MiniMax-H3 video dynamics and LIBERO actions.

This is the full-rank H3-WAM path.  It keeps H3's native video-flow target,
adds a lightweight multi-depth action expert, and lets both losses update the
selected H3 transformer blocks under FSDP.  Observation-row attention is
masked from future video/audio rows so action features cannot leak targets.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import shutil
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_h3_bf16_fsdp import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    KEYFRAME_TIMESTEP,
    audio_latent_count,
    load_training_rows,
    replicated_non_block_modules,
    set_trainable_tail,
    shifted_video_timestep,
    validate_cached_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--task-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--last-blocks", type=int, default=50)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep released H3 fixed and train only the multi-layer action head.",
    )
    parser.add_argument(
        "--frozen-action-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For a frozen backbone, train from dense first-frame/action caches "
            "without storing or scoring future-video latents."
        ),
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--action-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-objective",
        choices=("flow", "regression"),
        default="flow",
        help=(
            "Train stochastic action-flow velocity (the historical default) or "
            "directly regress a deterministic normalized action chunk."
        ),
    )
    parser.add_argument(
        "--explicit-language-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Expose the raw H3 text-encoder tokens directly to the action decoder "
            "instead of relying only on their indirect effect on video rows."
        ),
    )
    parser.add_argument(
        "--phase-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append a task-agnostic normalized episode-progress coordinate to "
            "each action query. The scale is the median demonstration length "
            "for the requested language task."
        ),
    )
    parser.add_argument(
        "--previous-action-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the normalized action immediately before each window.",
    )
    parser.add_argument(
        "--previous-action-cache",
        type=Path,
        help="Compact sidecar produced by build_libero_previous_action_cache.py.",
    )
    parser.add_argument(
        "--initialize-action-from",
        type=Path,
        help=(
            "Warm-start the action head without resuming optimizer/scheduler state. "
            "New conditioning columns are zero initialized for step-zero equivalence."
        ),
    )
    parser.add_argument(
        "--train-previous-action-projection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze the inherited action head and update only the seven new "
            "previous-action columns of state_projection.weight."
        ),
    )
    parser.add_argument(
        "--history-frame-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Condition each action decoder layer on a separately encoded earlier "
            "observation through a zero-initialized residual gate."
        ),
    )
    parser.add_argument(
        "--history-frame-map",
        type=Path,
        help="Sidecar produced by build_libero_history_frame_map.py.",
    )
    parser.add_argument(
        "--train-history-gate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the inherited action head and train only its history gate.",
    )
    parser.add_argument(
        "--history-adapter-rank",
        type=int,
        default=0,
        help="Per-action-layer low-rank nonlinear adapter over history-current deltas.",
    )
    parser.add_argument(
        "--train-history-adapter-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze inherited parameters and train only the temporal adapter.",
    )
    parser.add_argument("--action-flow-shift", type=float, default=5.0)
    parser.add_argument(
        "--action-loss-reweight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--empty-cache-before-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-hidden-dim", type=int, default=1024)
    parser.add_argument("--action-layers", type=int, default=1)
    parser.add_argument("--action-heads", type=int, default=16)
    parser.add_argument("--action-ffn-dim", type=int, default=4096)
    parser.add_argument(
        "--layer-mix-initialization",
        choices=("spaced", "uniform"),
        default="spaced",
    )
    parser.add_argument(
        "--layer-mix-learning-rate",
        type=float,
        help="Optional dedicated AdamW learning rate for H3 depth routing.",
    )
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        help="Official H3 block indices; default captures all 50 layers.",
    )
    parser.add_argument(
        "--fp32-master-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-batches-per-rank", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--routing-diagnostics-every",
        type=int,
        default=0,
        help=(
            "Emit H3 layer-feature diversity, layer-mix gradients, and the "
            "optimizer update every N steps. Zero disables diagnostics."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def minmax_normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scale = (high.float() - low.float()).clamp_min(1e-6)
    return ((value.float() - low.float()) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def cosine_with_warmup_factor(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    """Return a restart-safe warmup + cosine learning-rate multiplier."""

    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def checkpoint_steps(output_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^step(\d{6})$")
    checkpoints = []
    if not output_dir.exists():
        return checkpoints
    for path in output_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints)


def action_state_slices(
    *, previous_action_conditioning: bool, phase_conditioning: bool
) -> dict[str, slice]:
    offset = 0
    result = {"proprio": slice(offset, offset + 8)}
    offset += 8
    if previous_action_conditioning:
        result["previous_action"] = slice(offset, offset + 7)
        offset += 7
    if phase_conditioning:
        result["phase"] = slice(offset, offset + 1)
    return result


def migrate_state_projection(
    saved: torch.Tensor,
    target: torch.Tensor,
    *,
    saved_previous_action: bool,
    saved_phase: bool,
    target_previous_action: bool,
    target_phase: bool,
) -> torch.Tensor:
    if saved.ndim != 2 or target.ndim != 2 or saved.shape[0] != target.shape[0]:
        raise ValueError("state projection migration requires matching 2-D output width")
    old_slices = action_state_slices(
        previous_action_conditioning=saved_previous_action,
        phase_conditioning=saved_phase,
    )
    new_slices = action_state_slices(
        previous_action_conditioning=target_previous_action,
        phase_conditioning=target_phase,
    )
    migrated = torch.zeros_like(target)
    for name in old_slices.keys() & new_slices.keys():
        old_slice, new_slice = old_slices[name], new_slices[name]
        if old_slice.stop - old_slice.start != new_slice.stop - new_slice.start:
            raise ValueError(f"conditioning width changed for {name}")
        migrated[:, new_slice] = saved[:, old_slice].to(migrated.dtype)
    return migrated


def migrate_action_initialization_state(
    saved_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    *,
    saved_previous_action: bool,
    saved_phase: bool,
    target_previous_action: bool,
    target_phase: bool,
    target_history: bool,
    target_history_adapter: bool = False,
) -> dict[str, torch.Tensor]:
    missing_keys = current_state.keys() - saved_state.keys()
    unexpected_keys = saved_state.keys() - current_state.keys()
    allowed_missing = {"history_gate"} if target_history else set()
    if target_history_adapter:
        allowed_missing.update(
            name
            for name in current_state
            if name.startswith(("history_down.", "history_up."))
        )
    if not missing_keys <= allowed_missing or unexpected_keys:
        raise ValueError(
            "action initialization checkpoint keys do not match: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
        )
    migrated_state = {}
    for name, target in current_state.items():
        if name in missing_keys:
            migrated_state[name] = target
            continue
        saved = saved_state[name]
        if name == "state_projection.weight" and saved.shape != target.shape:
            saved = migrate_state_projection(
                saved,
                target,
                saved_previous_action=saved_previous_action,
                saved_phase=saved_phase,
                target_previous_action=target_previous_action,
                target_phase=target_phase,
            )
        if saved.shape != target.shape:
            raise ValueError(
                f"action initialization shape mismatch for {name}: "
                f"{tuple(saved.shape)} != {tuple(target.shape)}"
            )
        migrated_state[name] = saved
    return migrated_state


def main() -> None:
    args = parse_args()
    if (
        args.steps <= 0
        or args.gradient_accumulation_steps <= 0
        or args.last_blocks <= 0
        or args.action_horizon <= 0
    ):
        raise ValueError(
            "steps, gradient-accumulation-steps, last-blocks and action-horizon "
            "must be positive"
        )
    if args.video_loss_weight <= 0 or args.action_loss_weight <= 0:
        raise ValueError("joint loss weights must be positive")
    if args.layer_mix_learning_rate is not None and args.layer_mix_learning_rate <= 0:
        raise ValueError("layer-mix-learning-rate must be positive")
    if args.routing_diagnostics_every < 0:
        raise ValueError("routing-diagnostics-every cannot be negative")
    if args.frozen_action_only and not args.freeze_backbone:
        raise ValueError("frozen-action-only requires freeze-backbone")
    if args.resume_from is not None and args.initialize_action_from is not None:
        raise ValueError("resume-from and initialize-action-from are mutually exclusive")
    if args.train_previous_action_projection_only:
        if not args.previous_action_conditioning or args.initialize_action_from is None:
            raise ValueError(
                "previous-action projection-only training requires both previous "
                "action conditioning and initialize-action-from"
            )
        if args.weight_decay != 0.0:
            raise ValueError(
                "previous-action projection-only training requires zero weight decay"
            )
    if args.history_frame_conditioning:
        if not args.freeze_backbone or not args.frozen_action_only:
            raise ValueError(
                "history-frame conditioning currently requires frozen-action-only"
            )
        if args.history_frame_map is None:
            raise ValueError("history-frame conditioning requires history-frame-map")
    if args.train_history_gate_only:
        if not args.history_frame_conditioning or args.initialize_action_from is None:
            raise ValueError(
                "history-gate-only training requires history conditioning and "
                "initialize-action-from"
            )
        if args.weight_decay != 0.0:
            raise ValueError("history-gate-only training requires zero weight decay")
    if args.train_previous_action_projection_only and args.train_history_gate_only:
        raise ValueError("only one projection/gate-only training mode may be active")
    if args.history_adapter_rank < 0:
        raise ValueError("history-adapter-rank cannot be negative")
    if args.history_adapter_rank and not args.history_frame_conditioning:
        raise ValueError("history adapter requires history-frame conditioning")
    if args.train_history_adapter_only:
        if (
            args.history_adapter_rank <= 0
            or not args.history_frame_conditioning
            or args.initialize_action_from is None
        ):
            raise ValueError(
                "history-adapter-only training requires a positive adapter rank, "
                "history conditioning and initialize-action-from"
            )
        if args.weight_decay != 0.0:
            raise ValueError("history-adapter-only training requires zero weight decay")
    if sum(
        int(value)
        for value in (
            args.train_previous_action_projection_only,
            args.train_history_gate_only,
            args.train_history_adapter_only,
        )
    ) > 1:
        raise ValueError("only one projection/gate/adapter-only mode may be active")
    if args.action_objective == "flow" and args.action_flow_shift <= 0:
        raise ValueError("action-flow-shift must be positive")
    if args.validation_every < 0 or args.validation_batches_per_rank <= 0:
        raise ValueError("invalid validation settings")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        raise ValueError("warmup-steps must be non-negative and smaller than steps")
    if not 0.0 <= args.minimum_lr_ratio <= 1.0:
        raise ValueError("minimum-lr-ratio must be in [0, 1]")
    if args.checkpoint_every < 0 or args.keep_last_checkpoints <= 0:
        raise ValueError("invalid checkpoint settings")
    if (args.validation_manifest is None) != (args.validation_every == 0):
        raise ValueError(
            "validation-manifest and positive validation-every must be set together"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 2 or not torch.cuda.is_available():
        raise RuntimeError("joint H3-WAM training requires multi-GPU CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    video_generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 10_000 + rank
    )

    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
    )
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from fastwam.models.h3wam import (
        H3BlockAttentionMask,
        H3MultiLayerActionTransformer,
        H3OfficialFeatureCapture,
        build_h3_observation_attention_mask,
    )
    from fastwam.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    output_dir = args.output_dir.resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    data_root = args.data_root.resolve()

    def read_action_only_manifest(path: Path) -> list[dict]:
        rows = [
            json.loads(line)
            for line in path.resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any("id" not in row for row in rows):
            raise ValueError("training manifest must contain sample IDs")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError("training manifest contains duplicate sample IDs")
        # Dense caches can contain O(100k) small files. Repeating is_file() for
        # every row on every rank creates hundreds of thousands of metadata
        # operations. prepare_row validates the concrete window/context pair
        # before it is used, while cache construction verifies the total count.
        return rows

    training_rows = (
        read_action_only_manifest(args.manifest.resolve())
        if args.frozen_action_only
        else load_training_rows(data_root, args.manifest.resolve(), None)
    )
    random.Random(args.seed).shuffle(training_rows)
    validation_rows = []
    if args.validation_manifest is not None:
        validation_rows = (
            read_action_only_manifest(args.validation_manifest.resolve())
            if args.frozen_action_only
            else load_training_rows(
                data_root, args.validation_manifest.resolve(), None
            )
        )
    training_by_task: dict[str, list[dict]] = {}
    validation_by_task: dict[str, list[dict]] = {}
    if args.task_balanced:
        for rows, grouped in (
            (training_rows, training_by_task),
            (validation_rows, validation_by_task),
        ):
            for row in rows:
                task = str(row.get("task", ""))
                if not task:
                    raise ValueError("task-balanced manifests require a task per row")
                grouped.setdefault(task, []).append(row)
    training_tasks = sorted(training_by_task)
    validation_tasks = sorted(validation_by_task)
    phase_tasks = sorted({str(row.get("task", "")) for row in training_rows})
    if args.phase_conditioning and (not phase_tasks or phase_tasks[0] == ""):
        raise ValueError("phase conditioning requires a task per training row")
    phase_lengths_by_task = {
        task: int(
            round(
                statistics.median(
                    int(row["length"])
                    for row in training_rows
                    if str(row.get("task", "")) == task
                )
            )
        )
        for task in phase_tasks
        if task
    }
    previous_actions_by_id: dict[str, torch.Tensor] = {}
    previous_action_cache_path = None
    if args.previous_action_conditioning:
        previous_action_cache_path = (
            data_root / "previous_actions.pt"
            if args.previous_action_cache is None
            else args.previous_action_cache.resolve()
        )
        payload = torch.load(
            previous_action_cache_path, map_location="cpu", weights_only=False
        )
        if payload.get("format") != "h3wam-libero-previous-action-v1":
            raise ValueError("unsupported previous-action cache format")
        ids, values = payload["ids"], payload["values"].float()
        if values.ndim != 2 or tuple(values.shape) != (len(ids), 7):
            raise ValueError("previous-action cache has an invalid shape")
        previous_actions_by_id = dict(zip(map(str, ids), values.unbind(0), strict=True))
        missing = sorted(
            str(row["id"])
            for row in (*training_rows, *validation_rows)
            if str(row["id"]) not in previous_actions_by_id
        )
        if missing:
            raise ValueError(f"previous-action cache misses rows: {missing[:3]}")
    history_source_by_id: dict[str, str] = {}
    history_frame_map_path = None
    if args.history_frame_conditioning:
        history_frame_map_path = args.history_frame_map.resolve()
        payload = torch.load(
            history_frame_map_path, map_location="cpu", weights_only=False
        )
        if payload.get("format") != "h3wam-libero-history-frame-map-v1":
            raise ValueError("unsupported history-frame map format")
        ids, source_ids = payload["ids"], payload["source_ids"]
        if len(ids) != len(source_ids):
            raise ValueError("history-frame map ids and source_ids differ in length")
        history_source_by_id = dict(
            zip(map(str, ids), map(str, source_ids), strict=True)
        )
        required_ids = {
            str(row["id"]) for row in (*training_rows, *validation_rows)
        }
        missing = sorted(required_ids - history_source_by_id.keys())
        if missing:
            raise ValueError(f"history-frame map misses rows: {missing[:3]}")
        unavailable = sorted(
            source_id
            for source_id in {history_source_by_id[row_id] for row_id in required_ids}
            if not (data_root / "windows" / f"{source_id}.pt").is_file()
        )
        if unavailable:
            raise ValueError(f"history-frame cache windows are missing: {unavailable[:3]}")
    validation_rows_for_eval = validation_rows
    validation_by_task_for_eval = validation_by_task
    validation_tasks_for_eval = validation_tasks
    if args.history_frame_conditioning and validation_rows:
        validation_rows_for_eval = [
            row
            for row in validation_rows
            if history_source_by_id[str(row["id"])] != str(row["id"])
        ]
        if not validation_rows_for_eval:
            raise ValueError("validation manifest has no nonzero-history windows")
        if args.task_balanced:
            validation_by_task_for_eval = {}
            for row in validation_rows_for_eval:
                validation_by_task_for_eval.setdefault(str(row["task"]), []).append(row)
            validation_tasks_for_eval = sorted(validation_by_task_for_eval)
            missing_tasks = sorted(set(validation_tasks) - set(validation_tasks_for_eval))
            if missing_tasks:
                raise ValueError(
                    "validation tasks have no nonzero-history windows: "
                    f"{missing_tasks[:3]}"
                )
    action_state_dim = (
        8 + 7 * int(args.previous_action_conditioning) + int(args.phase_conditioning)
    )

    def select_row(
        rows: list[dict], grouped: dict[str, list[dict]], tasks: list[str], index: int
    ) -> dict:
        if not args.task_balanced:
            return rows[index % len(rows)]
        task = tasks[index % len(tasks)]
        task_cycle = index // len(tasks)
        return grouped[task][task_cycle % len(grouped[task])]

    stats = torch.load(data_root / "stats.pt", map_location="cpu", weights_only=False)
    language_feature_dim = None
    if args.explicit_language_conditioning:
        first_context_id = str(
            training_rows[0].get("context_id", training_rows[0]["id"])
        )
        first_conditioning = torch.load(
            data_root / "contexts" / f"{first_context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        language_feature_dim = int(first_conditioning["context"].shape[-1])
        del first_conditioning

    load_started = time.perf_counter()
    h3 = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model.resolve(),
        subfolder="transformer",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.freeze_backbone:
        h3.requires_grad_(False)
        trainable_names: set[str] = set()
    else:
        trainable_names = set_trainable_tail(h3, args.last_blocks)
    if args.fp32_master_weights and not args.freeze_backbone:
        for block in h3.transformer_blocks[-args.last_blocks :]:
            block.to(torch.float32)
    capture_layers = tuple(
        range(len(h3.transformer_blocks))
        if args.capture_layers is None
        else sorted(set(args.capture_layers))
    )
    if not capture_layers or capture_layers[0] < 0 or capture_layers[-1] >= len(
        h3.transformer_blocks
    ):
        raise ValueError("capture-layers are outside the H3 backbone")
    if args.freeze_backbone:
        h3.disable_gradient_checkpointing()
        h3.eval()
    else:
        h3.enable_gradient_checkpointing()
        h3.train()
    replicated_modules = replicated_non_block_modules(h3)
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
    patch_size = tuple(h3.module.config.patch_size)
    h3_feature_dim = int(h3.module.config.hidden_size)
    torch.manual_seed(args.seed)
    action_head = H3MultiLayerActionTransformer(
        action_dim=7,
        state_dim=action_state_dim,
        num_h3_layers=len(capture_layers),
        h3_feature_dim=h3_feature_dim,
        hidden_dim=args.action_hidden_dim,
        num_layers=args.action_layers,
        num_heads=args.action_heads,
        ffn_dim=args.action_ffn_dim,
        max_horizon=max(64, args.action_horizon),
        language_feature_dim=language_feature_dim,
        layer_mix_initialization=args.layer_mix_initialization,
        history_conditioning=args.history_frame_conditioning,
        history_adapter_rank=args.history_adapter_rank,
    ).to(device)
    initialized_action_step = None
    if args.initialize_action_from is not None:
        initialization_dir = args.initialize_action_from.resolve()
        initialization_manifest = json.loads(
            (initialization_dir / "manifest.json").read_text(encoding="utf-8")
        )
        initialized_action_step = int(initialization_manifest["step"])
        if rank == 0:
            saved_state = torch.load(
                initialization_dir / "action_head.pt",
                map_location="cpu",
                weights_only=True,
            )
            current_state = action_head.state_dict()
            migrated_state = migrate_action_initialization_state(
                saved_state,
                current_state,
                saved_previous_action=bool(
                    initialization_manifest.get(
                        "previous_action_conditioning", False
                    )
                ),
                saved_phase=bool(
                    initialization_manifest.get("phase_conditioning", False)
                ),
                target_previous_action=args.previous_action_conditioning,
                target_phase=args.phase_conditioning,
                target_history=args.history_frame_conditioning,
                target_history_adapter=args.history_adapter_rank > 0,
            )
            action_head.load_state_dict(migrated_state, strict=True)
        dist.barrier()
    if args.train_previous_action_projection_only:
        action_head.requires_grad_(False)
        state_projection_weight = action_head.state_projection.weight
        state_projection_weight.requires_grad_(True)
        previous_slice = action_state_slices(
            previous_action_conditioning=True,
            phase_conditioning=args.phase_conditioning,
        )["previous_action"]
        previous_action_gradient_mask = torch.zeros_like(state_projection_weight)
        previous_action_gradient_mask[:, previous_slice] = 1.0
        state_projection_weight.register_hook(
            lambda gradient: gradient * previous_action_gradient_mask
        )
    if args.train_history_gate_only:
        action_head.requires_grad_(False)
        assert action_head.history_gate is not None
        action_head.history_gate.requires_grad_(True)
    if args.train_history_adapter_only:
        action_head.requires_grad_(False)
        assert action_head.history_down is not None
        assert action_head.history_up is not None
        action_head.history_down.requires_grad_(True)
        action_head.history_up.requires_grad_(True)
    action_head = DDP(action_head, device_ids=[local_rank], broadcast_buffers=False)
    action_flow_scheduler = WanContinuousFlowMatchScheduler(
        shift=args.action_flow_shift
    )
    h3_trainable = [parameter for parameter in h3.parameters() if parameter.requires_grad]
    layer_mix_parameters = [action_head.module.layer_mix_logits]
    layer_mix_parameter_ids = {id(parameter) for parameter in layer_mix_parameters}
    action_parameters = [
        parameter
        for parameter in action_head.parameters()
        if id(parameter) not in layer_mix_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": h3_trainable, "lr": args.backbone_learning_rate},
            {"params": action_parameters, "lr": args.action_learning_rate},
            {
                "params": layer_mix_parameters,
                "lr": (
                    args.action_learning_rate
                    if args.layer_mix_learning_rate is None
                    else args.layer_mix_learning_rate
                ),
            },
        ],
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_factor(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
            minimum_ratio=args.minimum_lr_ratio,
        ),
    )
    feature_capture = H3OfficialFeatureCapture(
        h3.module.transformer_blocks,
        capture_layers,
        torch.tensor([0]),
    )
    attention_mask_hooks = H3BlockAttentionMask(h3.module.transformer_blocks)
    load_seconds = time.perf_counter() - load_started

    def save_checkpoint(step: int) -> Path:
        """Save BF16 rank-local model shards without the prohibitively large Adam state."""

        checkpoint_dir = output_dir / f"step{step:06d}"
        partial_dir = output_dir / f"step{step:06d}.partial"
        if rank == 0:
            if checkpoint_dir.exists() or partial_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite checkpoint path: {checkpoint_dir}"
                )
            # A full-H3 checkpoint is roughly 65 GB.  Prune before writing so
            # retaining N checkpoints never needs temporary disk space for
            # N+1.  At least N-1 known-good checkpoints remain if this save
            # is interrupted.
            old_checkpoints = checkpoint_steps(output_dir)
            while len(old_checkpoints) >= args.keep_last_checkpoints:
                old_step, old_path = old_checkpoints.pop(0)
                shutil.rmtree(old_path)
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_pruned",
                            "step": old_step,
                            "path": str(old_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            partial_dir.mkdir(parents=False)
        dist.barrier()
        local_parameters = {
            name: parameter.detach().to(device="cpu", dtype=torch.bfloat16)
            for name, parameter in h3.named_parameters()
            if parameter.requires_grad
        }
        rank_payload = {
            "format": "h3wam-fsdp-local-bf16-v1",
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "parameters": local_parameters,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "video_generator_state": video_generator.get_state(),
            "action_generator_state": action_generator.get_state(),
        }
        rank_path = partial_dir / f"h3_rank{rank:05d}.pt"
        rank_temporary = partial_dir / f".h3_rank{rank:05d}.{os.getpid()}.tmp"
        torch.save(rank_payload, rank_temporary)
        os.replace(rank_temporary, rank_path)
        del rank_payload, local_parameters
        if rank == 0:
            action_state = {
                name: value.detach().to(
                    device="cpu",
                    dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                )
                for name, value in action_head.module.state_dict().items()
            }
            action_temporary = partial_dir / f".action_head.{os.getpid()}.tmp"
            torch.save(action_state, action_temporary)
            os.replace(action_temporary, partial_dir / "action_head.pt")
            torch.save(
                {
                    "step": step,
                    "total_steps": args.steps,
                    "world_size": world_size,
                    "warmup_steps": args.warmup_steps,
                    "minimum_lr_ratio": args.minimum_lr_ratio,
                    "scheduler": scheduler.state_dict(),
                    "optimizer_state_saved": False,
                    "optimizer_resume_policy": "fresh AdamW moments",
                },
                partial_dir / "trainer_state.pt",
            )
        dist.barrier()
        if rank == 0:
            files = {
                path.name: path.stat().st_size
                for path in sorted(partial_dir.iterdir())
                if path.is_file()
            }
            (partial_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "h3wam-fsdp-local-bf16-v1",
                        "step": step,
                        "world_size": world_size,
                        "action_objective": args.action_objective,
                        "backbone_frozen": args.freeze_backbone,
                        "frozen_action_only": args.frozen_action_only,
                        "last_blocks": args.last_blocks,
                        "explicit_language_conditioning": (
                            args.explicit_language_conditioning
                        ),
                        "language_feature_dim": language_feature_dim,
                        "phase_conditioning": args.phase_conditioning,
                        "previous_action_conditioning": (
                            args.previous_action_conditioning
                        ),
                        "history_frame_conditioning": (
                            args.history_frame_conditioning
                        ),
                        "history_frame_map": (
                            None
                            if history_frame_map_path is None
                            else str(history_frame_map_path)
                        ),
                        "train_history_gate_only": args.train_history_gate_only,
                        "history_adapter_rank": args.history_adapter_rank,
                        "train_history_adapter_only": (
                            args.train_history_adapter_only
                        ),
                        "phase_lengths_by_task": phase_lengths_by_task,
                        "action_state_dim": action_state_dim,
                        "action_horizon": args.action_horizon,
                        "action_hidden_dim": args.action_hidden_dim,
                        "action_layers": args.action_layers,
                        "action_heads": args.action_heads,
                        "action_ffn_dim": args.action_ffn_dim,
                        "capture_layers": capture_layers,
                        "layer_mix_initialization": args.layer_mix_initialization,
                        "layer_mix_learning_rate": (
                            args.action_learning_rate
                            if args.layer_mix_learning_rate is None
                            else args.layer_mix_learning_rate
                        ),
                        "files": files,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(partial_dir, checkpoint_dir)
        dist.barrier()
        return checkpoint_dir

    def load_checkpoint(checkpoint_dir: Path) -> int:
        checkpoint_dir = checkpoint_dir.resolve()
        manifest_path = checkpoint_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["world_size"]) != world_size:
            raise ValueError(
                f"checkpoint world size {manifest['world_size']} != current {world_size}"
            )
        rank_payload = torch.load(
            checkpoint_dir / f"h3_rank{rank:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        current = {
            name: parameter
            for name, parameter in h3.named_parameters()
            if parameter.requires_grad
        }
        saved = rank_payload["parameters"]
        if current.keys() != saved.keys():
            missing = sorted(current.keys() - saved.keys())
            unexpected = sorted(saved.keys() - current.keys())
            raise ValueError(
                f"checkpoint parameter mismatch; missing={missing[:3]}, "
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
        action_state = torch.load(
            checkpoint_dir / "action_head.pt", map_location="cpu", weights_only=True
        )
        action_head.module.load_state_dict(action_state, strict=True)
        trainer_state = torch.load(
            checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        expected_schedule = (
            int(trainer_state.get("total_steps", -1)),
            int(trainer_state.get("warmup_steps", -1)),
            float(trainer_state.get("minimum_lr_ratio", -1.0)),
        )
        current_schedule = (
            args.steps,
            args.warmup_steps,
            args.minimum_lr_ratio,
        )
        if expected_schedule != current_schedule:
            raise ValueError(
                f"resume schedule mismatch: checkpoint={expected_schedule}, "
                f"current={current_schedule}"
            )
        scheduler.load_state_dict(trainer_state["scheduler"])
        for group, learning_rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = learning_rate
        torch.set_rng_state(rank_payload["torch_rng_state"])
        torch.cuda.set_rng_state(rank_payload["cuda_rng_state"], device)
        video_generator.set_state(rank_payload["video_generator_state"])
        action_generator.set_state(rank_payload["action_generator_state"])
        return int(trainer_state["step"])

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from)
        if start_step >= args.steps:
            raise ValueError(
                f"resume step {start_step} must be smaller than total steps {args.steps}"
            )
        dist.barrier()

    def prepare_row(row: dict) -> dict:
        sample_id = str(row["id"])
        context_id = str(row.get("context_id", sample_id))
        window_path = data_root / "windows" / f"{sample_id}.pt"
        context_path = data_root / "contexts" / f"{context_id}.pt"
        if not window_path.is_file() or not context_path.is_file():
            raise FileNotFoundError(
                f"missing cached window/context pair: {window_path}, {context_path}"
            )
        window = torch.load(window_path, map_location="cpu", weights_only=False)
        conditioning = torch.load(context_path, map_location="cpu", weights_only=False)
        if args.frozen_action_only:
            for key in ("first_frame_latents", "actions", "state"):
                if key not in window:
                    raise ValueError(f"dense action cache {sample_id} lacks {key}")
        else:
            validate_cached_sample(window, conditioning)
        actions = window["actions"][: args.action_horizon]
        if tuple(actions.shape) != (args.action_horizon, 7):
            raise ValueError(
                f"{row['id']} actions {tuple(actions.shape)} do not match "
                f"({args.action_horizon},7)"
            )
        clean = (
            None
            if args.frozen_action_only
            else window["video_latents"].to(device=device, dtype=torch.float32)
        )
        first = window["first_frame_latents"].to(device=device, dtype=torch.float32)
        history_first = None
        history_source_id = None
        if args.history_frame_conditioning:
            history_source_id = history_source_by_id[sample_id]
            history_window = torch.load(
                data_root / "windows" / f"{history_source_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if "first_frame_latents" not in history_window:
                raise ValueError(
                    f"history cache {history_source_id} lacks first_frame_latents"
                )
            history_first = history_window["first_frame_latents"].to(
                device=device, dtype=torch.float32
            )
            if tuple(history_first.shape) != tuple(first.shape):
                raise ValueError(
                    f"history/current latent shape mismatch for {sample_id}: "
                    f"{tuple(history_first.shape)} != {tuple(first.shape)}"
                )
        context = conditioning["context"].to(device=device, dtype=torch.float32)
        text_tags = conditioning["token_tags"].to(dtype=torch.long)
        if clean is None:
            _, _, _, latent_height, latent_width = first.shape
            pixel_frames = int(window.get("h3_frame_count", 39))
            num_latent_frames = (
                2
                if pixel_frames == 5
                else ((pixel_frames - 5) // 17) * 5 + 2
            )
        else:
            _, _, num_latent_frames, latent_height, latent_width = clean.shape
            pixel_frames = int(window.get("h3_frame_count", num_latent_frames))
        num_audio_latents = audio_latent_count(pixel_frames)
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=text_tags,
            num_latent_frames=num_latent_frames,
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
        normalized_state = minmax_normalize(
            window["state"], stats["state_min"], stats["state_max"]
        ).to(device)
        state_parts = [
            normalized_state.reshape(1, 8).expand(args.action_horizon, -1)
        ]
        if args.previous_action_conditioning:
            normalized_previous_action = minmax_normalize(
                previous_actions_by_id[sample_id],
                stats["action_min"],
                stats["action_max"],
            ).to(device)
            state_parts.append(
                normalized_previous_action.reshape(1, 7).expand(
                    args.action_horizon, -1
                )
            )
        if args.phase_conditioning:
            phase_steps = torch.arange(
                args.action_horizon, device=device, dtype=torch.float32
            )
            phase_steps.add_(int(row["start"])).clamp_max_(int(row["length"]) - 1)
            phase = (
                2.0 * phase_steps / max(int(row["length"]) - 1, 1) - 1.0
            ).unsqueeze(-1)
            state_parts.append(phase)
        action_state = torch.cat(state_parts, dim=-1)
        return {
            "id": str(row["id"]),
            "clean": clean,
            "first": first,
            "history_first": history_first,
            "history_source_id": history_source_id,
            "num_latent_frames": num_latent_frames,
            "context": context,
            "num_audio_latents": num_audio_latents,
            "position_ids": position_ids.to(device),
            "token_tags": token_tags.to(device),
            "video_indices": video_indices.to(device),
            "audio_indices": audio_indices.to(device),
            "text_indices": text_indices.to(device),
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
            "actions": minmax_normalize(
                actions, stats["action_min"], stats["action_max"]
            ).to(device),
            "state": action_state,
        }

    routing_diagnostics_enabled = False
    latest_routing_feature_diagnostics: dict | None = None

    def joint_loss(batch: dict, *, deterministic_seed: int | None = None):
        nonlocal latest_routing_feature_diagnostics
        if deterministic_seed is None:
            video_rng, action_rng = video_generator, action_generator
        else:
            video_rng = torch.Generator(device=device).manual_seed(deterministic_seed)
            action_rng = torch.Generator(device=device).manual_seed(
                deterministic_seed + 1
            )
        clean, first = batch["clean"], batch["first"]
        if args.frozen_action_only:
            video_timestep = torch.tensor(1.0, device=device)
            target = torch.zeros(
                (
                    1,
                    first.shape[1],
                    batch["num_latent_frames"],
                    first.shape[-2],
                    first.shape[-1],
                ),
                device=device,
                dtype=torch.float32,
            )
            noisy_video = target
            target_video = None
            noised_first = first
        else:
            video_timestep = shifted_video_timestep(generator=video_rng, device=device)
            video_noise = torch.randn(
                clean.shape, generator=video_rng, device=device, dtype=torch.float32
            )
            noisy_video = video_timestep * clean + (1.0 - video_timestep) * video_noise
            target_video = patchify_video_latents(clean - video_noise, patch_size)[None]
            condition_noise = torch.randn(
                first.shape, generator=video_rng, device=device, dtype=torch.float32
            )
            noised_first = (
                KEYFRAME_TIMESTEP * first
                + (1.0 - KEYFRAME_TIMESTEP) * condition_noise
            )
        video_rows = torch.cat(
            (patchify_video_latents(noised_first, patch_size), patchify_video_latents(noisy_video, patch_size)),
            dim=0,
        )[None]
        audio_shape = (
            1,
            batch["num_audio_latents"] * AUDIO_CHANNELS,
            AUDIO_LATENT_CHANNELS,
        )
        audio_rows = (
            torch.zeros(audio_shape, device=device, dtype=torch.float32)
            if args.frozen_action_only
            else torch.randn(
                audio_shape,
                generator=video_rng,
                device=device,
                dtype=torch.float32,
            )
        )
        unique_timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=batch["video_indices"].cpu(),
            audio_indices=batch["audio_indices"].cpu(),
            num_condition_video_rows=batch["num_condition_video_rows"],
            num_condition_audio_rows=batch["num_condition_audio_rows"],
            num_text_tokens=batch["text_indices"].numel(),
            video_timestep=float(video_timestep),
            audio_timestep=0.0,
            condition_video_timestep=max(float(video_timestep), KEYFRAME_TIMESTEP),
            condition_audio_timestep=1.0,
        )
        condition_indices = batch["video_indices"][: batch["num_condition_video_rows"]]
        attention_mask = build_h3_observation_attention_mask(
            sequence_length=int(batch["position_ids"].shape[0]),
            text_indices=batch["text_indices"],
            condition_video_indices=condition_indices,
            device=device,
        )
        feature_capture.set_condition_video_indices(condition_indices)
        attention_mask_hooks.set(attention_mask)
        def run_backbone(rows: torch.Tensor):
            backbone_context = (
                torch.no_grad() if args.freeze_backbone else contextlib.nullcontext()
            )
            with backbone_context:
                return h3(
                    hidden_states=rows,
                    audio_hidden_states=audio_rows,
                    encoder_hidden_states=batch["context"],
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=batch["token_tags"],
                    position_ids=batch["position_ids"],
                    video_indices=batch["video_indices"],
                    audio_indices=batch["audio_indices"],
                    text_indices=batch["text_indices"],
                    return_dict=True,
                )

        history_features = None
        if args.history_frame_conditioning:
            history_first = batch["history_first"]
            history_video_rows = torch.cat(
                (
                    patchify_video_latents(history_first, patch_size),
                    patchify_video_latents(noisy_video, patch_size),
                ),
                dim=0,
            )[None]
            run_backbone(history_video_rows)
            history_features = feature_capture.stacked()
        output = run_backbone(video_rows)
        features = feature_capture.stacked()
        if routing_diagnostics_enabled:
            flattened = features.detach().float()[0].reshape(len(capture_layers), -1)
            reference = flattened[0:1].expand_as(flattened)
            latest_routing_feature_diagnostics = {
                "feature_rms": flattened.square().mean(dim=-1).sqrt().cpu().tolist(),
                "cosine_to_first": F.cosine_similarity(
                    flattened, reference, dim=-1, eps=1e-8
                ).cpu().tolist(),
                "relative_l2_to_first": (
                    (flattened - reference).norm(dim=-1)
                    / reference.norm(dim=-1).clamp_min(1e-8)
                ).cpu().tolist(),
            }
        if args.frozen_action_only:
            video_loss = torch.zeros((), device=device, dtype=torch.float32)
        else:
            predicted_video = output.sample[:, batch["num_condition_video_rows"] :]
            video_loss = F.mse_loss(predicted_video.float(), target_video.float())

        clean_actions = batch["actions"].unsqueeze(0)
        if args.action_objective == "flow":
            action_uniform = torch.rand(
                (1,), generator=action_rng, device=device, dtype=torch.float32
            )
            action_sigma = args.action_flow_shift * action_uniform / (
                1.0 + (args.action_flow_shift - 1.0) * action_uniform
            )
            action_timestep = 1.0 - action_sigma
            action_noise = torch.randn(
                clean_actions.shape,
                generator=action_rng,
                device=device,
                dtype=torch.float32,
            )
            noisy_actions = action_timestep[:, None, None] * clean_actions + (
                1.0 - action_timestep[:, None, None]
            ) * action_noise
            target_actions = clean_actions - action_noise
        else:
            action_sigma = torch.zeros((1,), device=device, dtype=torch.float32)
            action_timestep = torch.zeros((1,), device=device, dtype=torch.float32)
            noisy_actions = torch.zeros_like(clean_actions)
            target_actions = clean_actions
        predicted_actions = action_head(
            noisy_actions,
            state=batch["state"].unsqueeze(0),
            h3_features=features,
            action_timestep=action_timestep,
            language_features=(
                batch["context"] if args.explicit_language_conditioning else None
            ),
            history_h3_features=history_features,
        )
        action_loss = F.mse_loss(predicted_actions.float(), target_actions.float())
        if args.action_objective == "flow" and args.action_loss_reweight:
            action_weight = action_flow_scheduler.training_weight(
                action_sigma * action_flow_scheduler.num_train_timesteps
            ).to(device=device, dtype=action_loss.dtype)
            action_loss = action_loss * action_weight
        total = args.video_loss_weight * video_loss + args.action_loss_weight * action_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("joint H3-WAM loss is non-finite")
        return total, video_loss, action_loss

    @torch.no_grad()
    def validate() -> dict[str, float]:
        h3.eval()
        action_head.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for index in range(args.validation_batches_per_rank):
            row_index = (index * world_size + rank) % len(validation_rows_for_eval)
            row = select_row(
                validation_rows_for_eval,
                validation_by_task_for_eval,
                validation_tasks_for_eval,
                row_index,
            )
            losses = joint_loss(
                prepare_row(row),
                deterministic_seed=args.seed + 100_000 + row_index,
            )
            totals += torch.tensor(
                [float(loss.detach()) for loss in losses],
                device=device,
                dtype=torch.float64,
            )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        totals /= args.validation_batches_per_rank * world_size
        h3.train(not args.freeze_backbone)
        action_head.train()
        return {
            "total": float(totals[0]),
            "video": float(totals[1]),
            "action": float(totals[2]),
        }

    metadata = {
        "event": "ready",
        "host": socket.gethostname(),
        "world_size": world_size,
        "model": str(args.model.resolve()),
        "data_root": str(data_root),
        "training_windows": len(training_rows),
        "validation_windows": len(validation_rows),
        "validation_windows_used": len(validation_rows_for_eval),
        "training_tasks": len(training_tasks) if args.task_balanced else None,
        "validation_tasks": len(validation_tasks) if args.task_balanced else None,
        "task_balanced": args.task_balanced,
        "action_objective": args.action_objective,
        "action_flow_shift": args.action_flow_shift,
        "action_loss_reweight": args.action_loss_reweight,
        "start_step": start_step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_size": world_size * args.gradient_accumulation_steps,
        "last_blocks": args.last_blocks,
        "backbone_frozen": args.freeze_backbone,
        "frozen_action_only": args.frozen_action_only,
        "explicit_language_conditioning": args.explicit_language_conditioning,
        "language_feature_dim": language_feature_dim,
        "phase_conditioning": args.phase_conditioning,
        "previous_action_conditioning": args.previous_action_conditioning,
        "previous_action_cache": (
            None
            if previous_action_cache_path is None
            else str(previous_action_cache_path)
        ),
        "history_frame_conditioning": args.history_frame_conditioning,
        "history_frame_map": (
            None
            if history_frame_map_path is None
            else str(history_frame_map_path)
        ),
        "initialize_action_from": (
            None
            if args.initialize_action_from is None
            else str(args.initialize_action_from.resolve())
        ),
        "initialized_action_step": initialized_action_step,
        "phase_lengths_by_task": phase_lengths_by_task,
        "action_state_dim": action_state_dim,
        "capture_layers": capture_layers,
        "action_horizon": args.action_horizon,
        "action_hidden_dim": args.action_hidden_dim,
        "action_layers": args.action_layers,
        "action_heads": args.action_heads,
        "action_ffn_dim": args.action_ffn_dim,
        "layer_mix_initialization": args.layer_mix_initialization,
        "layer_mix_learning_rate": (
            args.action_learning_rate
            if args.layer_mix_learning_rate is None
            else args.layer_mix_learning_rate
        ),
        "trainable_h3_parameters": sum(p.numel() for p in h3_trainable) * world_size,
        "action_parameters": sum(p.numel() for p in action_head.module.parameters()),
        "trainable_action_parameters": sum(
            p.numel() for p in action_head.module.parameters() if p.requires_grad
        ),
        "train_previous_action_projection_only": (
            args.train_previous_action_projection_only
        ),
        "train_history_gate_only": args.train_history_gate_only,
        "history_adapter_rank": args.history_adapter_rank,
        "train_history_adapter_only": args.train_history_adapter_only,
        "fp32_master_weights": args.fp32_master_weights,
        "warmup_steps": args.warmup_steps,
        "minimum_lr_ratio": args.minimum_lr_ratio,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_optimizer_state": False,
        "optimizer_foreach": False,
        "empty_cache_before_step": args.empty_cache_before_step,
        "load_seconds": load_seconds,
    }
    if rank == 0:
        print(json.dumps(metadata, sort_keys=True), flush=True)
    initial_validation = validate() if validation_rows and start_step == 0 else None
    if rank == 0 and initial_validation is not None:
        print(
            json.dumps({"event": "validation", "step": 0, **initial_validation}),
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_losses = None
    last_checkpoint_step = None
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = torch.zeros(3, device=device, dtype=torch.float64)
        diagnose_routing = (
            args.routing_diagnostics_every > 0
            and step % args.routing_diagnostics_every == 0
        )
        latest_routing_feature_diagnostics = None
        for micro_step in range(args.gradient_accumulation_steps):
            routing_diagnostics_enabled = diagnose_routing and micro_step == 0
            global_micro_step = (
                (step - 1) * args.gradient_accumulation_steps + micro_step
            )
            sample_index = (
                global_micro_step * world_size + rank
            )
            row = select_row(
                training_rows,
                training_by_task,
                training_tasks,
                sample_index,
            )
            losses = joint_loss(prepare_row(row))
            (losses[0] / args.gradient_accumulation_steps).backward()
            accumulated += torch.tensor(
                [float(loss.detach()) for loss in losses],
                device=device,
                dtype=torch.float64,
            ) / args.gradient_accumulation_steps
        routing_diagnostics_enabled = False
        layer_mix = action_head.module.layer_mix_logits
        layer_mix_before = layer_mix.detach().float().clone() if diagnose_routing else None
        layer_mix_gradient = (
            None
            if not diagnose_routing or layer_mix.grad is None
            else layer_mix.grad.detach().float().clone()
        )
        h3_grad_norm = (
            torch.zeros((), device=device)
            if args.freeze_backbone
            else h3.clip_grad_norm_(args.gradient_clip)
        )
        action_grad_norm = torch.nn.utils.clip_grad_norm_(
            action_head.parameters(), args.gradient_clip
        )
        if args.empty_cache_before_step:
            torch.cuda.empty_cache()
        optimizer.step()
        scheduler.step()
        if diagnose_routing and rank == 0:
            layer_mix_after = layer_mix.detach().float()
            print(
                json.dumps(
                    {
                        "event": "routing_diagnostics",
                        "step": step,
                        **(latest_routing_feature_diagnostics or {}),
                        "layer_mix_before": layer_mix_before.cpu().tolist(),
                        "layer_mix_gradient": (
                            None
                            if layer_mix_gradient is None
                            else layer_mix_gradient.cpu().tolist()
                        ),
                        "layer_mix_gradient_norm": (
                            None
                            if layer_mix_gradient is None
                            else float(layer_mix_gradient.norm())
                        ),
                        "layer_mix_update": (
                            layer_mix_after - layer_mix_before
                        ).cpu().tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        reduced = accumulated
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
        final_losses = {
            "total": float(reduced[0]),
            "video": float(reduced[1]),
            "action": float(reduced[2]),
        }
        if rank == 0 and (step % args.log_every == 0 or step == args.steps):
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        **final_losses,
                        "h3_grad_norm": float(h3_grad_norm),
                        "action_grad_norm": float(action_grad_norm),
                        "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                        "action_learning_rate": optimizer.param_groups[1]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if validation_rows and step % args.validation_every == 0:
            validation = validate()
            if rank == 0:
                print(
                    json.dumps(
                        {"event": "validation", "step": step, **validation},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            checkpoint_path = save_checkpoint(step)
            last_checkpoint_step = step
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "path": str(checkpoint_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if args.save_final and last_checkpoint_step != args.steps:
        checkpoint_path = save_checkpoint(args.steps)
        last_checkpoint_step = args.steps
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": args.steps,
                        "path": str(checkpoint_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    completed_steps = args.steps - start_step
    report = {
        **metadata,
        "event": "complete",
        "steps": args.steps,
        "completed_steps_this_run": completed_steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / completed_steps,
        "peak_allocated_gib_per_rank": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_train": final_losses,
        "initial_validation": initial_validation,
        "last_checkpoint_step": last_checkpoint_step,
    }
    if rank == 0:
        (output_dir / "joint_training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    feature_capture.close()
    attention_mask_hooks.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
