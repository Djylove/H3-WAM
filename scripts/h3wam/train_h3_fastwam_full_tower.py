#!/usr/bin/env python3
"""Train the C58 full30 official-FastWAM action tower on frozen H3 K/V.

The data, shifted-flow objective, normalization and dense-window loader are
imported from the audited D0 trainer.  The sole architecture change is the
function-preserving expansion from D0's five action blocks to the official
FastWAM 30-block ActionDiT tower.  H3 remains frozen and layer49 is repeated
for every action block.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    FASTWAM_ACTION_DIT_SHA256,
    FASTWAM_COMMIT,
    FASTWAM_GRADIENT_SHA256,
    FASTWAM_MOT_SHA256,
    FASTWAM_VIDEO_DIT_SHA256,
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)
from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Dataset,
    C58OnlineFrozenH3Provider,
    attach_online_h3_kv,
    collate_c58_online,
    move_c58_online_batch,
)


def _load_parent():
    path = Path(__file__).with_name("train_h3_int8_dreamwam_kv_carrier.py")
    spec = importlib.util.spec_from_file_location("_c58_d0_parent_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load D0 trainer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = _load_parent()
CHECKPOINT_SCHEMA = 1
CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "completed_steps",
        "model",
        "optimizer",
        "lr_scheduler",
        "contract",
        "probe_prediction",
        "probe_sample_ids",
        "rng_states",
        "data_state",
    }
)
DATA_STATE_KEYS = PARENT.DATA_STATE_KEYS
REPEAT_LAYER49_MODE = PARENT.REPEAT_LAYER49_CARRIER_SOURCE
LAYERWISE_H3_50_TO_ACTION_30_MODE = "uniform_h3_50_to_action30"


@dataclass(frozen=True)
class ModelSpec:
    action_dim: int = 7
    proprio_dim: int = 8
    context_dim: int = 5120
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 56
    attn_head_dim: int = 128
    freq_dim: int = 256
    action_layers: int = 30
    carrier_layers: tuple[int, ...] = DEFAULT_H3_CARRIER_LAYERS
    carrier_source_mode: str = REPEAT_LAYER49_MODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir")
    parser.add_argument(
        "--online-h3-checkpoint",
        type=Path,
        help=(
            "Run frozen INT8 H3 independently on every rank and keep K/V in "
            "memory. This mode never reads a precomputed H3 K/V cache."
        ),
    )
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument("--verify-h3-checkpoint-sha256", action="store_true")
    parser.add_argument(
        "--expected-h3-checkpoint-sha256", default=PARENT.H3_INT8_CHECKPOINT_SHA256
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--probe-sample-offset",
        type=int,
        help="Fixed restore probe row; defaults to the invocation sample offset.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-horizon", type=int, default=6000)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument("--capture-token-count", type=int, default=32)
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--carrier-mode",
        choices=(REPEAT_LAYER49_MODE, LAYERWISE_H3_50_TO_ACTION_30_MODE),
        default=REPEAT_LAYER49_MODE,
        help=(
            "C58 repeats layer49; C58b maps 30 uniformly-spaced H3 layers "
            "one-to-one onto the 30 official ActionDiT blocks."
        ),
    )
    parser.add_argument(
        "--matched-d0-control",
        action="store_true",
        help=(
            "Use the exact five-layer D0 action expert with parent weights but "
            "a fresh optimizer/scheduler. All data and flow contracts remain "
            "identical to C58; this is the required depth-only control."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def distributed_setup() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def build_model(
    spec: ModelSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool = False,
) -> H3FastWAMFullTowerPolicy:
    if spec.action_layers == 5:
        if spec.carrier_source_mode != REPEAT_LAYER49_MODE:
            raise ValueError("matched D0 control requires repeat_layer49")
        parent_spec = PARENT.ModelSpec(
            action_dim=spec.action_dim,
            proprio_dim=spec.proprio_dim,
            context_dim=spec.context_dim,
            hidden_dim=spec.hidden_dim,
            ffn_dim=spec.ffn_dim,
            num_heads=spec.num_heads,
            attn_head_dim=spec.attn_head_dim,
            freq_dim=spec.freq_dim,
            carrier_layers=spec.carrier_layers,
            carrier_source_mode=spec.carrier_source_mode,
            history_action_steps=0,
        )
        return PARENT.build_model(parent_spec, device=device, dtype=dtype)
    if spec.action_layers != 30:
        raise ValueError("C58 pair permits only 5-layer control or 30-layer candidate")
    if spec.carrier_source_mode == REPEAT_LAYER49_MODE:
        block_mapping = None
    elif spec.carrier_source_mode == LAYERWISE_H3_50_TO_ACTION_30_MODE:
        block_mapping = LAYERWISE_H3_50_TO_ACTION_30
    else:
        raise ValueError(f"unsupported C58 carrier mode {spec.carrier_source_mode!r}")
    return H3FastWAMFullTowerPolicy(
        enabled=True,
        carrier_layers=spec.carrier_layers,
        action_dim=spec.action_dim,
        proprio_dim=spec.proprio_dim,
        context_dim=spec.context_dim,
        hidden_dim=spec.hidden_dim,
        ffn_dim=spec.ffn_dim,
        num_heads=spec.num_heads,
        attn_head_dim=spec.attn_head_dim,
        freq_dim=spec.freq_dim,
        num_layers=spec.action_layers,
        use_gradient_checkpointing=gradient_checkpointing,
        action_block_to_h3_layer=block_mapping,
    ).to(device=device, dtype=dtype)


def forward_policy(
    model: nn.Module,
    batch: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    autocast = (
        torch.autocast(device_type=noisy.device.type, dtype=torch.bfloat16)
        if noisy.dtype == torch.bfloat16 and noisy.device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with autocast:
        return model(
            noisy,
            timesteps,
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            video_kv_cache=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
        )


def require_d0_parent(
    payload: dict[str, Any],
    *,
    dataset: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if payload.get("schema_version") != PARENT.CHECKPOINT_SCHEMA:
        raise ValueError("D0 parent checkpoint schema mismatch")
    if set(payload) != PARENT.CHECKPOINT_KEYS:
        raise ValueError("D0 parent top-level fields mismatch")
    contract = payload.get("contract")
    required = {
        "candidate": "D0",
        "classification": "action-only-on-frozen-features",
        "carrier_source_mode": PARENT.REPEAT_LAYER49_CARRIER_SOURCE,
        "h3_checkpoint_sha256": args.expected_h3_checkpoint_sha256,
        "split_manifest_sha256": dataset.manifest_sha256,
        "stats_sha256": dataset.stats_sha256,
        "action_horizon": args.action_horizon,
        "action_shift": args.action_shift,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
    }
    mismatches = {
        key: {"parent": contract.get(key), "required": value}
        for key, value in required.items()
        if not isinstance(contract, dict) or contract.get(key) != value
    }
    model_spec = {} if not isinstance(contract, dict) else contract.get("model_spec", {})
    expected_spec = {
        "action_dim": 7,
        "proprio_dim": 8,
        "context_dim": 5120,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 56,
        "attn_head_dim": 128,
        "freq_dim": 256,
        "carrier_layers": list(DEFAULT_H3_CARRIER_LAYERS),
        "carrier_source_mode": PARENT.REPEAT_LAYER49_CARRIER_SOURCE,
    }
    for key, value in expected_spec.items():
        actual = model_spec.get(key)
        if key == "carrier_layers" and isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            mismatches[f"model_spec.{key}"] = {"parent": actual, "required": value}
    if mismatches:
        raise ValueError(f"D0 parent contract mismatch: {mismatches}")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("D0 parent model state is empty")
    return contract


def checkpoint_contract(
    args: argparse.Namespace,
    spec: ModelSpec,
    dataset: Any,
    *,
    world_size: int,
    d0_parent_sha256: str,
    d0_parent_steps: int,
    initialization_report: dict[str, Any],
) -> dict[str, Any]:
    layerwise = spec.carrier_source_mode == LAYERWISE_H3_50_TO_ACTION_30_MODE
    matched_control = spec.action_layers == 5
    online_h3 = args.online_h3_checkpoint is not None
    contract = {
        "candidate": (
            "C58B_FASTWAM_FULL30_H3_LAYERWISE"
            if layerwise
            else (
                "C58_MATCHED_D0_FRESH_OPTIMIZER"
                if matched_control
                else "C58_FASTWAM_FULL30_H3_LAYER49"
            )
        ),
        "classification": (
            "action-only-on-frozen-layerwise-h3-kv_backbone_port"
            if layerwise
            else (
                "matched-five-layer-depth-control_fresh-optimizer"
                if matched_control
                else "action-only-on-frozen-features_backbone_port"
            )
        ),
        "fastwam_commit": FASTWAM_COMMIT,
        "fastwam_action_dit_sha256": FASTWAM_ACTION_DIT_SHA256,
        "fastwam_video_dit_sha256": FASTWAM_VIDEO_DIT_SHA256,
        "fastwam_gradient_sha256": FASTWAM_GRADIENT_SHA256,
        "fastwam_mot_sha256": FASTWAM_MOT_SHA256,
        "d0_parent_sha256": d0_parent_sha256,
        "d0_parent_completed_steps": d0_parent_steps,
        "d0_parent_optimizer_restored": False,
        "initialization": initialization_report,
        "carrier_source_mode": spec.carrier_source_mode,
        "h3_checkpoint_path": str(dataset.first_checkpoint_path),
        "h3_checkpoint_sha256": args.expected_h3_checkpoint_sha256,
        "verify_h3_checkpoint_sha256": args.verify_h3_checkpoint_sha256,
        "h3_execution": (
            "online_frozen_int8_per_rank_v1" if online_h3 else "precomputed_kv_v1"
        ),
        "disk_kv_training_input": not online_h3,
        "kv_subdir": None if online_h3 else args.kv_subdir,
        "kv_schema": PARENT.DREAMWAM_KV_SCHEMA,
        "kv_strategy": PARENT.DREAMWAM_KV_STRATEGY,
        "kv_layers": list(spec.carrier_layers),
        "kv_tokens": args.capture_token_count,
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "source_manifest_items": dataset.source_manifest_items,
        "split_manifest_sha256": dataset.manifest_sha256,
        "split_manifest_items": dataset.manifest_items,
        "stats_sha256": dataset.stats_sha256,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
        "action_horizon": args.action_horizon,
        "action_shift": args.action_shift,
        "flow_timestep_contract": "continuous_fp32_no_bf16_endpoint_rounding_v2",
        "flow_rng_contract": "base_plus_step1000003_plus_rank10000019_v2",
        "training_topology": {
            "world_size": world_size,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
        "optimizer": {
            "type": "AdamW",
            "betas": [0.9, 0.95],
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "parent_state_policy": "fresh_for_both_c58_and_matched_d0_control",
        },
        "lr_schedule": {
            "type": "linear_warmup_then_cosine",
            "warmup_steps": args.warmup_steps,
            "scheduler_horizon": args.scheduler_horizon,
            "min_learning_rate": args.min_learning_rate,
        },
        "model_spec": asdict(spec),
        "gradient_checkpointing": args.use_gradient_checkpointing,
    }
    if layerwise:
        contract["action_block_to_h3_layer"] = list(
            LAYERWISE_H3_50_TO_ACTION_30
        )
    return contract


def validate_c58_checkpoint(
    payload: dict[str, Any], expected_contract: dict[str, Any]
) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or set(payload) != CHECKPOINT_KEYS:
        raise ValueError("C58 checkpoint schema mismatch")
    if payload.get("contract") != expected_contract:
        actual = payload.get("contract", {})
        keys = sorted(
            key
            for key in set(actual) | set(expected_contract)
            if actual.get(key) != expected_contract.get(key)
        )
        raise ValueError(f"C58 checkpoint contract mismatch: {keys}")
    data_state = payload.get("data_state")
    if not isinstance(data_state, dict) or set(data_state) != DATA_STATE_KEYS:
        raise ValueError("C58 checkpoint data state mismatch")


def infinite_batches(loader: Iterable[dict[str, Any]]):
    while True:
        yield from loader


def probe_dataset_selection(
    args: argparse.Namespace,
    *,
    manifest_rows: list[dict[str, Any]] | None = None,
    loaded_checkpoint: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Keep the initialization probe inside the exact training slice.

    C58b's first cached canary starts deep in the dense manifest.  Returning
    to row zero here silently makes the trainer depend on a cache outside its
    declared slice and, more importantly, probes a different sample contract.
    """

    probe_offset = getattr(args, "probe_sample_offset", None)
    if probe_offset is None and loaded_checkpoint is not None:
        probe_ids = loaded_checkpoint.get("probe_sample_ids")
        if not isinstance(probe_ids, list) or len(probe_ids) != 1:
            raise ValueError("loaded C58 checkpoint must freeze exactly one probe id")
        if manifest_rows is None:
            raise ValueError("manifest rows are required to restore a frozen probe")
        matches = [
            index
            for index, row in enumerate(manifest_rows)
            if str(row["id"]) == str(probe_ids[0])
        ]
        if len(matches) != 1:
            raise ValueError(
                "loaded C58 checkpoint probe id is not unique in the manifest"
            )
        probe_offset = matches[0]
    if probe_offset is None:
        probe_offset = args.sample_offset
    return {"limit": 1, "sample_offset": int(probe_offset)}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    tensor_bytes = PARENT._tensor_bytes(payload)
    required = int(tensor_bytes * 1.1) + 64 * 1024**2
    available = shutil.disk_usage(path.parent).free
    if available < required:
        raise OSError(
            f"insufficient free space for C58 checkpoint: {available} < {required}"
        )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def main() -> None:
    args = parse_args()
    if args.restore_check_only and args.load_checkpoint is None:
        raise ValueError("--restore-check-only requires --load-checkpoint")
    if args.matched_d0_control and args.carrier_mode != REPEAT_LAYER49_MODE:
        raise ValueError("matched D0 control cannot use the layer-wise carrier")
    online_h3 = args.online_h3_checkpoint is not None
    if online_h3 and (
        args.carrier_mode != LAYERWISE_H3_50_TO_ACTION_30_MODE
        or args.matched_d0_control
    ):
        raise ValueError("online H3 is restricted to the C58b layer-wise full30 path")
    if not online_h3 and not args.kv_subdir:
        raise ValueError("cached training requires --kv-subdir")
    if online_h3 and args.per_device_batch_size != 1:
        raise ValueError("online H3 requires per-device-batch-size=1")
    positive = (
        args.steps,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.action_horizon,
        args.capture_token_count,
        args.learning_rate,
        args.max_grad_norm,
        args.action_shift,
        args.scheduler_horizon,
    )
    if min(positive) <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("invalid positive training arguments")

    rank, world_size, device = distributed_setup()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    started = time.perf_counter()
    manifest_rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loaded_payload = (
        None
        if args.load_checkpoint is None
        else torch.load(
            args.load_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
    )
    probe_selection = probe_dataset_selection(
        args,
        manifest_rows=manifest_rows,
        loaded_checkpoint=loaded_payload,
    )

    carrier_layers = (
        LAYERWISE_H3_50_TO_ACTION_30
        if args.carrier_mode == LAYERWISE_H3_50_TO_ACTION_30_MODE
        else DEFAULT_H3_CARRIER_LAYERS
    )
    if online_h3:
        dataset = C58OnlineFrozenH3Dataset(
            args.manifest,
            args.source_manifest,
            args.cache_root,
            args.online_h3_checkpoint,
            action_horizon=args.action_horizon,
            limit=args.limit,
            sample_offset=args.sample_offset,
        )
    else:
        dataset = PARENT.CachedDreamWAMKVDataset(
            args.manifest,
            args.cache_root,
            args.kv_subdir,
            source_manifest=args.source_manifest,
            carrier_layers=carrier_layers,
            capture_token_count=args.capture_token_count,
            kv_pool_strategy=PARENT.DREAMWAM_KV_STRATEGY,
            num_heads=56,
            attn_head_dim=128,
            action_horizon=args.action_horizon,
            limit=args.limit,
            sample_offset=args.sample_offset,
        )
    actual_h3_sha256 = PARENT.verify_h3_checkpoint_sha256(
        dataset.first_checkpoint_path,
        expected_sha256=args.expected_h3_checkpoint_sha256,
        enabled=args.verify_h3_checkpoint_sha256,
        rank=rank,
    )
    d0_path = args.d0_parent_checkpoint.resolve()
    d0_payload = torch.load(d0_path, map_location="cpu", weights_only=False)
    require_d0_parent(d0_payload, dataset=dataset, args=args)
    parent_consumed = set(d0_payload["data_state"]["sample_ids"])
    selected_now = {str(row["id"]) for row in dataset.rows}
    parent_overlap = parent_consumed & selected_now
    if parent_overlap:
        raise ValueError(
            "C58 must continue on rows disjoint from the D0 parent, but the "
            f"requested stage overlaps {len(parent_overlap)} consumed rows"
        )
    d0_sha256 = sha256_file(d0_path) if rank == 0 else None
    if dist.is_initialized():
        values = [d0_sha256]
        dist.broadcast_object_list(values, src=0)
        d0_sha256 = values[0]
    assert isinstance(d0_sha256, str)

    spec = ModelSpec(
        carrier_layers=carrier_layers,
        carrier_source_mode=args.carrier_mode,
        action_layers=5 if args.matched_d0_control else 30,
    )
    model = build_model(
        spec,
        device=device,
        dtype=dtype,
        gradient_checkpointing=args.use_gradient_checkpointing,
    )
    if args.matched_d0_control:
        model.load_state_dict(d0_payload["model"], strict=True)
        initialization = {
            "source_layers": 5,
            "target_layers": 5,
            "anchor_target_indices": [0, 1, 2, 3, 4],
            "nearest_source_indices": [0, 1, 2, 3, 4],
            "identity_target_indices": [],
            "source_prefix": "action_expert.blocks",
            "target_prefix": "action_expert.blocks",
            "alpha_scaling_applied": False,
            "width_interpolation_applied": False,
            "initialization_contract": "exact_d0_weights_fresh_optimizer_v1",
        }
    else:
        initialization = initialize_full_tower_from_d0(
            model, d0_payload["model"]
        ).to_dict()
    contract = checkpoint_contract(
        args,
        spec,
        dataset,
        world_size=world_size,
        d0_parent_sha256=d0_sha256,
        d0_parent_steps=int(d0_payload["completed_steps"]),
        initialization_report=initialization,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = PARENT.PARENT.build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        scheduler_horizon=args.scheduler_horizon,
        min_learning_rate=args.min_learning_rate,
    )

    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_c58_online if online_h3 else PARENT.collate_cached_batch,
    )
    if online_h3:
        probe_dataset = C58OnlineFrozenH3Dataset(
            args.manifest,
            args.source_manifest,
            args.cache_root,
            args.online_h3_checkpoint,
            action_horizon=args.action_horizon,
            **probe_selection,
        )
        probe_cpu = collate_c58_online([probe_dataset[0]])
        online_provider = C58OnlineFrozenH3Provider(
            args.online_h3_checkpoint, layers=carrier_layers
        ).to(device=device).eval()
    else:
        probe_dataset = PARENT.CachedDreamWAMKVDataset(
            args.manifest,
            args.cache_root,
            args.kv_subdir,
            source_manifest=args.source_manifest,
            carrier_layers=carrier_layers,
            capture_token_count=args.capture_token_count,
            kv_pool_strategy=PARENT.DREAMWAM_KV_STRATEGY,
            num_heads=56,
            attn_head_dim=128,
            action_horizon=args.action_horizon,
            **probe_selection,
        )
        probe_cpu = PARENT.collate_cached_batch([probe_dataset[0]])
        online_provider = None
    expected_probe_offset = probe_selection["sample_offset"]
    expected_probe_id = str(manifest_rows[expected_probe_offset]["id"])
    if probe_cpu["sample_ids"] != [expected_probe_id]:
        raise RuntimeError("C58 initialization probe escaped the training slice")
    if online_h3:
        probe = move_c58_online_batch(probe_cpu, device, dtype)
        probe = attach_online_h3_kv(probe, online_provider)
    else:
        probe = PARENT.move_batch(probe_cpu, device, dtype)
    flow_scheduler = PARENT.FlowMatchScheduler(
        num_train_timesteps=1000, shift=args.action_shift
    )

    # Prove C58's depth expansion exactly preserves D0.  For C58b, prove the
    # same property under a degenerate all-layer49 carrier, then separately
    # record the effect of replacing it with the real layer-wise H3 carrier.
    d0_spec_values = dict(d0_payload["contract"]["model_spec"])
    d0_spec_values["carrier_layers"] = tuple(d0_spec_values["carrier_layers"])
    d0_model = PARENT.build_model(
        PARENT.ModelSpec(**d0_spec_values), device=device, dtype=dtype
    )
    d0_model.load_state_dict(d0_payload["model"], strict=True)
    d0_model.eval()
    model.eval()
    with torch.no_grad():
        parity_noisy, _, parity_t = PARENT.PARENT.deterministic_flow_batch(
            probe["actions"], flow_scheduler, seed=args.seed + 8_000_001
        )
        if args.carrier_mode == REPEAT_LAYER49_MODE:
            parent_probe = probe
            expanded_probe = probe
        else:
            layer49 = probe["video_kv_cache"][49]
            parent_probe = dict(probe)
            parent_probe["video_kv_cache"] = {
                layer: {name: tensor.clone() for name, tensor in layer49.items()}
                for layer in DEFAULT_H3_CARRIER_LAYERS
            }
            expanded_probe = dict(probe)
            expanded_probe["video_kv_cache"] = {
                layer: {name: tensor.clone() for name, tensor in layer49.items()}
                for layer in LAYERWISE_H3_50_TO_ACTION_30
            }
        parent_prediction = PARENT.forward_policy(
            d0_model, parent_probe, parity_noisy, parity_t
        ).float()
        expanded_prediction = forward_policy(
            model, expanded_probe, parity_noisy, parity_t
        ).float()
        actual_carrier_prediction = forward_policy(
            model, probe, parity_noisy, parity_t
        ).float()
    step0_max_abs = float((expanded_prediction - parent_prediction).abs().max())
    layerwise_carrier_delta = float(
        (actual_carrier_prediction - expanded_prediction).abs().max()
    )
    if step0_max_abs != 0.0:
        raise RuntimeError(
            f"C58/C58b degenerate-carrier D0 parity failed: max_abs={step0_max_abs}"
        )
    if (
        args.carrier_mode == LAYERWISE_H3_50_TO_ACTION_30_MODE
        and layerwise_carrier_delta <= 0.0
    ):
        raise RuntimeError("C58b real layer-wise carrier has no effect at step zero")
    del d0_model
    del d0_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()

    completed_steps = 0
    restore_max_abs = None
    if args.load_checkpoint is not None:
        assert loaded_payload is not None
        validate_c58_checkpoint(loaded_payload, contract)
        model.load_state_dict(loaded_payload["model"], strict=True)
        optimizer.load_state_dict(loaded_payload["optimizer"])
        lr_scheduler.load_state_dict(loaded_payload["lr_scheduler"])
        completed_steps = int(loaded_payload["completed_steps"])
        if not args.restore_check_only:
            overlap = set(loaded_payload["data_state"]["sample_ids"]) & {
                str(row["id"]) for row in dataset.rows
            }
            if overlap:
                raise ValueError(
                    f"C58 stage slice overlaps {len(overlap)} previously consumed rows"
                )
        model.eval()
        with torch.no_grad():
            noisy, _, timesteps = PARENT.PARENT.deterministic_flow_batch(
                probe["actions"], flow_scheduler, seed=args.seed + 9_000_001
            )
            restored = forward_policy(model, probe, noisy, timesteps).float()
        expected = loaded_payload["probe_prediction"].to(restored)
        if loaded_payload["probe_sample_ids"] != probe_cpu["sample_ids"]:
            raise ValueError("C58 restore probe sample identity mismatch")
        restore_max_abs = float((restored - expected).abs().max())
        if restore_max_abs != 0.0:
            raise RuntimeError(f"C58 restore prediction mismatch: {restore_max_abs}")

    unwrapped = model
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        unwrapped = model.module

    history: list[dict[str, Any]] = []
    if not args.restore_check_only:
        model.train()
        iterator = iter(infinite_batches(loader))
        optimizer.zero_grad(set_to_none=True)
        tracked = unwrapped.action_expert.head.weight
        tracked_before = tracked.detach().float().clone()
        for local_step in range(1, args.steps + 1):
            losses: list[float] = []
            sample_ids: list[str] = []
            timestep_means: list[float] = []
            flow_seeds: list[int] = []
            for accumulation_index in range(args.gradient_accumulation_steps):
                batch_cpu = next(iterator)
                if online_h3:
                    assert online_provider is not None
                    batch = move_c58_online_batch(batch_cpu, device, dtype)
                    batch = attach_online_h3_kv(batch, online_provider)
                else:
                    batch = PARENT.move_batch(batch_cpu, device, dtype)
                sample_ids.extend(batch["sample_ids"])
                flow_seed = PARENT.PARENT.distributed_flow_seed(
                    base_seed=args.seed,
                    completed_step=completed_steps + local_step,
                    accumulation_index=accumulation_index,
                    rank=rank,
                )
                flow_seeds.append(flow_seed)
                noisy, target, timesteps = PARENT.PARENT.deterministic_flow_batch(
                    batch["actions"],
                    flow_scheduler,
                    seed=flow_seed,
                )
                prediction = forward_policy(model, batch, noisy, timesteps)
                loss = PARENT.flow_matching_loss(
                    prediction,
                    target,
                    timesteps,
                    flow_scheduler,
                    is_pad_mask=batch["action_is_pad"],
                )
                (loss / args.gradient_accumulation_steps).backward()
                losses.append(float(loss.detach()))
                timestep_means.append(float(timesteps.float().mean()))
            block_gradients = [
                PARENT.PARENT.module_grad_norm(block)
                for block in unwrapped.action_expert.blocks
            ]
            proprio_gradient = PARENT.PARENT.module_grad_norm(
                unwrapped.proprio_encoder
            )
            all_gradients = [*block_gradients, proprio_gradient]
            if not all(math.isfinite(value) and value > 0 for value in all_gradients):
                raise RuntimeError(f"C58 non-finite/zero gradient path: {all_gradients}")
            clipped = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update = float((tracked.detach().float() - tracked_before).abs().max())
            if not math.isfinite(update) or update <= 0:
                raise RuntimeError("C58 optimizer did not update the action head")
            tracked_before = tracked.detach().float().clone()
            record = {
                "step": completed_steps + local_step,
                "loss": sum(losses) / len(losses),
                "timestep_mean": sum(timestep_means) / len(timestep_means),
                "block_gradient_norms": block_gradients,
                "proprio_gradient_norm": proprio_gradient,
                "clipped_gradient_norm": clipped,
                "head_update_max_abs": update,
                "learning_rate": learning_rate,
                "sample_ids": sample_ids,
                "flow_seeds": flow_seeds,
            }
            history.append(record)
            if rank == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
        completed_steps += args.steps

    unwrapped.eval()
    with torch.no_grad():
        noisy, _, timesteps = PARENT.PARENT.deterministic_flow_batch(
            probe["actions"], flow_scheduler, seed=args.seed + 9_000_001
        )
        probe_prediction = forward_policy(unwrapped, probe, noisy, timesteps).float()
    if not torch.isfinite(probe_prediction).all() or float(probe_prediction.std()) <= 0:
        raise RuntimeError("C58 probe prediction is non-finite or constant")

    checkpoint_bytes = None
    if args.save_checkpoint is not None and not args.restore_check_only:
        local_rng = PARENT.capture_rng_state(device)
        local_ids = [sample_id for record in history for sample_id in record["sample_ids"]]
        if dist.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, {"rng": local_rng, "ids": local_ids})
            if any(item is None for item in gathered):
                raise RuntimeError("failed to gather C58 rank state")
            complete = [item for item in gathered if item is not None]
        else:
            complete = [{"rng": local_rng, "ids": local_ids}]
        stage_ids = PARENT.flatten_consumed_sample_ids(
            [item["ids"] for item in complete],
            expected_per_rank=(
                args.per_device_batch_size
                * args.gradient_accumulation_steps
                * args.steps
            ),
        )
        historical = (
            [] if loaded_payload is None else loaded_payload["data_state"]["sample_ids"]
        )
        cumulative = PARENT.merge_cumulative_consumed_sample_ids(
            list(historical), stage_ids
        )
        data_state = {
            "resume_mode": "explicit_stage_slice_v1",
            "sample_offset": args.sample_offset,
            "limit": args.limit,
            "selected_windows": len(dataset),
            "steps_in_invocation": args.steps,
            "sample_ids": cumulative,
            "sampler_cursor_restorable": False,
        }
        if rank == 0:
            checkpoint_bytes = _atomic_torch_save(
                {
                    "schema_version": CHECKPOINT_SCHEMA,
                    "completed_steps": completed_steps,
                    "model": unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "contract": contract,
                    "probe_prediction": probe_prediction.cpu(),
                    "probe_sample_ids": list(probe_cpu["sample_ids"]),
                    "rng_states": [item["rng"] for item in complete],
                    "data_state": data_state,
                },
                args.save_checkpoint,
            )
        if dist.is_initialized():
            dist.barrier()

    local_rank_audit = {
        "rank": rank,
        "sample_ids": [
            sample_id for record in history for sample_id in record["sample_ids"]
        ],
        "flow_seeds": [seed for record in history for seed in record["flow_seeds"]],
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    if dist.is_initialized():
        rank_audits: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_audits, local_rank_audit)
        if any(item is None for item in rank_audits):
            raise RuntimeError("failed to gather C58 per-rank runtime audit")
    else:
        rank_audits = [local_rank_audit]

    if rank == 0:
        trainable_parameters = sum(
            parameter.numel() for parameter in unwrapped.parameters() if parameter.requires_grad
        )
        report = {
            "event": (
                "h3_c58b_online_frozen_h3_full30_train"
                if online_h3
                else (
                    "h3_c58_matched_d0_fresh_optimizer_probe"
                    if args.matched_d0_control
                    else "h3_c58_fastwam_full30_probe"
                )
            ),
            "classification": contract["classification"],
            "status": "mechanical_probe_not_effectiveness_evidence",
            "resolved_argv": sys.argv,
            "world_size": world_size,
            "global_batch_size": (
                world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
            ),
            "selected_windows": len(dataset),
            "manifest_items": dataset.manifest_items,
            "completed_steps": completed_steps,
            "training_samples": (
                0
                if args.restore_check_only
                else world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
                * args.steps
            ),
            "effective_epochs": (
                0.0
                if args.restore_check_only
                else world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
                * args.steps
                / dataset.manifest_items
            ),
            "trainable_parameters": trainable_parameters,
            "step0_parent_parity_max_abs": step0_max_abs,
            "step0_layerwise_vs_degenerate_layer49_max_abs": layerwise_carrier_delta,
            "contract": contract,
            "h3_checkpoint_sha256_verified": actual_h3_sha256,
            "loaded_checkpoint": (
                None if args.load_checkpoint is None else str(args.load_checkpoint)
            ),
            "saved_checkpoint": (
                None if args.save_checkpoint is None else str(args.save_checkpoint)
            ),
            "checkpoint_file_size_bytes": checkpoint_bytes,
            "restore_probe_max_abs": restore_max_abs,
            "probe_prediction_mean": float(probe_prediction.mean()),
            "probe_prediction_std": float(probe_prediction.std()),
            "history": history,
            "per_rank_runtime": rank_audits,
            "elapsed_seconds": time.perf_counter() - started,
            "seconds_per_step": (
                None
                if args.restore_check_only or not history
                else (time.perf_counter() - started) / len(history)
            ),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
