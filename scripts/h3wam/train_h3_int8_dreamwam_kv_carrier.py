#!/usr/bin/env python3
"""Train Candidate D over frozen, layer-specific MiniMax-H3 K/V caches.

This trainer is intentionally separate from the historical last32 StarWAM
trainer.  It is opt-in, action-only-on-frozen-features, and uses five pinned
DreamWAM ActionDiT blocks for H3 layers 9, 19, 29, 39 and 49.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    DREAMWAM_COMMIT,
    DREAMWAM_EXPERTS_SHA256,
    DREAMWAM_LAYERS_SHA256,
    DREAMWAM_MOT_SHA256,
    H3DreamWAMKVCarrierPolicy,
    h3_kv_cache_bytes,
)


def _load_parent_flow_contract():
    path = Path(__file__).with_name("train_h3_int8_starwam_action.py")
    spec = importlib.util.spec_from_file_location("_h3_last32_parent_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load parent shifted-flow trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = _load_parent_flow_contract()
FlowMatchScheduler = PARENT.FlowMatchScheduler
flow_matching_loss = PARENT.flow_matching_loss

DREAMWAM_KV_SCHEMA = "h3_dreamwam_kv_v1"
DREAMWAM_KV_STRATEGY = "adaptive_avg_pool1d_sequence_v1"
CACHE_BACKBONE = "H3Int8FeatureBackbone"
CACHE_QUANTIZATION = "int8_tensorwise_convrot"
CHECKPOINT_SCHEMA = 1
PARENT_OBJECTIVE_COMMIT = PARENT.STARWAM_COMMIT


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
    carrier_layers: tuple[int, ...] = DEFAULT_H3_CARRIER_LAYERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--kv-subdir", default="h3_int8_dreamwam_kv_5x32"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument(
        "--enable-dreamwam-kv-carrier",
        action="store_true",
        help="Required explicit opt-in; there is no implicit replacement of last32.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-horizon", type=int, default=21700)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument("--capture-token-count", type=int, default=32)
    parser.add_argument(
        "--carrier-layers",
        type=int,
        nargs="+",
        default=DEFAULT_H3_CARRIER_LAYERS,
    )
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--freq-dim", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_signature(tensor: torch.Tensor) -> int:
    """Identify shared storage even when aliases have different offsets."""

    return tensor.untyped_storage().data_ptr()


def collate_cached_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    context, context_mask = PARENT._pad_contexts(items)
    layers = tuple(items[0]["video_kv_cache"])
    if any(tuple(item["video_kv_cache"]) != layers for item in items):
        raise ValueError("batch samples disagree on carrier layer order")
    video_kv_cache = {
        layer: {
            name: torch.stack(
                [item["video_kv_cache"][layer][name] for item in items]
            )
            for name in ("k", "v")
        }
        for layer in layers
    }
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "video_kv_cache": video_kv_cache,
        "actions": torch.stack([item["actions"] for item in items]),
        "proprio": torch.stack([item["proprio"] for item in items]),
        "action_is_pad": torch.stack([item["action_is_pad"] for item in items]),
        "text_context": context,
        "text_mask": context_mask,
    }


class CachedDreamWAMKVDataset(Dataset):
    """Strict reader for the independent five-layer projected K/V schema."""

    def __init__(
        self,
        manifest: Path,
        cache_root: Path,
        kv_subdir: str,
        *,
        source_manifest: Path | None = None,
        carrier_layers: tuple[int, ...] = DEFAULT_H3_CARRIER_LAYERS,
        capture_token_count: int = 32,
        num_heads: int = 56,
        attn_head_dim: int = 128,
        action_horizon: int = 32,
        limit: int = 0,
        sample_offset: int = 0,
    ) -> None:
        self.manifest = manifest.resolve()
        self.cache_root = cache_root.resolve()
        self.kv_root = self.cache_root / kv_subdir
        self.carrier_layers = tuple(int(layer) for layer in carrier_layers)
        self.capture_token_count = int(capture_token_count)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.action_horizon = int(action_horizon)
        if tuple(sorted(set(self.carrier_layers))) != self.carrier_layers:
            raise ValueError("carrier_layers must be strictly increasing and unique")
        if min(
            self.capture_token_count,
            self.num_heads,
            self.attn_head_dim,
            self.action_horizon,
        ) <= 0:
            raise ValueError("cache/action dimensions must be positive")
        all_rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all_rows:
            raise ValueError("manifest is empty")
        if sample_offset < 0 or sample_offset >= len(all_rows):
            raise ValueError("sample_offset must select an existing manifest row")
        split_ids = [str(row["id"]) for row in all_rows]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("split manifest contains duplicate window ids")
        self.source_manifest = (
            self.manifest if source_manifest is None else source_manifest.resolve()
        )
        source_rows = (
            all_rows
            if self.source_manifest == self.manifest
            else [
                json.loads(line)
                for line in self.source_manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        source_by_id = {str(row["id"]): row for row in source_rows}
        if len(source_by_id) != len(source_rows):
            raise ValueError("source manifest contains duplicate window ids")
        for row in all_rows:
            if source_by_id.get(str(row["id"])) != row:
                raise ValueError(
                    f"split row {row['id']} is not byte-equivalent source provenance"
                )
        selected = all_rows[sample_offset:]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError("selected dataset is empty")
        self.rows = selected
        self.manifest_items = len(all_rows)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.source_manifest_items = len(source_rows)
        self.source_manifest_sha256 = sha256_file(self.source_manifest)
        self.stats_path = self.cache_root / "stats.pt"
        self.stats_sha256 = sha256_file(self.stats_path)
        stats = torch.load(self.stats_path, map_location="cpu", weights_only=False)
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        first_cache_path = self.kv_root / f"{self.rows[0]['id']}.pt"
        if not first_cache_path.is_file():
            raise FileNotFoundError(f"missing DreamWAM K/V cache: {first_cache_path}")
        first_payload = torch.load(
            first_cache_path, map_location="cpu", weights_only=False
        )
        if not first_payload.get("checkpoint"):
            raise ValueError("DreamWAM K/V cache is missing its H3 checkpoint identity")
        self.first_checkpoint_path = Path(first_payload["checkpoint"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        cache_path = self.kv_root / f"{sample_id}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing DreamWAM K/V cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = {
            "schema": DREAMWAM_KV_SCHEMA,
            "layers": self.carrier_layers,
            "capture_token_count": self.capture_token_count,
            "num_heads": self.num_heads,
            "attn_head_dim": self.attn_head_dim,
            "capture_token_strategy": DREAMWAM_KV_STRATEGY,
            "dreamwam_commit": DREAMWAM_COMMIT,
            "context_id": str(row["context_id"]),
            "action_horizon": self.action_horizon,
            "backbone": CACHE_BACKBONE,
            "quantization": CACHE_QUANTIZATION,
            "manifest_items": self.source_manifest_items,
        }
        for key, expected_value in expected.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"DreamWAM K/V cache mismatch for {sample_id}: "
                    f"{key}={actual!r}, expected {expected_value!r}"
                )
        if not math.isclose(float(payload.get("timestep", -1.0)), 1.0):
            raise ValueError(f"K/V cache timestep must be 1.0 for {sample_id}")
        if Path(payload.get("checkpoint", "")) != self.first_checkpoint_path:
            raise ValueError(f"mixed H3 checkpoint identities in K/V cache: {sample_id}")
        video_kv_cache = payload.get("video_kv_cache")
        if not isinstance(video_kv_cache, dict):
            raise ValueError(f"missing video_kv_cache for {sample_id}")
        normalized_cache: dict[int, dict[str, torch.Tensor]] = {}
        signatures = set()
        expected_shape = (
            self.capture_token_count,
            self.num_heads,
            self.attn_head_dim,
        )
        if set(video_kv_cache) != set(self.carrier_layers):
            raise ValueError(f"K/V layer mapping mismatch for {sample_id}")
        for layer in self.carrier_layers:
            item = video_kv_cache[layer]
            if set(item) != {"k", "v"}:
                raise ValueError(f"layer {layer} cache must contain k and v exactly")
            normalized_cache[layer] = {}
            for name in ("k", "v"):
                tensor = item[name]
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"layer {layer} {name} shape mismatch: {tuple(tensor.shape)}"
                    )
                signature = _storage_signature(tensor)
                if signature in signatures:
                    raise ValueError(
                        f"layer-specific K/V storage aliases at layer {layer} {name}"
                    )
                signatures.add(signature)
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite layer {layer} {name} for {sample_id}")
                normalized_cache[layer][name] = tensor

        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        context_item = torch.load(
            self.cache_root / "contexts" / f"{row['context_id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if context_item.get("text_only") is not True:
            raise ValueError(f"context is not text-only for {sample_id}")
        actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"unexpected action shape for {sample_id}: {actions.shape}")
        return {
            "sample_id": sample_id,
            "video_kv_cache": normalized_cache,
            "actions": PARENT.normalize_minmax(
                actions, self.action_min, self.action_max
            ),
            "proprio": PARENT.normalize_minmax(
                window["state"].float(), self.state_min, self.state_max
            ),
            "action_is_pad": action_is_pad,
            "text_context": context_item["context"][0].float(),
        }


def build_model(spec: ModelSpec, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    return H3DreamWAMKVCarrierPolicy(
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
    ).to(device=device, dtype=dtype)


def forward_policy(
    model: nn.Module,
    batch: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    return model(
        noisy,
        timesteps,
        text_context=batch["text_context"],
        proprio=batch["proprio"],
        video_kv_cache=batch["video_kv_cache"],
        text_mask=batch["text_mask"],
    )


def move_batch(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result = dict(batch)
    for key in ("actions", "proprio", "text_context"):
        result[key] = batch[key].to(device=device, dtype=dtype, non_blocking=True)
    result["video_kv_cache"] = {
        layer: {
            name: tensor.to(device=device, dtype=dtype, non_blocking=True)
            for name, tensor in item.items()
        }
        for layer, item in batch["video_kv_cache"].items()
    }
    result["action_is_pad"] = batch["action_is_pad"].to(
        device=device, non_blocking=True
    )
    result["text_mask"] = batch["text_mask"].to(device=device, non_blocking=True)
    return result


def checkpoint_contract(
    args: argparse.Namespace, spec: ModelSpec, dataset: CachedDreamWAMKVDataset
) -> dict[str, Any]:
    return {
        "candidate": "D",
        "classification": "action-only-on-frozen-features",
        "dreamwam_commit": DREAMWAM_COMMIT,
        "dreamwam_layers_sha256": DREAMWAM_LAYERS_SHA256,
        "dreamwam_experts_sha256": DREAMWAM_EXPERTS_SHA256,
        "dreamwam_mot_sha256": DREAMWAM_MOT_SHA256,
        "parent_shifted_flow_commit": PARENT_OBJECTIVE_COMMIT,
        "h3_checkpoint_path": str(dataset.first_checkpoint_path),
        "kv_subdir": args.kv_subdir,
        "kv_schema": DREAMWAM_KV_SCHEMA,
        "kv_strategy": DREAMWAM_KV_STRATEGY,
        "kv_layers": list(spec.carrier_layers),
        "kv_tokens": args.capture_token_count,
        "kv_num_heads": spec.num_heads,
        "kv_attn_head_dim": spec.attn_head_dim,
        "kv_bytes_per_sample": h3_kv_cache_bytes(
            layers=len(spec.carrier_layers),
            tokens=args.capture_token_count,
            heads=spec.num_heads,
            head_dim=spec.attn_head_dim,
        ),
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
        "lr_schedule": {
            "type": "linear_warmup_then_cosine",
            "base_learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_steps": args.warmup_steps,
            "scheduler_horizon": args.scheduler_horizon,
        },
        "model_spec": asdict(spec),
    }


def save_checkpoint_atomic(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_steps: int,
    contract: dict[str, Any],
    probe_prediction: torch.Tensor,
    probe_sample_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": int(completed_steps),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "contract": contract,
            "probe_prediction": probe_prediction.detach().float().cpu(),
            "probe_sample_ids": list(probe_sample_ids),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint_strict(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    actual_contract = payload.get("contract", {})
    mismatches = [
        key
        for key in sorted(set(expected_contract) | set(actual_contract))
        if expected_contract.get(key) != actual_contract.get(key)
    ]
    if mismatches:
        raise ValueError(f"checkpoint contract mismatch: {mismatches}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    return payload


def infinite_batches(loader: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


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


def main() -> None:
    args = parse_args()
    if not args.enable_dreamwam_kv_carrier:
        raise ValueError("Candidate D requires --enable-dreamwam-kv-carrier")
    if args.restore_check_only and args.load_checkpoint is None:
        raise ValueError("--restore-check-only requires --load-checkpoint")
    positive = (
        args.steps,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.action_horizon,
        args.capture_token_count,
        args.hidden_dim,
        args.ffn_dim,
        args.num_heads,
        args.attn_head_dim,
        args.freq_dim,
        args.scheduler_horizon,
    )
    if min(positive) <= 0:
        raise ValueError("positive training/model arguments are required")
    carrier_layers = tuple(int(layer) for layer in args.carrier_layers)
    if carrier_layers != DEFAULT_H3_CARRIER_LAYERS:
        raise ValueError(
            f"Candidate D requires the audited layer mapping {DEFAULT_H3_CARRIER_LAYERS}"
        )
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.action_shift <= 0:
        raise ValueError("invalid optimizer or shifted-flow arguments")

    rank, world_size, device = distributed_setup()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    started = time.perf_counter()
    dataset = CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        args.kv_subdir,
        source_manifest=args.source_manifest,
        carrier_layers=carrier_layers,
        capture_token_count=args.capture_token_count,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        action_horizon=args.action_horizon,
        limit=args.limit,
        sample_offset=args.sample_offset,
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
        collate_fn=collate_cached_batch,
    )
    spec = ModelSpec(
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        freq_dim=args.freq_dim,
        carrier_layers=carrier_layers,
    )
    model = build_model(spec, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = PARENT.build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        scheduler_horizon=args.scheduler_horizon,
        min_learning_rate=args.min_learning_rate,
    )
    contract = checkpoint_contract(args, spec, dataset)
    completed_steps = 0
    loaded_payload = None
    if args.load_checkpoint is not None:
        loaded_payload = load_checkpoint_strict(
            args.load_checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            expected_contract=contract,
        )
        completed_steps = int(loaded_payload["completed_steps"])

    unwrapped_model = model
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        unwrapped_model = model.module
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=args.action_shift)
    probe_dataset = CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        args.kv_subdir,
        source_manifest=args.source_manifest,
        carrier_layers=carrier_layers,
        capture_token_count=args.capture_token_count,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        action_horizon=args.action_horizon,
        limit=1,
        sample_offset=0,
    )
    probe_cpu = collate_cached_batch([probe_dataset[0]])
    probe = move_batch(probe_cpu, device, dtype)

    restore_max_abs = None
    if loaded_payload is not None:
        unwrapped_model.eval()
        with torch.no_grad():
            noisy, _, timesteps = PARENT.deterministic_flow_batch(
                probe["actions"], scheduler, seed=args.seed + 9_000_001
            )
            restored_prediction = forward_policy(
                unwrapped_model, probe, noisy, timesteps
            ).float()
        expected = loaded_payload["probe_prediction"].to(restored_prediction)
        if loaded_payload.get("probe_sample_ids") != probe_cpu["sample_ids"]:
            raise ValueError("checkpoint restore probe sample identity mismatch")
        restore_max_abs = float((restored_prediction - expected).abs().max())
        if restore_max_abs != 0.0:
            raise RuntimeError(
                f"checkpoint restore prediction mismatch: max_abs={restore_max_abs}"
            )

    history: list[dict[str, Any]] = []
    if not args.restore_check_only:
        model.train()
        iterator = iter(infinite_batches(loader))
        optimizer.zero_grad(set_to_none=True)
        tracked = unwrapped_model.action_expert.head.weight
        tracked_before = tracked.detach().float().clone()
        for local_step in range(1, args.steps + 1):
            losses = []
            sample_ids = []
            timestep_means = []
            prediction_stds = []
            for accumulation_index in range(args.gradient_accumulation_steps):
                batch = move_batch(next(iterator), device, dtype)
                sample_ids.extend(batch["sample_ids"])
                noisy, target, timesteps = PARENT.deterministic_flow_batch(
                    batch["actions"],
                    scheduler,
                    seed=PARENT.distributed_flow_seed(
                        base_seed=args.seed,
                        completed_step=completed_steps + local_step,
                        accumulation_index=accumulation_index,
                        rank=rank,
                    ),
                )
                prediction = forward_policy(model, batch, noisy, timesteps)
                loss = flow_matching_loss(
                    prediction,
                    target,
                    timesteps,
                    scheduler,
                    is_pad_mask=batch["action_is_pad"],
                )
                (loss / args.gradient_accumulation_steps).backward()
                losses.append(float(loss.detach()))
                timestep_means.append(float(timesteps.float().mean()))
                prediction_stds.append(float(prediction.detach().float().std()))
            block_gradient_norms = [
                PARENT.module_grad_norm(block)
                for block in unwrapped_model.action_expert.blocks
            ]
            proprio_gradient_norm = PARENT.module_grad_norm(
                unwrapped_model.proprio_encoder
            )
            gradient_values = [*block_gradient_norms, proprio_gradient_norm]
            if not all(math.isfinite(value) and value > 0 for value in gradient_values):
                raise RuntimeError(f"non-finite/zero gradient path: {gradient_values}")
            clipped_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_max_abs = float(
                (tracked.detach().float() - tracked_before).abs().max()
            )
            if not math.isfinite(update_max_abs) or update_max_abs <= 0:
                raise RuntimeError("optimizer did not update the DreamWAM ActionDiT head")
            tracked_before = tracked.detach().float().clone()
            item = {
                "step": completed_steps + local_step,
                "loss": sum(losses) / len(losses),
                "timestep_mean": sum(timestep_means) / len(timestep_means),
                "prediction_std": sum(prediction_stds) / len(prediction_stds),
                "block_gradient_norms": block_gradient_norms,
                "proprio_gradient_norm": proprio_gradient_norm,
                "clipped_gradient_norm": clipped_norm,
                "head_update_max_abs": update_max_abs,
                "learning_rate": learning_rate,
                "sample_ids": sample_ids,
            }
            history.append(item)
            if rank == 0:
                print(json.dumps(item, sort_keys=True), flush=True)
        completed_steps += args.steps

    unwrapped_model.eval()
    with torch.no_grad():
        noisy, _, timesteps = PARENT.deterministic_flow_batch(
            probe["actions"], scheduler, seed=args.seed + 9_000_001
        )
        probe_prediction = forward_policy(
            unwrapped_model, probe, noisy, timesteps
        ).float()
    if not torch.isfinite(probe_prediction).all() or float(probe_prediction.std()) <= 0:
        raise RuntimeError("probe prediction is non-finite or constant")

    if args.save_checkpoint is not None and not args.restore_check_only:
        if rank == 0:
            save_checkpoint_atomic(
                args.save_checkpoint,
                model=unwrapped_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                completed_steps=completed_steps,
                contract=contract,
                probe_prediction=probe_prediction,
                probe_sample_ids=probe_cpu["sample_ids"],
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in unwrapped_model.parameters()
            if parameter.requires_grad
        )
        report = {
            "event": "h3_int8_dreamwam_kv_carrier_probe",
            "classification": "action-only-on-frozen-features",
            "candidate": "D",
            "resolved_argv": sys.argv,
            "world_size": world_size,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
            "contract": contract,
            "loaded_checkpoint": (
                None if args.load_checkpoint is None else str(args.load_checkpoint)
            ),
            "saved_checkpoint": (
                None if args.save_checkpoint is None else str(args.save_checkpoint)
            ),
            "restore_probe_max_abs": restore_max_abs,
            "probe_prediction_mean": float(probe_prediction.mean()),
            "probe_prediction_std": float(probe_prediction.std()),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
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
