#!/usr/bin/env python3
"""Faithful C57 persistent observation/action K/V training on frozen H3 K/V.

Unlike the legacy flattened-history adapter, every teacher-forced replan uses
the same ``commit_executed_feedback`` lifecycle as deployment.  The wrapper's
entire history replay occurs inside DDP forward so gradients from clean action
K/V are visible to the reducer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam.c57_lingbot_interfaces import (  # noqa: E402
    LingBotTeacherForcedFeedback,
    forward_teacher_forced_history,
)
from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    REPEAT_LAYER49_CARRIER_SOURCE,
)
from fastwam.models.h3wam.lingbot_persistent_kv import (  # noqa: E402
    H3LingBotPersistentKVPolicy,
    merge_observation_kv_sequence,
)


def load_parent():
    path = Path(__file__).with_name("train_h3_int8_dreamwam_kv_carrier.py")
    spec = importlib.util.spec_from_file_location("_c57_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import parent trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = load_parent()
SEQUENCE_SCHEMA = "c57_lingbot_replan8_v1"
CHECKPOINT_SCHEMA = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--cache-source-manifest", type=Path, required=True,
        help="Full manifest identity embedded in the frozen H3 K/V cache.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", default="h3_int8_dreamwam_kv_5x32_dense_v1")
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--freq-dim", type=int, default=256)
    return parser.parse_args()


class C57SequenceDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None:
        all_rows = [
            json.loads(line)
            for line in args.sequence_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all_rows or any(row.get("sequence_schema") != SEQUENCE_SCHEMA for row in all_rows):
            raise ValueError("C57 sequence manifest schema mismatch")
        self.rows = all_rows if not args.limit else all_rows[: args.limit]
        self.base = PARENT.CachedDreamWAMKVDataset(
            args.source_manifest,
            args.cache_root,
            args.kv_subdir,
            source_manifest=args.cache_source_manifest,
            carrier_layers=DEFAULT_H3_CARRIER_LAYERS,
            capture_token_count=32,
            kv_pool_strategy=PARENT.DREAMWAM_KV_STRATEGY,
            num_heads=args.num_heads,
            attn_head_dim=args.attn_head_dim,
            action_horizon=32,
        )
        self.index_by_id = {
            str(row["id"]): index for index, row in enumerate(self.base.rows)
        }
        if len(self.index_by_id) != len(self.base.rows):
            raise ValueError("source manifest contains duplicate IDs")
        missing = {
            str(row["current_id"]) for row in self.rows
        } - set(self.index_by_id)
        if missing:
            raise ValueError(f"sequence manifest has {len(missing)} missing current IDs")

    def __len__(self) -> int:
        return len(self.rows)

    def item(self, sample_id: str) -> dict[str, Any]:
        return self.base[self.index_by_id[sample_id]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        current = self.item(str(row["current_id"]))
        history = []
        for chunk in row["history"]:
            action_item = self.item(str(chunk["action_source_id"]))
            observation_items = [
                self.item(str(sample_id))
                for sample_id in chunk["observation_source_ids"]
            ]
            history.append(
                {
                    "observation_kv": merge_observation_kv_sequence(
                        [item["video_kv_cache"] for item in observation_items],
                        layers=DEFAULT_H3_CARRIER_LAYERS,
                    ),
                    "observed_frame_count": len(observation_items),
                    "executed_actions": action_item["actions"][:8],
                    "proprio": action_item["proprio"],
                }
            )
        return {"current": current, "history": history}


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("C57 follows LingBot's official microbatch=1 contract")
    item = items[0]
    current = PARENT.collate_cached_batch([item["current"]])
    history = []
    for feedback in item["history"]:
        history.append(
            {
                "observation_kv": {
                    layer: {
                        name: tensor.unsqueeze(0)
                        for name, tensor in cache.items()
                    }
                    for layer, cache in feedback["observation_kv"].items()
                },
                "observed_frame_count": int(feedback["observed_frame_count"]),
                "executed_actions": feedback["executed_actions"].unsqueeze(0),
                "proprio": feedback["proprio"].unsqueeze(0),
            }
        )
    current["history"] = history
    return current


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    result = PARENT.move_batch({k: v for k, v in batch.items() if k != "history"}, device, dtype)
    result["history"] = []
    for item in batch["history"]:
        result["history"].append(
            {
                "observation_kv": {
                    layer: {
                        name: value.to(device=device, dtype=dtype, non_blocking=True)
                        for name, value in cache.items()
                    }
                    for layer, cache in item["observation_kv"].items()
                },
                "observed_frame_count": item["observed_frame_count"],
                "executed_actions": item["executed_actions"].to(device=device, dtype=dtype, non_blocking=True),
                "proprio": item["proprio"].to(device=device, dtype=dtype, non_blocking=True),
            }
        )
    return result


class C57TrainingModel(nn.Module):
    """Keep all differentiable history replay inside one DDP forward."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.policy = H3LingBotPersistentKVPolicy(
            enabled=True,
            persistent_enabled=True,
            persistent_window_chunks=15,
            observation_tokens_per_chunk=32,
            action_tokens_per_chunk=4,
            carrier_layers=DEFAULT_H3_CARRIER_LAYERS,
            carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
            action_dim=7,
            proprio_dim=8,
            context_dim=5120,
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            num_heads=args.num_heads,
            attn_head_dim=args.attn_head_dim,
            freq_dim=args.freq_dim,
        )

    def forward(self, batch: dict[str, Any], noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        history = [
            LingBotTeacherForcedFeedback(
                observation_kv=item["observation_kv"],
                observed_frame_count=item["observed_frame_count"],
                executed_actions=item["executed_actions"],
                proprio=item["proprio"],
            )
            for item in batch["history"]
        ]
        prediction, _ = forward_teacher_forced_history(
            self.policy,
            episode_key=batch["sample_ids"][0],
            history=history,
            noisy_actions=noisy,
            timestep=timesteps,
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            current_observation_kv=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
        )
        return prediction


def infinite(loader: DataLoader, sampler: DistributedSampler | None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if min(args.steps, args.gradient_accumulation_steps, args.save_every, args.warmup_steps) <= 0:
        raise ValueError("positive steps/accumulation/save/warmup are required")
    rank, world_size, device = PARENT.distributed_setup()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    started = time.perf_counter()
    dataset = C57SequenceDataset(args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_one,
    )
    model = C57TrainingModel(args).to(device=device, dtype=dtype)
    parent_payload = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(parent_payload["model"], strict=True)
    parent_contract = parent_payload.get("contract", {})
    if parent_contract.get("candidate") != "D0" or parent_contract.get("carrier_source_mode") != REPEAT_LAYER49_CARRIER_SOURCE:
        raise ValueError("C57 must initialize from an audited D0 checkpoint")
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(float(step + 1) / args.warmup_steps, 1.0)
    )
    flow_scheduler = PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=args.action_shift)
    iterator = iter(infinite(loader, sampler))
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    tracked = raw_model.policy.action_expert.head.weight
    tracked_before = tracked.detach().float().clone()
    for step in range(1, args.steps + 1):
        losses, ids = [], []
        step_started = time.perf_counter()
        for accumulation in range(args.gradient_accumulation_steps):
            batch = move_batch(next(iterator), device, dtype)
            ids.extend(batch["sample_ids"])
            noisy, target, timesteps = PARENT.PARENT.deterministic_flow_batch(
                batch["actions"], flow_scheduler,
                seed=PARENT.PARENT.distributed_flow_seed(
                    base_seed=args.seed, completed_step=step,
                    accumulation_index=accumulation, rank=rank,
                ),
            )
            sync = nullcontext() if accumulation + 1 == args.gradient_accumulation_steps or not isinstance(model, DDP) else model.no_sync()
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with sync, autocast:
                prediction = model(batch, noisy, timesteps)
                loss = PARENT.flow_matching_loss(
                    prediction, target, timesteps, flow_scheduler,
                    is_pad_mask=batch["action_is_pad"],
                )
                (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach()))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True))
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update = float((tracked.detach().float() - tracked_before).abs().max())
        tracked_before = tracked.detach().float().clone()
        if not math.isfinite(update) or update <= 0:
            raise RuntimeError("C57 action expert did not update")
        item = {
            "step": step, "loss": sum(losses) / len(losses), "gradient_norm": grad_norm,
            "head_update_max_abs": update, "learning_rate": lr,
            "seconds": time.perf_counter() - step_started, "sample_ids": ids,
        }
        history.append(item)
        if rank == 0:
            print(json.dumps(item, sort_keys=True), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                atomic_torch_save(
                    {
                        "schema_version": CHECKPOINT_SCHEMA,
                        "completed_steps": step,
                        "model": raw_model.policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                        "contract": {
                            "candidate": "C57",
                            "classification": "action-only-on-frozen-h3-kv",
                            "method": "lingbot_persistent_observation_action_kv",
                            "source_candidate": "D0",
                            "source_checkpoint": str(args.initialize_from.resolve()),
                            "source_completed_steps": int(parent_payload["completed_steps"]),
                            "sequence_schema": SEQUENCE_SCHEMA,
                            "sequence_manifest": str(args.sequence_manifest.resolve()),
                            "sequence_manifest_sha256": PARENT.sha256_file(args.sequence_manifest),
                            "source_manifest_sha256": dataset.base.source_manifest_sha256,
                            "world_size": world_size,
                            "microbatch": 1,
                            "gradient_accumulation_steps": args.gradient_accumulation_steps,
                            "optimizer": "AdamW_fused",
                            "learning_rate": args.learning_rate,
                            "betas": [0.9, 0.95], "weight_decay": args.weight_decay,
                            "warmup_steps": args.warmup_steps,
                            "replan": 8, "observe_every": 4,
                            "persistent_token_capacity": raw_model.policy.persistent_token_capacity,
                        },
                    },
                    args.checkpoint_dir / f"c57_step{step:05d}.pt",
                )
            if dist.is_initialized():
                dist.barrier()
    if rank == 0:
        report = {
            "event": "c57_lingbot_persistent_kv_training",
            "status": "PASS",
            "gate": "PASS",
            "completed_steps": args.steps,
            "world_size": world_size,
            "steps": args.steps,
            "training_samples": world_size * args.gradient_accumulation_steps * args.steps,
            "selected_windows": len(dataset),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
            "history": history,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, args.output)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
