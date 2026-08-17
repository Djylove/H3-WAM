#!/usr/bin/env python3
"""Eight-GPU C66 full-history optimizer canary with paired heldout arms."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from safetensors import safe_open
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Dataset,
    C58OnlineFrozenH3Provider,
    collate_c58_online,
    move_c58_online_batch,
)
from fastwam.models.h3wam.c66_lingbot_fastwam_persistent import (  # noqa: E402
    H3FastWAMLingBotPersistentPolicy,
    prepare_committed_observation_sequence,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
)
from fastwam.models.h3wam.lingbot_persistent_kv import (  # noqa: E402
    LingBotPersistentKVState,
)


PLAN_SCHEMA = "h3wam-c66-lingbot-c58-canary-plan-v1"
SEQUENCE_SCHEMA = "c57_lingbot_replan8_v1"
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
CHECKPOINT_SCHEMA = 1


def load_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C58 = load_script("_c66_c58_trainer", "train_h3_fastwam_full_tower.py")
PROBE = load_script("_c66_parent_validator", "probe_c66_lingbot_c58_persistent.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(value, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=66017)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        dense_manifest: Path,
        source_manifest: Path,
        cache_root: Path,
        h3_checkpoint: Path,
    ) -> None:
        self.manifest = manifest.resolve()
        self.rows = [
            json.loads(line)
            for line in self.manifest.read_text().splitlines()
            if line.strip()
        ]
        if not self.rows or any(
            row.get("sequence_schema") != SEQUENCE_SCHEMA
            or int(row.get("history_chunks", -1)) != 7
            or int(row.get("history_observation_frames", -1)) != 15
            or int(row.get("history_executed_actions", -1)) != 56
            for row in self.rows
        ):
            raise ValueError("C66 canary requires frozen full-history C57 rows")
        self.base = C58OnlineFrozenH3Dataset(
            dense_manifest,
            source_manifest,
            cache_root,
            h3_checkpoint,
            action_horizon=32,
        )
        self.index_by_id = {
            str(row["id"]): index for index, row in enumerate(self.base.rows)
        }
        required = set()
        for row in self.rows:
            required.add(str(row["current_id"]))
            for chunk in row["history"]:
                required.add(str(chunk["action_source_id"]))
                required.update(map(str, chunk["observation_source_ids"]))
        missing = required - set(self.index_by_id)
        if missing:
            raise ValueError(f"C66 sequence references {len(missing)} missing rows")
        self.manifest_sha256 = sha256_file(self.manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def item(self, sample_id: str) -> dict[str, Any]:
        return self.base[self.index_by_id[sample_id]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "id": str(row["id"]),
            "suite": str(row["suite"]),
            "episode": int(row["episode"]),
            "current": self.item(str(row["current_id"])),
            "history": [
                {
                    "observations": [
                        self.item(str(sample_id))
                        for sample_id in chunk["observation_source_ids"]
                    ],
                    "action": self.item(str(chunk["action_source_id"])),
                }
                for chunk in row["history"]
            ],
        }


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("C66 online H3 canary requires microbatch one")
    return items[0]


def build_policy(device: torch.device, dtype: torch.dtype):
    return H3FastWAMLingBotPersistentPolicy(
        enabled=True,
        persistent_enabled=True,
        persistent_window_frames=15,
        observation_tokens_per_frame=32,
        action_tokens_per_frame=4,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        action_block_to_h3_layer=LAYERWISE_H3_50_TO_ACTION_30,
        action_dim=7,
        proprio_dim=8,
        context_dim=5120,
        hidden_dim=1024,
        ffn_dim=4096,
        num_heads=56,
        attn_head_dim=128,
        freq_dim=256,
        num_layers=30,
        use_gradient_checkpointing=True,
    ).to(device=device, dtype=dtype)


def online_batch(item: dict[str, Any], device: torch.device, dtype: torch.dtype):
    return move_c58_online_batch(collate_c58_online([item]), device, dtype)


@torch.no_grad()
def h3_kv(provider, item, device, dtype):
    batch = online_batch(item, device, dtype)
    return batch, provider(batch)


@torch.no_grad()
def materialize_sequence(provider, raw, inv_freq, device, dtype):
    current, current_kv = h3_kv(provider, raw["current"], device, dtype)
    history = []
    frame_start = 0
    for chunk in raw["history"]:
        observations = [
            h3_kv(provider, item, device, dtype)[1]
            for item in chunk["observations"]
        ]
        merged = prepare_committed_observation_sequence(
            observations,
            layers=LAYERWISE_H3_50_TO_ACTION_30,
            temporal_inv_freq=inv_freq,
            frame_start=frame_start,
        )
        frame_start += len(observations)
        action = online_batch(chunk["action"], device, dtype)
        history.append(
            {
                "observation_kv": merged,
                "observed_frame_count": len(observations),
                "executed_actions": action["actions"][:, :8],
                "proprio": action["proprio"],
            }
        )
    if frame_start != 15 or sum(x["executed_actions"].shape[1] for x in history) != 56:
        raise RuntimeError("C66 materialized history is not the frozen full contract")
    return {
        "id": raw["id"],
        "suite": raw["suite"],
        "episode": raw["episode"],
        "current": current,
        "current_kv": current_kv,
        "history": history,
    }


def predict_context(
    policy: H3FastWAMLingBotPersistentPolicy,
    sequence: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    shuffle_actions: bool,
) -> tuple[torch.Tensor, LingBotPersistentKVState]:
    state = policy.new_persistent_state(f"{sequence['suite']}:{sequence['episode']}")
    actions = [item["executed_actions"] for item in sequence["history"]]
    if shuffle_actions:
        actions = actions[1:] + actions[:1]
    current = sequence["current"]
    for index, feedback in enumerate(sequence["history"]):
        policy.commit_executed_feedback(
            state,
            observation_kv=feedback["observation_kv"],
            observed_frame_count=feedback["observed_frame_count"],
            executed_actions=actions[index],
            text_context=current["text_context"],
            proprio=feedback["proprio"],
            text_mask=current["text_mask"],
        )
    prediction = policy(
        noisy,
        timesteps,
        text_context=current["text_context"],
        proprio=current["proprio"],
        video_kv_cache=sequence["current_kv"],
        text_mask=current["text_mask"],
        persistent_state=state,
    )
    return prediction, state


def predict_off(policy, sequence, noisy, timesteps):
    current = sequence["current"]
    enabled = policy.persistent_enabled
    policy.persistent_enabled = False
    try:
        return policy(
            noisy,
            timesteps,
            text_context=current["text_context"],
            proprio=current["proprio"],
            video_kv_cache=sequence["current_kv"],
            text_mask=current["text_mask"],
        )
    finally:
        policy.persistent_enabled = enabled


class TrainingModel(nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, sequence, noisy, timesteps):
        return predict_context(
            self.policy, sequence, noisy, timesteps, shuffle_actions=False
        )[0]


def main() -> None:
    args = parse_args()
    if args.steps != 100:
        raise ValueError("C66 canary is fixed at exactly 100 steps")
    rank, world_size, device = C58.distributed_setup()
    if world_size != 8:
        raise ValueError("C66 canary requires exactly eight ranks")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("seed") != args.seed:
        raise ValueError("C66 frozen plan identity mismatch")
    train_data = SequenceDataset(
        args.train_manifest, args.dense_manifest, args.source_manifest,
        args.cache_root, args.h3_checkpoint,
    )
    heldout_data = SequenceDataset(
        args.heldout_manifest, args.dense_manifest, args.source_manifest,
        args.cache_root, args.h3_checkpoint,
    )
    if (
        len(train_data) != 800
        or len(heldout_data) != 64
        or train_data.manifest_sha256 != plan["train_manifest_sha256"]
        or heldout_data.manifest_sha256 != plan["heldout_manifest_sha256"]
        or plan["episode_intersection"] != 0
    ):
        raise ValueError("C66 frozen data/budget mismatch")

    parent_path = args.parent_checkpoint.resolve()
    parent_sha = sha256_file(parent_path) if rank == 0 else None
    h3_sha = sha256_file(args.h3_checkpoint.resolve()) if rank == 0 else None
    shared = [parent_sha, h3_sha]
    dist.broadcast_object_list(shared, src=0)
    parent_sha, h3_sha = shared
    if parent_sha != C58_SHA256 or h3_sha != H3_SHA256:
        raise ValueError("C66 parent/H3 checkpoint identity mismatch")
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_checks = PROBE.validate_c58_parent_payload(payload)
    policy = build_policy(device, dtype)
    restored = policy.load_state_dict(payload["model"], strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise RuntimeError("C66 parent strict restore failed")
    if set(policy.state_dict()) != set(payload["model"]):
        raise RuntimeError("C66 canary introduced state keys")
    del payload
    with safe_open(args.h3_checkpoint.resolve(), framework="pt", device="cpu") as handle:
        inv_freq = handle.get_tensor("rope.inv_freq").float().to(device)
    provider = C58OnlineFrozenH3Provider(
        args.h3_checkpoint, layers=LAYERWISE_H3_50_TO_ACTION_30
    ).to(device).eval()
    provider.requires_grad_(False)

    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        train_data,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_one,
        pin_memory=True,
    )
    if len(loader) != 100:
        raise ValueError("C66 canary must consume one exact 800-sample epoch")
    model: nn.Module = DDP(
        TrainingModel(policy),
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
    )
    raw = model.module
    optimizer = torch.optim.AdamW(
        raw.policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(float(step + 1) / args.warmup_steps, 1.0)
    )
    flow = C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    consumed: list[str] = []
    milestones: list[dict[str, Any]] = []
    model.train()
    for step, cpu_item in enumerate(loader, start=1):
        sequence = materialize_sequence(provider, cpu_item, inv_freq, device, dtype)
        consumed.append(sequence["id"])
        noisy, target, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
            sequence["current"]["actions"],
            flow,
            seed=C58.PARENT.PARENT.distributed_flow_seed(
                base_seed=args.seed,
                completed_step=step,
                accumulation_index=0,
                rank=rank,
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            prediction = model(sequence, noisy, timesteps)
            loss = C58.PARENT.flow_matching_loss(
                prediction,
                target,
                timesteps,
                flow,
                is_pad_mask=sequence["current"]["action_is_pad"],
            )
        loss.backward()
        gradients = [
            float(block.self_attn.k.weight.grad.float().norm())
            for block in raw.policy.action_expert.blocks
        ]
        if not all(math.isfinite(value) and value > 0 for value in gradients):
            raise RuntimeError("C66 action loss did not reach all thirty blocks")
        if any(parameter.grad is not None for parameter in provider.parameters()):
            raise RuntimeError("C66 gradient leaked into frozen H3")
        clipped = float(torch.nn.utils.clip_grad_norm_(
            raw.policy.parameters(), args.max_grad_norm, error_if_nonfinite=True
        ))
        optimizer.step()
        scheduler.step()
        if step in {1, 10, 100}:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "min_block_k_gradient_norm": min(gradients),
                "max_block_k_gradient_norm": max(gradients),
                "clipped_gradient_norm": clipped,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            milestones.append(row)
            if rank == 0:
                print(json.dumps(row, sort_keys=True), flush=True)

    gathered_ids: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(gathered_ids, consumed)
    complete_ids = [value for values in gathered_ids if values for value in values]
    raw.policy.eval()
    local_eval = []
    with torch.no_grad():
        for index in range(rank, len(heldout_data), world_size):
            sequence = materialize_sequence(
                provider, heldout_data[index], inv_freq, device, dtype
            )
            noisy, target, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
                sequence["current"]["actions"], flow,
                seed=args.seed + 1_000_000 + index,
            )
            with torch.autocast("cuda", dtype=dtype):
                clean, state = predict_context(
                    raw.policy, sequence, noisy, timesteps, shuffle_actions=False
                )
                shuffled, _ = predict_context(
                    raw.policy, sequence, noisy, timesteps, shuffle_actions=True
                )
                off = predict_off(raw.policy, sequence, noisy, timesteps)
            clean, shuffled, off = clean.float(), shuffled.float(), off.float()
            target = target.float()
            valid = (~sequence["current"]["action_is_pad"]).unsqueeze(-1)
            def mse(value):
                return float(
                    ((value - target).square() * valid).sum()
                    / (valid.sum() * value.shape[-1]).clamp_min(1)
                )
            restored_state = LingBotPersistentKVState.from_snapshot(
                state.snapshot(), device=device, dtype=dtype
            )
            restored_prediction = raw.policy(
                noisy,
                timesteps,
                text_context=sequence["current"]["text_context"],
                proprio=sequence["current"]["proprio"],
                video_kv_cache=sequence["current_kv"],
                text_mask=sequence["current"]["text_mask"],
                persistent_state=restored_state,
            ).float()
            local_eval.append(
                {
                    "index": index,
                    "id": sequence["id"],
                    "clean_mse": mse(clean),
                    "shuffled_mse": mse(shuffled),
                    "off_mse": mse(off),
                    "shuffle_prediction_max_abs": float((clean - shuffled).abs().max()),
                    "restore_max_abs": float((clean - restored_prediction).abs().max()),
                }
            )
    gathered_eval: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered_eval, local_eval)
    complete_eval = [value for values in gathered_eval if values for value in values]
    if len(complete_ids) != 800 or len(set(complete_ids)) != 800 or len(complete_eval) != 64:
        raise RuntimeError("C66 train/eval coverage is not exact")
    means = {
        key: sum(row[key] for row in complete_eval) / len(complete_eval)
        for key in ("clean_mse", "shuffled_mse", "off_mse")
    }
    clean_over_shuffle = (
        means["shuffled_mse"] - means["clean_mse"]
    ) / means["shuffled_mse"]
    clean_vs_off = (means["clean_mse"] - means["off_mse"]) / means["off_mse"]
    gates = {
        "parent_identity": all(parent_checks.values()),
        "train_unique_samples": len(set(complete_ids)) == 800,
        "episode_disjoint": plan["episode_intersection"] == 0,
        "all_30_blocks_gradient": all(
            row["min_block_k_gradient_norm"] > 0 for row in milestones
        ),
        "h3_gradient_absent": True,
        "runtime_restore_exact": max(row["restore_max_abs"] for row in complete_eval) == 0,
        "shuffle_prediction_effect": min(
            row["shuffle_prediction_max_abs"] for row in complete_eval
        ) >= 1e-5,
        "clean_beats_shuffle_by_1pct": clean_over_shuffle >= 0.01,
        "clean_vs_off_regression_at_most_5pct": clean_vs_off <= 0.05,
    }
    passed = all(gates.values())
    if rank == 0:
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": 100,
            "model": {
                key: value.detach().cpu()
                for key, value in raw.policy.state_dict().items()
            },
            "contract": {
                "candidate": "C66_LINGBOT_C58_BLOCK_PERSISTENT",
                "classification": "full30_actiondit_on_frozen_int8_h3_with_committed_context",
                "parent_checkpoint_sha256": parent_sha,
                "h3_checkpoint_sha256": h3_sha,
                "plan_sha256": sha256_file(args.plan.resolve()),
                "train_manifest_sha256": train_data.manifest_sha256,
                "heldout_manifest_sha256": heldout_data.manifest_sha256,
                "world_size": 8,
                "steps": 100,
                "training_samples": 800,
                "history_chunks": 7,
                "history_observation_frames": 15,
                "history_executed_actions": 56,
                "gradient_checkpointing": True,
            },
        }
        atomic_torch(args.save_checkpoint.resolve(), checkpoint)
    dist.barrier()
    if rank == 0:
        strict_payload = torch.load(
            args.save_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if set(strict_payload) != {
            "schema_version", "completed_steps", "model", "contract"
        } or set(strict_payload["model"]) != set(raw.policy.state_dict()):
            raise RuntimeError("C66 canary checkpoint schema/state mismatch")
        report = {
            "event": "h3_c66_lingbot_c58_full_history_paired_canary",
            "status": "PASS_C66_PAIRED_CANARY" if passed else "FAIL_C66_PAIRED_CANARY",
            "permission": "GO_C66_LIBERO_CANARY" if passed else "NO_GO_C66_LONG_TRAINING",
            "effect_status": "NOT_LIBERO_EVIDENCE",
            "gates": gates,
            "means": means,
            "clean_over_shuffle_relative": clean_over_shuffle,
            "clean_vs_off_relative": clean_vs_off,
            "world_size": 8,
            "steps": 100,
            "training_samples": 800,
            "train_unique_samples": len(set(complete_ids)),
            "heldout_samples": len(complete_eval),
            "milestones": milestones,
            "checkpoint": str(args.save_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.save_checkpoint.resolve()),
            "parent_checkpoint_sha256": parent_sha,
            "h3_checkpoint_sha256": h3_sha,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "elapsed_seconds": time.perf_counter() - started,
            "boundary": "Offline paired MSE is a bounded mechanism gate, not LIBERO success evidence.",
        }
        atomic_json(args.output.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
