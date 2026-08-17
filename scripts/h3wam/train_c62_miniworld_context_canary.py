#!/usr/bin/env python3
"""Bounded C62 bridge-only causal/optimizer canary on the fixed C58 parent."""

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
from fastwam.models.h3wam.c62_miniworld_context import (  # noqa: E402
    C62MiniWorldRollingContextPolicy,
    MiniWorldRollingContextState,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
)


SEQUENCE_SCHEMA = "c62_miniworld_replan8_real_history_v1"
PLAN_SCHEMA = "h3wam-c62-causal-canary-plan-v1"
CHECKPOINT_SCHEMA = 1
C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"


def load_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C58 = load_script("_c62_c58_trainer", "train_h3_fastwam_full_tower.py")
PROBE = load_script("_c62_parent_validator", "probe_c62_miniworld_context.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_manifest", type=Path)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=62017)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequence_manifest: Path,
        dense_manifest: Path,
        source_manifest: Path,
        cache_root: Path,
        h3_checkpoint: Path,
    ) -> None:
        self.manifest = sequence_manifest.resolve()
        self.rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows or any(
            row.get("sequence_schema") != SEQUENCE_SCHEMA for row in self.rows
        ):
            raise ValueError("C62 sequence manifest schema mismatch")
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
        required = {
            str(row["current_id"])
            for row in self.rows
        } | {
            str(item[key])
            for row in self.rows
            for item in row["history"]
            for key in ("observation_id", "actions_before_observation_id")
            if item[key] is not None
        } | {str(row["actions_before_current_id"]) for row in self.rows}
        missing = required - set(self.index_by_id)
        if missing:
            raise ValueError(f"C62 sequence references {len(missing)} missing rows")
        self.manifest_sha256 = sha256_file(self.manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def _item(self, sample_id: str) -> dict[str, Any]:
        return self.base[self.index_by_id[sample_id]]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        history = []
        for item in row["history"]:
            observation = self._item(str(item["observation_id"]))
            action_id = item["actions_before_observation_id"]
            actions = None if action_id is None else self._item(str(action_id))["actions"][:8]
            history.append({"observation": observation, "actions": actions})
        return {
            "sequence_id": str(row["current_id"]),
            "current": self._item(str(row["current_id"])),
            "history": history,
            "actions_before_current": self._item(
                str(row["actions_before_current_id"])
            )["actions"][:8],
        }


def collate_one(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("C62 online H3 canary requires microbatch one")
    return items[0]


def build_parent(device: torch.device, dtype: torch.dtype) -> H3FastWAMFullTowerPolicy:
    return H3FastWAMFullTowerPolicy(
        enabled=True,
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


def provider_batch(item: dict[str, Any], device: torch.device, dtype: torch.dtype):
    return move_c58_online_batch(collate_c58_online([item]), device, dtype)


@torch.no_grad()
def extract_kv(
    provider: C58OnlineFrozenH3Provider,
    item: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    batch = provider_batch(item, device, dtype)
    kv = provider(batch)
    if any(tensor.requires_grad for value in kv.values() for tensor in value.values()):
        raise RuntimeError("frozen online H3 emitted a differentiable tensor")
    return batch, kv


@torch.no_grad()
def materialize_sequence(
    provider: C58OnlineFrozenH3Provider,
    item: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    current, current_kv = extract_kv(provider, item["current"], device, dtype)
    history = []
    for raw in item["history"]:
        _, observation_kv = extract_kv(provider, raw["observation"], device, dtype)
        history.append(
            {
                "observation_kv": observation_kv,
                "actions": (
                    None
                    if raw["actions"] is None
                    else raw["actions"].to(device=device, dtype=dtype).unsqueeze(0)
                ),
            }
        )
    result = {
        "sequence_id": item["sequence_id"],
        "current": current,
        "current_kv": current_kv,
        "history": history,
        "actions_before_current": item["actions_before_current"].to(
            device=device, dtype=dtype
        ).unsqueeze(0),
    }
    action_tensors = [
        value["actions"] for value in result["history"] if value["actions"] is not None
    ] + [result["actions_before_current"]]
    if any(tuple(value.shape) != (1, 8, 7) for value in action_tensors):
        raise RuntimeError("C62 executed-action history must be batched [1,8,7]")
    return result


def context_state(
    model: C62MiniWorldRollingContextPolicy,
    sequence: dict[str, Any],
    *,
    shuffled: bool,
) -> tuple[MiniWorldRollingContextState, torch.Tensor]:
    state = model.new_context_state(sequence["sequence_id"])
    conditioned = [
        item["actions"] for item in sequence["history"] if item["actions"] is not None
    ]
    conditioned.append(sequence["actions_before_current"])
    if shuffled:
        conditioned = conditioned[1:] + conditioned[:1]
    cursor = 0
    for item in sequence["history"]:
        actions = None
        if item["actions"] is not None:
            actions = conditioned[cursor]
            cursor += 1
        model.commit_real_observation(
            state,
            observation_kv=item["observation_kv"],
            actions_before_observation=actions,
        )
    actions_before_current = conditioned[cursor]
    return state, actions_before_current


def predict(
    model: nn.Module,
    sequence: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    use_context: bool,
    shuffled: bool = False,
) -> torch.Tensor:
    current = sequence["current"]
    kwargs: dict[str, Any] = {}
    raw_model = model.module if isinstance(model, DDP) else model
    if use_context:
        state, actions = context_state(raw_model, sequence, shuffled=shuffled)
        kwargs.update(context_state=state, actions_before_current=actions)
    return model(
        noisy,
        timesteps,
        text_context=current["text_context"],
        proprio=current["proprio"],
        video_kv_cache=sequence["current_kv"],
        text_mask=current["text_mask"],
        use_context=use_context,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    if args.steps != 100:
        raise ValueError("C62 causal canary budget is fixed at exactly 100 steps")
    rank, world_size, device = C58.distributed_setup()
    if world_size != 8:
        raise ValueError("C62 causal canary requires exactly eight ranks")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("C62 canary plan schema mismatch")
    train_dataset = SequenceDataset(
        args.sequence_manifest,
        args.dense_manifest,
        args.source_manifest,
        args.cache_root,
        args.h3_checkpoint,
    )
    heldout_dataset = SequenceDataset(
        args.heldout_manifest,
        args.dense_manifest,
        args.source_manifest,
        args.cache_root,
        args.h3_checkpoint,
    )
    if (
        len(train_dataset) != 800
        or len(heldout_dataset) != 64
        or plan["train_manifest_sha256"] != train_dataset.manifest_sha256
        or plan["heldout_manifest_sha256"] != heldout_dataset.manifest_sha256
        or plan["canary_budget"]["training_samples"] != 800
    ):
        raise ValueError("C62 frozen data/budget identity mismatch")

    parent_path = args.parent_checkpoint.resolve()
    parent_sha = sha256_file(parent_path) if rank == 0 else None
    if dist.is_initialized():
        shared = [parent_sha]
        dist.broadcast_object_list(shared, src=0)
        parent_sha = shared[0]
    if parent_sha != C58_SHA256:
        raise ValueError("C62 parent is not the fixed C58 s10000 champion")
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_checks = PROBE.validate_c58_parent_payload(payload)
    parent = build_parent(device, dtype)
    restored = parent.load_state_dict(payload["model"], strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise RuntimeError("C62 C58 parent strict restore failed")
    del payload
    parent.requires_grad_(False)
    parent.eval()
    model = C62MiniWorldRollingContextPolicy(
        parent, context_enabled=True, max_cache_chunks=3
    ).to(device=device, dtype=dtype)
    model.parent.requires_grad_(False)
    provider = C58OnlineFrozenH3Provider(
        args.h3_checkpoint, layers=LAYERWISE_H3_50_TO_ACTION_30
    ).to(device=device).eval()
    provider.requires_grad_(False)

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_one,
        pin_memory=device.type == "cuda",
    )
    if len(loader) != args.steps:
        raise ValueError("C62 canary must consume one exact episode-shuffled epoch")
    flow_scheduler = C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    # Default-off is checked before any optimizer state exists.
    probe_sequence = materialize_sequence(provider, heldout_dataset[rank], device, dtype)
    probe_noisy, _, probe_t = C58.PARENT.PARENT.deterministic_flow_batch(
        probe_sequence["current"]["actions"], flow_scheduler, seed=args.seed + 9_000_001
    )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        parent_probe = parent(
            probe_noisy,
            probe_t,
            text_context=probe_sequence["current"]["text_context"],
            proprio=probe_sequence["current"]["proprio"],
            video_kv_cache=probe_sequence["current_kv"],
            text_mask=probe_sequence["current"]["text_mask"],
        ).float()
        off_probe = predict(
            model, probe_sequence, probe_noisy, probe_t, use_context=False
        ).float()
    pre_default_off_max_abs = float((parent_probe - off_probe).abs().max())
    if pre_default_off_max_abs != 0.0:
        raise RuntimeError("C62 pre-train default-off parent parity failed")

    ddp: nn.Module = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
    )
    raw_model = ddp.module
    trainable = list(raw_model.modulator.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(float(step + 1) / args.warmup_steps, 1.0)
    )
    ddp.train()
    raw_model.parent.eval()
    history = []
    consumed_ids = []
    for step, cpu_item in enumerate(loader, start=1):
        sequence = materialize_sequence(provider, cpu_item, device, dtype)
        consumed_ids.append(sequence["sequence_id"])
        noisy, target, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
            sequence["current"]["actions"],
            flow_scheduler,
            seed=C58.PARENT.PARENT.distributed_flow_seed(
                base_seed=args.seed,
                completed_step=step,
                accumulation_index=0,
                rank=rank,
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            prediction = predict(ddp, sequence, noisy, timesteps, use_context=True)
            loss = C58.PARENT.flow_matching_loss(
                prediction,
                target,
                timesteps,
                flow_scheduler,
                is_pad_mask=sequence["current"]["action_is_pad"],
            )
        loss.backward()
        refiner_gradients = [
            float(refiner[-1].weight.grad.float().norm())
            for refiner in raw_model.modulator.layer_refiners.values()
        ]
        shared_gradient = float(
            raw_model.modulator.shared_modulation.weight.grad.float().norm()
        )
        if not all(math.isfinite(value) and value > 0 for value in refiner_gradients):
            raise RuntimeError("C62 action loss did not reach all 30 bridge refiners")
        if not math.isfinite(shared_gradient) or shared_gradient <= 0:
            raise RuntimeError("C62 action loss did not reach shared action modulation")
        if any(parameter.grad is not None for parameter in raw_model.parent.parameters()):
            raise RuntimeError("C62 bridge-only canary leaked gradients into C58 parent")
        clipped = float(
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
        )
        optimizer.step()
        scheduler.step()
        if step in {1, 10, 100}:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "shared_gradient_norm": shared_gradient,
                    "min_refiner_gradient_norm": min(refiner_gradients),
                    "max_refiner_gradient_norm": max(refiner_gradients),
                    "clipped_gradient_norm": clipped,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )

    # Paired heldout: identical model/noise/target, changing only causal action
    # pairing among the same real observation chunks.
    raw_model.eval()
    local_eval = []
    with torch.no_grad():
        for index in range(rank, len(heldout_dataset), world_size):
            sequence = materialize_sequence(provider, heldout_dataset[index], device, dtype)
            noisy, target, timesteps = C58.PARENT.PARENT.deterministic_flow_batch(
                sequence["current"]["actions"], flow_scheduler, seed=args.seed + 1_000_000 + index
            )
            with torch.autocast("cuda", dtype=dtype):
                clean = predict(raw_model, sequence, noisy, timesteps, use_context=True).float()
                shuffled = predict(
                    raw_model, sequence, noisy, timesteps, use_context=True, shuffled=True
                ).float()
                off = predict(raw_model, sequence, noisy, timesteps, use_context=False).float()
            target_float = target.float()
            valid = (~sequence["current"]["action_is_pad"]).unsqueeze(-1)
            mse = lambda value: float(((value - target_float).square() * valid).sum() / (valid.sum() * value.shape[-1]).clamp_min(1))
            local_eval.append(
                {
                    "index": index,
                    "id": sequence["sequence_id"],
                    "clean_mse": mse(clean),
                    "shuffled_mse": mse(shuffled),
                    "off_mse": mse(off),
                    "shuffle_prediction_max_abs": float((clean - shuffled).abs().max()),
                }
            )
    gathered_eval: list[list[dict[str, Any]] | None] = [None] * world_size
    gathered_ids: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(gathered_eval, local_eval)
    dist.all_gather_object(gathered_ids, consumed_ids)
    complete_eval = [item for rank_items in gathered_eval if rank_items for item in rank_items]
    complete_ids = [item for rank_items in gathered_ids if rank_items for item in rank_items]
    if len(complete_eval) != 64 or len(complete_ids) != 800 or len(set(complete_ids)) != 800:
        raise RuntimeError("C62 canary sample coverage is not exact")
    means = {
        key: sum(float(item[key]) for item in complete_eval) / len(complete_eval)
        for key in ("clean_mse", "shuffled_mse", "off_mse")
    }
    shuffle_relative = (means["shuffled_mse"] - means["clean_mse"]) / means["shuffled_mse"]
    off_regression = (means["clean_mse"] - means["off_mse"]) / means["off_mse"]
    shuffle_delta_min = min(item["shuffle_prediction_max_abs"] for item in complete_eval)

    # Save only the delta; C58 remains an immutable external parent.
    bridge_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "completed_steps": args.steps,
        "bridge": {key: value.detach().cpu() for key, value in raw_model.modulator.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "contract": {
            "candidate": "C62_MINIWORLD_C58_ROLLING_CONTEXT_BRIDGE",
            "classification": "bridge-only-on-frozen-c58-and-frozen-int8-h3",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": parent_sha,
            "parent_completed_steps": 10_000,
            "train_manifest_sha256": train_dataset.manifest_sha256,
            "heldout_manifest_sha256": heldout_dataset.manifest_sha256,
            "plan_sha256": sha256_file(args.plan),
            "world_size": world_size,
            "steps": args.steps,
            "training_samples": 800,
            "history_chunks": 3,
            "replan": 8,
            "action_group": 4,
            "parent_trainable": False,
            "h3_trainable": False,
        },
    }
    if rank == 0:
        atomic_torch(args.save_checkpoint.resolve(), bridge_payload)
    dist.barrier()
    restored_payload = torch.load(args.save_checkpoint.resolve(), map_location="cpu", weights_only=False)
    strict = raw_model.modulator.load_state_dict(restored_payload["bridge"], strict=True)
    if strict.missing_keys or strict.unexpected_keys:
        raise RuntimeError("C62 bridge strict restore failed")
    optimizer.load_state_dict(restored_payload["optimizer"])
    scheduler.load_state_dict(restored_payload["lr_scheduler"])
    restored_state = MiniWorldRollingContextState.from_snapshot(
        context_state(raw_model, probe_sequence, shuffled=False)[0].snapshot(),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        expected = predict(raw_model, probe_sequence, probe_noisy, probe_t, use_context=True).float()
        restored_prediction = raw_model(
            probe_noisy,
            probe_t,
            text_context=probe_sequence["current"]["text_context"],
            proprio=probe_sequence["current"]["proprio"],
            video_kv_cache=probe_sequence["current_kv"],
            text_mask=probe_sequence["current"]["text_mask"],
            context_state=restored_state,
            actions_before_current=probe_sequence["actions_before_current"],
            use_context=True,
        ).float()
        post_off = predict(
            raw_model, probe_sequence, probe_noisy, probe_t, use_context=False
        ).float()
    restore_max_abs = float((expected - restored_prediction).abs().max())
    post_default_off_max_abs = float((parent_probe - post_off).abs().max())
    gates = {
        "parent_identity": all(parent_checks.values()),
        "train_unique_samples": len(set(complete_ids)) == 800,
        "heldout_episode_disjoint": plan["episode_intersection"] == 0,
        "all_30_bridge_refiners_gradient": all(
            row["min_refiner_gradient_norm"] > 0 for row in history
        ),
        "parent_gradients_absent": True,
        "pre_default_off_exact": pre_default_off_max_abs == 0.0,
        "post_default_off_exact": post_default_off_max_abs == 0.0,
        "bridge_restore_exact": restore_max_abs == 0.0,
        "shuffle_prediction_effect": shuffle_delta_min >= 1e-5,
        "clean_beats_shuffle_by_1pct": shuffle_relative >= 0.01,
        "clean_vs_off_regression_at_most_5pct": off_regression <= 0.05,
    }
    status = "PASS_C62_CAUSAL_OPTIMIZER_CANARY" if all(gates.values()) else "FAIL_C62_CAUSAL_OPTIMIZER_CANARY"
    report = {
        "event": "h3_c62_miniworld_c58_bridge_causal_optimizer_canary",
        "status": status,
        "permission": "GO_BOUNDED_C62_ABLATION" if all(gates.values()) else "NO_GO_C62_TRAINING",
        "effect_status": "NOT_LIBERO_EVIDENCE",
        "parent_checkpoint_sha256": parent_sha,
        "checkpoint": str(args.save_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.save_checkpoint.resolve()) if rank == 0 else None,
        "world_size": world_size,
        "steps": args.steps,
        "training_samples": 800,
        "effective_epochs": 1.0,
        "train_unique_samples": len(set(complete_ids)),
        "heldout_samples": len(complete_eval),
        "heldout": {
            **means,
            "clean_vs_shuffle_relative_improvement": shuffle_relative,
            "clean_vs_off_relative_regression": off_regression,
            "shuffle_prediction_max_abs_min": shuffle_delta_min,
        },
        "pre_default_off_max_abs": pre_default_off_max_abs,
        "post_default_off_max_abs": post_default_off_max_abs,
        "restore_max_abs": restore_max_abs,
        "gates": gates,
        "history": history,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "Short causal/optimizer canary only; closed-loop LIBERO is required for effectiveness.",
    }
    if rank == 0:
        atomic_json(args.output.resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
