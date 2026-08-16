#!/usr/bin/env python3
"""Train the full C56b FACT backbone port with frozen INT8 H3 online."""

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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam import (  # noqa: E402
    H3Int8FeatureBackbone,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    OnlineH3FACTDemoDataset,
    OnlineH3FACTRolloutDataset,
    encode_h3_vae_condition_standalone,
    fact_backbone_port_losses,
)
from fastwam.models.h3wam.fact_layerwise_tower import H3FACTLayerwiseTowerPolicy  # noqa: E402
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)
from fastwam.models.h3wam.int8_online import SEQUENCE_KV_POOL  # noqa: E402


FORMAT = "h3wam-c56b-fact-online-training-v1"
SCHEMA = 1
EXPECTED_H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
EXPECTED_D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
EXPECTED_C60_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
EXPECTED_C60_OBS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
FUTURE_DIM = 56 * 128
RANK_CATEGORIES = (
    "expert_demo", "expert_demo", "expert_demo", "expert_demo",
    "success_rollout", "success_rollout", "observational_failure", "causal_failure",
)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_script("_c56b_train_base", ROOT / "scripts/h3wam/probe_c56b_fact_online.py")
NORM = load_script("_c56b_train_norm", ROOT / "scripts/h3wam/fit_c56b_fact_online_target_norm.py")
C58 = BASE.C58
C58_ONLINE = BASE.C58_ONLINE


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--demo-cache-root", type=Path, required=True)
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--c59-overlay-root", type=Path, required=True)
    parser.add_argument("--c60-dataset", type=Path, required=True)
    parser.add_argument("--c60-observations", type=Path, required=True)
    parser.add_argument(
        "--expected-causal-dataset-sha256", default=EXPECTED_C60_SHA256
    )
    parser.add_argument(
        "--expected-causal-observations-sha256",
        default=EXPECTED_C60_OBS_SHA256,
    )
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--c58-parent-checkpoint", type=Path)
    parser.add_argument("--target-norm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--action-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--scheduler-horizon", type=int, default=10000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


class MixedPools:
    def __init__(self, a: argparse.Namespace) -> None:
        self.demo = OnlineH3FACTDemoDataset(
            a.demo_manifest, a.source_manifest, a.demo_cache_root, split="train"
        )
        self.c48 = OnlineH3FACTRolloutDataset(
            a.c48_dataset, a.c48_observations, a.source_manifest, a.demo_cache_root,
            split="train", c59_overlay_root=a.c59_overlay_root,
        )
        self.c60 = OnlineH3FACTRolloutDataset(
            a.c60_dataset, a.c60_observations, a.source_manifest, a.demo_cache_root,
            split="train",
            expected_dataset_sha256=a.expected_causal_dataset_sha256,
            expected_observations_sha256=a.expected_causal_observations_sha256,
        )
        pools: dict[str, list[list[int]]] = {
            "expert_demo": [list(v) for v in self.demo.episode_to_indices.values()],
            "success_rollout": [], "observational_failure": [],
            "causal_failure": [list(v) for v in self.c60.episode_to_indices.values()],
        }
        for indices in self.c48.episode_to_indices.values():
            first = self.c48.rows[indices[0]]
            target = self.c48.labels.for_sample(int(first["sample_id"]))
            category = (
                "success_rollout" if float(target["action_loss_mask"]) == 1.0
                else "observational_failure"
            )
            if any(
                float(self.c48.labels.for_sample(int(self.c48.rows[i]["sample_id"]))["action_loss_mask"])
                != float(target["action_loss_mask"])
                for i in indices
            ):
                raise ValueError("C48 action mask changes inside an episode")
            pools[category].append(list(indices))
        if any(not episodes for episodes in pools.values()):
            raise ValueError("C56b mixed training pool is empty")
        self.pools = pools

    def item(self, category: str, *, absolute_step: int, rank: int, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed + absolute_step * 1_000_003 + rank * 10_000_019)
        episodes = self.pools[category]
        episode = episodes[int(rng.integers(len(episodes)))]
        index = episode[int(rng.integers(len(episode)))]
        dataset = self.demo if category == "expert_demo" else (
            self.c60 if category == "causal_failure" else self.c48
        )
        item = dataset[index]
        expected = 1.0 if category in {"expert_demo", "success_rollout"} else 0.0
        if float(item["action_loss_mask"]) != expected:
            raise RuntimeError("C56b sampled action mask mismatch")
        return item


def encode(item: dict[str, Any], key: str, vae, device: torch.device) -> torch.Tensor:
    tensor = item[key].unsqueeze(0)
    if item["input_mode"] == "vae_latents":
        return tensor.to(device=device, dtype=torch.float32)
    if vae is None:
        raise ValueError("pixel stream requires online H3 VAE")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        return encode_h3_vae_condition_standalone(
            vae, tensor.to(device), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ).to(device=device, dtype=torch.float32)


def optimizer_groups(model: H3FACTLayerwiseTowerPolicy, base_lr: float, action_lr: float, wd: float):
    action_prefixes = (
        "tower.proprio_encoder.", "tower.action_expert.action_encoder.",
        "tower.action_expert.head.", "future_state_encoder.", "value_encoder.",
        "future_state_decoder.", "value_decoder.",
    )
    buckets: dict[tuple[str, bool], list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "action" if name.startswith(action_prefixes) else "base"
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        buckets.setdefault((group, no_decay), []).append(parameter)
    return [
        {
            "params": values,
            "lr": action_lr if group == "action" else base_lr,
            "weight_decay": 0.0 if no_decay else wd,
            "name": f"{group}{'_no_decay' if no_decay else ''}",
        }
        for (group, no_decay), values in buckets.items()
    ]


def schedule(optimizer, warmup: int, horizon: int):
    def factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(horizon - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def atomic_save(payload: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    required = sum(v.numel() * v.element_size() for v in payload["model"].values()) * 4
    if shutil.disk_usage(path.parent).free < required:
        raise OSError("insufficient storage for C56b resumable checkpoint")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def build_step(
    item: dict[str, Any], *, vae, provider, mean, std, model_device, seed: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    current = encode(item, "current_h3_input", vae, model_device)
    future = encode(item, "future_h3_input", vae, model_device)
    context = item["text_context"].unsqueeze(0).to(model_device, torch.float32)
    tags = item["text_token_tags"].to(model_device, torch.long)
    current_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(provider(current, context, tags))
    future_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(provider(future, context, tags))
    future_rep = BASE.future_representation_from_online_kv(future_kv).float()
    future_rep = (future_rep - mean.unsqueeze(0)) / std.unsqueeze(0)
    del current, future, future_kv
    actions = item["actions"].unsqueeze(0).to(model_device, torch.bfloat16)
    scheduler = C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy_actions, action_target, timestep = C58.PARENT.PARENT.deterministic_flow_batch(
        actions, scheduler, seed=seed
    )
    future_state = item["future_state"].unsqueeze(0).to(model_device, torch.bfloat16)
    value = item["value"].reshape(1, 1).to(model_device, torch.bfloat16)
    noisy_state, state_target = BASE.flow_corrupt(future_state, timestep, seed=seed + 1)
    noisy_value, value_target = BASE.flow_corrupt(value, timestep, seed=seed + 2)
    noisy_rep, rep_target = BASE.flow_corrupt(future_rep, timestep, seed=seed + 3)
    inputs = {
        "noisy_actions": noisy_actions, "timestep": timestep,
        "clean_actions": actions, "noisy_future_state": noisy_state,
        "noisy_value": noisy_value, "noisy_future_representation": noisy_rep,
        "text_context": context.to(torch.bfloat16),
        "proprio": item["proprio"].unsqueeze(0).to(model_device, torch.bfloat16),
        "video_kv_cache": current_kv,
        "text_mask": torch.ones(1, context.shape[1], dtype=torch.bool, device=model_device),
    }
    targets = {
        "action_target": action_target, "future_state_target": state_target,
        "value_target": value_target, "future_representation_target": rep_target,
        "action_is_pad": item["action_is_pad"].unsqueeze(0).to(model_device),
        "action_loss_mask": item["action_loss_mask"].reshape(1).to(model_device),
        "future_loss_mask": item["future_representation_loss_mask"].reshape(1).to(model_device),
        "future_state_loss_mask": item["future_state_loss_mask"].reshape(1).to(model_device),
        "value_loss_mask": item["value_loss_mask"].reshape(1).to(model_device),
    }
    return inputs, targets


def globally_normalize_masked_losses(
    losses: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], world: int
) -> dict[str, torch.Tensor]:
    """Make per-rank microbatch-one DDP equal FACT's global masked means."""

    masks = torch.stack(
        (
            targets["action_loss_mask"].float().sum(),
            targets["future_loss_mask"].float().sum(),
            targets["future_state_loss_mask"].float().sum(),
            targets["value_loss_mask"].float().sum(),
        )
    )
    dist.all_reduce(masks, op=dist.ReduceOp.SUM)
    if (masks <= 0).any():
        raise RuntimeError("C56b global FACT batch lost a supervision stream")
    names = (
        "action_loss", "future_representation_loss", "future_state_loss", "value_loss"
    )
    scaled = {
        name: losses[name] * (float(world) / masks[index])
        for index, name in enumerate(names)
    }
    scaled["loss"] = (
        10.0 * scaled["action_loss"]
        + scaled["future_representation_loss"]
        + 0.4 * scaled["future_state_loss"]
        + 0.4 * scaled["value_loss"]
    )
    return scaled


def main() -> None:
    a = args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world != 8 or not torch.cuda.is_available():
        raise RuntimeError("C56b training requires exactly eight CUDA ranks")
    if a.restore_check_only and a.load_checkpoint is None:
        raise ValueError("restore-check-only requires load-checkpoint")
    if min(a.steps, a.base_lr, a.action_lr, a.max_grad_norm) <= 0:
        raise ValueError("invalid C56b training arguments")
    for name, value in (
        ("causal dataset", a.expected_causal_dataset_sha256),
        ("causal observations", a.expected_causal_observations_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid expected {name} SHA256")
    if sha(a.h3_checkpoint.resolve()) != EXPECTED_H3_SHA256 or sha(a.d0_parent_checkpoint.resolve()) != EXPECTED_D0_SHA256:
        raise ValueError("C56b backbone identity mismatch")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    pools = MixedPools(a)
    norm = torch.load(a.target_norm.resolve(), map_location="cpu", weights_only=False)
    if norm.get("format") != NORM.FORMAT or norm.get("split") != "train" or int(norm.get("sample_count", 0)) != 512:
        raise ValueError("C56b normalization identity mismatch")
    mean, std = norm["mean"].float().to(device), norm["std"].float().to(device)

    from diffusers import AutoencoderKLMiniMaxH3
    vae = None
    if RANK_CATEGORIES[rank] != "expert_demo":
        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            a.h3_model.resolve(), subfolder="vae", torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        vae.requires_grad_(False)
    backbone = H3Int8FeatureBackbone.from_checkpoint(a.h3_checkpoint.resolve()).to(device).eval()
    backbone.requires_grad_(False)
    provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=LAYERWISE_H3_50_TO_ACTION_30, action_horizon=32,
            target_latent_frames=12, video_timestep=1.0,
            condition_video_timestep=1.0, capture_token_count=32,
            pool_strategy=SEQUENCE_KV_POOL,
        ),
    ).eval()
    spec = C58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        action_layers=30,
    )
    tower = C58.build_model(spec, device=device, dtype=torch.bfloat16, gradient_checkpointing=a.gradient_checkpointing)
    d0 = torch.load(a.d0_parent_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if a.c58_parent_checkpoint is None:
        initialization = initialize_full_tower_from_d0(tower, d0["model"]).to_dict()
        c58_parent_sha256 = None
    else:
        c58_parent = torch.load(
            a.c58_parent_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        c58_contract = c58_parent.get("contract", {})
        if (
            c58_contract.get("candidate") != "C58B_FASTWAM_FULL30_H3_LAYERWISE"
            or c58_contract.get("h3_execution") != "online_frozen_int8_per_rank_v1"
            or c58_contract.get("disk_kv_training_input") is not False
            or int(c58_parent.get("completed_steps", -1)) != 10000
        ):
            raise ValueError("C56b formal parent is not the fixed online C58b s10000 layerwise arm")
        tower.load_state_dict(c58_parent["model"], strict=True)
        c58_parent_sha256 = sha(a.c58_parent_checkpoint.resolve())
        initialization = {
            "initialization_contract": "strict_online_c58b_parent_v1",
            "c58_completed_steps": int(c58_parent["completed_steps"]),
        }
        del c58_parent
    del d0
    model = H3FACTLayerwiseTowerPolicy(tower, future_state_dim=8, future_representation_dim=FUTURE_DIM).to(device, torch.bfloat16)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, a.base_lr, a.action_lr, a.weight_decay),
        betas=(0.9, 0.95), eps=1e-8,
    )
    lr_scheduler = schedule(optimizer, a.warmup_steps, a.scheduler_horizon)
    contract = {
        "format": FORMAT, "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": list(RANK_CATEGORIES), "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": sha(a.target_norm.resolve()), "h3_sha256": EXPECTED_H3_SHA256,
        "d0_sha256": EXPECTED_D0_SHA256, "initialization": initialization,
        "c58_parent_sha256": c58_parent_sha256,
        # These identities make the C60/C61 pair independently auditable from
        # the checkpoint bytes.  The paired evaluator permits only the two
        # causal-failure hashes below to differ between arms.
        "demo_manifest_sha256": sha(a.demo_manifest.resolve()),
        "source_manifest_sha256": sha(a.source_manifest.resolve()),
        "demo_stats_sha256": sha(a.demo_cache_root.resolve() / "stats.pt"),
        "c48_dataset_sha256": sha(a.c48_dataset.resolve()),
        "c48_observations_sha256": sha(a.c48_observations.resolve()),
        "c59_completed_sha256": sha(a.c59_overlay_root.resolve() / "COMPLETED.json"),
        "c59_sample_labels_sha256": sha(pools.c48.labels.labels_path),
        "causal_failure_dataset_sha256": a.expected_causal_dataset_sha256,
        "causal_failure_observations_sha256": a.expected_causal_observations_sha256,
        "base_lr": a.base_lr, "action_lr": a.action_lr, "warmup_steps": a.warmup_steps,
        "scheduler_horizon": a.scheduler_horizon, "weight_decay": a.weight_decay,
        "max_grad_norm": a.max_grad_norm, "seed": a.seed,
        "gradient_checkpointing": a.gradient_checkpointing,
        "action_horizon": 32, "action_shift": 5.0,
        "h3_carrier_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }
    completed = 0
    loaded = None
    if a.load_checkpoint is not None:
        loaded = torch.load(a.load_checkpoint.resolve(), map_location="cpu", weights_only=False)
        if loaded.get("schema_version") != SCHEMA or loaded.get("contract") != contract:
            raise ValueError("C56b checkpoint schema mismatch")
        model.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        lr_scheduler.load_state_dict(loaded["lr_scheduler"])
        completed = int(loaded["completed_steps"])

    def probe_prediction(step: int) -> torch.Tensor:
        item = pools.item(RANK_CATEGORIES[rank], absolute_step=step, rank=rank, seed=a.seed)
        inputs, _ = build_step(
            item, vae=vae, provider=provider, mean=mean, std=std,
            model_device=device, seed=a.seed + step * 1_000_003 + rank * 10_000_019,
        )
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.forward_action(
                inputs["noisy_actions"], inputs["timestep"],
                text_context=inputs["text_context"], proprio=inputs["proprio"],
                video_kv_cache=inputs["video_kv_cache"], text_mask=inputs["text_mask"],
            ).float().cpu()
        return result

    restore_max_abs = None
    if loaded is not None:
        restored = probe_prediction(int(loaded["probe_step"]))
        restore_max_abs = float((restored - loaded["probe_predictions"][rank]).abs().max())
        if restore_max_abs != 0.0:
            raise RuntimeError(f"C56b restore mismatch: {restore_max_abs}")
    if a.restore_check_only:
        values = [torch.tensor(restore_max_abs, device=device)]
        dist.all_reduce(values[0], op=dist.ReduceOp.MAX)
        if rank == 0:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({
                "format": FORMAT, "status": "PASS_C56B_STRICT_RESTORE",
                "restore_max_abs": float(values[0]), "checkpoint": str(a.load_checkpoint),
                "effect_status": "NOT_EVIDENCE_READY",
            }, indent=2) + "\n")
        dist.destroy_process_group()
        return

    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for local_step in range(1, a.steps + 1):
        step_started = time.perf_counter()
        absolute = completed + local_step
        category = RANK_CATEGORIES[rank]
        item = pools.item(category, absolute_step=absolute, rank=rank, seed=a.seed)
        inputs, targets = build_step(
            item, vae=vae, provider=provider, mean=mean, std=std,
            model_device=device, seed=a.seed + absolute * 1_000_003 + rank * 10_000_019,
        )
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = ddp(**inputs)
            losses = fact_backbone_port_losses(predictions, **targets)
            losses = globally_normalize_masked_losses(losses, targets, world)
        action_value = float(losses["action_loss"].detach())
        if (action_value > 0) != (category in {"expert_demo", "success_rollout"}):
            raise RuntimeError("C56b per-step action mask failed")
        leak = 0.0
        if local_step == 1:
            changed = dict(inputs)
            changed["noisy_future_representation"] = torch.zeros_like(inputs["noisy_future_representation"])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                alternate = ddp(**changed)["action"].float()
            leak = float((predictions["action"].detach().float() - alternate).abs().max())
            if leak != 0.0:
                raise RuntimeError("C56b future target leaked into action")
        losses["loss"].backward()
        block_gradients = [C58.PARENT.PARENT.module_grad_norm(block) for block in model.shared_blocks]
        if not all(math.isfinite(v) and v > 0 for v in block_gradients):
            raise RuntimeError("C56b shared block gradient gate failed")
        if any(parameter.grad is not None for parameter in backbone.parameters()):
            raise RuntimeError("frozen H3 received gradients during C56b training")
        clipped = float(torch.nn.utils.clip_grad_norm_(ddp.parameters(), a.max_grad_norm, error_if_nonfinite=True))
        optimizer.step()
        lr_scheduler.step()
        metrics = torch.tensor([
            float(losses["loss"].detach()), float(losses["action_loss"].detach()),
            float(losses["future_representation_loss"].detach()),
            float(losses["future_state_loss"].detach()), float(losses["value_loss"].detach()),
            min(block_gradients), max(block_gradients), clipped, leak,
        ], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics[:5] /= world
        block_tensor = torch.tensor(block_gradients, device=device)
        dist.all_reduce(block_tensor, op=dist.ReduceOp.SUM)
        block_tensor /= world
        record = {
            "step": absolute, "loss": float(metrics[0]), "action_loss": float(metrics[1]),
            "future_representation_loss": float(metrics[2]), "future_state_loss": float(metrics[3]),
            "value_loss": float(metrics[4]), "sum_rank_min_block_grad": float(metrics[5]),
            "sum_rank_max_block_grad": float(metrics[6]), "sum_rank_clipped_grad": float(metrics[7]),
            "sum_rank_future_leak_abs": float(metrics[8]),
            "block_gradient_norms_mean_across_ranks": block_tensor.cpu().tolist(),
            "step_seconds": time.perf_counter() - step_started,
            "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
        }
        history.append(record)
        if rank == 0:
            print(json.dumps(record, sort_keys=True), flush=True)
    completed += a.steps

    model.eval()
    probe = probe_prediction(completed)
    probes: list[torch.Tensor | None] = [None] * world
    dist.all_gather_object(probes, probe)
    checkpoint_bytes = None
    if rank == 0 and a.save_checkpoint is not None:
        checkpoint_bytes = atomic_save({
            "schema_version": SCHEMA, "completed_steps": completed,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(), "contract": contract,
            "probe_step": completed, "probe_predictions": probes,
        }, a.save_checkpoint.resolve())
    dist.barrier()
    local_peak = torch.tensor(torch.cuda.max_memory_reserved(device), device=device, dtype=torch.long)
    dist.all_reduce(local_peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({
            "format": FORMAT, "status": "PASS_C56B_ONLINE_TRAINING_INVOCATION",
            "effect_status": "NOT_EVIDENCE_READY", "completed_steps": completed,
            "history": history, "contract": contract, "checkpoint": str(a.save_checkpoint),
            "checkpoint_bytes": checkpoint_bytes, "restore_at_load_max_abs": restore_max_abs,
            "max_peak_reserved_bytes": int(local_peak), "wall_seconds": time.perf_counter() - started,
            "claim_boundary": "Optimizer/restore evidence only; no held-out or rollout effect claim.",
        }, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
