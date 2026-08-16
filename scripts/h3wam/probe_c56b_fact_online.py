#!/usr/bin/env python3
"""Eight-rank one-step gate for online frozen-H3 C56b FACT training.

Every rank reads one immutable C60 RGB/state/action sample, encodes its current
and future observations, runs the frozen INT8 H3 twice, and keeps the resulting
thirty-layer K/V only in process memory.  Future/value losses then backpropagate
through the same thirty ActionDiT blocks used for action generation.  No H3
feature or K/V cache is accepted, read, or written by this probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam import (  # noqa: E402
    H3Int8FeatureBackbone,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    OnlineH3FACTRolloutDataset,
    collate_online_fact,
    encode_h3_vae_condition_standalone,
    fact_backbone_port_losses,
)
from fastwam.models.h3wam.fact_layerwise_tower import (  # noqa: E402
    H3FACTLayerwiseTowerPolicy,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
    initialize_full_tower_from_d0,
)
from fastwam.models.h3wam.int8_online import SEQUENCE_KV_POOL  # noqa: E402


FORMAT = "h3wam-c56b-fact-online-eight-gpu-one-step-v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
EXPECTED_D0_SHA256 = (
    "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
)
EXPECTED_C60_SHA256 = (
    "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
)
EXPECTED_C60_OBSERVATIONS_SHA256 = (
    "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
)
C58_ONLINE_COMMITS = (
    "ea43479",
    "2e1a0c8",
)
FUTURE_REPRESENTATION_DIM = 56 * 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C58 = load_script(
    "_c56b_online_c58_train",
    REPO_ROOT / "scripts/h3wam/train_h3_fastwam_full_tower.py",
)
C58_ONLINE = load_script(
    "_c56b_online_c58_probe",
    REPO_ROOT / "scripts/h3wam/probe_c58b_online_frozen_h3.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prepared-data-root", type=Path, required=True)
    parser.add_argument("--c60-dataset", type=Path, required=True)
    parser.add_argument("--c60-observations", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def flow_corrupt(
    clean: torch.Tensor, timestep: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    sigma = (timestep.float() / 1000.0).to(clean.dtype)
    sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
    return clean * (1.0 - sigma) + noise * sigma, noise - clean


def future_representation_from_online_kv(
    online_kv: dict[int, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Use the future H3 layer-49 value carrier without a learned projection."""

    if tuple(online_kv) != LAYERWISE_H3_50_TO_ACTION_30:
        raise ValueError("future H3 K/V must contain the exact thirty-layer order")
    value = online_kv[49].get("v")
    if value is None or value.ndim != 4 or tuple(value.shape[2:]) != (56, 128):
        raise ValueError("future H3 layer49 V must be [B,tokens,56,128]")
    result = value.mean(dim=1).flatten(1).clone()
    if result.shape[1] != FUTURE_REPRESENTATION_DIM:
        raise RuntimeError("future H3 representation width mismatch")
    if not torch.isfinite(result.float()).all():
        raise RuntimeError("future H3 representation is non-finite")
    return result


def cuda_memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8 or not torch.cuda.is_available():
        raise RuntimeError("C56b online probe requires exactly eight CUDA ranks")
    if min(args.learning_rate, args.max_grad_norm) <= 0 or args.weight_decay < 0:
        raise ValueError("invalid C56b optimizer arguments")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    output_root = args.output_root.resolve()
    rank_root = output_root / "ranks" / f"rank{rank:02d}"
    if rank_root.exists():
        raise FileExistsError(f"refusing existing rank output: {rank_root}")
    started = time.perf_counter()

    h3_path = args.h3_checkpoint.resolve()
    parent_path = args.d0_parent_checkpoint.resolve()
    if sha256_file(h3_path) != EXPECTED_H3_SHA256:
        raise ValueError("online C56b H3 checkpoint identity mismatch")
    if sha256_file(parent_path) != EXPECTED_D0_SHA256:
        raise ValueError("online C56b D0 parent identity mismatch")
    dataset = OnlineH3FACTRolloutDataset(
        args.c60_dataset,
        args.c60_observations,
        args.source_manifest,
        args.prepared_data_root,
        split="train",
        expected_dataset_sha256=EXPECTED_C60_SHA256,
        expected_observations_sha256=EXPECTED_C60_OBSERVATIONS_SHA256,
    )
    if len(dataset) < world_size:
        raise ValueError("C60 train split cannot provide one sample per rank")
    batch_cpu = collate_online_fact([dataset[rank]])
    if batch_cpu["stream"] != "c60_causal_failure" or batch_cpu["input_mode"] != "pixels":
        raise RuntimeError("C56b online probe escaped the C60 RGB stream")
    forbidden = {"video_kv_cache", "h3_features", "future_h3_target"}
    if forbidden & set(batch_cpu):
        raise RuntimeError("C56b online probe received a cached H3 artifact")

    from diffusers import AutoencoderKLMiniMaxH3

    vae_started = time.perf_counter()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    vae.requires_grad_(False)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        current_latents = encode_h3_vae_condition_standalone(
            vae,
            batch_cpu["current_h3_input"].to(device),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ).to(device=device, dtype=torch.float32)
        future_latents = encode_h3_vae_condition_standalone(
            vae,
            batch_cpu["future_h3_input"].to(device),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ).to(device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    del vae
    torch.cuda.empty_cache()
    vae_seconds = time.perf_counter() - vae_started

    h3_started = time.perf_counter()
    backbone = H3Int8FeatureBackbone.from_checkpoint(h3_path).to(device).eval()
    backbone.requires_grad_(False)
    provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=LAYERWISE_H3_50_TO_ACTION_30,
            action_horizon=32,
            target_latent_frames=12,
            video_timestep=1.0,
            condition_video_timestep=1.0,
            capture_token_count=32,
            pool_strategy=SEQUENCE_KV_POOL,
        ),
    ).eval()
    context = batch_cpu["text_context"].to(device=device, dtype=torch.float32)
    tags = batch_cpu["text_token_tags"][0, batch_cpu["text_mask"][0]].to(device)
    context = context[:, batch_cpu["text_mask"][0]]
    current_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(
        provider(current_latents, context, tags)
    )
    future_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(
        provider(future_latents, context, tags)
    )
    future_representation_clean = future_representation_from_online_kv(future_kv)
    torch.cuda.synchronize(device)
    del future_kv, current_latents, future_latents
    torch.cuda.empty_cache()
    h3_seconds = time.perf_counter() - h3_started

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    contract = parent.get("contract", {})
    expected_parent = {
        "candidate": "D0",
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "action_horizon": 32,
        "action_shift": 5.0,
    }
    mismatches = [
        key for key, value in expected_parent.items() if contract.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"online C56b parent contract mismatch: {mismatches}")
    spec = C58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        action_layers=30,
    )
    tower = C58.build_model(
        spec, device=device, dtype=dtype, gradient_checkpointing=False
    )
    expansion = initialize_full_tower_from_d0(tower, parent["model"])
    del parent
    model = H3FACTLayerwiseTowerPolicy(
        tower,
        future_state_dim=8,
        future_representation_dim=FUTURE_REPRESENTATION_DIM,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    actions = batch_cpu["actions"].to(device=device, dtype=dtype)
    proprio = batch_cpu["proprio"].to(device=device, dtype=dtype)
    future_state_clean = batch_cpu["future_state"].to(device=device, dtype=dtype)
    value_clean = batch_cpu["value"].to(device=device, dtype=dtype).reshape(-1, 1)
    text_mask = batch_cpu["text_mask"].to(device)
    action_is_pad = batch_cpu["action_is_pad"].to(device)
    action_loss_mask = batch_cpu["action_loss_mask"].to(device)
    future_mask = batch_cpu["future_representation_loss_mask"].to(device)
    value_mask = batch_cpu["value_loss_mask"].to(device)
    scheduler = C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy_actions, action_target, timestep = C58.PARENT.PARENT.deterministic_flow_batch(
        actions, scheduler, seed=args.seed + rank * 1009
    )
    noisy_state, state_target = flow_corrupt(
        future_state_clean, timestep, seed=args.seed + rank * 1009 + 1
    )
    noisy_value, value_target = flow_corrupt(
        value_clean, timestep, seed=args.seed + rank * 1009 + 2
    )
    noisy_representation, representation_target = flow_corrupt(
        future_representation_clean,
        timestep,
        seed=args.seed + rank * 1009 + 3,
    )
    model_inputs = {
        "noisy_actions": noisy_actions,
        "timestep": timestep,
        "clean_actions": actions,
        "noisy_future_state": noisy_state,
        "noisy_value": noisy_value,
        "noisy_future_representation": noisy_representation,
        "text_context": context.to(dtype=dtype),
        "proprio": proprio,
        "video_kv_cache": current_kv,
        "text_mask": text_mask[:, : context.shape[1]],
    }
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), autocast:
        parent_stage1 = model.tower(
            noisy_actions,
            timestep,
            text_context=model_inputs["text_context"],
            proprio=proprio,
            video_kv_cache=current_kv,
            text_mask=model_inputs["text_mask"],
        )
        c56b_stage1 = model.forward_action(
            noisy_actions,
            timestep,
            text_context=model_inputs["text_context"],
            proprio=proprio,
            video_kv_cache=current_kv,
            text_mask=model_inputs["text_mask"],
        )
    stage1_max_abs = float((parent_stage1.float() - c56b_stage1.float()).abs().max())
    if stage1_max_abs != 0.0:
        raise RuntimeError(f"online C56b changed C58 Stage1: {stage1_max_abs}")

    torch.cuda.reset_peak_memory_stats(device)
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(**model_inputs)
        losses = fact_backbone_port_losses(
            predictions,
            action_target=action_target,
            future_state_target=state_target,
            value_target=value_target,
            future_representation_target=representation_target,
            action_is_pad=action_is_pad,
            action_loss_mask=action_loss_mask,
            future_loss_mask=future_mask,
            value_loss_mask=value_mask,
        )
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("online C56b produced a non-finite loss")
    losses["loss"].backward()
    block_gradients = [
        C58.PARENT.PARENT.module_grad_norm(block) for block in model.shared_blocks
    ]
    if len(block_gradients) != 30 or not all(
        math.isfinite(value) and value > 0 for value in block_gradients
    ):
        raise RuntimeError("online FACT future loss did not update all shared blocks")
    encoder_gradient = C58.PARENT.PARENT.module_grad_norm(
        model.tower.action_expert.action_encoder
    )
    if not math.isfinite(encoder_gradient) or encoder_gradient <= 0:
        raise RuntimeError("online FACT future loss missed shared A/G encoder")
    h3_gradient_count = sum(
        parameter.grad is not None for parameter in backbone.parameters()
    )
    if h3_gradient_count != 0 or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("frozen online H3 unexpectedly received gradients")
    total_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - step_started
    memory = cuda_memory(device)

    rank_root.mkdir(parents=True)
    report = {
        "format": FORMAT,
        "status": "PASS_C56B_ONLINE_ONE_STEP",
        "effect_status": "NOT_EVIDENCE_READY",
        "rank": rank,
        "world_size": world_size,
        "sample_id": batch_cpu["sample_ids"][0],
        "stream": batch_cpu["stream"],
        "input_mode": batch_cpu["input_mode"],
        "h3_cache_role": "none_online_in_memory_only",
        "h3_trainable": False,
        "h3_gradient_parameter_count": h3_gradient_count,
        "h3_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
        "future_representation": "mean_tokens_layer49_value_7168d_no_projection",
        "stage1_max_abs": stage1_max_abs,
        "shared_blocks": len(model.shared_blocks),
        "min_shared_block_gradient_norm": min(block_gradients),
        "max_shared_block_gradient_norm": max(block_gradients),
        "shared_action_encoder_gradient_norm": encoder_gradient,
        "total_gradient_norm_before_clip": total_gradient_norm,
        "losses": {key: float(value.detach().float().cpu()) for key, value in losses.items()},
        "timing_seconds": {
            "online_vae_current_future": vae_seconds,
            "online_h3_current_future": h3_seconds,
            "fact_forward_backward_optimizer": step_seconds,
            "total": time.perf_counter() - started,
        },
        "memory": memory,
        "initialization": expansion.to_dict(),
        "c58_online_commits": list(C58_ONLINE_COMMITS),
        "claim_boundary": "Eight-rank mechanical one-step only; no optimization or rollout claim.",
    }
    (rank_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    if rank == 0:
        reports = [
            json.loads(
                (output_root / "ranks" / f"rank{index:02d}" / "report.json").read_text()
            )
            for index in range(world_size)
        ]
        aggregate = {
            "format": FORMAT,
            "status": "PASS_C56B_ONLINE_EIGHT_GPU_ONE_STEP",
            "effect_status": "NOT_EVIDENCE_READY",
            "world_size": world_size,
            "samples": [item["sample_id"] for item in reports],
            "all_no_h3_cache": all(
                item["h3_cache_role"] == "none_online_in_memory_only" for item in reports
            ),
            "all_h3_frozen_no_grad": all(
                item["h3_gradient_parameter_count"] == 0 for item in reports
            ),
            "max_stage1_abs": max(item["stage1_max_abs"] for item in reports),
            "min_shared_block_gradient_norm": min(
                item["min_shared_block_gradient_norm"] for item in reports
            ),
            "max_peak_reserved_bytes": max(
                item["memory"]["peak_reserved_bytes"] for item in reports
            ),
            "max_fact_step_seconds": max(
                item["timing_seconds"]["fact_forward_backward_optimizer"]
                for item in reports
            ),
            "claim_boundary": "Mechanical gate only; NOT_EVIDENCE_READY.",
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "AGGREGATE.json").write_text(
            json.dumps(aggregate, indent=2) + "\n"
        )
        print(json.dumps(aggregate, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
