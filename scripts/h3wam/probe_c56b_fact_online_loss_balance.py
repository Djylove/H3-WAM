#!/usr/bin/env python3
"""Compare raw and train-normalized future-H3 losses on mixed FACT streams."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam import (  # noqa: E402
    H3Int8FeatureBackbone,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    OnlineH3FACTDemoDataset,
    OnlineH3FACTRolloutDataset,
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


FORMAT = "h3wam-c56b-fact-online-loss-balance-v1"
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
FUTURE_REPRESENTATION_DIM = 56 * 128


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ONLINE = load_script(
    "_c56b_balance_online", REPO_ROOT / "scripts/h3wam/probe_c56b_fact_online.py"
)
NORM = load_script(
    "_c56b_balance_norm",
    REPO_ROOT / "scripts/h3wam/fit_c56b_fact_online_target_norm.py",
)
C58 = ONLINE.C58
C58_ONLINE = ONLINE.C58_ONLINE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--demo-cache-root", type=Path, required=True)
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--c59-overlay-root", type=Path, required=True)
    parser.add_argument("--c60-dataset", type=Path, required=True)
    parser.add_argument("--c60-observations", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--d0-parent-checkpoint", type=Path, required=True)
    parser.add_argument("--target-norm", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def gradient_norm(loss: torch.Tensor, parameters: Iterable[torch.Tensor]) -> float:
    params = tuple(parameters)
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    square = 0.0
    for gradient in gradients:
        if gradient is not None:
            square += float(gradient.detach().float().square().sum().cpu())
    return math.sqrt(square)


def select_rank_sample(args: argparse.Namespace, rank: int) -> tuple[str, dict[str, Any]]:
    demo = OnlineH3FACTDemoDataset(
        args.demo_manifest,
        args.source_manifest,
        args.demo_cache_root,
        split="train",
    )
    c48 = OnlineH3FACTRolloutDataset(
        args.c48_dataset,
        args.c48_observations,
        args.source_manifest,
        args.demo_cache_root,
        split="train",
        c59_overlay_root=args.c59_overlay_root,
    )
    c60 = OnlineH3FACTRolloutDataset(
        args.c60_dataset,
        args.c60_observations,
        args.source_manifest,
        args.demo_cache_root,
        split="train",
        expected_dataset_sha256=EXPECTED_C60_SHA256,
        expected_observations_sha256=EXPECTED_C60_OBSERVATIONS_SHA256,
    )
    c48_success: list[int] = []
    c48_failure: list[int] = []
    for index, row in enumerate(c48.rows):
        target = c48.labels.for_sample(int(row["sample_id"]))
        (c48_success if float(target["action_loss_mask"]) == 1.0 else c48_failure).append(index)
    if rank < 4:
        indices = NORM.deterministic_pool_indices(len(demo), 4, args.seed + 11)
        return "expert_demo", demo[indices[rank]]
    if rank < 6:
        positions = NORM.deterministic_pool_indices(len(c48_success), 2, args.seed + 22)
        return "success_rollout", c48[c48_success[positions[rank - 4]]]
    if rank == 6:
        position = NORM.deterministic_pool_indices(len(c48_failure), 1, args.seed + 33)[0]
        return "observational_failure", c48[c48_failure[position]]
    position = NORM.deterministic_pool_indices(len(c60), 1, args.seed + 44)[0]
    return "causal_failure", c60[position]


def encode_input(
    item: dict[str, Any], key: str, vae: torch.nn.Module | None, device: torch.device
) -> torch.Tensor:
    value = item[key].unsqueeze(0)
    if item["input_mode"] == "vae_latents":
        return value.to(device=device, dtype=torch.float32)
    if item["input_mode"] != "pixels" or vae is None:
        raise ValueError("online FACT input mode/VAE mismatch")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        return encode_h3_vae_condition_standalone(
            vae,
            value.to(device),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ).to(device=device, dtype=torch.float32)


def evaluate_arm(
    model: H3FACTLayerwiseTowerPolicy,
    *,
    clean_representation: torch.Tensor,
    common: dict[str, torch.Tensor | dict],
    selected_parameters: tuple[torch.Tensor, ...],
    seed: int,
) -> dict[str, Any]:
    noisy_representation, representation_target = ONLINE.flow_corrupt(
        clean_representation,
        common["timestep"],
        seed=seed,
    )
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictions = model(
            noisy_actions=common["noisy_actions"],
            timestep=common["timestep"],
            clean_actions=common["actions"],
            noisy_future_state=common["noisy_state"],
            noisy_value=common["noisy_value"],
            noisy_future_representation=noisy_representation,
            text_context=common["context"],
            proprio=common["proprio"],
            video_kv_cache=common["current_kv"],
            text_mask=common["text_mask"],
        )
        losses = fact_backbone_port_losses(
            predictions,
            action_target=common["action_target"],
            future_state_target=common["state_target"],
            value_target=common["value_target"],
            future_representation_target=representation_target,
            action_is_pad=common["action_is_pad"],
            action_loss_mask=common["action_loss_mask"],
            future_loss_mask=common["future_representation_loss_mask"],
            future_state_loss_mask=common["future_state_loss_mask"],
            value_loss_mask=common["value_loss_mask"],
        )
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("C56b balance arm produced non-finite loss")
    weighted_action_gradient = gradient_norm(
        10.0 * losses["action_loss"], selected_parameters
    )
    weighted_future_gradient = gradient_norm(
        losses["future_representation_loss"], selected_parameters
    )
    losses["loss"].backward()
    block_gradients = [
        C58.PARENT.PARENT.module_grad_norm(block) for block in model.shared_blocks
    ]
    if not all(math.isfinite(value) and value > 0 for value in block_gradients):
        raise RuntimeError("C56b balance arm missed a shared block")
    return {
        "clean_representation_rms": float(clean_representation.float().square().mean().sqrt()),
        "losses": {key: float(value.detach().float().cpu()) for key, value in losses.items()},
        "weighted_action_selected_gradient_norm": weighted_action_gradient,
        "weighted_future_selected_gradient_norm": weighted_future_gradient,
        "weighted_future_to_action_gradient_ratio": (
            weighted_future_gradient / weighted_action_gradient
            if weighted_action_gradient > 0
            else None
        ),
        "min_shared_block_gradient_norm": min(block_gradients),
        "max_shared_block_gradient_norm": max(block_gradients),
        "action_prediction": predictions["action"].detach().float().cpu(),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8 or not torch.cuda.is_available():
        raise RuntimeError("C56b loss-balance probe requires exactly eight CUDA ranks")
    if sha256_file(args.h3_checkpoint.resolve()) != EXPECTED_H3_SHA256:
        raise ValueError("C56b balance H3 identity mismatch")
    if sha256_file(args.d0_parent_checkpoint.resolve()) != EXPECTED_D0_SHA256:
        raise ValueError("C56b balance D0 identity mismatch")
    norm_payload = torch.load(args.target_norm.resolve(), map_location="cpu", weights_only=False)
    if (
        norm_payload.get("format") != NORM.FORMAT
        or norm_payload.get("norm_contract") != NORM.NORM_CONTRACT
        or norm_payload.get("split") != "train"
        or int(norm_payload.get("sample_count", 0)) != 512
        or norm_payload.get("mixture_counts")
        != {name: count * 32 for name, count in NORM.MIXTURE_COUNTS_PER_16.items()}
    ):
        raise ValueError("C56b target normalization contract mismatch")
    mean = norm_payload["mean"].float()
    std = norm_payload["std"].float()
    if mean.shape != (FUTURE_REPRESENTATION_DIM,) or std.shape != mean.shape:
        raise ValueError("C56b target normalization shape mismatch")

    output_root = args.output_root.resolve()
    rank_root = output_root / "ranks" / f"rank{rank:02d}"
    if rank_root.exists():
        raise FileExistsError(f"refusing existing balance rank output: {rank_root}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)
    category, item = select_rank_sample(args, rank)
    expected_action_mask = 1.0 if category in {"expert_demo", "success_rollout"} else 0.0
    if float(item["action_loss_mask"]) != expected_action_mask:
        raise RuntimeError("C56b mixed rank action mask mismatch")

    vae = None
    if item["input_mode"] == "pixels":
        from diffusers import AutoencoderKLMiniMaxH3

        vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.h3_model.resolve(),
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        vae.requires_grad_(False)
    current_latents = encode_input(item, "current_h3_input", vae, device)
    future_latents = encode_input(item, "future_h3_input", vae, device)
    del vae
    torch.cuda.empty_cache()

    backbone = H3Int8FeatureBackbone.from_checkpoint(args.h3_checkpoint.resolve()).to(device).eval()
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
    context = item["text_context"].unsqueeze(0).to(device=device, dtype=torch.float32)
    tags = item["text_token_tags"].to(device=device, dtype=torch.long)
    current_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(
        provider(current_latents, context, tags)
    )
    future_kv = C58_ONLINE.materialize_kv_for_autograd_consumer(
        provider(future_latents, context, tags)
    )
    raw_representation = ONLINE.future_representation_from_online_kv(future_kv).float()
    normalized_representation = (
        raw_representation - mean.to(device).unsqueeze(0)
    ) / std.to(device).unsqueeze(0)
    del future_kv, current_latents, future_latents
    torch.cuda.empty_cache()

    parent = torch.load(args.d0_parent_checkpoint.resolve(), map_location="cpu", weights_only=False)
    spec = C58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        action_layers=30,
    )
    tower = C58.build_model(spec, device=device, dtype=torch.bfloat16, gradient_checkpointing=False)
    expansion = initialize_full_tower_from_d0(tower, parent["model"])
    del parent
    model = H3FACTLayerwiseTowerPolicy(
        tower,
        future_state_dim=8,
        future_representation_dim=FUTURE_REPRESENTATION_DIM,
    ).to(device=device, dtype=torch.bfloat16)
    model.train()

    actions = item["actions"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    proprio = item["proprio"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    future_state = item["future_state"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    value = item["value"].reshape(1, 1).to(device=device, dtype=torch.bfloat16)
    scheduler = C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy_actions, action_target, timestep = C58.PARENT.PARENT.deterministic_flow_batch(
        actions, scheduler, seed=args.seed + rank * 1009
    )
    noisy_state, state_target = ONLINE.flow_corrupt(
        future_state, timestep, seed=args.seed + rank * 1009 + 1
    )
    noisy_value, value_target = ONLINE.flow_corrupt(
        value, timestep, seed=args.seed + rank * 1009 + 2
    )
    common = {
        "actions": actions,
        "noisy_actions": noisy_actions,
        "action_target": action_target,
        "timestep": timestep,
        "noisy_state": noisy_state,
        "state_target": state_target,
        "noisy_value": noisy_value,
        "value_target": value_target,
        "context": context.to(torch.bfloat16),
        "proprio": proprio,
        "current_kv": current_kv,
        "text_mask": torch.ones(1, context.shape[1], dtype=torch.bool, device=device),
        "action_is_pad": item["action_is_pad"].unsqueeze(0).to(device),
        "action_loss_mask": item["action_loss_mask"].reshape(1).to(device),
        "future_representation_loss_mask": item["future_representation_loss_mask"].reshape(1).to(device),
        "future_state_loss_mask": item["future_state_loss_mask"].reshape(1).to(device),
        "value_loss_mask": item["value_loss_mask"].reshape(1).to(device),
    }
    selected_parameters = (
        model.tower.action_expert.action_encoder.weight,
        *(block.modulation for block in model.shared_blocks),
    )
    torch.cuda.reset_peak_memory_stats(device)
    raw = evaluate_arm(
        model,
        clean_representation=raw_representation,
        common=common,
        selected_parameters=selected_parameters,
        seed=args.seed + rank * 1009 + 3,
    )
    normalized = evaluate_arm(
        model,
        clean_representation=normalized_representation,
        common=common,
        selected_parameters=selected_parameters,
        seed=args.seed + rank * 1009 + 3,
    )
    action_max_abs = float(
        (raw.pop("action_prediction") - normalized.pop("action_prediction")).abs().max()
    )
    expected_nonzero = expected_action_mask == 1.0
    action_loss = normalized["losses"]["action_loss"]
    if (action_loss > 0) != expected_nonzero:
        raise RuntimeError("C56b success/failure action loss gate failed")
    if action_max_abs != 0.0:
        raise RuntimeError("future target leaked into the causal action prediction")
    if normalized["losses"]["future_representation_loss"] >= raw["losses"]["future_representation_loss"]:
        raise RuntimeError("train-only target scaling did not reduce the raw future loss")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError("frozen H3 received gradients in balance probe")

    rank_root.mkdir(parents=True)
    report = {
        "format": FORMAT,
        "status": "PASS_C56B_MIXED_LOSS_BALANCE_RANK",
        "effect_status": "NOT_EVIDENCE_READY",
        "rank": rank,
        "category": category,
        "sample_id": str(item["sample_id"]),
        "input_mode": str(item["input_mode"]),
        "action_loss_mask": float(item["action_loss_mask"]),
        "future_representation_loss_mask": float(item["future_representation_loss_mask"]),
        "future_state_loss_mask": float(item["future_state_loss_mask"]),
        "value_loss_mask": float(item["value_loss_mask"]),
        "target_norm_sha256": sha256_file(args.target_norm.resolve()),
        "raw": raw,
        "normalized": normalized,
        "raw_to_normalized_future_loss_ratio": (
            raw["losses"]["future_representation_loss"]
            / normalized["losses"]["future_representation_loss"]
        ),
        "raw_normalized_action_prediction_max_abs": action_max_abs,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "initialization": expansion.to_dict(),
        "claim_boundary": "Mixed-stream raw/normalized gradient diagnostic only; no optimizer step or checkpoint.",
    }
    (rank_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    if rank == 0:
        reports = [
            json.loads((output_root / "ranks" / f"rank{index:02d}" / "report.json").read_text())
            for index in range(world_size)
        ]
        aggregate = {
            "format": FORMAT,
            "status": "PASS_C56B_MIXED_LOSS_BALANCE",
            "effect_status": "NOT_EVIDENCE_READY",
            "categories": [item["category"] for item in reports],
            "successful_action_losses_nonzero": all(
                item["normalized"]["losses"]["action_loss"] > 0
                for item in reports
                if item["action_loss_mask"] == 1.0
            ),
            "failure_action_losses_zero": all(
                item["normalized"]["losses"]["action_loss"] == 0
                for item in reports
                if item["action_loss_mask"] == 0.0
            ),
            "max_action_target_leak_abs": max(
                item["raw_normalized_action_prediction_max_abs"] for item in reports
            ),
            "raw_future_loss_range": [
                min(item["raw"]["losses"]["future_representation_loss"] for item in reports),
                max(item["raw"]["losses"]["future_representation_loss"] for item in reports),
            ],
            "normalized_future_loss_range": [
                min(item["normalized"]["losses"]["future_representation_loss"] for item in reports),
                max(item["normalized"]["losses"]["future_representation_loss"] for item in reports),
            ],
            "successful_weighted_future_to_action_gradient_ratio_range": [
                min(
                    item["normalized"]["weighted_future_to_action_gradient_ratio"]
                    for item in reports if item["action_loss_mask"] == 1.0
                ),
                max(
                    item["normalized"]["weighted_future_to_action_gradient_ratio"]
                    for item in reports if item["action_loss_mask"] == 1.0
                ),
            ],
            "min_normalized_shared_block_gradient_norm": min(
                item["normalized"]["min_shared_block_gradient_norm"] for item in reports
            ),
            "max_peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in reports),
            "target_norm_sha256": reports[0]["target_norm_sha256"],
            "permission": "PROBE_COMPLETE_REVIEW_BEFORE_MIXED_ONE_STEP",
            "claim_boundary": "No effectiveness or GO_LONG claim.",
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "AGGREGATE.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        print(json.dumps(aggregate, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
