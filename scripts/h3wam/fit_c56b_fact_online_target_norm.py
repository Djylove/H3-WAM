#!/usr/bin/env python3
"""Fit C56b future-H3 scaling from online train samples only.

This is a bounded distributed calibration probe, not feature caching.  Each
sample is decoded/encoded and passed through frozen INT8 H3 online; only the
7168-dimensional train-distribution mean/std is retained.  The registered
mixture is 8 expert demo, 4 successful rollout, 2 observational failure, and
2 causal failure samples per logical batch of sixteen, matching the intended
FACT training mixture while guaranteeing all supervision regimes occur.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
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
    OnlineH3FACTDemoDataset,
    OnlineH3FACTRolloutDataset,
    encode_h3_vae_condition_standalone,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
)
from fastwam.models.h3wam.int8_online import SEQUENCE_KV_POOL  # noqa: E402


FORMAT = "h3wam-c56b-fact-online-target-norm-v1"
NORM_CONTRACT = "train-only-mixture-weighted-per-dimension-zscore-v1"
FUTURE_REPRESENTATION_DIM = 56 * 128
MIXTURE_COUNTS_PER_16 = {
    "expert_demo": 8,
    "success_rollout": 4,
    "observational_failure": 2,
    "causal_failure": 2,
}
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
EXPECTED_C60_SHA256 = (
    "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
)
EXPECTED_C60_OBSERVATIONS_SHA256 = (
    "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ONLINE = load_script(
    "_c56b_online_norm_base", REPO_ROOT / "scripts/h3wam/probe_c56b_fact_online.py"
)


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def registered_stream_schedule(sample_count: int) -> list[str]:
    if sample_count <= 0 or sample_count % 16:
        raise ValueError("sample_count must be a positive multiple of 16")
    block: list[str] = []
    for stream, count in MIXTURE_COUNTS_PER_16.items():
        block.extend([stream] * count)
    if len(block) != 16:
        raise RuntimeError("registered FACT mixture must contain sixteen samples")
    return block * (sample_count // 16)


def deterministic_pool_indices(size: int, count: int, seed: int) -> list[int]:
    if size <= 0 or count <= 0 or count > size:
        raise ValueError("normalization pool cannot satisfy unique sampling")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator)[:count].tolist()


def fit_mean_std_from_moments(
    total: torch.Tensor, square_total: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if total.shape != square_total.shape or total.ndim != 1 or count <= 1:
        raise ValueError("invalid future-H3 calibration moments")
    mean = total / float(count)
    variance = (square_total / float(count) - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("future-H3 calibration statistics are non-finite")
    return mean, std


def select_train_samples(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, Any, int]], dict[str, int]]:
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
        destination = (
            c48_success if float(target["action_loss_mask"]) == 1.0 else c48_failure
        )
        destination.append(index)
    pools: dict[str, tuple[Any, list[int]]] = {
        "expert_demo": (demo, list(range(len(demo)))),
        "success_rollout": (c48, c48_success),
        "observational_failure": (c48, c48_failure),
        "causal_failure": (c60, list(range(len(c60)))),
    }
    schedule = registered_stream_schedule(args.sample_count)
    requested = {name: schedule.count(name) for name in MIXTURE_COUNTS_PER_16}
    selected: dict[str, list[int]] = {}
    for ordinal, (name, (_, indices)) in enumerate(pools.items()):
        positions = deterministic_pool_indices(
            len(indices), requested[name], args.seed + 100003 * (ordinal + 1)
        )
        selected[name] = [indices[position] for position in positions]
    cursors = {name: 0 for name in pools}
    samples: list[tuple[str, Any, int]] = []
    for name in schedule:
        dataset, _ = pools[name]
        index = selected[name][cursors[name]]
        cursors[name] += 1
        samples.append((name, dataset, index))
    return samples, requested


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8 or not torch.cuda.is_available():
        raise RuntimeError("C56b online target calibration requires exactly eight CUDA ranks")
    if args.sample_count % world_size:
        raise ValueError("sample_count must divide evenly across eight ranks")
    if sha256_file(args.h3_checkpoint.resolve()) != EXPECTED_H3_SHA256:
        raise ValueError("online C56b H3 checkpoint identity mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing existing normalization output: {output_root}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    samples, requested = select_train_samples(args)
    local_descriptors = samples[rank::world_size]
    local_samples = []
    for expected_stream, dataset, index in local_descriptors:
        item = dataset[index]
        if str(item["stream"]) == "expert_demo":
            actual_stream = "expert_demo"
        elif str(item["stream"]) == "c60_causal_failure":
            actual_stream = "causal_failure"
        elif float(item["action_loss_mask"]) == 1.0:
            actual_stream = "success_rollout"
        else:
            actual_stream = "observational_failure"
        if actual_stream != expected_stream:
            raise RuntimeError(
                f"FACT normalization schedule mismatch: {expected_stream} != {actual_stream}"
            )
        local_samples.append(item)
    if len(local_samples) * world_size != len(samples):
        raise RuntimeError("rank calibration samples are imbalanced")

    from diffusers import AutoencoderKLMiniMaxH3

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(),
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    vae.requires_grad_(False)
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

    total = torch.zeros(FUTURE_REPRESENTATION_DIM, dtype=torch.float64, device=device)
    square_total = torch.zeros_like(total)
    local_counts = {name: 0 for name in MIXTURE_COUNTS_PER_16}
    for item in local_samples:
        mode = str(item["input_mode"])
        future_input = item["future_h3_input"].unsqueeze(0)
        if mode == "pixels":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                latents = encode_h3_vae_condition_standalone(
                    vae,
                    future_input.to(device),
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ).to(device=device, dtype=torch.float32)
        elif mode == "vae_latents":
            latents = future_input.to(device=device, dtype=torch.float32)
        else:
            raise ValueError(f"unsupported online H3 input mode {mode!r}")
        context = item["text_context"].unsqueeze(0).to(device=device, dtype=torch.float32)
        tags = item["text_token_tags"].to(device=device, dtype=torch.long)
        future_kv = provider(latents, context, tags)
        representation = ONLINE.future_representation_from_online_kv(future_kv)[0].double()
        total += representation
        square_total += representation.square()
        if str(item["stream"]) == "expert_demo":
            label = "expert_demo"
        elif str(item["stream"]) == "c60_causal_failure":
            label = "causal_failure"
        elif float(item["action_loss_mask"]) == 1.0:
            label = "success_rollout"
        else:
            label = "observational_failure"
        local_counts[label] += 1
        del future_kv, representation, latents

    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(square_total, op=dist.ReduceOp.SUM)
    count_tensor = torch.tensor([len(local_samples)], dtype=torch.long, device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    count = int(count_tensor.item())
    mean, std = fit_mean_std_from_moments(total, square_total, count)
    raw_rms = math.sqrt(float((square_total / count).mean().item()))
    normalized_second_moment = float(
        (((square_total / count) - 2 * mean * mean + mean.square()) / std.square()).mean().item()
    )
    count_values = torch.tensor(
        [local_counts[name] for name in MIXTURE_COUNTS_PER_16],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(count_values, op=dist.ReduceOp.SUM)
    observed = {
        name: int(value)
        for name, value in zip(MIXTURE_COUNTS_PER_16, count_values.tolist())
    }
    if observed != requested or count != args.sample_count:
        raise RuntimeError(f"calibration mixture mismatch: {observed} != {requested}")

    if rank == 0:
        output_root.mkdir(parents=True)
        stats_path = output_root / "target_norm.pt"
        torch.save(
            {
                "format": FORMAT,
                "norm_contract": NORM_CONTRACT,
                "split": "train",
                "sample_count": count,
                "mixture_counts": observed,
                "mean": mean.float().cpu(),
                "std": std.float().cpu(),
            },
            stats_path,
        )
        quantiles = torch.quantile(
            std.float(), torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=device)
        ).cpu().tolist()
        report = {
            "format": FORMAT,
            "status": "PASS_C56B_TRAIN_ONLY_TARGET_NORM",
            "effect_status": "NOT_EVIDENCE_READY",
            "split": "train",
            "sample_count": count,
            "mixture_counts": observed,
            "norm_contract": NORM_CONTRACT,
            "future_representation": "mean_tokens_layer49_value_7168d_no_projection",
            "raw_target_rms": raw_rms,
            "normalized_target_rms": math.sqrt(max(normalized_second_moment, 0.0)),
            "std_quantiles_0_01_50_99_100": quantiles,
            "clamped_std_dimensions": int((std <= 1e-6).sum().item()),
            "target_norm_sha256": sha256_file(stats_path),
            "official_fact_boundary": (
                "FACT whitens Wan VAE latents with pretrained latents_mean/std before "
                "flow corruption; H3 K/V has no upstream whitening constants, so this "
                "backbone port fits equivalent scale from train samples only."
            ),
            "claim_boundary": (
                "Train-only online scale calibration; no H3 feature/KV cache, optimizer "
                "step, validation input, checkpoint, or effectiveness claim."
            ),
        }
        (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
