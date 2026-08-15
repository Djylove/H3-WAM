#!/usr/bin/env python3
"""Extract exact live H3 features for the 32 canonical C26 branch states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fastwam.models.h3wam import (
    H3Int8FeatureBackbone,
    H3Int8OnlineFeatureContract,
    H3Int8OnlineFeatureProvider,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    compact_h3_kv_progress_feature,
    encode_h3_vae_condition_standalone,
    preprocess_libero_cameras,
)


FORMAT_BY_DATASET = {
    "h3wam-c26-causal-critic-dataset-v1": "h3wam-c26-live-h3-features-v1",
    "h3wam-c27-causal-critic-dataset-v1": "h3wam-c27-live-h3-features-v1",
    "h3wam-c31-action-conditioned-consequence-dataset-v1": "h3wam-c31-live-h3-consequence-features-v1",
    "h3wam-c34-combined-consequence-ranking-dataset-v1": "h3wam-c34-live-h3-consequence-features-v1",
    "h3wam-c44-powered-consequence-ranking-dataset-v1": "h3wam-c44-live-h3-consequence-features-v1",
}
CONSEQUENCE_DATASETS = {
    "h3wam-c31-action-conditioned-consequence-dataset-v1",
    "h3wam-c34-combined-consequence-ranking-dataset-v1",
    "h3wam-c44-powered-consequence-ranking-dataset-v1",
}
EXPECTED_H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
EXPECTED_SOURCE_SHA256 = "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def task_context_ids(source_manifest: Path, tasks: set[str]) -> dict[str, str]:
    matches: dict[str, set[str]] = defaultdict(set)
    with source_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task = str(row["task"])
            if task in tasks:
                matches[task].add(str(row["context_id"]))
    missing = sorted(tasks - set(matches))
    ambiguous = {task: sorted(ids) for task, ids in matches.items() if len(ids) != 1}
    if missing or ambiguous:
        raise ValueError(f"task/context mapping invalid: missing={missing}, ambiguous={ambiguous}")
    return {task: next(iter(ids)) for task, ids in matches.items()}


def pool_feature_tokens(features: torch.Tensor, token_count: int = 32) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError("features must be [layers,tokens,hidden]")
    if features.shape[1] <= token_count:
        return features
    return F.adaptive_avg_pool1d(features.transpose(1, 2), token_count).transpose(1, 2)


def feature_samples(dataset: dict) -> list[dict]:
    """Return current-state samples, plus future branch targets for C31."""
    states = dataset.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("causal H3 feature extraction requires non-empty states")
    if [int(state["group_id"]) for state in states] != list(range(len(states))):
        raise ValueError("causal state group ids must be contiguous and ordered")
    samples = [
        {
            "kind": "current", "index": int(state["group_id"]),
            "task_language": state["task_language"],
            "agentview_image": state["agentview_image"],
            "wristview_image": state["wristview_image"],
        }
        for state in states
    ]
    if dataset.get("format") in CONSEQUENCE_DATASETS:
        branches = dataset.get("branches")
        if not isinstance(branches, list) or not branches:
            raise ValueError("C31 consequence dataset requires branches")
        if [int(branch["ordinal"]) for branch in branches] != list(range(len(branches))):
            raise ValueError("C31 branch ordinals must be contiguous and ordered")
        for branch in branches:
            state = states[int(branch["group_id"])]
            samples.append({
                "kind": "future", "index": int(branch["ordinal"]),
                "task_language": state["task_language"],
                "agentview_image": branch["future_agentview_image"],
                "wristview_image": branch["future_wristview_image"],
            })
    return samples


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    dataset_path = args.dataset.resolve()
    source_manifest = args.source_manifest.resolve()
    h3_checkpoint = args.h3_checkpoint.resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset.get("format") not in FORMAT_BY_DATASET:
        raise ValueError("unsupported C26 dataset format")
    samples = feature_samples(dataset)
    h3_sha256 = sha256_file(h3_checkpoint)
    source_sha256 = sha256_file(source_manifest)
    if h3_sha256 != EXPECTED_H3_SHA256:
        raise ValueError(f"H3 checkpoint SHA256 mismatch: {h3_sha256}")
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"source manifest SHA256 mismatch: {source_sha256}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("live INT8 H3 feature extraction requires CUDA")
    torch.cuda.set_device(device)
    task_to_context = task_context_ids(
        source_manifest, {str(sample["task_language"]) for sample in samples}
    )
    contexts = {}
    cache_root = args.cache_root.resolve()
    for task, context_id in task_to_context.items():
        payload = torch.load(
            cache_root / "contexts" / f"{context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("text_only") is not True:
            raise ValueError(f"context is not text-only: {context_id}")
        token_tags = payload.get("token_tags")
        if token_tags is None or torch.any(token_tags != 1):
            raise ValueError(f"context tags are not text-only: {context_id}")
        contexts[task] = {
            "id": context_id,
            "context": payload["context"].to(device=device, dtype=torch.bfloat16),
            "token_tags": token_tags.to(device),
        }

    from diffusers import AutoencoderKLMiniMaxH3

    backbone = H3Int8FeatureBackbone.from_checkpoint(h3_checkpoint).to(device).eval()
    backbone.requires_grad_(False)
    kv_provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=(49,), action_horizon=32, target_latent_frames=12,
            video_timestep=1.0, condition_video_timestep=1.0,
            capture_token_count=32,
            pool_strategy="adaptive_avg_pool1d_sequence_v1",
        ),
    )
    hidden_provider = H3Int8OnlineFeatureProvider(
        backbone,
        H3Int8OnlineFeatureContract(
            layers=(49,), action_horizon=32, target_latent_frames=12,
            video_timestep=1.0, condition_video_timestep=1.0,
            capture_compatibility="none",
        ),
    )
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(), subfolder="vae", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    vae.requires_grad_(False)

    compact_features = []
    hidden_features = []
    context_ids = []
    started = time.perf_counter()
    for index, sample in enumerate(samples):
        task = str(sample["task_language"])
        context = contexts[task]
        pixels = preprocess_libero_cameras(
            sample["agentview_image"].numpy(), sample["wristview_image"].numpy()
        )
        video = (
            pixels.mul(255.0).round().to(torch.uint8).permute(0, 3, 1, 2)
            .unsqueeze(2).to(device)
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            first_frame = encode_h3_vae_condition_standalone(
                vae, video, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            kv = kv_provider(first_frame, context["context"], context["token_tags"])
            hidden = hidden_provider(
                first_frame, context["context"], context["token_tags"]
            )[0]
        compact = compact_h3_kv_progress_feature(kv[49]).to(torch.bfloat16).cpu()
        hidden = pool_feature_tokens(hidden, 32).to(torch.bfloat16).cpu()
        if compact.shape != (512,) or hidden.shape != (1, 32, 5376):
            raise RuntimeError(
                f"unexpected C26 H3 feature shape: {compact.shape}/{hidden.shape}"
            )
        if not torch.isfinite(compact.float()).all() or not torch.isfinite(hidden.float()).all():
            raise FloatingPointError(f"non-finite H3 feature for group {index}")
        compact_features.append(compact)
        hidden_features.append(hidden)
        context_ids.append(context["id"])
        if (index + 1) % args.progress_every == 0 or index + 1 == len(samples):
            print(json.dumps({
                "completed": index + 1, "total": len(samples),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            }), flush=True)

    result = {
        "format": FORMAT_BY_DATASET[dataset["format"]],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "context_ids": context_ids,
        "d0_layer49_kv_compact": torch.stack(compact_features),
        "fact_layer49_hidden": torch.stack(hidden_features),
        "contracts": {
            "d0_kv": "layer49 concat(mean_k,std_k,mean_v,std_v) over 32x56 -> 512",
            "fact_hidden": "layer49 live hidden, exact adaptive pool to 32x5376",
            "h3_checkpoint": str(h3_checkpoint),
            "h3_checkpoint_sha256": h3_sha256,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_sha256,
            "vae_model": str(args.h3_model.resolve()),
            "video_timestep": 1.0,
            "condition_video_timestep": 1.0,
            "action_horizon": 32,
        },
        "duration_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    if dataset["format"] in CONSEQUENCE_DATASETS:
        result["sample_kinds"] = [sample["kind"] for sample in samples]
        result["sample_indices"] = torch.tensor(
            [int(sample["index"]) for sample in samples]
        )
    else:
        # Preserve the established C26/C27 output contract byte-for-byte at the
        # schema level; downstream critic loaders expect this field.
        result["group_ids"] = torch.tensor(
            [int(state["group_id"]) for state in dataset["states"]]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(result, temporary)
    os.replace(temporary, output)
    count_name = (
        "samples"
        if dataset["format"] in CONSEQUENCE_DATASETS
        else "groups"
    )
    print(json.dumps({
        "output": str(output), "dataset_sha256": result["dataset_sha256"],
        count_name: len(samples), "duration_seconds": round(result["duration_seconds"], 2),
    }, indent=2))


if __name__ == "__main__":
    main()
