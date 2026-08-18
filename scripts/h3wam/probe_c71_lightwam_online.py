#!/usr/bin/env python3
"""One-real-batch mechanical probe for the C71 Light-WAM/H3 architecture port.

The probe executes frozen INT8 H3 online, consumes only three selected H3 V
states, and backpropagates a masked direct-action regression loss through the
official byte-pinned Light-WAM state-fusion expert.  It deliberately performs
zero optimizer steps and writes no checkpoint, so its output cannot be used as
effectiveness evidence or as permission for a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Dataset,
    C58OnlineFrozenH3Provider,
    attach_online_h3_kv,
    collate_c58_online,
    move_c58_online_batch,
)
from fastwam.models.h3wam.lightwam_state_fusion import (  # noqa: E402
    LIGHTWAM_COMMIT,
    LIGHTWAM_H3_CARRIER_LAYERS,
    LIGHTWAM_STATE_FUSION_SHA256,
    H3LightWAMStateFusionPolicy,
)


FORMAT = "h3wam-c71-lightwam-online-one-batch-probe-v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def masked_direct_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over action scalars from non-padding timesteps."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,T,A] shape")
    if action_is_pad.shape != prediction.shape[:2]:
        raise ValueError("action_is_pad must be [B,T]")
    valid = (~action_is_pad.bool()).unsqueeze(-1).expand_as(prediction)
    if int(valid.sum()) <= 0:
        raise ValueError("masked action loss requires at least one valid scalar")
    return prediction.float().sub(target.float()).square().masked_select(valid).mean()


def parameter_grad_norm(module: torch.nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().float().square().sum().cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def cuda_memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        raise ValueError("C71 real online probe requires CUDA")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be non-negative")
    if len(args.expected_source_freeze_sha256) != 64:
        raise ValueError("expected source-freeze SHA256 must be 64 hex characters")
    resolved = {
        key: value.resolve()
        for key, value in {
            "manifest": args.manifest,
            "source_manifest": args.source_manifest,
            "cache_root": args.cache_root,
            "h3_checkpoint": args.h3_checkpoint,
            "source_freeze": args.source_freeze,
            "output": args.output,
        }.items()
    }
    for key in ("manifest", "source_manifest", "h3_checkpoint", "source_freeze"):
        if not resolved[key].is_file():
            raise FileNotFoundError(resolved[key])
    if not resolved["cache_root"].is_dir():
        raise FileNotFoundError(resolved["cache_root"])
    if resolved["output"].exists():
        raise FileExistsError(resolved["output"])

    identity_started = time.perf_counter()
    h3_sha = sha256_file(resolved["h3_checkpoint"])
    freeze_sha = sha256_file(resolved["source_freeze"])
    if h3_sha != EXPECTED_H3_SHA256:
        raise ValueError(f"C71 H3 checkpoint SHA mismatch: {h3_sha}")
    if freeze_sha != args.expected_source_freeze_sha256:
        raise ValueError(f"C71 source freeze SHA mismatch: {freeze_sha}")
    identity_seconds = time.perf_counter() - identity_started

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    dataset = C58OnlineFrozenH3Dataset(
        resolved["manifest"],
        resolved["source_manifest"],
        resolved["cache_root"],
        resolved["h3_checkpoint"],
        action_horizon=32,
        sample_offset=args.sample_offset,
        limit=1,
    )
    sample = dataset[0]
    batch = collate_c58_online([sample])
    batch = move_c58_online_batch(batch, device, torch.bfloat16)

    h3_load_started = time.perf_counter()
    provider = C58OnlineFrozenH3Provider(
        resolved["h3_checkpoint"], layers=LIGHTWAM_H3_CARRIER_LAYERS
    ).to(device=device).eval()
    provider.requires_grad_(False)
    torch.cuda.synchronize(device)
    h3_load_seconds = time.perf_counter() - h3_load_started
    memory_after_h3_load = cuda_memory(device)

    model_started = time.perf_counter()
    policy = H3LightWAMStateFusionPolicy(enabled=True).to(
        device=device, dtype=torch.bfloat16
    )
    policy.train()
    torch.cuda.synchronize(device)
    model_init_seconds = time.perf_counter() - model_started
    memory_after_models = cuda_memory(device)

    torch.cuda.reset_peak_memory_stats(device)
    extract_started = time.perf_counter()
    batch = attach_online_h3_kv(batch, provider)
    torch.cuda.synchronize(device)
    extract_seconds = time.perf_counter() - extract_started
    cache = batch["video_kv_cache"]
    if set(cache) != set(LIGHTWAM_H3_CARRIER_LAYERS):
        raise RuntimeError("online provider escaped the three-layer C71 contract")
    cache_tensors = [tensor for item in cache.values() for tensor in item.values()]
    if any(tensor.requires_grad or not torch.isfinite(tensor).all() for tensor in cache_tensors):
        raise RuntimeError("C71 frozen H3 emitted grad-bearing or non-finite states")

    policy.zero_grad(set_to_none=True)
    forward_started = time.perf_counter()
    prediction = policy(
        torch.zeros_like(batch["actions"]),
        torch.zeros((1,), device=device, dtype=torch.bfloat16),
        text_context=batch["text_context"],
        text_mask=batch["text_mask"],
        proprio=batch["proprio"],
        video_kv_cache=cache,
    )
    loss = masked_direct_action_mse(
        prediction, batch["actions"], batch["action_is_pad"]
    )
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise RuntimeError("C71 produced non-finite prediction or loss")
    loss.backward()
    torch.cuda.synchronize(device)
    forward_backward_seconds = time.perf_counter() - forward_started

    expert = policy.state_fusion_action_expert
    if expert is None or policy.proprio_encoder is None:
        raise RuntimeError("C71 enabled policy failed to construct trainable modules")
    gradient_norms = {
        "query_poolers": parameter_grad_norm(expert.layer_poolers),
        "layer_compressors": parameter_grad_norm(expert.layer_compressors),
        "fusion_trunk": parameter_grad_norm(expert.fused_proj)
        + parameter_grad_norm(expert.trunk),
        "step_position": parameter_grad_norm(expert.step_pos_proj),
        "proprio_encoder": parameter_grad_norm(policy.proprio_encoder),
        "output_head": parameter_grad_norm(expert.output),
    }
    gates = {
        "real_registered_dense_sample": sample.get("format")
        == "h3wam-c56b-online-fact-sample-v1",
        "frozen_online_int8_h3": not any(
            parameter.requires_grad for parameter in provider.parameters()
        )
        and not any(parameter.grad is not None for parameter in provider.parameters()),
        "exact_three_h3_layers": tuple(sorted(cache)) == LIGHTWAM_H3_CARRIER_LAYERS,
        "finite_nonconstant_prediction": bool(torch.isfinite(prediction).all())
        and float(prediction.detach().float().std().cpu()) > 0.0,
        "finite_positive_loss": math.isfinite(float(loss.detach().cpu()))
        and float(loss.detach().cpu()) > 0.0,
        "all_declared_gradient_paths_positive": all(
            math.isfinite(value) and value > 0.0 for value in gradient_norms.values()
        ),
        "no_future_input": not any(key.startswith("future") for key in batch),
        "no_optimizer_step": True,
        "no_checkpoint_written": True,
    }
    status = "PASS_C71_LIGHTWAM_ONE_BATCH_PROBE" if all(gates.values()) else "FAIL_C71_LIGHTWAM_ONE_BATCH_PROBE"
    trainable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    total_seconds = time.perf_counter() - started
    report = {
        "format": FORMAT,
        "status": status,
        "permission": "PROBE_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "hypothesis": (
            "Three frozen H3 value-state taps at layers 14/27/41 can drive the "
            "byte-pinned Light-WAM direct-action expert with finite nonconstant "
            "outputs and positive gradients on one real dense LIBERO window."
        ),
        "parent": "C58 frozen online INT8 H3 execution boundary",
        "unique_variable": "official Light-WAM three-state direct-action fusion head",
        "identity": {
            "sample_id": str(sample["sample_id"]),
            "sample_offset": args.sample_offset,
            "manifest": str(resolved["manifest"]),
            "manifest_sha256": dataset.manifest_sha256,
            "manifest_items": dataset.manifest_items,
            "selected_train_rows": len(dataset.rows),
            "source_manifest": str(resolved["source_manifest"]),
            "source_manifest_sha256": dataset.source_manifest_sha256,
            "source_manifest_items": dataset.source_manifest_items,
            "stats_sha256": dataset.stats_sha256,
            "h3_checkpoint": str(resolved["h3_checkpoint"]),
            "h3_checkpoint_sha256": h3_sha,
            "source_freeze": str(resolved["source_freeze"]),
            "source_freeze_sha256": freeze_sha,
            "lightwam_commit": LIGHTWAM_COMMIT,
            "lightwam_state_fusion_sha256": LIGHTWAM_STATE_FUSION_SHA256,
        },
        "contract": {
            "h3_quantization": "int8_tensorwise_convrot",
            "h3_trainable": False,
            "h3_layers": list(LIGHTWAM_H3_CARRIER_LAYERS),
            "h3_feature": "value_state",
            "h3_capture_tokens": 32,
            "action_horizon": 32,
            "action_dim": 7,
            "proprio_dim": 8,
            "loss": "masked_direct_normalized_action_mse",
            "future_target_consumed": False,
            "optimizer_steps": 0,
            "training_samples": 0,
            "effective_epochs": 0.0,
            "data_forward_samples": 1,
            "checkpoint": None,
            "trainable_parameters": trainable_parameters,
        },
        "measurement": {
            "loss": float(loss.detach().cpu()),
            "prediction_mean": float(prediction.detach().float().mean().cpu()),
            "prediction_std": float(prediction.detach().float().std().cpu()),
            "valid_action_timesteps": int((~batch["action_is_pad"]).sum().cpu()),
            "gradient_norms": gradient_norms,
            "gates": gates,
            "timing_seconds": {
                "identity_hashing": identity_seconds,
                "h3_checkpoint_load": h3_load_seconds,
                "lightwam_head_init": model_init_seconds,
                "online_h3_extract": extract_seconds,
                "head_forward_backward": forward_backward_seconds,
                "probe_after_identity": total_seconds,
            },
            "memory": {
                "after_h3_load": memory_after_h3_load,
                "after_h3_and_head": memory_after_models,
                "measured_extract_forward_backward_peak": cuda_memory(device),
            },
        },
        "excluded_sources": [
            "Light-WAM Wan backbone LoRA/adapted/delta states",
            "Light-WAM proprio-as-Wan-text-token injection",
            "future-frame/future-state/value supervision",
            "cached H3 K/V training input",
        ],
        "claim_boundary": (
            "Mechanical A800 feasibility only. No optimizer step, retained model, "
            "offline improvement, checkpoint selection, or closed-loop effect is claimed."
        ),
    }
    atomic_json(resolved["output"], report)
    print(json.dumps(report, indent=2), flush=True)
    if status != "PASS_C71_LIGHTWAM_ONE_BATCH_PROBE":
        raise SystemExit(64)


if __name__ == "__main__":
    main()
