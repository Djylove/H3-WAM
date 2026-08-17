#!/usr/bin/env python3
"""Read-only sharded C63 probe of C60's integrated FACT Stage-2 value head."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRS = load_script("_c63_pairs", "prepare_c63_fact_stage2_pairs.py")
PAIRED = load_script("_c63_c56b_paired", "evaluate_c56b_fact_online_paired.py")
TRAIN = PAIRED.TRAIN

FORMAT = "h3wam-c63-fact-stage2-within-state-shard-v1"
C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
INFERENCE_STEPS = 10
FLOW_SHIFT = 5.0
ORDER_ATOL = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_actions(
    actions: np.ndarray, action_min: torch.Tensor, action_max: torch.Tensor
) -> torch.Tensor:
    environment = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    dataset = TRAIN.OnlineH3FACTRolloutDataset.__module__  # import-path audit only
    del dataset
    from fastwam.models.h3wam.fact_online_data import environment_actions_to_dataset
    from fastwam.models.h3wam.deployment import minmax_normalize

    return minmax_normalize(
        environment_actions_to_dataset(environment), action_min, action_max
    )


def _batch_cache(cache: dict[int, dict[str, torch.Tensor]], batch: int):
    return {
        layer: {name: value.repeat(batch, *([1] * (value.ndim - 1))) for name, value in item.items()}
        for layer, item in cache.items()
    }


@torch.inference_mode()
def score_order(
    model,
    scheduler,
    clean_actions: torch.Tensor,
    *,
    base_state_noise: torch.Tensor,
    base_value_noise: torch.Tensor,
    base_rep_noise: torch.Tensor,
    text_context: torch.Tensor,
    proprio: torch.Tensor,
    video_kv_cache: dict[int, dict[str, torch.Tensor]],
    text_mask: torch.Tensor,
) -> torch.Tensor:
    batch = clean_actions.shape[0]
    state = base_state_noise.repeat(batch, 1, 1)
    value = base_value_noise.repeat(batch, 1, 1)
    representation = base_rep_noise.repeat(batch, 1, 1)
    context = text_context.repeat(batch, 1, 1)
    robot_state = proprio.repeat(batch, 1)
    mask = text_mask.repeat(batch, 1)
    cache = _batch_cache(video_kv_cache, batch)
    timesteps, deltas = scheduler.build_inference_schedule(
        INFERENCE_STEPS, clean_actions.device, clean_actions.dtype
    )
    for timestep, delta in zip(timesteps, deltas, strict=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model.forward_consequence(
                clean_actions=clean_actions,
                timestep=timestep.expand(batch),
                noisy_future_state=state,
                noisy_value=value,
                noisy_future_representation=representation,
                text_context=context,
                proprio=robot_state,
                video_kv_cache=cache,
                text_mask=mask,
            )
        PAIRS.assert_stage2_track_shapes(
            prediction["future_state"], prediction["value"],
            prediction["future_representation"], batch=batch,
            future_dim=TRAIN.FUTURE_DIM,
        )
        state = scheduler.step(prediction["future_state"], delta, state)
        value = scheduler.step(prediction["value"], delta, value)
        representation = scheduler.step(
            prediction["future_representation"], delta, representation
        )
        PAIRS.assert_stage2_track_shapes(
            state, value, representation, batch=batch,
            future_dim=TRAIN.FUTURE_DIM,
        )
    return PAIRS.assert_stage2_track_shapes(
        state, value, representation, batch=batch,
        future_dim=TRAIN.FUTURE_DIM,
    ).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--c60-dataset", type=Path, required=True)
    parser.add_argument("--c60-observations", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-model", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.shard < args.num_shards:
        raise ValueError("C63 requires CUDA and a valid shard")
    if args.num_shards != 8:
        raise ValueError("C63 fixed execution contract requires eight shards")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C63 shard: {output}")
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    pair_path = args.pairs.resolve()
    pair_manifest_sha = sha256_file(pair_path)
    manifest = json.loads(pair_path.read_text())
    if (
        manifest.get("format") != PAIRS.FORMAT
        or manifest.get("status") != "PASS_C63_FIXED_32_PAIR_MECHANICAL_PREPARATION"
        or manifest.get("effect_status") != "NOT_EVALUATED"
        or manifest.get("pair_count") != 32
        or manifest.get("suite_counts") != PAIRS.EXPECTED_SUITE_COUNTS
        or not all(manifest.get("identity_gate", {}).values())
    ):
        raise ValueError("C63 pair manifest gate failed")
    selected = [row for row in manifest["pairs"] if int(row["pair_index"]) % 8 == args.shard]
    if len(selected) != 4:
        raise ValueError("C63 shard must own exactly four fixed pairs")

    ready, checkpoint = PAIRED._load_ready(args.ready.resolve(), "C60_MAIN")
    checkpoint_path = Path(ready["checkpoint"]).resolve()
    if sha256_file(checkpoint_path) != C60_SHA256:
        raise ValueError("C63 C60 checkpoint identity mismatch")
    model = PAIRED.restore_model(checkpoint["model"], device=device, dtype=dtype).model
    model.requires_grad_(False).eval()
    del checkpoint

    dataset = TRAIN.OnlineH3FACTRolloutDataset(
        args.c60_dataset.resolve(), args.c60_observations.resolve(),
        args.source_manifest.resolve(), args.cache_root.resolve(), split="validation",
        expected_dataset_sha256=PAIRS.DATASET_SHA256,
        expected_observations_sha256=PAIRS.OBSERVATIONS_SHA256,
    )
    sample_to_index = {int(row["sample_id"]): index for index, row in enumerate(dataset.rows)}

    from diffusers import AutoencoderKLMiniMaxH3
    from fastwam.models.h3wam import H3Int8FeatureBackbone, H3Int8OnlineKVContract, H3Int8OnlineKVProvider
    from fastwam.models.h3wam.fastwam_full_tower import LAYERWISE_H3_50_TO_ACTION_30
    from fastwam.models.h3wam.int8_online import SEQUENCE_KV_POOL

    if sha256_file(args.h3_checkpoint.resolve()) != TRAIN.EXPECTED_H3_SHA256:
        raise ValueError("C63 H3 checkpoint identity mismatch")
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(), subfolder="vae", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).requires_grad_(False).eval()
    backbone = H3Int8FeatureBackbone.from_checkpoint(args.h3_checkpoint.resolve()).to(device)
    backbone.requires_grad_(False).eval()
    provider = H3Int8OnlineKVProvider(
        backbone,
        H3Int8OnlineKVContract(
            layers=LAYERWISE_H3_50_TO_ACTION_30, action_horizon=32,
            target_latent_frames=12, video_timestep=1.0,
            condition_video_timestep=1.0, capture_token_count=32,
            pool_strategy=SEQUENCE_KV_POOL,
        ),
    ).eval()
    scheduler = TRAIN.C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=FLOW_SHIFT)

    rows = []
    for pair in selected:
        for path_key, sha_key in (
            ("branch_trajectory", "branch_trajectory_sha256"),
            ("parent_trajectory", "parent_trajectory_sha256"),
        ):
            if sha256_file(Path(pair[path_key])) != pair[sha_key]:
                raise ValueError(f"C63 trajectory changed: {path_key}")
        item = dataset[sample_to_index[int(pair["sample_id"])]]
        if int(item["episode_id"]) != int(pair["episode_id"]):
            raise ValueError("C63 dataset item identity mismatch")
        failed_env = np.asarray(pair["failed_actions"], dtype=np.float32)
        success_env = np.asarray(pair["success_actions"], dtype=np.float32)
        failed_pad = np.asarray(pair["failed_is_pad"], dtype=bool)
        success_pad = np.asarray(pair["success_is_pad"], dtype=bool)
        if (
            PAIRS.tensor_identity(failed_env, failed_pad) != pair["failed_action_sha256"]
            or PAIRS.tensor_identity(success_env, success_pad) != pair["success_action_sha256"]
        ):
            raise ValueError("C63 action bytes did not round-trip through JSON")
        failed = _normalize_actions(failed_env, dataset.action_min, dataset.action_max)
        success = _normalize_actions(success_env, dataset.action_min, dataset.action_max)
        if not torch.equal(failed, item["actions"]):
            raise ValueError("C63 failed action differs from the C60 loader")
        clean = torch.stack((success, failed)).to(device=device, dtype=dtype)
        context = item["text_context"].unsqueeze(0).to(device=device, dtype=dtype)
        tags = item["text_token_tags"].to(device=device, dtype=torch.long)
        text_mask = torch.ones((1, context.shape[1]), device=device, dtype=torch.bool)
        proprio = item["proprio"].unsqueeze(0).to(device=device, dtype=dtype)
        current_latents = TRAIN.encode(item, "current_h3_input", vae, device)
        # The frozen provider owns its inference-mode boundary.  Clone only
        # after it returns, matching C56b training, so the consequence tower
        # receives ordinary tensors rather than inference tensors.
        inference_kv = provider(current_latents, context.float(), tags)
        current_kv = TRAIN.C58_ONLINE.materialize_kv_for_autograd_consumer(
            inference_kv
        )
        del inference_kv
        del current_latents
        generator = torch.Generator(device=device).manual_seed(
            args.seed + int(pair["pair_index"]) * 1_000_003
        )
        # Official FACT prepares future state/value as [B,1,D]/[B,1,1].
        # Keeping the token axis prevents accidental [B,B,1] broadcasting.
        base_state = torch.randn((1, 1, 8), generator=generator, device=device, dtype=dtype)
        base_value = torch.randn((1, 1, 1), generator=generator, device=device, dtype=dtype)
        base_rep = torch.randn(
            (1, 1, TRAIN.FUTURE_DIM), generator=generator, device=device, dtype=dtype
        )
        normal = score_order(
            model, scheduler, clean,
            base_state_noise=base_state, base_value_noise=base_value,
            base_rep_noise=base_rep, text_context=context, proprio=proprio,
            video_kv_cache=current_kv, text_mask=text_mask,
        )
        swapped = score_order(
            model, scheduler, clean.flip(0),
            base_state_noise=base_state, base_value_noise=base_value,
            base_rep_noise=base_rep, text_context=context, proprio=proprio,
            video_kv_cache=current_kv, text_mask=text_mask,
        ).flip(0)
        order_max_abs = float((normal - swapped).abs().max())
        success_score, failure_score = map(float, normal.tolist())
        rows.append(
            {
                "pair_index": int(pair["pair_index"]), "episode_id": int(pair["episode_id"]),
                "sample_id": int(pair["sample_id"]), "suite": pair["suite"],
                "task": int(pair["task"]), "trial": int(pair["trial"]),
                "success_score": success_score, "failure_score": failure_score,
                "failure_minus_success": failure_score - success_score,
                "success_preferred": bool(success_score < failure_score),
                "score_finite": bool(math.isfinite(success_score) and math.isfinite(failure_score)),
                "action_conditioned_value_delta_nonzero": bool(success_score != failure_score),
                "order_invariance_max_abs": order_max_abs,
                "order_invariance_pass": bool(order_max_abs <= ORDER_ATOL),
            }
        )
        del current_kv
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    result = {
        "format": FORMAT,
        "status": "PASS_C63_STAGE2_SHARD_MECHANICS" if all(
            row["score_finite"] and row["action_conditioned_value_delta_nonzero"]
            and row["order_invariance_pass"] for row in rows
        ) else "FAIL_C63_STAGE2_SHARD_MECHANICS",
        "effect_status": "SHARD_ONLY_NOT_INTERPRETABLE",
        "shard": args.shard, "num_shards": args.num_shards,
        "pair_manifest_sha256": pair_manifest_sha,
        "checkpoint_sha256": C60_SHA256,
        "dataset_sha256": PAIRS.DATASET_SHA256,
        "observations_sha256": PAIRS.OBSERVATIONS_SHA256,
        "solver": {"inference_steps": INFERENCE_STEPS, "flow_shift": FLOW_SHIFT},
        "same_noise_within_pair": True,
        "candidate_order_repeated": True,
        "rows": sorted(rows, key=lambda row: row["pair_index"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
