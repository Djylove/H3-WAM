#!/usr/bin/env python3
"""Read-only sharded scoring of the frozen C65 cross-suite action pairs."""

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


C63 = load_script("_c65_c63_stage2", "evaluate_c63_fact_stage2_within_state.py")
C65_DATA = load_script("_c65_data_finalizer", "finalize_c65_c60_pair_collection.py")
PAIRED = C63.PAIRED
TRAIN = C63.TRAIN

FORMAT = "h3wam-c65-fact-stage2-cross-suite-shard-v1"
DATA_GATE_FORMAT = C65_DATA.GATE_FORMAT
PAIR_FORMAT = C65_DATA.PAIR_FORMAT
C60_SHA256 = C65_DATA.C60_SHA256
SUITES = tuple(C65_DATA.SUITES)
INFERENCE_STEPS = 10
FLOW_SHIFT = 5.0
ORDER_ATOL = 1e-5
VALUE_CONTRACT = dict(C63.VALUE_CONTRACT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_pair_inputs(data_gate_path: Path, pair_path: Path) -> tuple[dict[str, Any], str]:
    gate = json.loads(data_gate_path.read_text())
    pair_sha = sha256_file(pair_path)
    pairs = json.loads(pair_path.read_text())
    if (
        gate.get("format") != DATA_GATE_FORMAT
        or gate.get("status") != "PASS_C65_FOUR_SUITE_PAIR_DATA_GATE"
        or gate.get("permission") != "GO_SCORE_C65"
        or gate.get("effect_status") != "NOT_EVALUATED"
        or gate.get("pair_count") != 80
        or gate.get("pairs_sha256") != pair_sha
        or Path(gate.get("pairs_path", "")).resolve() != pair_path
        or not gate.get("all_results_audited")
        or gate.get("model_forwards") != 0
        or gate.get("training_steps") != 0
    ):
        raise ValueError("C65 data gate does not authorize scoring")
    if (
        pairs.get("format") != PAIR_FORMAT
        or pairs.get("status") != "PASS_C65_FIXED_80_PAIR_MECHANICAL_PREPARATION"
        or pairs.get("effect_status") != "NOT_EVALUATED"
        or pairs.get("pair_count") != 80
        or pairs.get("suite_counts") != {suite: 20 for suite in SUITES}
        or pairs.get("checkpoint_sha256") != C60_SHA256
        or pairs.get("prepared_sha256") != C65_DATA.PREPARED_SHA256
        or pairs.get("jobs_sha256") != C65_DATA.JOBS_SHA256
        or pairs.get("source_independent") is not True
        or pairs.get("same_state_same_execution_contract") is not True
        or [int(row["pair_index"]) for row in pairs.get("pairs", [])] != list(range(80))
    ):
        raise ValueError("C65 pair manifest contract failed")
    return pairs, pair_sha


def load_observation(
    pair: dict[str, Any],
    *,
    state_min: torch.Tensor,
    state_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from fastwam.models.h3wam.deployment import (
        libero_observation_state,
        minmax_normalize,
    )
    from fastwam.models.h3wam.fact_online_data import OnlineH3FACTRolloutDataset

    observation = dict(pair["observation"])
    trajectory = Path(observation["trajectory"]).resolve()
    if (
        sha256_file(trajectory) != observation["trajectory_sha256"]
        or int(observation["row_index"]) != 0
        or int(observation["step"]) != int(pair["start_step"])
    ):
        raise ValueError("C65 current observation identity changed")
    with np.load(trajectory, allow_pickle=False) as archive:
        if (
            int(archive["step"][0]) != int(pair["start_step"])
            or sha256_array(archive["sim_state"][0]) != pair["sim_state_sha256"]
        ):
            raise ValueError("C65 current trajectory state changed")
        state = libero_observation_state(
            {
                "eef_pos": archive["eef_pos"][0],
                "eef_quat": archive["eef_quat"][0],
                "gripper_qpos": archive["gripper_qpos"][0],
            }
        )
    observation["kind"] = "row"
    pixels = OnlineH3FACTRolloutDataset._pixels(observation)
    return pixels, minmax_normalize(state, state_min, state_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--data-gate", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
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
        raise ValueError("C65 scoring requires CUDA and a valid shard")
    if args.num_shards != 8:
        raise ValueError("C65 fixed execution contract requires eight shards")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C65 shard: {output}")
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    pair_path = args.pairs.resolve()
    pairs, pair_sha = validate_pair_inputs(args.data_gate.resolve(), pair_path)
    selected = [
        row for row in pairs["pairs"]
        if int(row["pair_index"]) % args.num_shards == args.shard
    ]
    if len(selected) != 10:
        raise ValueError("C65 shard must own exactly ten fixed pairs")

    ready, checkpoint = PAIRED._load_ready(args.ready.resolve(), "C60_MAIN")
    checkpoint_path = Path(ready["checkpoint"]).resolve()
    if sha256_file(checkpoint_path) != C60_SHA256:
        raise ValueError("C65 C60 checkpoint identity mismatch")
    model = PAIRED.restore_model(checkpoint["model"], device=device, dtype=dtype).model
    model.requires_grad_(False).eval()
    del checkpoint

    from diffusers import AutoencoderKLMiniMaxH3
    from fastwam.models.h3wam import (
        H3Int8FeatureBackbone,
        H3Int8OnlineKVContract,
        H3Int8OnlineKVProvider,
    )
    from fastwam.models.h3wam.fastwam_full_tower import LAYERWISE_H3_50_TO_ACTION_30
    from fastwam.models.h3wam.fact_online_data import (
        _load_text_context,
        _task_context_ids,
    )
    from fastwam.models.h3wam.int8_online import SEQUENCE_KV_POOL

    if sha256_file(args.h3_checkpoint.resolve()) != TRAIN.EXPECTED_H3_SHA256:
        raise ValueError("C65 H3 checkpoint identity mismatch")
    source_manifest = args.source_manifest.resolve()
    cache_root = args.cache_root.resolve()
    tasks = {str(row["task_language"]) for row in selected}
    context_by_task = _task_context_ids(source_manifest, tasks)
    stats = torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)
    action_min, action_max = stats["action_min"].float(), stats["action_max"].float()
    state_min, state_max = stats["state_min"].float(), stats["state_max"].float()

    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.h3_model.resolve(), subfolder="vae", torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).requires_grad_(False).eval()
    backbone = H3Int8FeatureBackbone.from_checkpoint(args.h3_checkpoint.resolve()).to(device)
    backbone.requires_grad_(False).eval()
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
    scheduler = TRAIN.C58.PARENT.FlowMatchScheduler(
        num_train_timesteps=1000, shift=FLOW_SHIFT
    )

    rows = []
    for pair in selected:
        success_env = np.asarray(pair["success_actions"], dtype=np.float32)
        failure_env = np.asarray(pair["failure_actions"], dtype=np.float32)
        if (
            C65_DATA.action_identity(success_env) != pair["success_action_sha256"]
            or C65_DATA.action_identity(failure_env) != pair["failure_action_sha256"]
            or np.array_equal(success_env, failure_env)
        ):
            raise ValueError("C65 candidate action identity changed")
        success = C63._normalize_actions(success_env, action_min, action_max)
        failure = C63._normalize_actions(failure_env, action_min, action_max)
        clean = torch.stack((success, failure)).to(device=device, dtype=dtype)

        pixels, proprio = load_observation(
            pair, state_min=state_min, state_max=state_max
        )
        context_payload = _load_text_context(
            cache_root, context_by_task[str(pair["task_language"])]
        )
        context = context_payload["context"].unsqueeze(0).to(device=device, dtype=dtype)
        tags = context_payload["token_tags"].to(device=device, dtype=torch.long)
        text_mask = torch.ones((1, context.shape[1]), device=device, dtype=torch.bool)
        current_latents = TRAIN.encode(
            {"input_mode": "pixels", "current_h3_input": pixels},
            "current_h3_input", vae, device,
        )
        inference_kv = provider(current_latents, context.float(), tags)
        current_kv = TRAIN.C58_ONLINE.materialize_kv_for_autograd_consumer(inference_kv)
        del inference_kv, current_latents

        generator = torch.Generator(device=device).manual_seed(
            args.seed + int(pair["pair_index"]) * 1_000_003
        )
        base_state = torch.randn((1, 1, 8), generator=generator, device=device, dtype=dtype)
        base_value = torch.randn((1, 1, 1), generator=generator, device=device, dtype=dtype)
        base_rep = torch.randn(
            (1, 1, TRAIN.FUTURE_DIM), generator=generator, device=device, dtype=dtype
        )
        normal = C63.score_order(
            model, scheduler, clean,
            base_state_noise=base_state,
            base_value_noise=base_value,
            base_rep_noise=base_rep,
            text_context=context,
            proprio=proprio.unsqueeze(0).to(device=device, dtype=dtype),
            video_kv_cache=current_kv,
            text_mask=text_mask,
        )
        swapped = C63.score_order(
            model, scheduler, clean.flip(0),
            base_state_noise=base_state,
            base_value_noise=base_value,
            base_rep_noise=base_rep,
            text_context=context,
            proprio=proprio.unsqueeze(0).to(device=device, dtype=dtype),
            video_kv_cache=current_kv,
            text_mask=text_mask,
        ).flip(0)
        order_max_abs = float((normal - swapped).abs().max())
        success_normalized, failure_normalized = map(float, normal.tolist())
        success_score, failure_score = success_normalized + 1.0, failure_normalized + 1.0
        finite = math.isfinite(success_score) and math.isfinite(failure_score)
        rows.append(
            {
                "pair_index": int(pair["pair_index"]),
                "source_id": int(pair["source_id"]),
                "group_id": int(pair["group_id"]),
                "suite": str(pair["suite"]),
                "task": int(pair["task"]),
                "trial": int(pair["trial"]),
                "start_step": int(pair["start_step"]),
                "sim_state_sha256": pair["sim_state_sha256"],
                "observation_trajectory_sha256": pair["observation"]["trajectory_sha256"],
                "success_action_sha256": pair["success_action_sha256"],
                "failure_action_sha256": pair["failure_action_sha256"],
                "success_score": success_score,
                "failure_score": failure_score,
                "success_score_normalized": success_normalized,
                "failure_score_normalized": failure_normalized,
                "failure_minus_success": failure_score - success_score,
                "success_preferred": bool(success_score < failure_score),
                "tie": bool(success_score == failure_score),
                "score_finite": bool(finite),
                "order_invariance_max_abs": order_max_abs,
                "order_invariance_pass": bool(order_max_abs <= ORDER_ATOL),
                "identity_pass": True,
            }
        )
        del current_kv
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    mechanics = all(
        row["score_finite"] and row["order_invariance_pass"] and row["identity_pass"]
        for row in rows
    )
    result = {
        "format": FORMAT,
        "status": (
            "PASS_C65_STAGE2_SHARD_MECHANICS"
            if mechanics else "FAIL_C65_STAGE2_SHARD_MECHANICS"
        ),
        "effect_status": "SHARD_ONLY_NOT_INTERPRETABLE",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "data_gate_sha256": sha256_file(args.data_gate.resolve()),
        "pair_manifest_sha256": pair_sha,
        "checkpoint_sha256": C60_SHA256,
        "solver": {"inference_steps": INFERENCE_STEPS, "flow_shift": FLOW_SHIFT},
        "value_contract": VALUE_CONTRACT,
        "same_noise_within_pair": True,
        "candidate_order_repeated": True,
        "model_input_fields": [
            "current_agentview", "current_wristview", "current_proprio",
            "task_language", "candidate_action",
        ],
        "forbidden_model_input_fields": [
            "future_observation", "terminal_state", "outcome", "success_label",
        ],
        "rows": sorted(rows, key=lambda row: row["pair_index"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
