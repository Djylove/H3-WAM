#!/usr/bin/env python3
"""Matched balanced-80 evaluation for C56b/C56b+C61 online-H3 arms.

The program permits exactly one checkpoint-contract difference: the causal
failure dataset/observation identity.  H3 K/V is materialized in GPU memory
from canonical first-frame latents for every sample and is never read from or
written to a disk cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam.fact_layerwise_tower import (  # noqa: E402
    H3FACTLayerwiseTowerPolicy,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
)


def load_script(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ONLINE = load_script(
    "_c56b_online_eval_protocol", "evaluate_h3_fastwam_full_tower_online.py"
)
TRAIN = load_script("_c56b_online_train", "train_c56b_fact_online.py")
PROTOCOL = ONLINE.PROTOCOL

FORMAT = "h3wam-c56b-fact-online-paired-balanced80-v1"
SELECTED_IDS_SHA256 = (
    "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
)
EXPECTED_ITEMS = 80
CAUSAL_KEYS = frozenset(
    {"causal_failure_dataset_sha256", "causal_failure_observations_sha256"}
)
EXPECTED_ARMS = {"C60_MAIN", "C61_MATCHED"}


@dataclass(frozen=True)
class EvalConfig:
    main_ready: Path
    c61_ready: Path
    h3_checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path
    device: str = "cuda:0"
    seed: int = 42
    inference_steps: int = 10


class ActionOnly(nn.Module):
    def __init__(self, model: H3FACTLayerwiseTowerPolicy) -> None:
        super().__init__()
        self.model = model

    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.model.forward_action(noisy_actions, timestep, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ready(path: Path, expected_arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ready = json.loads(path.read_text(encoding="utf-8"))
    if (
        ready.get("format") != "h3wam-c56b-fact-online-long10000-ready-v1"
        or ready.get("status") != "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE"
        or ready.get("permission") != "READY_FOR_PAIRED_HELDOUT"
        or ready.get("effect_status") != "NOT_EVIDENCE_READY"
        or ready.get("arm") != expected_arm
        or ready.get("completed_steps") != 10_000
        or ready.get("world_size") != 8
        or ready.get("global_batch") != 8
        or not isinstance(ready.get("gate"), dict)
        or not ready["gate"]
        or not all(ready["gate"].values())
    ):
        raise ValueError(f"{expected_arm} endpoint READY gate failed")
    checkpoint = Path(ready.get("checkpoint", "")).resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(ready.get("checkpoint_size_bytes", -1))
        or sha256_file(checkpoint) != ready.get("checkpoint_sha256")
    ):
        raise ValueError(f"{expected_arm} checkpoint bytes differ from READY")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {
        "schema_version", "completed_steps", "model", "optimizer",
        "lr_scheduler", "contract", "probe_step", "probe_predictions",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("completed_steps") != 10_000
        or payload.get("probe_step") != 10_000
        or not isinstance(payload.get("probe_predictions"), list)
        or len(payload["probe_predictions"]) != 8
    ):
        raise ValueError(f"{expected_arm} checkpoint schema/step gate failed")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{expected_arm} checkpoint contract missing")
    required_contract = {
        "format": TRAIN.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": list(TRAIN.RANK_CATEGORIES),
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
        "target_norm_sha256": (
            "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000"
        ),
        "h3_sha256": TRAIN.EXPECTED_H3_SHA256,
        "d0_sha256": TRAIN.EXPECTED_D0_SHA256,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 10_000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
    }
    mismatch = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in required_contract.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"{expected_arm} fixed training contract mismatch: {mismatch}")
    initialization = contract.get("initialization", {})
    if (
        initialization.get("initialization_contract")
        != "strict_online_c58b_parent_v1"
        or initialization.get("c58_completed_steps") != 10_000
        or not isinstance(contract.get("c58_parent_sha256"), str)
        or len(contract["c58_parent_sha256"]) != 64
    ):
        raise ValueError(f"{expected_arm} fixed C58 parent initialization mismatch")
    for key in (
        "demo_manifest_sha256", "source_manifest_sha256", "demo_stats_sha256",
        "c48_dataset_sha256", "c48_observations_sha256",
        "c59_completed_sha256", "c59_sample_labels_sha256",
    ):
        value = contract.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{expected_arm} lacks immutable shared-data identity: {key}")
    if (
        contract.get("c58_parent_sha256") != ready.get("c58_parent_sha256")
        or contract.get("causal_failure_dataset_sha256")
        != ready.get("causal_failure_dataset_sha256")
        or contract.get("causal_failure_observations_sha256")
        != ready.get("causal_failure_observations_sha256")
    ):
        raise ValueError(f"{expected_arm} READY/checkpoint contract mismatch")
    return ready, payload


def assert_only_causal_failure_diff(
    main_contract: dict[str, Any], c61_contract: dict[str, Any]
) -> dict[str, Any]:
    if set(main_contract) != set(c61_contract):
        raise ValueError("paired checkpoint contract key sets differ")
    actual_differences = {
        key for key in main_contract if main_contract[key] != c61_contract[key]
    }
    if actual_differences != CAUSAL_KEYS:
        raise ValueError(
            "paired arms must differ only in causal failure identities; "
            f"actual={sorted(actual_differences)}"
        )
    for name, contract in (("main", main_contract), ("c61", c61_contract)):
        for key in CAUSAL_KEYS:
            value = contract.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} has invalid {key}")
    return {
        "status": "PASS_ONLY_CAUSAL_FAILURE_POOL_DIFFERS",
        "different_keys": sorted(CAUSAL_KEYS),
        "identical_keys": sorted(set(main_contract) - CAUSAL_KEYS),
        "main": {key: main_contract[key] for key in sorted(CAUSAL_KEYS)},
        "c61": {key: c61_contract[key] for key in sorted(CAUSAL_KEYS)},
    }


def restore_model(
    state: dict[str, torch.Tensor], *, device: torch.device, dtype: torch.dtype
) -> ActionOnly:
    spec = TRAIN.C58.ModelSpec(
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        carrier_source_mode=TRAIN.C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        action_layers=30,
    )
    tower = TRAIN.C58.build_model(
        spec, device=device, dtype=dtype, gradient_checkpointing=False
    )
    model = H3FACTLayerwiseTowerPolicy(
        tower, future_state_dim=8, future_representation_dim=TRAIN.FUTURE_DIM
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)
    return ActionOnly(model).eval()


def _replacement_provider_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    context = batch["replacement_text_context"]
    return {
        "current_h3_input": batch["current_h3_input"],
        "text_context": context,
        "text_token_tags": torch.ones(
            context.shape[:2], dtype=torch.long, device=context.device
        ),
    }


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> float:
    valid = (~pad).unsqueeze(-1).expand_as(prediction)
    return float(((prediction - target).float().square()[valid]).mean().cpu())


@torch.no_grad()
def evaluate_arm(
    *,
    name: str,
    payload: dict[str, Any],
    loader: DataLoader,
    provider: nn.Module,
    dataset: Any,
    scheduler: Any,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    first = ONLINE.move_batch(next(iter(loader)), device, dtype)
    first_cache = provider(ONLINE._provider_batch(first, shuffled=False))
    restore_noise = PROTOCOL.deterministic_noise_like(
        first["actions"], seed + 9_000_001
    )
    first_model = restore_model(payload["model"], device=device, dtype=dtype)
    first_prediction = ONLINE.sample_action_flow(
        first_model, first, first_cache, scheduler,
        initial_noise=restore_noise, inference_steps=10,
    ).float().cpu()
    del first_model
    gc.collect()
    torch.cuda.empty_cache()
    model = restore_model(payload["model"], device=device, dtype=dtype)
    second_prediction = ONLINE.sample_action_flow(
        model, first, first_cache, scheduler,
        initial_noise=restore_noise, inference_steps=10,
    ).float().cpu()
    restore_max_abs = float((first_prediction - second_prediction).abs().max())
    if restore_max_abs != 0.0:
        raise RuntimeError(f"{name} independent strict restore mismatch")

    normalized = PROTOCOL.DomainMetricAccumulator(7)
    physical = PROTOCOL.DomainMetricAccumulator(7)
    gripper = PROTOCOL.GripperSignAccumulator(6)
    language_action_path = PROTOCOL.LanguageSensitivityAccumulator()
    language_end_to_end = PROTOCOL.LanguageSensitivityAccumulator()
    visual_delta = PROTOCOL.DomainMetricAccumulator(7)
    evaluated_ids: list[str] = []
    tasks: Counter[str] = Counter()
    per_sample: dict[str, dict[str, float]] = {}
    seconds: list[float] = []
    for index, cpu_batch in enumerate(loader):
        started = time.perf_counter()
        batch = ONLINE.move_batch(cpu_batch, device, dtype)
        cache = provider(ONLINE._provider_batch(batch, shuffled=False))
        shuffled_cache = provider(ONLINE._provider_batch(batch, shuffled=True))
        replacement_cache = provider(_replacement_provider_batch(batch))
        noise = PROTOCOL.deterministic_noise_like(
            batch["actions"], seed + 1_000_003 * index
        )
        prediction = ONLINE.sample_action_flow(
            model, batch, cache, scheduler, initial_noise=noise,
            inference_steps=10,
        )
        shuffled = ONLINE.sample_action_flow(
            model, batch, shuffled_cache, scheduler, initial_noise=noise,
            inference_steps=10,
        )
        replacement_action = ONLINE.sample_action_flow(
            model, batch, cache, scheduler, initial_noise=noise,
            inference_steps=10, replacement_language=True,
        )
        replacement_full = ONLINE.sample_action_flow(
            model, batch, replacement_cache, scheduler, initial_noise=noise,
            inference_steps=10, replacement_language=True,
        )
        action_min = dataset.action_min.to(device=device, dtype=dtype)
        action_max = dataset.action_max.to(device=device, dtype=dtype)
        prediction_physical = PROTOCOL.denormalize_minmax_official(
            prediction, action_min, action_max
        )
        pad = batch["action_is_pad"]
        normalized.update(prediction, batch["actions"], pad)
        physical.update(prediction_physical, batch["raw_actions"], pad)
        gripper.update(prediction, batch["actions"], pad)
        language_action_path.update(prediction, replacement_action, pad)
        language_end_to_end.update(prediction, replacement_full, pad)
        visual_delta.update(shuffled, prediction, pad)
        sample_id = cpu_batch["sample_ids"][0]
        per_sample[sample_id] = {
            "normalized_action_mse": _masked_mse(
                prediction, batch["actions"], pad
            ),
            "physical_action_mse": _masked_mse(
                prediction_physical, batch["raw_actions"], pad
            ),
        }
        evaluated_ids.append(sample_id)
        tasks.update(cpu_batch["tasks"])
        seconds.append(time.perf_counter() - started)
    if PROTOCOL.sha256_strings(evaluated_ids) != SELECTED_IDS_SHA256:
        raise RuntimeError(f"{name} evaluated sample order drifted")
    return {
        "checkpoint_completed_steps": payload["completed_steps"],
        "strict_fresh_restore": {
            "independent_instances": 2,
            "same_noise": True,
            "max_abs": restore_max_abs,
        },
        "evaluated_ids_sha256": PROTOCOL.sha256_strings(evaluated_ids),
        "task_counts": dict(sorted(tasks.items())),
        "metrics": {
            "normalized_clip5_model_domain": normalized.finalize(),
            "denormalized_official_minmax_clamp": physical.finalize(),
            "gripper_sign": gripper.finalize(),
            "language_replacement_action_cross_attention_only":
                language_action_path.finalize(),
            "language_replacement_end_to_end_h3_and_action":
                language_end_to_end.finalize(),
            "visual_feature_shuffle_baseline_delta": visual_delta.finalize(),
        },
        "per_sample": per_sample,
        "timing": {"mean_seconds_per_sample": sum(seconds) / len(seconds)},
    }


def _conditioning_gate(arm: dict[str, Any]) -> dict[str, bool]:
    metrics = arm["metrics"]
    normalized = metrics["normalized_clip5_model_domain"]
    gripper = metrics["gripper_sign"]
    language = metrics["language_replacement_end_to_end_h3_and_action"]
    visual = metrics["visual_feature_shuffle_baseline_delta"]
    return {
        "prediction_not_constant": float(normalized["prediction_std"]) >= 0.01,
        "gripper_metric_finite": math.isfinite(float(gripper["macro_f1"])),
        "language_sensitive": float(language["mean_abs_prediction_delta"]) >= 0.01,
        "visual_sensitive": float(visual["action_mse"]) >= 1e-4,
    }


def _paired_wins(main: dict[str, Any], c61: dict[str, Any]) -> dict[str, Any]:
    if set(main["per_sample"]) != set(c61["per_sample"]):
        raise ValueError("paired sample sets differ")
    result: dict[str, Any] = {}
    for metric in ("normalized_action_mse", "physical_action_mse"):
        main_wins = c61_wins = ties = 0
        deltas = []
        for sample_id in main["per_sample"]:
            left = main["per_sample"][sample_id][metric]
            right = c61["per_sample"][sample_id][metric]
            deltas.append(right - left)
            if abs(left - right) <= 1e-12:
                ties += 1
            elif left < right:
                main_wins += 1
            else:
                c61_wins += 1
        result[metric] = {
            "main_wins": main_wins,
            "c61_wins": c61_wins,
            "ties": ties,
            "c61_minus_main_mean": sum(deltas) / len(deltas),
        }
    return result


def run(config: EvalConfig) -> dict[str, Any]:
    if config.seed != 42 or config.inference_steps != 10:
        raise ValueError("paired balanced80 seed/solver is fixed")
    main_ready, main_payload = _load_ready(config.main_ready.resolve(), "C60_MAIN")
    c61_ready, c61_payload = _load_ready(config.c61_ready.resolve(), "C61_MATCHED")
    pair_contract = assert_only_causal_failure_diff(
        main_payload["contract"], c61_payload["contract"]
    )
    if main_ready["c58_parent_sha256"] != c61_ready["c58_parent_sha256"]:
        raise ValueError("paired arms do not share the exact C58 parent")

    source_rows = PROTOCOL.read_jsonl(config.source_manifest)
    train_rows = PROTOCOL.read_jsonl(config.train_manifest)
    val_rows = PROTOCOL.read_jsonl(config.val_manifest)
    split = PROTOCOL.validate_episode_disjoint_manifests(
        source_rows, train_rows, val_rows
    )
    selected, selection = PROTOCOL.select_validation_rows(
        val_rows, samples_per_task=2
    )
    if (
        len(selected) != EXPECTED_ITEMS
        or selection["selected_ids_sha256"] != SELECTED_IDS_SHA256
        or selection["selected_task_count"] != 40
        or any(value != 2 for value in selection["task_counts"].values())
    ):
        raise ValueError("paired balanced80 selection drifted")
    actual_shared_data = {
        "source_manifest_sha256": sha256_file(config.source_manifest.resolve()),
        "demo_manifest_sha256": sha256_file(config.train_manifest.resolve()),
        "demo_stats_sha256": sha256_file(config.cache_root.resolve() / "stats.pt"),
    }
    if any(
        main_payload["contract"].get(key) != value
        for key, value in actual_shared_data.items()
    ):
        raise ValueError("heldout evaluator data differs from paired training contract")
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected)
    dataset = ONLINE.OnlineC58bValidationDataset(
        selected, cache_root=config.cache_root.resolve(), action_horizon=32,
        visual_mapping=visual_mapping,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ONLINE.collate_online,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    if device.type != "cuda":
        raise ValueError("online INT8 H3 paired evaluation requires CUDA")
    if sha256_file(config.h3_checkpoint.resolve()) != TRAIN.EXPECTED_H3_SHA256:
        raise ValueError("paired evaluator H3 checkpoint identity mismatch")
    provider = ONLINE.C58OnlineFrozenH3Provider(
        config.h3_checkpoint.resolve(), layers=tuple(LAYERWISE_H3_50_TO_ACTION_30)
    ).to(device=device).eval()
    scheduler = PROTOCOL.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    started = time.perf_counter()
    arms = {}
    for name, payload in (("c60_main", main_payload), ("c61_matched", c61_payload)):
        arms[name] = evaluate_arm(
            name=name, payload=payload, loader=loader, provider=provider,
            dataset=dataset, scheduler=scheduler, device=device, dtype=dtype,
            seed=config.seed,
        )
        gc.collect()
        torch.cuda.empty_cache()
    gates = {name: _conditioning_gate(value) for name, value in arms.items()}
    passed = all(all(values.values()) for values in gates.values())
    report = {
        "format": FORMAT,
        "status": "PASS_PAIRED_BALANCED80" if passed else "FAIL_CONDITIONING_COLLAPSE",
        "permission": "GO_PAIRED_LIBERO" if passed else "NO_GO_PAIRED_LIBERO",
        "effect_status": "NOT_EVIDENCE_READY",
        "hypothesis": (
            "Replacing only C60 causal failures with the finalized C61 pool improves "
            "fresh-execution LIBERO success without conditioning collapse."
        ),
        "checkpoint_identity": {
            "c60_main_ready": str(config.main_ready.resolve()),
            "c60_main_ready_sha256": sha256_file(config.main_ready.resolve()),
            "c60_main_checkpoint_sha256": main_ready["checkpoint_sha256"],
            "c61_matched_ready": str(config.c61_ready.resolve()),
            "c61_matched_ready_sha256": sha256_file(config.c61_ready.resolve()),
            "c61_matched_checkpoint_sha256": c61_ready["checkpoint_sha256"],
            "shared_c58_parent_sha256": main_ready["c58_parent_sha256"],
        },
        "paired_contract": pair_contract,
        "data": {
            "source_manifest_sha256": sha256_file(config.source_manifest.resolve()),
            "train_manifest_sha256": sha256_file(config.train_manifest.resolve()),
            "validation_manifest_sha256": sha256_file(config.val_manifest.resolve()),
            "stats_sha256": sha256_file(config.cache_root.resolve() / "stats.pt"),
            "selection": selection,
            "split_audit": split,
            "visual_shuffle": visual_contract,
        },
        "execution": {
            "h3": "online_frozen_int8",
            "h3_checkpoint_sha256": TRAIN.EXPECTED_H3_SHA256,
            "disk_kv_read": False,
            "disk_kv_write": False,
            "disk_feature_read": False,
            "carrier_layers": list(LAYERWISE_H3_50_TO_ACTION_30),
            "same_selected_samples_noise_solver_normalization": True,
            "seed": 42,
            "inference_steps": 10,
            "shift": 5.0,
        },
        "arms": arms,
        "paired_offline": _paired_wins(arms["c60_main"], arms["c61_matched"]),
        "conditioning_gates": gates,
        "timing": {"elapsed_seconds": time.perf_counter() - started},
        "claim_boundary": (
            "Offline regression and counterfactual sensitivity only. Permission "
            "launches matched fresh executions; LIBERO success decides the effect."
        ),
    }
    output = config.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing paired evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-ready", type=Path, required=True)
    parser.add_argument("--c61-ready", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    values = parser.parse_args()
    return EvalConfig(**vars(values))


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2), flush=True)
