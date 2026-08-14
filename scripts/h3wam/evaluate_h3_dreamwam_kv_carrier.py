#!/usr/bin/env python3
"""Balanced-80 evaluator adapter for Candidate D DreamWAM H3 K/V carrier.

This file is deliberately separate from the StarWAM evaluator.  It imports
that evaluator as the frozen evaluation protocol and reuses its selection,
shuffle, collation, shifted-flow sampler, normalization and metric code.  The
only carrier-specific adaptation is packing/unpacking a complete five-layer
H3 K/V bundle through the protocol's ``features`` slot.
"""

from __future__ import annotations

import argparse
import gc
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


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_sibling(
    "_h3_starwam_balanced80_protocol", "evaluate_h3_int8_starwam_action.py"
)
CANDIDATE_D = _load_sibling(
    "_h3_dreamwam_candidate_d_trainer", "train_h3_int8_dreamwam_kv_carrier.py"
)

EXPECTED_SEED = 42
EXPECTED_BATCH_SIZE = 1
EXPECTED_SAMPLES_PER_TASK = 2
EXPECTED_SELECTED_ITEMS = 80
EXPECTED_KV_TIMESTEP = 1.0
EXPECTED_H3_CHECKPOINT_SHA256 = CANDIDATE_D.PARENT.H3_INT8_CHECKPOINT_SHA256
CANDIDATE_D_V4_SELECTED_IDS_SHA256 = (
    "b507e1ff6031f01c88cd6181aaeb4cba33b76e2c67737a986bf764c76be87519"
)
R1_G_V8_SELECTED_IDS_SHA256 = (
    "75d888fbb4298bef3517b623c00861ac6fe036495dee3bf4f0c68b5c097c5f54"
)
MODEL_SPEC_KEYS = {
    "action_dim",
    "proprio_dim",
    "context_dim",
    "hidden_dim",
    "ffn_dim",
    "num_heads",
    "attn_head_dim",
    "freq_dim",
    "carrier_layers",
    "carrier_source_mode",
}
CONTRACT_KEYS = {
    "candidate",
    "classification",
    "dreamwam_commit",
    "dreamwam_layers_sha256",
    "dreamwam_experts_sha256",
    "dreamwam_mot_sha256",
    "parent_shifted_flow_commit",
    "carrier_source_mode",
    "h3_checkpoint_path",
    "h3_checkpoint_sha256",
    "verify_h3_checkpoint_sha256",
    "kv_subdir",
    "kv_schema",
    "kv_strategy",
    "kv_layers",
    "kv_tokens",
    "kv_num_heads",
    "kv_attn_head_dim",
    "kv_bytes_per_sample",
    "source_manifest_sha256",
    "source_manifest_items",
    "split_manifest_sha256",
    "split_manifest_items",
    "stats_sha256",
    "action_normalization",
    "state_normalization",
    "action_horizon",
    "action_shift",
    "flow_timestep_contract",
    "flow_rng_contract",
    "training_topology",
    "lr_schedule",
    "model_spec",
}


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path | None = None
    kv_subdir: str | None = None
    device: str = "cpu"
    num_workers: int = 0
    seed: int = EXPECTED_SEED
    batch_size: int = EXPECTED_BATCH_SIZE
    samples_per_task: int = EXPECTED_SAMPLES_PER_TASK
    inference_steps: int = PROTOCOL.EXPECTED_INFERENCE_STEPS
    action_shift: float = PROTOCOL.EXPECTED_ACTION_SHIFT
    language_sensitivity: bool = True
    visual_feature_shuffle: bool = True
    cache_audit_aggregate_sha256: str | None = None
    expected_selected_ids_sha256: str = CANDIDATE_D_V4_SELECTED_IDS_SHA256


def require_balanced80_protocol(config: EvalConfig) -> None:
    expected = {
        "seed": EXPECTED_SEED,
        "batch_size": EXPECTED_BATCH_SIZE,
        "samples_per_task": EXPECTED_SAMPLES_PER_TASK,
        "inference_steps": PROTOCOL.EXPECTED_INFERENCE_STEPS,
        "action_shift": PROTOCOL.EXPECTED_ACTION_SHIFT,
        "language_sensitivity": True,
        "visual_feature_shuffle": True,
    }
    mismatches = {
        key: {"requested": getattr(config, key), "required": value}
        for key, value in expected.items()
        if getattr(config, key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate D balanced-80 protocol mismatch: {mismatches}")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    sha_values = {
        "expected selected IDs": config.expected_selected_ids_sha256,
    }
    if config.cache_audit_aggregate_sha256 is not None:
        sha_values["external cache audit aggregate"] = (
            config.cache_audit_aggregate_sha256
        )
    for name, value in sha_values.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} SHA256 must be lowercase hex")


def require_selected_ids(selection: dict[str, Any], config: EvalConfig) -> str:
    actual = str(selection.get("selected_ids_sha256", ""))
    if actual != config.expected_selected_ids_sha256:
        raise ValueError(
            "Candidate D selected IDs differ from the expected manifest-specific "
            f"gate: actual={actual}, expected={config.expected_selected_ids_sha256}"
        )
    return actual


def _require_contract(
    payload: dict[str, Any],
    *,
    config: EvalConfig,
    source_manifest_sha256: str,
    source_manifest_items: int,
    train_manifest_sha256: str,
    train_manifest_items: int,
    stats_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if set(payload) != set(CANDIDATE_D.CHECKPOINT_KEYS):
        raise ValueError("Candidate D checkpoint top-level schema mismatch")
    if payload.get("schema_version") != CANDIDATE_D.CHECKPOINT_SCHEMA:
        raise ValueError("Candidate D checkpoint schema mismatch")
    if not isinstance(payload.get("completed_steps"), int) or payload["completed_steps"] < 0:
        raise ValueError("checkpoint completed_steps must be a non-negative integer")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("checkpoint model state is missing or empty")
    contract = payload.get("contract")
    if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
        raise ValueError("Candidate D checkpoint contract schema mismatch")
    required = {
        "classification": "action-only-on-frozen-features",
        "dreamwam_commit": CANDIDATE_D.DREAMWAM_COMMIT,
        "dreamwam_layers_sha256": CANDIDATE_D.DREAMWAM_LAYERS_SHA256,
        "dreamwam_experts_sha256": CANDIDATE_D.DREAMWAM_EXPERTS_SHA256,
        "dreamwam_mot_sha256": CANDIDATE_D.DREAMWAM_MOT_SHA256,
        "parent_shifted_flow_commit": CANDIDATE_D.PARENT_OBJECTIVE_COMMIT,
        "h3_checkpoint_sha256": EXPECTED_H3_CHECKPOINT_SHA256,
        "verify_h3_checkpoint_sha256": True,
        "kv_schema": CANDIDATE_D.DREAMWAM_KV_SCHEMA,
        "kv_strategy": CANDIDATE_D.DREAMWAM_KV_STRATEGY,
        "kv_layers": list(CANDIDATE_D.DEFAULT_H3_CARRIER_LAYERS),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_items": source_manifest_items,
        "split_manifest_sha256": train_manifest_sha256,
        "split_manifest_items": train_manifest_items,
        "stats_sha256": stats_sha256,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
    }
    mismatches = {
        key: {"checkpoint": contract.get(key), "expected": expected}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Candidate D evaluation contract mismatch: {mismatches}")
    if not math.isclose(float(contract["action_shift"]), config.action_shift):
        raise ValueError("checkpoint action_shift differs from balanced-80 protocol")
    model_spec = contract.get("model_spec")
    if not isinstance(model_spec, dict) or set(model_spec) != MODEL_SPEC_KEYS:
        raise ValueError("Candidate D model_spec schema mismatch")
    carrier_source_mode = str(contract.get("carrier_source_mode", ""))
    if carrier_source_mode not in (
        CANDIDATE_D.ALIGNED_5LAYER_CARRIER_SOURCE,
        CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE,
    ):
        raise ValueError("Candidate D/D0 carrier_source_mode is invalid")
    expected_candidate = (
        "D0"
        if carrier_source_mode == CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
        else "D"
    )
    if contract.get("candidate") != expected_candidate:
        raise ValueError("candidate label does not match carrier_source_mode")
    if model_spec.get("carrier_source_mode") != carrier_source_mode:
        raise ValueError("model_spec and contract carrier_source_mode disagree")
    layers = tuple(int(value) for value in model_spec["carrier_layers"])
    if layers != CANDIDATE_D.DEFAULT_H3_CARRIER_LAYERS:
        raise ValueError("Candidate D carrier layer mapping is not the audited mapping")
    numeric_contract = {
        "kv_tokens": 32,
        "kv_num_heads": int(model_spec["num_heads"]),
        "kv_attn_head_dim": int(model_spec["attn_head_dim"]),
        "action_horizon": 32,
    }
    for key, expected in numeric_contract.items():
        if int(contract.get(key, -1)) != expected:
            raise ValueError(f"Candidate D {key} differs from audited protocol")
    expected_bytes = CANDIDATE_D.h3_kv_cache_bytes(
        layers=len(layers),
        tokens=int(contract["kv_tokens"]),
        heads=int(contract["kv_num_heads"]),
        head_dim=int(contract["kv_attn_head_dim"]),
    )
    if int(contract.get("kv_bytes_per_sample", -1)) != expected_bytes:
        raise ValueError("Candidate D kv_bytes_per_sample is inconsistent")
    kv_subdir = str(contract.get("kv_subdir", ""))
    if not kv_subdir:
        raise ValueError("Candidate D kv_subdir is missing")
    if config.kv_subdir is not None and config.kv_subdir != kv_subdir:
        raise ValueError("requested kv_subdir differs from checkpoint contract")
    if not str(contract.get("h3_checkpoint_path", "")):
        raise ValueError("Candidate D H3 checkpoint identity is missing")
    return contract, model_spec, kv_subdir


class CachedDreamWAMKVValidationDataset(PROTOCOL.CachedLast32ValidationDataset):
    """Use the canonical val data path while replacing only the carrier load."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_root: Path,
        kv_subdir: str,
        source_manifest_items: int,
        model_spec: dict[str, Any],
        action_horizon: int,
        h3_checkpoint_path: str,
        visual_feature_shuffle: dict[str, str],
    ) -> None:
        self.carrier_layers = tuple(int(value) for value in model_spec["carrier_layers"])
        self.kv_tokens = 32
        self.kv_num_heads = int(model_spec["num_heads"])
        self.kv_attn_head_dim = int(model_spec["attn_head_dim"])
        self.h3_checkpoint_path = Path(h3_checkpoint_path)
        parent_spec = {
            "action_dim": int(model_spec["action_dim"]),
            "proprio_dim": int(model_spec["proprio_dim"]),
            "h3_feature_dim": self.kv_num_heads * self.kv_attn_head_dim,
            "context_dim": int(model_spec["context_dim"]),
        }
        super().__init__(
            rows,
            cache_root=cache_root,
            feature_subdir=kv_subdir,
            source_manifest_items=source_manifest_items,
            model_spec=parent_spec,
            action_horizon=action_horizon,
            visual_feature_shuffle=visual_feature_shuffle,
            limit=0,
            sample_offset=0,
        )
        self.kv_root = self.feature_root

    def _load_features(self, sample_id: str, expected_context_id: str) -> torch.Tensor:
        path = self.kv_root / f"{sample_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing DreamWAM K/V cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "schema": CANDIDATE_D.DREAMWAM_KV_SCHEMA,
            "layers": self.carrier_layers,
            "capture_token_count": self.kv_tokens,
            "num_heads": self.kv_num_heads,
            "attn_head_dim": self.kv_attn_head_dim,
            "capture_token_strategy": CANDIDATE_D.DREAMWAM_KV_STRATEGY,
            "dreamwam_commit": CANDIDATE_D.DREAMWAM_COMMIT,
            "context_id": expected_context_id,
            "action_horizon": self.action_horizon,
            "backbone": CANDIDATE_D.CACHE_BACKBONE,
            "quantization": CANDIDATE_D.CACHE_QUANTIZATION,
            "manifest_items": self.source_manifest_items,
        }
        for key, expected_value in expected.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"DreamWAM K/V cache mismatch for {sample_id}: "
                    f"{key}={actual!r}, expected {expected_value!r}"
                )
        if not math.isclose(float(payload.get("timestep", -1.0)), EXPECTED_KV_TIMESTEP):
            raise ValueError(f"K/V cache timestep must be 1.0 for {sample_id}")
        if Path(payload.get("checkpoint", "")) != self.h3_checkpoint_path:
            raise ValueError(f"H3 checkpoint identity mismatch for {sample_id}")
        cache = payload.get("video_kv_cache")
        if not isinstance(cache, dict) or set(cache) != set(self.carrier_layers):
            raise ValueError(f"K/V layer mapping mismatch for {sample_id}")
        tensors: list[torch.Tensor] = []
        storage_signatures: set[int] = set()
        expected_shape = (self.kv_tokens, self.kv_num_heads, self.kv_attn_head_dim)
        for layer in self.carrier_layers:
            item = cache[layer]
            if set(item) != {"k", "v"}:
                raise ValueError(f"layer {layer} cache must contain k and v exactly")
            pair = []
            for name in ("k", "v"):
                tensor = item[name]
                if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
                    raise ValueError(f"layer {layer} {name} shape mismatch")
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"layer {layer} {name} must be bfloat16")
                signature = tensor.untyped_storage().data_ptr()
                if signature in storage_signatures:
                    raise ValueError(f"layer-specific K/V storage aliases at {layer} {name}")
                storage_signatures.add(signature)
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite layer {layer} {name} for {sample_id}")
                pair.append(tensor)
            tensors.append(torch.stack(pair, dim=0))
        # [layers, K/V, tokens, heads, head_dim], treated as one visual carrier.
        return torch.stack(tensors, dim=0)


class DreamWAMKVProtocolPolicy(nn.Module):
    """Expose Candidate D through the canonical evaluator forward signature."""

    def __init__(self, policy: nn.Module, carrier_layers: tuple[int, ...]) -> None:
        super().__init__()
        self.policy = policy
        self.carrier_layers = carrier_layers

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        h3_features: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected_prefix = (noisy_actions.shape[0], len(self.carrier_layers), 2)
        if h3_features.ndim != 6 or tuple(h3_features.shape[:3]) != expected_prefix:
            raise ValueError(
                "packed DreamWAM carrier must be [B,layers,2,tokens,heads,head_dim]"
            )
        # Each clone restores the no-alias guarantee required by Candidate D.
        cache = {
            layer: {
                "k": h3_features[:, index, 0].clone(),
                "v": h3_features[:, index, 1].clone(),
            }
            for index, layer in enumerate(self.carrier_layers)
        }
        autocast = (
            torch.autocast(device_type=noisy_actions.device.type, dtype=torch.bfloat16)
            if noisy_actions.dtype == torch.bfloat16
            and noisy_actions.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast:
            return self.policy(
                noisy_actions,
                timestep,
                text_context=text_context,
                proprio=proprio,
                video_kv_cache=cache,
                text_mask=text_mask,
            )


def restore_model_strict(
    model_spec: dict[str, Any],
    model_state: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DreamWAMKVProtocolPolicy:
    normalized = dict(model_spec)
    normalized["carrier_layers"] = tuple(normalized["carrier_layers"])
    policy = CANDIDATE_D.build_model(
        CANDIDATE_D.ModelSpec(**normalized), device=device, dtype=dtype
    )
    policy.load_state_dict(model_state, strict=True)
    policy.eval()
    return DreamWAMKVProtocolPolicy(policy, tuple(normalized["carrier_layers"])).eval()


def _fresh_restore_check(
    *,
    model_spec: dict[str, Any],
    model_state: dict[str, Any],
    first_batch: dict[str, Any],
    scheduler: Any,
    config: EvalConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    noise = PROTOCOL.deterministic_noise_like(
        first_batch["actions"], config.seed + 9_000_001
    )
    predictions = []
    second_model = None
    for _ in range(2):
        model = restore_model_strict(
            model_spec, model_state, device=device, dtype=dtype
        )
        predictions.append(
            PROTOCOL.sample_action_flow(
                model,
                first_batch,
                scheduler,
                inference_steps=config.inference_steps,
                initial_noise=noise,
            ).float().cpu()
        )
        if second_model is not None:
            del second_model
        second_model = model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    max_abs = float((predictions[0] - predictions[1]).abs().max())
    if max_abs != 0.0:
        raise RuntimeError(f"fresh restore prediction mismatch: max_abs={max_abs}")
    assert second_model is not None
    return second_model, {
        "strict_state_dict": True,
        "independent_model_instances": 2,
        "same_noise": True,
        "max_abs": max_abs,
        "sample_ids": list(first_batch["sample_ids"]),
    }


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Run Candidate D while delegating every metric/protocol primitive upstream."""

    started = time.perf_counter()
    require_balanced80_protocol(config)
    source_manifest = config.source_manifest.resolve()
    train_manifest = config.train_manifest.resolve()
    val_manifest = config.val_manifest.resolve()
    cache_root = config.cache_root.resolve()
    source_rows = PROTOCOL.read_jsonl(source_manifest)
    train_rows = PROTOCOL.read_jsonl(train_manifest)
    val_rows = PROTOCOL.read_jsonl(val_manifest)
    split_audit = PROTOCOL.validate_episode_disjoint_manifests(
        source_rows, train_rows, val_rows
    )
    selected_rows, selection = PROTOCOL.select_validation_rows(
        val_rows, samples_per_task=config.samples_per_task
    )
    if len(selected_rows) != EXPECTED_SELECTED_ITEMS:
        raise ValueError("balanced-80 selection did not produce exactly 80 samples")
    selected_ids_gate = require_selected_ids(selection, config)
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected_rows)
    hashes = {
        "source_manifest_sha256": PROTOCOL.sha256_file(source_manifest),
        "train_manifest_sha256": PROTOCOL.sha256_file(train_manifest),
        "validation_manifest_sha256": PROTOCOL.sha256_file(val_manifest),
        "stats_sha256": PROTOCOL.sha256_file(cache_root / "stats.pt"),
    }
    payload, checkpoint_sha256 = PROTOCOL._load_checkpoint(config.checkpoint)
    contract, model_spec, kv_subdir = _require_contract(
        payload,
        config=config,
        source_manifest_sha256=hashes["source_manifest_sha256"],
        source_manifest_items=len(source_rows),
        train_manifest_sha256=hashes["train_manifest_sha256"],
        train_manifest_items=len(train_rows),
        stats_sha256=hashes["stats_sha256"],
    )
    dataset = CachedDreamWAMKVValidationDataset(
        selected_rows,
        cache_root=cache_root,
        kv_subdir=kv_subdir,
        source_manifest_items=len(source_rows),
        model_spec=model_spec,
        action_horizon=int(contract["action_horizon"]),
        h3_checkpoint_path=str(contract["h3_checkpoint_path"]),
        visual_feature_shuffle=visual_mapping,
    )
    if len(dataset.replacement_context) < 2:
        raise ValueError("language sensitivity requires at least two context ids")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=PROTOCOL.collate_eval_batch,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    scheduler = PROTOCOL.FlowMatchScheduler(
        num_train_timesteps=1000, shift=config.action_shift
    )
    first_batch = PROTOCOL.move_batch(next(iter(loader)), device, dtype)
    model, restore = _fresh_restore_check(
        model_spec=model_spec,
        model_state=payload["model"],
        first_batch=first_batch,
        scheduler=scheduler,
        config=config,
        device=device,
        dtype=dtype,
    )

    action_dim = int(model_spec["action_dim"])
    normalized = PROTOCOL.DomainMetricAccumulator(action_dim)
    physical = PROTOCOL.DomainMetricAccumulator(action_dim)
    gripper = PROTOCOL.GripperSignAccumulator(action_dim - 1)
    language = PROTOCOL.LanguageSensitivityAccumulator()
    shuffled_normalized = PROTOCOL.DomainMetricAccumulator(action_dim)
    shuffled_physical = PROTOCOL.DomainMetricAccumulator(action_dim)
    shuffled_gripper = PROTOCOL.GripperSignAccumulator(action_dim - 1)
    normalized_delta = PROTOCOL.DomainMetricAccumulator(action_dim)
    physical_delta = PROTOCOL.DomainMetricAccumulator(action_dim)
    evaluated_ids: list[str] = []
    evaluated_pairs: list[str] = []
    evaluated_tasks: Counter[str] = Counter()
    with torch.no_grad():
        for batch_index, batch_cpu in enumerate(loader):
            batch = PROTOCOL.move_batch(batch_cpu, device, dtype)
            noise = PROTOCOL.deterministic_noise_like(
                batch["actions"], config.seed + 1_000_003 * batch_index
            )
            prediction = PROTOCOL.sample_action_flow(
                model, batch, scheduler,
                inference_steps=config.inference_steps, initial_noise=noise,
            )
            shuffled = PROTOCOL.sample_action_flow(
                model, batch, scheduler,
                inference_steps=config.inference_steps, initial_noise=noise,
                replacement_visual=True,
            )
            replacement = PROTOCOL.sample_action_flow(
                model, batch, scheduler,
                inference_steps=config.inference_steps, initial_noise=noise,
                replacement_language=True,
            )
            prediction_physical = PROTOCOL.denormalize_minmax_official(
                prediction, dataset.action_min.to(device=device, dtype=dtype),
                dataset.action_max.to(device=device, dtype=dtype),
            )
            shuffled_physical_value = PROTOCOL.denormalize_minmax_official(
                shuffled, dataset.action_min.to(device=device, dtype=dtype),
                dataset.action_max.to(device=device, dtype=dtype),
            )
            pad = batch["action_is_pad"]
            normalized.update(prediction, batch["actions"], pad)
            physical.update(prediction_physical, batch["raw_actions"], pad)
            gripper.update(prediction, batch["actions"], pad)
            language.update(prediction, replacement, pad)
            shuffled_normalized.update(shuffled, batch["actions"], pad)
            shuffled_physical.update(shuffled_physical_value, batch["raw_actions"], pad)
            shuffled_gripper.update(shuffled, batch["actions"], pad)
            normalized_delta.update(shuffled, prediction, pad)
            physical_delta.update(shuffled_physical_value, prediction_physical, pad)
            evaluated_ids.extend(batch_cpu["sample_ids"])
            evaluated_tasks.update(batch_cpu["tasks"])
            evaluated_pairs.extend(
                f"{target}\0{source}" for target, source in zip(
                    batch_cpu["sample_ids"], batch_cpu["visual_shuffle_source_ids"],
                    strict=True,
                )
            )

    normalized_report = normalized.finalize()
    physical_report = physical.finalize()
    gripper_report = gripper.finalize()
    shuffled_normalized_report = shuffled_normalized.finalize()
    shuffled_physical_report = shuffled_physical.finalize()
    shuffled_gripper_report = shuffled_gripper.finalize()
    ids_sha = PROTOCOL.sha256_strings(evaluated_ids)
    pairs_sha = PROTOCOL.sha256_strings(evaluated_pairs)
    if ids_sha != selection["selected_ids_sha256"]:
        raise RuntimeError("evaluated sample order differs from frozen selection")
    if pairs_sha != visual_contract["ordered_mapping_sha256"]:
        raise RuntimeError("evaluated K/V shuffle differs from frozen mapping")
    if dict(sorted(evaluated_tasks.items())) != selection["task_counts"]:
        raise RuntimeError("evaluated task counts differ from frozen selection")
    report = {
        "event": (
            "h3_dreamwam_kv_candidate_d0_balanced80_offline_evaluation"
            if contract["carrier_source_mode"]
            == CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
            else "h3_dreamwam_kv_candidate_d_balanced80_offline_evaluation"
        ),
        "candidate": contract["candidate"],
        "carrier_source_mode": contract["carrier_source_mode"],
        "classification": "causal-action-on-frozen-five-layer-h3-kv",
        "status": "completed_not_closed_loop_evidence",
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "schema_version": payload["schema_version"],
            "completed_steps": payload["completed_steps"],
            "contract": contract,
            "h3_checkpoint_sha256": contract["h3_checkpoint_sha256"],
            "h3_checkpoint_sha256_verified": contract[
                "verify_h3_checkpoint_sha256"
            ],
            "fresh_restore": restore,
        },
        "protocol_identity": {
            "adapter": str(Path(__file__).resolve()),
            "starwam_evaluator_sha256": PROTOCOL.sha256_file(
                Path(PROTOCOL.__file__).resolve()
            ),
            "selection_salt": PROTOCOL.BALANCED_VAL_SELECTION_SALT,
            "visual_shuffle_salt": PROTOCOL.VISUAL_FEATURE_SHUFFLE_SALT,
            "dreamwam_commit": CANDIDATE_D.DREAMWAM_COMMIT,
            "carrier_source_mode": contract["carrier_source_mode"],
            "carrier_semantics": (
                "shuffle_source_bundle_then_repeat_source_layer49_per_block"
                if contract["carrier_source_mode"]
                == CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
                else "shuffle_complete_aligned_five_layer_kv_bundle"
            ),
            "h3_checkpoint_sha256": contract["h3_checkpoint_sha256"],
            "h3_checkpoint_sha256_verified": contract[
                "verify_h3_checkpoint_sha256"
            ],
            "cache_audit_aggregate_sha256": config.cache_audit_aggregate_sha256,
            "cache_aggregate_hash_binding": (
                "external_audit_declaration_not_recomputed_by_selected-cache-evaluator"
            ),
            "expected_selected_ids_sha256": config.expected_selected_ids_sha256,
            "actual_selected_ids_sha256": selected_ids_gate,
            "r1_g_v8_selected_ids_sha256": R1_G_V8_SELECTED_IDS_SHA256,
            "strict_parent_ids_matched": (
                selected_ids_gate == R1_G_V8_SELECTED_IDS_SHA256
            ),
            "parent_comparison_status": (
                "STRICT_PAIRED_IDS"
                if selected_ids_gate == R1_G_V8_SELECTED_IDS_SHA256
                else "NOT_STRICT_PAIRED_IDS"
            ),
        },
        "data": {
            "source_manifest_sha256": hashes["source_manifest_sha256"],
            "source_manifest_items": len(source_rows),
            "train_manifest_sha256": hashes["train_manifest_sha256"],
            "train_manifest_items": len(train_rows),
            "validation_manifest_sha256": hashes["validation_manifest_sha256"],
            "validation_manifest_items": len(val_rows),
            "stats_sha256": hashes["stats_sha256"],
            "selected_sample_ids_sha256": ids_sha,
            "selection": selection,
            "kv_root": str(dataset.kv_root),
            "split_audit": split_audit,
        },
        "inference": {
            "scheduler": "pinned StarWAM FlowMatchScheduler",
            "shift": config.action_shift,
            "steps": config.inference_steps,
            "integrator": "Euler sample += velocity * delta",
            "seed": config.seed,
            "batch_size": config.batch_size,
            "same_noise_for_baseline_language_visual": True,
            "carrier_source_mode": contract["carrier_source_mode"],
            "device": str(device),
            "dtype": str(dtype),
        },
        "metrics": {
            "normalized_clip5_model_domain": normalized_report,
            "denormalized_official_minmax_clamp": physical_report,
            "gripper_sign": gripper_report,
            "language_replacement_sensitivity": language.finalize(),
            "visual_feature_shuffle": {
                "contract": visual_contract,
                "evaluated_mapping_sha256": pairs_sha,
                "carrier_semantics": (
                    "replace_source_sample_then_repeat_its_layer49_per_block"
                    if contract["carrier_source_mode"]
                    == CANDIDATE_D.REPEAT_LAYER49_CARRIER_SOURCE
                    else "replace_complete_aligned_five_layer_kv_bundle"
                ),
                "same_initial_action_noise": True,
                "baseline_vs_shuffle_action_delta": {
                    "normalized_model_domain": normalized_delta.finalize(),
                    "denormalized_official_minmax_clamp": physical_delta.finalize(),
                },
                "shuffle_vs_target": {
                    "normalized_clip5_model_domain": shuffled_normalized_report,
                    "denormalized_official_minmax_clamp": shuffled_physical_report,
                    "gripper_sign": shuffled_gripper_report,
                },
                "metric_change_shuffle_minus_baseline": {
                    "normalized_clip5_model_domain": PROTOCOL.metric_changes(
                        normalized_report, shuffled_normalized_report,
                        PROTOCOL.DOMAIN_CHANGE_KEYS,
                    ),
                    "denormalized_official_minmax_clamp": PROTOCOL.metric_changes(
                        physical_report, shuffled_physical_report,
                        PROTOCOL.DOMAIN_CHANGE_KEYS,
                    ),
                    "gripper_sign": PROTOCOL.metric_changes(
                        gripper_report, shuffled_gripper_report,
                        PROTOCOL.GRIPPER_CHANGE_KEYS,
                    ),
                },
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if config.output is not None:
        output = config.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    return report


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-audit-aggregate-sha256",
        help="Aggregate cache hash supplied by a separate full-cache audit artifact.",
    )
    parser.add_argument(
        "--expected-selected-ids-sha256",
        default=CANDIDATE_D_V4_SELECTED_IDS_SHA256,
        help=(
            "Manifest-specific balanced-80 ID gate; defaults to audited Candidate D v4."
        ),
    )
    values = parser.parse_args()
    return EvalConfig(
        checkpoint=values.checkpoint,
        source_manifest=values.source_manifest,
        train_manifest=values.train_manifest,
        val_manifest=values.val_manifest,
        cache_root=values.cache_root,
        kv_subdir=values.kv_subdir,
        output=values.output,
        device=values.device,
        num_workers=values.num_workers,
        cache_audit_aggregate_sha256=values.cache_audit_aggregate_sha256,
        expected_selected_ids_sha256=values.expected_selected_ids_sha256,
    )


if __name__ == "__main__":
    run_evaluation(parse_args())
