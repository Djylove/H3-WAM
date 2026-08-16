#!/usr/bin/env python3
"""Strict balanced-80 evaluation for the online-H3 C58b full FastWAM tower.

Unlike the legacy C58 evaluator, this program cannot name or read a disk K/V
directory.  It loads only canonical first-frame VAE latents, text contexts,
actions and proprioception, then executes the frozen INT8 MiniMax H3 online to
materialize the exact thirty layer-wise K/V bundles consumed during training.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.c58_online_training import (  # noqa: E402
    C58OnlineFrozenH3Provider,
)
from fastwam.models.h3wam.fastwam_full_tower import (  # noqa: E402
    LAYERWISE_H3_50_TO_ACTION_30,
)


def _load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_sibling(
    "_c58b_online_balanced80_protocol", "evaluate_h3_int8_starwam_action.py"
)
C58 = _load_sibling(
    "_c58b_online_full_tower_trainer", "train_h3_fastwam_full_tower.py"
)

EXPECTED_SELECTED_IDS_SHA256 = (
    "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
)
EXPECTED_CHECKPOINT_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
EXPECTED_D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
EXPECTED_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_ITEMS = 80
EXPECTED_STEPS = 10_000
LAYERS = tuple(LAYERWISE_H3_50_TO_ACTION_30)
MODEL_SPEC_KEYS = set(asdict(C58.ModelSpec()))


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    ready: Path
    h3_checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path
    device: str = "cuda:0"
    num_workers: int = 0
    seed: int = 42
    batch_size: int = 1
    samples_per_task: int = 2
    inference_steps: int = 10
    action_shift: float = 5.0


def _load_context(cache_root: Path, context_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(
        cache_root / "contexts" / f"{context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    context = payload.get("context")
    tags = payload.get("token_tags")
    if (
        payload.get("text_only") is not True
        or not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or context.shape[0] != 1
        or context.shape[-1] != 5120
        or not isinstance(tags, torch.Tensor)
        or tags.ndim != 1
        or tags.numel() != context.shape[1]
    ):
        raise ValueError(f"invalid H3 text-only context: {context_id}")
    if torch.any(tags != 1):
        raise ValueError(f"context contains non-text tags: {context_id}")
    return context[0].float(), tags.long()


class OnlineC58bValidationDataset(Dataset):
    """Canonical validation data with no feature/K/V cache access."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_root: Path,
        action_horizon: int,
        visual_mapping: dict[str, str],
    ) -> None:
        self.rows = rows
        self.rows_by_id = {str(row["id"]): row for row in rows}
        self.cache_root = cache_root.resolve()
        self.action_horizon = int(action_horizon)
        self.visual_mapping = visual_mapping
        if set(visual_mapping) != set(self.rows_by_id) or set(
            visual_mapping.values()
        ) != set(self.rows_by_id):
            raise ValueError("visual mapping is not a permutation of selected rows")
        stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        if tuple(self.action_min.shape) != (7,) or tuple(self.state_min.shape) != (8,):
            raise ValueError("canonical stats shape mismatch")
        context_ids = sorted({str(row["context_id"]) for row in rows})
        if len(context_ids) < 2:
            raise ValueError("language replacement requires at least two contexts")
        self.replacement_context = {
            context_id: next(item for item in context_ids if item != context_id)
            for context_id in context_ids
        }

    def __len__(self) -> int:
        return len(self.rows)

    def _visual_input(self, row: dict[str, Any]) -> torch.Tensor:
        sample_id = str(row["id"])
        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        latent = window.get("first_frame_latents")
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or tuple(latent.shape[:3]) != (1, 24, 1)
        ):
            raise ValueError(f"invalid first-frame VAE latent: {sample_id}")
        return latent[0].float().clone()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        source_id = self.visual_mapping[sample_id]
        source_row = self.rows_by_id[source_id]
        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        raw_actions = window["actions"][: self.action_horizon].float()
        state = window["state"].float()
        pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(raw_actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"action shape mismatch: {sample_id}")
        if tuple(state.shape) != (8,) or not bool((~pad).any()):
            raise ValueError(f"state/padding mismatch: {sample_id}")
        context_id = str(row["context_id"])
        source_context_id = str(source_row["context_id"])
        context, tags = _load_context(self.cache_root, context_id)
        source_context, source_tags = _load_context(
            self.cache_root, source_context_id
        )
        replacement, _ = _load_context(
            self.cache_root, self.replacement_context[context_id]
        )
        return {
            "sample_id": sample_id,
            "task": str(row["task"]),
            "actions": PROTOCOL.normalize_minmax(
                raw_actions, self.action_min, self.action_max
            ),
            "raw_actions": raw_actions,
            "proprio": PROTOCOL.normalize_minmax(
                state, self.state_min, self.state_max
            ),
            "action_is_pad": pad,
            "text_context": context,
            "text_token_tags": tags,
            "replacement_text_context": replacement,
            "current_h3_input": self._visual_input(row),
            "shuffled_h3_input": self._visual_input(source_row),
            "shuffled_h3_text_context": source_context,
            "shuffled_h3_text_token_tags": source_tags,
            "visual_shuffle_source_id": source_id,
        }


def _pad(items: list[dict[str, Any]], key: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = int(items[0][key].shape[1])
    tokens = max(int(item[key].shape[0]) for item in items)
    result = torch.zeros(len(items), tokens, width, dtype=torch.float32)
    mask = torch.zeros(len(items), tokens, dtype=torch.bool)
    for index, item in enumerate(items):
        value = item[key]
        result[index, : value.shape[0]] = value
        mask[index, : value.shape[0]] = True
    return result, mask


def collate_online(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("online pruned H3 evaluation requires batch size 1")
    text, text_mask = _pad(items, "text_context")
    replacement, replacement_mask = _pad(items, "replacement_text_context")
    shuffled_text, _ = _pad(items, "shuffled_h3_text_context")
    item = items[0]
    return {
        "sample_ids": [item["sample_id"]],
        "tasks": [item["task"]],
        "actions": item["actions"].unsqueeze(0),
        "raw_actions": item["raw_actions"].unsqueeze(0),
        "proprio": item["proprio"].unsqueeze(0),
        "action_is_pad": item["action_is_pad"].unsqueeze(0),
        "text_context": text,
        "text_mask": text_mask,
        "text_token_tags": item["text_token_tags"].unsqueeze(0),
        "replacement_text_context": replacement,
        "replacement_text_mask": replacement_mask,
        "current_h3_input": item["current_h3_input"].unsqueeze(0),
        "shuffled_h3_input": item["shuffled_h3_input"].unsqueeze(0),
        "shuffled_h3_text_context": shuffled_text,
        "shuffled_h3_text_token_tags": item[
            "shuffled_h3_text_token_tags"
        ].unsqueeze(0),
        "visual_shuffle_source_ids": [item["visual_shuffle_source_id"]],
    }


def move_batch(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "actions", "raw_actions", "proprio", "text_context",
        "replacement_text_context", "shuffled_h3_text_context",
    ):
        result[key] = batch[key].to(device=device, dtype=dtype)
    for key in ("current_h3_input", "shuffled_h3_input"):
        result[key] = batch[key].to(device=device, dtype=torch.float32)
    for key in ("text_token_tags", "shuffled_h3_text_token_tags"):
        result[key] = batch[key].to(device=device, dtype=torch.long)
    for key in ("action_is_pad", "text_mask", "replacement_text_mask"):
        result[key] = batch[key].to(device=device)
    return result


def _provider_batch(batch: dict[str, Any], *, shuffled: bool) -> dict[str, Any]:
    if shuffled:
        return {
            "current_h3_input": batch["shuffled_h3_input"],
            "text_context": batch["shuffled_h3_text_context"],
            "text_token_tags": batch["shuffled_h3_text_token_tags"],
        }
    return {
        "current_h3_input": batch["current_h3_input"],
        "text_context": batch["text_context"],
        "text_token_tags": batch["text_token_tags"],
    }


def restore_model(
    model_spec: dict[str, Any], model_state: dict[str, Any],
    *, device: torch.device, dtype: torch.dtype,
) -> nn.Module:
    normalized = dict(model_spec)
    normalized["carrier_layers"] = tuple(normalized["carrier_layers"])
    model = C58.build_model(
        C58.ModelSpec(**normalized), device=device, dtype=dtype,
        gradient_checkpointing=False,
    )
    model.load_state_dict(model_state, strict=True)
    return model.eval()


@torch.no_grad()
def sample_action_flow(
    model: nn.Module,
    batch: dict[str, Any],
    cache: dict[int, dict[str, torch.Tensor]],
    scheduler: Any,
    *,
    initial_noise: torch.Tensor,
    inference_steps: int,
    replacement_language: bool = False,
) -> torch.Tensor:
    actions = initial_noise.clone()
    timesteps, deltas = scheduler.build_inference_schedule(
        inference_steps, actions.device, actions.dtype
    )
    text_key = "replacement_text_context" if replacement_language else "text_context"
    mask_key = "replacement_text_mask" if replacement_language else "text_mask"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if actions.device.type == "cuda" and actions.dtype == torch.bfloat16
        else nullcontext()
    )
    for timestep, delta in zip(timesteps, deltas, strict=True):
        with autocast:
            velocity = model(
                actions,
                timestep.expand(actions.shape[0]),
                text_context=batch[text_key],
                proprio=batch["proprio"],
                video_kv_cache=cache,
                text_mask=batch[mask_key],
            )
        actions = scheduler.step(velocity, delta, actions)
    return actions


def _require_ready_and_checkpoint(
    config: EvalConfig,
    *,
    source_sha: str,
    source_items: int,
    train_sha: str,
    train_items: int,
    stats_sha: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    ready = json.loads(config.ready.read_text(encoding="utf-8"))
    checkpoint = config.checkpoint.resolve()
    if (
        ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
        or Path(ready.get("checkpoint", "")).resolve() != checkpoint
        or ready.get("restore_probe_max_abs") != 0.0
    ):
        raise ValueError("C58b long-run READY identity/restore gate failed")
    payload, checkpoint_sha = PROTOCOL._load_checkpoint(checkpoint)
    if ready.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("checkpoint bytes differ from strict READY")
    if set(payload) != set(C58.CHECKPOINT_KEYS):
        raise ValueError("C58b checkpoint top-level schema mismatch")
    if payload.get("schema_version") != C58.CHECKPOINT_SCHEMA:
        raise ValueError("C58b checkpoint schema mismatch")
    if payload.get("completed_steps") != EXPECTED_STEPS:
        raise ValueError("balanced80 requires the exact s10000 checkpoint")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("C58b checkpoint contract missing")
    required = {
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "classification": "action-only-on-frozen-layerwise-h3-kv_backbone_port",
        "fastwam_commit": EXPECTED_FASTWAM_COMMIT,
        "d0_parent_sha256": EXPECTED_D0_SHA256,
        "d0_parent_completed_steps": 14_000,
        "d0_parent_optimizer_restored": False,
        "carrier_source_mode": C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
        "h3_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "verify_h3_checkpoint_sha256": True,
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "disk_kv_training_input": False,
        "kv_subdir": None,
        "kv_layers": list(LAYERS),
        "kv_tokens": 32,
        "source_manifest_sha256": source_sha,
        "source_manifest_items": source_items,
        "split_manifest_sha256": train_sha,
        "split_manifest_items": train_items,
        "stats_sha256": stats_sha,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
        "action_horizon": 32,
        "action_shift": 5.0,
        "action_block_to_h3_layer": list(LAYERS),
    }
    mismatches = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in required.items() if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"C58b evaluation contract mismatch: {mismatches}")
    spec = contract.get("model_spec")
    if not isinstance(spec, dict) or set(spec) != MODEL_SPEC_KEYS:
        raise ValueError("C58b model_spec schema mismatch")
    normalized_spec = dict(spec)
    normalized_spec["carrier_layers"] = tuple(normalized_spec["carrier_layers"])
    expected_spec = asdict(C58.ModelSpec(
        carrier_layers=LAYERS,
        carrier_source_mode=C58.LAYERWISE_H3_50_TO_ACTION_30_MODE,
    ))
    if normalized_spec != expected_spec:
        raise ValueError("C58b model_spec differs from exact full30 layerwise tower")
    h3_path = config.h3_checkpoint.resolve()
    if Path(contract.get("h3_checkpoint_path", "")).resolve() != h3_path:
        raise ValueError("requested H3 checkpoint path differs from training")
    if PROTOCOL.sha256_file(h3_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("H3 checkpoint SHA256 mismatch")
    return payload, checkpoint_sha, ready


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if (
        config.batch_size != 1 or config.samples_per_task != 2
        or config.inference_steps != 10 or config.action_shift != 5.0
        or config.seed != 42 or config.num_workers != 0
    ):
        raise ValueError("C58b balanced80 protocol is fixed")
    source_rows = PROTOCOL.read_jsonl(config.source_manifest)
    train_rows = PROTOCOL.read_jsonl(config.train_manifest)
    val_rows = PROTOCOL.read_jsonl(config.val_manifest)
    split_audit = PROTOCOL.validate_episode_disjoint_manifests(
        source_rows, train_rows, val_rows
    )
    selected, selection = PROTOCOL.select_validation_rows(
        val_rows, samples_per_task=2
    )
    if len(selected) != EXPECTED_ITEMS or selection[
        "selected_ids_sha256"
    ] != EXPECTED_SELECTED_IDS_SHA256:
        raise ValueError("balanced80 selected-ID gate mismatch")
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected)
    hashes = {
        "source_manifest_sha256": PROTOCOL.sha256_file(config.source_manifest),
        "train_manifest_sha256": PROTOCOL.sha256_file(config.train_manifest),
        "validation_manifest_sha256": PROTOCOL.sha256_file(config.val_manifest),
        "stats_sha256": PROTOCOL.sha256_file(config.cache_root / "stats.pt"),
    }
    payload, checkpoint_sha, ready = _require_ready_and_checkpoint(
        config,
        source_sha=hashes["source_manifest_sha256"],
        source_items=len(source_rows),
        train_sha=hashes["train_manifest_sha256"],
        train_items=len(train_rows),
        stats_sha=hashes["stats_sha256"],
    )
    dataset = OnlineC58bValidationDataset(
        selected, cache_root=config.cache_root,
        action_horizon=32, visual_mapping=visual_mapping,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_online,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    if device.type != "cuda":
        raise ValueError("online INT8 H3 balanced80 requires CUDA")
    provider = C58OnlineFrozenH3Provider(config.h3_checkpoint, layers=LAYERS).to(
        device=device
    ).eval()
    scheduler = PROTOCOL.FlowMatchScheduler(
        num_train_timesteps=1000, shift=5.0
    )
    first_batch = move_batch(next(iter(loader)), device, dtype)
    first_cache = provider(_provider_batch(first_batch, shuffled=False))
    restore_predictions: list[torch.Tensor] = []
    restored_model: nn.Module | None = None
    restore_noise = PROTOCOL.deterministic_noise_like(
        first_batch["actions"], config.seed + 9_000_001
    )
    for _ in range(2):
        restored_model = restore_model(
            payload["contract"]["model_spec"], payload["model"],
            device=device, dtype=dtype,
        )
        restore_predictions.append(sample_action_flow(
            restored_model, first_batch, first_cache, scheduler,
            initial_noise=restore_noise, inference_steps=10,
        ).float().cpu())
        gc.collect()
        torch.cuda.empty_cache()
    restore_max_abs = float(
        (restore_predictions[0] - restore_predictions[1]).abs().max()
    )
    if restore_max_abs != 0.0 or restored_model is None:
        raise RuntimeError(f"fresh restore mismatch: {restore_max_abs}")
    model = restored_model

    normalized = PROTOCOL.DomainMetricAccumulator(7)
    physical = PROTOCOL.DomainMetricAccumulator(7)
    gripper = PROTOCOL.GripperSignAccumulator(6)
    language = PROTOCOL.LanguageSensitivityAccumulator()
    shuffled_normalized = PROTOCOL.DomainMetricAccumulator(7)
    shuffled_physical = PROTOCOL.DomainMetricAccumulator(7)
    shuffled_gripper = PROTOCOL.GripperSignAccumulator(6)
    normalized_delta = PROTOCOL.DomainMetricAccumulator(7)
    physical_delta = PROTOCOL.DomainMetricAccumulator(7)
    evaluated_ids: list[str] = []
    evaluated_pairs: list[str] = []
    evaluated_tasks: Counter[str] = Counter()
    per_sample_seconds: list[float] = []
    for batch_index, cpu_batch in enumerate(loader):
        sample_started = time.perf_counter()
        batch = move_batch(cpu_batch, device, dtype)
        baseline_cache = provider(_provider_batch(batch, shuffled=False))
        shuffled_cache = provider(_provider_batch(batch, shuffled=True))
        noise = PROTOCOL.deterministic_noise_like(
            batch["actions"], config.seed + 1_000_003 * batch_index
        )
        prediction = sample_action_flow(
            model, batch, baseline_cache, scheduler,
            initial_noise=noise, inference_steps=10,
        )
        shuffled = sample_action_flow(
            model, batch, shuffled_cache, scheduler,
            initial_noise=noise, inference_steps=10,
        )
        replacement = sample_action_flow(
            model, batch, baseline_cache, scheduler,
            initial_noise=noise, inference_steps=10, replacement_language=True,
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
        evaluated_ids.extend(cpu_batch["sample_ids"])
        evaluated_tasks.update(cpu_batch["tasks"])
        evaluated_pairs.extend(
            f"{target}\0{source}" for target, source in zip(
                cpu_batch["sample_ids"], cpu_batch["visual_shuffle_source_ids"],
                strict=True,
            )
        )
        per_sample_seconds.append(time.perf_counter() - sample_started)

    ids_sha = PROTOCOL.sha256_strings(evaluated_ids)
    pairs_sha = PROTOCOL.sha256_strings(evaluated_pairs)
    if ids_sha != EXPECTED_SELECTED_IDS_SHA256:
        raise RuntimeError("evaluated IDs differ from selected IDs")
    if pairs_sha != visual_contract["ordered_mapping_sha256"]:
        raise RuntimeError("evaluated visual shuffle mapping drifted")
    if dict(sorted(evaluated_tasks.items())) != selection["task_counts"]:
        raise RuntimeError("evaluated task counts drifted")
    normalized_report = normalized.finalize()
    physical_report = physical.finalize()
    gripper_report = gripper.finalize()
    shuffled_normalized_report = shuffled_normalized.finalize()
    shuffled_physical_report = shuffled_physical.finalize()
    shuffled_gripper_report = shuffled_gripper.finalize()
    report = {
        "format": "h3wam-c58b-online-h3-balanced80-v1",
        "event": "h3_c58b_online_h3_fastwam_full30_balanced80",
        "status": "completed_offline_not_closed_loop_evidence",
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "completed_steps": payload["completed_steps"],
            "strict_long_ready_path": str(config.ready.resolve()),
            "strict_long_ready_sha256": PROTOCOL.sha256_file(config.ready),
            "fresh_restore": {
                "strict_state_dict": True,
                "independent_model_instances": 2,
                "same_noise": True,
                "max_abs": restore_max_abs,
                "sample_ids": first_batch["sample_ids"],
            },
        },
        "contract": payload["contract"],
        "execution": {
            "h3": "online_frozen_int8",
            "h3_checkpoint": str(config.h3_checkpoint.resolve()),
            "h3_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "disk_kv_read": False,
            "disk_feature_read": False,
            "input_artifacts": ["windows/first_frame_latents", "contexts", "stats"],
            "carrier_layers": list(LAYERS),
            "carrier_mapping": "one_to_one_uniform_h3_50_to_action30",
            "visual_shuffle": "rerun_online_h3_on_source_latent_and_source_context",
            "language_replacement": "action_text_only_keep_baseline_h3_kv",
        },
        "data": {
            **hashes,
            "source_manifest_items": len(source_rows),
            "train_manifest_items": len(train_rows),
            "validation_manifest_items": len(val_rows),
            "selected_sample_ids_sha256": ids_sha,
            "selection": selection,
            "split_audit": split_audit,
        },
        "inference": {
            "scheduler": "pinned StarWAM FlowMatchScheduler",
            "shift": 5.0,
            "steps": 10,
            "integrator": "Euler sample += velocity * delta",
            "seed": 42,
            "batch_size": 1,
            "same_noise_for_baseline_language_visual": True,
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
        "timing": {
            "elapsed_seconds": time.perf_counter() - started,
            "mean_seconds_per_sample": sum(per_sample_seconds) / len(per_sample_seconds),
        },
        "claim_boundary": (
            "Measures held-out action regression and causal sensitivity with online "
            "H3; it does not prove LIBERO closed-loop success."
        ),
    }
    output = config.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    return EvalConfig(**vars(args))


if __name__ == "__main__":
    print(json.dumps(run_evaluation(parse_args()), indent=2), flush=True)
