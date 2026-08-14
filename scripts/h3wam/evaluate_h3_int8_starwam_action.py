#!/usr/bin/env python3
"""Strict offline evaluator for cached-feature H3 + pinned StarWAM ActionDiT.

The evaluator is intentionally separate from the R1 trainer.  It restores a
schema-2 checkpoint into a freshly constructed model, verifies an
episode-disjoint validation split, samples actions with StarWAM's pinned
shift-5 FlowMatchScheduler for exactly ten Euler steps, and reports causal
chunk metrics.  H3 is never loaded: only completed ``last32`` feature caches
are accepted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.starwam_feature_action import (  # noqa: E402
    H3StarWAMFeatureActionPolicy,
    STARWAM_ACTION_DIT_SHA256,
    STARWAM_WAN_BLOCK_SHA256,
    _load_pinned_starwam_action_dit,
)


_load_pinned_starwam_action_dit()
from starwam.modules.scheduler import FlowMatchScheduler  # noqa: E402


CHECKPOINT_SCHEMA = 2
STARWAM_COMMIT = "cd76d96f273f81e228a05f40f9697fe2514e2356"
STARWAM_POLICY_SHA256 = "9d5630cf0a39a7124f0dd452fd6d5215277c532ed11ee25d467fa15a9a2657a1"
STARWAM_FEATURE_MODEL_SHA256 = "e61ff10c92355243b02701f80847f3d382e89a440a6fbd7718aed629c68a18b7"
STARWAM_SCHEDULER_SHA256 = "5f9df0c8be5380faf4cd61abb337b55eea68cc49c5bfcbdb8ee9b72a8e057796"
STARWAM_METRICS_SHA256 = "e7fba9d350403335ecfd8b3bc9068cb54581259a4ea1bc69a0f6dbd338edd84e"
FEATURE_STRATEGY = "starwam_adaptive_avg_pool1d_v1"
FEATURE_BACKBONE = "H3Int8FeatureBackbone"
FEATURE_QUANTIZATION = "int8_tensorwise_convrot"
FEATURE_LAYERS = (49,)
FEATURE_TOKENS = 32
FEATURE_TIMESTEP = 1.0
EXPECTED_ACTION_SHIFT = 5.0
EXPECTED_INFERENCE_STEPS = 10
EXPECTED_BALANCED_VAL_TASKS = 40
BALANCED_VAL_SELECTION_SALT = "h3-int8-starwam-balanced-val-v1"


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path | None = None
    feature_subdir: str | None = None
    limit: int = 0
    sample_offset: int = 0
    samples_per_task: int = 0
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 42
    inference_steps: int = EXPECTED_INFERENCE_STEPS
    action_shift: float = EXPECTED_ACTION_SHIFT
    language_sensitivity: bool = False


MODEL_SPEC_KEYS = {
    "action_dim",
    "proprio_dim",
    "h3_feature_dim",
    "context_dim",
    "hidden_dim",
    "ffn_dim",
    "num_heads",
    "attn_head_dim",
    "action_layers",
    "freq_dim",
    "max_seq_len",
    "gradient_checkpointing",
    "include_feature_timestep",
    "feature_timestep",
    "feature_input_scale",
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _balanced_sample_rank(sample_id: str) -> tuple[str, str]:
    digest = hashlib.sha256()
    digest.update(BALANCED_VAL_SELECTION_SALT.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(sample_id).encode("utf-8"))
    return digest.hexdigest(), str(sample_id)


def select_validation_rows(
    rows: list[dict[str, Any]],
    *,
    samples_per_task: int = 0,
    limit: int = 0,
    sample_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select validation rows under an explicit, reproducible contract.

    Balanced selection is independent of manifest row order: within each task,
    rows are ranked by SHA256(fixed salt + NUL + sample id).  It intentionally
    cannot be combined with the legacy positional limit/offset selection.
    """

    if not rows:
        raise ValueError("validation rows are empty")
    if samples_per_task < 0 or limit < 0 or sample_offset < 0:
        raise ValueError(
            "samples_per_task, limit and sample_offset must be non-negative"
        )
    if samples_per_task and (limit or sample_offset):
        raise ValueError(
            "samples_per_task is mutually exclusive with limit and sample_offset"
        )

    if samples_per_task:
        rows_by_task: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_task.setdefault(str(row["task"]), []).append(row)
        if len(rows_by_task) != EXPECTED_BALANCED_VAL_TASKS:
            raise ValueError(
                "balanced validation requires exactly "
                f"{EXPECTED_BALANCED_VAL_TASKS} tasks in the full frozen val "
                f"manifest, found {len(rows_by_task)}"
            )
        insufficient = {
            task: len(task_rows)
            for task, task_rows in sorted(rows_by_task.items())
            if len(task_rows) < samples_per_task
        }
        if insufficient:
            raise ValueError(
                "balanced validation has tasks with fewer than "
                f"{samples_per_task} samples: {insufficient}"
            )
        selected: list[dict[str, Any]] = []
        for task in sorted(rows_by_task):
            ranked = sorted(
                rows_by_task[task], key=lambda row: _balanced_sample_rank(row["id"])
            )
            selected.extend(ranked[:samples_per_task])
        mode = "deterministic_balanced_per_task"
        salt: str | None = BALANCED_VAL_SELECTION_SALT
    else:
        if sample_offset >= len(rows):
            raise ValueError("sample_offset must select a validation row")
        selected = rows[sample_offset:]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError("selected validation dataset is empty")
        mode = "manifest_order_slice"
        salt = None

    selected_ids = [str(row["id"]) for row in selected]
    task_counts = dict(sorted(Counter(str(row["task"]) for row in selected).items()))
    return selected, {
        "mode": mode,
        "salt": salt,
        "samples_per_task": samples_per_task or None,
        "required_task_count": (
            EXPECTED_BALANCED_VAL_TASKS if samples_per_task else None
        ),
        "selected_items": len(selected),
        "selected_task_count": len(task_counts),
        "task_counts": task_counts,
        "selected_ids_sha256": sha256_strings(selected_ids),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate window ids: {path}")
    return rows


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["dataset_root"]),
        str(row["suite"]),
        int(row["episode"]),
    )


def validate_episode_disjoint_manifests(
    source_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {str(row["id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source manifest contains duplicate window ids")
    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            source_row = source_by_id.get(str(row["id"]))
            if source_row is None or source_row != row:
                raise ValueError(
                    f"{split_name} row {row['id']} is not exact source-manifest provenance"
                )
    train_ids = {str(row["id"]) for row in train_rows}
    val_ids = {str(row["id"]) for row in val_rows}
    window_overlap = train_ids & val_ids
    if window_overlap:
        raise ValueError(f"train/validation window overlap: {sorted(window_overlap)[:5]}")
    train_episodes = {episode_key(row) for row in train_rows}
    val_episodes = {episode_key(row) for row in val_rows}
    episode_overlap = train_episodes & val_episodes
    if episode_overlap:
        raise ValueError(
            f"train/validation episode overlap: {sorted(episode_overlap)[:5]}"
        )
    return {
        "episode_key": ["dataset_root", "suite", "episode"],
        "source_windows": len(source_rows),
        "train_windows": len(train_rows),
        "validation_windows": len(val_rows),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "window_overlap": 0,
        "episode_overlap": 0,
        "validation_tasks": len({str(row["task"]) for row in val_rows}),
        "validation_suites": dict(
            sorted(Counter(str(row["suite"]) for row in val_rows).items())
        ),
    }


def normalize_minmax(
    values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    value_range = (upper.to(values) - lower.to(values)).clamp_min(1.0e-6)
    return (2.0 * (values - lower.to(values)) / value_range - 1.0).clamp(
        -5.0, 5.0
    )


def denormalize_minmax_official(
    values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Match StarWAM policy min-max denormalization, including [-1,1] clamp."""

    value_range = (upper.to(values) - lower.to(values)).clamp_min(1.0e-6)
    return (values.clamp(-1.0, 1.0) + 1.0) * 0.5 * value_range + lower.to(
        values
    )


def _pad_contexts(
    items: list[dict[str, Any]], key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = max(int(item[key].shape[0]) for item in items)
    width = int(items[0][key].shape[1])
    context = torch.zeros(len(items), max_tokens, width, dtype=torch.float32)
    mask = torch.zeros(len(items), max_tokens, dtype=torch.bool)
    for index, item in enumerate(items):
        tokens = item[key]
        if tokens.ndim != 2 or int(tokens.shape[1]) != width:
            raise ValueError(f"inconsistent {key} shape")
        context[index, : tokens.shape[0]] = tokens
        mask[index, : tokens.shape[0]] = True
    return context, mask


def collate_eval_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    context, context_mask = _pad_contexts(items, "text_context")
    replacement, replacement_mask = _pad_contexts(
        items, "replacement_text_context"
    )
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "tasks": [str(item["task"]) for item in items],
        "features": torch.stack([item["features"] for item in items]),
        "actions": torch.stack([item["actions"] for item in items]),
        "raw_actions": torch.stack([item["raw_actions"] for item in items]),
        "proprio": torch.stack([item["proprio"] for item in items]),
        "action_is_pad": torch.stack([item["action_is_pad"] for item in items]),
        "text_context": context,
        "text_mask": context_mask,
        "replacement_text_context": replacement,
        "replacement_text_mask": replacement_mask,
    }


class CachedLast32ValidationDataset(Dataset):
    """Read held-out cached H3 last32 features under the checkpoint contract."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_root: Path,
        feature_subdir: str,
        source_manifest_items: int,
        model_spec: dict[str, Any],
        action_horizon: int,
        limit: int = 0,
        sample_offset: int = 0,
    ) -> None:
        if sample_offset < 0 or sample_offset >= len(rows):
            raise ValueError("sample_offset must select a validation row")
        selected = rows[sample_offset:]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError("selected validation dataset is empty")
        self.rows = selected
        self.cache_root = Path(cache_root).resolve()
        self.feature_root = self.cache_root / feature_subdir
        self.source_manifest_items = int(source_manifest_items)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(model_spec["action_dim"])
        self.proprio_dim = int(model_spec["proprio_dim"])
        self.h3_feature_dim = int(model_spec["h3_feature_dim"])
        self.context_dim = int(model_spec["context_dim"])
        stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        if tuple(self.action_min.shape) != (self.action_dim,) or tuple(
            self.action_max.shape
        ) != (self.action_dim,):
            raise ValueError("action stats do not match checkpoint action_dim")
        if tuple(self.state_min.shape) != (self.proprio_dim,) or tuple(
            self.state_max.shape
        ) != (self.proprio_dim,):
            raise ValueError("state stats do not match checkpoint proprio_dim")

        context_ids = sorted({str(row["context_id"]) for row in rows})
        self.replacement_context: dict[str, str] = {}
        for context_id in context_ids:
            replacement = next(
                (candidate for candidate in context_ids if candidate != context_id),
                context_id,
            )
            self.replacement_context[context_id] = replacement

    def __len__(self) -> int:
        return len(self.rows)

    def _load_context(self, context_id: str) -> torch.Tensor:
        path = self.cache_root / "contexts" / f"{context_id}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("text_only") is not True:
            raise ValueError(f"context is not text-only: {context_id}")
        tags = payload.get("token_tags")
        if tags is not None and torch.any(tags != 1):
            raise ValueError(f"context contains non-text tags: {context_id}")
        context = payload["context"]
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError(f"unexpected context shape for {context_id}: {context.shape}")
        context = context[0].float()
        if tuple(context.shape[1:]) != (self.context_dim,):
            raise ValueError(
                f"context width mismatch for {context_id}: {context.shape[1]}"
            )
        return context

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        context_id = str(row["context_id"])
        feature_path = self.feature_root / f"{sample_id}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing completed feature cache: {feature_path}")
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        expected = {
            "layers": FEATURE_LAYERS,
            "context_id": context_id,
            "action_horizon": self.action_horizon,
            "capture_token_count": FEATURE_TOKENS,
            "capture_token_strategy": FEATURE_STRATEGY,
            "backbone": FEATURE_BACKBONE,
            "quantization": FEATURE_QUANTIZATION,
            "manifest_items": self.source_manifest_items,
        }
        for key, expected_value in expected.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"feature cache contract mismatch for {sample_id}: "
                    f"{key}={actual!r}, expected {expected_value!r}"
                )
        if not math.isclose(float(payload.get("timestep", -1.0)), FEATURE_TIMESTEP):
            raise ValueError(f"feature cache timestep must be 1.0 for {sample_id}")
        features = payload["features"]
        expected_shape = (1, FEATURE_TOKENS, self.h3_feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"unexpected last32 feature shape for {sample_id}: {features.shape}"
            )
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"non-finite feature cache for {sample_id}")

        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        raw_actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(raw_actions.shape) != (self.action_horizon, self.action_dim):
            raise ValueError(f"unexpected action shape for {sample_id}: {raw_actions.shape}")
        if bool((~action_is_pad).sum() == 0):
            raise ValueError(f"fully padded action chunk: {sample_id}")
        state = window["state"].float()
        if tuple(state.shape) != (self.proprio_dim,):
            raise ValueError(f"unexpected proprio shape for {sample_id}: {state.shape}")
        replacement_id = self.replacement_context[context_id]
        return {
            "sample_id": sample_id,
            "task": str(row["task"]),
            "features": features,
            "actions": normalize_minmax(
                raw_actions, self.action_min, self.action_max
            ),
            "raw_actions": raw_actions,
            "proprio": normalize_minmax(state, self.state_min, self.state_max),
            "action_is_pad": action_is_pad,
            "text_context": self._load_context(context_id),
            "replacement_text_context": self._load_context(replacement_id),
        }


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
    expected_top_level = {
        "schema_version",
        "completed_steps",
        "model",
        "optimizer",
        "lr_scheduler",
        "contract",
        "probe_prediction",
        "probe_sample_ids",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "checkpoint top-level schema mismatch: "
            f"missing={sorted(expected_top_level - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_top_level)}"
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch: expected schema_version=2")
    if int(payload.get("completed_steps", -1)) < 0:
        raise ValueError("checkpoint completed_steps must be non-negative")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("checkpoint model state is missing or empty")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint contract is missing")
    required_contract = {
        "starwam_commit": STARWAM_COMMIT,
        "starwam_action_dit_sha256": STARWAM_ACTION_DIT_SHA256,
        "starwam_wan_block_sha256": STARWAM_WAN_BLOCK_SHA256,
        "feature_strategy": FEATURE_STRATEGY,
        "feature_layers": list(FEATURE_LAYERS),
        "feature_tokens": FEATURE_TOKENS,
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
        for key, expected in required_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint evaluation contract mismatch: {mismatches}")
    if not math.isclose(float(contract.get("action_shift", -1.0)), config.action_shift):
        raise ValueError("checkpoint action_shift does not match evaluator shift")
    if not math.isclose(config.action_shift, EXPECTED_ACTION_SHIFT):
        raise ValueError("R1 offline evaluation is fixed to action shift 5")
    if config.inference_steps != EXPECTED_INFERENCE_STEPS:
        raise ValueError("R1 offline evaluation is fixed to 10 inference steps")
    model_spec = contract.get("model_spec")
    if not isinstance(model_spec, dict) or set(model_spec) != MODEL_SPEC_KEYS:
        raise ValueError(
            "checkpoint model_spec schema mismatch: "
            f"missing={sorted(MODEL_SPEC_KEYS - set(model_spec or {}))}, "
            f"unexpected={sorted(set(model_spec or {}) - MODEL_SPEC_KEYS)}"
        )
    action_horizon = int(contract.get("action_horizon", -1))
    if action_horizon <= 0 or action_horizon > int(model_spec["max_seq_len"]):
        raise ValueError("checkpoint action_horizon is invalid for model_spec")
    feature_subdir = str(contract.get("feature_subdir", ""))
    if not feature_subdir:
        raise ValueError("checkpoint feature_subdir is missing")
    if config.feature_subdir is not None and config.feature_subdir != feature_subdir:
        raise ValueError("requested feature_subdir differs from checkpoint contract")
    return contract, model_spec, feature_subdir


def build_model_from_spec(
    model_spec: dict[str, Any], *, device: torch.device, dtype: torch.dtype
) -> H3StarWAMFeatureActionPolicy:
    return H3StarWAMFeatureActionPolicy(
        action_dim=int(model_spec["action_dim"]),
        proprio_dim=int(model_spec["proprio_dim"]),
        h3_feature_dim=int(model_spec["h3_feature_dim"]),
        context_dim=int(model_spec["context_dim"]),
        hidden_dim=int(model_spec["hidden_dim"]),
        ffn_dim=int(model_spec["ffn_dim"]),
        num_heads=int(model_spec["num_heads"]),
        attn_head_dim=int(model_spec["attn_head_dim"]),
        num_layers=int(model_spec["action_layers"]),
        freq_dim=int(model_spec["freq_dim"]),
        max_seq_len=int(model_spec["max_seq_len"]),
        use_gradient_checkpointing=bool(model_spec["gradient_checkpointing"]),
        include_feature_timestep=bool(model_spec["include_feature_timestep"]),
        feature_timestep=float(model_spec["feature_timestep"]),
        feature_input_scale=float(model_spec["feature_input_scale"]),
    ).to(device=device, dtype=dtype)


def restore_model_strict(
    model_spec: dict[str, Any],
    model_state: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> H3StarWAMFeatureActionPolicy:
    model = build_model_from_spec(model_spec, device=device, dtype=dtype)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def move_batch(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "features",
        "actions",
        "raw_actions",
        "proprio",
        "text_context",
        "replacement_text_context",
    ):
        result[key] = batch[key].to(device=device, dtype=dtype)
    for key in ("action_is_pad", "text_mask", "replacement_text_mask"):
        result[key] = batch[key].to(device=device)
    return result


@torch.no_grad()
def sample_action_flow(
    model: nn.Module,
    batch: dict[str, Any],
    scheduler: FlowMatchScheduler,
    *,
    inference_steps: int,
    initial_noise: torch.Tensor,
    replacement_language: bool = False,
) -> torch.Tensor:
    actions = initial_noise.clone()
    timesteps, deltas = scheduler.build_inference_schedule(
        inference_steps, actions.device, actions.dtype
    )
    text_key = (
        "replacement_text_context" if replacement_language else "text_context"
    )
    mask_key = "replacement_text_mask" if replacement_language else "text_mask"
    for timestep, delta in zip(timesteps, deltas, strict=True):
        velocity = model(
            actions,
            timestep.expand(actions.shape[0]),
            text_context=batch[text_key],
            h3_features=batch["features"],
            proprio=batch["proprio"],
            text_mask=batch[mask_key],
        )
        actions = scheduler.step(velocity, delta, actions)
    return actions


def deterministic_noise_like(actions: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=actions.device).manual_seed(int(seed))
    return torch.randn(
        actions.shape,
        device=actions.device,
        dtype=actions.dtype,
        generator=generator,
    )


class DomainMetricAccumulator:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)
        self.element_count = 0
        self.step_count = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.dim_squared_error = torch.zeros(action_dim, dtype=torch.float64)
        self.dim_absolute_error = torch.zeros(action_dim, dtype=torch.float64)
        self.dim_count = torch.zeros(action_dim, dtype=torch.float64)
        self.pred_sum = torch.zeros(action_dim, dtype=torch.float64)
        self.pred_square_sum = torch.zeros(action_dim, dtype=torch.float64)
        self.target_sum = torch.zeros(action_dim, dtype=torch.float64)
        self.target_square_sum = torch.zeros(action_dim, dtype=torch.float64)
        self.ade_sum = 0.0
        self.endpoint_sum = 0.0
        self.endpoint_count = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor
    ) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().float().cpu()
        is_pad = is_pad.detach().bool().cpu()
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("metric actions must be matching [B,T,D] tensors")
        if prediction.shape[-1] != self.action_dim or tuple(is_pad.shape) != tuple(
            prediction.shape[:2]
        ):
            raise ValueError("metric action dimension or padding shape mismatch")
        valid = ~is_pad
        if not bool(valid.any()):
            raise ValueError("metric batch has no valid action steps")
        pred_valid = prediction[valid].double()
        target_valid = target[valid].double()
        diff = pred_valid - target_valid
        self.step_count += int(pred_valid.shape[0])
        self.element_count += int(pred_valid.numel())
        self.squared_error += float(diff.square().sum())
        self.absolute_error += float(diff.abs().sum())
        self.dim_squared_error += diff.square().sum(dim=0)
        self.dim_absolute_error += diff.abs().sum(dim=0)
        self.dim_count += float(pred_valid.shape[0])
        self.pred_sum += pred_valid.sum(dim=0)
        self.pred_square_sum += pred_valid.square().sum(dim=0)
        self.target_sum += target_valid.sum(dim=0)
        self.target_square_sum += target_valid.square().sum(dim=0)
        self.ade_sum += float(torch.linalg.vector_norm(diff, dim=-1).sum())
        for batch_index in range(prediction.shape[0]):
            valid_indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            endpoint = int(valid_indices[-1])
            self.endpoint_sum += float(
                torch.linalg.vector_norm(
                    prediction[batch_index, endpoint].double()
                    - target[batch_index, endpoint].double()
                )
            )
            self.endpoint_count += 1

    def finalize(self) -> dict[str, Any]:
        if self.element_count <= 0 or self.step_count <= 0 or self.endpoint_count <= 0:
            raise ValueError("no metric observations accumulated")
        dim_count = self.dim_count.clamp_min(1.0)
        pred_mean = self.pred_sum / dim_count
        target_mean = self.target_sum / dim_count
        pred_variance = self.pred_square_sum / dim_count - pred_mean.square()
        target_variance = self.target_square_sum / dim_count - target_mean.square()
        global_pred_sum = float(self.pred_sum.sum())
        global_pred_square_sum = float(self.pred_square_sum.sum())
        global_count = float(self.element_count)
        global_variance = max(
            global_pred_square_sum / global_count
            - (global_pred_sum / global_count) ** 2,
            0.0,
        )
        return {
            "valid_steps": self.step_count,
            "valid_elements": self.element_count,
            "action_mse": self.squared_error / self.element_count,
            "action_mae": self.absolute_error / self.element_count,
            "action_mse_per_dim": (self.dim_squared_error / dim_count).tolist(),
            "action_mae_per_dim": (self.dim_absolute_error / dim_count).tolist(),
            "chunk_ade_l2": self.ade_sum / self.step_count,
            "chunk_endpoint_l2": self.endpoint_sum / self.endpoint_count,
            "prediction_mean_per_dim": pred_mean.tolist(),
            "prediction_std_per_dim": pred_variance.clamp_min(0.0).sqrt().tolist(),
            "prediction_std": math.sqrt(global_variance),
            "target_mean_per_dim": target_mean.tolist(),
            "target_std_per_dim": target_variance.clamp_min(0.0).sqrt().tolist(),
        }


class GripperSignAccumulator:
    def __init__(self, gripper_dim: int) -> None:
        self.gripper_dim = int(gripper_dim)
        self.tp = self.tn = self.fp = self.fn = 0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor
    ) -> None:
        valid = ~is_pad.detach().bool().cpu()
        pred = prediction.detach().float().cpu()[..., self.gripper_dim][valid] >= 0
        truth = target.detach().float().cpu()[..., self.gripper_dim][valid] >= 0
        self.tp += int((pred & truth).sum())
        self.tn += int((~pred & ~truth).sum())
        self.fp += int((pred & ~truth).sum())
        self.fn += int((~pred & truth).sum())

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def finalize(self) -> dict[str, Any]:
        total = self.tp + self.tn + self.fp + self.fn
        precision = self._ratio(self.tp, self.tp + self.fp)
        recall = self._ratio(self.tp, self.tp + self.fn)
        f1 = self._ratio(2 * precision * recall, precision + recall)
        negative_precision = self._ratio(self.tn, self.tn + self.fn)
        negative_recall = self._ratio(self.tn, self.tn + self.fp)
        negative_f1 = self._ratio(
            2 * negative_precision * negative_recall,
            negative_precision + negative_recall,
        )
        return {
            "positive_semantics": "normalized gripper >= 0",
            "accuracy": self._ratio(self.tp + self.tn, total),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": 0.5 * (f1 + negative_f1),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }


class LanguageSensitivityAccumulator:
    def __init__(self) -> None:
        self.absolute_delta = 0.0
        self.squared_delta = 0.0
        self.reference_square = 0.0
        self.element_count = 0
        self.step_l2 = 0.0
        self.step_count = 0

    def update(
        self,
        reference: torch.Tensor,
        replacement: torch.Tensor,
        is_pad: torch.Tensor,
    ) -> None:
        valid = ~is_pad.detach().bool().cpu()
        reference = reference.detach().float().cpu()
        replacement = replacement.detach().float().cpu()
        delta = reference[valid].double() - replacement[valid].double()
        self.absolute_delta += float(delta.abs().sum())
        self.squared_delta += float(delta.square().sum())
        self.reference_square += float(reference[valid].double().square().sum())
        self.element_count += int(delta.numel())
        self.step_l2 += float(torch.linalg.vector_norm(delta, dim=-1).sum())
        self.step_count += int(delta.shape[0])

    def finalize(self) -> dict[str, Any]:
        if self.element_count == 0:
            raise ValueError("no language sensitivity observations")
        return {
            "same_noise": True,
            "mean_abs_prediction_delta": self.absolute_delta / self.element_count,
            "rms_prediction_delta": math.sqrt(
                self.squared_delta / self.element_count
            ),
            "mean_step_l2_prediction_delta": self.step_l2 / self.step_count,
            "relative_l2_prediction_delta": math.sqrt(
                self.squared_delta / max(self.reference_square, 1.0e-30)
            ),
        }


def _resolve_device_dtype(device_text: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return device, dtype


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload, checkpoint_sha256


def _fresh_restore_check(
    *,
    model_spec: dict[str, Any],
    model_state: dict[str, Any],
    first_batch: dict[str, Any],
    scheduler: FlowMatchScheduler,
    config: EvalConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, Any]]:
    noise = deterministic_noise_like(first_batch["actions"], config.seed + 9_000_001)
    first_model = restore_model_strict(
        model_spec, model_state, device=device, dtype=dtype
    )
    with torch.no_grad():
        first_prediction = sample_action_flow(
            first_model,
            first_batch,
            scheduler,
            inference_steps=config.inference_steps,
            initial_noise=noise,
        ).float().cpu()
    del first_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    second_model = restore_model_strict(
        model_spec, model_state, device=device, dtype=dtype
    )
    with torch.no_grad():
        second_prediction = sample_action_flow(
            second_model,
            first_batch,
            scheduler,
            inference_steps=config.inference_steps,
            initial_noise=noise,
        ).float().cpu()
    max_abs = float((first_prediction - second_prediction).abs().max())
    if max_abs != 0.0:
        raise RuntimeError(f"fresh restore prediction mismatch: max_abs={max_abs}")
    return second_model, {
        "strict_state_dict": True,
        "independent_model_instances": 2,
        "same_noise": True,
        "max_abs": max_abs,
        "sample_ids": list(first_batch["sample_ids"]),
    }


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    source_manifest = config.source_manifest.resolve()
    train_manifest = config.train_manifest.resolve()
    val_manifest = config.val_manifest.resolve()
    cache_root = config.cache_root.resolve()
    source_rows = read_jsonl(source_manifest)
    train_rows = read_jsonl(train_manifest)
    val_rows = read_jsonl(val_manifest)
    split_audit = validate_episode_disjoint_manifests(
        source_rows, train_rows, val_rows
    )
    selected_val_rows, selection = select_validation_rows(
        val_rows,
        samples_per_task=config.samples_per_task,
        limit=config.limit,
        sample_offset=config.sample_offset,
    )
    hashes = {
        "source_manifest_sha256": sha256_file(source_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(val_manifest),
        "stats_sha256": sha256_file(cache_root / "stats.pt"),
    }
    payload, checkpoint_sha256 = _load_checkpoint(config.checkpoint)
    contract, model_spec, feature_subdir = _require_contract(
        payload,
        config=config,
        source_manifest_sha256=hashes["source_manifest_sha256"],
        source_manifest_items=len(source_rows),
        train_manifest_sha256=hashes["train_manifest_sha256"],
        train_manifest_items=len(train_rows),
        stats_sha256=hashes["stats_sha256"],
    )
    action_horizon = int(contract["action_horizon"])
    dataset = CachedLast32ValidationDataset(
        selected_val_rows,
        cache_root=cache_root,
        feature_subdir=feature_subdir,
        source_manifest_items=len(source_rows),
        model_spec=model_spec,
        action_horizon=action_horizon,
        limit=0,
        sample_offset=0,
    )
    if config.language_sensitivity and len(dataset.replacement_context) < 2:
        raise ValueError("language sensitivity requires at least two context ids")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_eval_batch,
    )
    device, dtype = _resolve_device_dtype(config.device)
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=config.action_shift)
    first_batch_cpu = next(iter(loader))
    first_batch = move_batch(first_batch_cpu, device, dtype)
    model, restore = _fresh_restore_check(
        model_spec=model_spec,
        model_state=payload["model"],
        first_batch=first_batch,
        scheduler=scheduler,
        config=config,
        device=device,
        dtype=dtype,
    )

    normalized_metrics = DomainMetricAccumulator(int(model_spec["action_dim"]))
    physical_metrics = DomainMetricAccumulator(int(model_spec["action_dim"]))
    gripper = GripperSignAccumulator(int(model_spec["action_dim"]) - 1)
    language = LanguageSensitivityAccumulator() if config.language_sensitivity else None
    evaluated_sample_ids: list[str] = []
    evaluated_tasks: Counter[str] = Counter()
    with torch.no_grad():
        for batch_index, batch_cpu in enumerate(loader):
            batch = move_batch(batch_cpu, device, dtype)
            noise = deterministic_noise_like(
                batch["actions"], config.seed + 1_000_003 * batch_index
            )
            prediction = sample_action_flow(
                model,
                batch,
                scheduler,
                inference_steps=config.inference_steps,
                initial_noise=noise,
            )
            normalized_metrics.update(
                prediction, batch["actions"], batch["action_is_pad"]
            )
            prediction_physical = denormalize_minmax_official(
                prediction,
                dataset.action_min.to(device=device, dtype=dtype),
                dataset.action_max.to(device=device, dtype=dtype),
            )
            physical_metrics.update(
                prediction_physical,
                batch["raw_actions"],
                batch["action_is_pad"],
            )
            gripper.update(prediction, batch["actions"], batch["action_is_pad"])
            if language is not None:
                replacement = sample_action_flow(
                    model,
                    batch,
                    scheduler,
                    inference_steps=config.inference_steps,
                    initial_noise=noise,
                    replacement_language=True,
                )
                language.update(prediction, replacement, batch["action_is_pad"])
            evaluated_sample_ids.extend(batch_cpu["sample_ids"])
            evaluated_tasks.update(batch_cpu["tasks"])

    normalized_report = normalized_metrics.finalize()
    physical_report = physical_metrics.finalize()
    evaluated_ids_sha256 = sha256_strings(evaluated_sample_ids)
    if evaluated_ids_sha256 != selection["selected_ids_sha256"]:
        raise RuntimeError("evaluated sample order differs from frozen selection")
    if dict(sorted(evaluated_tasks.items())) != selection["task_counts"]:
        raise RuntimeError("evaluated task counts differ from frozen selection")
    report = {
        "event": "h3_int8_starwam_offline_evaluation",
        "classification": "causal-action-on-frozen-cached-h3-features",
        "status": "completed_not_closed_loop_evidence",
        "checkpoint": {
            "path": str(config.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "schema_version": payload["schema_version"],
            "completed_steps": int(payload["completed_steps"]),
            "contract": contract,
            "fresh_restore": restore,
        },
        "source_identity": {
            "starwam_commit": STARWAM_COMMIT,
            "action_dit_sha256": STARWAM_ACTION_DIT_SHA256,
            "wan_block_sha256": STARWAM_WAN_BLOCK_SHA256,
            "policy_sha256": STARWAM_POLICY_SHA256,
            "feature_conditioned_model_sha256": STARWAM_FEATURE_MODEL_SHA256,
            "scheduler_sha256": STARWAM_SCHEDULER_SHA256,
            "official_metrics_sha256": STARWAM_METRICS_SHA256,
        },
        "data": {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": hashes["source_manifest_sha256"],
            "source_manifest_items": len(source_rows),
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": hashes["train_manifest_sha256"],
            "train_manifest_items": len(train_rows),
            "validation_manifest": str(val_manifest),
            "validation_manifest_sha256": hashes["validation_manifest_sha256"],
            "validation_manifest_items": len(val_rows),
            "selected_validation_items": len(dataset),
            "selected_sample_ids_sha256": evaluated_ids_sha256,
            "selected_tasks": len(evaluated_tasks),
            "selected_windows_by_task": dict(sorted(evaluated_tasks.items())),
            "selection": selection,
            "stats_path": str((cache_root / "stats.pt").resolve()),
            "stats_sha256": hashes["stats_sha256"],
            "feature_root": str(dataset.feature_root),
            "split_audit": split_audit,
        },
        "inference": {
            "causal_conditioning": "cached current-observation H3 layer49 last32 + text + proprio",
            "scheduler": "pinned StarWAM FlowMatchScheduler",
            "num_train_timesteps": 1000,
            "shift": config.action_shift,
            "steps": config.inference_steps,
            "integrator": "Euler sample += velocity * delta",
            "seed": config.seed,
            "batch_size": config.batch_size,
            "device": str(device),
            "dtype": str(dtype),
        },
        "metrics": {
            "normalized_clip5_model_domain": normalized_report,
            "denormalized_official_minmax_clamp": physical_report,
            "gripper_sign": gripper.finalize(),
            "language_replacement_sensitivity": (
                None if language is None else language.finalize()
            ),
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
    parser.add_argument("--feature-subdir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--samples-per-task", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=EXPECTED_INFERENCE_STEPS)
    parser.add_argument("--action-shift", type=float, default=EXPECTED_ACTION_SHIFT)
    parser.add_argument("--language-sensitivity", action="store_true")
    values = parser.parse_args()
    return EvalConfig(
        checkpoint=values.checkpoint,
        source_manifest=values.source_manifest,
        train_manifest=values.train_manifest,
        val_manifest=values.val_manifest,
        cache_root=values.cache_root,
        output=values.output,
        feature_subdir=values.feature_subdir,
        limit=values.limit,
        sample_offset=values.sample_offset,
        samples_per_task=values.samples_per_task,
        batch_size=values.batch_size,
        num_workers=values.num_workers,
        device=values.device,
        seed=values.seed,
        inference_steps=values.inference_steps,
        action_shift=values.action_shift,
        language_sensitivity=values.language_sensitivity,
    )


def main() -> None:
    report = run_evaluation(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
