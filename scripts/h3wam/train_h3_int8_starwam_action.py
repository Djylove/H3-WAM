#!/usr/bin/env python3
"""Train pinned StarWAM ActionDiT on cached standalone INT8-H3 features.

This is intentionally an action-only-on-frozen-features trainer.  H3 is not
loaded and cannot receive gradients here; every sample must carry the strict
metadata written by ``precompute_h3_int8_features.py``.  The action expert is
the byte-verified implementation from the pinned StarWAM submodule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.starwam_feature_action import (  # noqa: E402
    H3StarWAMFeatureActionPolicy,
    STARWAM_ACTION_DIT_SHA256,
    STARWAM_WAN_BLOCK_SHA256,
    _load_pinned_starwam_action_dit,
)

# Import through the namespace installed by the verified ActionDiT loader.
_load_pinned_starwam_action_dit()
from starwam.modules.scheduler import FlowMatchScheduler  # noqa: E402
from starwam.training.loss import flow_matching_loss  # noqa: E402


STARWAM_COMMIT = "cd76d96f273f81e228a05f40f9697fe2514e2356"
H3_INT8_CHECKPOINT_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
FEATURE_SCALE_AUDIT_SAMPLES = 512
FEATURE_GLOBAL_RMS = 7322.443
TEXT_CONTEXT_COUNT = 40
TEXT_CONTEXT_GLOBAL_RMS = 70.346
RMS_MATCH_FEATURE_INPUT_SCALE = 0.009606920816877307
CACHE_STRATEGY = "starwam_adaptive_avg_pool1d_v1"
CACHE_BACKBONE = "H3Int8FeatureBackbone"
CACHE_QUANTIZATION = "int8_tensorwise_convrot"
CHECKPOINT_SCHEMA = 2


@dataclass(frozen=True)
class ModelSpec:
    action_dim: int = 7
    proprio_dim: int = 8
    h3_feature_dim: int = 5376
    context_dim: int = 5120
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 40
    attn_head_dim: int = 128
    action_layers: int = 30
    freq_dim: int = 256
    max_seq_len: int = 64
    gradient_checkpointing: bool = True
    include_feature_timestep: bool = False
    feature_timestep: float = 0.0
    feature_input_scale: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Full cache-source manifest when manifest is a train/val subset.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--feature-subdir", default="h3_int8_libero40_last32_starwam"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-horizon", type=int, default=21700)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-layers", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--ffn-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument(
        "--clean-action-regression-weight",
        type=float,
        default=0.0,
        help=(
            "Candidate F only: add masked clean-action reconstruction MSE from "
            "the same flow prediction. Zero preserves the pinned R1 objective."
        ),
    )
    parser.add_argument(
        "--paired-visual-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Candidate G only: require the correct cached H3 observation to have "
            "lower weighted flow loss than a cross-sample H3 observation. Zero "
            "preserves the pinned R1 objective."
        ),
    )
    parser.add_argument(
        "--paired-visual-margin",
        type=float,
        default=0.05,
        help="Candidate G hinge margin in weighted flow-loss units.",
    )
    parser.add_argument("--feature-input-scale", type=float, default=1.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument("--verify-h3-checkpoint-sha256", action="store_true")
    parser.add_argument(
        "--expected-h3-checkpoint-sha256", default=H3_INT8_CHECKPOINT_SHA256
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_minmax(values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Match StarWAM's min/max normalization and deployment clipping range."""

    value_range = (upper.to(values) - lower.to(values)).clamp_min(1.0e-6)
    normalized = 2.0 * (values - lower.to(values)) / value_range - 1.0
    return normalized.clamp(-5.0, 5.0)


def _pad_contexts(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = max(int(item["text_context"].shape[0]) for item in items)
    width = int(items[0]["text_context"].shape[1])
    context = torch.zeros(len(items), max_tokens, width, dtype=torch.float32)
    mask = torch.zeros(len(items), max_tokens, dtype=torch.bool)
    for index, item in enumerate(items):
        tokens = item["text_context"]
        context[index, : tokens.shape[0]] = tokens
        mask[index, : tokens.shape[0]] = True
    return context, mask


def collate_cached_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    context, context_mask = _pad_contexts(items)
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "features": torch.stack([item["features"] for item in items]),
        "actions": torch.stack([item["actions"] for item in items]),
        "proprio": torch.stack([item["proprio"] for item in items]),
        "action_is_pad": torch.stack([item["action_is_pad"] for item in items]),
        "text_context": context,
        "text_mask": context_mask,
    }


class CachedLast32Dataset(Dataset):
    """Strict reader for H3 INT8 last-layer, 32-token observation caches."""

    def __init__(
        self,
        manifest: Path,
        cache_root: Path,
        feature_subdir: str,
        *,
        source_manifest: Path | None = None,
        action_horizon: int = 32,
        limit: int = 0,
        sample_offset: int = 0,
    ) -> None:
        self.manifest = manifest.resolve()
        self.cache_root = cache_root.resolve()
        self.feature_root = self.cache_root / feature_subdir
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        all_rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all_rows:
            raise ValueError("manifest is empty")
        if sample_offset < 0 or sample_offset >= len(all_rows):
            raise ValueError("sample_offset must select an existing manifest row")
        split_ids = [str(row["id"]) for row in all_rows]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("split manifest contains duplicate window ids")
        self.source_manifest = (
            self.manifest
            if source_manifest is None
            else source_manifest.resolve()
        )
        source_rows = (
            all_rows
            if self.source_manifest == self.manifest
            else [
                json.loads(line)
                for line in self.source_manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        source_by_id = {str(row["id"]): row for row in source_rows}
        if len(source_by_id) != len(source_rows):
            raise ValueError("source manifest contains duplicate window ids")
        for row in all_rows:
            source_row = source_by_id.get(str(row["id"]))
            if source_row is None or source_row != row:
                raise ValueError(
                    f"split row {row['id']} is not byte-equivalent JSON provenance from source manifest"
                )
        selected = all_rows[sample_offset:]
        if limit:
            selected = selected[:limit]
        if not selected:
            raise ValueError("selected dataset is empty")
        self.rows = selected
        self.manifest_items = len(all_rows)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.source_manifest_items = len(source_rows)
        self.source_manifest_sha256 = sha256_file(self.source_manifest)
        self.stats_path = self.cache_root / "stats.pt"
        self.stats_sha256 = sha256_file(self.stats_path)
        stats = torch.load(self.stats_path, map_location="cpu", weights_only=False)
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        feature_path = self.feature_root / f"{sample_id}.pt"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing completed feature cache: {feature_path}")
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        expected_metadata = {
            "layers": (49,),
            "context_id": str(row["context_id"]),
            "action_horizon": self.action_horizon,
            "capture_token_count": 32,
            "capture_token_strategy": CACHE_STRATEGY,
            "backbone": CACHE_BACKBONE,
            "quantization": CACHE_QUANTIZATION,
            "manifest_items": self.source_manifest_items,
        }
        for key, expected in expected_metadata.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(
                    f"feature cache contract mismatch for {sample_id}: "
                    f"{key}={actual!r}, expected {expected!r}"
                )
        if not math.isclose(float(payload.get("timestep", -1.0)), 1.0):
            raise ValueError(f"feature cache timestep must be 1.0 for {sample_id}")
        features = payload["features"]
        if tuple(features.shape) != (1, 32, 5376):
            raise ValueError(f"unexpected last32 feature shape for {sample_id}: {features.shape}")
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"non-finite feature cache for {sample_id}")

        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        context_item = torch.load(
            self.cache_root / "contexts" / f"{row['context_id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if context_item.get("text_only") is not True:
            raise ValueError(f"context is not text-only for {sample_id}")
        actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"unexpected action shape for {sample_id}: {actions.shape}")
        return {
            "sample_id": sample_id,
            "features": features,
            "actions": normalize_minmax(actions, self.action_min, self.action_max),
            "proprio": normalize_minmax(
                window["state"].float(), self.state_min, self.state_max
            ),
            "action_is_pad": action_is_pad,
            "text_context": context_item["context"][0].float(),
        }


def build_model(spec: ModelSpec, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    return H3StarWAMFeatureActionPolicy(
        action_dim=spec.action_dim,
        proprio_dim=spec.proprio_dim,
        h3_feature_dim=spec.h3_feature_dim,
        context_dim=spec.context_dim,
        hidden_dim=spec.hidden_dim,
        ffn_dim=spec.ffn_dim,
        num_heads=spec.num_heads,
        attn_head_dim=spec.attn_head_dim,
        num_layers=spec.action_layers,
        freq_dim=spec.freq_dim,
        max_seq_len=spec.max_seq_len,
        use_gradient_checkpointing=spec.gradient_checkpointing,
        include_feature_timestep=spec.include_feature_timestep,
        feature_timestep=spec.feature_timestep,
        feature_input_scale=spec.feature_input_scale,
    ).to(device=device, dtype=dtype)


def deterministic_flow_batch(
    actions: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    uniform = torch.rand(
        actions.shape[0], device=actions.device, dtype=torch.float32, generator=generator
    )
    sigma = scheduler._phi(uniform, scheduler.shift)
    # Keep the continuous timestep in FP32.  BF16 rounds values near 1000 to
    # the exact endpoint, where StarWAM's importance weight is zero.  With a
    # shared per-step seed this made every DDP rank deterministically produce
    # zero gradients at the same optimizer step.
    timesteps = sigma * float(scheduler.num_train_timesteps)
    noise = torch.randn(
        actions.shape, device=actions.device, dtype=actions.dtype, generator=generator
    )
    noisy = scheduler.add_noise(actions, noise, timesteps)
    target = scheduler.training_target(actions, noise, timesteps)
    return noisy, target, timesteps


def distributed_flow_seed(
    *, base_seed: int, completed_step: int, accumulation_index: int, rank: int
) -> int:
    """Deterministic but rank-distinct action-flow RNG contract."""

    if min(completed_step, accumulation_index, rank) < 0:
        raise ValueError("flow seed coordinates must be non-negative")
    return (
        int(base_seed)
        + 1_000_003 * int(completed_step)
        + 10_000_019 * int(rank)
        + int(accumulation_index)
    )


def forward_policy(model: nn.Module, batch: dict[str, Any], noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    return model(
        noisy,
        timesteps,
        text_context=batch["text_context"],
        h3_features=batch["features"],
        proprio=batch["proprio"],
        text_mask=batch["text_mask"],
    )


def reconstruct_clean_action_from_flow(
    noisy_actions: torch.Tensor,
    predicted_velocity: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    """Recover x0 from FastWAM/StarWAM's v=noise-clean parameterization."""

    if noisy_actions.shape != predicted_velocity.shape or noisy_actions.ndim != 3:
        raise ValueError("flow clean reconstruction requires matching [B,T,D] tensors")
    if timesteps.ndim != 1 or timesteps.shape[0] != noisy_actions.shape[0]:
        raise ValueError("flow clean reconstruction requires one timestep per batch item")
    sigma = scheduler.timestep_to_sigma(timesteps).to(
        device=noisy_actions.device, dtype=noisy_actions.dtype
    )
    return noisy_actions - sigma[:, None, None] * predicted_velocity


def masked_chunk_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    """Direct clean-action chunk MSE with the released action padding contract."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("chunk regression requires matching [B,T,D] tensors")
    element_loss = (prediction.float() - target.float()).square()
    if action_is_pad is None:
        return element_loss.mean(dim=(1, 2)).mean()
    if tuple(action_is_pad.shape) != tuple(prediction.shape[:2]):
        raise ValueError("action padding mask must match [B,T]")
    valid = (~action_is_pad.bool()).to(element_loss).unsqueeze(-1)
    valid = valid.expand_as(element_loss)
    per_sample = (element_loss * valid).sum(dim=(1, 2)) / valid.sum(
        dim=(1, 2)
    ).clamp_min(1.0)
    return per_sample.mean()


def paired_visual_negative_features(features: torch.Tensor) -> torch.Tensor:
    """Return detached cross-sample H3 features without weakening batch-size one DDP.

    A local cyclic shift is sufficient when a rank owns multiple samples.  The
    production R1 launch uses one sample per rank, so in that case every rank
    receives the next rank's fixed-shape cached feature tensor.  Cached H3
    features are frozen inputs; the negative branch must never create a second
    gradient route into them.
    """

    if features.ndim < 2 or features.shape[0] <= 0:
        raise ValueError("paired visual negatives require a non-empty batch tensor")
    detached = features.detach()
    if features.shape[0] > 1:
        return detached.roll(shifts=1, dims=0)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [torch.empty_like(detached) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, detached)
        source_rank = (dist.get_rank() + 1) % dist.get_world_size()
        return gathered[source_rank]
    raise ValueError(
        "paired visual margin needs local batch >=2 or initialized distributed world_size >=2"
    )


def paired_visual_margin_loss(
    correct_flow_loss: torch.Tensor,
    wrong_visual_flow_loss: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Hinge that ranks the correct H3 observation above a paired wrong one."""

    if correct_flow_loss.ndim != 0 or wrong_visual_flow_loss.ndim != 0:
        raise ValueError("paired visual margin expects scalar flow losses")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("paired visual margin must be finite and positive")
    return torch.relu(correct_flow_loss - wrong_visual_flow_loss + margin)


def optimizer_step(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchScheduler,
    *,
    seed: int,
    max_grad_norm: float,
    gradient_accumulation_steps: int = 1,
    clean_action_regression_weight: float = 0.0,
    paired_visual_margin_weight: float = 0.0,
    paired_visual_margin: float = 0.05,
) -> dict[str, float]:
    if (
        not math.isfinite(clean_action_regression_weight)
        or clean_action_regression_weight < 0
    ):
        raise ValueError("clean action regression weight must be finite and non-negative")
    if not math.isfinite(paired_visual_margin_weight) or paired_visual_margin_weight < 0:
        raise ValueError("paired visual margin weight must be finite and non-negative")
    if clean_action_regression_weight > 0 and paired_visual_margin_weight > 0:
        raise ValueError("Candidates F and G are mutually exclusive controlled ablations")
    if paired_visual_margin_weight > 0 and (
        not math.isfinite(paired_visual_margin) or paired_visual_margin <= 0
    ):
        raise ValueError("paired visual margin must be finite and positive")
    noisy, target, timesteps = deterministic_flow_batch(
        batch["actions"], scheduler, seed=seed
    )
    prediction = forward_policy(model, batch, noisy, timesteps)
    flow_loss = flow_matching_loss(
        prediction,
        target,
        timesteps,
        scheduler,
        is_pad_mask=batch["action_is_pad"],
    )
    auxiliary_losses = []
    if clean_action_regression_weight > 0:
        predicted_clean_action = reconstruct_clean_action_from_flow(
            noisy, prediction, timesteps, scheduler
        )
        clean_action_regression_loss = masked_chunk_regression_loss(
            predicted_clean_action,
            batch["actions"],
            batch["action_is_pad"],
        )
        auxiliary_losses.append(
            clean_action_regression_weight * clean_action_regression_loss
        )
    else:
        clean_action_regression_loss = flow_loss.detach().new_zeros(())
    if paired_visual_margin_weight > 0:
        wrong_visual_batch = dict(batch)
        wrong_visual_batch["features"] = paired_visual_negative_features(
            batch["features"]
        )
        wrong_visual_prediction = forward_policy(
            model, wrong_visual_batch, noisy, timesteps
        )
        wrong_visual_flow_loss = flow_matching_loss(
            wrong_visual_prediction,
            target,
            timesteps,
            scheduler,
            is_pad_mask=batch["action_is_pad"],
        )
        visual_margin_loss = paired_visual_margin_loss(
            flow_loss,
            wrong_visual_flow_loss,
            margin=paired_visual_margin,
        )
        visual_prediction_delta = (
            prediction.float() - wrong_visual_prediction.float()
        ).square().mean()
        auxiliary_losses.append(paired_visual_margin_weight * visual_margin_loss)
    else:
        wrong_visual_flow_loss = flow_loss.detach().new_zeros(())
        visual_margin_loss = flow_loss.detach().new_zeros(())
        visual_prediction_delta = flow_loss.detach().new_zeros(())
    total_loss = flow_loss + sum(auxiliary_losses, start=flow_loss.new_zeros(()))
    (total_loss / gradient_accumulation_steps).backward()
    return {
        "loss": float(total_loss.detach()),
        "flow_loss": float(flow_loss.detach()),
        "clean_action_regression_loss": float(
            clean_action_regression_loss.detach()
        ),
        "weighted_clean_action_regression_loss": float(
            clean_action_regression_weight * clean_action_regression_loss.detach()
        ),
        "wrong_visual_flow_loss": float(wrong_visual_flow_loss.detach()),
        "paired_visual_margin_loss": float(visual_margin_loss.detach()),
        "weighted_paired_visual_margin_loss": float(
            paired_visual_margin_weight * visual_margin_loss.detach()
        ),
        "paired_visual_prediction_delta_mse": float(
            visual_prediction_delta.detach()
        ),
        "timestep_mean": float(timesteps.detach().float().mean()),
        "prediction_std": float(prediction.detach().float().std()),
    }


def module_grad_norm(module: nn.Module) -> float:
    total = torch.zeros((), device=next(module.parameters()).device, dtype=torch.float32)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    if dist.is_initialized():
        dist.all_reduce(total)
    return float(total.sqrt())


def probe_action_state_inference_flags(
    model: H3StarWAMFeatureActionPolicy,
    batch: dict[str, Any],
    scheduler: FlowMatchScheduler,
    *,
    seed: int,
) -> dict[str, bool]:
    """Audit every tensor retained by ActionDiT pre_dit after restore."""

    noisy, _, timesteps = deterministic_flow_batch(batch["actions"], scheduler, seed=seed)
    with torch.no_grad():
        context, context_mask = model.compose_context(
            batch["text_context"],
            batch["features"],
            batch["proprio"],
            batch["text_mask"],
        )
        state = model.action_expert.pre_dit(
            noisy, timesteps, context, context_mask
        )
    names = {
        "tokens": state["tokens"],
        "ctx": state["context"],
        "t_mod": state["t_mod"],
        "freqs": state["freqs"],
        "mask": state["context_mask"],
    }
    return {
        name: bool(tensor.is_inference()) if tensor is not None else False
        for name, tensor in names.items()
    }


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    scheduler_horizon: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Reproduce StarWAM's released linear-warmup then cosine schedule."""

    if warmup_steps < 0 or scheduler_horizon <= 1:
        raise ValueError("warmup_steps must be non-negative and scheduler_horizon > 1")
    warmup_steps = min(warmup_steps, scheduler_horizon - 1)
    remaining_steps = max(scheduler_horizon - warmup_steps, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,
        eta_min=min_learning_rate,
    )
    if warmup_steps == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def checkpoint_contract(
    args: argparse.Namespace, spec: ModelSpec, dataset: CachedLast32Dataset
) -> dict[str, Any]:
    paired_visual_margin_weight = float(
        getattr(args, "paired_visual_margin_weight", 0.0)
    )
    if args.clean_action_regression_weight > 0 and paired_visual_margin_weight > 0:
        raise ValueError("Candidates F and G are mutually exclusive controlled ablations")
    contract = {
        "starwam_commit": STARWAM_COMMIT,
        "starwam_action_dit_sha256": STARWAM_ACTION_DIT_SHA256,
        "starwam_wan_block_sha256": STARWAM_WAN_BLOCK_SHA256,
        "h3_checkpoint_sha256": args.expected_h3_checkpoint_sha256,
        "feature_subdir": args.feature_subdir,
        "feature_strategy": CACHE_STRATEGY,
        "feature_layers": [49],
        "feature_tokens": 32,
        "feature_timestep": 1.0,
        "feature_timestep_token": {
            "value": spec.feature_timestep,
            "semantics": "starwam_clean_observation_flow_timestep",
            "h3_curve_timestep": 1.0,
            "enabled": spec.include_feature_timestep,
            "embedding_trainable": spec.include_feature_timestep,
        },
        "feature_input_scale": spec.feature_input_scale,
        "feature_scale_audit": {
            "feature_samples": FEATURE_SCALE_AUDIT_SAMPLES,
            "feature_global_rms": FEATURE_GLOBAL_RMS,
            "text_contexts": TEXT_CONTEXT_COUNT,
            "text_context_global_rms": TEXT_CONTEXT_GLOBAL_RMS,
            "rms_match_scale": RMS_MATCH_FEATURE_INPUT_SCALE,
        },
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "source_manifest_items": dataset.source_manifest_items,
        "split_manifest_sha256": dataset.manifest_sha256,
        "split_manifest_items": dataset.manifest_items,
        "stats_sha256": dataset.stats_sha256,
        "action_normalization": "starwam_minmax_clip5",
        "state_normalization": "starwam_minmax_clip5",
        "action_horizon": args.action_horizon,
        "action_shift": args.action_shift,
        "flow_timestep_contract": "continuous_fp32_no_bf16_endpoint_rounding_v2",
        "flow_rng_contract": "base_plus_step1000003_plus_rank10000019_v2",
        "lr_schedule": {
            "type": "linear_warmup_then_cosine",
            "base_learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_steps": args.warmup_steps,
            "scheduler_horizon": args.scheduler_horizon,
        },
        "model_spec": asdict(spec),
    }
    if args.clean_action_regression_weight > 0:
        contract["clean_action_regression_complement"] = {
            "candidate": "F",
            "weight": args.clean_action_regression_weight,
            "base_target": "noise_minus_clean_action",
            "clean_reconstruction": "noisy_action_minus_sigma_times_predicted_velocity",
            "loss": "masked_clean_action_chunk_mse",
            "same_forward": True,
            "same_flow_noise": True,
            "extra_parameters": 0,
        }
    paired_visual_margin = float(getattr(args, "paired_visual_margin", 0.05))
    if paired_visual_margin_weight > 0:
        contract["paired_visual_margin_complement"] = {
            "candidate": "G",
            "weight": paired_visual_margin_weight,
            "margin": paired_visual_margin,
            "positive": "same_sample_cached_h3_features",
            "negative": "detached_cross_sample_cached_h3_features",
            "fixed_inputs": [
                "noisy_action",
                "flow_target",
                "timestep",
                "text_context",
                "proprio",
            ],
            "loss": "relu(correct_weighted_flow_loss-wrong_weighted_flow_loss+margin)",
            "extra_parameters": 0,
        }
    return contract


def _contract_mismatch(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    return [
        key
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def save_checkpoint_atomic(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_steps: int,
    contract: dict[str, Any],
    probe_prediction: torch.Tensor,
    probe_sample_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": int(completed_steps),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "contract": contract,
            "probe_prediction": probe_prediction.detach().float().cpu(),
            "probe_sample_ids": list(probe_sample_ids),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint_strict(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    mismatches = _contract_mismatch(expected_contract, payload.get("contract", {}))
    if mismatches:
        raise ValueError(f"checkpoint contract mismatch: {mismatches}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    return payload


def move_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    result = dict(batch)
    for key in ("features", "actions", "proprio", "text_context"):
        result[key] = batch[key].to(device=device, dtype=dtype, non_blocking=True)
    result["action_is_pad"] = batch["action_is_pad"].to(device=device, non_blocking=True)
    result["text_mask"] = batch["text_mask"].to(device=device, non_blocking=True)
    return result


def infinite_batches(loader: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def distributed_setup() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def main() -> None:
    args = parse_args()
    if min(
        args.steps,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.action_horizon,
        args.action_layers,
        args.hidden_dim,
        args.ffn_dim,
        args.num_heads,
        args.attn_head_dim,
        args.scheduler_horizon,
    ) <= 0:
        raise ValueError("positive training/model arguments are required")
    if (
        args.learning_rate <= 0
        or args.weight_decay < 0
        or args.action_shift <= 0
        or args.min_learning_rate < 0
        or args.min_learning_rate > args.learning_rate
        or args.warmup_steps < 0
        or not math.isfinite(args.clean_action_regression_weight)
        or args.clean_action_regression_weight < 0
        or not math.isfinite(args.feature_input_scale)
        or args.feature_input_scale <= 0
    ):
        raise ValueError("invalid optimizer or flow-shift value")
    if args.restore_check_only and args.load_checkpoint is None:
        raise ValueError("--restore-check-only requires --load-checkpoint")

    rank, world_size, device = distributed_setup()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    started = time.perf_counter()

    dataset = CachedLast32Dataset(
        args.manifest,
        args.cache_root,
        args.feature_subdir,
        source_manifest=args.source_manifest,
        action_horizon=args.action_horizon,
        limit=args.limit,
        sample_offset=args.sample_offset,
    )
    first_feature = torch.load(
        dataset.feature_root / f"{dataset.rows[0]['id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_path = Path(first_feature["checkpoint"])
    actual_h3_sha256 = None
    if args.verify_h3_checkpoint_sha256:
        if rank == 0:
            actual_h3_sha256 = sha256_file(checkpoint_path)
        if dist.is_initialized():
            value = [actual_h3_sha256]
            dist.broadcast_object_list(value, src=0)
            actual_h3_sha256 = value[0]
        if actual_h3_sha256 != args.expected_h3_checkpoint_sha256:
            raise ValueError(
                f"H3 checkpoint SHA256 mismatch: {actual_h3_sha256}"
            )

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_cached_batch,
    )
    spec = ModelSpec(
        action_layers=args.action_layers,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        max_seq_len=max(64, args.action_horizon),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        include_feature_timestep=False,
        feature_timestep=0.0,
        feature_input_scale=args.feature_input_scale,
    )
    model = build_model(spec, device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        scheduler_horizon=args.scheduler_horizon,
        min_learning_rate=args.min_learning_rate,
    )
    contract = checkpoint_contract(args, spec, dataset)
    loaded_payload = None
    completed_steps = 0
    if args.load_checkpoint is not None:
        loaded_payload = load_checkpoint_strict(
            args.load_checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            expected_contract=contract,
        )
        completed_steps = int(loaded_payload["completed_steps"])

    unwrapped_model = model
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
        unwrapped_model = model.module
    trainable_parameters = sum(
        parameter.numel() for parameter in unwrapped_model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in unwrapped_model.parameters())
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=args.action_shift)
    # Restore identity must not depend on which disjoint slice a resumed stage
    # trains on.  Always anchor the checkpoint probe to manifest row zero.
    probe_dataset = CachedLast32Dataset(
        args.manifest,
        args.cache_root,
        args.feature_subdir,
        source_manifest=args.source_manifest,
        action_horizon=args.action_horizon,
        limit=1,
        sample_offset=0,
    )
    probe_cpu = collate_cached_batch([probe_dataset[0]])
    probe = move_batch(probe_cpu, device, dtype)

    restore_max_abs = None
    resume_tensor_inference_flags = None
    if loaded_payload is not None:
        unwrapped_model.eval()
        # no_grad (rather than inference_mode) is required because ActionDiT
        # lazily retains its RoPE cache and resumed training must reuse it in
        # autograd/checkpointed forwards.
        with torch.no_grad():
            loaded_noisy, _, loaded_timesteps = deterministic_flow_batch(
                probe["actions"], scheduler, seed=args.seed + 9_000_001
            )
            loaded_prediction = forward_policy(
                unwrapped_model, probe, loaded_noisy, loaded_timesteps
            ).float()
        expected_prediction = loaded_payload["probe_prediction"].to(loaded_prediction)
        if loaded_payload.get("probe_sample_ids") != probe_cpu["sample_ids"]:
            raise ValueError("checkpoint restore probe sample identity mismatch")
        restore_max_abs = float((loaded_prediction - expected_prediction).abs().max())
        if restore_max_abs != 0.0:
            raise RuntimeError(
                f"checkpoint restore prediction mismatch: max_abs={restore_max_abs}"
            )
        resume_tensor_inference_flags = probe_action_state_inference_flags(
            unwrapped_model,
            probe,
            scheduler,
            seed=args.seed + 9_000_001,
        )
        if any(resume_tensor_inference_flags.values()):
            raise RuntimeError(
                "restore left inference tensors in ActionDiT state: "
                f"{resume_tensor_inference_flags}"
            )

    history: list[dict[str, Any]] = []
    if not args.restore_check_only:
        model.train()
        iterator = iter(infinite_batches(loader))
        optimizer.zero_grad(set_to_none=True)
        tracked = unwrapped_model.action_expert.head.weight
        tracked_before = tracked.detach().float().clone()
        for local_step in range(1, args.steps + 1):
            micro_metrics = []
            step_sample_ids: list[str] = []
            for accumulation_index in range(args.gradient_accumulation_steps):
                batch = move_batch(next(iterator), device, dtype)
                local_sample_ids = list(batch["sample_ids"])
                if dist.is_initialized():
                    gathered_sample_ids: list[list[str] | None] = [
                        None for _ in range(world_size)
                    ]
                    dist.all_gather_object(gathered_sample_ids, local_sample_ids)
                    for rank_sample_ids in gathered_sample_ids:
                        if rank_sample_ids is not None:
                            step_sample_ids.extend(rank_sample_ids)
                else:
                    step_sample_ids.extend(local_sample_ids)
                micro_metrics.append(
                    optimizer_step(
                        model,
                        batch,
                        optimizer,
                        scheduler,
                        seed=distributed_flow_seed(
                            base_seed=args.seed,
                            completed_step=completed_steps + local_step,
                            accumulation_index=accumulation_index,
                            rank=rank,
                        ),
                        max_grad_norm=args.max_grad_norm,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        clean_action_regression_weight=(
                            args.clean_action_regression_weight
                        ),
                        paired_visual_margin_weight=(
                            args.paired_visual_margin_weight
                        ),
                        paired_visual_margin=args.paired_visual_margin,
                    )
                )
            expert_gradient = module_grad_norm(unwrapped_model.action_expert)
            projector_gradient = module_grad_norm(unwrapped_model.feature_projector)
            proprio_gradient = module_grad_norm(unwrapped_model.proprio_encoder)
            gradient_values = (expert_gradient, projector_gradient, proprio_gradient)
            if not all(math.isfinite(value) and value > 0 for value in gradient_values):
                raise RuntimeError(f"non-finite/zero gradient path: {gradient_values}")
            clipped_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_max_abs = float(
                (tracked.detach().float() - tracked_before).abs().max()
            )
            if not math.isfinite(update_max_abs) or update_max_abs <= 0:
                raise RuntimeError("optimizer did not update the ActionDiT head")
            tracked_before = tracked.detach().float().clone()
            item = {
                "step": completed_steps + local_step,
                "loss": sum(x["loss"] for x in micro_metrics) / len(micro_metrics),
                "flow_loss": sum(x["flow_loss"] for x in micro_metrics)
                / len(micro_metrics),
                "clean_action_regression_loss": sum(
                    x["clean_action_regression_loss"] for x in micro_metrics
                )
                / len(micro_metrics),
                "weighted_clean_action_regression_loss": sum(
                    x["weighted_clean_action_regression_loss"]
                    for x in micro_metrics
                )
                / len(micro_metrics),
                "wrong_visual_flow_loss": sum(
                    x["wrong_visual_flow_loss"] for x in micro_metrics
                )
                / len(micro_metrics),
                "paired_visual_margin_loss": sum(
                    x["paired_visual_margin_loss"] for x in micro_metrics
                )
                / len(micro_metrics),
                "weighted_paired_visual_margin_loss": sum(
                    x["weighted_paired_visual_margin_loss"] for x in micro_metrics
                )
                / len(micro_metrics),
                "paired_visual_prediction_delta_mse": sum(
                    x["paired_visual_prediction_delta_mse"] for x in micro_metrics
                )
                / len(micro_metrics),
                "timestep_mean": sum(x["timestep_mean"] for x in micro_metrics) / len(micro_metrics),
                "prediction_std": sum(x["prediction_std"] for x in micro_metrics) / len(micro_metrics),
                "expert_gradient_norm": expert_gradient,
                "feature_projector_gradient_norm": projector_gradient,
                "proprio_gradient_norm": proprio_gradient,
                "clipped_gradient_norm": clipped_norm,
                "head_update_max_abs": update_max_abs,
                "learning_rate": learning_rate,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "sample_ids": step_sample_ids,
            }
            history.append(item)
            if rank == 0:
                print(json.dumps(item, sort_keys=True), flush=True)
        completed_steps += args.steps

    model.eval()
    with torch.inference_mode():
        noisy, _, timesteps = deterministic_flow_batch(
            probe["actions"], scheduler, seed=args.seed + 9_000_001
        )
        probe_prediction = forward_policy(
            unwrapped_model, probe, noisy, timesteps
        ).float()
    if not torch.isfinite(probe_prediction).all() or float(probe_prediction.std()) <= 0:
        raise RuntimeError("restore probe output is non-finite or constant")

    if args.save_checkpoint is not None and not args.restore_check_only:
        if rank == 0:
            save_checkpoint_atomic(
                args.save_checkpoint,
                model=unwrapped_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                completed_steps=completed_steps,
                contract=contract,
                probe_prediction=probe_prediction,
                probe_sample_ids=probe_cpu["sample_ids"],
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        report = {
            "event": "h3_int8_starwam_action_canary",
            "classification": "action-only-on-frozen-features",
            "resolved_argv": sys.argv,
            "world_size": world_size,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_batch_size": (
                world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
            ),
            "selected_windows": len(dataset),
            "manifest_items": dataset.manifest_items,
            "source_manifest_items": dataset.source_manifest_items,
            "source_manifest_sha256": dataset.source_manifest_sha256,
            "split_manifest_sha256": dataset.manifest_sha256,
            "completed_steps": completed_steps,
            "training_samples": (
                0
                if args.restore_check_only
                else world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
                * args.steps
            ),
            "effective_epochs": (
                0.0
                if args.restore_check_only
                else world_size
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
                * args.steps
                / dataset.manifest_items
            ),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "model_spec": asdict(spec),
            "action_objective": {
                "base": "pinned_starwam_weighted_masked_flow",
                "clean_action_regression_weight": (
                    args.clean_action_regression_weight
                ),
                "candidate_f_enabled": args.clean_action_regression_weight > 0,
                "paired_visual_margin_weight": args.paired_visual_margin_weight,
                "paired_visual_margin": args.paired_visual_margin,
                "candidate_g_enabled": args.paired_visual_margin_weight > 0,
            },
            "contract": contract,
            "h3_checkpoint_path": str(checkpoint_path),
            "h3_checkpoint_sha256_verified": actual_h3_sha256,
            "loaded_checkpoint": (
                None if args.load_checkpoint is None else str(args.load_checkpoint)
            ),
            "saved_checkpoint": (
                None if args.save_checkpoint is None else str(args.save_checkpoint)
            ),
            "restore_probe_max_abs": restore_max_abs,
            "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "lr_scheduler_last_epoch": int(lr_scheduler.last_epoch),
            "resume_tensor_inference_flags": resume_tensor_inference_flags,
            "probe_prediction_mean": float(probe_prediction.mean()),
            "probe_prediction_std": float(probe_prediction.std()),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda"
                else 0.0
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
