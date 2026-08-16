#!/usr/bin/env python3
"""Paired C55 action-only vs FACT-style joint-aux continuation training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    DEFAULT_H3_CARRIER_LAYERS,
    DREAMWAM_COMMIT,
    REPEAT_LAYER49_CARRIER_SOURCE,
)
from fastwam.models.h3wam.fact_joint_aux import (  # noqa: E402
    FACT_COMMIT,
    H3FactJointAuxPolicy,
)


def load_candidate_d_module():
    path = Path(__file__).with_name("train_h3_int8_dreamwam_kv_carrier.py")
    spec = importlib.util.spec_from_file_location("_c55_candidate_d", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load D0 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D0 = load_candidate_d_module()
PARENT = D0.PARENT
FORMAT = "h3wam-c55-fact-joint-action-v2"
CHECKPOINT_SCHEMA = 1
FUTURE_H3_TARGET_NORM = "train-sample-weighted-per-dimension-zscore-v1"
EXPECTED_H3_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_index(seed: int, stream: str, ordinal: int, size: int) -> int:
    if size <= 0:
        raise ValueError("sampling pool must be non-empty")
    value = hashlib.blake2b(
        f"{seed}:{stream}:{ordinal}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(value, "little") % size


def fit_future_h3_target_norm(
    observation_ids: torch.Tensor,
    features: torch.Tensor,
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Fit target scaling on the immutable train sample distribution only."""

    positions = {int(value): index for index, value in enumerate(observation_ids)}
    train_future_ids = [
        int(row["future_observation_id"])
        for row in rows
        if row["split"] == "train"
    ]
    if not train_future_ids:
        raise ValueError("C55 future-H3 target normalization has no train rows")
    try:
        train_positions = torch.tensor(
            [positions[value] for value in train_future_ids], dtype=torch.long
        )
    except KeyError as error:
        raise ValueError(
            f"C55 future-H3 normalization target is missing: {error.args[0]}"
        ) from error
    train_targets = features.index_select(0, train_positions).float()
    mean = train_targets.mean(dim=0)
    std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("C55 future-H3 target normalization is non-finite")
    digest = hashlib.sha256()
    digest.update(FUTURE_H3_TARGET_NORM.encode("utf-8"))
    digest.update(mean.contiguous().numpy().tobytes())
    digest.update(std.contiguous().numpy().tobytes())
    return mean, std, digest.hexdigest()


def environment_actions_to_dataset(actions: torch.Tensor) -> torch.Tensor:
    """Invert the exact LIBERO deployment gripper conversion."""

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("environment actions must be [T,7]")
    result = actions.float().clone()
    result[:, -1] = (1.0 - result[:, -1]) / 2.0
    return result


def dataset_actions_to_environment(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("dataset actions must be [T,7]")
    result = actions.float().clone()
    result[:, -1] = -(2.0 * result[:, -1] - 1.0)
    return result


class C55RolloutDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        projected_features_path: Path,
        kv_root: Path,
        demo_cache_root: Path,
        *,
        split: str = "train",
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.projected_features_path = projected_features_path.resolve()
        self.kv_root = kv_root.resolve()
        self.demo_cache_root = demo_cache_root.resolve()
        payload = torch.load(self.dataset_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "h3wam-c48-fact-dense-value-dataset-v1":
            raise ValueError("C55 requires the immutable C48 dataset")
        all_rows = payload["samples"]
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"C55 rollout split is empty: {split}")
        if len({int(row["sample_id"]) for row in self.rows}) != len(self.rows):
            raise ValueError("C55 rollout sample ids are duplicated")
        self.success_indices = [i for i, row in enumerate(self.rows) if row["success"]]
        self.failure_indices = [i for i, row in enumerate(self.rows) if not row["success"]]
        if not self.success_indices or not self.failure_indices:
            raise ValueError("C55 rollout split requires both success and failure")

        projected = torch.load(
            self.projected_features_path, map_location="cpu", weights_only=False
        )
        if projected.get("format") != "h3wam-c49-dense-value-projected-features-v1":
            raise ValueError("C55 projected H3 feature format mismatch")
        ids = projected["observation_ids"].long()
        features = projected["fact_layer49_projected"].float()
        if ids.ndim != 1 or features.shape != (len(ids), 256):
            raise ValueError("C55 projected H3 feature tensor mismatch")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("C55 projected H3 observation ids are duplicated")
        feature_mean, feature_std, target_norm_sha256 = fit_future_h3_target_norm(
            ids, features, all_rows
        )
        normalized_features = (features - feature_mean) / feature_std
        self.feature_by_id = {
            int(i): normalized_features[pos] for pos, i in enumerate(ids)
        }
        self.future_h3_target_mean = feature_mean
        self.future_h3_target_std = feature_std
        self.future_h3_target_norm_sha256 = target_norm_sha256

        stats = torch.load(
            self.demo_cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        self.dataset_sha256 = sha256_file(self.dataset_path)
        self.projected_features_sha256 = sha256_file(self.projected_features_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        observation_id = int(row["current_observation_id"])
        future_observation_id = int(row["future_observation_id"])
        path = self.kv_root / "items" / f"obs_{observation_id:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "schema": D0.DREAMWAM_KV_SCHEMA,
            "format": "h3wam-c55-rollout-kv-shard-v1",
            "observation_id": observation_id,
            "split": str(row["split"]),
            "layers": DEFAULT_H3_CARRIER_LAYERS,
            "capture_token_count": 32,
            "num_heads": 56,
            "attn_head_dim": 128,
            "capture_token_strategy": D0.DREAMWAM_KV_STRATEGY,
            "dreamwam_commit": DREAMWAM_COMMIT,
            "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
            "dataset_sha256": self.dataset_sha256,
        }
        for key, expected_value in expected.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"C55 rollout K/V mismatch for obs{observation_id}: "
                    f"{key}={actual!r}, expected={expected_value!r}"
                )
        video_kv_cache = payload["video_kv_cache"]
        if set(video_kv_cache) != set(DEFAULT_H3_CARRIER_LAYERS):
            raise ValueError("C55 rollout K/V layer set mismatch")
        signatures = set()
        for layer in DEFAULT_H3_CARRIER_LAYERS:
            if set(video_kv_cache[layer]) != {"k", "v"}:
                raise ValueError("C55 rollout K/V item must contain k and v")
            for name in ("k", "v"):
                tensor = video_kv_cache[layer][name]
                if tensor.shape != (32, 56, 128) or tensor.dtype != torch.bfloat16:
                    raise ValueError("C55 rollout K/V tensor contract mismatch")
                signature = tensor.untyped_storage().data_ptr()
                if signature in signatures:
                    raise ValueError("C55 rollout K/V storage alias")
                signatures.add(signature)
                if not torch.isfinite(tensor.float()).all():
                    raise ValueError("C55 rollout K/V is non-finite")

        environment_actions = row["executed_actions"].float()
        dataset_actions = environment_actions_to_dataset(environment_actions)
        valid = ~row["action_is_pad"].bool()
        roundtrip = dataset_actions_to_environment(dataset_actions)
        if valid.any() and not torch.allclose(
            roundtrip[valid], environment_actions[valid], rtol=0.0, atol=1e-7
        ):
            raise ValueError("C55 environment/dataset action roundtrip failed")
        normalized_actions = PARENT.normalize_minmax(
            dataset_actions, self.action_min, self.action_max
        )
        context_id = str(payload["context_id"])
        context = torch.load(
            self.demo_cache_root / "contexts" / f"{context_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if context.get("text_only") is not True:
            raise ValueError("C55 rollout context is not text-only")
        if future_observation_id not in self.feature_by_id:
            raise ValueError("C55 future H3 target is missing")
        return {
            "sample_id": f"rollout_{int(row['sample_id']):06d}",
            "video_kv_cache": video_kv_cache,
            "actions": normalized_actions,
            "proprio": PARENT.normalize_minmax(
                row["current_proprio"].float(), self.state_min, self.state_max
            ),
            "action_is_pad": row["action_is_pad"].bool(),
            "text_context": context["context"][0].float(),
            "future_h3": self.feature_by_id[future_observation_id].clone(),
            "future_state": PARENT.normalize_minmax(
                row["future_proprio"].float(), self.state_min, self.state_max
            ),
            # FACT default min/max is 0/2, hence normalized value = raw - 1.
            "value": torch.tensor(float(row["value_target"]) - 1.0),
            "action_loss_mask": torch.tensor(float(bool(row["success"]))),
            "success": bool(row["success"]),
        }


def collate_rollout(item: dict[str, Any]) -> dict[str, Any]:
    batch = D0.collate_cached_batch([item])
    batch.update(
        {
            "future_h3": item["future_h3"].unsqueeze(0),
            "future_state": item["future_state"].unsqueeze(0),
            "value": item["value"].reshape(1),
            "action_loss_mask": item["action_loss_mask"].reshape(1),
            "success": item["success"],
        }
    )
    return batch


def move_targets(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = D0.move_batch(batch, device, torch.bfloat16)
    for key in ("future_h3", "future_state", "value", "action_loss_mask"):
        if key in batch:
            result[key] = batch[key].to(device=device, dtype=torch.float32)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("action_only", "joint_aux"), required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--demo-manifest", type=Path, required=True)
    parser.add_argument("--demo-source-manifest", type=Path, required=True)
    parser.add_argument("--demo-cache-root", type=Path, required=True)
    parser.add_argument("--demo-kv-subdir", required=True)
    parser.add_argument("--rollout-dataset", type=Path, required=True)
    parser.add_argument("--rollout-projected-features", type=Path, required=True)
    parser.add_argument("--rollout-kv-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--scheduler-horizon", type=int, default=6000)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--restore-check-only", action="store_true")
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device("cuda", local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("C55 training requires CUDA")
    torch.cuda.set_device(0)
    return 0, 1, torch.device("cuda", 0)


def distributed_mean(value: torch.Tensor, world_size: int) -> float:
    scalar = value.detach().float().reshape(())
    if dist.is_initialized():
        scalar = scalar.clone()
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        scalar /= world_size
    return float(scalar.cpu())


def carrier_forward(
    model: nn.Module,
    batch: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    arguments = {
        "text_context": batch["text_context"],
        "proprio": batch["proprio"],
        "video_kv_cache": batch["video_kv_cache"],
        "text_mask": batch["text_mask"],
    }
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if noisy.device.type == "cuda"
        else nullcontext()
    )
    with context:
        return model(noisy, timesteps, **arguments)


def joint_forward(
    model: nn.Module,
    batch: dict[str, Any],
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
) -> dict[str, torch.Tensor]:
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if noisy.device.type == "cuda"
        else nullcontext()
    )
    with context:
        return model(
            noisy,
            timesteps,
            clean_executed_actions=batch["actions"],
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            video_kv_cache=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
        )


def action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    is_pad_mask: torch.Tensor,
) -> torch.Tensor:
    return D0.flow_matching_loss(
        prediction, target, timesteps, scheduler, is_pad_mask=is_pad_mask
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0 or args.scheduler_horizon <= 0:
        raise ValueError("C55 steps/LR/scheduler horizon must be positive")
    if args.restore_check_only and args.load_checkpoint is None:
        raise ValueError("restore-check-only requires --load-checkpoint")
    if len(args.expected_parent_sha256) != 64:
        raise ValueError("expected parent SHA256 must be explicit")
    parent_path = args.parent_checkpoint.resolve()
    parent_sha256 = sha256_file(parent_path)
    if parent_sha256 != args.expected_parent_sha256:
        raise ValueError(
            f"C55 D0 parent identity mismatch: {parent_sha256}"
        )

    rank, world_size, device = setup_distributed()
    if world_size != 8:
        raise ValueError("C55 paired contract requires exactly 8 ranks per arm")
    torch.manual_seed(args.seed + rank)
    started = time.perf_counter()

    demo = D0.CachedDreamWAMKVDataset(
        args.demo_manifest,
        args.demo_cache_root,
        args.demo_kv_subdir,
        source_manifest=args.demo_source_manifest,
        carrier_layers=DEFAULT_H3_CARRIER_LAYERS,
        capture_token_count=32,
        kv_pool_strategy=D0.DREAMWAM_KV_STRATEGY,
        num_heads=56,
        attn_head_dim=128,
        action_horizon=32,
    )
    rollout = C55RolloutDataset(
        args.rollout_dataset,
        args.rollout_projected_features,
        args.rollout_kv_root,
        args.demo_cache_root,
        split="train",
    )
    spec = D0.ModelSpec(
        carrier_layers=DEFAULT_H3_CARRIER_LAYERS,
        carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
    )
    carrier = D0.build_model(spec, device=device, dtype=torch.bfloat16)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_contract = parent.get("contract", {})
    required_parent = {
        "candidate": "D0",
        "carrier_source_mode": REPEAT_LAYER49_CARRIER_SOURCE,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "action_horizon": 32,
        "kv_strategy": D0.DREAMWAM_KV_STRATEGY,
    }
    mismatch = [
        key for key, expected in required_parent.items()
        if parent_contract.get(key) != expected
    ]
    if mismatch:
        raise ValueError(f"C55 parent checkpoint contract mismatch: {mismatch}")
    carrier.load_state_dict(parent["model"], strict=True)

    raw_model: nn.Module
    if args.arm == "joint_aux":
        raw_model = H3FactJointAuxPolicy(
            carrier, hidden_dim=1024, future_h3_dim=256, future_state_dim=8
        ).to(device=device, dtype=torch.bfloat16)
    else:
        raw_model = carrier
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    lr_scheduler = PARENT.build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        scheduler_horizon=args.scheduler_horizon,
        min_learning_rate=args.min_learning_rate,
    )
    completed_steps = 0
    saved = None
    if args.load_checkpoint is not None:
        saved = torch.load(
            args.load_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if saved.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("C55 checkpoint schema mismatch")
        contract = saved.get("contract", {})
        if (
            contract.get("arm") != args.arm
            or contract.get("parent_sha256") != parent_sha256
            or contract.get("future_h3_target_norm") != FUTURE_H3_TARGET_NORM
            or contract.get("future_h3_target_norm_sha256")
            != rollout.future_h3_target_norm_sha256
        ):
            raise ValueError("C55 resume contract mismatch")
        raw_model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        lr_scheduler.load_state_dict(saved["lr_scheduler"])
        completed_steps = int(saved["completed_steps"])

    model: nn.Module = DistributedDataParallel(
        raw_model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=args.arm == "joint_aux",
    )
    scheduler = D0.FlowMatchScheduler(num_train_timesteps=1000, shift=args.action_shift)

    def probe_prediction() -> torch.Tensor:
        probe_batch = move_targets(
            D0.collate_cached_batch([demo[0]]), device
        )
        noisy, _, timesteps = PARENT.deterministic_flow_batch(
            probe_batch["actions"], scheduler, seed=314159265
        )
        raw_model.eval()
        with torch.inference_mode():
            prediction = carrier_forward(raw_model, probe_batch, noisy, timesteps)
        raw_model.train()
        return prediction.detach().float().cpu()

    if args.restore_check_only:
        assert saved is not None
        actual = probe_prediction()
        expected = saved.get("probe_prediction")
        if not torch.is_tensor(expected):
            raise ValueError("C55 checkpoint is missing its restore probe")
        max_abs = float((actual - expected.float()).abs().max())
        if max_abs != 0.0:
            raise RuntimeError(f"C55 restore probe mismatch: {max_abs}")
        if rank == 0:
            report = {
                "format": FORMAT,
                "status": "PASS_CHECKPOINT_RESTORE",
                "effect_status": "NOT_EVIDENCE_READY",
                "completed_steps": completed_steps,
                "restore_probe_max_abs": max_abs,
            }
            output = args.output.resolve()
            if output.exists():
                raise FileExistsError(f"refusing to overwrite C55 report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, sort_keys=True))
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    model.train()
    history = []
    for local_step in range(1, args.steps + 1):
        step = completed_steps + local_step
        global_ordinal = (step - 1) * world_size + rank
        demo_index = stable_index(args.seed, "demo", global_ordinal, len(demo))
        pool = rollout.success_indices if rank % 2 == 0 else rollout.failure_indices
        rollout_index = pool[
            stable_index(args.seed, "rollout", global_ordinal, len(pool))
        ]
        demo_batch = move_targets(
            D0.collate_cached_batch([demo[demo_index]]), device
        )
        rollout_batch = move_targets(
            collate_rollout(rollout[rollout_index]), device
        )
        demo_noisy, demo_target, demo_t = PARENT.deterministic_flow_batch(
            demo_batch["actions"],
            scheduler,
            seed=PARENT.distributed_flow_seed(
                base_seed=args.seed,
                completed_step=step,
                accumulation_index=0,
                rank=rank,
            ),
        )
        rollout_noisy, rollout_target, rollout_t = PARENT.deterministic_flow_batch(
            rollout_batch["actions"],
            scheduler,
            seed=PARENT.distributed_flow_seed(
                base_seed=args.seed,
                completed_step=step,
                accumulation_index=1,
                rank=rank,
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        demo_prediction = carrier_forward(model, demo_batch, demo_noisy, demo_t)
        demo_action_loss = action_loss(
            demo_prediction,
            demo_target,
            demo_t,
            scheduler,
            demo_batch["action_is_pad"],
        )
        (0.5 * demo_action_loss).backward()
        if args.arm == "joint_aux":
            assert isinstance(raw_model, H3FactJointAuxPolicy)
            outputs = joint_forward(model, rollout_batch, rollout_noisy, rollout_t)
            rollout_prediction = outputs["action"]
        else:
            rollout_prediction = carrier_forward(
                model, rollout_batch, rollout_noisy, rollout_t
            )
            outputs = None
        rollout_action_loss = action_loss(
            rollout_prediction,
            rollout_target,
            rollout_t,
            scheduler,
            rollout_batch["action_is_pad"],
        )
        # Exactly four success and four failure ranks.  Scaling valid rollout
        # action losses by two reproduces a global masked mean over successes.
        masked_rollout_action = (
            2.0 * rollout_batch["action_loss_mask"].mean() * rollout_action_loss
        )
        future_h3_loss = torch.zeros((), device=device)
        future_state_loss = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        rollout_total = masked_rollout_action
        if outputs is not None:
            future_h3_loss = torch.nn.functional.mse_loss(
                outputs["future_h3"].float(), rollout_batch["future_h3"]
            )
            future_state_loss = torch.nn.functional.mse_loss(
                outputs["future_state"].float(), rollout_batch["future_state"]
            )
            value_loss = torch.nn.functional.mse_loss(
                outputs["value"].float(), rollout_batch["value"]
            )
            # C48 knows only the terminal episode outcome, not FACT's causal
            # failure_active onset.  Do not fabricate an onset penalty: value
            # trains on successful rows only, while observed future H3/state
            # remain valid consequence targets for both outcomes.
            masked_value_loss = (
                2.0 * rollout_batch["action_loss_mask"].mean() * value_loss
            )
            # FACT weights 10:1:0.4:0.4 divided by action weight 10 keeps the
            # existing D0 action LR scale unchanged.
            rollout_total = (
                masked_rollout_action
                + 0.1 * future_h3_loss
                + 0.04 * future_state_loss
                + 0.04 * masked_value_loss
            )
        loss = 0.5 * (demo_action_loss + rollout_total)
        (0.5 * rollout_total).backward()
        carrier_for_grad = raw_model.carrier if isinstance(raw_model, H3FactJointAuxPolicy) else raw_model
        block_gradients = [PARENT.module_grad_norm(block) for block in carrier_for_grad.action_expert.blocks]
        if not all(math.isfinite(value) and value > 0 for value in block_gradients):
            raise RuntimeError(f"C55 shared action gradient path failed: {block_gradients}")
        clipped = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
        )
        optimizer.step()
        lr_scheduler.step()
        history.append(
            {
                "step": step,
                "loss": distributed_mean(loss, world_size),
                "demo_action_loss": distributed_mean(demo_action_loss, world_size),
                "rollout_action_loss": distributed_mean(rollout_action_loss, world_size),
                "rollout_action_mask": distributed_mean(
                    rollout_batch["action_loss_mask"].mean(), world_size
                ),
                "future_h3_loss": distributed_mean(future_h3_loss, world_size),
                "future_state_loss": distributed_mean(future_state_loss, world_size),
                "value_loss": distributed_mean(value_loss, world_size),
                "block_gradient_norms": block_gradients,
                "clipped_gradient_norm": clipped,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "demo_sample_id": demo_batch["sample_ids"][0],
                "rollout_sample_id": rollout_batch["sample_ids"][0],
            }
        )

    completed_steps += args.steps
    contract = {
        "format": FORMAT,
        "arm": args.arm,
        "classification": "backbone_port",
        "parent_sha256": parent_sha256,
        "parent_completed_steps": int(parent["completed_steps"]),
        "fact_commit": FACT_COMMIT,
        "dreamwam_commit": DREAMWAM_COMMIT,
        "h3_checkpoint_sha256": EXPECTED_H3_SHA256,
        "demo_manifest_sha256": demo.manifest_sha256,
        "demo_source_manifest_sha256": demo.source_manifest_sha256,
        "rollout_dataset_sha256": rollout.dataset_sha256,
        "rollout_projected_features_sha256": rollout.projected_features_sha256,
        "future_h3_target_norm": FUTURE_H3_TARGET_NORM,
        "future_h3_target_norm_sha256": rollout.future_h3_target_norm_sha256,
        "action_contract": "inverse exact deployment gripper map; D0 minmax; no clamp",
        "sampling_contract": "each step: 8 demo + 4 success rollout + 4 failure rollout",
        "loss_contract": "0.5*(demo_action + masked_success_rollout_action + optional 0.1*train-zscored-H3_all+0.04state_all+0.04value_success_only)",
        "failure_boundary": "failure actions and value are masked because causal failure_active onset is unavailable; observed future H3/state train on all rows",
        "action_shift": args.action_shift,
        "seed": args.seed,
        "world_size": world_size,
    }
    if rank == 0 and args.save_checkpoint is not None:
        output = args.save_checkpoint.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite C55 checkpoint: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "completed_steps": completed_steps,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "contract": contract,
            "probe_prediction": probe_prediction(),
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        torch.save(payload, temporary)
        os.replace(temporary, output)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        report = {
            "format": FORMAT,
            "status": "PASS_MECHANICAL" if args.steps <= 10 else "TRAINED_NOT_EVALUATED",
            "effect_status": "NOT_EVIDENCE_READY",
            "completed_steps": completed_steps,
            "training_samples_this_invocation": args.steps * world_size * 2,
            "stream_effective_epochs": {
                "demo": completed_steps * world_size / len(demo),
                "rollout": completed_steps * world_size / len(rollout),
            },
            "contract": contract,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "save_checkpoint": None if args.save_checkpoint is None else str(args.save_checkpoint.resolve()),
        }
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite C55 report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(json.dumps(report, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
