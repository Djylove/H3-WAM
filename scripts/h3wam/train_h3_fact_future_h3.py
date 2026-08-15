#!/usr/bin/env python3
"""Run the isolated H3-FACT future-observation representation canary."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

from fastwam.models.h3wam.fact_lite_consequence import (  # noqa: E402
    FutureH3ConsequenceModel,
    TemporalFutureH3ConsequenceModel,
    actions_for_arm,
    future_h3_mse,
)
from train_h3_fact_future_proprio import (  # noqa: E402
    ARMS,
    CACHE_BACKBONE,
    CACHE_QUANTIZATION,
    CACHE_STRATEGY,
    FACT_COMMIT,
    H3_INT8_CHECKPOINT_SHA256,
    atomic_json_save,
    atomic_torch_save,
    audit_manifests,
    normalize_minmax,
    read_jsonl,
    relative_gain,
    select_rows,
    sha256_file,
    sha256_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--feature-subdir", default="h3_int8_starwam_last32_dense_v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--model-variant", choices=("flattened", "temporal"), default="flattened"
    )
    parser.add_argument("--actions-per-latent", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-h3-checkpoint-sha256", action="store_true")
    parser.add_argument(
        "--expected-h3-checkpoint-sha256", default=H3_INT8_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--feature-input-scale", type=float, default=0.009606920816877307
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-shuffle-degradation", type=float, default=0.01)
    parser.add_argument(
        "--selection-salt", default="h3-fact-future-proprio-f1-v1"
    )
    return parser.parse_args()


def temporal_key(row: dict[str, Any], *, start: int | None = None) -> tuple[str, str, int, int]:
    return (
        str(Path(row["dataset_root"]).resolve()),
        str(row["suite"]),
        int(row["episode"]),
        int(row["start"] if start is None else start),
    )


def build_future_index(source_rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    index: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in source_rows:
        key = temporal_key(row)
        if key in index:
            raise ValueError(f"duplicate temporal source key: {key}")
        index[key] = row
    return index


def eligible_rows(
    rows: list[dict[str, Any]],
    future_index: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    action_horizon: int,
) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        future_start = int(row["start"]) + action_horizon
        if bool(row.get("padded_tail", False)) or future_start >= int(row["length"]):
            continue
        future = future_index.get(temporal_key(row, start=future_start))
        if future is None:
            continue
        if str(future["context_id"]) != str(row["context_id"]):
            raise ValueError(f"future context changed within episode: {row['id']}")
        eligible.append(row)
    return eligible


class FutureH3Dataset(Dataset):
    """Join each current dense window to the H3 observation at start+horizon."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        source_rows: list[dict[str, Any]],
        cache_root: Path,
        feature_subdir: str,
        action_horizon: int,
    ) -> None:
        if not rows:
            raise ValueError("selected future-H3 rows are empty")
        self.rows = rows
        self.source_items = len(source_rows)
        self.future_index = build_future_index(source_rows)
        self.cache_root = cache_root.resolve()
        self.feature_root = self.cache_root / feature_subdir
        self.action_horizon = int(action_horizon)
        stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        if tuple(self.action_min.shape) != (7,) or tuple(self.state_min.shape) != (8,):
            raise ValueError("v7 normalization stats have unexpected shapes")
        first = self._feature_payload(self.rows[0])
        if not first.get("checkpoint"):
            raise ValueError("H3 feature cache is missing checkpoint identity")
        self.h3_checkpoint_path = Path(first["checkpoint"]).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def _feature_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = torch.load(
            self.feature_root / f"{row['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        expected = {
            "layers": (49,),
            "context_id": str(row["context_id"]),
            "action_horizon": self.action_horizon,
            "capture_token_count": 32,
            "capture_token_strategy": CACHE_STRATEGY,
            "backbone": CACHE_BACKBONE,
            "quantization": CACHE_QUANTIZATION,
            "manifest_items": self.source_items,
        }
        for key, expected_value in expected.items():
            actual = payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"feature cache contract mismatch for {row['id']}: "
                    f"{key}={actual!r}, expected {expected_value!r}"
                )
        if not math.isclose(float(payload.get("timestep", -1.0)), 1.0):
            raise ValueError(f"feature cache timestep must be 1.0 for {row['id']}")
        features = payload["features"]
        if tuple(features.shape) != (1, 32, 5376):
            raise ValueError(f"unexpected H3 feature shape for {row['id']}: {features.shape}")
        if not bool(torch.isfinite(features.float()).all()):
            raise ValueError(f"non-finite H3 feature for {row['id']}")
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        future_start = int(row["start"]) + self.action_horizon
        future_row = self.future_index.get(temporal_key(row, start=future_start))
        if future_row is None:
            raise ValueError(f"future H3 row is absent for {row['id']}")
        current_payload = self._feature_payload(row)
        future_payload = self._feature_payload(future_row)
        current_checkpoint = Path(current_payload["checkpoint"]).resolve()
        future_checkpoint = Path(future_payload["checkpoint"]).resolve()
        if current_checkpoint != self.h3_checkpoint_path or future_checkpoint != self.h3_checkpoint_path:
            raise ValueError(f"mixed H3 checkpoint identities for {row['id']}")
        window = torch.load(
            self.cache_root / "windows" / f"{row['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"unexpected action shape for {row['id']}: {actions.shape}")
        if bool(action_is_pad.any()):
            raise ValueError(f"future-H3 canary requires unpadded actions: {row['id']}")
        return {
            "sample_id": str(row["id"]),
            "future_sample_id": str(future_row["id"]),
            "features": current_payload["features"],
            "future_features": future_payload["features"],
            "actions": normalize_minmax(actions, self.action_min, self.action_max),
            "current_proprio": normalize_minmax(
                window["state"].float(), self.state_min, self.state_max
            ),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "future_sample_ids": [str(item["future_sample_id"]) for item in items],
        "features": torch.stack([item["features"] for item in items]),
        "future_features": torch.stack([item["future_features"] for item in items]),
        "actions": torch.stack([item["actions"] for item in items]),
        "current_proprio": torch.stack([item["current_proprio"] for item in items]),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        **{
            key: batch[key].to(device=device, non_blocking=True)
            for key in ("features", "future_features", "actions", "current_proprio")
        },
    }


@torch.inference_mode()
def evaluate(
    models: dict[str, FutureH3ConsequenceModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    for model in models.values():
        model.eval()
    sums = Counter()
    elements = 0
    items = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        inputs = {
            "conditioned_true": (models["conditioned"], actions_for_arm(batch["actions"], "conditioned")),
            "conditioned_shuffled": (models["conditioned"], actions_for_arm(batch["actions"], "shuffled")),
            "shuffled_train_true": (models["shuffled"], actions_for_arm(batch["actions"], "conditioned")),
            "independent": (models["independent"], actions_for_arm(batch["actions"], "independent")),
        }
        for name, (model, actions) in inputs.items():
            prediction = model(batch["current_proprio"], batch["features"], actions)
            target = model.project_features(batch["future_features"])
            sums[name] += float((prediction.float() - target.float()).square().sum())
        batch_items = int(batch["actions"].shape[0])
        items += batch_items
        elements += batch_items * int(models["conditioned"].target_dim)
    return {name: {"projected_mse": sums[name] / elements} for name in sums} | {
        "samples": items,
        "projection_dimensions": int(models["conditioned"].target_dim),
    }


def train(
    models: dict[str, FutureH3ConsequenceModel],
    loader: DataLoader,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for name, model in models.items()
    }
    iterator = iter(loader)
    history = []
    for step in range(1, steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        item: dict[str, Any] = {"step": step, "sample_ids": batch["sample_ids"], "loss": {}, "grad_norm": {}}
        for name in ARMS:
            model = models[name]
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            _, loss = future_h3_mse(
                model,
                current_proprio=batch["current_proprio"],
                h3_features=batch["features"],
                candidate_actions=actions_for_arm(batch["actions"], name),
                future_h3_features=batch["future_features"],
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {name} loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0.0:
                raise FloatingPointError(f"non-finite/zero {name} gradient at step {step}")
            optimizer.step()
            item["loss"][name] = float(loss.detach())
            item["grad_norm"][name] = float(grad_norm.detach())
        if step == 1 or step == steps or step % 25 == 0:
            history.append(item)
            print(json.dumps(item), flush=True)
    return history


def fresh_restore_probe(
    models: dict[str, FutureH3ConsequenceModel | TemporalFutureH3ConsequenceModel],
    batch: dict[str, Any],
    *,
    model_kwargs: dict[str, Any],
    model_class: type[FutureH3ConsequenceModel] | type[TemporalFutureH3ConsequenceModel],
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    batch = move_batch(batch, device)
    states = {
        name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    max_abs = 0.0
    with torch.inference_mode():
        for name, model in models.items():
            actions = actions_for_arm(batch["actions"], name)
            expected = model(batch["current_proprio"], batch["features"], actions).cpu()
            restored = model_class(**model_kwargs).to(device)
            restored.load_state_dict(states[name], strict=True)
            actual = restored(batch["current_proprio"], batch["features"], actions).cpu()
            max_abs = max(max_abs, float((expected - actual).abs().max()))
    return states, max_abs


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("refusing to overwrite future-H3 report or checkpoint")
    if min(args.steps, args.train_limit, args.val_limit, args.batch_size, args.action_horizon, args.target_dim, args.hidden_dim, args.actions_per_latent, args.num_heads) <= 0:
        raise ValueError("steps, limits, batch size and dimensions must be positive")
    if args.batch_size < 2 or args.train_limit % args.batch_size or args.val_limit % args.batch_size:
        raise ValueError("limits must be divisible by batch-size >=2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    source_manifest = args.source_manifest.resolve()
    cache_root = args.cache_root.resolve()
    all_train_rows = read_jsonl(train_manifest)
    all_val_rows = read_jsonl(val_manifest)
    source_rows = read_jsonl(source_manifest)
    split_audit = audit_manifests(all_train_rows, all_val_rows, source_rows)
    future_index = build_future_index(source_rows)
    eligible_train = eligible_rows(all_train_rows, future_index, action_horizon=args.action_horizon)
    eligible_val = eligible_rows(all_val_rows, future_index, action_horizon=args.action_horizon)
    train_rows = select_rows(eligible_train, limit=args.train_limit, salt=f"{args.selection_salt}|train")
    val_rows = select_rows(eligible_val, limit=args.val_limit, salt=f"{args.selection_salt}|validation")
    dataset_kwargs = {
        "source_rows": source_rows,
        "cache_root": cache_root,
        "feature_subdir": args.feature_subdir,
        "action_horizon": args.action_horizon,
    }
    train_dataset = FutureH3Dataset(train_rows, **dataset_kwargs)
    val_dataset = FutureH3Dataset(val_rows, **dataset_kwargs)
    if train_dataset.h3_checkpoint_path != val_dataset.h3_checkpoint_path:
        raise ValueError("train/validation caches use different H3 checkpoints")
    actual_h3_sha = None
    if args.verify_h3_checkpoint_sha256:
        actual_h3_sha = sha256_file(train_dataset.h3_checkpoint_path)
        if actual_h3_sha != args.expected_h3_checkpoint_sha256:
            raise ValueError("H3 checkpoint SHA256 mismatch")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, drop_last=True, collate_fn=collate_batch, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_batch, pin_memory=device.type == "cuda")
    model_kwargs = {
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": args.action_horizon,
        "h3_feature_dim": 5376,
        "target_dim": args.target_dim,
        "hidden_dim": args.hidden_dim,
        "feature_input_scale": args.feature_input_scale,
        "projection_seed": args.projection_seed,
    }
    if args.model_variant == "temporal":
        model_class = TemporalFutureH3ConsequenceModel
        model_kwargs.update({
            "actions_per_latent": args.actions_per_latent,
            "num_heads": args.num_heads,
        })
    else:
        model_class = FutureH3ConsequenceModel
    initial = model_class(**model_kwargs)
    models = {name: copy.deepcopy(initial).to(device) for name in ARMS}
    initial_metrics = evaluate(models, val_loader, device)
    started = time.perf_counter()
    history = train(models, train_loader, steps=args.steps, learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, device=device)
    final_metrics = evaluate(models, val_loader, device)
    states, restore_max_abs = fresh_restore_probe(
        models, next(iter(val_loader)), model_kwargs=model_kwargs,
        model_class=model_class, device=device,
    )

    conditioned = float(final_metrics["conditioned_true"]["projected_mse"])
    independent = float(final_metrics["independent"]["projected_mse"])
    shuffled_train = float(final_metrics["shuffled_train_true"]["projected_mse"])
    shuffled_eval = float(final_metrics["conditioned_shuffled"]["projected_mse"])
    independent_gain = relative_gain(independent, conditioned)
    shuffled_train_gain = relative_gain(shuffled_train, conditioned)
    shuffle_degradation = (shuffled_eval - conditioned) / max(conditioned, 1.0e-12)
    finite = all(math.isfinite(float(final_metrics[name]["projected_mse"])) for name in ("conditioned_true", "conditioned_shuffled", "shuffled_train_true", "independent"))
    mechanism_pass = finite and restore_max_abs == 0.0 and independent_gain >= args.minimum_relative_improvement and shuffled_train_gain >= args.minimum_relative_improvement and shuffle_degradation >= args.minimum_shuffle_degradation

    checkpoint_payload = {
        "schema_version": 1,
        "classification": "novel_composition_fact_lite_future_h3_only",
        "model_variant": args.model_variant,
        "completed_steps": args.steps,
        "model_kwargs": model_kwargs,
        "models": states,
        "contract": {
            "fact_commit": FACT_COMMIT,
            "h3_checkpoint_path": str(train_dataset.h3_checkpoint_path),
            "h3_checkpoint_sha256": actual_h3_sha or args.expected_h3_checkpoint_sha256,
            "future_target": "fixed_random_projection(mean(H3_features[start+32]))",
            "projection_trainable": False,
            "action_boundary": "candidate_actions_detached_inside_consequence_forward",
            "action_generator_present": False,
            "arms": list(ARMS),
            "model_variant": args.model_variant,
            "action_temporal_alignment": (
                f"{args.actions_per_latent}_raw_actions_per_latent_token"
                if args.model_variant == "temporal" else "flattened_32x7"
            ),
        },
    }
    atomic_torch_save(checkpoint_payload, args.checkpoint.resolve())
    report = {
        "experiment_id": "h3_fact_lite_future_h3_f1h_v1",
        "classification": "novel_composition",
        "model_variant": args.model_variant,
        "status": "PASS_MECHANISM_GATE" if mechanism_pass else "FAIL_MECHANISM_GATE",
        "claim_boundary": "Future-H3 projection mechanism only; no value, ranking, policy or LIBERO success claim.",
        "source": {
            "fact_commit": FACT_COMMIT,
            "h3_checkpoint_path": str(train_dataset.h3_checkpoint_path),
            "h3_checkpoint_sha256": actual_h3_sha or args.expected_h3_checkpoint_sha256,
            "h3_checkpoint_sha256_verified": args.verify_h3_checkpoint_sha256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest": str(val_manifest),
            "validation_manifest_sha256": sha256_file(val_manifest),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "stats_sha256": sha256_file(cache_root / "stats.pt"),
            "feature_subdir": args.feature_subdir,
        },
        "data": {
            **split_audit,
            "eligible_train": len(eligible_train),
            "eligible_validation": len(eligible_val),
            "train_selected": len(train_rows),
            "validation_selected": len(val_rows),
            "train_selected_ids_sha256": sha256_ids([str(row["id"]) for row in train_rows]),
            "validation_selected_ids_sha256": sha256_ids([str(row["id"]) for row in val_rows]),
            "train_tasks": len({str(row["task"]) for row in train_rows}),
            "validation_tasks": len({str(row["task"]) for row in val_rows}),
            "future_offset": args.action_horizon,
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "duration_seconds": time.perf_counter() - started,
            "history": history,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mechanism": {
            "conditioned_gain_over_independent": independent_gain,
            "conditioned_gain_over_shuffled_train": shuffled_train_gain,
            "conditioned_shuffle_degradation": shuffle_degradation,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "minimum_shuffle_degradation": args.minimum_shuffle_degradation,
            "all_metrics_finite": finite,
            "fresh_restore_max_abs": restore_max_abs,
            "future_to_action_generator_gradient_possible": False,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": None,
        "next": "Only a PASS permits a same-contract s500 repeat; value and ranking remain separate gated children.",
    }
    report["checkpoint_sha256"] = sha256_file(args.checkpoint.resolve())
    atomic_json_save(report, args.output.resolve())
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
