#!/usr/bin/env python3
"""Train/evaluate C31 action-conditioned future-H3 consequence adapters.

The source split is frozen by C30 before outcomes exist.  Future observations
are label-side only, and candidate actions are detached inside the model.  A
correct-action arm is compared with within-state shuffled-action and zero-action
controls on previously unseen source episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from fastwam.models.h3wam.fact_lite_consequence import (
    FutureH3ConsequenceModel,
    TemporalFutureH3ConsequenceModel,
)


DATA_FORMAT = "h3wam-c31-action-conditioned-consequence-dataset-v1"
FEATURE_FORMAT = "h3wam-c31-live-h3-consequence-features-v1"
FORMATS = {
    DATA_FORMAT: FEATURE_FORMAT,
    "h3wam-c34-combined-consequence-ranking-dataset-v1":
        "h3wam-c34-live-h3-consequence-features-v1",
}
ARMS = ("conditioned", "shuffled", "independent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("flattened", "temporal"), required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actions-per-latent", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection-seed", type=int, default=20260815)
    parser.add_argument("--feature-input-scale", type=float, default=0.009606920816877307)
    parser.add_argument(
        "--target-error-scaling",
        choices=("raw", "train_delta_std"),
        default="raw",
        help="Optionally weight future-current errors by train-only per-dimension std.",
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    parser.add_argument("--minimum-shuffle-degradation", type=float, default=0.01)
    parser.add_argument("--condition-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--mechanism-gate", choices=("separate_independent", "paired_null"),
        default="separate_independent",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experiment-id", default="h3_c31_action_conditioned_consequence_v1"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def atomic_json_save(payload: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def model_spec(args: argparse.Namespace) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": 32,
        "h3_feature_dim": 5376,
        "target_dim": args.target_dim,
        "hidden_dim": args.hidden_dim,
        "feature_input_scale": args.feature_input_scale,
        "projection_seed": args.projection_seed,
    }
    if args.model_variant == "temporal":
        kwargs.update({
            "actions_per_latent": args.actions_per_latent,
            "num_heads": args.num_heads,
        })
        return TemporalFutureH3ConsequenceModel, kwargs
    return FutureH3ConsequenceModel, kwargs


def fixed_project_all(
    model: torch.nn.Module, raw_features: torch.Tensor, *, chunk_size: int = 64
) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(raw_features), chunk_size):
            outputs.append(model.project_features(raw_features[start:start + chunk_size]).cpu())
    return torch.cat(outputs)


class C31Dataset(Dataset):
    def __init__(
        self,
        indices: list[int],
        *,
        states: list[dict],
        branches: list[dict],
        current_projected: torch.Tensor,
        future_projected: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        shuffled_ordinals: dict[int, int],
    ) -> None:
        self.indices = indices
        self.states = states
        self.branches = branches
        self.current_projected = current_projected
        self.future_projected = future_projected
        self.state_mean = state_mean
        self.state_std = state_std
        self.shuffled_ordinals = shuffled_ordinals

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ordinal = self.indices[index]
        branch = self.branches[ordinal]
        group_id = int(branch["group_id"])
        state = self.states[group_id]
        actions = branch["environment_actions"].float()
        is_pad = branch["action_is_pad"].bool()
        if actions.shape != (32, 7) or is_pad.shape != (32,):
            raise ValueError(f"invalid action contract for branch {ordinal}")
        if bool((actions[is_pad] != 0).any()):
            raise ValueError(f"unexecuted action tail is not zero for branch {ordinal}")
        shuffled = self.branches[self.shuffled_ordinals[ordinal]]["environment_actions"].float()
        return {
            "ordinal": ordinal,
            "group_id": group_id,
            "suite": str(state["suite"]),
            "current_proprio": (state["proprio"].float() - self.state_mean) / self.state_std,
            "current_projected": self.current_projected[group_id],
            "future_projected": self.future_projected[ordinal],
            "actions": actions,
            "shuffled_actions": shuffled,
        }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ordinals": [item["ordinal"] for item in items],
        "group_ids": [item["group_id"] for item in items],
        "suites": [item["suite"] for item in items],
        **{
            key: torch.stack([item[key] for item in items])
            for key in (
                "current_proprio", "current_projected", "future_projected",
                "actions", "shuffled_actions",
            )
        },
    }


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        **{
            key: batch[key].to(device=device, non_blocking=True)
            for key in (
                "current_proprio", "current_projected", "future_projected",
                "actions", "shuffled_actions",
            )
        },
    }


def arm_actions(batch: dict[str, Any], arm: str) -> torch.Tensor:
    if arm == "conditioned":
        return batch["actions"]
    if arm == "shuffled":
        return batch["shuffled_actions"]
    if arm == "independent":
        return torch.zeros_like(batch["actions"])
    raise ValueError(arm)


@torch.inference_mode()
def evaluate(
    models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device,
    target_error_scale: torch.Tensor,
) -> dict[str, Any]:
    for model in models.values():
        model.eval()
    sums: dict[str, float] = defaultdict(float)
    raw_sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    suite_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    suite_counts: dict[str, int] = defaultdict(int)
    for raw in loader:
        batch = move(raw, device)
        probes = {
            "conditioned_true": (models["conditioned"], batch["actions"]),
            "conditioned_with_zero": (
                models["conditioned"], torch.zeros_like(batch["actions"])
            ),
            "conditioned_within_state_shuffled": (
                models["conditioned"], batch["shuffled_actions"]
            ),
            "shuffled_train_true": (models["shuffled"], batch["actions"]),
            "shuffled_train_mismatch": (
                models["shuffled"], batch["shuffled_actions"]
            ),
            "independent": (models["independent"], torch.zeros_like(batch["actions"])),
        }
        squared: dict[str, torch.Tensor] = {}
        for name, (model, actions) in probes.items():
            prediction = model.forward_projected(
                batch["current_proprio"], batch["current_projected"], actions
            )
            raw_error = prediction.float() - batch["future_projected"].float()
            raw_squared = raw_error.square().mean(1)
            squared[name] = (raw_error / target_error_scale).square().mean(1)
            sums[name] += float(squared[name].sum())
            raw_sums[name] += float(raw_squared.sum())
            counts[name] += int(len(squared[name]))
        for row, suite in enumerate(batch["suites"]):
            suite_counts[suite] += 1
            for name in probes:
                suite_sums[suite][name] += float(squared[name][row])
    return {
        "samples": sum(suite_counts.values()),
        "mse": {name: sums[name] / counts[name] for name in sorted(sums)},
        "raw_mse": {name: raw_sums[name] / counts[name] for name in sorted(raw_sums)},
        "per_suite_mse": {
            suite: {
                name: values[name] / suite_counts[suite] for name in sorted(values)
            }
            for suite, values in sorted(suite_sums.items())
        },
    }


def checkpoint_payload(
    *, args: argparse.Namespace, step: int, models: dict[str, torch.nn.Module],
    model_kwargs: dict[str, Any], state_mean: torch.Tensor, state_std: torch.Tensor,
    dataset_sha: str, feature_sha: str, target_error_scale: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "c31_action_conditioned_future_h3",
        "model_variant": args.model_variant,
        "completed_steps": step,
        "model_kwargs": model_kwargs,
        "models": {
            name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for name, model in models.items()
        },
        "normalization": {
            "state_mean": state_mean, "state_std": state_std,
            "target_error_scaling": args.target_error_scaling,
            "target_error_scale": target_error_scale.detach().cpu(),
        },
        "contract": {
            "dataset_sha256": dataset_sha,
            "features_sha256": feature_sha,
            "future_observation_is_label_only": True,
            "candidate_actions_detached_in_model": True,
            "unexecuted_action_tail_zero_masked": True,
            "shuffle_control": "cyclic_other_candidate_within_same_exact_state",
            "condition_dropout_prob": args.condition_dropout_prob,
            "mechanism_gate": args.mechanism_gate,
            "source_split_frozen_before_c30_outcomes": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if min(args.steps, args.save_every, args.batch_size) <= 0:
        raise ValueError("steps, save-every and batch-size must be positive")
    if not 0.0 <= args.condition_dropout_prob < 1.0:
        raise ValueError("condition-dropout-prob must be in [0,1)")
    if args.steps % args.save_every:
        raise ValueError("steps must be divisible by save-every")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_path, feature_path = args.dataset.resolve(), args.features.resolve()
    dataset_sha, feature_sha = sha256_file(dataset_path), sha256_file(feature_path)
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    features = torch.load(feature_path, map_location="cpu", weights_only=False)
    if data.get("format") not in FORMATS or features.get("format") != FORMATS[data["format"]]:
        raise ValueError("C31 data/feature format mismatch")
    if features.get("dataset_sha256") != dataset_sha:
        raise ValueError("C31 feature/dataset identity mismatch")
    states, branches = data["states"], data["branches"]
    if [int(row["group_id"]) for row in states] != list(range(len(states))):
        raise ValueError("state group order mismatch")
    if [int(row["ordinal"]) for row in branches] != list(range(len(branches))):
        raise ValueError("branch order mismatch")
    kinds = list(features["sample_kinds"])
    indices = features["sample_indices"].tolist()
    expected_kinds = ["current"] * len(states) + ["future"] * len(branches)
    expected_indices = list(range(len(states))) + list(range(len(branches)))
    if kinds != expected_kinds or indices != expected_indices:
        raise ValueError("C31 feature sample order mismatch")

    model_class, model_kwargs = model_spec(args)
    initial = model_class(**model_kwargs)
    projected = fixed_project_all(initial, features["fact_layer49_hidden"])
    current_projected = projected[:len(states)]
    future_projected = projected[len(states):]
    grouped: dict[int, list[int]] = defaultdict(list)
    for branch in branches:
        grouped[int(branch["group_id"])].append(int(branch["ordinal"]))
    shuffled_ordinals: dict[int, int] = {}
    for group_id, ordinals in grouped.items():
        if len(ordinals) != 4:
            raise ValueError(f"group {group_id} must contain four candidates")
        ordered = sorted(ordinals)
        for index, ordinal in enumerate(ordered):
            shuffled_ordinals[ordinal] = ordered[(index + 1) % len(ordered)]

    train_state_ids = [
        int(s["group_id"]) for s in states if s["consequence_split"] == "train"
    ]
    val_state_ids = [
        int(s["group_id"]) for s in states if s["consequence_split"] == "validation"
    ]
    train_indices = [
        int(b["ordinal"]) for b in branches if b["consequence_split"] == "train"
    ]
    val_indices = [
        int(b["ordinal"]) for b in branches if b["consequence_split"] == "validation"
    ]
    train_sources = {str(states[group]["source_episode"]) for group in train_state_ids}
    val_sources = {str(states[group]["source_episode"]) for group in val_state_ids}
    if not train_indices or not val_indices or train_sources & val_sources:
        raise ValueError("C31 requires non-empty source-disjoint train/validation")
    train_proprio = torch.stack([states[group]["proprio"].float() for group in train_state_ids])
    state_mean = train_proprio.mean(0)
    state_std = train_proprio.std(0, correction=0).clamp_min(1.0e-6)
    if args.target_error_scaling == "train_delta_std":
        train_deltas = torch.stack([
            future_projected[ordinal]
            - current_projected[int(branches[ordinal]["group_id"])]
            for ordinal in train_indices
        ])
        target_error_scale = train_deltas.std(0, correction=0).clamp_min(1.0e-6)
    else:
        target_error_scale = torch.ones(args.target_dim)
    target_error_scale = target_error_scale.to(device=device)
    common = {
        "states": states, "branches": branches,
        "current_projected": current_projected,
        "future_projected": future_projected,
        "state_mean": state_mean, "state_std": state_std,
        "shuffled_ordinals": shuffled_ordinals,
    }
    train_dataset = C31Dataset(train_indices, **common)
    val_dataset = C31Dataset(val_indices, **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, drop_last=True, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    models = {name: copy.deepcopy(initial).to(device) for name in ARMS}
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for name, model in models.items()
    }
    initial_metrics = evaluate(models, val_loader, device, target_error_scale)
    iterator = iter(train_loader)
    history = []
    started = time.perf_counter()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw = next(iterator)
        batch = move(raw, device)
        losses, grad_norms = {}, {}
        for name in ARMS:
            model, optimizer = models[name], optimizers[name]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            actions = arm_actions(batch, name)
            if name == "conditioned" and args.condition_dropout_prob > 0:
                drop = torch.rand(
                    actions.shape[0], device=actions.device
                ) < args.condition_dropout_prob
                actions = actions.masked_fill(drop[:, None, None], 0)
            prediction = model.forward_projected(
                batch["current_proprio"], batch["current_projected"],
                actions,
            )
            error = (
                prediction.float() - batch["future_projected"].detach().float()
            ) / target_error_scale
            loss = error.square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {name} loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if not bool(torch.isfinite(grad)) or float(grad) <= 0:
                raise FloatingPointError(f"invalid {name} gradient at step {step}")
            optimizer.step()
            losses[name], grad_norms[name] = float(loss.detach()), float(grad.detach())
        if step == 1 or step % 100 == 0:
            item = {"step": step, "loss": losses, "grad_norm": grad_norms}
            history.append(item)
            print(json.dumps(item), flush=True)
        if step % args.save_every == 0:
            payload = checkpoint_payload(
                args=args, step=step, models=models, model_kwargs=model_kwargs,
                state_mean=state_mean, state_std=state_std,
                dataset_sha=dataset_sha, feature_sha=feature_sha,
                target_error_scale=target_error_scale,
            )
            atomic_torch_save(
                payload,
                args.checkpoint_dir / f"{args.model_variant}_seed{args.seed}_step{step:05d}.pt",
            )

    final_metrics = evaluate(models, val_loader, device, target_error_scale)
    restore_batch = move(next(iter(val_loader)), device)
    final_checkpoint = args.checkpoint_dir / f"{args.model_variant}_seed{args.seed}_step{args.steps:05d}.pt"
    restored_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    restore_max_abs = 0.0
    with torch.inference_mode():
        for name, model in models.items():
            restored = model_class(**model_kwargs).to(device)
            restored.load_state_dict(restored_payload["models"][name], strict=True)
            restored.eval()
            actions = arm_actions(restore_batch, name)
            expected = model.forward_projected(
                restore_batch["current_proprio"], restore_batch["current_projected"], actions
            )
            actual = restored.forward_projected(
                restore_batch["current_proprio"], restore_batch["current_projected"], actions
            )
            restore_max_abs = max(
                restore_max_abs, float((expected - actual).abs().max())
            )
    conditioned = final_metrics["mse"]["conditioned_true"]
    paired_null = final_metrics["mse"]["conditioned_with_zero"]
    independent = final_metrics["mse"]["independent"]
    shuffled_train = final_metrics["mse"]["shuffled_train_true"]
    conditioned_shuffle = final_metrics["mse"]["conditioned_within_state_shuffled"]
    independent_gain = (independent - conditioned) / max(independent, 1.0e-12)
    paired_null_gain = (paired_null - conditioned) / max(paired_null, 1.0e-12)
    shuffled_train_gain = (shuffled_train - conditioned) / max(shuffled_train, 1.0e-12)
    shuffle_degradation = (conditioned_shuffle - conditioned) / max(conditioned, 1.0e-12)
    finite = all(math.isfinite(float(value)) for value in final_metrics["mse"].values())
    gate = {
        "all_metrics_finite": finite,
        "fresh_restore_exact": restore_max_abs == 0.0,
        "conditioned_beats_independent": independent_gain >= args.minimum_relative_improvement,
        "conditioned_beats_paired_null": paired_null_gain >= args.minimum_relative_improvement,
        "conditioned_beats_shuffled_train": shuffled_train_gain >= args.minimum_relative_improvement,
        "within_state_shuffle_hurts_conditioned": shuffle_degradation >= args.minimum_shuffle_degradation,
    }
    required_gate_keys = [
        "all_metrics_finite", "fresh_restore_exact",
        "conditioned_beats_shuffled_train",
        "within_state_shuffle_hurts_conditioned",
        (
            "conditioned_beats_paired_null"
            if args.mechanism_gate == "paired_null"
            else "conditioned_beats_independent"
        ),
    ]
    passed = all(gate[key] for key in required_gate_keys)
    report = {
        "experiment_id": args.experiment_id,
        "status": "PASS_C31_ACTION_CONDITIONED_CONSEQUENCE" if passed else "FAIL_C31_ACTION_CONDITIONED_CONSEQUENCE",
        "claim_boundary": "Held-out-source future-H3 consequence mechanism only; no ranking or closed-loop claim.",
        "model_variant": args.model_variant,
        "source": {
            "dataset": str(dataset_path), "dataset_sha256": dataset_sha,
            "features": str(feature_path), "features_sha256": feature_sha,
        },
        "data": {
            "train_sources": len(train_sources), "validation_sources": len(val_sources),
            "source_overlap": len(train_sources & val_sources),
            "reserved_ranking_validation_sources": int(
                data["audit"].get(
                    "reserved_ranking_validation_sources",
                    data["audit"].get("fresh_ranking_validation_sources", 0),
                )
            ),
            "train_states": len(train_state_ids), "validation_states": len(val_state_ids),
            "train_branches": len(train_indices), "validation_branches": len(val_indices),
            "partial_action_branches": int(data["audit"]["partial_action_branches"]),
        },
        "optimization": {
            "steps": args.steps, "batch_size": args.batch_size,
            "effective_train_examples": args.steps * args.batch_size,
            "effective_epochs": args.steps * args.batch_size / len(train_indices),
            "save_every": args.save_every, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "seed": args.seed,
            "target_error_scaling": args.target_error_scaling,
            "condition_dropout_prob": args.condition_dropout_prob,
            "mechanism_gate": args.mechanism_gate,
            "target_error_scale_min": float(target_error_scale.min()),
            "target_error_scale_max": float(target_error_scale.max()),
            "duration_seconds": time.perf_counter() - started,
            "parameter_count_per_arm": sum(p.numel() for p in initial.parameters()),
            "history": history,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mechanism": {
            "conditioned_gain_over_independent": independent_gain,
            "conditioned_gain_over_paired_null": paired_null_gain,
            "conditioned_gain_over_shuffled_train": shuffled_train_gain,
            "conditioned_within_state_shuffle_degradation": shuffle_degradation,
            "fresh_restore_max_abs": restore_max_abs,
            "thresholds": {
                "relative_improvement": args.minimum_relative_improvement,
                "shuffle_degradation": args.minimum_shuffle_degradation,
            },
            "required_gate_keys": required_gate_keys,
            "gate": gate,
        },
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "next": "Only repeated-seed PASS permits frozen-consequence value ranking on C31 mixed groups.",
    }
    atomic_json_save(report, args.output.resolve())
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
