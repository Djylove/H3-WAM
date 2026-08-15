#!/usr/bin/env python3
"""Evaluate a D0 history adapter against its exact frozen parent."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_trainer():
    path = Path(__file__).with_name("train_h3_int8_dreamwam_kv_carrier.py")
    spec = importlib.util.spec_from_file_location("_history_candidate_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=10)
    return parser.parse_args()


def sample_actions(model, batch: dict, scheduler, noise: torch.Tensor, steps: int):
    actions = noise.clone()
    timesteps, deltas = scheduler.build_inference_schedule(
        steps, actions.device, actions.dtype
    )
    for timestep, delta in zip(timesteps, deltas, strict=True):
        velocity = TRAINER.forward_policy(
            model,
            batch,
            actions,
            timestep.float().expand(actions.shape[0]),
        )
        actions = scheduler.step(velocity, delta, actions)
    return actions


class Metrics:
    def __init__(self) -> None:
        self.normalized_sq = 0.0
        self.physical_sq = 0.0
        self.count = 0
        self.gripper_correct = 0
        self.gripper_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        raw_target: torch.Tensor,
        is_pad: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ) -> None:
        valid = ~is_pad
        error = prediction - target
        self.normalized_sq += float(error[valid].float().square().sum())
        self.count += int(valid.sum()) * prediction.shape[-1]
        physical = 0.5 * (prediction.float() + 1.0) * (
            action_max - action_min
        ) + action_min
        self.physical_sq += float(
            (physical[valid] - raw_target[valid].float()).square().sum()
        )
        predicted_gripper = physical[..., -1] >= 0.5
        target_gripper = raw_target[..., -1] >= 0.5
        self.gripper_correct += int(
            (predicted_gripper[valid] == target_gripper[valid]).sum()
        )
        self.gripper_count += int(valid.sum())

    def result(self) -> dict:
        return {
            "normalized_mse": self.normalized_sq / self.count,
            "physical_mse": self.physical_sq / self.count,
            "gripper_accuracy": self.gripper_correct / self.gripper_count,
            "active_action_values": self.count,
        }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    contract = checkpoint["contract"]
    history_steps = int(contract.get("history_action_steps", 0))
    if history_steps <= 0 or contract.get("train_history_adapter_only") is not True:
        raise ValueError("checkpoint is not a history-adapter-only candidate")
    model_spec = dict(contract["model_spec"])
    model_spec["carrier_layers"] = tuple(model_spec["carrier_layers"])
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dataset = TRAINER.CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        contract["kv_subdir"],
        source_manifest=args.source_manifest,
        carrier_layers=tuple(model_spec["carrier_layers"]),
        capture_token_count=int(contract["kv_tokens"]),
        num_heads=int(model_spec["num_heads"]),
        attn_head_dim=int(model_spec["attn_head_dim"]),
        action_horizon=int(contract["action_horizon"]),
        history_action_steps=history_steps,
        executed_action_history_root=Path(contract["executed_action_history_root"]),
    )
    items = [TRAINER.collate_cached_batch([dataset[index]]) for index in range(len(dataset))]
    if len(items) < 2:
        raise ValueError("history evaluation requires at least two examples")
    child = TRAINER.build_model(
        TRAINER.ModelSpec(**model_spec), device=device, dtype=dtype
    )
    child.load_state_dict(checkpoint["model"], strict=True)
    child.eval()
    scheduler = TRAINER.FlowMatchScheduler(
        num_train_timesteps=1000, shift=float(contract["action_shift"])
    )
    modes = {
        "parent_equivalent_zero_history_input": Metrics(),
        "child_correct_history": Metrics(),
        "child_shuffled_history": Metrics(),
    }
    condition_delta = {"history": 0.0, "language": 0.0, "visual": 0.0}
    delta_values = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        for index, raw_batch in enumerate(items):
            batch = TRAINER.move_batch(raw_batch, device, dtype)
            replacement = TRAINER.move_batch(items[(index + 1) % len(items)], device, dtype)
            noise = torch.randn(
                batch["actions"].shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            zero_history_batch = dict(batch)
            zero_history_batch["executed_action_history"] = torch.zeros_like(
                batch["executed_action_history"]
            )
            zero_history_batch["executed_action_history_valid"] = torch.zeros_like(
                batch["executed_action_history_valid"]
            )
            parent_prediction = sample_actions(
                child, zero_history_batch, scheduler, noise, args.inference_steps
            )
            prediction = sample_actions(
                child, batch, scheduler, noise, args.inference_steps
            )
            shuffled_history_batch = dict(batch)
            shuffled_history_batch["executed_action_history"] = replacement[
                "executed_action_history"
            ]
            shuffled_history_batch["executed_action_history_valid"] = replacement[
                "executed_action_history_valid"
            ]
            history_prediction = sample_actions(
                child, shuffled_history_batch, scheduler, noise, args.inference_steps
            )
            language_batch = dict(batch)
            language_batch["text_context"] = replacement["text_context"]
            language_batch["text_mask"] = replacement["text_mask"]
            language_prediction = sample_actions(
                child, language_batch, scheduler, noise, args.inference_steps
            )
            visual_batch = dict(batch)
            visual_batch["video_kv_cache"] = replacement["video_kv_cache"]
            visual_prediction = sample_actions(
                child, visual_batch, scheduler, noise, args.inference_steps
            )
            raw_actions = 0.5 * (batch["actions"].float() + 1.0) * (
                dataset.action_max.to(device) - dataset.action_min.to(device)
            ) + dataset.action_min.to(device)
            for mode, value in (
                ("parent_equivalent_zero_history_input", parent_prediction),
                ("child_correct_history", prediction),
                ("child_shuffled_history", history_prediction),
            ):
                modes[mode].update(
                    value,
                    batch["actions"],
                    raw_actions,
                    batch["action_is_pad"],
                    dataset.action_min.to(device),
                    dataset.action_max.to(device),
                )
            valid = (~batch["action_is_pad"]).unsqueeze(-1)
            delta_values += int(valid.sum()) * prediction.shape[-1]
            condition_delta["history"] += float(
                ((prediction - history_prediction).abs() * valid).sum()
            )
            condition_delta["language"] += float(
                ((prediction - language_prediction).abs() * valid).sum()
            )
            condition_delta["visual"] += float(
                ((prediction - visual_prediction).abs() * valid).sum()
            )
    result = {
        "format": "h3-d0-history-adapter-eval-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "completed_steps": int(checkpoint["completed_steps"]),
        "history_action_steps": history_steps,
        "examples": len(items),
        "inference_steps": args.inference_steps,
        "seed": args.seed,
        "parent_equivalence": (
            "The adapter is bias-free and additive; an all-zero history with an all-false "
            "valid mask exactly removes the only trainable child branch while keeping one "
            "model instance and identical solver noise."
        ),
        "metrics": {name: metric.result() for name, metric in modes.items()},
        "mean_abs_condition_delta": {
            name: value / delta_values for name, value in condition_delta.items()
        },
        "evidence_boundary": "Offline held-out action and condition-response gate; not rollout success.",
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite history evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{output.stat().st_ino if output.exists() else 'partial'}")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
