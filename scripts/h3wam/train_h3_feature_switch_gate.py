#!/usr/bin/env python3
"""Train a phase-free recovery gate from cached H3 features and proprioception."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument(
        "--task",
        help="Optionally train the gate on one task from a multi-task manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switch-step", type=int, default=72)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_helpers():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.switch_step <= 0 or args.steps <= 0 or args.batch_size <= 1:
        raise ValueError("switch-step, steps and batch-size must be positive")
    if args.learning_rate <= 0 or args.hidden_dim <= 0:
        raise ValueError("learning-rate and hidden-dim must be positive")
    helpers = load_helpers()
    from fastwam.models.h3wam import H3FeatureSwitchGate

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache_root = args.cache_root.resolve()
    items = helpers.read_manifest(args.manifest.resolve())
    if args.task is not None:
        items = [item for item in items if str(item["task"]) == args.task]
        if not items:
            raise ValueError(f"manifest contains no rows for task {args.task!r}")
    train_items, validation_items = helpers.split_by_episode(
        items, args.val_episodes_per_task
    )
    stats = torch.load(cache_root / "stats.pt", map_location="cpu", weights_only=False)

    started = time.perf_counter()
    pooled_by_id: dict[str, torch.Tensor] = {}
    state_by_id: dict[str, torch.Tensor] = {}
    first_feature = None
    for index, item in enumerate(items, start=1):
        item_id = str(item["id"])
        feature_artifact = torch.load(
            cache_root / args.feature_subdir / f"{item_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        feature = feature_artifact["features"]
        if feature.ndim != 3:
            raise ValueError(f"cached feature {item_id} must be [L,S,D]")
        pooled_by_id[item_id] = feature.float().mean(dim=(0, 1))
        window = torch.load(
            cache_root / "windows" / f"{item_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state_by_id[item_id] = helpers.minmax_normalize(
            window["state"].float(), stats["state_min"], stats["state_max"]
        ).reshape(8)
        if first_feature is None:
            first_feature = feature_artifact
        if index % 500 == 0 or index == len(items):
            print(
                json.dumps(
                    {
                        "cached": index,
                        "total": len(items),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    assert first_feature is not None

    def tensors(selected: list[dict]):
        features = torch.stack([pooled_by_id[str(item["id"])] for item in selected])
        states = torch.stack([state_by_id[str(item["id"])] for item in selected])
        labels = torch.tensor(
            [int(item["start"]) >= args.switch_step for item in selected],
            dtype=torch.float32,
        )
        return features, states, labels

    train_feature, train_state, train_label = tensors(train_items)
    val_feature, val_state, val_label = tensors(validation_items)
    negative_indices = torch.nonzero(train_label == 0).flatten()
    positive_indices = torch.nonzero(train_label == 1).flatten()
    if not len(negative_indices) or not len(positive_indices):
        raise ValueError("training split must contain both gate classes")

    device = torch.device("cuda")
    model = H3FeatureSwitchGate(
        h3_feature_dim=int(train_feature.shape[-1]),
        state_dim=8,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    @torch.inference_mode()
    def validate() -> dict[str, float]:
        model.eval()
        logits = []
        for offset in range(0, len(val_label), 256):
            logits.append(
                model(
                    val_feature[offset : offset + 256].to(device),
                    val_state[offset : offset + 256].to(device),
                ).cpu()
            )
        logits_tensor = torch.cat(logits)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tensor, val_label
        )
        prediction = logits_tensor >= 0
        target = val_label.bool()
        accuracy = (prediction == target).float().mean()
        positive_recall = (prediction[target].float().mean() if target.any() else torch.tensor(0.0))
        negative_recall = ((~prediction[~target]).float().mean() if (~target).any() else torch.tensor(0.0))
        model.train()
        return {
            "loss": float(loss.item()),
            "accuracy": float(accuracy.item()),
            "positive_recall": float(positive_recall.item()),
            "negative_recall": float(negative_recall.item()),
        }

    def checkpoint(step: int, metrics: dict[str, float], best_loss: float) -> dict:
        return {
            "policy_type": "h3_feature_switch_gate",
            "model": model.state_dict(),
            "normalization": stats,
            "h3_feature_dim": int(train_feature.shape[-1]),
            "state_dim": 8,
            "hidden_dim": args.hidden_dim,
            "feature_shape": tuple(first_feature["features"].shape),
            "feature_layers": tuple(first_feature["layers"]),
            "feature_subdir": args.feature_subdir,
            "training_switch_step": args.switch_step,
            "threshold": 0.5,
            "step": step,
            "validation": metrics,
            "best_validation_loss": best_loss,
        }

    best_loss = float("inf")
    last_metrics = {"loss": float("nan")}
    half = args.batch_size // 2
    model.train()
    for step in range(1, args.steps + 1):
        negative = negative_indices[
            torch.randint(len(negative_indices), (half,))
        ]
        positive = positive_indices[
            torch.randint(len(positive_indices), (args.batch_size - half,))
        ]
        indices = torch.cat((negative, positive))
        indices = indices[torch.randperm(len(indices))]
        logits = model(train_feature[indices].to(device), train_state[indices].to(device))
        target = train_label[indices].to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            last_metrics = validate()
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "validation": last_metrics,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            if last_metrics["loss"] < best_loss:
                best_loss = last_metrics["loss"]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint(step, last_metrics, best_loss),
                    args.output.with_name(
                        args.output.stem + "_best" + args.output.suffix
                    ),
                )
    torch.save(checkpoint(args.steps, last_metrics, best_loss), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_windows": len(train_items),
                "validation_windows": len(validation_items),
                "train_class_counts": {
                    "base": int((train_label == 0).sum()),
                    "recovery": int((train_label == 1).sum()),
                },
                "best_validation_loss": best_loss,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "total_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
