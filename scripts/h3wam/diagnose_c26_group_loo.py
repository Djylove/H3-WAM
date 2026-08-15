#!/usr/bin/env python3
"""Select C26 critic regularization using train groups only; val remains diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import train_c26_causal_action_critic as c26


STEPS_GRID = (10, 30, 100, 300)
WEIGHT_DECAY_GRID = (0.03, 0.3, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--h3-features", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--fact-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--projection-seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = {
        "dataset": args.dataset.resolve(), "h3_features": args.h3_features.resolve(),
        "stats": args.stats.resolve(), "fact_checkpoint": args.fact_checkpoint.resolve(),
    }
    dataset = torch.load(paths["dataset"], map_location="cpu", weights_only=False)
    features = torch.load(paths["h3_features"], map_location="cpu", weights_only=False)
    stats = torch.load(paths["stats"], map_location="cpu", weights_only=False)
    fact = torch.load(paths["fact_checkpoint"], map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    designs, _ = c26.build_designs(
        dataset, features, stats, fact,
        projection_dim=args.projection_dim, projection_seed=args.projection_seed,
        device=device,
    )
    group_ids = torch.tensor([int(row["group_id"]) for row in dataset["branches"]])
    labels = torch.tensor([float(row["success"]) for row in dataset["branches"]])
    train_groups = c26.mixed_group_ids(dataset, "train")
    val_groups = c26.mixed_group_ids(dataset, "val")
    arms = {}
    for arm in c26.ARMS:
        candidates = []
        for steps in STEPS_GRID:
            for weight_decay in WEIGHT_DECAY_GRID:
                fold_correct = 0.0
                fold_pairs = 0
                fold_top1 = 0
                folds = []
                for heldout in train_groups:
                    fit_groups = [group for group in train_groups if group != heldout]
                    fitted = c26.fit_pairwise_linear(
                        designs[arm], group_ids, labels, fit_groups,
                        steps=steps, learning_rate=args.learning_rate,
                        weight_decay=weight_decay, device=device,
                    )
                    scores = designs[arm] @ fitted["weights"]
                    metric = c26.ranking_metrics(scores, group_ids, labels, [heldout])
                    fold_correct += metric["pairwise_correct"]
                    fold_pairs += metric["pairwise_total"]
                    fold_top1 += metric["top1_successes"]
                    folds.append({
                        "heldout_group": heldout,
                        "pairwise_correct": metric["pairwise_correct"],
                        "pairwise_total": metric["pairwise_total"],
                        "top1_success": metric["top1_successes"] == 1,
                    })
                candidates.append({
                    "steps": steps, "weight_decay": weight_decay,
                    "pairwise_correct": fold_correct, "pairwise_total": fold_pairs,
                    "pairwise_accuracy": fold_correct / fold_pairs,
                    "top1_successes": fold_top1, "top1_total": len(train_groups),
                    "folds": folds,
                })
        selected = sorted(
            candidates,
            key=lambda row: (
                -row["pairwise_accuracy"], -row["top1_successes"],
                row["steps"], -row["weight_decay"],
            ),
        )[0]
        fitted = c26.fit_pairwise_linear(
            designs[arm], group_ids, labels, train_groups,
            steps=selected["steps"], learning_rate=args.learning_rate,
            weight_decay=selected["weight_decay"], device=device,
        )
        scores = designs[arm] @ fitted["weights"]
        old_val = c26.ranking_metrics(scores, group_ids, labels, val_groups)
        arms[arm] = {
            "selection": selected,
            "all_candidates": candidates,
            "old_c25_val_exploratory_only": old_val,
        }
    report = {
        "format": "h3wam-c26-train-group-loo-regularization-v1",
        "classification": "training_only_model_selection_after_c26_failure",
        "grid": {"steps": list(STEPS_GRID), "weight_decay": list(WEIGHT_DECAY_GRID)},
        "learning_rate": args.learning_rate,
        "train_groups": train_groups,
        "old_val_groups": val_groups,
        "arms": arms,
        "permission": "CONFIG_SELECTION_FOR_FRESH_C27_VALIDATION_ONLY",
        "boundary": (
            "C25 validation was already observed and is exploratory only. No selected "
            "configuration may be promoted without untouched C27 validation."
        ),
        "sources": {name + "_sha256": c26.sha256_file(path) for name, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        arm: {
            "selected_steps": value["selection"]["steps"],
            "selected_weight_decay": value["selection"]["weight_decay"],
            "loo_pairwise": value["selection"]["pairwise_accuracy"],
            "loo_top1": value["selection"]["top1_successes"],
            "old_val_pairwise": value["old_c25_val_exploratory_only"]["pairwise_accuracy"],
            "old_val_top1": value["old_c25_val_exploratory_only"]["top1_successes"],
        }
        for arm, value in arms.items()
    }, indent=2))


if __name__ == "__main__":
    main()
