#!/usr/bin/env python3
"""Confirm the train-only-selected H3 action critic on untouched C27 episodes."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path

import torch

import train_c26_causal_action_critic as c26


FORMAT = "h3wam-c28-fresh-causal-action-critic-v1"
SELECTED_STEPS = 10
SELECTED_WEIGHT_DECAY = 3.0
SELECTED_LEARNING_RATE = 0.03
SELECTED_PROJECTION_DIM = 32
SELECTED_PROJECTION_SEED = 20260815
SHUFFLE_SEEDS = (71, 73, 79, 83, 89)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c25-dataset", type=Path, required=True)
    parser.add_argument("--c25-features", type=Path, required=True)
    parser.add_argument("--c27-dataset", type=Path, required=True)
    parser.add_argument("--c27-features", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--fact-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def combine_datasets(c25: dict, c27: dict) -> dict:
    if c25.get("format") != "h3wam-c26-causal-critic-dataset-v1":
        raise ValueError("C25 critic dataset format mismatch")
    if c27.get("format") != "h3wam-c27-causal-critic-dataset-v1":
        raise ValueError("C27 critic dataset format mismatch")
    states = []
    branches = []
    for state in c25["states"]:
        row = copy.copy(state)
        row["split"] = "train"
        states.append(row)
    for branch in c25["branches"]:
        row = copy.copy(branch)
        row["split"] = "train"
        branches.append(row)
    offset = len(states)
    for state in c27["states"]:
        row = copy.copy(state)
        row["group_id"] = int(row["group_id"]) + offset
        states.append(row)
    for branch in c27["branches"]:
        row = copy.copy(branch)
        row["group_id"] = int(row["group_id"]) + offset
        branches.append(row)
    if [int(row["group_id"]) for row in states] != list(range(len(states))):
        raise ValueError("combined causal group ids are not contiguous")
    train_sources = {row["source_episode"] for row in states if row["split"] == "train"}
    val_sources = {row["source_episode"] for row in states if row["split"] == "val"}
    if train_sources & val_sources:
        raise ValueError("combined source episodes overlap train and validation")
    return {"states": states, "branches": branches}


def combine_features(c25: dict, c27: dict, group_count: int) -> dict:
    if c25.get("format") != "h3wam-c26-live-h3-features-v1":
        raise ValueError("C25 H3 feature format mismatch")
    if c27.get("format") != "h3wam-c27-live-h3-features-v1":
        raise ValueError("C27 H3 feature format mismatch")
    compact = torch.cat((c25["d0_layer49_kv_compact"], c27["d0_layer49_kv_compact"]))
    hidden = torch.cat((c25["fact_layer49_hidden"], c27["fact_layer49_hidden"]))
    if compact.shape[0] != group_count or hidden.shape[0] != group_count:
        raise ValueError("combined H3 features do not cover all causal groups")
    return {"d0_layer49_kv_compact": compact, "fact_layer49_hidden": hidden}


def permutation_pvalue(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    groups: list[int],
    observed_correct: float,
    *,
    maximum_exact: int = 200_000,
    monte_carlo_samples: int = 100_000,
    seed: int = 20260815,
) -> dict:
    assignment_counts = []
    for group in groups:
        members = torch.nonzero(group_ids == group, as_tuple=False).flatten()
        assignment_counts.append(math.comb(len(members), int(labels[members].sum())))
    total = math.prod(assignment_counts)
    if total <= maximum_exact:
        value, enumerated = c26.exact_label_permutation_pvalue(
            scores, group_ids, labels, groups, observed_correct
        )
        return {"value": value, "mode": "exact", "samples": enumerated, "space": total}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    at_least = 0
    for _ in range(monte_carlo_samples):
        permuted = labels.clone()
        for group in groups:
            members = torch.nonzero(group_ids == group, as_tuple=False).flatten()
            order = members[torch.randperm(len(members), generator=generator)]
            permuted[members] = labels[order]
        metric = c26.ranking_metrics(scores, group_ids, permuted, groups)
        at_least += int(metric["pairwise_correct"] >= observed_correct)
    return {
        "value": (at_least + 1) / (monte_carlo_samples + 1),
        "mode": "fixed_seed_monte_carlo", "samples": monte_carlo_samples,
        "space": total, "seed": seed,
    }


def main() -> None:
    args = parse_args()
    paths = {
        "c25_dataset": args.c25_dataset.resolve(),
        "c25_features": args.c25_features.resolve(),
        "c27_dataset": args.c27_dataset.resolve(),
        "c27_features": args.c27_features.resolve(),
        "stats": args.stats.resolve(),
        "fact_checkpoint": args.fact_checkpoint.resolve(),
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    dataset = combine_datasets(payloads["c25_dataset"], payloads["c27_dataset"])
    features = combine_features(
        payloads["c25_features"], payloads["c27_features"], len(dataset["states"])
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    designs, design_contract = c26.build_designs(
        dataset, features, payloads["stats"], payloads["fact_checkpoint"],
        projection_dim=SELECTED_PROJECTION_DIM,
        projection_seed=SELECTED_PROJECTION_SEED,
        device=device,
    )
    group_ids = torch.tensor([int(row["group_id"]) for row in dataset["branches"]])
    labels = torch.tensor([float(row["success"]) for row in dataset["branches"]])
    train_groups = c26.mixed_group_ids(dataset, "train")
    val_groups = c26.mixed_group_ids(dataset, "val")
    if len(val_groups) < 4:
        raise ValueError("C28 requires at least four fresh C27 validation mixed groups")
    arms, weights = {}, {}
    for arm in ("action_only", "h3_interaction"):
        fitted = c26.fit_pairwise_linear(
            designs[arm], group_ids, labels, train_groups,
            steps=SELECTED_STEPS, learning_rate=SELECTED_LEARNING_RATE,
            weight_decay=SELECTED_WEIGHT_DECAY, device=device,
        )
        scores = designs[arm] @ fitted["weights"]
        train_metric = c26.ranking_metrics(scores, group_ids, labels, train_groups)
        val_metric = c26.ranking_metrics(scores, group_ids, labels, val_groups)
        permutation = permutation_pvalue(
            scores, group_ids, labels, val_groups, val_metric["pairwise_correct"]
        )
        shuffle_metrics = []
        for seed in SHUFFLE_SEEDS:
            shuffled = c26.shuffled_train_labels(labels, group_ids, train_groups, seed)
            control = c26.fit_pairwise_linear(
                designs[arm], group_ids, shuffled, train_groups,
                steps=SELECTED_STEPS, learning_rate=SELECTED_LEARNING_RATE,
                weight_decay=SELECTED_WEIGHT_DECAY, device=device,
            )
            control_scores = designs[arm] @ control["weights"]
            shuffle_metrics.append(
                c26.ranking_metrics(control_scores, group_ids, labels, val_groups)
            )
        arms[arm] = {
            "optimization": {key: value for key, value in fitted.items() if key != "weights"},
            "train": train_metric,
            "val": {**val_metric, "permutation": permutation},
            "shuffle_mean_pairwise_accuracy": sum(
                row["pairwise_accuracy"] for row in shuffle_metrics
            ) / len(shuffle_metrics),
            "shuffled_train_controls": shuffle_metrics,
        }
        weights[arm] = fitted["weights"]
    baseline = arms["action_only"]["val"]
    candidate = arms["h3_interaction"]
    val = candidate["val"]
    gate = {
        "pairwise_accuracy_at_least_0_65": val["pairwise_accuracy"] >= 0.65,
        "top1_success_rate_at_least_0_75": val["top1_success_rate"] >= 0.75,
        "beats_action_only_by_at_least_one_pair": (
            val["pairwise_correct"] >= baseline["pairwise_correct"] + 1
        ),
        "beats_action_only_by_at_least_0_10": (
            val["pairwise_accuracy"] >= baseline["pairwise_accuracy"] + 0.10
        ),
        "permutation_p_at_most_0_10": val["permutation"]["value"] <= 0.10,
        "beats_shuffle_mean_by_at_least_0_10": (
            val["pairwise_accuracy"]
            >= candidate["shuffle_mean_pairwise_accuracy"] + 0.10
        ),
        "all_validation_groups_have_score_variation": all(
            row["score_range"] > 1e-8 for row in val["groups"]
        ),
    }
    gate["passed"] = all(gate.values())
    status = (
        "PASS_C28_FRESH_HELDOUT_WITHIN_STATE_RANKING"
        if gate["passed"] else "FAIL_C28_FRESH_HELDOUT_WITHIN_STATE_RANKING"
    )
    train_pairs = arms["h3_interaction"]["optimization"]["positive_pairs"]
    report = {
        "format": FORMAT,
        "hypothesis": (
            "The C25-train-only LOO-selected frozen H3-action interaction critic "
            "outperforms an action-only shortcut on untouched C27 source episodes."
        ),
        "status": status,
        "training_permission": (
            "GO_BEST_OF_N_CLOSED_LOOP_CANARY" if gate["passed"] else "NO_GO_BEST_OF_N"
        ),
        "effect_conclusion": "NOT_EVIDENCE_READY_UNTIL_CLOSED_LOOP",
        "preselected_config": {
            "selection_artifact": "/mnt/h3-wam/eval/c26-causal-critic-v1/train_group_loo.json",
            "steps": SELECTED_STEPS, "learning_rate": SELECTED_LEARNING_RATE,
            "weight_decay": SELECTED_WEIGHT_DECAY,
            "projection_dim": SELECTED_PROJECTION_DIM,
            "projection_seed": SELECTED_PROJECTION_SEED,
        },
        "data": {
            "c25_groups_reclassified_train_only": len(payloads["c25_dataset"]["states"]),
            "c27_train_groups": sum(row["split"] == "train" for row in payloads["c27_dataset"]["states"]),
            "c27_fresh_val_groups": sum(row["split"] == "val" for row in payloads["c27_dataset"]["states"]),
            "train_mixed_groups": train_groups, "fresh_val_mixed_groups": val_groups,
            "train_positive_pairs": train_pairs,
            "fresh_val_positive_pairs": val["pairwise_total"],
        },
        "budget": {
            "optimizer_steps_per_arm": SELECTED_STEPS,
            "effective_pair_epochs": float(SELECTED_STEPS),
            "balanced_pair_examples_per_epoch": 2 * train_pairs,
            "examples_seen_per_real_arm": 2 * train_pairs * SELECTED_STEPS,
            "parent_parameters_updated": 0,
        },
        "sources": {name + "_sha256": c26.sha256_file(path) for name, path in paths.items()},
        "arms": arms, "gate": gate,
        "boundary": (
            "Passing permits only a fixed-parent N=1 versus N=4 closed-loop canary. "
            "C27 validation cannot be reused for further configuration selection."
        ),
    }
    checkpoint = {
        "format": FORMAT, "status": status, "weights": weights,
        "design_contract": design_contract, "preselected_config": report["preselected_config"],
        "sources": report["sources"], "gate": gate,
    }
    c26.atomic_torch_save(checkpoint, args.checkpoint)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": status, "training_permission": report["training_permission"],
        "train_pairs": train_pairs, "val_pairs": val["pairwise_total"],
        "action_only": {"pairs": baseline["pairwise_correct"], "top1": baseline["top1_successes"]},
        "h3_interaction": {
            "pairs": val["pairwise_correct"], "top1": val["top1_successes"],
            "p": val["permutation"]["value"],
        },
        "gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
