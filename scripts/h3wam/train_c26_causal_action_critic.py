#!/usr/bin/env python3
"""Train tiny frozen-parent C26 critics on causal first-action outcomes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from fastwam.models.h3wam import libero_dataset_actions, minmax_normalize
from fastwam.models.h3wam.fact_lite_consequence import FutureH3ConsequenceModel


FORMAT = "h3wam-c26-causal-action-critic-v1"
FACT_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
ARMS = ("action_only", "h3_interaction", "fact_consequence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--h3-features", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--fact-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--projection-seed", type=int, default=20260815)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=(71, 73, 79, 83, 89))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_fit(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.double().mean(dim=0).float()
    std = values.double().std(dim=0, unbiased=False).float().clamp_min(1e-5)
    return mean, std


def standardize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values.float() - mean) / std


def fixed_projection(width: int, output: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(width, output, generator=generator) / math.sqrt(width)


def mixed_group_ids(dataset: dict, split: str) -> list[int]:
    return [
        int(state["group_id"])
        for state in dataset["states"]
        if state["split"] == split and state["mixed_outcomes"]
    ]


def pair_indices(
    group_ids: torch.Tensor, labels: torch.Tensor, selected_groups: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    positive, negative = [], []
    for group_id in selected_groups:
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten().tolist()
        successes = [index for index in members if labels[index] > 0.5]
        failures = [index for index in members if labels[index] < 0.5]
        if not successes or not failures:
            raise ValueError(f"selected group {group_id} is not mixed")
        for success in successes:
            for failure in failures:
                positive.append(success)
                negative.append(failure)
    return torch.tensor(positive), torch.tensor(negative)


def fit_pairwise_linear(
    design: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    selected_groups: list[int],
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    success, failure = pair_indices(group_ids, labels, selected_groups)
    differences = design[success] - design[failure]
    balanced = torch.cat((differences, -differences), dim=0).to(device)
    targets = torch.cat((torch.ones(len(differences)), torch.zeros(len(differences)))).to(device)
    weights = torch.zeros(design.shape[1], device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        (weights,), lr=learning_rate, weight_decay=weight_decay
    )
    started = time.perf_counter()
    initial_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(balanced @ weights, targets)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    final_loss = float(
        F.binary_cross_entropy_with_logits(balanced @ weights, targets).detach()
    )
    return {
        "weights": weights.detach().cpu(),
        "positive_pairs": len(differences),
        "balanced_examples_per_epoch": len(balanced),
        "steps": steps,
        "effective_pair_epochs": float(steps),
        "examples_seen": int(steps * len(balanced)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "duration_seconds": time.perf_counter() - started,
    }


def ranking_metrics(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    selected_groups: list[int],
) -> dict:
    correct = 0.0
    pairs = 0
    top1 = 0
    margins = []
    groups = []
    for group_id in selected_groups:
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        successes = members[labels[members] > 0.5]
        failures = members[labels[members] < 0.5]
        group_correct = 0.0
        group_pairs = 0
        for success in successes:
            for failure in failures:
                margin = float(scores[success] - scores[failure])
                group_correct += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
                group_pairs += 1
                margins.append(margin)
        winner = int(members[torch.argmax(scores[members])])
        winner_success = bool(labels[winner] > 0.5)
        top1 += int(winner_success)
        correct += group_correct
        pairs += group_pairs
        groups.append({
            "group_id": group_id,
            "successes": len(successes),
            "pairwise_correct": group_correct,
            "pairwise_total": group_pairs,
            "top1_success": winner_success,
            "score_range": float(scores[members].max() - scores[members].min()),
        })
    return {
        "pairwise_correct": correct,
        "pairwise_total": pairs,
        "pairwise_accuracy": correct / pairs,
        "top1_successes": top1,
        "top1_total": len(selected_groups),
        "top1_success_rate": top1 / len(selected_groups),
        "mean_success_failure_margin": sum(margins) / len(margins),
        "groups": groups,
    }


def exact_label_permutation_pvalue(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    selected_groups: list[int],
    observed_correct: float,
) -> tuple[float, int]:
    group_assignments = []
    for group_id in selected_groups:
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten().tolist()
        successes = int(labels[members].sum().item())
        group_assignments.append(
            [set(combo) for combo in itertools.combinations(members, successes)]
        )
    at_least = 0
    total = 0
    for assignment in itertools.product(*group_assignments):
        permuted = torch.zeros_like(labels)
        for successes in assignment:
            for index in successes:
                permuted[index] = 1.0
        metric = ranking_metrics(scores, group_ids, permuted, selected_groups)
        at_least += int(metric["pairwise_correct"] >= observed_correct)
        total += 1
    return at_least / total, total


def shuffled_train_labels(
    labels: torch.Tensor, group_ids: torch.Tensor, groups: list[int], seed: int
) -> torch.Tensor:
    result = labels.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for group_id in groups:
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        permutation = members[torch.randperm(len(members), generator=generator)]
        result[members] = labels[permutation]
    return result


def build_designs(
    dataset: dict,
    features: dict,
    stats: dict,
    fact_payload: dict,
    *,
    projection_dim: int,
    projection_seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict]:
    branches = dataset["branches"]
    group_ids = torch.tensor([int(row["group_id"]) for row in branches])
    train_rows = torch.tensor([row["split"] == "train" for row in branches])
    actions_environment = torch.stack([row["environment_actions"] for row in branches])
    actions_dataset = libero_dataset_actions(actions_environment)
    actions_normalized = minmax_normalize(
        actions_dataset, stats["action_min"], stats["action_max"]
    ).clamp(-5.0, 5.0)
    action_flat = actions_normalized.flatten(1)
    action_mean, action_std = standardize_fit(action_flat[train_rows])
    action_projection = fixed_projection(
        action_flat.shape[1], projection_dim, projection_seed
    )
    action_design = standardize(action_flat, action_mean, action_std) @ action_projection

    compact = features["d0_layer49_kv_compact"].float()
    proprio = torch.stack([state["proprio"] for state in dataset["states"]])
    proprio_normalized = minmax_normalize(
        proprio, stats["state_min"], stats["state_max"]
    ).clamp(-5.0, 5.0)
    state_raw = torch.cat((compact, proprio_normalized), dim=1)
    train_state = torch.tensor([state["split"] == "train" for state in dataset["states"]])
    state_mean, state_std = standardize_fit(state_raw[train_state])
    state_projection = fixed_projection(
        state_raw.shape[1], projection_dim, projection_seed + 1
    )
    state_design = standardize(state_raw, state_mean, state_std) @ state_projection
    interaction = action_design * state_design[group_ids]

    if fact_payload.get("contract", {}).get("fact_commit") != FACT_COMMIT:
        raise ValueError("FACT consequence checkpoint commit mismatch")
    fact_model = FutureH3ConsequenceModel(**fact_payload["model_kwargs"]).to(device)
    fact_model.load_state_dict(fact_payload["models"]["conditioned"], strict=True)
    fact_model.requires_grad_(False).eval()
    hidden = features["fact_layer49_hidden"].float()[group_ids]
    current_proprio = proprio_normalized[group_ids]
    consequence_batches = []
    with torch.inference_mode():
        for start in range(0, len(branches), 32):
            stop = min(start + 32, len(branches))
            prediction = fact_model(
                current_proprio[start:stop].to(device),
                hidden[start:stop].to(device),
                actions_normalized[start:stop].to(device),
            )
            current = fact_model.project_features(hidden[start:stop].to(device))
            consequence_batches.append((prediction - current).cpu())
    consequence = torch.cat(consequence_batches)
    consequence_mean, consequence_std = standardize_fit(consequence[train_rows])
    consequence_projection = fixed_projection(
        consequence.shape[1], projection_dim, projection_seed + 2
    )
    consequence_design = (
        standardize(consequence, consequence_mean, consequence_std)
        @ consequence_projection
    )

    designs = {
        "action_only": action_design,
        "h3_interaction": torch.cat((action_design, interaction), dim=1),
        "fact_consequence": torch.cat((action_design, consequence_design), dim=1),
    }
    contract = {
        "environment_to_dataset_action": "motion_identity; gripper=(1-env)/2",
        "normalization": "C25 executed action -> v7 minmax clip5",
        "action_projection": action_projection,
        "state_projection": state_projection,
        "consequence_projection": consequence_projection,
        "action_mean": action_mean, "action_std": action_std,
        "state_mean": state_mean, "state_std": state_std,
        "consequence_mean": consequence_mean, "consequence_std": consequence_std,
        "projection_seed": projection_seed, "projection_dim": projection_dim,
        "fact_model_frozen": True, "h3_parent_frozen": True, "action_parent_frozen": True,
    }
    return designs, contract


def atomic_torch_save(payload: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    if min(args.steps, args.projection_dim) <= 0:
        raise ValueError("steps and projection-dim must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimization arguments")
    paths = {
        "dataset": args.dataset.resolve(), "h3_features": args.h3_features.resolve(),
        "stats": args.stats.resolve(), "fact_checkpoint": args.fact_checkpoint.resolve(),
    }
    dataset = torch.load(paths["dataset"], map_location="cpu", weights_only=False)
    features = torch.load(paths["h3_features"], map_location="cpu", weights_only=False)
    stats = torch.load(paths["stats"], map_location="cpu", weights_only=False)
    fact_payload = torch.load(paths["fact_checkpoint"], map_location="cpu", weights_only=False)
    if dataset.get("format") != "h3wam-c26-causal-critic-dataset-v1":
        raise ValueError("C26 dataset format mismatch")
    if features.get("format") != "h3wam-c26-live-h3-features-v1":
        raise ValueError("C26 H3 feature format mismatch")
    if features.get("dataset_sha256") != sha256_file(paths["dataset"]):
        raise ValueError("C26 feature/dataset identity mismatch")
    if torch.any(features["group_ids"] != torch.arange(32)):
        raise ValueError("C26 feature group order mismatch")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    designs, design_contract = build_designs(
        dataset, features, stats, fact_payload,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        device=device,
    )
    group_ids = torch.tensor([int(row["group_id"]) for row in dataset["branches"]])
    labels = torch.tensor([float(row["success"]) for row in dataset["branches"]])
    train_groups = mixed_group_ids(dataset, "train")
    val_groups = mixed_group_ids(dataset, "val")
    arms = {}
    weights = {}
    for arm in ARMS:
        fitted = fit_pairwise_linear(
            designs[arm], group_ids, labels, train_groups,
            steps=args.steps, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, device=device,
        )
        scores = designs[arm] @ fitted["weights"]
        train_metrics = ranking_metrics(scores, group_ids, labels, train_groups)
        val_metrics = ranking_metrics(scores, group_ids, labels, val_groups)
        pvalue, permutations = exact_label_permutation_pvalue(
            scores, group_ids, labels, val_groups, val_metrics["pairwise_correct"]
        )
        shuffle_metrics = []
        for seed in args.shuffle_seeds:
            shuffled = shuffled_train_labels(labels, group_ids, train_groups, seed)
            control = fit_pairwise_linear(
                designs[arm], group_ids, shuffled, train_groups,
                steps=args.steps, learning_rate=args.learning_rate,
                weight_decay=args.weight_decay, device=device,
            )
            control_scores = designs[arm] @ control["weights"]
            shuffle_metrics.append(
                ranking_metrics(control_scores, group_ids, labels, val_groups)
            )
        arms[arm] = {
            "design_dim": designs[arm].shape[1],
            "optimization": {key: value for key, value in fitted.items() if key != "weights"},
            "train": train_metrics,
            "val": {**val_metrics, "exact_permutation_pvalue": pvalue,
                    "exact_permutations": permutations},
            "shuffled_train_controls": shuffle_metrics,
            "shuffle_mean_pairwise_accuracy": sum(
                item["pairwise_accuracy"] for item in shuffle_metrics
            ) / len(shuffle_metrics),
        }
        weights[arm] = fitted["weights"]

    baseline = arms["action_only"]["val"]
    for arm in ("h3_interaction", "fact_consequence"):
        val = arms[arm]["val"]
        arms[arm]["gate"] = {
            "pairwise_at_least_6_of_9": val["pairwise_correct"] >= 6,
            "top1_all_heldout_groups": val["top1_successes"] == 3,
            "beats_action_only_by_one_pair": (
                val["pairwise_correct"] >= baseline["pairwise_correct"] + 1
            ),
            "exact_permutation_p_at_most_0_10": val["exact_permutation_pvalue"] <= 0.10,
            "beats_shuffle_mean_by_one_pair": (
                val["pairwise_accuracy"]
                >= arms[arm]["shuffle_mean_pairwise_accuracy"] + 1 / 9
            ),
            "train_pairwise_at_least_0_80": arms[arm]["train"]["pairwise_accuracy"] >= 0.80,
        }
        arms[arm]["gate"]["passed"] = all(arms[arm]["gate"].values())
    winners = [arm for arm in ("h3_interaction", "fact_consequence") if arms[arm]["gate"]["passed"]]
    status = "PASS_C26_HELDOUT_WITHIN_STATE_RANKING" if winners else "FAIL_C26_HELDOUT_WITHIN_STATE_RANKING"
    report = {
        "format": FORMAT,
        "hypothesis": (
            "A frozen H3-conditioned or FACT-consequence critic ranks successful first-action "
            "chunks above failed alternatives in unseen source episodes, beyond an action-only shortcut."
        ),
        "status": status,
        "training_permission": "GO_BEST_OF_N_CLOSED_LOOP_CANARY" if winners else "NO_GO_BEST_OF_N",
        "effect_conclusion": "NOT_EVIDENCE_READY_UNTIL_CLOSED_LOOP" if winners else "REJECT_CURRENT_CRITIC",
        "winners": winners,
        "data": {
            "branches": 128, "groups": 32,
            "train_mixed_groups": train_groups, "val_mixed_groups": val_groups,
            "train_positive_pairs": 21, "val_positive_pairs": 9,
            "source_episode_split_isolated": True,
        },
        "budget": {
            "optimizer_steps_per_arm": args.steps,
            "effective_pair_epochs": float(args.steps),
            "balanced_pair_examples_per_epoch": 42,
            "examples_seen_per_real_arm": 42 * args.steps,
            "shuffle_control_seeds": list(args.shuffle_seeds),
            "parent_parameters_updated": 0,
        },
        "sources": {
            "fact_commit": FACT_COMMIT,
            **{name + "_sha256": sha256_file(path) for name, path in paths.items()},
        },
        "arms": arms,
        "boundary": (
            "Passing permits only a fixed-parent N=1 versus N=4 closed-loop canary. "
            "It does not yet prove LIBERO success-rate improvement or general WAM capability."
        ),
    }
    checkpoint = {
        "format": FORMAT,
        "status": status,
        "winners": winners,
        "weights": weights,
        "design_contract": design_contract,
        "sources": report["sources"],
        "arms": arms,
    }
    atomic_torch_save(checkpoint, args.checkpoint)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": status, "training_permission": report["training_permission"],
        "winners": winners,
        "val": {arm: {
            "pairs": arms[arm]["val"]["pairwise_correct"],
            "top1": arms[arm]["val"]["top1_successes"],
            "p": arms[arm]["val"]["exact_permutation_pvalue"],
        } for arm in ARMS},
    }, indent=2))


if __name__ == "__main__":
    main()
