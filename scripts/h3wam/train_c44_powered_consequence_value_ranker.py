#!/usr/bin/env python3
"""Fit on expanded causal sources and evaluate once on untouched C43 final groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

import train_c26_causal_action_critic as c26
import train_c28_fresh_causal_action_critic as c28
from fastwam.models.h3wam.fact_lite_consequence import TemporalFutureH3ConsequenceModel


FORMAT = "h3wam-c44-powered-consequence-value-ranker-v1"
EXPECTED_CONSEQUENCE_TRAIN_DATASET_SHA256 = "2a6c9252b8e77975f58920425bc18110fa8ea63bdc12c4c15571cfffeb9f7459"
C40_FRESH_PAIRWISE = 44 / 81
CV_STEPS = (10, 30, 100)
CV_WEIGHT_DECAYS = (0.3, 3.0, 10.0)
LEARNING_RATE = 0.03
PROJECTION_DIM = 32
PROJECTION_SEED = 20260815
SHUFFLE_SEEDS = (71, 73, 79, 83, 89, 97, 101, 103, 107, 109)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--consequence-checkpoints", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_fold(source: str, folds: int = 5) -> int:
    return int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % folds


def mixed_groups(states: list[dict], role: str) -> list[int]:
    return [
        int(row["group_id"]) for row in states
        if row["consequence_split"] == role and row["mixed_outcomes"]
    ]


def select_config(
    design: torch.Tensor,
    group_ids: torch.Tensor,
    labels: torch.Tensor,
    states: list[dict],
    train_groups: list[int],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    folds = defaultdict(list)
    for group in train_groups:
        folds[source_fold(str(states[group]["source_episode"]))].append(group)
    active = sorted(key for key, value in folds.items() if value)
    if len(active) < 3:
        raise ValueError("C44 train groups cover fewer than three source folds")
    rows = []
    for steps in CV_STEPS:
        for weight_decay in CV_WEIGHT_DECAYS:
            correct = total = 0.0
            for fold in active:
                validation = folds[fold]
                training = [group for key in active if key != fold for group in folds[key]]
                fitted = c26.fit_pairwise_linear(
                    design, group_ids, labels, training,
                    steps=steps, learning_rate=LEARNING_RATE,
                    weight_decay=weight_decay, device=device,
                )
                metric = c26.ranking_metrics(
                    design @ fitted["weights"], group_ids, labels, validation
                )
                correct += metric["pairwise_correct"]
                total += metric["pairwise_total"]
            rows.append({
                "steps": steps,
                "weight_decay": weight_decay,
                "pairwise_correct": correct,
                "pairwise_total": total,
                "pairwise_accuracy": correct / total,
            })
    selected = max(rows, key=lambda row: (
        row["pairwise_accuracy"], -row["steps"], row["weight_decay"]
    ))
    return selected, rows


def project_consequences(
    data: dict,
    features: dict,
    checkpoints: list[Path],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    states, branches = data["states"], data["branches"]
    hidden = features["fact_layer49_hidden"]
    current_hidden = hidden[:len(states)]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    current_projected = None
    predictions = []
    group_ids = torch.tensor([int(row["group_id"]) for row in branches])
    actions = torch.stack([row["environment_actions"].float() for row in branches])
    checkpoint_hashes = []
    for path, payload in zip(checkpoints, payloads, strict=True):
        if payload.get("model_variant") != "temporal":
            raise ValueError("C44 requires temporal consequence checkpoints")
        if payload["contract"]["dataset_sha256"] != EXPECTED_CONSEQUENCE_TRAIN_DATASET_SHA256:
            raise ValueError("C44 consequence checkpoint training identity mismatch")
        model = TemporalFutureH3ConsequenceModel(**payload["model_kwargs"]).to(device)
        model.load_state_dict(payload["models"]["conditioned"], strict=True)
        model.requires_grad_(False).eval()
        projected = []
        with torch.inference_mode():
            for start in range(0, len(states), 32):
                projected.append(
                    model.project_features(current_hidden[start:start + 32].to(device)).cpu()
                )
        this_current = torch.cat(projected)
        if current_projected is None:
            current_projected = this_current
        elif not torch.equal(current_projected, this_current):
            raise ValueError("C44 checkpoint fixed projections differ")
        mean = payload["normalization"]["state_mean"]
        std = payload["normalization"]["state_std"]
        proprio = torch.stack([states[int(group)]["proprio"].float() for group in group_ids])
        normalized = (proprio - mean) / std
        batches = []
        with torch.inference_mode():
            for start in range(0, len(branches), 64):
                stop = min(start + 64, len(branches))
                batches.append(model.forward_projected(
                    normalized[start:stop].to(device),
                    this_current[group_ids[start:stop]].to(device),
                    actions[start:stop].to(device),
                ).cpu())
        predictions.append(torch.cat(batches))
        checkpoint_hashes.append(sha256_file(path))
    if current_projected is None:
        raise RuntimeError("empty consequence ensemble")
    return current_projected, torch.stack(predictions).mean(0), {
        "members": len(payloads),
        "checkpoint_sha256": checkpoint_hashes,
        "training_dataset_sha256": EXPECTED_CONSEQUENCE_TRAIN_DATASET_SHA256,
        "ensemble": "arithmetic mean of four frozen C38 temporal future-H3 predictions",
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("refusing to overwrite C44 output")
    dataset_path = args.dataset.resolve()
    feature_path = args.features.resolve()
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    features = torch.load(feature_path, map_location="cpu", weights_only=False)
    if data.get("format") != "h3wam-c44-powered-consequence-ranking-dataset-v1":
        raise ValueError("C44 dataset format mismatch")
    if features.get("format") != "h3wam-c44-live-h3-consequence-features-v1":
        raise ValueError("C44 feature format mismatch")
    dataset_sha = sha256_file(dataset_path)
    if features.get("dataset_sha256") != dataset_sha:
        raise ValueError("C44 feature/dataset identity mismatch")
    device = torch.device(args.device)
    current, predicted, ensemble_contract = project_consequences(
        data, features, [path.resolve() for path in args.consequence_checkpoints], device
    )
    states, branches = data["states"], data["branches"]
    group_ids = torch.tensor([int(row["group_id"]) for row in branches])
    labels = torch.tensor([float(row["success"]) for row in branches])
    train_groups = mixed_groups(states, "ranker_train")
    final_groups = mixed_groups(states, "reserved_powered_final")
    if len(train_groups) < 46 or len(final_groups) < 24:
        raise ValueError(f"C44 expected at least46/24 mixed groups, got {len(train_groups)}/{len(final_groups)}")
    train_rows = torch.tensor([
        states[int(row["group_id"])]["consequence_split"] == "ranker_train"
        for row in branches
    ])
    actions = torch.stack([row["environment_actions"].float() for row in branches]).flatten(1)
    action_mean, action_std = c26.standardize_fit(actions[train_rows])
    action_projection = c26.fixed_projection(actions.shape[1], PROJECTION_DIM, PROJECTION_SEED)
    action_design = c26.standardize(actions, action_mean, action_std) @ action_projection
    delta = predicted - current[group_ids]
    delta_mean, delta_std = c26.standardize_fit(delta[train_rows])
    consequence_projection = c26.fixed_projection(
        delta.shape[1], PROJECTION_DIM, PROJECTION_SEED + 1
    )
    consequence_design = c26.standardize(delta, delta_mean, delta_std) @ consequence_projection
    designs = {
        "action_only": action_design,
        "consequence_ensemble": torch.cat((action_design, consequence_design), dim=1),
    }
    arms, weights = {}, {}
    for name, design in designs.items():
        selected, cv = select_config(
            design, group_ids, labels, states, train_groups, device
        )
        fitted = c26.fit_pairwise_linear(
            design, group_ids, labels, train_groups,
            steps=int(selected["steps"]), learning_rate=LEARNING_RATE,
            weight_decay=float(selected["weight_decay"]), device=device,
        )
        scores = design @ fitted["weights"]
        train_metric = c26.ranking_metrics(scores, group_ids, labels, train_groups)
        final_metric = c26.ranking_metrics(scores, group_ids, labels, final_groups)
        permutation = c28.permutation_pvalue(
            scores, group_ids, labels, final_groups, final_metric["pairwise_correct"],
            maximum_exact=0, monte_carlo_samples=100000, seed=20260816,
        )
        shuffled_metrics = []
        for seed in SHUFFLE_SEEDS:
            shuffled = c26.shuffled_train_labels(labels, group_ids, train_groups, seed)
            control = c26.fit_pairwise_linear(
                design, group_ids, shuffled, train_groups,
                steps=int(selected["steps"]), learning_rate=LEARNING_RATE,
                weight_decay=float(selected["weight_decay"]), device=device,
            )
            shuffled_metrics.append(c26.ranking_metrics(
                design @ control["weights"], group_ids, labels, final_groups
            ))
        arms[name] = {
            "selected_train_only_config": selected,
            "train_cv": cv,
            "optimization": {key: value for key, value in fitted.items() if key != "weights"},
            "train": train_metric,
            "fresh_final": {**final_metric, "permutation": permutation},
            "shuffle_mean_pairwise_accuracy": sum(
                row["pairwise_accuracy"] for row in shuffled_metrics
            ) / len(shuffled_metrics),
        }
        weights[name] = fitted["weights"]
    baseline = arms["action_only"]["fresh_final"]
    candidate = arms["consequence_ensemble"]
    final = candidate["fresh_final"]
    gate = {
        "pairwise_accuracy_at_least_0_60": final["pairwise_accuracy"] >= 0.60,
        "top1_success_rate_at_least_0_60": final["top1_success_rate"] >= 0.60,
        "beats_action_only_by_at_least_0_05": (
            final["pairwise_accuracy"] >= baseline["pairwise_accuracy"] + 0.05
        ),
        "beats_c40_by_at_least_0_05": final["pairwise_accuracy"] >= C40_FRESH_PAIRWISE + 0.05,
        "permutation_p_at_most_0_05": final["permutation"]["value"] <= 0.05,
        "beats_shuffle_mean_by_at_least_0_05": (
            final["pairwise_accuracy"] >= candidate["shuffle_mean_pairwise_accuracy"] + 0.05
        ),
        "train_pairwise_accuracy_at_least_0_75": (
            candidate["train"]["pairwise_accuracy"] >= 0.75
        ),
        "all_fresh_groups_have_score_variation": all(
            row["score_range"] > 1e-8 for row in final["groups"]
        ),
    }
    gate["passed"] = all(gate.values())
    status = (
        "PASS_C44_POWERED_CONSEQUENCE_VALUE_RANKING"
        if gate["passed"] else "FAIL_C44_POWERED_CONSEQUENCE_VALUE_RANKING"
    )
    report = {
        "format": FORMAT,
        "status": status,
        "permission": "GO_BEST_OF_N_CLOSED_LOOP_CANARY" if gate["passed"] else "NO_GO_BEST_OF_N",
        "claim_boundary": "Untouched-source offline ranking only; closed-loop improvement remains unproven.",
        "data": {
            "train_mixed_groups": len(train_groups),
            "fresh_final_mixed_groups": len(final_groups),
            "train_pairs": candidate["train"]["pairwise_total"],
            "fresh_final_pairs": final["pairwise_total"],
            "fresh_final_sources": len({states[group]["source_episode"] for group in final_groups}),
        },
        "sources": {
            "dataset_sha256": dataset_sha,
            "features_sha256": sha256_file(feature_path),
            "c38_corrected_audit_sha256": "c5709b8d2009deebff42a801592618305d30887ae17ed6fccf2d8a7d862818f5",
            "c40_report_sha256": "9a3148b04d412ee0d3307bf352249c2a762e8def56693926daa23fac48b1235c",
        },
        "ensemble": ensemble_contract,
        "arms": arms,
        "gate": gate,
    }
    checkpoint = {
        "format": FORMAT,
        "status": status,
        "weights": weights,
        "normalization": {
            "action_mean": action_mean,
            "action_std": action_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "action_projection": action_projection,
            "consequence_projection": consequence_projection,
        },
        "ensemble": ensemble_contract,
        "sources": report["sources"],
        "gate": gate,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_name(f".{args.checkpoint.name}.{os.getpid()}.partial")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": status,
        "permission": report["permission"],
        "gate": gate,
        "action_only": {
            "pairwise": baseline["pairwise_accuracy"],
            "top1": baseline["top1_success_rate"],
        },
        "consequence": {
            "pairwise": final["pairwise_accuracy"],
            "top1": final["top1_success_rate"],
            "p": final["permutation"]["value"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
