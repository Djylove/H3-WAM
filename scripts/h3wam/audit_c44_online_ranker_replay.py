#!/usr/bin/env python3
"""Prove that the online C44 scorer reproduces the frozen offline report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam import FrozenConsequenceActionRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--consequence-checkpoints", type=Path, nargs=4, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    report = json.loads(args.report.read_text())
    expected_rows = {
        int(row["group_id"]): row
        for row in report["arms"]["consequence_ensemble"]["fresh_final"]["groups"]
    }
    ranker = FrozenConsequenceActionRanker(
        args.ranker, args.consequence_checkpoints, device=torch.device(args.device)
    )
    branches_by_group: dict[int, list[dict]] = {}
    for branch in data["branches"]:
        branches_by_group.setdefault(int(branch["group_id"]), []).append(branch)
    correct = total = top1 = 0
    maximum_score_range_error = 0.0
    for group, expected in expected_rows.items():
        rows = branches_by_group[group]
        actions = torch.stack([row["environment_actions"].float() for row in rows]).to(args.device)
        scores = ranker.score(
            data["states"][group]["proprio"].float().to(args.device),
            features["fact_layer49_hidden"][group].float().to(args.device),
            actions,
        ).cpu()
        labels = torch.tensor([bool(row["success"]) for row in rows])
        success_scores = scores[labels]
        failure_scores = scores[~labels]
        this_correct = int((success_scores[:, None] > failure_scores[None]).sum())
        this_total = success_scores.numel() * failure_scores.numel()
        this_top1 = bool(labels[int(scores.argmax())])
        score_range = float(scores.max() - scores.min())
        maximum_score_range_error = max(
            maximum_score_range_error, abs(score_range - float(expected["score_range"]))
        )
        if this_correct != int(expected["pairwise_correct"]):
            raise RuntimeError(f"group {group} pairwise replay mismatch")
        if this_total != int(expected["pairwise_total"]):
            raise RuntimeError(f"group {group} pair count mismatch")
        if this_top1 is not bool(expected["top1_success"]):
            raise RuntimeError(f"group {group} top1 replay mismatch")
        correct += this_correct
        total += this_total
        top1 += int(this_top1)
    frozen = report["arms"]["consequence_ensemble"]["fresh_final"]
    if (correct, total, top1, len(expected_rows)) != (
        int(frozen["pairwise_correct"]), int(frozen["pairwise_total"]),
        int(frozen["top1_successes"]), int(frozen["top1_total"]),
    ):
        raise RuntimeError("aggregate C44 replay mismatch")
    if maximum_score_range_error > 1e-4:
        raise RuntimeError("C44 replay score ranges differ beyond tolerance")
    print(json.dumps({
        "status": "PASS_C44_ONLINE_SCORER_REPLAY",
        "pairwise_correct": correct,
        "pairwise_total": total,
        "top1_successes": top1,
        "top1_total": len(expected_rows),
        "maximum_score_range_error": maximum_score_range_error,
        "ranker_sha256": ranker.ranker_checkpoint_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
