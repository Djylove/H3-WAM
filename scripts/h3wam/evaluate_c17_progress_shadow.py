#!/usr/bin/env python3
"""Gate a frozen H3 progress probe on paired, action-identical rollouts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--incumbent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="C17")
    return parser.parse_args()


def episode(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload["tasks"][0]["episodes"][0]


def pairwise_auc(labels: list[bool], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores, strict=True) if label]
    negative = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        raise ValueError("AUROC requires both successful and failed shadow episodes")
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positive
        for neg in negative
    )
    return wins / (len(positive) * len(negative))


def main() -> None:
    args = parse_args()
    selected = []
    for line in (args.shadow_root / "selection.txt").read_text().splitlines():
        suite, task, trial, expected = line.split()
        selected.append((suite, int(task), int(trial), expected))
    if len(selected) != 16 or len(set(selected)) != 16:
        raise ValueError("C17 shadow selection must contain 16 unique episodes")

    rows = []
    noninterference = True
    for suite, task, trial, expected in selected:
        slug = suite.removeprefix("libero_")
        shadow_path = (
            args.shadow_root / "runs" / f"{slug}_task{task}_trial{trial}" / "results.json"
        )
        parent_path = (
            args.incumbent_root
            / f"d0_h32_s14000_{slug}_task{task}_trial{trial}_replan8"
            / "results.json"
        )
        shadow_payload, shadow = episode(shadow_path)
        parent_payload, parent = episode(parent_path)
        action_equal = (
            shadow["first_environment_action_chunk"]
            == parent["first_environment_action_chunk"]
        )
        contract_equal = all(
            shadow_payload[key] == parent_payload[key]
            for key in (
                "checkpoint", "suite", "max_steps", "replan_steps",
                "action_horizon", "wait_steps", "normalized_action_pre_clamp",
                "model_evaluations",
            )
        )
        noninterference &= action_equal and contract_equal
        values = [float(value) for value in shadow["progress_values"]]
        if not values or len(values) != int(shadow["replans"]):
            raise ValueError(f"missing C17 progress trace: {shadow_path}")
        rows.append(
            {
                "suite": suite,
                "task": task,
                "trial": trial,
                "selection_label": expected,
                "parent_success": bool(parent["success"]),
                "shadow_success": bool(shadow["success"]),
                "first_action_chunk_exact": action_equal,
                "rollout_contract_exact": contract_equal,
                "replans": len(values),
                "first": values[0],
                "last": values[-1],
                "delta": values[-1] - values[0],
                "nonincreasing_fraction": sum(
                    right <= left for left, right in zip(values, values[1:])
                ) / max(len(values) - 1, 1),
            }
        )

    success = [row for row in rows if row["shadow_success"]]
    failure = [row for row in rows if not row["shadow_success"]]
    labels = [row["shadow_success"] for row in rows]
    # Lower predicted remaining progress is treated as the success score.
    auc = pairwise_auc(labels, [-row["last"] for row in rows])
    success_median_delta = statistics.median(row["delta"] for row in success)
    success_median_last = statistics.median(row["last"] for row in success)
    failure_median_last = statistics.median(row["last"] for row in failure)
    outcome_reproduced = all(
        row["parent_success"] == row["shadow_success"] for row in rows
    )
    pass_gate = (
        noninterference
        and outcome_reproduced
        and auc >= 0.65
        and success_median_delta < 0.0
        and success_median_last < failure_median_last
    )
    report = {
        "format": f"h3wam-{args.experiment_id.lower()}-progress-shadow-gate-v1",
        "experiment_class": "controlled_ablation",
        "falsifiable_hypothesis": (
            "Without changing actions, successful incumbent rollouts have lower final "
            "predicted remaining progress than failures and decrease over the episode."
        ),
        "parent": "D0-H32-s14000/replan8/no ensemble",
        "only_variable": (
            f"validated {args.experiment_id} progress ridge emits shadow metadata"
        ),
        "episodes": len(rows),
        "successful_episodes": len(success),
        "failed_episodes": len(failure),
        "first_action_chunks_exact": noninterference,
        "outcomes_exact": outcome_reproduced,
        "final_remaining_progress_auroc": auc,
        "success_median_delta": success_median_delta,
        "success_median_last": success_median_last,
        "failure_median_last": failure_median_last,
        "promotion": (
            "all first chunks/contracts/outcomes exact; AUROC>=0.65; success median "
            "delta<0; success median final<failure median final"
        ),
        "status": "PASS_PROGRESS_SHADOW_GATE" if pass_gate else "FAIL_PROGRESS_SHADOW_GATE",
        "training_permission": "GO_CANARY",
        "effect_conclusion": "NOT_EVIDENCE_READY",
        "boundary": (
            "Diagnostic only. Passing does not authorize action reranking, best-of-N, "
            "or claim improved LIBERO success."
        ),
        "rows": rows,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
