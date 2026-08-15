#!/usr/bin/env python3
"""Freeze C47 final successes for a no-tuning counterfactual value test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


FORMAT = "h3wam-c52-fresh-counterfactual-value-ranking-v1"
DISTANCES = (3, 5)
OFFSETS = (0, 1_000_000, 2_000_000, 3_000_000)
FINAL_TRIALS = (26, 27)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c47-completed", type=Path, required=True)
    parser.add_argument("--c51-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    c47 = json.loads(args.c47_completed.read_text())
    c51 = json.loads(args.c51_report.read_text())
    if c47.get("status") != "PASS_C47_DENSE_VALUE_SOURCES":
        raise ValueError("C47 source gate did not pass")
    if c51.get("permission") != "GO_FRESH_COUNTERFACTUAL_VALUE_RANKING":
        raise ValueError("C51 did not permit fresh counterfactual ranking")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()

    records: list[dict] = []
    for row in c47["rows"]:
        if int(row["trial"]) not in FINAL_TRIALS or not bool(row["success"]):
            continue
        trajectory = Path(row["trajectory"]).resolve()
        with np.load(trajectory, allow_pickle=False) as archive:
            state_count = int(archive["step"].shape[0])
        available = [distance for distance in DISTANCES if state_count > distance]
        if len(available) != len(DISTANCES):
            continue
        records.append({
            "source_episode": f"{row['suite']}:task{row['task']}:trial{row['trial']}",
            "suite": row["suite"], "task": int(row["task"]),
            "trial": int(row["trial"]), "trajectory": str(trajectory),
            "state_count": state_count, "available_distances": available,
            "split": "fresh_counterfactual_final",
        })
    if len(records) != 30:
        raise ValueError(f"expected 30 eligible frozen sources, found {len(records)}")

    groups: list[dict] = []
    selections: list[dict] = []
    for record in sorted(records, key=lambda item: item["source_episode"]):
        for distance in record["available_distances"]:
            group_id = len(groups)
            index = int(record["state_count"]) - int(distance)
            continuation = 252_000_000 + group_id * 100_000
            group = {
                **record, "group_id": group_id, "distance_replans": distance,
                "index": index,
                "continuation_policy_noise_seed_base": continuation,
            }
            groups.append(group)
            base_seed = 42 + record["task"] * 100_000 + record["trial"] * 1_000 + index
            for offset in OFFSETS:
                selections.append({
                    "ordinal": len(selections), "group_id": group_id,
                    "source_episode": record["source_episode"],
                    "suite": record["suite"], "task": record["task"],
                    "trial": record["trial"], "split": record["split"],
                    "distance_replans": distance, "index": index,
                    "trajectory": record["trajectory"],
                    "first_policy_noise_seed": base_seed + offset,
                    "noise_offset": offset,
                    "continuation_policy_noise_seed_base": continuation,
                })
    if len(groups) != 60 or len(selections) != 240:
        raise RuntimeError("C52 frozen inventory mismatch")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selections)
    )
    source_suites = Counter(record["suite"] for record in records)
    preregistration = {
        "format": FORMAT,
        "hypothesis": (
            "The C51-frozen dense value expert ranks newly executed action alternatives "
            "above chance on untouched counterfactual outcomes."
        ),
        "model_frozen_before_outcomes": True,
        "selected_checkpoint": c51["selected"]["checkpoint"],
        "selected_checkpoint_sha256": c51["sources"]["checkpoint_sha256"],
        "source_contract": (
            "C47 final successful parent trials26/27; source visual trajectories were part "
            "of C51 final, but none of these four alternative branch outcomes existed then"
        ),
        "only_variable_within_group": "first-replan diffusion noise offset",
        "execution_contract": "first chunk32; continuation replan8 with identical seed schedule",
        "candidate_offsets": list(OFFSETS),
        "sources": len(records), "source_suite_counts": dict(sorted(source_suites.items())),
        "groups": len(groups), "branches": len(selections), "groups_detail": groups,
        "outcome_gate": (
            "all actions/seeds/consequences exact; >=10 mixed groups across >=3 suites"
        ),
        "ranking_gate_frozen_before_outcomes": {
            "pairwise_success_failure_accuracy_at_least": 0.60,
            "mixed_group_top1_success_at_least": 0.60,
            "one_sided_within_group_permutation_p_at_most": 0.05,
            "all_group_score_ranges_greater_than": 1e-6,
        },
        "pass_permission": "GO_C53_DENSE_VALUE_CLOSED_LOOP_CANARY",
        "fail_permission": "NO_GO_DENSE_VALUE_CLOSED_LOOP_CANARY",
        "claim_boundary": (
            "A pass proves fresh counterfactual action ranking only, not online rollout gain"
        ),
        "sources_sha256": {
            "c47": sha256_file(args.c47_completed),
            "c51": sha256_file(args.c51_report),
        },
        "max_environment_steps": len(selections) * 400,
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({
        "output_root": str(output), "sources": len(records),
        "source_suite_counts": dict(sorted(source_suites.items())),
        "groups": len(groups), "branches": len(selections),
    }, indent=2))


if __name__ == "__main__":
    main()
