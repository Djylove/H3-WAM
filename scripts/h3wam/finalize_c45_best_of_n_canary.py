#!/usr/bin/env python3
"""Apply the preregistered C45 paired closed-loop canary gates exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


FORMAT = "h3wam-c45-best-of-n-closed-loop-canary-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite C45 final report")
    for shard in range(4):
        if not (args.root / f"shard{shard}.COMPLETED").is_file():
            raise FileNotFoundError(f"C45 shard {shard} is incomplete")

    rows = {}
    expected = []
    ordinal = 0
    for suite in ("libero_goal", "libero_object", "libero_spatial", "libero_10"):
        for task in range(5):
            for arm in ("parent", "bestof4"):
                expected.append((ordinal, suite, task, arm))
                ordinal += 1
    for ordinal, suite, task, arm in expected:
        path = args.root / "runs" / f"{ordinal}_{arm}_{suite[7:]}_task{task}_trial22" / "results.json"
        payload = json.loads(path.read_text())
        episode = payload["tasks"][0]["episodes"][0]
        rows[(suite, task, arm)] = {"path": path, "episode": episode}

    parent_success = candidate_success = wins = losses = 0
    per_suite = defaultdict(lambda: {"parent": 0, "bestof4": 0})
    candidate0_exact = True
    selected_indices = []
    score_ranges = []
    seed_schedules_valid = True
    pairs = []
    for suite in ("libero_goal", "libero_object", "libero_spatial", "libero_10"):
        for task in range(5):
            parent = rows[(suite, task, "parent")]["episode"]
            candidate = rows[(suite, task, "bestof4")]["episode"]
            p = bool(parent["success"])
            c = bool(candidate["success"])
            parent_success += int(p)
            candidate_success += int(c)
            wins += int(c and not p)
            losses += int(p and not c)
            per_suite[suite]["parent"] += int(p)
            per_suite[suite]["bestof4"] += int(c)
            candidate0_exact &= (
                parent["first_environment_action_chunk"]
                == candidate["first_consequence_candidate0_chunk"]
            )
            indices = [int(value) for value in candidate["consequence_selected_indices"]]
            ranges = [float(value) for value in candidate["consequence_score_ranges"]]
            seeds = candidate["consequence_candidate_seeds"]
            bases = candidate["replan_noise_seeds"]
            seed_schedules_valid &= len(indices) == len(ranges) == len(seeds) == len(bases)
            seed_schedules_valid &= all(
                list(map(int, proposed)) == list(range(int(base), int(base) + 4))
                for proposed, base in zip(seeds, bases, strict=True)
            )
            selected_indices.extend(indices)
            score_ranges.extend(ranges)
            pairs.append({
                "suite": suite, "task": task, "trial": 22,
                "parent_success": p, "bestof4_success": c,
                "parent_steps": int(parent["steps"]),
                "bestof4_steps": int(candidate["steps"]),
                "bestof4_replans": int(candidate["replans"]),
            })
    nonzero = sum(index != 0 for index in selected_indices)
    suite_floor = all(
        values["bestof4"] >= values["parent"] - 1 for values in per_suite.values()
    )
    gate = {
        "all_20_candidate0_first_chunks_exact_parent": candidate0_exact,
        "all_candidate_seed_schedules_are_base_through_base_plus_3": seed_schedules_valid,
        "all_replans_have_nonzero_score_range": bool(score_ranges) and min(score_ranges) > 1e-8,
        "nonzero_candidate_selected_at_least_20_percent": (
            bool(selected_indices) and nonzero / len(selected_indices) >= 0.20
        ),
        "candidate_success_at_least_parent_plus_2": candidate_success >= parent_success + 2,
        "paired_wins_at_least_3": wins >= 3,
        "paired_net_wins_at_least_2": wins - losses >= 2,
        "no_suite_regresses_by_more_than_1_of_5": suite_floor,
    }
    gate["passed"] = all(gate.values())
    status = "PASS_C45_BEST_OF_N_CANARY" if gate["passed"] else "FAIL_C45_BEST_OF_N_CANARY"
    report = {
        "format": FORMAT,
        "status": status,
        "permission": "GO_C46_POWERED_BEST_OF_N" if gate["passed"] else "NO_GO_ONLINE_RANKER",
        "claim_boundary": "Twenty paired episodes are a mechanism canary, not a full LIBERO benchmark claim.",
        "contract": {
            "suites": ["libero_goal", "libero_object", "libero_spatial", "libero_10"],
            "tasks_per_suite": [0, 1, 2, 3, 4], "trial": 22,
            "parent": "D0-H32-s14000/replan8/sample1",
            "candidate": "same parent plus frozen C44 consequence best-of-4",
            "candidate_seed_schedule": "base replan noise seed + candidate index",
        },
        "results": {
            "parent_successes": parent_success, "candidate_successes": candidate_success,
            "paired_wins": wins, "paired_losses": losses,
            "paired_ties": 20 - wins - losses,
            "per_suite": dict(per_suite),
            "selection_decisions": len(selected_indices),
            "nonzero_selections": nonzero,
            "nonzero_selection_fraction": nonzero / max(len(selected_indices), 1),
            "minimum_score_range": min(score_ranges, default=0.0),
            "pairs": pairs,
        },
        "sources": {
            "ranker_sha256": sha256_file(Path("/mnt/h3-wam/outputs/c44-powered-consequence-ranking-v1/ranker.pt")),
            "c44_report_sha256": sha256_file(Path("/mnt/h3-wam/outputs/c44-powered-consequence-ranking-v1/report.json")),
        },
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
