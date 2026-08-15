#!/usr/bin/env python3
"""Select C22 high-entropy states for a first-action-only intervention canary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c22-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def entropy(successes: int) -> float:
    probability = successes / 4
    if probability in (0.0, 1.0):
        return 0.0
    return -sum(value * math.log2(value) for value in (probability, 1 - probability))


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    c22 = json.loads((args.c22_root / "COMPLETED").read_text())
    if not c22["all_groups_action_diverse"] or c22["mixed_outcome_groups"] < 1:
        raise ValueError("C23 requires action-diverse C22 with at least one mixed group")
    ranked = sorted(
        c22["group_reports"],
        key=lambda row: (
            -entropy(row["successes"]), row["suite"], row["task"],
            row["trial"], row["distance_replans"],
        ),
    )
    selected = []
    for suite in sorted({row["suite"] for row in ranked}):
        selected.append(next(row for row in ranked if row["suite"] == suite))
    selected_ids = {
        (row["suite"], row["task"], row["trial"], row["distance_replans"])
        for row in selected
    }
    for row in ranked:
        identity = (row["suite"], row["task"], row["trial"], row["distance_replans"])
        if identity not in selected_ids:
            selected.append(row)
            selected_ids.add(identity)
        if len(selected) == 8:
            break
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "runs").mkdir(parents=True)
    (output / "logs").mkdir()
    rows = []
    group_records = []
    for rank, group in enumerate(selected):
        continuation_seed = 20_000_000 + rank * 100_000
        group_records.append(
            {
                key: group[key] for key in (
                    "suite", "task", "trial", "distance_replans", "index",
                    "successes", "mixed_outcomes", "state_sha256",
                )
            }
            | {"selection_entropy_bits": entropy(group["successes"]),
               "continuation_policy_noise_seed_base": continuation_seed}
        )
        for reference in sorted(group["branches_detail"], key=lambda row: row["noise_offset"]):
            result_path = Path(reference["result"])
            payload = json.loads(result_path.read_text())
            branch = payload["tasks"][0]["episodes"][0]["branch_start"]
            rows.append(
                {
                    "ordinal": len(rows),
                    "group_rank": rank,
                    "suite": group["suite"],
                    "task": group["task"],
                    "trial": group["trial"],
                    "distance_replans": group["distance_replans"],
                    "index": group["index"],
                    "trajectory": branch["trajectory"],
                    "first_policy_noise_seed": reference["policy_noise_seed"],
                    "noise_offset": reference["noise_offset"],
                    "continuation_policy_noise_seed_base": continuation_seed,
                    "c22_reference_result": str(result_path.resolve()),
                }
            )
    if len(rows) != 32:
        raise AssertionError(f"expected 32 C23 branches, got {len(rows)}")
    (output / "selection.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    preregistration = {
        "format": "h3wam-c23-first-action-causal-canary-v1",
        "falsifiable_hypothesis": (
            "At fixed state and fixed continuation noise schedule, changing only first-replan "
            "noise reproduces its C22 first action, keeps all later exogenous seeds identical, "
            "and yields at least one same-state mixed-outcome group."
        ),
        "parent": "D0-H32-s14000/replan8/no ensemble",
        "selection_rule": (
            "highest C22 Bernoulli outcome entropy: one group per suite, then global rank, "
            "with deterministic lexical tie-breaks"
        ),
        "within_group_only_variable": "first-replan policy diffusion noise seed",
        "fixed_continuation_contract": (
            "replan i>=1 uses continuation_policy_noise_seed_base + i - 1"
        ),
        "groups": 8,
        "branches": 32,
        "max_environment_steps": 12_800,
        "selected_groups": group_records,
        "promotion": (
            "all first chunks bit-exact to C22 references, all continuation schedules valid, "
            "all groups action-diverse, and >=1 mixed-outcome group"
        ),
        "pass_permission": "GO_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY",
        "fail_permission": "NO_GO_FIRST_ACTION_CRITIC_DATA",
        "effect_conclusion": "NOT_EVIDENCE_READY",
    }
    atomic_json(output / "preregistration.json", preregistration)
    print(json.dumps({"output_root": str(output), "groups": 8, "branches": 32}, indent=2))


if __name__ == "__main__":
    main()
