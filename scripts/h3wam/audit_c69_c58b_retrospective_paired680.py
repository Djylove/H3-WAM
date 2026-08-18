#!/usr/bin/env python3
"""Cross-bind historical C69 and C58b rollouts on the same 680 LIBERO states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = [row for task in payload["tasks"] for row in task["episodes"]]
    if len(episodes) != 1:
        raise ValueError(f"not a one-episode result: {path}")
    return episodes[0]


def one_sided(wins: int, losses: int) -> float:
    count = wins + losses
    if not count:
        return 1.0
    return sum(math.comb(count, k) for k in range(wins, count + 1)) / (2**count)


def summary(rows: list[dict]) -> dict:
    c58 = sum(row["c58_success"] for row in rows)
    c69 = sum(row["c69_success"] for row in rows)
    wins = sum((not row["c58_success"]) and row["c69_success"] for row in rows)
    losses = sum(row["c58_success"] and (not row["c69_success"]) for row in rows)
    return {
        "pairs": len(rows), "c58b_successes": c58, "c69_successes": c69,
        "c58b_success_rate": c58 / len(rows), "c69_success_rate": c69 / len(rows),
        "success_rate_delta_c69_minus_c58b": (c69 - c58) / len(rows),
        "c69_wins": wins, "c58b_wins": losses,
        "one_sided_mcnemar_p_c69_better": one_sided(wins, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c58-evidence", type=Path, required=True)
    parser.add_argument("--c69-evidence", type=Path, required=True)
    parser.add_argument("--c58-trial33-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    c58 = {}
    for line in args.c58_evidence.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["suite"], int(row["task"]), int(row["trial"]))
        path = Path(row["candidate_result"])
        if sha256(path) != row["candidate_result_sha256"] or sha256(Path(row["candidate_trajectory"])) != row["candidate_trajectory_sha256"]:
            raise ValueError(f"C58 evidence byte mismatch: {key}")
        c58[key] = row
    c69 = {}
    for line in args.c69_evidence.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["suite"], int(row["task"]), int(row["trial"]))
        path = Path(row["c69_result"])
        if sha256(path) != row["c69_result_sha256"] or sha256(Path(row["c69_trajectory"])) != row["c69_trajectory_sha256"]:
            raise ValueError(f"C69 evidence byte mismatch: {key}")
        c69[key] = row

    expected_640 = {(suite, task, trial) for suite in SUITES for task in range(10) for trial in range(34, 50)}
    if set(c58) != expected_640 or not expected_640.issubset(c69):
        raise ValueError("historical 640-pair grid mismatch")
    rows = []
    for key in sorted(expected_640):
        if c58[key]["initial_state_sha256"] != c69[key]["initial_state_sha256"]:
            raise ValueError(f"initial trajectory state mismatch: {key}")
        rows.append({
            "key": key, "identity": "exact_initial_trajectory_digest",
            "c58_success": bool(episode(Path(c58[key]["candidate_result"]))["success"]),
            "c69_success": bool(episode(Path(c69[key]["c69_result"]))["success"]),
        })

    # Trial 33 predates C58 trajectory persistence, but its deterministic seed,
    # trial and full initial object-joint mapping are present in both result files.
    for suite in SUITES:
        source = args.c58_trial33_root / "candidate_c58b" / suite / "results.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        by_task = {int(task["task_id"]): task["episodes"][0] for task in payload["tasks"]}
        for task in range(10):
            key = (suite, task, 33)
            a = by_task[task]
            b = episode(Path(c69[key]["c69_result"]))
            if a["trial"] != b["trial"] or a["episode_seed"] != b["episode_seed"] or a["initial_object_joints"] != b["initial_object_joints"]:
                raise ValueError(f"trial33 initial-state contract mismatch: {key}")
            rows.append({"key": key, "identity": "exact_seed_and_initial_object_joints", "c58_success": bool(a["success"]), "c69_success": bool(b["success"])})

    if len(rows) != 680:
        raise AssertionError("retrospective grid is not 680 pairs")
    overall = summary(rows)
    per_suite = {suite: summary([row for row in rows if row["key"][0] == suite]) for suite in SUITES}
    suite_floor = min(row["success_rate_delta_c69_minus_c58b"] for row in per_suite.values())
    output = {
        "format": "h3wam-c69-c58b-retrospective-paired680-audit-v1",
        "status": "PASS_EXACT_HISTORICAL_PAIR_IDENTITY_AND_RESULT_AUDIT",
        "effect_status": "RETROSPECTIVE_DIRECT_PAIRED_EVIDENCE_NOT_PREREGISTERED_NOT_UNSEEN",
        "overall": overall, "per_suite": per_suite,
        "identity": {"exact_initial_trajectory_digest_pairs": 640, "exact_seed_and_initial_object_joints_pairs": 40},
        "directional_gates": {
            "overall_delta_at_least_3pp": overall["success_rate_delta_c69_minus_c58b"] >= 0.03,
            "net_wins_at_least_20": overall["c69_wins"] - overall["c58b_wins"] >= 20,
            "one_sided_p_at_most_0_05": overall["one_sided_mcnemar_p_c69_better"] <= 0.05,
            "suite_floor_at_least_minus_3pp": suite_floor >= -0.03,
        },
        "suite_floor": suite_floor,
        "decision": "DIRECT_CONFIRMATION_RERUN_REQUIRED_BEFORE_PROMOTION",
        "sources": {
            "c58_evidence": str(args.c58_evidence.resolve()), "c58_evidence_sha256": sha256(args.c58_evidence),
            "c69_evidence": str(args.c69_evidence.resolve()), "c69_evidence_sha256": sha256(args.c69_evidence),
            "c58_trial33_root": str(args.c58_trial33_root.resolve()),
        },
        "claim_boundary": "All outcomes existed before this audit. It is strong direct paired evidence, but not preregistered evidence and not evaluation on unseen LIBERO initial states.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
