#!/usr/bin/env python3
"""Finalize the preregistered C55 fresh tri-arm LIBERO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ARMS = {"d0_parent", "action_only", "joint_aux"}
SUITES = {"libero_goal", "libero_object", "libero_spatial", "libero_10"}
INITIAL_STATE_KEYS = (
    "step",
    "agentview_image",
    "wristview_image",
    "eef_pos",
    "eef_quat",
    "gripper_qpos",
    "previous_action",
    "sim_state",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def one_sided_mcnemar(wins: int, losses: int) -> float:
    """Exact P[X >= wins], X~Binomial(wins+losses, 0.5)."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (
        2**discordant
    )


def paired_effect(groups: list[dict], candidate: str, reference: str) -> dict:
    candidate_successes = sum(row[candidate] for row in groups)
    reference_successes = sum(row[reference] for row in groups)
    wins = sum(row[candidate] and not row[reference] for row in groups)
    losses = sum(row[reference] and not row[candidate] for row in groups)
    count = len(groups)
    by_suite = {}
    for suite in sorted(SUITES):
        rows = [row for row in groups if row["suite"] == suite]
        c_successes = sum(row[candidate] for row in rows)
        r_successes = sum(row[reference] for row in rows)
        by_suite[suite] = {
            "pairs": len(rows),
            "candidate_successes": c_successes,
            "reference_successes": r_successes,
            "candidate_rate": c_successes / len(rows),
            "reference_rate": r_successes / len(rows),
            "absolute_gain": (c_successes - r_successes) / len(rows),
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "pairs": count,
        "candidate_successes": candidate_successes,
        "reference_successes": reference_successes,
        "candidate_rate": candidate_successes / count,
        "reference_rate": reference_successes / count,
        "absolute_gain": (candidate_successes - reference_successes) / count,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "ties": count - wins - losses,
        "one_sided_exact_mcnemar_p": one_sided_mcnemar(wins, losses),
        "per_suite": by_suite,
    }


def load_stage(root: Path, expected_stage: str, expected_trials: set[int]) -> tuple[list[dict], dict]:
    prepared_path = root / "PREPARED.json"
    manifest_path = root / "jobs.jsonl"
    prepared = json.loads(prepared_path.read_text())
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if (
        prepared["stage"] != expected_stage
        or set(prepared["trials"]) != expected_trials
        or set(prepared["suites"]) != SUITES
        or set(prepared["arms"]) != ARMS
        or prepared["manifest_sha256"] != sha256_file(manifest_path)
        or len(rows) != len(expected_trials) * len(SUITES) * 10 * len(ARMS)
    ):
        raise ValueError(f"invalid frozen C55 stage contract: {root}")
    if not all((root / "workers" / f"worker{worker:02d}.COMPLETED").is_file() for worker in range(32)):
        raise RuntimeError(f"C55 stage workers are incomplete: {root}")
    return rows, prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--mechanical-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    mechanical_audit = json.loads(args.mechanical_audit.resolve().read_text())
    if (
        mechanical_audit.get("status") != "PASS_C55_MECHANICAL_CANARY"
        or mechanical_audit.get("permission") != "GO_C55_REMAINING_FRESH_TRIALS"
    ):
        raise ValueError("C55 finalization requires a passing mechanical-only audit")
    stage_specs = (
        (args.mechanical_root.resolve(), "mechanical_canary", set(range(33, 37))),
        (args.fresh_root.resolve(), "fresh_final", set(range(37, 50))),
    )
    jobs = []
    prepared_reports = []
    for root, stage, trials in stage_specs:
        rows, prepared = load_stage(root, stage, trials)
        jobs.extend(rows)
        prepared_reports.append(prepared)
    if len(jobs) != 2040:
        raise ValueError("C55 final effect requires exactly 2040 tri-arm episodes")

    grouped: dict[tuple[int, str, int], dict] = defaultdict(dict)
    result_hashes = {}
    mechanical_exact = True
    for row in jobs:
        result_path = Path(row["output"]) / "results.json"
        payload = json.loads(result_path.read_text())
        episode = payload["tasks"][0]["episodes"][0]
        trajectory = Path(episode["trajectory"])
        if (
            payload["replan_steps"] != 8
            or payload["action_horizon"] != 32
            or payload["normalized_action_pre_clamp"] is not True
            or payload["trial_indices"] != [row["trial"]]
            or payload["task_ids"] != [row["task"]]
            or payload["suite"] != row["suite"]
            or Path(payload["checkpoint"]).resolve() != Path(row["checkpoint"]).resolve()
        ):
            raise ValueError(f"C55 rollout contract mismatch: {result_path}")
        key = (row["trial"], row["suite"], row["task"])
        grouped[key][row["arm"]] = {
            "success": bool(episode["success"]),
            "episode_seed": episode["episode_seed"],
            "replan_noise_seeds": episode["replan_noise_seeds"],
            "trajectory": trajectory,
        }
        result_hashes[str(result_path)] = sha256_file(result_path)

    pairs = []
    for (trial, suite, task), arms in sorted(grouped.items()):
        if set(arms) != ARMS:
            raise ValueError(f"C55 tri-arm group incomplete: {(trial, suite, task)}")
        reference = arms["d0_parent"]
        reference_archive = np.load(reference["trajectory"])
        for arm in ("action_only", "joint_aux"):
            candidate = arms[arm]
            candidate_archive = np.load(candidate["trajectory"])
            shared = min(
                len(candidate["replan_noise_seeds"]),
                len(reference["replan_noise_seeds"]),
            )
            mechanical_exact &= (
                candidate["episode_seed"] == reference["episode_seed"]
                and shared > 0
                and candidate["replan_noise_seeds"][:shared]
                == reference["replan_noise_seeds"][:shared]
                and all(
                    np.array_equal(candidate_archive[name][0], reference_archive[name][0])
                    for name in INITIAL_STATE_KEYS
                )
            )
        pairs.append(
            {
                "trial": trial,
                "suite": suite,
                "task": task,
                **{arm: arms[arm]["success"] for arm in sorted(ARMS)},
            }
        )
    if len(pairs) != 680:
        raise ValueError("C55 final effect requires exactly 680 paired task/trial groups")

    primary = paired_effect(pairs, "joint_aux", "action_only")
    incumbent = paired_effect(pairs, "joint_aux", "d0_parent")
    primary_gate = {
        "mechanical_contract_exact": mechanical_exact,
        "absolute_gain_at_least_0_03": primary["absolute_gain"] >= 0.03,
        "net_wins_at_least_20": primary["net_wins"] >= 20,
        "one_sided_exact_mcnemar_p_at_most_0_05": primary[
            "one_sided_exact_mcnemar_p"
        ]
        <= 0.05,
        "no_suite_regresses_by_more_than_0_03": all(
            row["absolute_gain"] >= -0.03 for row in primary["per_suite"].values()
        ),
    }
    primary_gate["passed"] = all(primary_gate.values())
    promotion_gate = {
        "joint_not_below_d0_overall": incumbent["absolute_gain"] >= 0.0,
        "no_suite_regresses_vs_d0_by_more_than_0_03": all(
            row["absolute_gain"] >= -0.03 for row in incumbent["per_suite"].values()
        ),
    }
    promotion_gate["passed"] = primary_gate["passed"] and all(promotion_gate.values())
    if promotion_gate["passed"]:
        status = "PASS_C55_CLOSED_LOOP_PROMOTION"
        permission = "GO_PROMOTE_C55_JOINT_ACTION_EXPERT"
    elif primary_gate["passed"]:
        status = "PASS_C55_ACTION_EXPERT_EFFECT_ONLY"
        permission = "KEEP_D0_INCUMBENT_CONTINUE_C55_RESEARCH"
    else:
        status = "FAIL_C55_CLOSED_LOOP_EFFECT"
        permission = "NO_GO_C55_PROMOTION"
    report = {
        "format": "h3wam-c55-fresh-triarm-final-v1",
        "status": status,
        "permission": permission,
        "claim_boundary": (
            "Fresh LIBERO trials 33-49, all 40 benchmark tasks, paired tri-arm seeds, "
            "H32 action horizon with replan=8. No claim beyond this simulator contract."
        ),
        "counts": {"episodes": len(jobs), "paired_groups": len(pairs)},
        "primary_joint_vs_action_only": primary,
        "incumbent_joint_vs_d0": incumbent,
        "primary_gate": primary_gate,
        "promotion_gate": promotion_gate,
        "mechanical_audit_sha256": sha256_file(args.mechanical_audit.resolve()),
        "stage_manifest_sha256": [row["manifest_sha256"] for row in prepared_reports],
        "result_sha256": result_hashes,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": status,
                "permission": permission,
                "primary": primary,
                "incumbent": incumbent,
                "primary_gate": primary_gate,
                "promotion_gate": promotion_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
