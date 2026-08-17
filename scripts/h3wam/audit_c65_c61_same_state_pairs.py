#!/usr/bin/env python3
"""Read-only source/data audit for C65; this script never evaluates a model.

C65 asks whether the four C61 action resamples at an exact restored state can
provide source-independent success/failure pairs for a balanced four-suite
FACT Stage-2 ranking test.  Outcomes are read only to establish eligibility;
no value score, checkpoint forward, training, or best-of-N selection occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "h3wam-c65-c61-same-state-source-data-audit-v1"
C61_FORMAT = "h3wam-c61-failure-rollout-expansion-v1"
C63_FORMAT = "h3wam-c63-fact-stage2-within-state-pairs-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZON = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def action_identity(actions: np.ndarray, is_pad: np.ndarray | None = None) -> str:
    actions = np.asarray(actions, dtype="<f4")
    if actions.shape != (HORIZON, 7):
        raise ValueError(f"action shape is not {(HORIZON, 7)}: {actions.shape}")
    if is_pad is None:
        is_pad = np.zeros(HORIZON, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(actions.tobytes(order="C"))
    digest.update(np.asarray(is_pad, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_result(root: Path, job: dict[str, Any]) -> Path:
    prefix = f"{int(job['ordinal'])}_g{int(job['group_id'])}_c{int(job['candidate'])}_"
    matches = sorted((root / "runs").glob(f"{prefix}*/results.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one result for {prefix}, got {len(matches)}")
    return matches[0].resolve()


def _suite_counter() -> dict[str, Counter[str]]:
    return {suite: Counter() for suite in SUITES}


def audit(
    c61_root: Path,
    c63_pairs_path: Path,
    *,
    expected_frozen_sha256: str,
    expected_jobs_sha256: str,
    minimum_sources_per_suite: int,
) -> dict[str, Any]:
    frozen_path = c61_root / "FROZEN.json"
    jobs_path = c61_root / "jobs.jsonl"
    frozen_sha = sha256_file(frozen_path)
    jobs_sha = sha256_file(jobs_path)
    if frozen_sha != expected_frozen_sha256 or jobs_sha != expected_jobs_sha256:
        raise ValueError("C61 frozen inventory SHA256 mismatch")
    frozen = json.loads(frozen_path.read_text())
    jobs = load_jsonl(jobs_path)
    if (
        frozen.get("format") != C61_FORMAT
        or frozen.get("jobs_sha256") != jobs_sha
        or len(jobs) != int(frozen["jobs"])
        or [int(row["ordinal"]) for row in jobs] != list(range(len(jobs)))
    ):
        raise ValueError("C61 frozen inventory contract failed")

    c63 = json.loads(c63_pairs_path.read_text())
    if c63.get("format") != C63_FORMAT:
        raise ValueError("C63 pair manifest format mismatch")
    c63_parent_hashes = {row["parent_trajectory_sha256"] for row in c63["pairs"]}
    c63_branch_hashes = {row["branch_trajectory_sha256"] for row in c63["pairs"]}
    c63_action_hashes = {
        value
        for row in c63["pairs"]
        for value in (row["success_action_sha256"], row["failed_action_sha256"])
    }
    c63_coordinates = {
        (row["suite"], int(row["task"]), int(row["trial"]), int(row["onset_step"]))
        for row in c63["pairs"]
    }
    c63_state_hashes: set[str] = set()
    for row in c63["pairs"]:
        with np.load(row["parent_trajectory"], allow_pickle=False) as archive:
            c63_state_hashes.add(sha256_array(archive["sim_state"][int(row["parent_index"])]))

    records_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    policy_checkpoints: set[str] = set()
    parent_sha_cache: dict[str, str] = {}
    branch_hashes: set[str] = set()
    all_state_hashes: set[str] = set()
    all_action_hashes: set[str] = set()
    parent_proposed_action_hashes: set[str] = set()
    exact = Counter()
    for job in jobs:
        result_path = find_result(c61_root, job)
        result = json.loads(result_path.read_text())
        episodes = result.get("tasks", [{}])[0].get("episodes", [])
        if len(episodes) != 1:
            raise ValueError("C61 result does not contain one branch episode")
        episode = episodes[0]
        required = {
            "suite": str(job["suite"]),
            "task_ids": [int(job["task"])],
            "trial_indices": [int(job["trial"])],
            "start_trajectory": str(Path(job["trajectory"]).resolve()),
            "start_index": int(job["index"]),
            "first_policy_noise_seed": int(job["first_policy_noise_seed"]),
            "continuation_policy_noise_seed_base": int(
                job["continuation_policy_noise_seed_base"]
            ),
            "first_replan_steps": 32,
            "replan_steps": 8,
            "save_trajectories": True,
        }
        if any(result.get(key) != value for key, value in required.items()):
            raise ValueError(f"C61 result identity drift: {result_path}")
        checkpoint = str(result.get("checkpoint", ""))
        if not checkpoint:
            raise ValueError("C61 result lacks policy checkpoint")
        policy_checkpoints.add(checkpoint)
        parent_path = Path(job["trajectory"]).resolve()
        branch_path = Path(episode["trajectory"]).resolve()
        parent_sha = parent_sha_cache.setdefault(str(parent_path), sha256_file(parent_path))
        if parent_sha != str(job["trajectory_sha256"]):
            raise ValueError("C61 parent trajectory changed")
        branch_sha = sha256_file(branch_path)
        branch_hashes.add(branch_sha)
        with np.load(parent_path, allow_pickle=False) as parent, np.load(
            branch_path, allow_pickle=False
        ) as branch:
            index = int(job["index"])
            if int(parent["step"][index]) != int(job["start_step"]):
                raise ValueError("C61 parent step drift")
            if int(branch["step"][0]) != int(job["start_step"]):
                raise ValueError("C61 branch step drift")
            state_equal = np.array_equal(parent["sim_state"][index], branch["sim_state"][0])
            previous_equal = np.array_equal(
                parent["previous_action"][index], branch["previous_action"][0]
            )
            if not state_equal or not previous_equal:
                raise ValueError("C61 restored state/previous_action is not byte exact")
            exact["parent_branch_sim_state"] += int(state_equal)
            exact["parent_branch_previous_action"] += int(previous_equal)
            state_sha = sha256_array(branch["sim_state"][0])
            previous_sha = sha256_array(branch["previous_action"][0])
            action = np.asarray(branch["policy_actions"][0], dtype=np.float32)
            json_action = np.asarray(episode["first_environment_action_chunk"], dtype=np.float32)
            if not np.array_equal(action, json_action):
                raise ValueError("C61 JSON/trajectory first action differs")
            action_sha = action_identity(action)
            parent_action_sha = action_identity(parent["policy_actions"][index])
        all_state_hashes.add(state_sha)
        all_action_hashes.add(action_sha)
        parent_proposed_action_hashes.add(parent_action_sha)
        records_by_group[int(job["group_id"])].append(
            {
                "source_id": int(job["source_id"]),
                "group_id": int(job["group_id"]),
                "candidate": int(job["candidate"]),
                "suite": str(job["suite"]),
                "task": int(job["task"]),
                "trial": int(job["trial"]),
                "start_step": int(job["start_step"]),
                "distance_replans": int(job["distance_replans"]),
                "success": bool(episode["success"]),
                "state_sha256": state_sha,
                "previous_action_sha256": previous_sha,
                "action_sha256": action_sha,
                "parent_trajectory_sha256": parent_sha,
                "branch_trajectory_sha256": branch_sha,
            }
        )

    if len(policy_checkpoints) != 1:
        raise ValueError("C61 branches were generated by multiple checkpoints")
    policy_checkpoint = Path(next(iter(policy_checkpoints)))
    suite_counts = _suite_counter()
    mixed_groups: list[dict[str, Any]] = []
    failure_sources: dict[str, set[int]] = {suite: set() for suite in SUITES}
    mixed_sources: dict[str, set[int]] = {suite: set() for suite in SUITES}
    for group_id, rows in sorted(records_by_group.items()):
        if sorted(row["candidate"] for row in rows) != list(range(4)):
            raise ValueError(f"C61 group {group_id} lacks candidates 0..3")
        if len({row["state_sha256"] for row in rows}) != 1:
            raise ValueError(f"C61 group {group_id} state mismatch")
        if len({row["previous_action_sha256"] for row in rows}) != 1:
            raise ValueError(f"C61 group {group_id} previous_action mismatch")
        if len({row["action_sha256"] for row in rows}) != 4:
            raise ValueError(f"C61 group {group_id} action candidates are not unique")
        suite, source_id = rows[0]["suite"], rows[0]["source_id"]
        successes = [row for row in rows if row["success"]]
        failures = [row for row in rows if not row["success"]]
        count = suite_counts[suite]
        count["groups"] += 1
        count["successful_jobs"] += len(successes)
        count["failed_jobs"] += len(failures)
        if failures:
            failure_sources[suite].add(source_id)
        if successes and failures:
            count["mixed_groups"] += 1
            count["mixed_success_jobs"] += len(successes)
            count["mixed_failure_jobs"] += len(failures)
            count["combinatorial_pairs"] += len(successes) * len(failures)
            mixed_sources[suite].add(source_id)
            mixed_groups.append(
                {
                    "group_id": group_id,
                    "source_id": source_id,
                    "suite": suite,
                    "task": rows[0]["task"],
                    "trial": rows[0]["trial"],
                    "start_step": rows[0]["start_step"],
                    "distance_replans": rows[0]["distance_replans"],
                    "state_sha256": rows[0]["state_sha256"],
                    "success_candidates": [row["candidate"] for row in successes],
                    "failure_candidates": [row["candidate"] for row in failures],
                    "fixed_pair": {
                        "success_candidate": successes[0]["candidate"],
                        "failure_candidate": failures[0]["candidate"],
                    },
                }
            )
        elif successes:
            count["all_success_groups"] += 1
        else:
            count["all_failure_groups"] += 1

    suite_report: dict[str, Any] = {}
    for suite in SUITES:
        suite_report[suite] = {
            **dict(sorted(suite_counts[suite].items())),
            "mixed_independent_sources": len(mixed_sources[suite]),
            "any_failure_independent_sources": len(failure_sources[suite]),
            "required_independent_sources": minimum_sources_per_suite,
            "strict_data_gate": (
                "PASS"
                if len(mixed_sources[suite]) >= minimum_sources_per_suite
                else "FAIL"
            ),
        }
    balanced_sources = min(len(mixed_sources[suite]) for suite in SUITES)
    distribution_note = (
        "All C61 candidates are genuine frozen D0-H32-s14000 diffusion samples, "
        "but C65 would rank candidates from the current C60/C58 deployment carrier; "
        "therefore C61 is a near-neighbor, not an exact deployment-candidate distribution."
    )
    return {
        "format": FORMAT,
        "status": "FAIL_C65_C61_FOUR_SUITE_DATA_GATE_NO_SCORE",
        "effect_status": "NOT_EVALUATED",
        "read_only": True,
        "model_forwards": 0,
        "training_steps": 0,
        "best_of_n_deployment": False,
        "inputs": {
            "c61_root": str(c61_root),
            "c61_frozen_sha256": frozen_sha,
            "c61_jobs_sha256": jobs_sha,
            "c63_pairs": str(c63_pairs_path),
            "c63_pairs_sha256": sha256_file(c63_pairs_path),
        },
        "inventory": {
            "sources": int(frozen["sources"]),
            "groups": len(records_by_group),
            "jobs": len(jobs),
            "successful_jobs": sum(row["successful_jobs"] for row in suite_report.values()),
            "failed_jobs": sum(row["failed_jobs"] for row in suite_report.values()),
            "mixed_groups": len(mixed_groups),
            "strict_balanced_independent_sources_per_suite": balanced_sources,
        },
        "generation_identity": {
            "policy_checkpoint": str(policy_checkpoint),
            "policy_checkpoint_sha256": sha256_file(policy_checkpoint),
            "first_replan_steps": 32,
            "continuation_replan_steps": 8,
            "candidates_per_state": 4,
            "same_continuation_noise_within_group": True,
            "distribution_assessment": distribution_note,
        },
        "mechanical_gates": {
            "all_results_present": len(records_by_group) == int(frozen["groups"]),
            "parent_branch_sim_state_byte_exact": exact["parent_branch_sim_state"] == len(jobs),
            "parent_branch_previous_action_byte_exact": exact["parent_branch_previous_action"] == len(jobs),
            "four_unique_actions_per_group": True,
            "single_policy_checkpoint": True,
        },
        "suite_inventory": suite_report,
        "c63_disjointness": {
            "parent_trajectory_sha_overlap": len(set(parent_sha_cache.values()) & c63_parent_hashes),
            "branch_trajectory_sha_overlap": len(branch_hashes & c63_branch_hashes),
            "state_sha_overlap": len(all_state_hashes & c63_state_hashes),
            "candidate_action_sha_overlap": len(all_action_hashes & c63_action_hashes),
            "parent_proposed_action_sha_overlap": len(
                parent_proposed_action_hashes & c63_action_hashes
            ),
            "suite_task_trial_step_overlap": len(
                {
                    (row["suite"], row["task"], row["trial"], row["start_step"])
                    for rows in records_by_group.values()
                    for row in rows
                }
                & c63_coordinates
            ),
            "pass": not (
                set(parent_sha_cache.values()) & c63_parent_hashes
                or branch_hashes & c63_branch_hashes
                or all_state_hashes & c63_state_hashes
                or all_action_hashes & c63_action_hashes
            ),
        },
        "parent_success_pair_diagnostic": {
            "independent_sources_with_any_failure_by_suite": {
                suite: len(failure_sources[suite]) for suite in SUITES
            },
            "not_primary_reason": (
                "The successful parent executed replan=8 while C61 interventions executed "
                "their first chunk for 32 steps. Pairing those outcomes would use asymmetric "
                "execution contracts and cannot repair the strict mixed-candidate data gap."
            ),
        },
        "mixed_groups": mixed_groups,
        "decision": {
            "data_gate": "FAIL",
            "reason": (
                f"Strict four-suite source-balanced capacity is {balanced_sources}; "
                f"C65 requires at least {minimum_sources_per_suite} independent mixed sources per suite."
            ),
            "permission": "NO_SCORE_COLLECT_C65_DEPLOYMENT_DISTRIBUTION_PAIRS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c61-root", type=Path, required=True)
    parser.add_argument("--c63-pairs", type=Path, required=True)
    parser.add_argument("--expected-frozen-sha256", required=True)
    parser.add_argument("--expected-jobs-sha256", required=True)
    parser.add_argument("--minimum-sources-per-suite", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C65 audit: {output}")
    report = audit(
        args.c61_root.resolve(),
        args.c63_pairs.resolve(),
        expected_frozen_sha256=args.expected_frozen_sha256,
        expected_jobs_sha256=args.expected_jobs_sha256,
        minimum_sources_per_suite=args.minimum_sources_per_suite,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in report.items() if key != "mixed_groups"}, indent=2))


if __name__ == "__main__":
    main()
