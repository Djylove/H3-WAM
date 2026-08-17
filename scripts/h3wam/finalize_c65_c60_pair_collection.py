#!/usr/bin/env python3
"""Complete-only C65 data gate and deterministic same-state pair freezer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "h3wam-c65-c60-deployment-pair-collection-v1"
GATE_FORMAT = "h3wam-c65-c60-deployment-pair-data-gate-v1"
PAIR_FORMAT = "h3wam-c65-c60-deployment-pairs-v1"
PREPARED_SHA256 = "a883db2662acbb8a2bb31fa9ebbd7ff344ab01d1af5626d5d02def07a0e1158a"
JOBS_SHA256 = "c9a13ede1ea111450ff4bd4f893fd729fc55190f92ce45e76e0240a9001b52cf"
C60_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MINIMUM_PER_SUITE = 20
HORIZON = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def action_identity(actions: np.ndarray) -> str:
    value = np.asarray(actions, dtype="<f4")
    if value.shape != (HORIZON, 7):
        raise ValueError("C65 candidate action shape mismatch")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_result(root: Path, job: dict[str, Any]) -> Path:
    prefix = f"{int(job['ordinal'])}_g{int(job['group_id'])}_c{int(job['candidate'])}_"
    matches = sorted((root / "runs").glob(f"{prefix}*/results.json"))
    if len(matches) != 1:
        raise ValueError(f"C65 expected one result for {prefix}, got {len(matches)}")
    return matches[0].resolve()


def validate_markers(root: Path) -> dict[str, str]:
    expected = {
        "node-n1-spatial-object.COMPLETED": {
            "node_tag": "n1-spatial-object",
            "suites": "libero_spatial,libero_object",
            "jobs": 1536,
        },
        "node-n2-goal-10.COMPLETED": {
            "node_tag": "n2-goal-10",
            "suites": "libero_goal,libero_10",
            "jobs": 1536,
        },
    }
    hashes = {}
    for name, fixed in expected.items():
        path = root / name
        payload = json.loads(path.read_text())
        if any(payload.get(key) != value for key, value in fixed.items()):
            raise ValueError(f"C65 completion marker mismatch: {name}")
        if int(payload.get("duration_seconds", -1)) < 0:
            raise ValueError(f"C65 completion duration invalid: {name}")
        hashes[name] = sha256_file(path)
    unexpected = {path.name for path in root.glob("node-*.COMPLETED")} - set(expected)
    if unexpected:
        raise ValueError(f"unexpected C65 completion markers: {sorted(unexpected)}")
    return hashes


def finalize(root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prepared_path, jobs_path = root / "PREPARED.json", root / "jobs.jsonl"
    if sha256_file(prepared_path) != PREPARED_SHA256 or sha256_file(jobs_path) != JOBS_SHA256:
        raise ValueError("C65 prepared/jobs identity mismatch")
    prepared = json.loads(prepared_path.read_text())
    jobs = jsonl(jobs_path)
    if (
        prepared.get("format") != FORMAT
        or prepared.get("status") != "PASS_C65_COLLECTION_FROZEN_NOT_EXECUTED"
        or prepared.get("jobs") != 3072
        or len(jobs) != 3072
        or [int(row["ordinal"]) for row in jobs] != list(range(3072))
    ):
        raise ValueError("C65 frozen collection contract mismatch")
    marker_hashes = validate_markers(root)
    expected_results = {find_result(root, job) for job in jobs}
    actual_results = {path.resolve() for path in (root / "runs").glob("*/results.json")}
    if actual_results != expected_results:
        raise ValueError("C65 result inventory is not exactly 3072 frozen jobs")

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    result_digest = hashlib.sha256()
    for job in jobs:
        result_path = find_result(root, job)
        result_sha = sha256_file(result_path)
        result_digest.update(f"{int(job['ordinal'])}:{result_sha}\n".encode())
        result = json.loads(result_path.read_text())
        episodes = result.get("tasks", [{}])[0].get("episodes", [])
        required = {
            "policy": "h3_fact_online_int8",
            "checkpoint": prepared["candidate_checkpoint"],
            "suite": str(job["suite"]),
            "task_ids": [int(job["task"])],
            "trial_indices": [int(job["trial"])],
            "start_trajectory": str(Path(job["trajectory"]).resolve()),
            "start_index": int(job["index"]),
            "first_policy_noise_seed": int(job["first_policy_noise_seed"]),
            "continuation_policy_noise_seed_base": int(
                job["continuation_policy_noise_seed_base"]
            ),
            "first_replan_steps": 8,
            "replan_steps": 8,
            "save_trajectories": True,
        }
        if len(episodes) != 1 or any(result.get(key) != value for key, value in required.items()):
            raise ValueError(f"C65 branch identity mismatch: {result_path}")
        episode = episodes[0]
        trajectory = Path(episode["trajectory"]).resolve()
        parent = Path(job["trajectory"]).resolve()
        with np.load(parent, allow_pickle=False) as source, np.load(
            trajectory, allow_pickle=False
        ) as branch:
            index = int(job["index"])
            if (
                int(source["step"][index]) != int(job["start_step"])
                or int(branch["step"][0]) != int(job["start_step"])
                or not np.array_equal(source["sim_state"][index], branch["sim_state"][0])
                or not np.array_equal(
                    source["previous_action"][index], branch["previous_action"][0]
                )
                or sha256_array(branch["sim_state"][0]) != job["sim_state_sha256"]
            ):
                raise ValueError("C65 exact restored-state gate failed")
            actions = np.asarray(branch["policy_actions"][0], dtype=np.float32)
            if not np.array_equal(
                actions,
                np.asarray(episode["first_environment_action_chunk"], dtype=np.float32),
            ):
                raise ValueError("C65 result/trajectory first action mismatch")
            observation_identity = {
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256_file(trajectory),
                "row_index": 0,
                "step": int(branch["step"][0]),
            }
        groups[int(job["group_id"])].append(
            {
                "job": job,
                "success": bool(episode["success"]),
                "action": actions,
                "action_sha256": action_identity(actions),
                "observation": observation_identity,
                "task_language": str(result["tasks"][0]["task"]),
                "result": str(result_path),
                "result_sha256": result_sha,
            }
        )

    mixed_by_source: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    suite_group_counts = {suite: Counter() for suite in SUITES}
    for group_id, rows in sorted(groups.items()):
        if sorted(int(row["job"]["candidate"]) for row in rows) != list(range(8)):
            raise ValueError("C65 group lacks candidates 0..7")
        if len({row["job"]["sim_state_sha256"] for row in rows}) != 1:
            raise ValueError("C65 group state identity drift")
        if len({row["action_sha256"] for row in rows}) != 8:
            raise ValueError("C65 group candidate actions are not unique")
        successes = [row for row in rows if row["success"]]
        failures = [row for row in rows if not row["success"]]
        suite, source_id = str(rows[0]["job"]["suite"]), int(rows[0]["job"]["source_id"])
        counts = suite_group_counts[suite]
        counts["groups"] += 1
        counts["successful_jobs"] += len(successes)
        counts["failed_jobs"] += len(failures)
        if successes and failures:
            counts["mixed_groups"] += 1
            mixed_by_source[(suite, source_id)].append(
                {
                    "group_id": group_id,
                    "rows": rows,
                    "successes": successes,
                    "failures": failures,
                }
            )
        elif successes:
            counts["all_success_groups"] += 1
        else:
            counts["all_failure_groups"] += 1

    eligible_sources = {
        suite: sorted(
            source_id for candidate_suite, source_id in mixed_by_source
            if candidate_suite == suite
        )
        for suite in SUITES
    }
    passed = all(len(eligible_sources[suite]) >= MINIMUM_PER_SUITE for suite in SUITES)
    pairs: list[dict[str, Any]] = []
    if passed:
        for suite in SUITES:
            for source_id in eligible_sources[suite][:MINIMUM_PER_SUITE]:
                # group_id follows the frozen distance order 3,5,7,9.
                selected_group = min(
                    mixed_by_source[(suite, source_id)], key=lambda row: row["group_id"]
                )
                success = min(
                    selected_group["successes"],
                    key=lambda row: int(row["job"]["candidate"]),
                )
                failure = min(
                    selected_group["failures"],
                    key=lambda row: int(row["job"]["candidate"]),
                )
                observation = min(
                    selected_group["rows"],
                    key=lambda row: int(row["job"]["candidate"]),
                )["observation"]
                job = success["job"]
                pairs.append(
                    {
                        "pair_index": len(pairs),
                        "source_id": source_id,
                        "group_id": int(job["group_id"]),
                        "suite": suite,
                        "task": int(job["task"]),
                        "trial": int(job["trial"]),
                        "start_step": int(job["start_step"]),
                        "distance_replans": int(job["distance_replans"]),
                        "sim_state_sha256": job["sim_state_sha256"],
                        "observation": observation,
                        "task_language": success["task_language"],
                        "success_candidate": int(success["job"]["candidate"]),
                        "success_actions": success["action"].tolist(),
                        "success_action_sha256": success["action_sha256"],
                        "failure_candidate": int(failure["job"]["candidate"]),
                        "failure_actions": failure["action"].tolist(),
                        "failure_action_sha256": failure["action_sha256"],
                    }
                )
        if len(pairs) != 80 or Counter(row["suite"] for row in pairs) != Counter(
            {suite: 20 for suite in SUITES}
        ):
            raise RuntimeError("C65 selected pair inventory drift")

    suite_inventory = {
        suite: {
            **dict(sorted(suite_group_counts[suite].items())),
            "frozen_parent_sources": 24,
            "eligible_mixed_sources": len(eligible_sources[suite]),
            "source_coverage": len(eligible_sources[suite]) / 24,
            "required_mixed_sources": MINIMUM_PER_SUITE,
            "gate": "PASS" if len(eligible_sources[suite]) >= MINIMUM_PER_SUITE else "FAIL",
        }
        for suite in SUITES
    }
    pair_manifest = None
    if passed:
        pair_manifest = {
            "format": PAIR_FORMAT,
            "status": "PASS_C65_FIXED_80_PAIR_MECHANICAL_PREPARATION",
            "effect_status": "NOT_EVALUATED",
            "prepared_sha256": PREPARED_SHA256,
            "jobs_sha256": JOBS_SHA256,
            "checkpoint_sha256": C60_SHA256,
            "pair_count": 80,
            "suite_counts": {suite: 20 for suite in SUITES},
            "source_independent": True,
            "same_state_same_execution_contract": True,
            "selection": "smallest source_id; smallest mixed group_id; smallest success/failure candidate ids",
            "pairs": pairs,
        }
    gate = {
        "format": GATE_FORMAT,
        "status": (
            "PASS_C65_FOUR_SUITE_PAIR_DATA_GATE"
            if passed else "FAIL_C65_FOUR_SUITE_PAIR_DATA_GATE_NO_SCORE"
        ),
        "effect_status": "NOT_EVALUATED",
        "permission": "GO_SCORE_C65" if passed else "NO_SCORE_DATA_COVERAGE_GAP",
        "prepared_sha256": PREPARED_SHA256,
        "jobs_sha256": JOBS_SHA256,
        "checkpoint_sha256": C60_SHA256,
        "completion_markers": marker_hashes,
        "result_inventory_sha256": result_digest.hexdigest(),
        "jobs": len(jobs),
        "suite_inventory": suite_inventory,
        "pair_count": len(pairs),
        "all_results_audited": True,
        "model_forwards": 0,
        "training_steps": 0,
        "claim_boundary": "Eligibility/data gate only; no model score, training, or best-of-N effect.",
    }
    return gate, pair_manifest


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    gate_path, pair_path = root / "DATA_GATE.json", root / "PAIRS.json"
    if gate_path.exists() or pair_path.exists():
        raise FileExistsError("refusing existing C65 finalizer outputs")
    gate, pairs = finalize(root)
    if pairs is not None:
        atomic_json(pair_path, pairs)
        gate["pairs_path"] = str(pair_path)
        gate["pairs_sha256"] = sha256_file(pair_path)
    atomic_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
