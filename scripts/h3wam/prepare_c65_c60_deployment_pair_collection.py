#!/usr/bin/env python3
"""Freeze fresh C60 deployment-distribution branches for the C65 data gate."""

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
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DISTANCES = (3, 5, 7, 9)
CANDIDATES = 8
C60_RESULTS_SHA256 = "d9280c5ad4aeac231a8da793ac5f5d667f005dbc8c5cfe3657b93a4895483ec3"
C60_PAIR_EVIDENCE_SHA256 = "b96421a1e5c6d6ff8fe729f5cf3128560a00e64eac3983d20ef505346f3e9b05"
C60_CHECKPOINT_SHA256 = "d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36"
C60_PAIRED_GATE_SHA256 = "c68b2f8bfff6308f97a5f181facd9841c84cfcf9ec22e5e2e561196084337220"
C63_PAIRS_SHA256 = "c3ead728d362d53ad57f02508cb66febfa128dad79e95911d0da89cd40d8c25a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def round_robin_sources(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministically maximize task coverage without reading branch outcomes."""

    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task"])].append(row)
    for task in by_task:
        by_task[task].sort(key=lambda row: (int(row["trial"]), row["trajectory_sha256"]))
    selected: list[dict[str, Any]] = []
    cursor = Counter()
    while len(selected) < count:
        progressed = False
        for task in sorted(by_task):
            index = cursor[task]
            if index < len(by_task[task]):
                selected.append(by_task[task][index])
                cursor[task] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"source pool has fewer than {count} rows")
    return selected


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    results_path = args.c60_results.resolve()
    c63_path = args.c63_pairs.resolve()
    c61_root = args.c61_root.resolve()
    checkpoint = args.checkpoint.resolve()
    paired_gate = args.paired_gate.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C65 collection root: {output_root}")
    if sha256_file(results_path) != C60_RESULTS_SHA256:
        raise ValueError("C60 expanded RESULTS identity mismatch")
    if sha256_file(c63_path) != C63_PAIRS_SHA256:
        raise ValueError("C63 pair manifest identity mismatch")
    if sha256_file(checkpoint) != C60_CHECKPOINT_SHA256:
        raise ValueError("C60 checkpoint identity mismatch")
    if sha256_file(paired_gate) != C60_PAIRED_GATE_SHA256:
        raise ValueError("C60 paired gate identity mismatch")
    gate = json.loads(paired_gate.read_text())
    if (
        gate.get("status") != "PASS_PAIRED_BALANCED80"
        or gate.get("permission") != "GO_PAIRED_LIBERO"
        or gate.get("checkpoint_identity", {}).get("c60_main_checkpoint_sha256")
        != C60_CHECKPOINT_SHA256
    ):
        raise ValueError("C60 paired gate contract mismatch")
    results = json.loads(results_path.read_text())
    if (
        results.get("format")
        != "h3wam-c60-fact-vs-c58-expanded-paired-libero-trials33-49-v1"
        or results.get("candidate_checkpoint_sha256") != C60_CHECKPOINT_SHA256
        or results.get("pair_evidence_sha256") != C60_PAIR_EVIDENCE_SHA256
    ):
        raise ValueError("C60 results contract mismatch")
    evidence_path = Path(results["pair_evidence"]).resolve()
    if sha256_file(evidence_path) != C60_PAIR_EVIDENCE_SHA256:
        raise ValueError("C60 pair evidence identity mismatch")
    evidence = {
        (row["suite"], int(row["task"]), int(row["trial"])): row
        for row in jsonl(evidence_path)
    }

    c63 = json.loads(c63_path.read_text())
    forbidden_parent_sha = {
        value
        for row in c63["pairs"]
        for value in (
            row["parent_trajectory_sha256"],
            row["branch_trajectory_sha256"],
        )
    }
    forbidden_states: set[str] = set()
    for row in c63["pairs"]:
        with np.load(row["parent_trajectory"], allow_pickle=False) as archive:
            forbidden_states.add(
                sha256_array(archive["sim_state"][int(row["parent_index"])]),
            )
    c61_jobs = jsonl(c61_root / "jobs.jsonl")
    for row in c61_jobs[::4]:
        ordinal, group_id, candidate = (
            int(row["ordinal"]), int(row["group_id"]), int(row["candidate"])
        )
        matches = list((c61_root / "runs").glob(
            f"{ordinal}_g{group_id}_c{candidate}_*/results.json"
        ))
        if len(matches) != 1:
            raise ValueError("C61 disjointness inventory is incomplete")
        episode = json.loads(matches[0].read_text())["tasks"][0]["episodes"][0]
        with np.load(episode["trajectory"], allow_pickle=False) as archive:
            forbidden_states.add(sha256_array(archive["sim_state"][0]))

    outcome = {
        (row["suite"], int(row["task"]), int(row["trial"])): bool(row["candidate"])
        for row in results["pair_outcomes"]
    }
    source_pool: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    for identity, row in sorted(evidence.items()):
        suite, task, trial = identity
        if suite not in source_pool or not outcome.get(identity, False):
            continue
        result_path = Path(row["candidate_result"]).resolve()
        trajectory = Path(row["candidate_trajectory"]).resolve()
        if sha256_file(result_path) != row["candidate_result_sha256"]:
            raise ValueError("C60 source result changed")
        if sha256_file(trajectory) != row["candidate_trajectory_sha256"]:
            raise ValueError("C60 source trajectory changed")
        source_result = json.loads(result_path.read_text())
        episodes = source_result["tasks"][0]["episodes"]
        if (
            source_result.get("checkpoint") != str(checkpoint)
            or len(episodes) != 1
            or not bool(episodes[0]["success"])
            or Path(episodes[0]["trajectory"]).resolve() != trajectory
        ):
            raise ValueError("C60 source is not an exact successful parent")
        if row["candidate_trajectory_sha256"] in forbidden_parent_sha:
            raise ValueError("C65 source trajectory overlaps C63")
        with np.load(trajectory, allow_pickle=False) as archive:
            if len(archive["step"]) < max(DISTANCES):
                continue
        source_pool[suite].append(
            {
                "suite": suite,
                "task": task,
                "trial": trial,
                "result": str(result_path),
                "result_sha256": row["candidate_result_sha256"],
                "trajectory": str(trajectory),
                "trajectory_sha256": row["candidate_trajectory_sha256"],
            }
        )

    sources: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    suite_counts: dict[str, Any] = {}
    for suite in SUITES:
        selected = round_robin_sources(source_pool[suite], args.sources_per_suite)
        suite_counts[suite] = {
            "available_success_sources": len(source_pool[suite]),
            "selected_sources": len(selected),
            "selected_tasks": dict(sorted(Counter(row["task"] for row in selected).items())),
        }
        for source_row in selected:
            source_id = len(sources)
            trajectory = Path(source_row["trajectory"])
            with np.load(trajectory, allow_pickle=False) as archive:
                steps = archive["step"].astype(np.int64)
                states = archive["sim_state"]
                source = {
                    "source_id": source_id,
                    **source_row,
                    "state_count": len(steps),
                }
                sources.append(source)
                for distance in DISTANCES:
                    index = len(steps) - distance
                    state_sha = sha256_array(states[index])
                    if state_sha in forbidden_states:
                        raise ValueError("C65 state overlaps C61/C63")
                    group_id = len(groups)
                    group = {
                        **source,
                        "group_id": group_id,
                        "distance_replans": distance,
                        "index": index,
                        "start_step": int(steps[index]),
                        "sim_state_sha256": state_sha,
                        "continuation_policy_noise_seed_base": 365_000_000
                        + group_id * 10_000,
                    }
                    groups.append(group)
                    base_seed = 65_000_000 + group_id * 100
                    for candidate in range(CANDIDATES):
                        jobs.append(
                            {
                                **group,
                                "ordinal": len(jobs),
                                "candidate": candidate,
                                "first_policy_noise_seed": base_seed
                                + candidate * 1_000_000,
                            }
                        )
    expected_sources = args.sources_per_suite * len(SUITES)
    if (
        len(sources) != expected_sources
        or len(groups) != expected_sources * len(DISTANCES)
        or len(jobs) != len(groups) * CANDIDATES
        or len({row["sim_state_sha256"] for row in groups}) != len(groups)
    ):
        raise ValueError("C65 frozen source/group/job inventory mismatch")

    output_root.mkdir(parents=True)
    jobs_path = output_root / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs),
        encoding="utf-8",
    )
    prepared = {
        "format": FORMAT,
        "status": "PASS_C65_COLLECTION_FROZEN_NOT_EXECUTED",
        "effect_status": "NOT_EVALUATED",
        "score_permission": "NO_SCORE_UNTIL_20_MIXED_SOURCES_PER_SUITE",
        "sources": len(sources),
        "groups": len(groups),
        "jobs": len(jobs),
        "sources_per_suite": args.sources_per_suite,
        "distances_replans": list(DISTANCES),
        "candidates_per_state": CANDIDATES,
        "suite_inventory": suite_counts,
        "candidate_policy": "h3_fact_online_int8",
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": C60_CHECKPOINT_SHA256,
        "paired_gate": str(paired_gate),
        "paired_gate_sha256": C60_PAIRED_GATE_SHA256,
        "source_results": str(results_path),
        "source_results_sha256": C60_RESULTS_SHA256,
        "source_pair_evidence_sha256": C60_PAIR_EVIDENCE_SHA256,
        "c63_pairs_sha256": C63_PAIRS_SHA256,
        "c61_jobs_sha256": sha256_file(c61_root / "jobs.jsonl"),
        "jobs_sha256": sha256_file(jobs_path),
        "execution_contract": {
            "wait_steps": 0,
            "first_replan_steps": 8,
            "continuation_replan_steps": 8,
            "action_horizon": 32,
            "model_evaluations": 10,
            "same_continuation_noise_within_group": True,
            "all_outcomes_hidden_until_complete": True,
        },
        "pair_contract": (
            "After complete collection, choose at most one state per parent source; "
            "within it use the smallest successful and smallest failed candidate ids."
        ),
        "data_gate": "exactly 20 source-independent mixed pairs per suite or fail closed",
    }
    prepared_path = output_root / "PREPARED.json"
    temporary = prepared_path.with_name(f".{prepared_path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, prepared_path)
    (output_root / "runs").mkdir()
    (output_root / "logs").mkdir()
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c60-results", type=Path, required=True)
    parser.add_argument("--c63-pairs", type=Path, required=True)
    parser.add_argument("--c61-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired-gate", type=Path, required=True)
    parser.add_argument("--sources-per-suite", type=int, default=24)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.sources_per_suite < 20:
        raise ValueError("C65 source pool must contain at least 20 parents per suite")
    report = prepare(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
