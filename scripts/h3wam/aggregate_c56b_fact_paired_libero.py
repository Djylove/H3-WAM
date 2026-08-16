#!/usr/bin/env python3
"""Aggregate matched fresh executions of C56b C60-main and C61 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


FORMAT = "h3wam-c56b-fact-paired-libero-trial33-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARMS = ("c60_main", "c61_matched")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(first_wins: int, second_wins: int) -> dict[str, Any]:
    discordant = first_wins + second_wins
    if discordant == 0:
        return {
            "discordant_pairs": 0, "first_wins": 0, "second_wins": 0,
            "one_sided_p_first_better": 1.0, "two_sided_p": 1.0,
            "method": "exact_binomial_mcnemar_p0_5",
        }
    denominator = 2**discordant
    upper = sum(
        math.comb(discordant, index)
        for index in range(first_wins, discordant + 1)
    ) / denominator
    lower = sum(
        math.comb(discordant, index)
        for index in range(0, min(first_wins, second_wins) + 1)
    ) / denominator
    return {
        "discordant_pairs": discordant,
        "first_wins": first_wins,
        "second_wins": second_wins,
        "one_sided_p_first_better": upper,
        "two_sided_p": min(1.0, 2.0 * lower),
        "method": "exact_binomial_mcnemar_p0_5",
    }


def _endpoint(path: Path, arm: str) -> tuple[dict[str, Any], Path]:
    ready = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(ready.get("checkpoint", "")).resolve()
    if (
        ready.get("status") != "PASS_C56B_ONLINE_LONG10000_STRICT_RESTORE"
        or ready.get("permission") != "READY_FOR_PAIRED_HELDOUT"
        or ready.get("arm") != arm
        or ready.get("completed_steps") != 10_000
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != ready.get("checkpoint_size_bytes")
        or sha256_file(checkpoint) != ready.get("checkpoint_sha256")
    ):
        raise ValueError(f"invalid C56b endpoint: {arm}")
    return ready, checkpoint


def _load_result(
    path: Path, *, label: str, suite: str, checkpoint: Path, policy: str
) -> tuple[dict[tuple[str, int, int], bool], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "policy": policy,
        "suite": suite,
        "task_ids": list(range(10)),
        "trial_indices": [33],
        "trials_per_task": 1,
        "max_steps": 400,
        "wait_steps": 30,
        "replan_steps": 8,
        "action_horizon": 32,
        "model_evaluations": 10,
        "environment_seed": None,
        "policy_noise_seed_base": None,
        "normalized_action_pre_clamp": True,
        "sample_ensemble_size": 1,
        "use_action_ensembler": False,
        "save_trajectories": False,
    }
    mismatch = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"rollout contract mismatch {label}/{suite}: {mismatch}")
    if Path(payload.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError(f"rollout checkpoint mismatch: {label}/{suite}")
    pairs: dict[tuple[str, int, int], bool] = {}
    for task in payload.get("tasks", []):
        task_id = int(task["task_id"])
        for episode in task.get("episodes", []):
            key = (suite, task_id, int(episode["trial"]))
            if key in pairs:
                raise ValueError(f"duplicate rollout episode: {label}/{key}")
            replans = int(episode.get("replans", -1))
            expected_seed = 42 + task_id * 100_000 + 33 * 1_000
            if (
                episode.get("episode_seed") != expected_seed
                or episode.get("environment_seed") is not None
                or replans < 0
                or episode.get("replan_noise_seeds")
                != list(range(expected_seed, expected_seed + replans))
            ):
                raise ValueError(f"rollout episode seed mismatch: {label}/{key}")
            pairs[key] = bool(episode["success"])
    if set(pairs) != {(suite, task, 33) for task in range(10)}:
        raise ValueError(f"incomplete rollout suite: {label}/{suite}")
    return pairs, path


def _load_c58_control(
    path: Path, *, expected_parent_sha256: str
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(report.get("candidate_checkpoint", "")).resolve()
    protocol = report.get("protocol", {})
    if (
        report.get("format") != "h3wam-c58b-vs-d0-paired-fresh-libero-trial33-v2"
        or report.get("status") != "COMPLETE"
        or report.get("paired_episodes_per_arm") != 40
        or report.get("candidate_checkpoint_sha256") != expected_parent_sha256
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != expected_parent_sha256
        or protocol.get("trial_indices") != [33]
        or protocol.get("action_horizon") != 32
        or protocol.get("replan_interval") != 8
        or protocol.get("inference_steps") != 10
        or protocol.get("wait_steps") != 30
        or protocol.get("environment_seed") is not None
        or protocol.get("policy_noise_seed_base") is not None
        or protocol.get("episode_seed_contract")
        != "seed_plus_task_id_times_100000_plus_trial_index_times_1000"
        or protocol.get("suites") != list(SUITES)
        or protocol.get("tasks_per_suite") != 10
        or report.get("paired_episode_keys") != "suite,task_id,trial_index"
    ):
        raise ValueError("C58b parent RESULTS identity/protocol mismatch")
    sources = report.get("sources", {}).get("candidate_c58b", {})
    if set(sources) != set(SUITES):
        raise ValueError("C58b parent RESULTS lacks exact four-suite sources")
    return report, checkpoint, sources


def aggregate(
    root: Path, gate_path: Path, main_ready_path: Path, c61_ready_path: Path,
    c58_results_path: Path,
) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("format") != "h3wam-c56b-fact-online-paired-balanced80-v1"
        or gate.get("status") != "PASS_PAIRED_BALANCED80"
        or gate.get("permission") != "GO_PAIRED_LIBERO"
    ):
        raise ValueError("paired heldout gate did not authorize LIBERO")
    main_ready, main_checkpoint = _endpoint(main_ready_path, "C60_MAIN")
    c61_ready, c61_checkpoint = _endpoint(c61_ready_path, "C61_MATCHED")
    identities = gate.get("checkpoint_identity", {})
    if (
        identities.get("c60_main_ready_sha256") != sha256_file(main_ready_path)
        or identities.get("c61_matched_ready_sha256") != sha256_file(c61_ready_path)
        or identities.get("c60_main_checkpoint_sha256")
        != main_ready["checkpoint_sha256"]
        or identities.get("c61_matched_checkpoint_sha256")
        != c61_ready["checkpoint_sha256"]
        or main_ready.get("c58_parent_sha256")
        != c61_ready.get("c58_parent_sha256")
    ):
        raise ValueError("paired heldout/checkpoint endpoint identity mismatch")
    c58_report, c58_checkpoint, c58_sources = _load_c58_control(
        c58_results_path, expected_parent_sha256=main_ready["c58_parent_sha256"]
    )

    all_results = {arm: {} for arm in (*ARMS, "c58_parent")}
    sources: dict[str, Any] = {arm: {} for arm in (*ARMS, "c58_parent")}
    suite_reports = []
    for suite in SUITES:
        main, main_path = _load_result(
            root / "c60_main" / suite / "results.json",
            label="c60_main", suite=suite, checkpoint=main_checkpoint,
            policy="h3_fact_online_int8",
        )
        c61, c61_path = _load_result(
            root / "c61_matched" / suite / "results.json",
            label="c61_matched", suite=suite, checkpoint=c61_checkpoint,
            policy="h3_fact_online_int8",
        )
        c58_source = c58_sources[suite]
        c58_path = Path(c58_source.get("path", "")).resolve()
        if (
            not c58_path.is_file()
            or sha256_file(c58_path) != c58_source.get("sha256")
        ):
            raise ValueError(f"C58b raw source bytes mismatch: {suite}")
        c58, _ = _load_result(
            c58_path, label="c58_parent", suite=suite,
            checkpoint=c58_checkpoint, policy="h3_fastwam_online_int8",
        )
        if set(main) != set(c61) or set(main) != set(c58):
            raise ValueError(f"three-arm paired episode keys differ: {suite}")
        main_wins = sum(main[key] and not c61[key] for key in main)
        c61_wins = sum(c61[key] and not main[key] for key in main)
        main_over_c58 = sum(main[key] and not c58[key] for key in main)
        c58_over_main = sum(c58[key] and not main[key] for key in main)
        c61_over_c58 = sum(c61[key] and not c58[key] for key in main)
        c58_over_c61 = sum(c58[key] and not c61[key] for key in main)
        suite_reports.append({
            "suite": suite,
            "episodes_per_arm": 10,
            "main_successes": sum(main.values()),
            "c61_successes": sum(c61.values()),
            "c58_parent_successes": sum(c58.values()),
            "main_success_rate": sum(main.values()) / 10,
            "c61_success_rate": sum(c61.values()) / 10,
            "c58_parent_success_rate": sum(c58.values()) / 10,
            "c61_minus_main_success_rate": (sum(c61.values()) - sum(main.values())) / 10,
            "main_minus_c58_success_rate": (sum(main.values()) - sum(c58.values())) / 10,
            "c61_minus_c58_success_rate": (sum(c61.values()) - sum(c58.values())) / 10,
            "c61_vs_main": exact_mcnemar(c61_wins, main_wins),
            "main_vs_c58": exact_mcnemar(main_over_c58, c58_over_main),
            "c61_vs_c58": exact_mcnemar(c61_over_c58, c58_over_c61),
        })
        all_results["c60_main"].update(main)
        all_results["c61_matched"].update(c61)
        all_results["c58_parent"].update(c58)
        sources["c60_main"][suite] = {
            "path": str(main_path.resolve()), "sha256": sha256_file(main_path)
        }
        sources["c61_matched"][suite] = {
            "path": str(c61_path.resolve()), "sha256": sha256_file(c61_path)
        }
        sources["c58_parent"][suite] = {
            "path": str(c58_path), "sha256": sha256_file(c58_path)
        }
    main, c61, c58 = (
        all_results["c60_main"], all_results["c61_matched"],
        all_results["c58_parent"],
    )
    if len(main) != 40 or set(main) != set(c61) or set(main) != set(c58):
        raise ValueError("global three-arm paired LIBERO grid is incomplete")
    main_wins = sum(main[key] and not c61[key] for key in main)
    c61_wins = sum(c61[key] and not main[key] for key in main)
    main_over_c58 = sum(main[key] and not c58[key] for key in main)
    c58_over_main = sum(c58[key] and not main[key] for key in main)
    c61_over_c58 = sum(c61[key] and not c58[key] for key in main)
    c58_over_c61 = sum(c58[key] and not c61[key] for key in main)
    main_successes, c61_successes, c58_successes = (
        sum(main.values()), sum(c61.values()), sum(c58.values())
    )
    return {
        "format": FORMAT,
        "status": "COMPLETE",
        "effect_status": "PAIRED_CLOSED_LOOP_MICRO_BENCHMARK_EVIDENCE",
        "hypothesis": (
            "Replacing only C60 causal failures with C61 improves matched "
            "LIBERO success at fixed execution settings."
        ),
        "main_checkpoint": str(main_checkpoint),
        "main_checkpoint_sha256": main_ready["checkpoint_sha256"],
        "c61_checkpoint": str(c61_checkpoint),
        "c61_checkpoint_sha256": c61_ready["checkpoint_sha256"],
        "shared_c58_parent_sha256": main_ready["c58_parent_sha256"],
        "c58_parent_results": str(c58_results_path.resolve()),
        "c58_parent_results_sha256": sha256_file(c58_results_path),
        "paired_heldout_gate": str(gate_path.resolve()),
        "paired_heldout_gate_sha256": sha256_file(gate_path),
        "protocol": {
            "execution_freshness": "new_policy_execution_on_fixed_heldout_state",
            "globally_unused_init_state": False,
            "reason": "LIBERO exposes only trials0..49 and project history consumed them; trial33 remains model-unseen for C56/C61 training.",
            "suites": list(SUITES), "tasks": list(range(10)),
            "trial_indices": [33], "replan_steps": 8,
            "action_horizon": 32, "inference_steps": 10,
            "wait_steps": 30,
            "environment_seed": None, "policy_noise_seed_base": None,
            "episode_seed_contract": (
                "seed_plus_task_id_times_100000_plus_trial_index_times_1000"
            ),
            "online_h3_no_disk_kv": True,
        },
        "paired_episodes_per_arm": 40,
        "main_successes": main_successes,
        "c61_successes": c61_successes,
        "c58_parent_successes": c58_successes,
        "main_success_rate": main_successes / 40,
        "c61_success_rate": c61_successes / 40,
        "c58_parent_success_rate": c58_successes / 40,
        "c61_minus_main_success_rate": (c61_successes - main_successes) / 40,
        "main_minus_c58_success_rate": (main_successes - c58_successes) / 40,
        "c61_minus_c58_success_rate": (c61_successes - c58_successes) / 40,
        "paired_effects": {
            "c61_vs_main": exact_mcnemar(c61_wins, main_wins),
            "main_vs_c58": exact_mcnemar(main_over_c58, c58_over_main),
            "c61_vs_c58": exact_mcnemar(c61_over_c58, c58_over_c61),
        },
        "suites": suite_reports,
        "sources": sources,
        "claim_boundary": (
            "This is a complete four-suite single-state-index paired micro-benchmark. "
            "It tests the C61-only variable but is not a full 50-state LIBERO claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--main-ready", type=Path, required=True)
    parser.add_argument("--c61-ready", type=Path, required=True)
    parser.add_argument("--c58-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.root.resolve(), args.gate.resolve(), args.main_ready.resolve(),
        args.c61_ready.resolve(),
        args.c58_results.resolve(),
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C56b paired result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
