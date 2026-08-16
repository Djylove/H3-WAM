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


def exact_mcnemar(c61_wins: int, main_wins: int) -> dict[str, Any]:
    discordant = c61_wins + main_wins
    if discordant == 0:
        return {
            "discordant_pairs": 0, "c61_wins": 0, "main_wins": 0,
            "one_sided_p_c61_better": 1.0, "two_sided_p": 1.0,
            "method": "exact_binomial_mcnemar_p0_5",
        }
    denominator = 2**discordant
    upper = sum(
        math.comb(discordant, index)
        for index in range(c61_wins, discordant + 1)
    ) / denominator
    lower = sum(
        math.comb(discordant, index)
        for index in range(0, min(c61_wins, main_wins) + 1)
    ) / denominator
    return {
        "discordant_pairs": discordant,
        "c61_wins": c61_wins,
        "main_wins": main_wins,
        "one_sided_p_c61_better": upper,
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


def _load_suite(
    root: Path, arm: str, suite: str, checkpoint: Path
) -> tuple[dict[tuple[str, int, int], bool], Path]:
    path = root / arm / suite / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "policy": "h3_fact_online_int8",
        "suite": suite,
        "task_ids": list(range(10)),
        "trial_indices": [33],
        "trials_per_task": 1,
        "max_steps": 400,
        "wait_steps": 0,
        "replan_steps": 8,
        "action_horizon": 32,
        "model_evaluations": 10,
        "environment_seed": 42,
        "policy_noise_seed_base": 330_042,
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
        raise ValueError(f"rollout contract mismatch {arm}/{suite}: {mismatch}")
    if Path(payload.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError(f"rollout checkpoint mismatch: {arm}/{suite}")
    pairs: dict[tuple[str, int, int], bool] = {}
    for task in payload.get("tasks", []):
        task_id = int(task["task_id"])
        for episode in task.get("episodes", []):
            key = (suite, task_id, int(episode["trial"]))
            if key in pairs:
                raise ValueError(f"duplicate rollout episode: {arm}/{key}")
            pairs[key] = bool(episode["success"])
    if set(pairs) != {(suite, task, 33) for task in range(10)}:
        raise ValueError(f"incomplete rollout suite: {arm}/{suite}")
    return pairs, path


def aggregate(
    root: Path, gate_path: Path, main_ready_path: Path, c61_ready_path: Path
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

    all_results = {arm: {} for arm in ARMS}
    sources: dict[str, Any] = {arm: {} for arm in ARMS}
    suite_reports = []
    for suite in SUITES:
        main, main_path = _load_suite(
            root, "c60_main", suite, main_checkpoint
        )
        c61, c61_path = _load_suite(
            root, "c61_matched", suite, c61_checkpoint
        )
        if set(main) != set(c61):
            raise ValueError(f"paired episode keys differ: {suite}")
        main_wins = sum(main[key] and not c61[key] for key in main)
        c61_wins = sum(c61[key] and not main[key] for key in main)
        suite_reports.append({
            "suite": suite,
            "episodes_per_arm": 10,
            "main_successes": sum(main.values()),
            "c61_successes": sum(c61.values()),
            "main_success_rate": sum(main.values()) / 10,
            "c61_success_rate": sum(c61.values()) / 10,
            "c61_minus_main_success_rate": (sum(c61.values()) - sum(main.values())) / 10,
            **exact_mcnemar(c61_wins, main_wins),
        })
        all_results["c60_main"].update(main)
        all_results["c61_matched"].update(c61)
        sources["c60_main"][suite] = {
            "path": str(main_path.resolve()), "sha256": sha256_file(main_path)
        }
        sources["c61_matched"][suite] = {
            "path": str(c61_path.resolve()), "sha256": sha256_file(c61_path)
        }
    main, c61 = all_results["c60_main"], all_results["c61_matched"]
    if len(main) != 40 or set(main) != set(c61):
        raise ValueError("global paired LIBERO grid is incomplete")
    main_wins = sum(main[key] and not c61[key] for key in main)
    c61_wins = sum(c61[key] and not main[key] for key in main)
    main_successes, c61_successes = sum(main.values()), sum(c61.values())
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
        "paired_heldout_gate": str(gate_path.resolve()),
        "paired_heldout_gate_sha256": sha256_file(gate_path),
        "protocol": {
            "execution_freshness": "new_policy_execution_on_fixed_heldout_state",
            "globally_unused_init_state": False,
            "reason": "LIBERO exposes only trials0..49 and project history consumed them; trial33 remains model-unseen for C56/C61 training.",
            "suites": list(SUITES), "tasks": list(range(10)),
            "trial_indices": [33], "replan_steps": 8,
            "action_horizon": 32, "inference_steps": 10,
            "environment_seed": 42, "policy_noise_seed_base": 330_042,
            "online_h3_no_disk_kv": True,
        },
        "paired_episodes_per_arm": 40,
        "main_successes": main_successes,
        "c61_successes": c61_successes,
        "main_success_rate": main_successes / 40,
        "c61_success_rate": c61_successes / 40,
        "c61_minus_main_success_rate": (c61_successes - main_successes) / 40,
        "paired_effect": exact_mcnemar(c61_wins, main_wins),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.root.resolve(), args.gate.resolve(), args.main_ready.resolve(),
        args.c61_ready.resolve(),
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
