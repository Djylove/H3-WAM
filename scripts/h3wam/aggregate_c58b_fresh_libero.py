#!/usr/bin/env python3
"""Strictly aggregate the four-suite C58b trial-33 fresh canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ARMS = {
    "candidate_c58b": "h3_fastwam_online_int8",
    "control_d0": "h3_dreamwam_kv_int8",
}


def _exact_mcnemar(candidate_wins: int, control_wins: int) -> dict[str, Any]:
    discordant = candidate_wins + control_wins
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "candidate_wins": 0,
            "control_wins": 0,
            "one_sided_p_candidate_better": 1.0,
            "two_sided_p": 1.0,
            "method": "exact_binomial_mcnemar_p0_5",
        }
    denominator = 2**discordant
    upper = sum(
        math.comb(discordant, index)
        for index in range(candidate_wins, discordant + 1)
    ) / denominator
    lower_tail_at_smaller = sum(
        math.comb(discordant, index)
        for index in range(0, min(candidate_wins, control_wins) + 1)
    ) / denominator
    return {
        "discordant_pairs": discordant,
        "candidate_wins": candidate_wins,
        "control_wins": control_wins,
        "one_sided_p_candidate_better": upper,
        "two_sided_p": min(1.0, 2.0 * lower_tail_at_smaller),
        "method": "exact_binomial_mcnemar_p0_5",
    }


def _load_arm(
    root: Path,
    *,
    arm: str,
    suite: str,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int, int], bool], Path]:
    path = root / arm / suite / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("policy") != ARMS[arm]
        or payload.get("suite") != suite
        or payload.get("task_ids") != list(range(10))
        or payload.get("trial_indices") != [33]
        or payload.get("trials_per_task") != 1
        or payload.get("replan_steps") != 8
        or payload.get("action_horizon") != 32
        or payload.get("model_evaluations") != 10
        or payload.get("wait_steps") != 30
        or payload.get("environment_seed") is not None
        or payload.get("policy_noise_seed_base") is not None
        or payload.get("normalized_action_pre_clamp") is not True
        or payload.get("sample_ensemble_size") != 1
        or payload.get("use_action_ensembler") is not False
        or payload.get("save_trajectories") is not False
    ):
        raise ValueError(f"paired rollout contract mismatch: {arm}/{suite}")
    if Path(payload.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError(f"paired rollout checkpoint mismatch: {arm}/{suite}")
    pairs: dict[tuple[str, int, int], bool] = {}
    for task in payload.get("tasks", []):
        task_id = int(task["task_id"])
        for episode in task.get("episodes", []):
            key = (suite, task_id, int(episode["trial"]))
            if key in pairs:
                raise ValueError(f"duplicate paired episode: {arm}/{key}")
            if (
                episode.get("episode_seed") != 33_042
                or episode.get("environment_seed") is not None
                or episode.get("replan_noise_seeds") != list(range(33_042, 33_092))
            ):
                raise ValueError(f"paired episode seed mismatch: {arm}/{key}")
            pairs[key] = bool(episode.get("success"))
    expected = {(suite, task_id, 33) for task_id in range(10)}
    if set(pairs) != expected:
        raise ValueError(f"paired suite is incomplete: {arm}/{suite}")
    return payload, pairs, path


def aggregate(root: Path, gate: Path, d0_checkpoint: Path) -> dict[str, Any]:
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("permission") != "GO_FRESH_LIBERO":
        raise ValueError("balanced80 gate did not authorize fresh LIBERO")
    candidate_checkpoint = Path(gate_payload["checkpoint"]).resolve()
    d0_checkpoint = d0_checkpoint.resolve()
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    d0_checkpoint_sha256 = sha256_file(d0_checkpoint)
    if d0_checkpoint_sha256 != EXPECTED_D0_SHA256:
        raise ValueError("D0 parent checkpoint SHA256 mismatch")
    rows = []
    sources: dict[str, Any] = {arm: {} for arm in ARMS}
    all_candidate: dict[tuple[str, int, int], bool] = {}
    all_control: dict[tuple[str, int, int], bool] = {}
    for suite in SUITES:
        _, candidate_pairs, candidate_path = _load_arm(
            root, arm="candidate_c58b", suite=suite,
            checkpoint=candidate_checkpoint,
        )
        _, control_pairs, control_path = _load_arm(
            root, arm="control_d0", suite=suite, checkpoint=d0_checkpoint,
        )
        if set(candidate_pairs) != set(control_pairs):
            raise ValueError(f"candidate/control pair keys differ: {suite}")
        candidate_successes = sum(candidate_pairs.values())
        control_successes = sum(control_pairs.values())
        candidate_wins = sum(
            candidate_pairs[key] and not control_pairs[key]
            for key in candidate_pairs
        )
        control_wins = sum(
            control_pairs[key] and not candidate_pairs[key]
            for key in candidate_pairs
        )
        rows.append({
            "suite": suite,
            "episodes": 10,
            "candidate_successes": candidate_successes,
            "control_successes": control_successes,
            "candidate_success_rate": candidate_successes / 10,
            "control_success_rate": control_successes / 10,
            "success_rate_delta": (candidate_successes - control_successes) / 10,
            **_exact_mcnemar(candidate_wins, control_wins),
        })
        all_candidate.update(candidate_pairs)
        all_control.update(control_pairs)
        sources["candidate_c58b"][suite] = {
            "path": str(candidate_path.resolve()),
            "sha256": sha256_file(candidate_path),
        }
        sources["control_d0"][suite] = {
            "path": str(control_path.resolve()),
            "sha256": sha256_file(control_path),
        }
    if len(all_candidate) != 40 or set(all_candidate) != set(all_control):
        raise ValueError("global paired episode identity mismatch")
    candidate_successes = sum(all_candidate.values())
    control_successes = sum(all_control.values())
    candidate_wins = sum(
        all_candidate[key] and not all_control[key] for key in all_candidate
    )
    control_wins = sum(
        all_control[key] and not all_candidate[key] for key in all_candidate
    )
    paired = _exact_mcnemar(candidate_wins, control_wins)
    return {
        "format": "h3wam-c58b-vs-d0-paired-fresh-libero-trial33-v2",
        "status": "COMPLETE",
        "effect_status": "PAIRED_CLOSED_LOOP_MICRO_BENCHMARK_EVIDENCE",
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "control": "D0_REPEAT_LAYER49_PARENT",
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": gate_payload["checkpoint_sha256"],
        "control_checkpoint": str(d0_checkpoint),
        "control_checkpoint_sha256": d0_checkpoint_sha256,
        "balanced80_gate": str(gate.resolve()),
        "balanced80_gate_sha256": sha256_file(gate),
        "protocol": gate_payload["closed_loop_protocol"],
        "paired_episode_keys": "suite,task_id,trial_index",
        "paired_episodes_per_arm": 40,
        "candidate_successes": candidate_successes,
        "control_successes": control_successes,
        "candidate_success_rate": candidate_successes / 40,
        "control_success_rate": control_successes / 40,
        "success_rate_delta": (candidate_successes - control_successes) / 40,
        "paired_effect": paired,
        "suites": rows,
        "sources": sources,
        "claim_boundary": (
            "Strictly paired fresh trial-33 single-seed micro-benchmark across all "
            "40 LIBERO tasks. It estimates a paired effect but a full multi-trial "
            "benchmark is required for a robust generalization claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--d0-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.root.resolve(), args.gate.resolve(), args.d0_checkpoint.resolve()
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
