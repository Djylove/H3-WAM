#!/usr/bin/env python3
"""Freeze the 32 held-out same-state action pairs for the C63 Stage-2 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


FORMAT = "h3wam-c63-fact-stage2-within-state-pairs-v1"
DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
EXPECTED_SUITE_COUNTS = {"libero_object": 2, "libero_spatial": 30}
HORIZON = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def executed_chunk(
    archive: Mapping[str, np.ndarray], start_index: int, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce C60's actual-between-replans action reconstruction exactly."""

    steps = np.asarray(archive["step"])
    policy_actions = np.asarray(archive["policy_actions"])
    terminal_step = int(np.asarray(archive["terminal_step"]).item())
    if (
        steps.ndim != 1
        or policy_actions.shape != (len(steps), HORIZON, 7)
        or not 0 <= start_index < len(steps)
        or horizon != HORIZON
    ):
        raise ValueError("trajectory action/step contract mismatch")
    chunks: list[np.ndarray] = []
    for cursor in range(start_index, len(steps)):
        segment_end = int(steps[cursor + 1]) if cursor + 1 < len(steps) else terminal_step
        take = max(0, min(horizon, segment_end - int(steps[cursor])))
        chunks.append(policy_actions[cursor, :take].astype(np.float32, copy=False))
        if sum(len(chunk) for chunk in chunks) >= horizon:
            break
    actions = np.concatenate(chunks, axis=0)[:horizon] if chunks else np.empty((0, 7), np.float32)
    padded = np.zeros((horizon, 7), dtype=np.float32)
    padded[: len(actions)] = actions
    is_pad = np.arange(horizon) >= len(actions)
    return padded, is_pad


def tensor_identity(actions: np.ndarray, is_pad: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(actions, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(is_pad, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_pairs(dataset_path: Path, observations_path: Path) -> dict[str, Any]:
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise ValueError("C63 C60 dataset SHA256 mismatch")
    if sha256_file(observations_path) != OBSERVATIONS_SHA256:
        raise ValueError("C63 C60 observations SHA256 mismatch")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "h3wam-c60-counterfactual-failure-dataset-v1":
        raise ValueError("C63 input is not the fixed C60 dataset")
    observations = {int(row["observation_id"]): row for row in _jsonl(observations_path)}
    rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["samples"]:
        if row.get("split") == "validation":
            rows_by_episode[int(row["episode_id"])].append(row)
    episodes = [row for row in payload["episodes"] if row.get("split") == "validation"]
    if len(episodes) != 32 or len(rows_by_episode) != 32:
        raise ValueError("C63 requires exactly 32 validation episodes")

    pairs: list[dict[str, Any]] = []
    drift_max: dict[str, float] = defaultdict(float)
    drift_exact: Counter[str] = Counter()
    identity_keys = ("sim_state", "previous_action")
    observed_keys = (
        "agentview_image", "wristview_image", "eef_pos", "eef_quat", "gripper_qpos"
    )
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        episode_id = int(episode["episode_index"])
        onset = int(episode["failure_active_from_step"])
        matches = [
            row for row in rows_by_episode[episode_id]
            if int(row["current_step"]) == onset
        ]
        if len(matches) != 1:
            raise ValueError(f"episode {episode_id} does not have one onset row")
        row = matches[0]
        observation = observations[int(row["current_observation_id"])]
        branch_path = Path(episode["trajectory"]).resolve()
        parent_path = Path(episode["successful_parent_trajectory"]).resolve()
        if (
            Path(observation["trajectory"]).resolve() != branch_path
            or observation.get("kind") != "row"
            or int(observation["step"]) != onset
        ):
            raise ValueError("C63 onset observation provenance mismatch")
        branch_sha = sha256_file(branch_path)
        if branch_sha != episode["trajectory_sha256"]:
            raise ValueError("C63 branch trajectory bytes changed")
        parent_sha = sha256_file(parent_path)
        with np.load(branch_path, allow_pickle=False) as branch, np.load(
            parent_path, allow_pickle=False
        ) as parent:
            branch_positions = np.flatnonzero(branch["step"] == onset)
            parent_positions = np.flatnonzero(parent["step"] == onset)
            if len(branch_positions) != 1 or len(parent_positions) != 1:
                raise ValueError("C63 onset step is not unique")
            branch_index, parent_index = int(branch_positions[0]), int(parent_positions[0])
            if branch_index != int(observation["row_index"]):
                raise ValueError("C63 branch observation index mismatch")
            for key in identity_keys:
                if not np.array_equal(branch[key][branch_index], parent[key][parent_index]):
                    raise ValueError(f"C63 exact restored-state gate failed: {key}")
            for key in observed_keys:
                left = branch[key][branch_index]
                right = parent[key][parent_index]
                exact = bool(np.array_equal(left, right))
                drift_exact[f"{key}:{str(exact).lower()}"] += 1
                drift_max[key] = max(
                    drift_max[key], float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                )
            failed_actions, failed_pad = executed_chunk(branch, branch_index)
            success_actions, success_pad = executed_chunk(parent, parent_index)
        embedded_failed = row["executed_actions"].float().numpy().astype(np.float32)
        embedded_pad = row["action_is_pad"].bool().numpy()
        if not np.array_equal(failed_actions, embedded_failed) or not np.array_equal(
            failed_pad, embedded_pad
        ):
            raise ValueError("C63 failed action reconstruction differs from C60")
        if np.array_equal(success_actions, failed_actions) and np.array_equal(
            success_pad, failed_pad
        ):
            raise ValueError("C63 pair candidates are identical")
        pairs.append(
            {
                "pair_index": len(pairs),
                "episode_id": episode_id,
                "sample_id": int(row["sample_id"]),
                "suite": str(row["suite"]),
                "task": int(row["task"]),
                "trial": int(row["trial"]),
                "onset_step": onset,
                "branch_trajectory": str(branch_path),
                "branch_trajectory_sha256": branch_sha,
                "branch_index": branch_index,
                "parent_trajectory": str(parent_path),
                "parent_trajectory_sha256": parent_sha,
                "parent_index": parent_index,
                "failed_actions": failed_actions.tolist(),
                "failed_is_pad": failed_pad.tolist(),
                "failed_action_sha256": tensor_identity(failed_actions, failed_pad),
                "success_actions": success_actions.tolist(),
                "success_is_pad": success_pad.tolist(),
                "success_action_sha256": tensor_identity(success_actions, success_pad),
            }
        )
    suite_counts = dict(sorted(Counter(row["suite"] for row in pairs).items()))
    if suite_counts != EXPECTED_SUITE_COUNTS:
        raise ValueError(f"C63 suite counts drifted: {suite_counts}")
    parent_count = len({row["parent_trajectory"] for row in pairs})
    if parent_count != 11:
        raise ValueError(f"C63 successful-parent count drifted: {parent_count}")
    return {
        "format": FORMAT,
        "status": "PASS_C63_FIXED_32_PAIR_MECHANICAL_PREPARATION",
        "effect_status": "NOT_EVALUATED",
        "dataset_sha256": DATASET_SHA256,
        "observations_sha256": OBSERVATIONS_SHA256,
        "pair_count": len(pairs),
        "successful_parent_trajectories": parent_count,
        "suite_counts": suite_counts,
        "identity_gate": {
            "same_simulator_step": True,
            "sim_state_byte_exact": True,
            "previous_action_byte_exact": True,
            "same_branch_model_input_for_both_candidates": True,
        },
        "pre_forward_erratum_observation_drift": {
            "exact_counts": dict(sorted(drift_exact.items())),
            "max_abs_by_field": dict(sorted(drift_max.items())),
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C63 pair manifest: {output}")
    result = build_pairs(args.dataset.resolve(), args.observations.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({**{k: v for k, v in result.items() if k != "pairs"}, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
