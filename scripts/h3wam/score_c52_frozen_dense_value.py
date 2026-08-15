#!/usr/bin/env python3
"""Score C52 action alternatives once with the frozen C51 dense value model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fastwam.models.h3wam import DenseTemporalFutureValueModel, libero_observation_state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_pairwise_permutation_p(group_rows: list[dict], observed: int) -> float:
    """Exact stratified randomization tail for integer pairwise-correct counts."""
    distribution = {0: 1.0}
    for group in group_rows:
        labels = tuple(int(candidate["success"]) for candidate in group["candidates"])
        successes = sum(labels)
        failures = 4 - successes
        if not successes or not failures:
            continue
        scores = tuple(float(candidate["predicted_value"]) for candidate in group["candidates"])
        local = Counter()
        for permutation in itertools.permutations(range(4)):
            permuted = tuple(scores[index] for index in permutation)
            correct = sum(
                permuted[positive] < permuted[negative]
                for positive in range(4) if labels[positive]
                for negative in range(4) if not labels[negative]
            )
            local[int(correct)] += 1
        next_distribution: dict[int, float] = {}
        for previous_count, previous_probability in distribution.items():
            for local_count, frequency in local.items():
                total = previous_count + local_count
                next_distribution[total] = next_distribution.get(total, 0.0) + previous_probability * frequency / math.factorial(4)
        distribution = next_distribution
    return float(sum(probability for count, probability in distribution.items() if count >= observed))


def proprio(archive: np.lib.npyio.NpzFile, index: int) -> torch.Tensor:
    return torch.as_tensor(libero_observation_state({
        "eef_pos": archive["eef_pos"][index],
        "eef_quat": archive["eef_quat"][index],
        "gripper_qpos": archive["gripper_qpos"][index],
    }), dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    outcomes = json.loads(args.outcomes.read_text())
    if outcomes.get("ranking_permission") != "GO_C52_FROZEN_VALUE_SCORING":
        raise ValueError("C52 outcome gate did not permit scoring")
    if sha256_file(args.checkpoint) != outcomes["selected_checkpoint_sha256"]:
        raise ValueError("frozen checkpoint identity mismatch")

    observation_ids: dict[tuple[str, int], int] = {}
    for line in args.observations.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "row":
            key = (str(Path(row["trajectory"]).resolve()), int(row["row_index"]))
            if key in observation_ids:
                raise ValueError(f"duplicate C49 observation key: {key}")
            observation_ids[key] = int(row["observation_id"])
    feature_payload = torch.load(args.features, map_location="cpu", weights_only=False)
    if feature_payload.get("format") != "h3wam-c49-dense-value-projected-features-v1":
        raise ValueError("C49 projected feature identity mismatch")
    projected = feature_payload["fact_layer49_projected"].float()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = DenseTemporalFutureValueModel(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    state_mean = checkpoint["normalization"]["state_mean"].to(device)
    state_std = checkpoint["normalization"]["state_std"].to(device)

    scored_groups = []
    action_hashes = set()
    for group in outcomes["group_reports"]:
        candidates = []
        source_trajectory = str(Path(group["candidates"][0]["trajectory"]).resolve())
        source_index = int(group["candidates"][0]["index"])
        observation_id = observation_ids[(source_trajectory, source_index)]
        with np.load(source_trajectory, allow_pickle=False) as archive:
            state = proprio(archive, source_index)
        actions = []
        for candidate in group["candidates"]:
            branch_files = tuple(Path(candidate["run_directory"]).glob("*trajectory.npz"))
            if len(branch_files) != 1:
                raise ValueError(f"branch trajectory mismatch: {candidate['run_directory']}")
            with np.load(branch_files[0], allow_pickle=False) as branch:
                action = torch.from_numpy(np.asarray(branch["policy_actions"][0], dtype=np.float32).copy())
            action_hashes.add(hashlib.sha256(action.numpy().tobytes()).hexdigest())
            actions.append(action)
        batch = len(actions)
        state_batch = ((state.to(device) - state_mean) / state_std).unsqueeze(0).expand(batch, -1)
        feature_batch = projected[observation_id].to(device).unsqueeze(0).expand(batch, -1)
        action_batch = torch.stack(actions).to(device)
        with torch.inference_mode():
            values = model.forward_projected(state_batch, feature_batch, action_batch)["value"].cpu()
        for source, value in zip(group["candidates"], values.tolist(), strict=True):
            candidates.append({**source, "predicted_value": float(value)})
        scored_groups.append({
            **{key: group[key] for key in ("group_id", "suite", "source_episode", "successes", "mixed_outcomes")},
            "observation_id": observation_id,
            "score_range": max(values).item() - min(values).item(),
            "selected_ordinal": candidates[int(torch.argmin(values))]["ordinal"],
            "selected_success": candidates[int(torch.argmin(values))]["success"],
            "candidate0_success": candidates[0]["success"],
            "candidates": candidates,
        })

    mixed = [group for group in scored_groups if group["mixed_outcomes"]]
    correct = total_pairs = 0
    for group in mixed:
        successful = [row for row in group["candidates"] if row["success"]]
        failed = [row for row in group["candidates"] if not row["success"]]
        total_pairs += len(successful) * len(failed)
        correct += sum(
            positive["predicted_value"] < negative["predicted_value"]
            for positive in successful for negative in failed
        )
    pairwise_accuracy = correct / total_pairs
    top1_success = sum(group["selected_success"] for group in mixed) / len(mixed)
    candidate0_success = sum(group["candidate0_success"] for group in mixed) / len(mixed)
    random_top1_expectation = sum(group["successes"] / 4 for group in mixed) / len(mixed)
    permutation_p = exact_pairwise_permutation_p(mixed, correct)
    thresholds = outcomes["ranking_gate_frozen_before_outcomes"]
    gate = {
        "pairwise_success_failure_accuracy_at_least_0_60": pairwise_accuracy >= thresholds["pairwise_success_failure_accuracy_at_least"],
        "mixed_group_top1_success_at_least_0_60": top1_success >= thresholds["mixed_group_top1_success_at_least"],
        "one_sided_exact_permutation_p_at_most_0_05": permutation_p <= thresholds["one_sided_within_group_permutation_p_at_most"],
        "all_group_score_ranges_greater_than_1e_6": all(group["score_range"] > thresholds["all_group_score_ranges_greater_than"] for group in scored_groups),
        "strict_checkpoint_restore": True,
    }
    gate["passed"] = all(gate.values())
    report = {
        "format": "h3wam-c52-frozen-dense-value-ranking-result-v1",
        "status": "PASS_C52_FROZEN_DENSE_VALUE_RANKING" if gate["passed"] else "FAIL_C52_FROZEN_DENSE_VALUE_RANKING",
        "permission": outcomes["pass_permission"] if gate["passed"] else outcomes["fail_permission"],
        "claim_boundary": outcomes["claim_boundary"],
        "mixed_groups": len(mixed), "mixed_suites": sorted({group["suite"] for group in mixed}),
        "success_failure_pairs": total_pairs, "correctly_ranked_pairs": correct,
        "pairwise_success_failure_accuracy": pairwise_accuracy,
        "mixed_group_top1_success": top1_success,
        "candidate0_success_on_mixed_groups": candidate0_success,
        "random_top1_expectation": random_top1_expectation,
        "one_sided_exact_within_group_permutation_p": permutation_p,
        "minimum_group_score_range": min(group["score_range"] for group in scored_groups),
        "unique_candidate_action_hashes": len(action_hashes),
        "gate": gate, "group_reports": scored_groups,
        "sources": {
            "outcomes_sha256": sha256_file(args.outcomes),
            "features_sha256": sha256_file(args.features),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
    }
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in report if key not in {"group_reports"}}, indent=2))


if __name__ == "__main__":
    main()
