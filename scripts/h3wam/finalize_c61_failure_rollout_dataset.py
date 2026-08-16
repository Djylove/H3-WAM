#!/usr/bin/env python3
"""Finalize completed C61 branches into C60-compatible FACT failure data.

The output deliberately keeps C60's loader format because it has the exact
failure-action/future/value contract consumed by the FACT trainer.  Provenance
records C61 as the generator.  Successful branches are audited but excluded;
failed branch actions are causal inputs only and always have imitation mask 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam.models.h3wam.deployment import libero_observation_state  # noqa: E402


C61_FORMAT = "h3wam-c61-failure-rollout-expansion-v1"
TRAINING_FORMAT = "h3wam-c60-counterfactual-failure-dataset-v1"
REPORT_FORMAT = "h3wam-c61-finalized-fact-failure-dataset-v1"
# Retain the exact annotation vocabulary enforced by the existing C60 loader;
# C61 provenance is recorded separately in ``generation`` and ``sources``.
ANNOTATION_SOURCE = "state_aligned_counterfactual_action_intervention"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_for_parent(job: dict[str, Any]) -> str:
    identity = (
        f"{int(job['source_id'])}:{int(job['episode_id'])}:"
        f"{job['suite']}:task{int(job['task'])}:trial{int(job['trial'])}"
    )
    digest = hashlib.blake2b(f"c61:{identity}".encode(), digest_size=8).digest()
    return "validation" if int.from_bytes(digest, "little") % 5 == 0 else "train"


def parent_identity(job: dict[str, Any]) -> str:
    return (
        f"source{int(job['source_id'])}:episode{int(job['episode_id'])}:"
        f"{job['suite']}:task{int(job['task'])}:trial{int(job['trial'])}"
    )


def branch_identity(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": int(job["ordinal"]),
        "source_id": int(job["source_id"]),
        "episode_id": int(job["episode_id"]),
        "group_id": int(job["group_id"]),
        "candidate": int(job["candidate"]),
        "distance_replans": int(job["distance_replans"]),
        "suite": str(job["suite"]),
        "task": int(job["task"]),
        "trial": int(job["trial"]),
        "trajectory": str(Path(job["trajectory"]).resolve()),
        "trajectory_sha256": str(job["trajectory_sha256"]),
        "index": int(job["index"]),
        "start_step": int(job["start_step"]),
        "first_policy_noise_seed": int(job["first_policy_noise_seed"]),
        "continuation_policy_noise_seed_base": int(
            job["continuation_policy_noise_seed_base"]
        ),
    }


def branch_identity_sha256(job: dict[str, Any]) -> str:
    payload = json.dumps(
        branch_identity(job), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_inventory(
    *,
    root: Path,
    c48_dataset_path: Path,
    c48_observations_path: Path,
    expected_frozen_sha256: str,
    expected_jobs_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_path, jobs_path = root / "FROZEN.json", root / "jobs.jsonl"
    if sha256_file(frozen_path) != expected_frozen_sha256:
        raise ValueError("C61 FROZEN.json SHA256 mismatch")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("format") != C61_FORMAT or frozen.get("status") != "PASS_C61_FROZEN_NOT_EXECUTED":
        raise ValueError("C61 frozen contract mismatch")
    actual_jobs_sha = sha256_file(jobs_path)
    if actual_jobs_sha != expected_jobs_sha256 or frozen.get("jobs_sha256") != actual_jobs_sha:
        raise ValueError("C61 jobs.jsonl SHA256 mismatch")
    if frozen.get("c48_dataset_sha256") != sha256_file(c48_dataset_path):
        raise ValueError("C61 C48 dataset identity mismatch")
    if frozen.get("c48_observations_sha256") != sha256_file(c48_observations_path):
        raise ValueError("C61 C48 observations identity mismatch")
    jobs = load_jsonl(jobs_path)
    if len(jobs) != int(frozen["jobs"]):
        raise ValueError("C61 job count differs from FROZEN.json")
    ordinals = [int(job["ordinal"]) for job in jobs]
    if ordinals != list(range(len(jobs))):
        raise ValueError("C61 job ordinals are not exact and contiguous")

    c48 = torch.load(c48_dataset_path, map_location="cpu", weights_only=False)
    if c48.get("format") != "h3wam-c48-fact-dense-value-dataset-v1":
        raise ValueError("C61 requires the immutable C48 training source")
    c48_by_episode: dict[int, dict[str, Any]] = {}
    for row in c48["samples"]:
        episode_id = int(row["episode_id"])
        identity = {
            "split": str(row["split"]),
            "success": bool(row["success"]),
            "suite": str(row["suite"]),
            "task": int(row["task"]),
            "trial": int(row["trial"]),
        }
        previous = c48_by_episode.setdefault(episode_id, identity)
        if previous != identity:
            raise ValueError("C48 episode rows disagree on parent identity")
    trajectory_by_episode: dict[int, str] = {}
    for row in load_jsonl(c48_observations_path):
        episode_id = int(row["episode_id"])
        trajectory = str(Path(row["trajectory"]).resolve())
        previous = trajectory_by_episode.setdefault(episode_id, trajectory)
        if previous != trajectory:
            raise ValueError("C48 episode has multiple parent trajectories")

    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    parent_hashes: dict[str, str] = {}
    for job in jobs:
        source_id = int(job["source_id"])
        episode_id = int(job["episode_id"])
        source = c48_by_episode.get(episode_id)
        expected = {
            "split": "train",
            "success": True,
            "suite": str(job["suite"]),
            "task": int(job["task"]),
            "trial": int(job["trial"]),
        }
        if source != expected:
            raise ValueError("C61 job is not an exact C48 train-success parent")
        parent_path = str(Path(job["trajectory"]).resolve())
        if trajectory_by_episode.get(episode_id) != parent_path:
            raise ValueError("C61 job trajectory differs from its C48 parent")
        if parent_path not in parent_hashes:
            parent_hashes[parent_path] = sha256_file(Path(parent_path))
        actual_parent_sha = parent_hashes[parent_path]
        if actual_parent_sha != str(job["trajectory_sha256"]):
            raise ValueError("C61 parent trajectory SHA256 mismatch")
        by_source[source_id].append(job)
        by_group[int(job["group_id"])].append(job)

    offsets = tuple(int(value) for value in frozen["offsets"])
    if len(offsets) != 4 or offsets[0] != 0:
        raise ValueError("C61 frozen candidate offsets are invalid")
    for group_id, rows in by_group.items():
        if sorted(int(row["candidate"]) for row in rows) != list(range(4)):
            raise ValueError(f"C61 group {group_id} lacks four exact candidates")
        invariant_keys = (
            "source_id", "episode_id", "group_id", "distance_replans",
            "suite", "task", "trial", "trajectory", "trajectory_sha256",
            "index", "start_step", "continuation_policy_noise_seed_base",
        )
        reference = {key: rows[0][key] for key in invariant_keys}
        if any({key: row[key] for key in invariant_keys} != reference for row in rows):
            raise ValueError(f"C61 group {group_id} branch identity is inconsistent")
        by_candidate = {int(row["candidate"]): row for row in rows}
        base_seed = int(by_candidate[0]["first_policy_noise_seed"])
        if any(
            int(by_candidate[candidate]["first_policy_noise_seed"])
            != base_seed + offsets[candidate]
            for candidate in range(4)
        ):
            raise ValueError(f"C61 group {group_id} candidate seeds differ from FROZEN")
    for source_id, rows in by_source.items():
        identities = {parent_identity(row) for row in rows}
        if len(identities) != 1:
            raise ValueError(f"C61 source_id {source_id} aliases parent episodes")
        distances = {int(row["distance_replans"]) for row in rows}
        if distances != {3, 5} or len(rows) != 8:
            raise ValueError(f"C61 source_id {source_id} lacks exact d3/d5 arms")
    if len(by_source) != int(frozen["sources"]) or len(by_group) != int(frozen["groups"]):
        raise ValueError("C61 source/group inventory differs from FROZEN")
    return frozen, jobs


def validate_completion_markers(root: Path, jobs: list[dict[str, Any]], num_nodes: int) -> dict[str, str]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    markers: dict[str, str] = {}
    for node in range(num_nodes):
        path = root / f"node{node}-of-{num_nodes}.COMPLETED"
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_jobs = sum(int(job["ordinal"]) % num_nodes == node for job in jobs)
        if payload != {
            "node": node,
            "num_nodes": num_nodes,
            "jobs": expected_jobs,
            "duration_seconds": int(payload.get("duration_seconds", -1)),
        } or int(payload["duration_seconds"]) < 0:
            raise ValueError(f"C61 completion marker mismatch: {path}")
        markers[path.name] = sha256_file(path)
    unexpected = {
        path.name for path in root.glob("node*-of-*.COMPLETED")
    } - set(markers)
    if unexpected:
        raise ValueError(f"unexpected C61 completion markers: {sorted(unexpected)}")
    return markers


def find_result(root: Path, job: dict[str, Any]) -> Path:
    prefix = f"{int(job['ordinal'])}_g{int(job['group_id'])}_c{int(job['candidate'])}_"
    matches = sorted((root / "runs").glob(f"{prefix}*/results.json"))
    if len(matches) != 1:
        raise ValueError(f"C61 expected one exact result for {prefix}, got {len(matches)}")
    return matches[0].resolve()


def proprio(archive: Any, prefix: str, index: int | None = None) -> torch.Tensor:
    def value(name: str):
        tensor = archive[f"{prefix}{name}"]
        return tensor if index is None else tensor[index]

    return libero_observation_state(
        {
            "eef_pos": value("eef_pos"),
            "eef_quat": value("eef_quat"),
            "gripper_qpos": value("gripper_qpos"),
        }
    ).float()


def validate_result(job: dict[str, Any], result_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required_top = {
        "suite": str(job["suite"]),
        "task_ids": [int(job["task"])],
        "trial_indices": [int(job["trial"])],
        "start_trajectory": str(Path(job["trajectory"]).resolve()),
        "start_index": int(job["index"]),
        "first_policy_noise_seed": int(job["first_policy_noise_seed"]),
        "continuation_policy_noise_seed_base": int(job["continuation_policy_noise_seed_base"]),
        "first_replan_steps": 32,
        "replan_steps": 8,
        "save_trajectories": True,
    }
    mismatches = [key for key, value in required_top.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"C61 result launch identity mismatch: {mismatches}")
    if len(result.get("tasks", [])) != 1 or len(result["tasks"][0].get("episodes", [])) != 1:
        raise ValueError("C61 result must contain exactly one task and branch episode")
    task, episode = result["tasks"][0], result["tasks"][0]["episodes"][0]
    if int(task["task_id"]) != int(job["task"]):
        raise ValueError("C61 result task identity mismatch")
    branch = episode.get("branch_start")
    expected_branch = {
        "trajectory": str(Path(job["trajectory"]).resolve()),
        "index": int(job["index"]),
        "step": int(job["start_step"]),
    }
    actual_branch = {
        "trajectory": str(Path(branch["trajectory"]).resolve()),
        "index": int(branch["index"]),
        "step": int(branch["step"]),
    }
    if actual_branch != expected_branch:
        raise ValueError("C61 result branch_start is not the frozen intervention onset")
    success = bool(episode["success"])
    if int(result.get("episodes", -1)) != 1 or int(result.get("successes", -1)) != int(success):
        raise ValueError("C61 result aggregate outcome disagrees with its branch")
    trajectory = Path(episode["trajectory"]).resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(f"missing C61 branch trajectory: {trajectory}")
    if not result.get("checkpoint"):
        raise ValueError("C61 result lacks frozen policy checkpoint identity")
    with np.load(trajectory, allow_pickle=False) as archive:
        if "policy_actions" not in archive.files or archive["policy_actions"].ndim != 3:
            raise ValueError("C61 branch trajectory lacks policy action identity")
        json_first_chunk = np.asarray(
            episode["first_environment_action_chunk"], dtype=np.float32
        )
        if not np.array_equal(json_first_chunk, archive["policy_actions"][0]):
            raise ValueError("C61 result/trajectory first branch action identity mismatch")
    return result, episode, trajectory


def build_failure_episode(
    *,
    job: dict[str, Any],
    result: dict[str, Any],
    episode: dict[str, Any],
    trajectory: Path,
    result_path: Path,
    episode_index: int,
    observations: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    split = split_for_parent(job)
    trajectory_sha = sha256_file(trajectory)
    with np.load(trajectory, allow_pickle=False) as archive:
        required = {
            "step", "eef_pos", "eef_quat", "gripper_qpos", "policy_actions",
            "terminal_step", "terminal_eef_pos", "terminal_eef_quat",
            "terminal_gripper_qpos",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"C61 trajectory lacks training fields: {trajectory}")
        steps = archive["step"].astype(np.int64)
        row_count, terminal_step = len(steps), int(archive["terminal_step"])
        if row_count <= 0 or row_count != int(episode["replans"]):
            raise ValueError("C61 trajectory/replan row count mismatch")
        if archive["policy_actions"].shape != (row_count, 32, 7):
            raise ValueError("C61 policy action tensor shape mismatch")
        if int(steps[0]) != int(job["start_step"]):
            raise ValueError("C61 first trajectory row is not the explicit branch onset")
        if np.any(np.diff(steps) <= 0) or terminal_step <= int(steps[-1]):
            raise ValueError("C61 branch time coordinates are not strictly causal")
        json_first_chunk = np.asarray(
            episode["first_environment_action_chunk"], dtype=np.float32
        )
        if not np.array_equal(json_first_chunk, archive["policy_actions"][0]):
            raise ValueError("C61 result/trajectory first branch action identity mismatch")

        observation_ids = []
        for row_index, step in enumerate(steps.tolist()):
            observation_id = len(observations)
            observation_ids.append(observation_id)
            observations.append(
                {
                    "observation_id": observation_id,
                    "episode_id": episode_index,
                    "split": split,
                    "trajectory": str(trajectory),
                    "trajectory_sha256": trajectory_sha,
                    "kind": "row",
                    "row_index": row_index,
                    "step": int(step),
                    "task_language": str(result["tasks"][0]["task"]),
                }
            )
        terminal_id = len(observations)
        observations.append(
            {
                "observation_id": terminal_id,
                "episode_id": episode_index,
                "split": split,
                "trajectory": str(trajectory),
                "trajectory_sha256": trajectory_sha,
                "kind": "terminal",
                "row_index": None,
                "step": terminal_step,
                "task_language": str(result["tasks"][0]["task"]),
            }
        )
        step_to_row = {int(step): index for index, step in enumerate(steps)}
        for row_index, start in enumerate(steps.tolist()):
            chunks = []
            for cursor in range(row_index, row_count):
                segment_end = int(steps[cursor + 1]) if cursor + 1 < row_count else terminal_step
                take = max(0, min(32, segment_end - int(steps[cursor])))
                chunks.append(torch.as_tensor(archive["policy_actions"][cursor, :take]).float())
                if sum(len(chunk) for chunk in chunks) >= 32:
                    break
            actions = torch.cat(chunks, dim=0)[:32]
            executed_steps = len(actions)
            if executed_steps <= 0:
                raise ValueError("C61 failure sample has no actually executed actions")
            padded = torch.zeros(32, 7)
            padded[:executed_steps] = actions
            future_step = min(int(start) + 32, terminal_step)
            if future_step == terminal_step:
                future_id, future_state = terminal_id, proprio(archive, "terminal_")
            elif future_step in step_to_row:
                future_row = step_to_row[future_step]
                future_id = observation_ids[future_row]
                future_state = proprio(archive, "", future_row)
            else:
                raise ValueError("C61 lacks the exact observed future required for supervision")
            progress = min(1.0, float(future_step) / max(float(terminal_step), 1.0))
            remaining = max(0.0, float(terminal_step - future_step) / max(float(terminal_step), 1.0))
            samples.append(
                {
                    "sample_id": len(samples),
                    "episode_id": episode_index,
                    "split": split,
                    "suite": str(job["suite"]),
                    "task": int(job["task"]),
                    "trial": int(job["trial"]),
                    "source_id": int(job["source_id"]),
                    "parent_episode_id": int(job["episode_id"]),
                    "group_id": int(job["group_id"]),
                    "candidate": int(job["candidate"]),
                    "success": False,
                    "current_observation_id": observation_ids[row_index],
                    "future_observation_id": future_id,
                    "current_step": int(start),
                    "future_step": future_step,
                    "terminal_step": terminal_step,
                    "failure_active_from_step": int(job["start_step"]),
                    "failure_active": True,
                    "action_loss_mask": 0.0,
                    "future_loss_mask": 1.0,
                    "value_loss_mask": 1.0,
                    "current_proprio": proprio(archive, "", row_index),
                    "future_proprio": future_state,
                    "executed_actions": padded,
                    "action_is_pad": torch.arange(32) >= executed_steps,
                    "executed_action_steps": executed_steps,
                    "fact_code_value_raw": remaining + 1.0,
                    "fact_paper_progress_target": max(0.0, progress - 1.0),
                    "supervision_source": "observed_post_intervention_branch_future_and_terminal",
                }
            )
    return {
        "episode_index": episode_index,
        # C60's existing loader uses this field for its disjointness check.
        # It is intentionally the parent source_id, not the branch ordinal.
        "source_ordinal": int(job["source_id"]),
        "source_id": int(job["source_id"]),
        "parent_episode_id": int(job["episode_id"]),
        "parent_identity": parent_identity(job),
        "group_id": int(job["group_id"]),
        "candidate": int(job["candidate"]),
        "split": split,
        "trajectory": str(trajectory),
        "trajectory_sha256": trajectory_sha,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "successful_parent_trajectory": str(Path(job["trajectory"]).resolve()),
        "successful_parent_trajectory_sha256": str(job["trajectory_sha256"]),
        "branch_identity": branch_identity(job),
        "branch_identity_sha256": branch_identity_sha256(job),
        "failure_episode": True,
        "failure_active_from_frame": 0,
        "failure_active_from_step": int(job["start_step"]),
        "annotation_source": ANNOTATION_SOURCE,
        "evidence": "frozen exact sim-state restore plus candidate-specific first action seed; C48 parent succeeded and this observed branch failed",
        "length": int(episode["replans"]),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root, output_root = args.c61_root.resolve(), args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C61 finalized data: {output_root}")
    frozen, jobs = validate_frozen_inventory(
        root=root,
        c48_dataset_path=args.c48_dataset.resolve(),
        c48_observations_path=args.c48_observations.resolve(),
        expected_frozen_sha256=args.expected_frozen_sha256,
        expected_jobs_sha256=args.expected_jobs_sha256,
    )
    marker_hashes = validate_completion_markers(root, jobs, args.num_nodes)
    expected_results = {find_result(root, job) for job in jobs}
    actual_results = {path.resolve() for path in (root / "runs").glob("*/results.json")}
    if actual_results != expected_results:
        raise ValueError("C61 result coverage is not exactly the frozen job inventory")

    observations: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    successful_jobs = 0
    policy_checkpoints: set[str] = set()
    counts = {
        split: {"episodes": 0, "samples": 0, "groups": set(), "sources": set(), "parents": set()}
        for split in ("train", "validation")
    }
    for job in jobs:
        result_path = find_result(root, job)
        result, episode, trajectory = validate_result(job, result_path)
        policy_checkpoints.add(str(result.get("checkpoint")))
        if bool(episode["success"]):
            successful_jobs += 1
            continue
        episode_index = len(episodes)
        before = len(samples)
        episode_row = build_failure_episode(
            job=job,
            result=result,
            episode=episode,
            trajectory=trajectory,
            result_path=result_path,
            episode_index=episode_index,
            observations=observations,
            samples=samples,
        )
        episodes.append(episode_row)
        split = episode_row["split"]
        counts[split]["episodes"] += 1
        counts[split]["samples"] += len(samples) - before
        counts[split]["groups"].add(int(job["group_id"]))
        counts[split]["sources"].add(int(job["source_id"]))
        counts[split]["parents"].add(parent_identity(job))
    if not episodes or successful_jobs + len(episodes) != len(jobs):
        raise ValueError("C61 finalizer did not classify every frozen job exactly once")
    if len(policy_checkpoints) != 1:
        raise ValueError("C61 jobs were collected from mixed policy checkpoints")
    if counts["train"]["episodes"] == 0 or counts["validation"]["episodes"] == 0:
        raise ValueError("C61 retained failures do not populate both frozen splits")
    for field in ("sources", "parents"):
        if counts["train"][field] & counts["validation"][field]:
            raise RuntimeError(f"C61 {field} split leakage")
    if any(
        float(row["action_loss_mask"]) != 0.0
        or float(row["future_loss_mask"]) != 1.0
        or float(row["value_loss_mask"]) != 1.0
        or int(row["current_step"]) < int(row["failure_active_from_step"])
        for row in samples
    ):
        raise RuntimeError("C61 finalized loss/onset contract failed")

    serializable_counts = {
        split: {
            "episodes": int(value["episodes"]),
            "samples": int(value["samples"]),
            "groups": len(value["groups"]),
            "source_ids": len(value["sources"]),
            "parent_episodes": len(value["parents"]),
        }
        for split, value in counts.items()
    }
    output_root.mkdir(parents=True)
    observations_path = output_root / "observations.jsonl"
    episodes_path = output_root / "failure_rollouts.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in observations), encoding="utf-8"
    )
    episodes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes), encoding="utf-8"
    )
    dataset = {
        "format": TRAINING_FORMAT,
        "generation": REPORT_FORMAT,
        "sources": {
            "c61_frozen_sha256": args.expected_frozen_sha256,
            "c61_jobs_sha256": args.expected_jobs_sha256,
            "c48_dataset_sha256": frozen["c48_dataset_sha256"],
            "c48_observations_sha256": frozen["c48_observations_sha256"],
            "completion_markers": marker_hashes,
            "policy_checkpoint": next(iter(policy_checkpoints)),
        },
        "claim_boundary": "Explicit onset is the frozen exact-state branch action intervention, not an inferred visual failure onset.",
        "action_contract": "all branch actions masked from imitation",
        "future_contract": "actual post-intervention observations supervise future prediction",
        "value_contract": "pinned FACT code remaining-time +1 and paper Eq.6 progress -1 stored separately",
        "split_contract": "source_id and successful C48 parent episode are disjoint across train/validation",
        "counts": serializable_counts,
        "collected_jobs": len(jobs),
        "excluded_successful_jobs": successful_jobs,
        "retained_failed_jobs": len(episodes),
        "observations": len(observations),
        "episodes": episodes,
        "samples": samples,
    }
    dataset_path = output_root / "dataset.pt"
    temporary = output_root / f".dataset.pt.{os.getpid()}.partial"
    torch.save(dataset, temporary)
    os.replace(temporary, dataset_path)
    report = {
        "format": REPORT_FORMAT,
        "status": "PASS_C61_FINALIZED_FACT_FAILURE_DATASET",
        "counts": serializable_counts,
        "collected_jobs": len(jobs),
        "excluded_successful_jobs": successful_jobs,
        "retained_failed_jobs": len(episodes),
        "observations": len(observations),
        "samples": len(samples),
        "dataset_sha256": sha256_file(dataset_path),
        "observations_sha256": sha256_file(observations_path),
        "failure_rollouts_sha256": sha256_file(episodes_path),
        "frozen_sha256": args.expected_frozen_sha256,
        "jobs_sha256": args.expected_jobs_sha256,
        "completion_marker_sha256": marker_hashes,
        "gates": {
            "exact_frozen_job_coverage": "PASS",
            "failed_branches_only": "PASS",
            "source_and_parent_split_disjoint": "PASS",
            "explicit_branch_onset": "PASS",
            "action_imitation_mask_zero": "PASS",
            "observed_future_and_terminal_value": "PASS",
            "exact_branch_identity": "PASS",
        },
    }
    completed = output_root / "COMPLETED.json"
    completed.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def audit_inventory(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only audit for an in-progress collection; never writes artifacts."""

    root = args.c61_root.resolve()
    frozen, jobs = validate_frozen_inventory(
        root=root,
        c48_dataset_path=args.c48_dataset.resolve(),
        c48_observations_path=args.c48_observations.resolve(),
        expected_frozen_sha256=args.expected_frozen_sha256,
        expected_jobs_sha256=args.expected_jobs_sha256,
    )
    jobs_by_ordinal = {int(job["ordinal"]): job for job in jobs}
    observations: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    successes = failures = 0
    paths = sorted((root / "runs").glob("*/results.json"))
    seen_ordinals: set[int] = set()
    for path in paths:
        try:
            ordinal = int(path.parent.name.split("_", 1)[0])
        except ValueError as error:
            raise ValueError(f"C61 result directory lacks frozen ordinal: {path}") from error
        if ordinal in seen_ordinals or ordinal not in jobs_by_ordinal:
            raise ValueError(f"C61 partial result has duplicate/unknown ordinal: {ordinal}")
        seen_ordinals.add(ordinal)
        job = jobs_by_ordinal[ordinal]
        if find_result(root, job) != path.resolve():
            raise ValueError("C61 partial result path is not its exact frozen branch")
        result, episode, trajectory = validate_result(job, path.resolve())
        if bool(episode["success"]):
            successes += 1
        else:
            build_failure_episode(
                job=job,
                result=result,
                episode=episode,
                trajectory=trajectory,
                result_path=path.resolve(),
                episode_index=failures,
                observations=observations,
                samples=samples,
            )
            failures += 1
    return {
        "format": REPORT_FORMAT,
        "status": "PASS_C61_READ_ONLY_INVENTORY_AUDIT",
        "read_only": True,
        "complete": len(paths) == len(jobs),
        "frozen_jobs": len(jobs),
        "audited_results": len(paths),
        "missing_results": len(jobs) - len(paths),
        "successful_results": successes,
        "failed_results": failures,
        "potential_failure_samples": len(samples),
        "frozen_sources": int(frozen["sources"]),
        "frozen_groups": int(frozen["groups"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c61-root", type=Path, required=True)
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--c48-observations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-frozen-sha256", required=True)
    parser.add_argument("--expected-jobs-sha256", required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument(
        "--audit-inventory",
        action="store_true",
        help="Read-only audit of currently completed jobs; never writes dataset/COMPLETED.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_inventory:
        report = audit_inventory(args)
    else:
        if args.output_root is None:
            raise ValueError("formal finalization requires --output-root")
        report = finalize(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
