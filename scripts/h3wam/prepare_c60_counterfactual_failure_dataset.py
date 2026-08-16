#!/usr/bin/env python3
"""Freeze C54 state-aligned failed branches as causal FACT failure data.

Every C54 branch starts from an exact state on a successful parent trajectory
and replaces the next policy chunk with a newly sampled chunk.  If that branch
fails, the branch start is an auditable intervention boundary: failed actions
are never imitation targets, but their observed futures and value are valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from fastwam.models.h3wam import libero_observation_state  # noqa: E402


FORMAT = "h3wam-c60-counterfactual-failure-dataset-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_for_source(suite: str, task: int, trial: int) -> str:
    """Keep both d3/d5 states and every action arm from one parent together."""

    identity = f"{suite}:task{task}:trial{trial}"
    digest = hashlib.blake2b(f"c60:{identity}".encode(), digest_size=8).digest()
    return "validation" if int.from_bytes(digest, "little") % 5 == 0 else "train"


def proprio(archive, prefix: str, index: int | None = None) -> torch.Tensor:
    def value(name: str):
        tensor = archive[f"{prefix}{name}"]
        return tensor if index is None else tensor[index]

    return torch.as_tensor(
        libero_observation_state(
            {
                "eef_pos": value("eef_pos"),
                "eef_quat": value("eef_quat"),
                "gripper_qpos": value("gripper_qpos"),
            }
        ),
        dtype=torch.float32,
    )


def find_result(root: Path, job: dict) -> Path:
    prefix = f"{int(job['ordinal'])}_g{int(job['group_id'])}_{job['arm']}_"
    matches = sorted((root / "runs").glob(f"{prefix}*/results.json"))
    if len(matches) != 1:
        raise ValueError(f"C60 expected one result for {prefix}, got {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c54-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    c54_root = args.c54_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C60 output: {output_root}")
    final_path = c54_root / "FINAL_REPORT.json"
    selection_path = c54_root / "selection.jsonl"
    sources_path = c54_root / "SOURCES.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    # A failed online ranker does not invalidate the raw, observed branches.
    if final.get("status") != "FAIL_C54_STATE_ALIGNED_REPLICATION":
        raise ValueError("C60 source outcome is not the frozen C54 final")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    successful_parent = {
        str(Path(row["trajectory"]).resolve()): row
        for row in sources["rows"]
        if bool(row["success"]) and bool(row["eligible"])
    }
    jobs = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observations = []
    samples = []
    episodes = []
    counts = {
        split: {"episodes": 0, "samples": 0, "groups": set(), "sources": set()}
        for split in ("train", "validation")
    }
    for job in jobs:
        result_path = find_result(c54_root, job)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        episode = result["tasks"][0]["episodes"][0]
        if bool(episode["success"]):
            continue
        branch = episode.get("branch_start") or {}
        parent_path = str(Path(job["trajectory"]).resolve())
        if parent_path not in successful_parent:
            raise ValueError(f"branch parent was not a verified success: {parent_path}")
        expected_branch = {
            "trajectory": parent_path,
            "index": int(job["index"]),
            "step": int(job["start_step"]),
        }
        actual_branch = {
            "trajectory": str(Path(branch["trajectory"]).resolve()),
            "index": int(branch["index"]),
            "step": int(branch["step"]),
        }
        if actual_branch != expected_branch:
            raise ValueError(f"branch identity mismatch: {actual_branch} != {expected_branch}")
        trajectory = Path(episode["trajectory"]).resolve()
        archive = np.load(trajectory, allow_pickle=False)
        row_count = int(archive["step"].shape[0])
        terminal_step = int(archive["terminal_step"])
        if row_count != int(episode["replans"]):
            raise ValueError(f"replan mismatch: {trajectory}")
        if archive["policy_actions"].shape != (row_count, 32, 7):
            raise ValueError(f"action shape mismatch: {trajectory}")
        if int(archive["step"][0]) != int(job["start_step"]):
            raise ValueError(f"trajectory does not start at intervention: {trajectory}")

        episode_id = len(episodes)
        source_identity = f"{job['suite']}:task{int(job['task'])}:trial{int(job['trial'])}"
        split = split_for_source(str(job["suite"]), int(job["task"]), int(job["trial"]))
        counts[split]["episodes"] += 1
        counts[split]["groups"].add(int(job["group_id"]))
        counts[split]["sources"].add(source_identity)
        observation_ids = []
        for index in range(row_count):
            observation_id = len(observations)
            observation_ids.append(observation_id)
            observations.append(
                {
                    "observation_id": observation_id,
                    "episode_id": episode_id,
                    "split": split,
                    "trajectory": str(trajectory),
                    "kind": "row",
                    "row_index": index,
                    "step": int(archive["step"][index]),
                    "task_language": result["tasks"][0]["task"],
                }
            )
        terminal_id = len(observations)
        observations.append(
            {
                "observation_id": terminal_id,
                "episode_id": episode_id,
                "split": split,
                "trajectory": str(trajectory),
                "kind": "terminal",
                "row_index": None,
                "step": terminal_step,
                "task_language": result["tasks"][0]["task"],
            }
        )
        for index in range(row_count):
            start = int(archive["step"][index])
            chunks = []
            for cursor in range(index, row_count):
                segment_end = (
                    int(archive["step"][cursor + 1])
                    if cursor + 1 < row_count
                    else terminal_step
                )
                take = max(0, min(32, segment_end - int(archive["step"][cursor])))
                chunks.append(torch.as_tensor(archive["policy_actions"][cursor, :take], dtype=torch.float32))
                if sum(len(chunk) for chunk in chunks) >= 32:
                    break
            actions = torch.cat(chunks, dim=0)[:32] if chunks else torch.empty(0, 7)
            executed_steps = len(actions)
            padded = torch.zeros(32, 7)
            padded[:executed_steps] = actions
            future_step = min(start + 32, terminal_step)
            future_positions = np.flatnonzero(archive["step"] == future_step)
            if len(future_positions):
                future_position = int(future_positions[0])
                future_id = observation_ids[future_position]
                future_state = proprio(archive, "", future_position)
            else:
                future_id = terminal_id
                future_state = proprio(archive, "terminal_")
            remaining = max(
                0.0,
                float(terminal_step - future_step) / max(float(terminal_step), 1.0),
            )
            progress = min(1.0, float(future_step) / max(float(terminal_step), 1.0))
            samples.append(
                {
                    "sample_id": len(samples),
                    "episode_id": episode_id,
                    "split": split,
                    "suite": str(job["suite"]),
                    "task": int(job["task"]),
                    "trial": int(job["trial"]),
                    "group_id": int(job["group_id"]),
                    "arm": str(job["arm"]),
                    "success": False,
                    "current_observation_id": observation_ids[index],
                    "future_observation_id": future_id,
                    "current_step": start,
                    "future_step": future_step,
                    "terminal_step": terminal_step,
                    "failure_active_from_step": int(job["start_step"]),
                    "failure_active": True,
                    "action_loss_mask": 0.0,
                    "future_loss_mask": 1.0,
                    "value_loss_mask": 1.0,
                    "current_proprio": proprio(archive, "", index),
                    "future_proprio": future_state,
                    "executed_actions": padded,
                    "action_is_pad": torch.arange(32) >= executed_steps,
                    "executed_action_steps": executed_steps,
                    "fact_code_value_raw": remaining + 1.0,
                    "fact_paper_progress_target": max(0.0, progress - 1.0),
                }
            )
            counts[split]["samples"] += 1
        episodes.append(
            {
                "episode_index": episode_id,
                "source_ordinal": int(job["ordinal"]),
                "group_id": int(job["group_id"]),
                "arm": str(job["arm"]),
                "split": split,
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256_file(trajectory),
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "successful_parent_trajectory": parent_path,
                "failure_episode": True,
                "failure_active_from_frame": 0,
                "failure_active_from_step": int(job["start_step"]),
                "annotation_source": "state_aligned_counterfactual_action_intervention",
                "evidence": "C54 exact sim-state restore plus changed first action seed; parent trajectory succeeded",
                "length": row_count,
            }
        )
        archive.close()

    if not episodes:
        raise ValueError("C60 found no failed counterfactual branches")
    train_groups = counts["train"]["groups"]
    validation_groups = counts["validation"]["groups"]
    if train_groups & validation_groups:
        raise RuntimeError("C60 group split leakage")
    if counts["train"]["sources"] & counts["validation"]["sources"]:
        raise RuntimeError("C60 parent source episode split leakage")
    output_root.mkdir(parents=True)
    observations_path = output_root / "observations.jsonl"
    episodes_path = output_root / "failure_rollouts.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in observations),
        encoding="utf-8",
    )
    episodes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes),
        encoding="utf-8",
    )
    serializable_counts = {
        split: {
            "episodes": value["episodes"],
            "samples": value["samples"],
            "groups": len(value["groups"]),
            "source_episodes": len(value["sources"]),
        }
        for split, value in counts.items()
    }
    dataset = {
        "format": FORMAT,
        "sources": {
            "c54_final_sha256": sha256_file(final_path),
            "c54_selection_sha256": sha256_file(selection_path),
            "c54_sources_sha256": sha256_file(sources_path),
        },
        "claim_boundary": (
            "Causal onset is the state-aligned action-resampling intervention, not a claim "
            "that an irreversible simulator failure is visually observable at that instant."
        ),
        "action_contract": "all branch actions masked from imitation",
        "future_contract": "actual post-intervention observations supervise future prediction",
        "value_contract": "pinned FACT code remaining-time +1 and paper Eq.6 progress -1 stored separately",
        "counts": serializable_counts,
        "observations": len(observations),
        "episodes": episodes,
        "samples": samples,
    }
    temporary = output_root / f".dataset.pt.{os.getpid()}.partial"
    torch.save(dataset, temporary)
    os.replace(temporary, output_root / "dataset.pt")
    report = {
        "format": FORMAT,
        "status": "PASS_C60_COUNTERFACTUAL_FAILURE_DATASET",
        "counts": serializable_counts,
        "observations": len(observations),
        "samples": len(samples),
        "dataset_sha256": sha256_file(output_root / "dataset.pt"),
        "observations_sha256": sha256_file(observations_path),
        "failure_rollouts_sha256": sha256_file(episodes_path),
    }
    (output_root / "COMPLETED.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
