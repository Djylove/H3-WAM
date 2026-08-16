from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fastwam.models.h3wam.fact_backbone_port import C60CausalFailureLabels


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/finalize_c61_failure_rollout_dataset.py"
SPEC = importlib.util.spec_from_file_location("_c61_finalizer_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizeC61FailureRolloutDatasetTest(unittest.TestCase):
    def build_fixture(self, directory: Path) -> argparse.Namespace:
        root, output = directory / "c61", directory / "final"
        (root / "runs").mkdir(parents=True)
        source_specs = []
        wanted = {"train", "validation"}
        candidate = 0
        while wanted:
            probe = {
                "source_id": candidate,
                "episode_id": 100 + candidate,
                "suite": "libero_goal",
                "task": candidate,
                "trial": 12,
            }
            split = MODULE.split_for_parent(probe)
            if split in wanted:
                source_specs.append((probe, split))
                wanted.remove(split)
            candidate += 1

        c48_samples, c48_observations, jobs = [], [], []
        group_id = 0
        for probe, _ in source_specs:
            source_id, episode_id = probe["source_id"], probe["episode_id"]
            parent = directory / f"parent_{source_id}.npz"
            steps = np.arange(5, dtype=np.int64) * 8
            self.write_trajectory(parent, steps=steps, terminal_step=40, action_value=0.0)
            c48_samples.append(
                {
                    "sample_id": len(c48_samples),
                    "episode_id": episode_id,
                    "split": "train",
                    "suite": probe["suite"],
                    "task": probe["task"],
                    "trial": probe["trial"],
                    "success": True,
                }
            )
            c48_observations.append(
                {
                    "observation_id": len(c48_observations),
                    "episode_id": episode_id,
                    "split": "train",
                    "trajectory": str(parent.resolve()),
                    "kind": "row",
                    "row_index": 0,
                    "step": 0,
                    "task_language": f"task {probe['task']}",
                }
            )
            for distance in (3, 5):
                index = len(steps) - distance
                base_seed = 61_000_000 + group_id * 100
                continuation = 361_000_000 + group_id * 10_000
                for arm, offset in enumerate((0, 1_000_000, 2_000_000, 3_000_000)):
                    jobs.append(
                        {
                            **probe,
                            "state_count": 5,
                            "trajectory": str(parent.resolve()),
                            "trajectory_sha256": sha(parent),
                            "group_id": group_id,
                            "distance_replans": distance,
                            "index": index,
                            "start_step": int(steps[index]),
                            "continuation_policy_noise_seed_base": continuation,
                            "ordinal": len(jobs),
                            "candidate": arm,
                            "first_policy_noise_seed": base_seed + offset,
                        }
                    )
                group_id += 1

        c48_path, observations_path = directory / "c48.pt", directory / "c48.jsonl"
        torch.save(
            {"format": "h3wam-c48-fact-dense-value-dataset-v1", "samples": c48_samples},
            c48_path,
        )
        observations_path.write_text(
            "".join(json.dumps(row) + "\n" for row in c48_observations), encoding="utf-8"
        )
        jobs_path = root / "jobs.jsonl"
        jobs_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8"
        )
        frozen = {
            "format": "h3wam-c61-failure-rollout-expansion-v1",
            "status": "PASS_C61_FROZEN_NOT_EXECUTED",
            "sources": len(source_specs),
            "groups": len(source_specs) * 2,
            "jobs": len(jobs),
            "offsets": [0, 1_000_000, 2_000_000, 3_000_000],
            "c48_dataset_sha256": sha(c48_path),
            "c48_observations_sha256": sha(observations_path),
            "jobs_sha256": sha(jobs_path),
        }
        frozen_path = root / "FROZEN.json"
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        for job in jobs:
            run = root / "runs" / (
                f"{job['ordinal']}_g{job['group_id']}_c{job['candidate']}_"
                f"goal_task{job['task']}_trial{job['trial']}"
            )
            run.mkdir()
            branch = run / "branch_trajectory.npz"
            action_value = float(job["ordinal"] + 1) / 100.0
            self.write_trajectory(
                branch,
                steps=np.array([job["start_step"]], dtype=np.int64),
                terminal_step=int(job["start_step"]) + 8,
                action_value=action_value,
            )
            success = int(job["candidate"]) == 3
            action_chunk = np.full((32, 7), action_value, dtype=np.float32)
            result = {
                "policy": "h3_dreamwam_kv_int8",
                "checkpoint": "/frozen/d0_h32_s14000.pt",
                "suite": job["suite"],
                "task_ids": [job["task"]],
                "trial_indices": [job["trial"]],
                "start_trajectory": job["trajectory"],
                "start_index": job["index"],
                "first_policy_noise_seed": job["first_policy_noise_seed"],
                "continuation_policy_noise_seed_base": job["continuation_policy_noise_seed_base"],
                "first_replan_steps": 32,
                "replan_steps": 8,
                "save_trajectories": True,
                "episodes": 1,
                "successes": int(success),
                "tasks": [
                    {
                        "task_id": job["task"],
                        "task": f"task {job['task']}",
                        "episodes": [
                            {
                                "success": success,
                                "replans": 1,
                                "trajectory": str(branch.resolve()),
                                "branch_start": {
                                    "trajectory": job["trajectory"],
                                    "index": job["index"],
                                    "step": job["start_step"],
                                },
                                "first_environment_action_chunk": action_chunk.tolist(),
                            }
                        ],
                    }
                ],
            }
            (run / "results.json").write_text(json.dumps(result), encoding="utf-8")
        (root / "node0-of-1.COMPLETED").write_text(
            json.dumps({"node": 0, "num_nodes": 1, "jobs": len(jobs), "duration_seconds": 10}) + "\n",
            encoding="utf-8",
        )
        return argparse.Namespace(
            c61_root=root,
            c48_dataset=c48_path,
            c48_observations=observations_path,
            output_root=output,
            expected_frozen_sha256=sha(frozen_path),
            expected_jobs_sha256=sha(jobs_path),
            num_nodes=1,
        )

    @staticmethod
    def write_trajectory(path: Path, *, steps: np.ndarray, terminal_step: int, action_value: float) -> None:
        count = len(steps)
        actions = np.full((count, 32, 7), action_value, dtype=np.float32)
        kwargs = {
            "step": steps,
            "eef_pos": np.zeros((count, 3), dtype=np.float32),
            "eef_quat": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (count, 1)),
            "gripper_qpos": np.zeros((count, 2), dtype=np.float32),
            "policy_actions": actions,
            "terminal_step": np.array(terminal_step, dtype=np.int64),
            "terminal_eef_pos": np.ones(3, dtype=np.float32),
            "terminal_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "terminal_gripper_qpos": np.ones(2, dtype=np.float32),
        }
        np.savez_compressed(path, **kwargs)

    def test_finalizer_is_failure_only_disjoint_and_c60_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            report = MODULE.finalize(args)
            self.assertEqual(report["status"], "PASS_C61_FINALIZED_FACT_FAILURE_DATASET")
            self.assertEqual(report["collected_jobs"], 16)
            self.assertEqual(report["excluded_successful_jobs"], 4)
            self.assertEqual(report["retained_failed_jobs"], 12)
            payload = torch.load(args.output_root / "dataset.pt", weights_only=False)
            self.assertTrue(all(not row["success"] for row in payload["samples"]))
            self.assertTrue(all(row["action_loss_mask"] == 0 for row in payload["samples"]))
            self.assertTrue(all(row["future_loss_mask"] == 1 for row in payload["samples"]))
            self.assertTrue(all(row["value_loss_mask"] == 1 for row in payload["samples"]))
            train_sources = {
                row["source_id"] for row in payload["episodes"] if row["split"] == "train"
            }
            validation_sources = {
                row["source_id"] for row in payload["episodes"] if row["split"] == "validation"
            }
            self.assertFalse(train_sources & validation_sources)
            labels = C60CausalFailureLabels(
                args.output_root / "dataset.pt",
                expected_sha256=report["dataset_sha256"],
                value_contract="fact_code_remaining_plus_penalty",
            )
            self.assertGreater(len(labels.split("train")), 0)
            self.assertGreater(len(labels.split("validation")), 0)

    def test_finalizer_rejects_nonexact_branch_action_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = next((args.c61_root / "runs").glob("*/results.json"))
            result = json.loads(path.read_text())
            result["tasks"][0]["episodes"][0]["first_environment_action_chunk"][0][0] += 1.0
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action identity"):
                MODULE.finalize(args)

    def test_finalizer_rejects_frozen_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            args.expected_frozen_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "FROZEN"):
                MODULE.finalize(args)

    def test_incomplete_collection_only_allows_read_only_inventory_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            next((args.c61_root / "runs").glob("*/results.json")).unlink()
            (args.c61_root / "node0-of-1.COMPLETED").unlink()
            with self.assertRaises(FileNotFoundError):
                MODULE.finalize(args)
            report = MODULE.audit_inventory(args)
            self.assertTrue(report["read_only"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["audited_results"], 15)
            self.assertEqual(report["missing_results"], 1)
            self.assertFalse(args.output_root.exists())


if __name__ == "__main__":
    unittest.main()
