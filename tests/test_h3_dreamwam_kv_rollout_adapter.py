import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SERVE = load_script(
    "serve_rollout_policy_dreamwam_kv_test",
    "scripts/h3wam/serve_rollout_policy.py",
)
ROLLOUT = load_script(
    "rollout_libero_dreamwam_kv_test",
    "scripts/h3wam/rollout_libero.py",
)


class H3DreamWAMKVRolloutAdapterTest(unittest.TestCase):
    def test_first_action_noise_intervention_fixes_continuation_schedule(self):
        resolve = ROLLOUT.resolve_replan_noise_seed
        common = {
            "episode_seed": 42,
            "fixed_replan_noise": False,
            "fixed_noise_seed": None,
            "continuation_policy_noise_seed_base": 9_000_000,
        }
        self.assertEqual(
            resolve(**common, replans=0, first_policy_noise_seed=101), 101
        )
        self.assertEqual(
            resolve(**common, replans=0, first_policy_noise_seed=202), 202
        )
        for first_seed in (101, 202):
            self.assertEqual(
                resolve(**common, replans=1, first_policy_noise_seed=first_seed),
                9_000_000,
            )
            self.assertEqual(
                resolve(**common, replans=4, first_policy_noise_seed=first_seed),
                9_000_003,
            )

    def test_branch_start_loads_only_state_step_and_previous_action(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(
                path,
                step=np.asarray([0, 8], dtype=np.int64),
                sim_state=np.arange(10, dtype=np.float64).reshape(2, 5),
                previous_action=np.arange(14, dtype=np.float32).reshape(2, 7),
                agentview_image=np.full((2, 2, 2, 3), 255, dtype=np.uint8),
            )
            branch = ROLLOUT.load_branch_start(path, 1)
        self.assertEqual(branch["index"], 1)
        self.assertEqual(branch["step"], 8)
        np.testing.assert_array_equal(branch["sim_state"], np.arange(5, 10))
        np.testing.assert_array_equal(branch["previous_action"], np.arange(7, 14))
        self.assertNotIn("agentview_image", branch)

    def test_branch_start_rejects_out_of_range_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(
                path,
                step=np.asarray([0]),
                sim_state=np.zeros((1, 5)),
                previous_action=np.zeros((1, 7)),
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                ROLLOUT.load_branch_start(path, 1)

    def test_online_adapter_accepts_paired_d_and_d0_source_modes(self):
        resolve = SERVE.H3DreamWAMKVInt8Policy._resolve_candidate_source_mode
        self.assertEqual(
            resolve({"candidate": "D", "carrier_source_mode": "aligned_5layer"}),
            ("D", "aligned_5layer"),
        )
        self.assertEqual(
            resolve({"candidate": "D0", "carrier_source_mode": "repeat_layer49"}),
            ("D0", "repeat_layer49"),
        )

    def test_online_adapter_rejects_crossed_or_unknown_source_modes(self):
        resolve = SERVE.H3DreamWAMKVInt8Policy._resolve_candidate_source_mode
        with self.assertRaisesRegex(ValueError, "source mode mismatch"):
            resolve({"candidate": "D", "carrier_source_mode": "repeat_layer49"})
        with self.assertRaisesRegex(ValueError, "only supports paired"):
            resolve({"candidate": "D1", "carrier_source_mode": "aligned_5layer"})

    def test_server_prefers_project_vendored_starwam_when_present(self):
        expected = ROOT / "third_party" / "StarWAM"
        self.assertTrue(expected.is_dir())
        self.assertIn(str(expected), sys.path)

    def test_server_parser_exposes_d0_policy_and_manifest(self):
        argv = [
            "serve",
            "--policy",
            "h3_dreamwam_kv_int8",
            "--checkpoint",
            "d0.pt",
            "--cache-root",
            "cache",
            "--port",
            "1234",
            "--ready-file",
            "ready.json",
            "--h3-checkpoint",
            "h3.safetensors",
            "--h3-model",
            "h3-model",
            "--dreamwam-source-manifest",
            "source.jsonl",
            "--model-evaluations",
            "10",
        ]
        with patch.object(sys, "argv", argv):
            args = SERVE.parse_args()
        self.assertEqual(args.policy, "h3_dreamwam_kv_int8")
        self.assertEqual(args.dreamwam_source_manifest, Path("source.jsonl"))

    def test_rollout_routes_d0_to_dedicated_server(self):
        argv = [
            "rollout",
            "--policy",
            "h3_dreamwam_kv_int8",
            "--checkpoint",
            "d0.pt",
            "--cache-root",
            "cache",
            "--policy-python",
            "python",
            "--output-dir",
            "out",
            "--h3-checkpoint",
            "h3.safetensors",
            "--h3-model",
            "h3-model",
            "--dreamwam-source-manifest",
            "source.jsonl",
            "--model-evaluations",
            "10",
        ]
        with patch.object(sys, "argv", argv):
            args = ROLLOUT.parse_args()
        command = ROLLOUT.policy_command(args, 1234, Path("ready.json"))
        self.assertIn("h3_dreamwam_kv_int8", command)
        manifest_index = command.index("--dreamwam-source-manifest")
        self.assertTrue(command[manifest_index + 1].endswith("source.jsonl"))

    def test_rollout_rejects_missing_d0_manifest(self):
        argv = [
            "rollout",
            "--policy",
            "h3_dreamwam_kv_int8",
            "--checkpoint",
            "d0.pt",
            "--cache-root",
            "cache",
            "--policy-python",
            "python",
            "--output-dir",
            "out",
            "--h3-checkpoint",
            "h3.safetensors",
            "--h3-model",
            "h3-model",
        ]
        with patch.object(sys, "argv", argv):
            args = ROLLOUT.parse_args()
        with self.assertRaisesRegex(ValueError, "dreamwam-source-manifest"):
            ROLLOUT.policy_command(args, 1234, Path("ready.json"))

    def test_d0_task_context_mapping_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"task": "pick", "context_id": "pick-v1"}) + "\n",
                encoding="utf-8",
            )
            policy = SERVE.H3DreamWAMKVInt8Policy.__new__(
                SERVE.H3DreamWAMKVInt8Policy
            )
            policy.source_manifest = manifest
            self.assertEqual(policy._load_task_context_ids(), {"pick": "pick-v1"})

            manifest.write_text(
                "\n".join(
                    (
                        json.dumps({"task": "pick", "context_id": "pick-v1"}),
                        json.dumps({"task": "pick", "context_id": "pick-v2"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ambiguous context IDs"):
                policy._load_task_context_ids()


if __name__ == "__main__":
    unittest.main()
