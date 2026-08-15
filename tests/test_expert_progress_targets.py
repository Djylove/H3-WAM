import json
import tempfile
import unittest
from pathlib import Path

from scripts.h3wam import prepare_expert_progress_targets as targets


class ExpertProgressTargetsTest(unittest.TestCase):
    def test_targets_and_episode_disjointness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            base = {"context_id": "c", "dataset_root": "/data", "length": 100,
                    "suite": "libero_10", "task": "task"}
            train.write_text(json.dumps({**base, "id": "a", "episode": 1, "start": 10, "split": "train"}) + "\n")
            val.write_text(json.dumps({**base, "id": "b", "episode": 2, "start": 80, "split": "validation"}) + "\n")
            train_value, train_report = targets.convert(train, expected_split="train", future_offset=32)
            val_value, val_report = targets.convert(val, expected_split="validation", future_offset=32)
            self.assertEqual(json.loads(train_value)["future_state_index"], 42)
            self.assertAlmostEqual(json.loads(train_value)["value_raw"], 57 / 99)
            self.assertEqual(json.loads(val_value)["future_state_index"], 99)
            self.assertEqual(json.loads(val_value)["value_raw"], 0.0)
            self.assertFalse(train_report["episode_ids"] & val_report["episode_ids"])

    def test_main_writes_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, val = root / "train.jsonl", root / "val.jsonl"
            base = {"context_id": "c", "dataset_root": "/data", "length": 64,
                    "suite": "libero_goal", "task": "task", "start": 0}
            train.write_text(json.dumps({**base, "id": "a", "episode": 1, "split": "train"}) + "\n")
            val.write_text(json.dumps({**base, "id": "b", "episode": 2, "split": "validation"}) + "\n")
            old = targets.parse_args
            targets.parse_args = lambda: type("Args", (), {
                "train_manifest": train, "val_manifest": val, "future_offset": 32,
                "output_train": root / "out_train.jsonl", "output_val": root / "out_val.jsonl",
                "output_report": root / "report.json",
            })()
            try:
                targets.main()
            finally:
                targets.parse_args = old
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["episode_overlap"], 0)
            self.assertEqual(report["train"]["windows"], 1)


if __name__ == "__main__":
    unittest.main()
