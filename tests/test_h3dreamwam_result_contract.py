import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.h3dreamwam.result_contract import completed_rollout, is_completed_rollout


class RolloutResultContractTest(unittest.TestCase):
    def test_only_complete_exact_counts_are_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            payload = {
                "status": "complete",
                "expected_tasks": 1,
                "expected_episodes": 2,
                "finished_episodes": 2,
                "tasks": [{"episodes": [{"success": False}, {"success": True}]}],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(completed_rollout(path), payload)
            self.assertTrue(is_completed_rollout(path))

    def test_running_or_incomplete_results_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "expected_tasks": 1,
                        "expected_episodes": 1,
                        "finished_episodes": 0,
                        "tasks": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(is_completed_rollout(path))
            with self.assertRaisesRegex(ValueError, "not complete"):
                completed_rollout(path)


if __name__ == "__main__":
    unittest.main()
