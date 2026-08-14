import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/h3wam/build_episode_disjoint_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_episode_disjoint_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EpisodeDisjointSplitTest(unittest.TestCase):
    def rows(self):
        rows = []
        for task_index in range(2):
            for episode in range(5):
                for start in range(3):
                    rows.append(
                        {
                            "id": f"suite_t{task_index}_ep{episode}_s{start}",
                            "dataset_root": "/data",
                            "suite": "suite",
                            "episode": task_index * 100 + episode,
                            "start": start,
                            "task": f"task {task_index}",
                            "context_id": f"task_{task_index}",
                        }
                    )
        return rows

    def test_split_is_deterministic_episode_disjoint_and_task_complete(self):
        rows = self.rows()
        first = MODULE.split_rows(rows, val_fraction=0.2, salt="fixed")
        second = MODULE.split_rows(rows, val_fraction=0.2, salt="fixed")
        self.assertEqual(first, second)
        train, val, audit = first
        train_eps = {MODULE.episode_key(row) for row in train}
        val_eps = {MODULE.episode_key(row) for row in val}
        self.assertFalse(train_eps & val_eps)
        self.assertEqual({row["id"] for row in train + val}, {row["id"] for row in rows})
        self.assertTrue(audit["all_tasks_present_in_both_splits"])
        self.assertEqual(audit["episode_overlap"], 0)

    def test_rejects_episode_with_multiple_tasks(self):
        rows = self.rows()
        rows[1]["task"] = "corrupt task"
        with self.assertRaisesRegex(ValueError, "multiple tasks"):
            MODULE.split_rows(rows, val_fraction=0.2, salt="fixed")


if __name__ == "__main__":
    unittest.main()
