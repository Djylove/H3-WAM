import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.h3wam import prepare_fact_value_target_dossier as dossier


class FactValueTargetDossierTest(unittest.TestCase):
    def _episode(self, root: Path, *, trial: int, success: bool) -> None:
        episode_root = root / f"d0_h32_s14000_goal_task0_trial{trial}_replan8"
        episode_root.mkdir(parents=True)
        trajectory = episode_root / "trajectory.npz"
        np.savez(
            trajectory,
            step=np.arange(4),
            policy_actions=np.zeros((4, 32, 7), dtype=np.float32),
        )
        result = {
            "tasks": [{"task_id": 0, "task": "pick the cup", "episodes": [{
                "success": success, "trajectory": str(trajectory),
            }]}]
        }
        (episode_root / "results.json").write_text(json.dumps(result))

    def test_success_targets_and_failure_censoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._episode(root, trial=0, success=True)
            self._episode(root, trial=3, success=True)
            self._episode(root, trial=1, success=False)
            manifest, report = root / "targets.jsonl", root / "report.json"
            old = dossier.parse_args
            dossier.parse_args = lambda: type("Args", (), {
                "result_root": root, "prefix": "d0_h32_s14000", "replan": 8,
                "validation_trial": 3, "output_manifest": manifest, "output_report": report,
            })()
            try:
                dossier.main()
            finally:
                dossier.parse_args = old
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["value_raw"] for row in rows[:3]], [2 / 3, 1 / 3, 0.0])
            summary = json.loads(report.read_text())
            self.assertEqual(summary["train_transitions"], 3)
            self.assertEqual(summary["val_transitions"], 3)
            self.assertEqual(summary["censored_failure_episodes"], 1)
            self.assertEqual(summary["gates"]["best_of_n_action_ranking"], "NO_GO_NO_COUNTERFACTUAL_OUTCOMES")


if __name__ == "__main__":
    unittest.main()
