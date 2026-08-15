import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "h3wam" / "audit_fact_rollout_transitions.py"
SPEC = importlib.util.spec_from_file_location("audit_fact_rollout_transitions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FactRolloutTransitionAuditTest(unittest.TestCase):
    def write_episode(self, root: Path, trial: int, success: bool) -> None:
        directory = root / f"d0_h32_s14000_object_task0_trial{trial}_replan8"
        directory.mkdir()
        trajectory = directory / f"task00_trial{trial:02d}_trajectory.npz"
        replans = 3
        np.savez(
            trajectory,
            step=np.arange(replans),
            agentview_image=np.zeros((replans, 2, 2, 3), dtype=np.uint8),
            wristview_image=np.zeros((replans, 2, 2, 3), dtype=np.uint8),
            eef_pos=np.zeros((replans, 3), dtype=np.float32),
            eef_quat=np.zeros((replans, 4), dtype=np.float32),
            gripper_qpos=np.zeros((replans, 2), dtype=np.float32),
            previous_action=np.zeros((replans, 7), dtype=np.float32),
            policy_actions=np.zeros((replans, 32, 7), dtype=np.float32),
            sim_state=np.zeros((replans, 5), dtype=np.float64),
        )
        payload = {
            "tasks": [
                {
                    "episodes": [
                        {
                            "success": success,
                            "trajectory": str(trajectory),
                        }
                    ]
                }
            ]
        }
        (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_audit_freezes_trial_split_and_blocks_failure_imitation(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            self.write_episode(root, trial=0, success=True)
            self.write_episode(root, trial=3, success=False)
            output = root / "audit.json"
            argv = ["audit", str(root), "--output", str(output)]
            with patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["causal_transitions"], 4)
            self.assertEqual(report["split"]["train"]["trials"], [0])
            self.assertEqual(report["split"]["val"]["trials"], [3])
            self.assertEqual(report["split_overlap"], 0)
            self.assertEqual(report["contract_gates"]["failure_onset"], "UNKNOWN")
            self.assertEqual(report["contract_gates"]["failure_imitation"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
