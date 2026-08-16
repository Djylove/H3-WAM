import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/h3wam/prepare_c59_failure_active_contract.py"
SPEC = importlib.util.spec_from_file_location("prepare_c59_failure_active_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sample(sample_id, episode_id, step, future, terminal, success):
    return {
        "sample_id": sample_id,
        "episode_id": episode_id,
        "current_step": step,
        "future_step": future,
        "terminal_step": terminal,
        "success": success,
        "value_target": 123.0,
    }


class FailureActiveContractTest(unittest.TestCase):
    def test_annotated_failure_masks_action_and_activates_on_future_crossing(self):
        rows = [
            sample(10, 3, 0, 32, 80, False),
            sample(11, 3, 8, 40, 80, False),
            sample(12, 3, 16, 48, 80, False),
        ]
        annotation = {
            "episode_id": 3,
            "failure_active_from_step": 40,
            "annotation_source": "explicit_intervention",
            "evidence": "event:7",
        }
        episode, labels = MODULE.derive_episode_overlay(3, rows, annotation)
        self.assertEqual(episode["failure_active_from_frame"], 1)
        self.assertEqual([row["action_loss_mask"] for row in labels], [0, 0, 0])
        self.assertEqual([row["failure_active_mask"] for row in labels], [0, 1, 1])
        self.assertAlmostEqual(labels[0]["fact_code_value_raw"], 0.6)
        self.assertAlmostEqual(labels[1]["fact_code_value_raw"], 1.5)
        self.assertEqual(labels[1]["fact_paper_progress_target"], 0.0)

    def test_unannotated_failure_uses_one_past_end_sentinel(self):
        rows = [sample(1, 9, 0, 32, 64, False), sample(2, 9, 8, 40, 64, False)]
        episode, labels = MODULE.derive_episode_overlay(9, rows, None)
        self.assertEqual(episode["failure_active_from_frame"], 2)
        self.assertFalse(episode["failure_onset_available"])
        self.assertEqual([row["failure_active_mask"] for row in labels], [0, 0])
        self.assertEqual([row["action_loss_mask"] for row in labels], [0, 0])

    def test_success_supervises_actions_and_rejects_failure_annotation(self):
        rows = [sample(1, 4, 0, 32, 64, True)]
        _, labels = MODULE.derive_episode_overlay(4, rows, None)
        self.assertEqual(labels[0]["action_loss_mask"], 1)
        with self.assertRaisesRegex(ValueError, "successful episode"):
            MODULE.derive_episode_overlay(
                4,
                rows,
                {
                    "episode_id": 4,
                    "failure_active_from_step": 5,
                    "annotation_source": "human_review",
                    "evidence": "video",
                },
            )

    def test_annotations_require_auditable_source_and_evidence(self):
        with self.assertRaisesRegex(ValueError, "source"):
            MODULE.index_annotations(
                [{"episode_id": 1, "failure_active_from_step": 5, "evidence": "x"}]
            )


if __name__ == "__main__":
    unittest.main()
