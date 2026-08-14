import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/h3wam/train_libero_h3_action.py"
SPEC = importlib.util.spec_from_file_location("train_libero_h3_action", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

FEATURE_SCRIPT = (
    Path(__file__).parents[1] / "scripts/h3wam/train_h3_feature_action.py"
)
FEATURE_SPEC = importlib.util.spec_from_file_location(
    "train_h3_feature_action", FEATURE_SCRIPT
)
FEATURE_MODULE = importlib.util.module_from_spec(FEATURE_SPEC)
assert FEATURE_SPEC.loader is not None
FEATURE_SPEC.loader.exec_module(FEATURE_MODULE)


class H3TrainingSplitTest(unittest.TestCase):
    def test_split_is_disjoint_by_episode_for_every_task(self):
        items = []
        for task in range(2):
            for episode in range(task * 10, task * 10 + 4):
                for start in (0, 10):
                    items.append(
                        {
                            "id": f"{task}-{episode}-{start}",
                            "task_group": task,
                            "episode": episode,
                            "start": start,
                        }
                    )
        train, validation = MODULE.split_by_episode(items, val_episodes_per_task=1)
        train_episodes = {item["episode"] for item in train}
        validation_episodes = {item["episode"] for item in validation}
        self.assertFalse(train_episodes & validation_episodes)
        self.assertEqual(len(validation_episodes), 2)

    def test_minmax_normalize_maps_endpoints(self):
        import torch

        value = torch.tensor([0.0, 5.0, 10.0])
        result = MODULE.minmax_normalize(value, torch.zeros(3), torch.full((3,), 10.0))
        torch.testing.assert_close(result, torch.tensor([-1.0, 0.0, 1.0]))

    def test_event_stage_boundaries_follow_gripper_events(self):
        import torch

        gripper = torch.ones(100)
        gripper[40:90] = 0
        boundaries = FEATURE_MODULE.event_stage_boundaries(gripper)

        self.assertEqual(boundaries, (22, 56, 59))
        self.assertEqual(
            [
                FEATURE_MODULE.event_stage_for_start(start, boundaries)
                for start in (0, 22, 56, 59)
            ],
            [0, 1, 2, 3],
        )

    def test_event_stage_release_respects_horizon_one(self):
        import torch

        gripper = torch.ones(100)
        gripper[40:90] = 0

        self.assertEqual(
            FEATURE_MODULE.event_stage_boundaries(gripper, action_horizon=1),
            (22, 56, 90),
        )

    def test_feature_split_qualifies_episode_by_suite(self):
        items = []
        for suite in ("libero_goal", "libero_object"):
            for episode in range(3):
                items.append(
                    {
                        "id": f"{suite}_ep{episode:06d}_s000000",
                        "suite": suite,
                        "task": f"{suite}-task",
                        "episode": episode,
                        "start": 0,
                    }
                )
        train, validation = FEATURE_MODULE.split_feature_items_by_episode(items, 1)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 2)
        self.assertEqual(
            {FEATURE_MODULE.episode_key(item) for item in validation},
            {"libero_goal:ep000002", "libero_object:ep000002"},
        )

    def test_explicit_feature_split_and_prefixed_previous_id(self):
        train_item = {
            "id": "libero_goal_ep000007_s000011",
            "suite": "libero_goal",
            "task": "task",
            "episode": 7,
            "start": 11,
            "split": "train",
        }
        val_item = dict(train_item, id="libero_goal_ep000008_s000011", episode=8, split="val")
        train, validation = FEATURE_MODULE.split_feature_items_by_episode(
            [train_item, val_item], 1
        )
        self.assertEqual(train, [train_item])
        self.assertEqual(validation, [val_item])
        self.assertEqual(
            FEATURE_MODULE.previous_window_id(train_item),
            "libero_goal_ep000007_s000010",
        )


if __name__ == "__main__":
    unittest.main()
