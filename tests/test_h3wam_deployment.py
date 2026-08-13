import unittest

import numpy as np
import torch

from fastwam.models.h3wam import (
    ActionEnsembler,
    action_denormalization_bounds,
    libero_dataset_action,
    libero_dataset_actions,
    libero_environment_actions,
    libero_observation_state,
    normalize_libero_environment_action_history,
    preprocess_libero_cameras,
    quaternion_to_axis_angle,
)


class H3WAMDeploymentTest(unittest.TestCase):
    def test_action_denormalization_bounds_follow_checkpoint_contract(self):
        cache = {
            "action_min": torch.full((7,), -2.0),
            "action_max": torch.full((7,), 2.0),
        }
        low, high = action_denormalization_bounds("minmax", cache)
        torch.testing.assert_close(low, cache["action_min"])
        torch.testing.assert_close(high, cache["action_max"])

        quantiles = {"q01": [-1.0] * 6 + [0.0], "q99": [1.0] * 7}
        low, high = action_denormalization_bounds("quantile", cache, quantiles)
        torch.testing.assert_close(low, torch.tensor(quantiles["q01"]))
        torch.testing.assert_close(high, torch.tensor(quantiles["q99"]))
        with self.assertRaises(ValueError):
            action_denormalization_bounds("quantile", cache)

    def test_action_ensembler_fuses_overlapping_absolute_timestamps(self):
        ensembler = ActionEnsembler()
        first = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        second = np.array([[6.0, 7.0], [8.0, 9.0]], dtype=np.float32)
        ensembler.add_actions(first, start_timestamp=0)
        ensembler.add_actions(second, start_timestamp=1)

        np.testing.assert_allclose(ensembler.get_action(0), [0.0, 1.0])
        np.testing.assert_allclose(ensembler.get_action(1), [4.0, 5.0])
        np.testing.assert_allclose(ensembler.get_action(2), [6.0, 7.0])

        ensembler.cleanup(current_timestamp=2)
        with self.assertRaises(ValueError):
            ensembler.get_action(1)
        np.testing.assert_allclose(ensembler.get_action(2), [6.0, 7.0])

    def test_action_ensembler_validates_shape(self):
        ensembler = ActionEnsembler()
        with self.assertRaises(ValueError):
            ensembler.add_actions(np.zeros((7,), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            ensembler.add_actions(np.zeros((2, 7), dtype=np.float32), -1)

    def test_quaternion_identity_and_half_turn(self):
        np.testing.assert_allclose(
            quaternion_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32)),
            np.zeros(3),
        )
        np.testing.assert_allclose(
            quaternion_to_axis_angle(np.array([1, 0, 0, 0], dtype=np.float32)),
            np.array([np.pi, 0, 0], dtype=np.float32),
            rtol=1e-6,
        )

    def test_observation_state_contract(self):
        state = libero_observation_state(
            {
                "eef_pos": np.array([1, 2, 3]),
                "eef_quat": np.array([0, 0, 0, 1]),
                "gripper_qpos": np.array([0.1, -0.1]),
            }
        )
        torch.testing.assert_close(
            state,
            torch.tensor([1, 2, 3, 0, 0, 0, 0.1, -0.1]),
        )

    def test_camera_preprocessing_rotates_and_concatenates(self):
        agent = np.zeros((2, 2, 3), dtype=np.uint8)
        wrist = np.zeros((2, 2, 3), dtype=np.uint8)
        agent[1, 1] = 255
        wrist[1, 1, 0] = 128
        result = preprocess_libero_cameras(
            agent,
            wrist,
            camera_height=2,
            camera_width=2,
        )
        self.assertEqual(tuple(result.shape), (1, 2, 4, 3))
        torch.testing.assert_close(result[0, 0, 0], torch.ones(3))
        self.assertAlmostEqual(float(result[0, 0, 2, 0]), 128 / 255)

    def test_environment_action_gripper_conversion_and_clipping(self):
        minimum = torch.tensor([-1.0] * 6 + [0.0])
        maximum = torch.tensor([1.0] * 7)
        normalized = torch.tensor(
            [
                [2.0, 0, 0, 0, 0, 0, 1.0],
                [0.0, 0, 0, 0, 0, 0, -1.0],
            ]
        )
        actions = libero_environment_actions(normalized, minimum, maximum)
        self.assertEqual(tuple(actions.shape), (2, 7))
        self.assertEqual(float(actions[0, 0]), 1.0)
        self.assertEqual(float(actions[0, -1]), -1.0)
        self.assertEqual(float(actions[1, -1]), 1.0)

        binary = libero_environment_actions(
            torch.tensor([[0.0, 0, 0, 0, 0, 0, 0.2]]),
            minimum,
            maximum,
            binarize_gripper=True,
        )
        self.assertEqual(abs(float(binary[0, -1])), 1.0)

    def test_dataset_action_inverts_environment_gripper(self):
        opened = libero_dataset_action(np.array([0, 0, 0, 0, 0, 0, -1]))
        closed = libero_dataset_action(np.array([0, 0, 0, 0, 0, 0, 1]))

        self.assertEqual(float(opened[-1]), 1.0)
        self.assertEqual(float(closed[-1]), 0.0)
        with self.assertRaises(ValueError):
            libero_dataset_action(np.zeros(6))

    def test_batched_dataset_action_inverts_only_the_gripper(self):
        environment = torch.tensor(
            [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.0], [0, 0, 0, 0, 0, 0, 1.0]]
        )
        dataset = libero_dataset_actions(environment)
        torch.testing.assert_close(dataset[:, :6], environment[:, :6])
        torch.testing.assert_close(dataset[:, -1], torch.tensor([1.0, 0.0]))

    def test_online_history_matches_dataset_normalization_and_preserves_padding(self):
        environment = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0.25, -0.5, 0, 0, 0, 0, -1],
                [-0.25, 0.5, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor([False, True, True])
        low = torch.tensor([-1.0] * 6 + [0.0])
        high = torch.tensor([1.0] * 7)
        normalized = normalize_libero_environment_action_history(
            environment, valid, low, high, clip=1.5
        )
        torch.testing.assert_close(normalized[0], torch.zeros(7))
        expected = 2.0 * (libero_dataset_actions(environment[valid]) - low) / (
            high - low
        ) - 1.0
        torch.testing.assert_close(normalized[valid], expected)
        self.assertEqual(float(normalized[1, -1]), 1.0)
        self.assertEqual(float(normalized[2, -1]), -1.0)

    def test_environment_action_temporal_median_rejects_single_spike(self):
        minimum = torch.tensor([-1.0] * 6 + [0.0])
        maximum = torch.tensor([1.0] * 7)
        normalized = torch.zeros((5, 7))
        normalized[2, 1] = -1.0
        actions = libero_environment_actions(
            normalized,
            minimum,
            maximum,
            temporal_median_window=3,
        )
        np.testing.assert_allclose(actions[:, 1], 0.0)

        with self.assertRaises(ValueError):
            libero_environment_actions(
                normalized,
                minimum,
                maximum,
                temporal_median_window=2,
            )


if __name__ == "__main__":
    unittest.main()
