import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy
from fastwam.models.h3wam.fact_joint_aux import (
    H3FactJointAuxPolicy,
    fact_joint_auxiliary_loss,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_c55_fact_joint_action_test",
    ROOT / "scripts/h3wam/train_c55_fact_joint_action.py",
)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_c55_deployment_checkpoint_test",
    ROOT / "scripts/h3wam/export_c55_deployment_checkpoint.py",
)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
sys.modules[EXPORT_SPEC.name] = EXPORT
EXPORT_SPEC.loader.exec_module(EXPORT)


class H3FactJointAuxPolicyTest(unittest.TestCase):
    @staticmethod
    def carrier() -> H3DreamWAMKVCarrierPolicy:
        return H3DreamWAMKVCarrierPolicy(
            enabled=True,
            carrier_layers=(1, 3),
            action_dim=2,
            proprio_dim=3,
            context_dim=6,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            attn_head_dim=4,
            freq_dim=8,
        )

    @staticmethod
    def batch() -> dict:
        return {
            "noisy_actions": torch.randn(2, 4, 2),
            "timestep": torch.tensor([100.0, 700.0]),
            "clean_executed_actions": torch.randn(2, 4, 2),
            "text_context": torch.randn(2, 5, 6),
            "proprio": torch.randn(2, 3),
            "video_kv_cache": {
                layer: {
                    name: torch.randn(2, 5, 2, 4)
                    for name in ("k", "v")
                }
                for layer in (1, 3)
            },
        }

    def test_parent_action_path_is_exact_after_wrapping(self):
        torch.manual_seed(11)
        parent = self.carrier()
        torch.manual_seed(19)
        candidate = H3FactJointAuxPolicy(
            self.carrier(), hidden_dim=8, future_h3_dim=5, future_state_dim=3
        )
        candidate.carrier.load_state_dict(parent.state_dict(), strict=True)
        batch = self.batch()
        common = {
            key: batch[key]
            for key in ("text_context", "proprio", "video_kv_cache")
        }
        with torch.no_grad():
            expected = parent(batch["noisy_actions"], batch["timestep"], **common)
            actual = candidate.forward_action(
                batch["noisy_actions"], batch["timestep"], **common
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_auxiliary_targets_cannot_change_action_forward(self):
        torch.manual_seed(23)
        model = H3FactJointAuxPolicy(
            self.carrier(), hidden_dim=8, future_h3_dim=5, future_state_dim=3
        )
        batch = self.batch()
        outputs = model.forward_joint(**batch)
        first = fact_joint_auxiliary_loss(
            outputs,
            action_loss=outputs["action"].square().mean(),
            future_h3_target=torch.zeros(2, 5),
            future_state_target=torch.zeros(2, 3),
            value_target=torch.zeros(2),
        )
        second = fact_joint_auxiliary_loss(
            outputs,
            action_loss=outputs["action"].square().mean(),
            future_h3_target=torch.full((2, 5), 100.0),
            future_state_target=torch.full((2, 3), -100.0),
            value_target=torch.full((2,), 50.0),
        )
        torch.testing.assert_close(
            first["action_loss"], second["action_loss"], rtol=0.0, atol=0.0
        )

    def test_auxiliary_losses_reach_shared_action_blocks(self):
        torch.manual_seed(29)
        model = H3FactJointAuxPolicy(
            self.carrier(), hidden_dim=8, future_h3_dim=5, future_state_dim=3
        )
        outputs = model.forward_joint(**self.batch())
        losses = fact_joint_auxiliary_loss(
            outputs,
            action_loss=outputs["action"].sum() * 0.0,
            future_h3_target=torch.randn(2, 5),
            future_state_target=torch.randn(2, 3),
            value_target=torch.randn(2),
        )
        losses["loss"].backward()
        grad = model.carrier.action_expert.blocks[0].self_attn.q.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.future_h3_decoder.weight.grad.abs().sum()), 0.0
        )

    def test_fact_relative_weights_preserve_parent_action_scale(self):
        prediction = {
            "future_h3": torch.zeros(2, 5),
            "future_state": torch.zeros(2, 3),
            "value": torch.zeros(2),
        }
        action_loss = torch.tensor(2.0)
        losses = fact_joint_auxiliary_loss(
            prediction,
            action_loss=action_loss,
            future_h3_target=torch.zeros(2, 5),
            future_state_target=torch.zeros(2, 3),
            value_target=torch.zeros(2),
        )
        torch.testing.assert_close(losses["loss"], action_loss)

    def test_libero_action_roundtrip_is_exact_including_gripper(self):
        environment = torch.tensor(
            [
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.0],
                [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 1.0],
            ]
        )
        dataset = TRAIN.environment_actions_to_dataset(environment)
        torch.testing.assert_close(dataset[:, -1], torch.tensor([1.0, 0.0]))
        restored = TRAIN.dataset_actions_to_environment(dataset)
        torch.testing.assert_close(restored, environment, rtol=0.0, atol=0.0)

    def test_paired_sampler_is_stateless_and_stream_specific(self):
        values = [TRAIN.stable_index(7, "demo", i, 97) for i in range(100)]
        self.assertEqual(values, [TRAIN.stable_index(7, "demo", i, 97) for i in range(100)])
        self.assertNotEqual(
            values, [TRAIN.stable_index(7, "rollout", i, 97) for i in range(100)]
        )
        self.assertTrue(all(0 <= value < 97 for value in values))

    def test_future_h3_target_norm_uses_train_samples_only(self):
        observation_ids = torch.tensor([10, 11, 12, 13])
        features = torch.tensor(
            [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [1000.0, -1000.0]]
        )
        rows = [
            {"split": "train", "future_observation_id": 10},
            {"split": "train", "future_observation_id": 11},
            {"split": "train", "future_observation_id": 11},
            {"split": "train", "future_observation_id": 12},
            {"split": "validation", "future_observation_id": 13},
        ]
        mean, std, identity = TRAIN.fit_future_h3_target_norm(
            observation_ids, features, rows
        )
        train = features[torch.tensor([0, 1, 1, 2])]
        normalized = (train - mean) / std
        torch.testing.assert_close(normalized.mean(0), torch.zeros(2), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            normalized.std(0, unbiased=False), torch.ones(2), atol=1e-6, rtol=0.0
        )
        self.assertEqual(len(identity), 64)

    def test_joint_deployment_export_strips_only_carrier_prefix(self):
        state = {
            "carrier.a": torch.tensor(1.0),
            "carrier.b": torch.tensor(2.0),
            "future_h3_decoder.weight": torch.tensor(3.0),
        }
        result = EXPORT.carrier_state(
            {"model": state, "contract": {"arm": "joint_aux"}}, {"a", "b"}
        )
        self.assertEqual(set(result), {"a", "b"})
        self.assertEqual(float(result["a"]), 1.0)


if __name__ == "__main__":
    unittest.main()
