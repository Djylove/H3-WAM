import unittest

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import (
    DREAMWAM_COMMIT,
    H3DreamWAMKVCarrierPolicy,
    _load_pinned_dreamwam_carrier,
    h3_kv_cache_bytes,
)


class H3DreamWAMKVCarrierPolicyTest(unittest.TestCase):
    def build_policy(self, *, enabled=True):
        return H3DreamWAMKVCarrierPolicy(
            enabled=enabled,
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

    def cache(self, *, batch=2, tokens=5):
        return {
            layer: {
                "k": torch.randn(batch, tokens, 2, 4),
                "v": torch.randn(batch, tokens, 2, 4),
            }
            for layer in (1, 3)
        }

    def inputs(self):
        return {
            "noisy_actions": torch.randn(2, 4, 2),
            "timestep": torch.tensor([100.0, 700.0]),
            "text_context": torch.randn(2, 5, 6),
            "proprio": torch.randn(2, 3),
        }

    def test_is_explicitly_disabled_by_default(self):
        policy = self.build_policy(enabled=False)
        self.assertFalse(policy.enabled)
        self.assertEqual(sum(p.numel() for p in policy.parameters()), 0)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            policy(video_kv_cache=self.cache(), **self.inputs())

    def test_pinned_official_action_expert_and_carrier_helpers_load(self):
        ActionDiT, JointMoT = _load_pinned_dreamwam_carrier()
        self.assertEqual(ActionDiT.__module__, "dreamwam.experts")
        self.assertEqual(JointMoT.__module__, "dreamwam.mot")
        self.assertEqual(DREAMWAM_COMMIT, "6e989facc0c452fd3488d75f60bc36411005558c")

    def test_forward_uses_distinct_layer_kv_and_backpropagates(self):
        torch.manual_seed(7)
        policy = self.build_policy()
        inputs = self.inputs()
        cache = self.cache()
        output = policy(video_kv_cache=cache, **inputs)
        self.assertEqual(tuple(output.shape), (2, 4, 2))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(policy.action_expert.blocks[0].self_attn.q.weight.grad)
        self.assertIsNotNone(policy.proprio_encoder.weight.grad)

        changed = {
            layer: {name: tensor.clone() for name, tensor in item.items()}
            for layer, item in cache.items()
        }
        changed[3]["v"].add_(2.0)
        with torch.no_grad():
            changed_output = policy(video_kv_cache=changed, **inputs)
        self.assertFalse(torch.equal(output.detach(), changed_output))

    def test_action_block_mapping_is_explicit(self):
        policy = self.build_policy()
        self.assertEqual(policy.action_block_to_h3_layer, (1, 3))
        self.assertEqual(len(policy.action_expert.blocks), 2)

    def test_rejects_missing_or_repeated_last_layer_cache(self):
        policy = self.build_policy()
        inputs = self.inputs()
        cache = self.cache()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            policy(video_kv_cache={1: cache[1]}, **inputs)

        repeated = self.cache()
        repeated[3]["k"] = repeated[1]["k"]
        with self.assertRaisesRegex(ValueError, "layer-specific"):
            policy(video_kv_cache=repeated, **inputs)

        shared = torch.randn(2, 6, 2, 4)
        offset_alias = self.cache()
        offset_alias[1]["k"] = shared[:, :5]
        offset_alias[3]["k"] = shared[:, 1:]
        with self.assertRaisesRegex(ValueError, "layer-specific"):
            policy(video_kv_cache=offset_alias, **inputs)

    def test_cache_budget_is_explicit(self):
        self.assertEqual(
            h3_kv_cache_bytes(layers=5, tokens=32),
            5 * 32 * 56 * 128 * 2 * 2,
        )
        self.assertEqual(
            h3_kv_cache_bytes(layers=5, tokens=98),
            5 * 98 * 56 * 128 * 2 * 2,
        )


if __name__ == "__main__":
    unittest.main()
