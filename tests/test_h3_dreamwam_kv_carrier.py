import unittest

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import (
    ALIGNED_5LAYER_CARRIER_SOURCE,
    DREAMWAM_COMMIT,
    H3DreamWAMKVCarrierPolicy,
    REPEAT_LAYER49_CARRIER_SOURCE,
    _load_pinned_dreamwam_carrier,
    h3_kv_cache_bytes,
)


class H3DreamWAMKVCarrierPolicyTest(unittest.TestCase):
    def build_policy(
        self,
        *,
        enabled=True,
        carrier_layers=(1, 3),
        carrier_source_mode=ALIGNED_5LAYER_CARRIER_SOURCE,
    ):
        return H3DreamWAMKVCarrierPolicy(
            enabled=enabled,
            carrier_layers=carrier_layers,
            carrier_source_mode=carrier_source_mode,
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

    def test_d0_repeats_layer49_with_independent_clones_and_same_parameters(self):
        layers = (1, 3, 49)
        torch.manual_seed(101)
        aligned = self.build_policy(carrier_layers=layers)
        torch.manual_seed(101)
        d0 = self.build_policy(
            carrier_layers=layers,
            carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in aligned.parameters()),
            sum(parameter.numel() for parameter in d0.parameters()),
        )
        for name, tensor in aligned.state_dict().items():
            torch.testing.assert_close(tensor, d0.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(d0.action_block_to_h3_layer, (49, 49, 49))

        cache = {
            layer: {
                "k": torch.randn(2, 5, 2, 4),
                "v": torch.randn(2, 5, 2, 4),
            }
            for layer in layers
        }
        resolved = d0._resolve_carrier_cache(cache, batch=2)
        self.assertEqual(len(resolved), len(layers))
        signatures = []
        for item in resolved:
            torch.testing.assert_close(item["k"], cache[49]["k"].flatten(2, 3))
            torch.testing.assert_close(item["v"], cache[49]["v"].flatten(2, 3))
            signatures.extend(
                item[name].untyped_storage().data_ptr() for name in ("k", "v")
            )
        self.assertEqual(len(signatures), len(set(signatures)))

        inputs = self.inputs()
        with torch.no_grad():
            reference = d0(video_kv_cache=cache, **inputs)
            changed_early = {
                layer: {name: tensor.clone() for name, tensor in item.items()}
                for layer, item in cache.items()
            }
            changed_early[1]["k"].add_(100.0)
            unchanged = d0(video_kv_cache=changed_early, **inputs)
            torch.testing.assert_close(reference, unchanged, rtol=0, atol=0)
            changed_layer49 = {
                layer: {name: tensor.clone() for name, tensor in item.items()}
                for layer, item in cache.items()
            }
            changed_layer49[49]["v"].add_(2.0)
            changed = d0(video_kv_cache=changed_layer49, **inputs)
            self.assertFalse(torch.equal(reference, changed))

    def test_d0_requires_layer49(self):
        with self.assertRaisesRegex(ValueError, "requires H3 layer49"):
            self.build_policy(carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE)

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
