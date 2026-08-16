import tempfile
import unittest
from pathlib import Path

import torch

from fastwam.models.h3wam.dreamwam_kv_carrier import (
    H3DreamWAMKVCarrierPolicy,
    REPEAT_LAYER49_CARRIER_SOURCE,
)
from fastwam.models.h3wam.fastwam_full_tower import (
    FASTWAM_COMMIT,
    H3FastWAMFullTowerPolicy,
    _load_pinned_fastwam_action_dit,
    depth_anchor_indices,
    fastwam_official_alpha,
    initialize_full_tower_from_d0,
    nearest_source_indices,
)


LAYERS = (9, 19, 29, 39, 49)


def build_d0() -> H3DreamWAMKVCarrierPolicy:
    return H3DreamWAMKVCarrierPolicy(
        enabled=True,
        carrier_layers=LAYERS,
        carrier_source_mode=REPEAT_LAYER49_CARRIER_SOURCE,
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
    )


def build_full(*, enabled: bool = True) -> H3FastWAMFullTowerPolicy:
    return H3FastWAMFullTowerPolicy(
        enabled=enabled,
        carrier_layers=LAYERS,
        action_dim=2,
        proprio_dim=3,
        context_dim=6,
        hidden_dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_head_dim=4,
        freq_dim=8,
        num_layers=30,
    )


def make_cache(*, batch: int = 2, tokens: int = 5):
    return {
        layer: {
            "k": torch.randn(batch, tokens, 2, 4),
            "v": torch.randn(batch, tokens, 2, 4),
        }
        for layer in LAYERS
    }


def make_inputs():
    return {
        "noisy_actions": torch.randn(2, 4, 2),
        "timestep": torch.tensor([100.0, 700.0]),
        "text_context": torch.randn(2, 5, 6),
        "proprio": torch.randn(2, 3),
        "text_mask": torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        ),
    }


class H3FastWAMFullTowerTest(unittest.TestCase):
    def test_official_source_is_pinned_and_full_depth(self):
        ActionDiT, _ = _load_pinned_fastwam_action_dit()
        self.assertEqual(
            ActionDiT.__module__, "_h3wam_fastwam_45d8e145.action_dit"
        )
        self.assertEqual(
            FASTWAM_COMMIT, "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
        )
        policy = build_full()
        self.assertEqual(len(policy.action_expert.blocks), 30)
        self.assertEqual(policy.action_block_to_h3_layer, (49,) * 30)

    def test_disabled_default_has_no_trainable_side_effect(self):
        policy = build_full(enabled=False)
        self.assertEqual(sum(parameter.numel() for parameter in policy.parameters()), 0)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            policy(video_kv_cache=make_cache(), **make_inputs())

    def test_depth_expansion_is_exact_d0_function_at_step_zero(self):
        torch.manual_seed(17)
        d0 = build_d0().eval()
        torch.manual_seed(91)
        full = build_full().eval()
        report = initialize_full_tower_from_d0(full, d0.state_dict())
        self.assertEqual(report.anchor_target_indices, (0, 7, 14, 22, 29))
        self.assertEqual(len(report.identity_target_indices), 25)
        self.assertFalse(report.alpha_scaling_applied)
        self.assertFalse(report.width_interpolation_applied)

        inputs = make_inputs()
        cache = make_cache()
        with torch.no_grad():
            expected = d0(video_kv_cache=cache, **inputs)
            actual = full(video_kv_cache=cache, **inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_added_identity_blocks_receive_gradients(self):
        torch.manual_seed(23)
        d0 = build_d0()
        full = build_full()
        report = initialize_full_tower_from_d0(full, d0.state_dict())
        output = full(video_kv_cache=make_cache(), **make_inputs())
        output.square().mean().backward()
        for index in report.identity_target_indices:
            block = full.action_expert.blocks[index]
            self.assertIsNotNone(block.self_attn.o.weight.grad)
            self.assertGreater(float(block.self_attn.o.weight.grad.norm()), 0.0)
        for index in report.anchor_target_indices:
            block = full.action_expert.blocks[index]
            self.assertGreater(float(block.self_attn.q.weight.grad.norm()), 0.0)
        self.assertGreater(float(full.proprio_encoder.weight.grad.norm()), 0.0)

    def test_only_layer49_changes_output_but_full_cache_is_required(self):
        torch.manual_seed(29)
        d0 = build_d0()
        full = build_full().eval()
        initialize_full_tower_from_d0(full, d0.state_dict())
        inputs = make_inputs()
        cache = make_cache()
        with torch.no_grad():
            baseline = full(video_kv_cache=cache, **inputs)
            early = {
                layer: {name: value.clone() for name, value in item.items()}
                for layer, item in cache.items()
            }
            early[9]["v"].add_(100.0)
            unchanged = full(video_kv_cache=early, **inputs)
            torch.testing.assert_close(baseline, unchanged, rtol=0, atol=0)
            last = {
                layer: {name: value.clone() for name, value in item.items()}
                for layer, item in cache.items()
            }
            last[49]["v"].add_(2.0)
            changed = full(video_kv_cache=last, **inputs)
        self.assertFalse(torch.equal(baseline, changed))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            full(video_kv_cache={49: cache[49]}, **inputs)

    def test_strict_state_restore_is_bit_exact(self):
        torch.manual_seed(31)
        d0 = build_d0()
        policy = build_full().eval()
        initialize_full_tower_from_d0(policy, d0.state_dict())
        inputs = make_inputs()
        cache = make_cache()
        with torch.no_grad():
            reference = policy(video_kv_cache=cache, **inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c58.pt"
            torch.save({"model": policy.state_dict()}, path)
            restored = build_full().eval()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(payload["model"], strict=True)
            with torch.no_grad():
                actual = restored(video_kv_cache=cache, **inputs)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_official_alpha_scaling_formula_and_depth_helpers(self):
        self.assertEqual(depth_anchor_indices(5, 30), (0, 7, 14, 22, 29))
        self.assertEqual(len(nearest_source_indices(5, 30)), 30)
        self.assertAlmostEqual(fastwam_official_alpha(3072, 1024), 3**0.5)


if __name__ == "__main__":
    unittest.main()
