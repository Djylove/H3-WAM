import importlib.util
import unittest

import torch

from fastwam.models.h3dreamwam import (
    H3DoTActionHead,
    H3DoTKVFusion,
    H3DoTWAM,
    apply_h3_rotary,
    expand_h3_rgb_flow_projections,
    inverse_h3_rotary,
)


class H3DoTDockingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_inverse_h3_rotary_recovers_canonical_key(self) -> None:
        canonical = torch.randn(2, 5, 3, 8)
        phase = torch.randn(5, 4)
        cos_half = phase.cos()
        sin_half = phase.sin()
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        rotated = apply_h3_rotary(canonical, cos, sin)
        recovered = inverse_h3_rotary(rotated, cos, sin)
        torch.testing.assert_close(recovered, canonical, rtol=1e-5, atol=1e-6)

    def test_uniform_layer_fusion_matches_mean_after_identity_channel_map(self) -> None:
        fusion = H3DoTKVFusion(
            video_layers=3,
            action_layers=1,
            video_num_heads=2,
            video_head_dim=4,
            action_num_heads=2,
            action_head_dim=4,
        )
        with torch.no_grad():
            fusion.key_channel.weight.copy_(torch.eye(8))
            fusion.value_channel.weight.copy_(torch.eye(8))
        keys = torch.randn(3, 1, 5, 2, 4)
        values = torch.randn_like(keys)
        cos = torch.ones(5, 4)
        sin = torch.zeros(5, 4)
        fused_key, fused_value = fusion(
            rotated_video_keys=keys,
            video_values=values,
            video_cos=cos,
            video_sin=sin,
        )
        self.assertEqual(fused_key.shape, (1, 1, 5, 2, 4))
        self.assertEqual(fused_value.shape, fused_key.shape)
        torch.testing.assert_close(fused_value[0], values.mean(dim=0))
        expected_key = fusion.key_norm(keys.mean(dim=0))
        torch.testing.assert_close(fused_key[0], expected_key)

    def test_layer_mixing_is_independent_per_attention_head(self) -> None:
        fusion = H3DoTKVFusion(
            video_layers=2,
            action_layers=1,
            video_num_heads=2,
            video_head_dim=2,
            action_num_heads=2,
            action_head_dim=2,
        )
        with torch.no_grad():
            fusion.key_channel.weight.copy_(torch.eye(4))
            fusion.value_channel.weight.copy_(torch.eye(4))
            fusion.layer_mix[0, 0].copy_(torch.tensor([1.0, 0.0]))
            fusion.layer_mix[0, 1].copy_(torch.tensor([0.0, 1.0]))
        keys = torch.randn(2, 1, 3, 2, 2)
        values = torch.zeros_like(keys)
        values[0, :, :, 0] = 3.0
        values[1, :, :, 1] = 7.0
        cos = torch.ones(3, 2)
        sin = torch.zeros(3, 2)
        _, fused_value = fusion(
            rotated_video_keys=keys,
            video_values=values,
            video_cos=cos,
            video_sin=sin,
        )
        torch.testing.assert_close(
            fused_value[0, :, :, 0],
            torch.full((1, 3, 2), 3.0),
        )
        torch.testing.assert_close(
            fused_value[0, :, :, 1],
            torch.full((1, 3, 2), 7.0),
        )

    def test_channel_map_supports_narrower_action_attention(self) -> None:
        fusion = H3DoTKVFusion(
            video_layers=3,
            action_layers=1,
            video_num_heads=4,
            video_head_dim=8,
            action_num_heads=2,
            action_head_dim=8,
        )
        keys = torch.randn(3, 2, 5, 4, 8)
        values = torch.randn_like(keys)
        docked_keys, docked_values = fusion(
            rotated_video_keys=keys,
            video_values=values,
            video_cos=torch.ones(5, 8),
            video_sin=torch.zeros(5, 8),
        )
        self.assertEqual(docked_keys.shape, (1, 2, 5, 2, 8))
        self.assertEqual(docked_values.shape, docked_keys.shape)

    def test_single_layer_action_head_has_no_text_cross_attention(self) -> None:
        head = H3DoTActionHead(
            action_dim=3,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=4,
        )
        self.assertFalse(any("cross_attn" in name for name, _ in head.named_parameters()))
        docked_keys = torch.randn(1, 2, 6, 2, 4)
        docked_values = torch.randn_like(docked_keys)
        output = head(
            noisy_actions=torch.randn(2, 5, 3),
            timestep=torch.tensor([100.0, 600.0]),
            docked_keys=docked_keys,
            docked_values=docked_values,
        )
        self.assertEqual(output.shape, (2, 5, 3))
        output.square().mean().backward()
        self.assertGreater(float(head.layers[0].attn.to_q.weight.grad.abs().sum()), 0.0)

    def test_deeper_action_carrier_consumes_one_fused_cache_per_layer(self) -> None:
        action_layers = 4
        fusion = H3DoTKVFusion(
            video_layers=3,
            action_layers=action_layers,
            video_num_heads=2,
            video_head_dim=4,
            action_num_heads=2,
            action_head_dim=4,
        )
        head = H3DoTActionHead(
            action_dim=3,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            num_layers=action_layers,
            frequency_dim=4,
        )
        keys = torch.randn(3, 2, 6, 2, 4)
        values = torch.randn_like(keys)
        docked_keys, docked_values = fusion(
            rotated_video_keys=keys,
            video_values=values,
            video_cos=torch.ones(6, 4),
            video_sin=torch.zeros(6, 4),
        )
        self.assertEqual(docked_keys.shape, (action_layers, 2, 6, 2, 4))
        output = head(
            noisy_actions=torch.randn(2, 5, 3),
            timestep=torch.tensor([100.0, 600.0]),
            docked_keys=docked_keys,
            docked_values=docked_values,
        )
        self.assertEqual(output.shape, (2, 5, 3))
        output.square().mean().backward()
        for layer in head.layers:
            self.assertGreater(float(layer.attn.to_q.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(fusion.layer_mix.grad.abs().sum()), 0.0)

    def test_action_loss_updates_layer_mixing_and_h3_cache(self) -> None:
        fusion = H3DoTKVFusion(
            video_layers=2,
            action_layers=1,
            video_num_heads=2,
            video_head_dim=4,
            action_num_heads=2,
            action_head_dim=4,
        )
        head = H3DoTActionHead(
            action_dim=2,
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            num_layers=1,
            frequency_dim=4,
        )
        keys = torch.randn(2, 1, 4, 2, 4, requires_grad=True)
        values = torch.randn_like(keys, requires_grad=True)
        cos = torch.ones(4, 4)
        sin = torch.zeros(4, 4)
        docked_keys, docked_values = fusion(
            rotated_video_keys=keys,
            video_values=values,
            video_cos=cos,
            video_sin=sin,
        )
        output = head(
            noisy_actions=torch.randn(1, 3, 2),
            timestep=torch.tensor([500.0]),
            docked_keys=docked_keys,
            docked_values=docked_values,
        )
        output.square().mean().backward()
        self.assertGreater(float(fusion.layer_mix.grad.abs().sum()), 0.0)
        self.assertGreater(float(keys.grad.abs().sum()), 0.0)
        self.assertGreater(float(values.grad.abs().sum()), 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("diffusers") is not None,
        "diffusers is only installed in the H3 training environment",
    )
    def test_tiny_h3_dotwam_forward_and_backward(self) -> None:
        from diffusers import MiniMaxH3Transformer3DModel

        h3 = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=64,
            in_channels=4,
            audio_in_channels=8,
            patch_size=(1, 2, 2),
            text_dim=32,
            freq_dim=16,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=2,
        )
        expand_h3_rgb_flow_projections(
            h3,
            flow_channels=4,
            flow_output_init_scale=0.0,
            generator=torch.Generator().manual_seed(8),
        )
        action_head = H3DoTActionHead(
            action_dim=3,
            hidden_dim=24,
            ffn_dim=48,
            num_heads=2,
            head_dim=16,
            num_layers=1,
            frequency_dim=16,
        )
        fusion = H3DoTKVFusion(
            video_layers=2,
            action_layers=1,
            video_num_heads=2,
            video_head_dim=16,
            action_num_heads=2,
            action_head_dim=16,
        )
        model = H3DoTWAM(
            h3,
            action_head,
            fusion,
            state_dim=4,
            text_dim=32,
            rgb_patch_width=16,
            use_gradient_checkpointing=True,
        )
        video_indices = torch.tensor([3, 4, 5, 6])
        text_indices = torch.tensor([0, 1, 2])
        audio_indices = torch.tensor([7, 8])
        token_tags = torch.tensor([1, 1, 1, 0, 0, 0, 0, 2, 2])
        timestep_indices = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 1])
        position_ids = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ]
        )
        output = model(
            video_rows=torch.randn(1, 4, 32),
            audio_rows=torch.randn(1, 2, 8),
            context=torch.randn(1, 2, 32),
            timestep=torch.tensor([0.4, 1.0]),
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            condition_video_indices=video_indices[:1],
            noisy_actions=torch.randn(1, 5, 3),
            action_timestep=torch.tensor([500.0]),
            state=torch.randn(1, 4),
            context_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertEqual(output.rgb_velocity_rows.shape, (1, 4, 16))
        self.assertEqual(output.flow_velocity_rows.shape, (1, 4, 16))
        self.assertEqual(output.action_velocity.shape, (1, 5, 3))
        self.assertIsNotNone(output.docked_keys)
        self.assertIsNotNone(output.docked_values)
        self.assertEqual(torch.count_nonzero(output.flow_velocity_rows), 0)
        loss = output.rgb_velocity_rows.square().mean()
        loss = loss + output.action_velocity.square().mean()
        loss.backward()
        self.assertGreater(
            float(model.kv_fusion.layer_mix.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.hub_layers[0].h3_block.attn.to_k.weight.grad.abs().sum()),
            0.0,
        )

        with torch.no_grad():
            cached = model(
                video_rows=torch.empty(0),
                audio_rows=torch.empty(0),
                context=torch.empty(0),
                timestep=torch.empty(0),
                timestep_indices=torch.empty(0, dtype=torch.long),
                token_tags=torch.empty(0, dtype=torch.long),
                position_ids=torch.empty(0, 3, dtype=torch.long),
                video_indices=torch.empty(0, dtype=torch.long),
                audio_indices=torch.empty(0, dtype=torch.long),
                text_indices=torch.empty(0, dtype=torch.long),
                condition_video_indices=torch.empty(0, dtype=torch.long),
                noisy_actions=torch.randn(1, 5, 3),
                action_timestep=torch.tensor([250.0]),
                state=torch.empty(0),
                cached_docked_keys=output.docked_keys.detach(),
                cached_docked_values=output.docked_values.detach(),
            )
        self.assertEqual(cached.action_velocity.shape, (1, 5, 3))
        self.assertEqual(cached.rgb_velocity_rows.numel(), 0)


if __name__ == "__main__":
    unittest.main()
