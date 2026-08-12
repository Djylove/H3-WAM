import unittest

import torch
from torch import nn

from fastwam.models.h3wam import (
    H3ActionAdapter,
    H3ActionBridge,
    make_first_frame_payload,
)


class _FakeH3(nn.Module):
    """Small differentiable stand-in for H3's frozen audio projection."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.audio_projection = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)

    def forward(
        self,
        x,
        timestep,
        context,
        transformer_options=None,
        minimax_payload=None,
    ):
        video, audio = x
        scale = timestep.reshape(-1, 1, 1, 1).to(audio.dtype) / 1000.0
        return [torch.zeros_like(video), self.audio_projection(audio) * scale]


class H3ActionAdapterTest(unittest.TestCase):
    def test_action_latent_layout_round_trip_shape(self):
        adapter = H3ActionAdapter(action_dim=14)
        actions = torch.randn(2, 32, 14)

        latents = adapter.encode_actions(actions)
        reconstructed = adapter.decode_velocity(latents)

        self.assertEqual(latents.shape, (2, 32, 2, 32))
        self.assertEqual(reconstructed.shape, actions.shape)

    def test_single_arm_action_shape_is_supported(self):
        adapter = H3ActionAdapter(action_dim=7)
        actions = torch.randn(1, 16, 7)

        self.assertEqual(adapter.encode_actions(actions).shape, (1, 32, 2, 16))

    def test_current_state_can_condition_every_action_step(self):
        adapter = H3ActionAdapter(action_dim=7, state_dim=8)
        actions = torch.randn(2, 16, 7)
        state = torch.randn(2, 8)

        latents = adapter.encode_actions(actions, state)

        self.assertEqual(latents.shape, (2, 32, 2, 16))
        with self.assertRaisesRegex(ValueError, "state with dimension 8 is required"):
            adapter.encode_actions(actions)

    def test_direct_decoder_conditioning_receives_gradients(self):
        adapter = H3ActionAdapter(
            action_dim=7,
            state_dim=8,
            context_dim=16,
            direct_conditioning=True,
        )
        actions = torch.randn(2, 4, 7)
        state = torch.randn(2, 8)
        context = torch.randn(2, 3, 16)
        latents = adapter.encode_actions(actions, state)

        output = adapter.decode_velocity(latents, state=state, context=context)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 4, 7))
        self.assertIsNotNone(adapter.decoder_state_projection.weight.grad)
        self.assertIsNotNone(adapter.decoder_context_projection[-1].weight.grad)

    def test_direct_decoder_requires_context(self):
        adapter = H3ActionAdapter(
            action_dim=7,
            state_dim=8,
            direct_conditioning=True,
        )
        latents = torch.randn(1, 32, 2, 4)
        with self.assertRaisesRegex(ValueError, "direct decoder context"):
            adapter.decode_velocity(latents, state=torch.randn(1, 8))

    def test_direct_decoder_accepts_per_action_state_sequence(self):
        adapter = H3ActionAdapter(
            action_dim=7,
            state_dim=8,
            context_dim=16,
            direct_conditioning=True,
        )
        latents = torch.randn(2, 32, 2, 4)
        state = torch.randn(2, 4, 8)
        context = torch.randn(2, 3, 16)

        output = adapter.decode_velocity(latents, state=state, context=context)

        self.assertEqual(tuple(output.shape), (2, 4, 7))

    def test_direct_action_residual_receives_gradients(self):
        adapter = H3ActionAdapter(
            action_dim=7,
            state_dim=8,
            context_dim=16,
            direct_conditioning=True,
            direct_action_residual=True,
        )
        latents = torch.randn(2, 32, 2, 4)
        state = torch.randn(2, 4, 8)
        context = torch.randn(2, 3, 16)

        output = adapter.decode_velocity(latents, state=state, context=context)
        output.square().mean().backward()

        self.assertIsNotNone(adapter.decoder_action_residual[-1].weight.grad)

    def test_bridge_freezes_h3_but_backpropagates_to_adapter(self):
        h3 = _FakeH3()
        adapter = H3ActionAdapter(action_dim=14)
        bridge = H3ActionBridge(h3, adapter, freeze_h3=True)
        bridge.train()

        output = bridge(
            video_latents=torch.randn(1, 24, 2, 4, 4),
            noisy_actions=torch.randn(1, 32, 14),
            timestep=torch.tensor([500.0]),
            context=torch.randn(1, 8, 16),
        )
        loss = output.action_velocity.square().mean()
        loss.backward()

        self.assertEqual(output.action_velocity.shape, (1, 32, 14))
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in h3.parameters()))
        self.assertTrue(all(not p.requires_grad for p in h3.parameters()))

    def test_invalid_action_dimension_is_rejected(self):
        adapter = H3ActionAdapter(action_dim=14)
        with self.assertRaisesRegex(ValueError, "expected action_dim=14"):
            adapter.encode_actions(torch.randn(1, 32, 7))

    def test_first_frame_payload_matches_fl2va_contract(self):
        latents = torch.randn(1, 24, 2, 24, 20)
        payload = make_first_frame_payload(latents, frame_count=22, seed=7)

        self.assertEqual(payload["frame_count"], 22)
        self.assertEqual(payload["seed"], 7)
        self.assertIs(payload["cond_video_latents"][0], latents)
        self.assertEqual(payload["keyframes"][0]["resolved_frame_index"], 0)

    def test_invalid_h3_frame_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"17n\+5"):
            make_first_frame_payload(torch.randn(1, 24, 2, 4, 4), frame_count=9)


if __name__ == "__main__":
    unittest.main()
