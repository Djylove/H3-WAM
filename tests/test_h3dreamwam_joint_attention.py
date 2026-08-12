import unittest

import torch
import torch.nn.functional as F
from torch import nn

from fastwam.models.h3dreamwam import (
    H3DreamActionBlock,
    align_h3_action_chunk_ids,
    build_lingbot_block_causal_mask,
    four_stream_h3_action_layer,
    lingbot_four_stream_attention,
    load_action_block_state,
    paired_h3_action_layer,
)
from fastwam.models.h3wam import build_h3_observation_attention_mask


def identity_rotary(
    hidden: torch.Tensor, _cos: torch.Tensor, _sin: torch.Tensor
) -> torch.Tensor:
    return hidden


class TinyAdaLN(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(time_dim, 18 * hidden_dim)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        projected = self.linear(F.silu(temb)).reshape(-1, 6 * self.hidden_dim)
        return projected.chunk(6, dim=-1)


class TinyH3Attention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        width = heads * head_dim
        self.to_q = nn.Linear(hidden_dim, width, bias=False)
        self.to_k = nn.Linear(hidden_dim, width, bias=False)
        self.to_v = nn.Linear(hidden_dim, width, bias=False)
        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)
        self.to_out = nn.ModuleList(
            [nn.Linear(width, hidden_dim, bias=False), nn.Dropout(0.0)]
        )

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        shape = (batch, sequence, self.heads, self.head_dim)
        query = self.norm_q(self.to_q(hidden).reshape(shape)).transpose(1, 2)
        key = self.norm_k(self.to_k(hidden).reshape(shape)).transpose(1, 2)
        value = self.to_v(hidden).reshape(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
        attended = attended.transpose(1, 2).flatten(2)
        return self.to_out[1](self.to_out[0](attended))


class TinyH3Block(nn.Module):
    def __init__(self, hidden_dim: int = 8, heads: int = 2, head_dim: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = TinyH3Attention(hidden_dim, heads, head_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.SiLU(), nn.Linear(16, hidden_dim)
        )
        self.adaln_proj = TinyAdaLN(4, hidden_dim)

    def forward_reference(
        self,
        hidden: torch.Tensor,
        temb: torch.Tensor,
        indices: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_proj(temb)
        residual = hidden
        normalized = self.norm1(hidden)
        normalized = normalized * (1 + scale_a.index_select(0, indices))
        normalized = normalized + shift_a.index_select(0, indices)
        hidden = residual + gate_a.index_select(0, indices) * self.attn(
            normalized, attention_mask
        )
        residual = hidden
        normalized = self.norm2(hidden)
        normalized = normalized * (1 + scale_f.index_select(0, indices))
        normalized = normalized + shift_f.index_select(0, indices)
        return residual + gate_f.index_select(0, indices) * self.ff(normalized)


class PairedAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.h3 = TinyH3Block()
        self.action = H3DreamActionBlock(
            hidden_dim=8, ffn_dim=16, num_heads=2, head_dim=4
        )
        self.h3_hidden = torch.randn(1, 6, 8)
        self.action_hidden = torch.randn(1, 3, 8)
        self.temb = torch.randn(1, 4)
        self.indices = torch.tensor([0, 1, 2, 0, 1, 2])
        self.rotary = (torch.empty(0), torch.empty(0))
        self.action_time = torch.randn(1, 6, 8)
        self.context = torch.randn(1, 2, 8)
        self.context_mask = torch.ones(1, 2, dtype=torch.bool)
        self.video_indices = torch.tensor([0, 2, 3, 5])

    def paired(
        self,
        h3_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        h3_attention_mask: torch.Tensor | None = None,
        action_to_video_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return paired_h3_action_layer(
            h3_block=self.h3,
            action_block=self.action,
            h3_hidden=h3_hidden,
            action_hidden=action_hidden,
            h3_temb=self.temb,
            h3_adaln_indices=self.indices,
            h3_rotary_emb=self.rotary,
            h3_apply_rotary=identity_rotary,
            video_indices=self.video_indices,
            action_time_modulation=self.action_time,
            action_context=self.context,
            action_context_mask=self.context_mask,
            h3_attention_mask=h3_attention_mask,
            action_to_video_indices=action_to_video_indices,
        )

    def test_video_matches_standalone_h3_and_ignores_action(self) -> None:
        expected = self.h3.forward_reference(
            self.h3_hidden, self.temb, self.indices, None
        )
        actual, _ = self.paired(self.h3_hidden, self.action_hidden)
        changed, _ = self.paired(self.h3_hidden, self.action_hidden + 100.0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(changed, actual, rtol=0, atol=0)

    def test_action_loss_updates_h3_but_video_loss_does_not_update_action(self) -> None:
        h3_hidden = self.h3_hidden.clone().requires_grad_(True)
        next_h3, next_action = self.paired(h3_hidden, self.action_hidden)
        next_action.square().mean().backward(retain_graph=True)
        self.assertIsNotNone(h3_hidden.grad)
        self.assertGreater(float(h3_hidden.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.h3.attn.to_k.weight.grad.abs().sum()), 0.0)

        self.action.zero_grad(set_to_none=True)
        self.h3.zero_grad(set_to_none=True)
        next_h3.sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in self.action.parameters())
        )

    def test_observation_rows_do_not_read_future_targets(self) -> None:
        mask = build_h3_observation_attention_mask(
            sequence_length=6,
            text_indices=torch.tensor([0]),
            condition_video_indices=torch.tensor([1]),
            device="cpu",
        )
        base, _ = self.paired(self.h3_hidden, self.action_hidden, mask)
        changed_hidden = self.h3_hidden.clone()
        changed_hidden[:, 2:] += 50.0
        changed, _ = self.paired(changed_hidden, self.action_hidden, mask)
        torch.testing.assert_close(base[:, :2], changed[:, :2], rtol=1e-5, atol=1e-5)

    def test_exact_dreamwam_norm_covers_the_full_attention_width(self) -> None:
        block = H3DreamActionBlock(
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            head_dim=4,
            full_width_rmsnorm=True,
        )
        self.assertEqual(block.attn.norm_q.weight.shape, (8,))
        self.assertEqual(block.cross_attn.norm_k.weight.shape, (8,))
        hidden = torch.randn(1, 3, 8)
        query, key, value = block.attn.qkv(hidden)
        self.assertEqual(query.shape, (1, 3, 2, 4))
        self.assertEqual(key.shape, query.shape)
        self.assertEqual(value.shape, query.shape)

    def test_video_residual_gate_starts_at_zero_and_receives_gradient(self) -> None:
        self.assertEqual(
            torch.count_nonzero(self.action.video_residual_gate),
            0,
        )
        _, baseline = self.paired(self.h3_hidden, self.action_hidden)
        baseline.square().mean().backward()
        gate_grad = self.action.video_residual_gate.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(float(gate_grad.abs().sum()), 0.0)

        self.action.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.action.video_residual_gate.fill_(0.5)
        _, gated = self.paired(self.h3_hidden, self.action_hidden)
        self.assertGreater(float((gated - baseline).detach().abs().max()), 0.0)

    def test_video_residual_adapter_is_function_preserving_and_trainable(self) -> None:
        output = self.action.video_residual_adapter[-1]
        self.assertEqual(torch.count_nonzero(output.weight), 0)
        _, baseline = self.paired(self.h3_hidden, self.action_hidden)
        baseline.square().mean().backward()
        self.assertIsNotNone(output.weight.grad)
        self.assertGreater(float(output.weight.grad.abs().sum()), 0.0)

        self.action.zero_grad(set_to_none=True)
        with torch.no_grad():
            output.weight.fill_(0.01)
        _, adapted = self.paired(self.h3_hidden, self.action_hidden)
        self.assertGreater(float((adapted - baseline).detach().abs().max()), 0.0)

    def test_legacy_action_block_checkpoint_migrates_zero_gate(self) -> None:
        legacy = {
            key: value.detach().clone()
            for key, value in self.action.state_dict().items()
            if key != "video_residual_gate"
            and key != "action_to_video_gate"
            and not key.startswith("video_residual_adapter.")
        }
        with torch.no_grad():
            self.action.video_residual_gate.fill_(0.75)
        migrated = load_action_block_state(self.action, legacy)
        self.assertTrue(migrated)
        self.assertEqual(
            torch.count_nonzero(self.action.video_residual_gate),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(self.action.video_residual_adapter[-1].weight),
            0,
        )

    def test_bidirectional_route_is_function_preserving_at_zero(self) -> None:
        baseline_h3, baseline_action = self.paired(
            self.h3_hidden, self.action_hidden
        )
        routed_h3, routed_action = self.paired(
            self.h3_hidden,
            self.action_hidden,
            action_to_video_indices=torch.tensor([2, 3, 4, 5]),
        )
        torch.testing.assert_close(routed_h3, baseline_h3, rtol=0, atol=0)
        torch.testing.assert_close(routed_action, baseline_action, rtol=0, atol=0)

    def test_video_loss_trains_reverse_gate_without_observation_leakage(self) -> None:
        target_indices = torch.tensor([2, 3, 4, 5])
        routed_h3, _ = self.paired(
            self.h3_hidden,
            self.action_hidden,
            action_to_video_indices=target_indices,
        )
        routed_h3.index_select(1, target_indices).square().mean().backward()
        gate_grad = self.action.action_to_video_gate.grad
        self.assertIsNotNone(gate_grad)
        self.assertGreater(float(gate_grad.abs().sum()), 0.0)

        self.action.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.action.action_to_video_gate.fill_(0.5)
        baseline_h3, _ = self.paired(self.h3_hidden, self.action_hidden)
        routed_h3, _ = self.paired(
            self.h3_hidden,
            self.action_hidden + 3.0,
            action_to_video_indices=target_indices,
        )
        torch.testing.assert_close(
            routed_h3[:, :2], baseline_h3[:, :2], rtol=0, atol=0
        )
        self.assertGreater(
            float((routed_h3[:, 2:] - baseline_h3[:, 2:]).abs().max()), 0.0
        )


class LingBotBlockCausalMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        # One token per stream/chunk makes the upstream four-stream contract
        # directly inspectable. Order: NV0,NV1,CV0,CV1,NA0,NA1,CA0,CA1.
        self.mask = build_lingbot_block_causal_mask(
            video_chunk_ids=torch.tensor([0, 1]),
            action_chunk_ids=torch.tensor([0, 1]),
        )[0, 0]

    def test_noisy_action_reads_same_chunk_clean_video_not_future(self) -> None:
        noisy_action_0 = 4
        clean_video_0 = 2
        clean_video_1 = 3
        self.assertTrue(self.mask[noisy_action_0, clean_video_0])
        self.assertFalse(self.mask[noisy_action_0, clean_video_1])

    def test_noisy_future_video_reads_previous_clean_action(self) -> None:
        noisy_video_1 = 1
        clean_action_0 = 6
        clean_action_1 = 7
        noisy_action_0 = 4
        self.assertTrue(self.mask[noisy_video_1, clean_action_0])
        self.assertFalse(self.mask[noisy_video_1, clean_action_1])
        self.assertFalse(self.mask[noisy_video_1, noisy_action_0])

    def test_noisy_streams_only_self_attend_inside_their_slot(self) -> None:
        self.assertTrue(self.mask[0, 0])
        self.assertFalse(self.mask[0, 1])
        self.assertTrue(self.mask[4, 4])
        self.assertFalse(self.mask[4, 5])

    def test_clean_stream_is_causal_and_window_is_in_slot_units(self) -> None:
        clean_action_1 = 7
        clean_video_0 = 2
        clean_video_1 = 3
        self.assertTrue(self.mask[clean_action_1, clean_video_0])
        self.assertTrue(self.mask[clean_action_1, clean_video_1])
        windowed = build_lingbot_block_causal_mask(
            video_chunk_ids=torch.tensor([0, 1]),
            action_chunk_ids=torch.tensor([0, 1]),
            window_size=1,
        )[0, 0]
        self.assertFalse(windowed[clean_action_1, clean_video_0])
        self.assertTrue(windowed[clean_action_1, clean_video_1])

    def test_invalid_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            build_lingbot_block_causal_mask(
                video_chunk_ids=torch.tensor([]),
                action_chunk_ids=torch.tensor([0]),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_lingbot_block_causal_mask(
                video_chunk_ids=torch.tensor([-1]),
                action_chunk_ids=torch.tensor([0]),
            )

    def test_h3_latent_frames_align_to_32_action_time_axis(self) -> None:
        # Mirror the cached H3 layout: 12 latent frames, 98 spatial tokens per
        # frame, versus 32 LIBERO actions. This cannot be represented by a
        # fixed integer actions-per-latent-frame reshape.
        frame_ids = torch.arange(12).repeat_interleave(98).float()
        video_chunks, action_chunks = align_h3_action_chunk_ids(
            video_frame_ids=frame_ids,
            action_horizon=32,
            actions_per_chunk=4,
        )
        self.assertEqual(video_chunks.shape, (12 * 98,))
        self.assertEqual(action_chunks.shape, (32,))
        self.assertEqual(torch.unique(video_chunks).tolist(), list(range(8)))
        self.assertEqual(torch.unique(action_chunks).tolist(), list(range(8)))
        for frame in range(12):
            self.assertEqual(
                torch.unique(video_chunks[frame * 98 : (frame + 1) * 98]).numel(),
                1,
            )
        self.assertTrue(torch.all(video_chunks[1:] >= video_chunks[:-1]))
        self.assertTrue(torch.all(action_chunks[1:] >= action_chunks[:-1]))

    def test_chunk_alignment_handles_non_divisible_horizon(self) -> None:
        video_chunks, action_chunks = align_h3_action_chunk_ids(
            video_frame_ids=torch.tensor([10.0, 20.0, 30.0]),
            action_horizon=10,
            actions_per_chunk=4,
        )
        self.assertEqual(video_chunks.tolist(), [0, 1, 2])
        self.assertEqual(action_chunks.tolist(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    def test_four_stream_attention_respects_teacher_forcing_boundary(self) -> None:
        def qkv(values: list[float]) -> tuple[torch.Tensor, ...]:
            value = torch.tensor(values).reshape(1, -1, 1, 1)
            query = torch.zeros_like(value)
            key = torch.zeros_like(value)
            return query, key, value

        streams = {
            "noisy_video_qkv": qkv([10.0, 11.0]),
            "clean_video_qkv": qkv([20.0, 21.0]),
            "noisy_action_qkv": qkv([30.0, 31.0]),
            "clean_action_qkv": qkv([40.0, 41.0]),
        }
        noisy_video, _, noisy_action, _ = lingbot_four_stream_attention(
            **streams, attention_mask=self.mask[None, None]
        )
        # NV1 sees its same-slot noisy peer (11) and the earlier clean
        # streams CV0 (20) + CA0 (40), never current/future clean rows.
        torch.testing.assert_close(
            noisy_video[0, 1, 0, 0], torch.tensor((11.0 + 20.0 + 40.0) / 3)
        )
        # NA0 sees itself (30) and current-chunk CV0 (20); CA0 is the target
        # for that noisy action and is intentionally hidden.
        torch.testing.assert_close(
            noisy_action[0, 0, 0, 0], torch.tensor((30.0 + 20.0) / 2)
        )

    def test_four_stream_geometry_mismatch_is_rejected(self) -> None:
        good = torch.zeros(1, 2, 1, 1)
        bad = torch.zeros(1, 2, 2, 1)
        with self.assertRaisesRegex(ValueError, "batch/head geometry"):
            lingbot_four_stream_attention(
                noisy_video_qkv=(good, good, good),
                clean_video_qkv=(good, good, good),
                noisy_action_qkv=(bad, bad, bad),
                clean_action_qkv=(good, good, good),
                attention_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool),
            )

    def test_read_only_context_is_visible_to_every_query(self) -> None:
        def qkv(values: list[float]) -> tuple[torch.Tensor, ...]:
            value = torch.tensor(values).reshape(1, -1, 1, 1)
            return torch.zeros_like(value), torch.zeros_like(value), value

        zero = qkv([0.0, 0.0])
        outputs = lingbot_four_stream_attention(
            noisy_video_qkv=zero,
            clean_video_qkv=zero,
            noisy_action_qkv=zero,
            clean_action_qkv=zero,
            attention_mask=self.mask[None, None],
            context_key_value=(
                torch.zeros(1, 1, 1, 1),
                torch.full((1, 1, 1, 1), 9.0),
            ),
        )
        self.assertTrue(all(torch.all(output > 0) for output in outputs))


class FourStreamLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.h3 = TinyH3Block()
        self.action = H3DreamActionBlock(
            hidden_dim=8, ffn_dim=16, num_heads=2, head_dim=4
        )
        self.noisy_video = torch.randn(1, 4, 8)
        self.clean_video = torch.randn(1, 4, 8)
        self.noisy_action = torch.randn(1, 4, 8)
        self.clean_action = torch.randn(1, 4, 8)
        self.video_chunks = torch.tensor([0, 0, 1, 1])
        self.action_chunks = torch.tensor([0, 0, 1, 1])
        self.temb = torch.randn(1, 4)
        self.indices = torch.zeros(4, dtype=torch.long)
        self.action_time = torch.randn(1, 6, 8)
        self.context = torch.randn(1, 2, 8)
        self.context_mask = torch.ones(1, 2, dtype=torch.bool)

    def run_layer(self) -> tuple[torch.Tensor, ...]:
        return four_stream_h3_action_layer(
            h3_block=self.h3,
            action_block=self.action,
            noisy_video_hidden=self.noisy_video,
            clean_video_hidden=self.clean_video,
            noisy_action_hidden=self.noisy_action,
            clean_action_hidden=self.clean_action,
            noisy_h3_temb=self.temb,
            clean_h3_temb=self.temb,
            noisy_h3_adaln_indices=self.indices,
            clean_h3_adaln_indices=self.indices,
            h3_rotary_emb=(torch.empty(0), torch.empty(0)),
            h3_apply_rotary=identity_rotary,
            noisy_action_time_modulation=self.action_time,
            clean_action_time_modulation=self.action_time,
            action_context=self.context,
            action_context_mask=self.context_mask,
            video_chunk_ids=self.video_chunks,
            action_chunk_ids=self.action_chunks,
        )

    def test_layer_preserves_stream_shapes(self) -> None:
        outputs = self.run_layer()
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(1, 4, 8)] * 4,
        )

    def test_video_objective_has_direct_action_expert_gradient(self) -> None:
        noisy_video, _, _, _ = self.run_layer()
        # Chunk 1 video reads chunk 0 clean-action tokens through the official
        # strict-causal teacher-forcing route.
        noisy_video[:, 2:].square().mean().backward()
        grad = self.action.attn.to_v.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_action_objective_has_direct_h3_gradient(self) -> None:
        _, _, noisy_action, _ = self.run_layer()
        noisy_action.square().mean().backward()
        grad = self.h3.attn.to_v.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_h3_context_metadata_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires temb"):
            four_stream_h3_action_layer(
                h3_block=self.h3,
                action_block=self.action,
                noisy_video_hidden=self.noisy_video,
                clean_video_hidden=self.clean_video,
                noisy_action_hidden=self.noisy_action,
                clean_action_hidden=self.clean_action,
                noisy_h3_temb=self.temb,
                clean_h3_temb=self.temb,
                noisy_h3_adaln_indices=self.indices,
                clean_h3_adaln_indices=self.indices,
                h3_rotary_emb=(torch.empty(0), torch.empty(0)),
                h3_apply_rotary=identity_rotary,
                noisy_action_time_modulation=self.action_time,
                clean_action_time_modulation=self.action_time,
                action_context=self.context,
                action_context_mask=self.context_mask,
                video_chunk_ids=self.video_chunks,
                action_chunk_ids=self.action_chunks,
                h3_context_hidden=torch.randn(1, 2, 8),
            )


if __name__ == "__main__":
    unittest.main()
