import unittest

import torch
import torch.nn.functional as F
from torch import nn

from fastwam.models.h3dreamwam import (
    H3DreamActionBlock,
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


if __name__ == "__main__":
    unittest.main()
