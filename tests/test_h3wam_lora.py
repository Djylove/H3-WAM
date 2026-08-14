import unittest

import torch
from torch import nn

from fastwam.models.h3wam import (
    H3LoRALinear,
    h3_lora_disabled,
    h3_lora_parameters,
    h3_lora_state_dict,
    inject_h3_attention_lora,
    load_h3_lora_state_dict,
)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(8, 24, bias=False)
        self.out_proj = nn.Linear(8, 8, bias=False)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 16, bias=False)
        self.mlp.fc2 = nn.Linear(16, 8, bias=False)


class FakeH3(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])


class FusedSwiGLUBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 4
        self.out_features = 3
        self.weight = nn.Parameter(torch.randn(3, 4))

    def forward(self, hidden, *, input_act=None):
        if input_act == "swiglu":
            gate, up = hidden.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(hidden, self.weight)


class H3LoRATest(unittest.TestCase):
    def test_injection_wraps_selected_attention_modules(self):
        model = FakeH3().requires_grad_(False)
        original = model.blocks[-1].attn.qkv_proj
        sample = torch.randn(3, 8)
        expected = original(sample)

        report = inject_h3_attention_lora(model, rank=2, last_n_blocks=1)

        self.assertEqual(report.blocks, 1)
        self.assertEqual(report.modules, 2)
        self.assertIsInstance(model.blocks[-1].attn.qkv_proj, H3LoRALinear)
        self.assertNotIsInstance(model.blocks[0].attn.qkv_proj, H3LoRALinear)
        torch.testing.assert_close(model.blocks[-1].attn.qkv_proj(sample), expected)

    def test_only_lora_parameters_receive_gradients(self):
        model = FakeH3().requires_grad_(False)
        inject_h3_attention_lora(model, rank=2)
        output = model.blocks[0].attn.qkv_proj(torch.randn(3, 8)).square().mean()
        output.backward()
        parameters = h3_lora_parameters(model)
        self.assertTrue(parameters)
        self.assertTrue(any(parameter.grad is not None for parameter in parameters))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.blocks[0].attn.qkv_proj.base.parameters()))
        self.assertEqual(len(h3_lora_state_dict(model)), 8)

    def test_optional_mlp_injection_and_temporary_disable(self):
        model = FakeH3().requires_grad_(False)
        sample = torch.randn(3, 8)
        base_expected = model.blocks[-1].mlp.fc1(sample)
        report = inject_h3_attention_lora(
            model, rank=2, last_n_blocks=1, include_mlp=True
        )

        self.assertEqual(report.modules, 4)
        self.assertIsInstance(model.blocks[-1].mlp.fc1, H3LoRALinear)
        with torch.no_grad():
            model.blocks[-1].mlp.fc1.lora_b.weight.fill_(0.25)
        adapted = model.blocks[-1].mlp.fc1(sample)
        with h3_lora_disabled(model):
            disabled = model.blocks[-1].mlp.fc1(sample)
        torch.testing.assert_close(disabled, base_expected)
        self.assertFalse(torch.equal(adapted, disabled))
        self.assertTrue(model.blocks[-1].mlp.fc1.enabled)

    def test_sparse_loader_copies_only_lora_parameters(self):
        source = FakeH3().requires_grad_(False)
        target = FakeH3().requires_grad_(False)
        inject_h3_attention_lora(source, rank=2)
        inject_h3_attention_lora(target, rank=2)
        with torch.no_grad():
            for parameter in h3_lora_parameters(source):
                parameter.fill_(0.25)
        base_before = target.blocks[0].attn.qkv_proj.base.weight.detach().clone()

        loaded = load_h3_lora_state_dict(target, h3_lora_state_dict(source))

        self.assertEqual(loaded, 8)
        for parameter in h3_lora_parameters(target):
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))
        torch.testing.assert_close(
            target.blocks[0].attn.qkv_proj.base.weight,
            base_before,
        )

    def test_sparse_loader_rejects_incomplete_checkpoint(self):
        model = FakeH3().requires_grad_(False)
        inject_h3_attention_lora(model, rank=2)
        state = h3_lora_state_dict(model)
        state.pop(next(iter(state)))
        with self.assertRaises(ValueError):
            load_h3_lora_state_dict(model, state)

    def test_lora_supports_native_fused_swiglu_input(self):
        base = FusedSwiGLUBase()
        layer = H3LoRALinear(base, rank=2, alpha=2)
        source = torch.randn(5, 8)
        expected = base(source, input_act="swiglu")
        torch.testing.assert_close(layer(source, input_act="swiglu"), expected)
        with torch.no_grad():
            layer.lora_b.weight.fill_(0.25)
        self.assertFalse(
            torch.equal(layer(source, input_act="swiglu"), expected)
        )


if __name__ == "__main__":
    unittest.main()
