#!/usr/bin/env python3
"""Verify H3-DreamWAM primitives against real Diffusers H3 modules."""

from __future__ import annotations

import json

import torch

from fastwam.models.h3dreamwam import (
    H3DreamActionBlock,
    expand_h3_rgb_flow_projections,
    paired_h3_action_layer,
)


def main() -> None:
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerBlock,
        _apply_rotary_emb,
    )

    torch.manual_seed(2026)
    model = MiniMaxH3Transformer3DModel(
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
    rgb = torch.randn(1, 5, 16)
    original_embedding = model.proj_in(rgb)
    report = expand_h3_rgb_flow_projections(
        model,
        flow_channels=4,
        generator=torch.Generator().manual_seed(17),
    )
    expanded_embedding = model.proj_in(torch.cat((rgb, torch.zeros_like(rgb)), -1))
    torch.testing.assert_close(
        expanded_embedding, original_embedding, rtol=1.0e-6, atol=1.0e-6
    )

    h3_block: MiniMaxH3TransformerBlock = model.transformer_blocks[0]
    action_block = H3DreamActionBlock(
        hidden_dim=24,
        ffn_dim=48,
        num_heads=2,
        head_dim=16,
    )
    h3_hidden = torch.randn(1, 7, 32, requires_grad=True)
    action_hidden = torch.randn(1, 4, 24, requires_grad=True)
    temb = torch.randn(1, 16)
    adaln_indices = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    cos = torch.ones(7, 8)
    sin = torch.zeros(7, 8)
    action_time = torch.randn(1, 6, 24)
    context = torch.randn(1, 3, 24)
    context_mask = torch.ones(1, 3, dtype=torch.bool)

    standalone = h3_block(
        h3_hidden,
        temb,
        adaln_indices,
        (cos, sin),
    )
    paired_video, paired_action = paired_h3_action_layer(
        h3_block=h3_block,
        action_block=action_block,
        h3_hidden=h3_hidden,
        action_hidden=action_hidden,
        h3_temb=temb,
        h3_adaln_indices=adaln_indices,
        h3_rotary_emb=(cos, sin),
        h3_apply_rotary=_apply_rotary_emb,
        video_indices=torch.tensor([0, 2, 4, 6]),
        action_time_modulation=action_time,
        action_context=context,
        action_context_mask=context_mask,
    )
    torch.testing.assert_close(
        paired_video, standalone, rtol=2.0e-5, atol=2.0e-6
    )
    paired_action.square().mean().backward()
    h3_key_grad = h3_block.attn.to_k.weight.grad
    if h3_key_grad is None or float(h3_key_grad.abs().sum()) == 0.0:
        raise RuntimeError("action loss did not reach H3 key projection")
    print(
        json.dumps(
            {
                "event": "h3dreamwam_primitive_smoke",
                "projection": report.__dict__,
                "standalone_pair_max_abs": float(
                    (paired_video.detach() - standalone.detach()).abs().max()
                ),
                "h3_key_gradient_l1": float(h3_key_grad.abs().sum()),
                "action_shape": list(paired_action.shape),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
