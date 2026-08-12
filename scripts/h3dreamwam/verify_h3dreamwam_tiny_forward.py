#!/usr/bin/env python3
"""End-to-end tiny-model verification of the H3-DreamWAM forward."""

from __future__ import annotations

import copy
import json

import torch

from fastwam.models.h3dreamwam import (
    H3DreamActionExpert,
    H3DreamWAM,
    expand_h3_rgb_flow_projections,
)


def tiny_h3():
    from diffusers import MiniMaxH3Transformer3DModel

    return MiniMaxH3Transformer3DModel(
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


def main() -> None:
    torch.manual_seed(2026)
    baseline = tiny_h3().eval()
    adapted_h3 = copy.deepcopy(baseline)
    expand_h3_rgb_flow_projections(
        adapted_h3,
        flow_channels=4,
        generator=torch.Generator().manual_seed(5),
    )
    action = H3DreamActionExpert(
        action_dim=7,
        state_dim=8,
        text_dim=32,
        hidden_dim=24,
        ffn_dim=48,
        num_heads=2,
        head_dim=16,
        num_layers=2,
        frequency_dim=16,
    )
    model = H3DreamWAM(
        adapted_h3,
        action,
        rgb_patch_width=16,
        use_gradient_checkpointing=True,
    ).eval()

    video_indices = torch.tensor([3, 4, 5, 6])
    text_indices = torch.tensor([0, 1, 2])
    audio_indices = torch.tensor([7, 8])
    token_tags = torch.tensor([1, 1, 1, 0, 0, 0, 0, 2, 2])
    timestep_indices = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 1])
    position_ids = torch.tensor(
        [
            [0, 0, 0], [0, 1, 0], [0, 2, 0],
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
            [0, 0, 0], [1, 0, 0],
        ]
    )
    rgb_rows = torch.randn(1, 4, 16)
    flow_rows = torch.zeros_like(rgb_rows)
    audio_rows = torch.randn(1, 2, 8)
    context = torch.randn(1, 3, 32)
    timestep = torch.tensor([0.4, 1.0])
    baseline_output = baseline(
        hidden_states=rgb_rows,
        audio_hidden_states=audio_rows,
        encoder_hidden_states=context,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=True,
    )
    adapted_output = model(
        video_rows=torch.cat((rgb_rows, flow_rows), dim=-1),
        audio_rows=audio_rows,
        context=context,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        noisy_actions=torch.randn(1, 5, 7),
        action_timestep=torch.tensor([500.0]),
        state=torch.randn(1, 8),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(
        adapted_output.rgb_velocity_rows,
        baseline_output.sample,
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    if torch.count_nonzero(adapted_output.flow_velocity_rows) != 0:
        raise RuntimeError("zero-initialized flow head produced a non-zero output")

    model.train()
    train_output = model(
        video_rows=torch.cat((rgb_rows, flow_rows), dim=-1),
        audio_rows=audio_rows,
        context=context,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        noisy_actions=torch.randn(1, 5, 7),
        action_timestep=torch.tensor([500.0]),
        state=torch.randn(1, 8),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    loss = (
        train_output.rgb_velocity_rows.square().mean()
        + train_output.flow_velocity_rows.square().mean()
        + train_output.action_velocity.square().mean()
    )
    loss.backward()
    action_grad = model.paired_layers[0].action_block.attn.to_q.weight.grad
    h3_grad = model.paired_layers[0].h3_block.attn.to_k.weight.grad
    if action_grad is None or h3_grad is None:
        raise RuntimeError("joint backward missed H3 or ActionDiT")
    print(
        json.dumps(
            {
                "event": "h3dreamwam_tiny_forward",
                "rgb_equivalence_max_abs": float(
                    (
                        adapted_output.rgb_velocity_rows
                        - baseline_output.sample
                    ).abs().max().detach()
                ),
                "flow_nonzero": int(
                    torch.count_nonzero(adapted_output.flow_velocity_rows)
                ),
                "loss": float(loss.detach()),
                "h3_gradient_l1": float(h3_grad.abs().sum()),
                "action_gradient_l1": float(action_grad.abs().sum()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
