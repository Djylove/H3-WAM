"""One-way paired H3 video / ActionDiT layer used by H3-DreamWAM."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .action_expert import H3DreamActionBlock, _attention


def h3_attention_input(
    block: nn.Module,
    hidden: torch.Tensor,
    *,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    apply_rotary: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        block.adaln_proj(temb)
    )
    residual = hidden
    normalized = block.norm1(hidden)
    normalized = normalized * (
        1.0 + scale_attn.index_select(0, adaln_indices)
    ) + shift_attn.index_select(0, adaln_indices)
    batch, sequence, _ = normalized.shape
    attention = block.attn
    shape = (batch, sequence, attention.heads, attention.head_dim)
    query = attention.norm_q(attention.to_q(normalized).reshape(shape))
    key = attention.norm_k(attention.to_k(normalized).reshape(shape))
    value = attention.to_v(normalized).reshape(shape)
    query = apply_rotary(query, *rotary_emb)
    key = apply_rotary(key, *rotary_emb)
    return (
        query,
        key,
        value,
        residual,
        gate_attn,
        shift_ffn,
        scale_ffn,
        gate_ffn,
    )


def h3_post_attention(
    block: nn.Module,
    *,
    attended: torch.Tensor,
    residual: torch.Tensor,
    gate_attn: torch.Tensor,
    shift_ffn: torch.Tensor,
    scale_ffn: torch.Tensor,
    gate_ffn: torch.Tensor,
    adaln_indices: torch.Tensor,
) -> torch.Tensor:
    attention_output = block.attn.to_out[0](attended.flatten(2))
    attention_output = block.attn.to_out[1](attention_output)
    hidden = residual + gate_attn.index_select(0, adaln_indices) * attention_output
    residual = hidden
    normalized = block.norm2(hidden)
    normalized = normalized * (
        1.0 + scale_ffn.index_select(0, adaln_indices)
    ) + shift_ffn.index_select(0, adaln_indices)
    return residual + gate_ffn.index_select(0, adaln_indices) * block.ff(normalized)


def paired_h3_action_layer(
    *,
    h3_block: nn.Module,
    action_block: H3DreamActionBlock,
    h3_hidden: torch.Tensor,
    action_hidden: torch.Tensor,
    h3_temb: torch.Tensor,
    h3_adaln_indices: torch.Tensor,
    h3_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    h3_apply_rotary: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    video_indices: torch.Tensor,
    action_time_modulation: torch.Tensor,
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor | None,
    h3_attention_mask: torch.Tensor | None = None,
    action_to_video_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one paired H3/ActionDiT layer.

    The inherited path is video-to-action only.  Supplying
    ``action_to_video_indices`` enables a zero-initialized reverse route for
    explicitly selected future-video rows.  Observation/text/audio rows stay
    isolated from action targets, which is the minimum causal contract needed
    before testing LingBot-VA-style bidirectional stream training.
    """

    h3_io = h3_attention_input(
        h3_block,
        h3_hidden,
        temb=h3_temb,
        adaln_indices=h3_adaln_indices,
        rotary_emb=h3_rotary_emb,
        apply_rotary=h3_apply_rotary,
    )
    h3_attended = _attention(h3_io[0], h3_io[1], h3_io[2], h3_attention_mask)

    video_indices = video_indices.to(device=h3_hidden.device, dtype=torch.long).reshape(-1)
    if video_indices.numel() == 0:
        raise ValueError("ActionDiT requires at least one H3 video row")
    if int(video_indices.min()) < 0 or int(video_indices.max()) >= h3_hidden.shape[1]:
        raise ValueError("video index is outside the packed H3 sequence")
    action_io = action_block.attention_input(action_hidden, action_time_modulation)
    if action_to_video_indices is not None:
        action_to_video_indices = action_to_video_indices.to(
            device=h3_hidden.device, dtype=torch.long
        ).reshape(-1)
        if action_to_video_indices.numel() == 0:
            raise ValueError("bidirectional fusion requires future-video rows")
        if (
            int(action_to_video_indices.min()) < 0
            or int(action_to_video_indices.max()) >= h3_hidden.shape[1]
        ):
            raise ValueError("action-to-video index is outside the packed H3 sequence")
        if h3_io[0].shape[2:] != action_io[1].shape[2:]:
            raise ValueError(
                "bidirectional fusion requires matching video/action attention geometry"
            )
        reverse_attended = _attention(
            h3_io[0].index_select(1, action_to_video_indices),
            action_io[1],
            action_io[2],
        )
        reverse_gate = torch.tanh(action_block.action_to_video_gate).to(
            dtype=reverse_attended.dtype,
            device=reverse_attended.device,
        )
        h3_attended = h3_attended.index_add(
            1,
            action_to_video_indices,
            reverse_gate * reverse_attended,
        )

    next_h3 = h3_post_attention(
        h3_block,
        attended=h3_attended,
        residual=h3_io[3],
        gate_attn=h3_io[4],
        shift_ffn=h3_io[5],
        scale_ffn=h3_io[6],
        gate_ffn=h3_io[7],
        adaln_indices=h3_adaln_indices,
    )
    video_key = h3_io[1].index_select(1, video_indices)
    video_value = h3_io[2].index_select(1, video_indices)
    action_key = torch.cat((video_key, action_io[1]), dim=1)
    action_value = torch.cat((video_value, action_io[2]), dim=1)
    action_attended = _attention(action_io[0], action_key, action_value)
    # Add an independently normalized video-only residual. Its per-head gate
    # starts at zero, preserving the old joint-attention checkpoint exactly,
    # and is warmed up before ActionDiT body parameters are unfrozen.
    video_attended = _attention(action_io[0], video_key, video_value)
    video_gate = torch.tanh(action_block.video_residual_gate).to(
        dtype=video_attended.dtype,
        device=video_attended.device,
    )
    action_attended = action_attended + video_gate * video_attended
    video_residual = action_block.video_residual_adapter(video_attended.flatten(2))
    next_action = action_block.post_attention(
        residual=action_io[3],
        attended=action_attended,
        gate_attn=action_io[4],
        shift_ffn=action_io[5],
        scale_ffn=action_io[6],
        gate_ffn=action_io[7],
        video_residual=video_residual,
        context=action_context,
        context_mask=action_context_mask,
    )
    return next_h3, next_action
