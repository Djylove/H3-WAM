"""One-way paired H3 video / ActionDiT layer used by H3-DreamWAM."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .action_expert import H3DreamActionBlock, _attention


def build_lingbot_block_causal_mask(
    *,
    video_chunk_ids: torch.Tensor,
    action_chunk_ids: torch.Tensor,
    window_size: int | None = None,
) -> torch.Tensor:
    """Build the four-stream training mask used by LingBot-VA.

    Rows and columns are ordered as ``[noisy_video, clean_video,
    noisy_action, clean_action]``.  Video chunk ``c`` occupies causal slot
    ``2*c`` and action chunk ``c`` occupies slot ``2*c+1``.  Consequently a
    noisy action chunk may read the clean video from the same chunk, while a
    noisy video chunk may read only clean actions from earlier chunks.  This
    is the key teacher-forced action-to-future-video route; it deliberately
    excludes same-chunk noisy-action leakage.

    The returned boolean mask has SDPA shape ``[1,1,Q,K]`` where ``True``
    means that the key is visible.  ``window_size`` uses LingBot-VA's
    interleaved video/action slot units, not raw frame units.
    """

    video_chunk_ids = video_chunk_ids.reshape(-1).long()
    action_chunk_ids = action_chunk_ids.reshape(-1).long()
    if video_chunk_ids.numel() == 0 or action_chunk_ids.numel() == 0:
        raise ValueError("video/action chunk ids cannot be empty")
    if int(video_chunk_ids.min()) < 0 or int(action_chunk_ids.min()) < 0:
        raise ValueError("chunk ids must be non-negative")
    if window_size is not None and window_size < 0:
        raise ValueError("window_size must be non-negative")
    if action_chunk_ids.device != video_chunk_ids.device:
        action_chunk_ids = action_chunk_ids.to(video_chunk_ids.device)

    video_slots = video_chunk_ids * 2
    action_slots = action_chunk_ids * 2 + 1
    frame_ids = torch.cat(
        (video_slots, video_slots, action_slots, action_slots), dim=0
    )
    # This naming follows the upstream code: 0 is a noisy/predicted stream
    # and 1 is a clean teacher-forcing stream.
    noise_ids = torch.cat(
        (
            torch.zeros_like(video_slots),
            torch.ones_like(video_slots),
            torch.zeros_like(action_slots),
            torch.ones_like(action_slots),
        ),
        dim=0,
    )
    query_frames = frame_ids[:, None]
    key_frames = frame_ids[None, :]
    query_is_clean = noise_ids[:, None] == 1
    key_is_clean = noise_ids[None, :] == 1

    clean_to_clean = query_is_clean & key_is_clean & (key_frames <= query_frames)
    noisy_to_clean = (~query_is_clean) & key_is_clean & (
        key_frames < query_frames
    )
    noisy_to_noisy = (~query_is_clean) & (~key_is_clean) & (
        key_frames == query_frames
    )
    allowed = clean_to_clean | noisy_to_clean | noisy_to_noisy
    if window_size is not None:
        allowed &= (query_frames - key_frames).abs() <= int(window_size)
    return allowed[None, None]


def align_h3_action_chunk_ids(
    *,
    video_frame_ids: torch.Tensor,
    action_horizon: int,
    actions_per_chunk: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align H3 latent-frame tokens and robot actions on shared time chunks.

    LingBot-VA can reshape actions to a fixed number per Wan latent frame. H3
    has a different temporal VAE geometry (the current LIBERO cache contains
    12 latent frames for 32 actions), so copying that reshape would silently
    misalign supervision.  This function instead assigns monotonically
    ordered H3 latent frames and actions to the same fixed-duration action
    chunks.  Every spatial token from one latent frame receives the same id.
    """

    video_frame_ids = video_frame_ids.reshape(-1)
    if video_frame_ids.numel() == 0:
        raise ValueError("video frame ids cannot be empty")
    if action_horizon <= 0 or actions_per_chunk <= 0:
        raise ValueError("action horizon and actions_per_chunk must be positive")
    # H3 positions are floating point and may repeat for every spatial token.
    # sorted_unique plus searchsorted gives a stable ordinal without assuming
    # a particular positional scaling used by Diffusers.
    unique_frames = torch.unique(video_frame_ids, sorted=True)
    frame_ordinals = torch.searchsorted(unique_frames, video_frame_ids)
    num_chunks = (int(action_horizon) + int(actions_per_chunk) - 1) // int(
        actions_per_chunk
    )
    video_chunks = torch.div(
        frame_ordinals * num_chunks,
        unique_frames.numel(),
        rounding_mode="floor",
    ).clamp_max(num_chunks - 1)
    action_chunks = torch.div(
        torch.arange(action_horizon, device=video_frame_ids.device),
        int(actions_per_chunk),
        rounding_mode="floor",
    )
    return video_chunks.long(), action_chunks.long()


def lingbot_four_stream_attention(
    *,
    noisy_video_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    clean_video_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    noisy_action_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    clean_action_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    context_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one official-order four-stream attention operation.

    Video tensors may be projected by H3 while action tensors are projected by
    the action expert; only their ``[heads, head_dim]`` geometry must agree.
    Keeping the projections modality-specific while concatenating attention is
    the mixture-of-transformers boundary needed for an H3 backbone port.
    """

    streams = (
        noisy_video_qkv,
        clean_video_qkv,
        noisy_action_qkv,
        clean_action_qkv,
    )
    for stream in streams:
        if len(stream) != 3 or any(tensor.ndim != 4 for tensor in stream):
            raise ValueError("every stream must contain Q/K/V tensors [B,S,H,D]")
    batch_head_geometry = streams[0][0].shape[0], streams[0][0].shape[2:]
    for query, key, value in streams:
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Q/K/V shapes must match within each stream")
        if (query.shape[0], query.shape[2:]) != batch_head_geometry:
            raise ValueError("all streams must share batch/head geometry")

    lengths = [stream[0].shape[1] for stream in streams]
    total_length = sum(lengths)
    context_length = 0
    if context_key_value is not None:
        context_key, context_value = context_key_value
        if context_key.ndim != 4 or context_key.shape != context_value.shape:
            raise ValueError("context K/V must have matching [B,S,H,D] shapes")
        if (context_key.shape[0], context_key.shape[2:]) != batch_head_geometry:
            raise ValueError("context must share four-stream batch/head geometry")
        context_length = context_key.shape[1]
    if attention_mask.shape[-2:] != (total_length, total_length):
        raise ValueError("attention mask does not match the four-stream sequence")
    query = torch.cat([stream[0] for stream in streams], dim=1)
    key = torch.cat([stream[1] for stream in streams], dim=1)
    value = torch.cat([stream[2] for stream in streams], dim=1)
    if context_key_value is not None:
        key = torch.cat((context_key, key), dim=1)
        value = torch.cat((context_value, value), dim=1)
        context_columns = torch.ones(
            (*attention_mask.shape[:-1], context_length),
            device=attention_mask.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat((context_columns, attention_mask.bool()), dim=-1)
    attended = _attention(query, key, value, attention_mask)
    return tuple(attended.split(lengths, dim=1))


def four_stream_h3_action_layer(
    *,
    h3_block: nn.Module,
    action_block: H3DreamActionBlock,
    noisy_video_hidden: torch.Tensor,
    clean_video_hidden: torch.Tensor,
    noisy_action_hidden: torch.Tensor,
    clean_action_hidden: torch.Tensor,
    noisy_h3_temb: torch.Tensor,
    clean_h3_temb: torch.Tensor,
    noisy_h3_adaln_indices: torch.Tensor,
    clean_h3_adaln_indices: torch.Tensor,
    h3_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    h3_apply_rotary: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    noisy_action_time_modulation: torch.Tensor,
    clean_action_time_modulation: torch.Tensor,
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor | None,
    video_chunk_ids: torch.Tensor,
    action_chunk_ids: torch.Tensor,
    window_size: int | None = None,
    h3_context_hidden: torch.Tensor | None = None,
    h3_context_temb: torch.Tensor | None = None,
    h3_context_adaln_indices: torch.Tensor | None = None,
    h3_context_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one H3/action MoT layer with LingBot-VA's four streams.

    Unlike :func:`paired_h3_action_layer`, this is a direct joint attention
    operation rather than a gated residual.  H3 owns both video streams and
    ActionDiT owns both action streams; Q/K/V share attention geometry and are
    concatenated only inside attention.  The clean streams provide causal
    teacher forcing and are not prediction outputs.
    """

    if noisy_video_hidden.shape != clean_video_hidden.shape:
        raise ValueError("noisy/clean video hidden shapes must match")
    if noisy_action_hidden.shape != clean_action_hidden.shape:
        raise ValueError("noisy/clean action hidden shapes must match")
    if noisy_video_hidden.shape[1] != video_chunk_ids.numel():
        raise ValueError("video chunk ids must cover every video token")
    if noisy_action_hidden.shape[1] != action_chunk_ids.numel():
        raise ValueError("action chunk ids must cover every action token")

    noisy_video_io = h3_attention_input(
        h3_block,
        noisy_video_hidden,
        temb=noisy_h3_temb,
        adaln_indices=noisy_h3_adaln_indices,
        rotary_emb=h3_rotary_emb,
        apply_rotary=h3_apply_rotary,
    )
    clean_video_io = h3_attention_input(
        h3_block,
        clean_video_hidden,
        temb=clean_h3_temb,
        adaln_indices=clean_h3_adaln_indices,
        rotary_emb=h3_rotary_emb,
        apply_rotary=h3_apply_rotary,
    )
    noisy_action_io = action_block.attention_input(
        noisy_action_hidden, noisy_action_time_modulation
    )
    clean_action_io = action_block.attention_input(
        clean_action_hidden, clean_action_time_modulation
    )
    context_key_value = None
    context_arguments = (
        h3_context_temb,
        h3_context_adaln_indices,
        h3_context_rotary_emb,
    )
    if h3_context_hidden is None:
        if any(argument is not None for argument in context_arguments):
            raise ValueError("context metadata requires h3_context_hidden")
    else:
        if any(argument is None for argument in context_arguments):
            raise ValueError("H3 context requires temb, AdaLN indices and RoPE")
        context_io = h3_attention_input(
            h3_block,
            h3_context_hidden,
            temb=h3_context_temb,
            adaln_indices=h3_context_adaln_indices,
            rotary_emb=h3_context_rotary_emb,
            apply_rotary=h3_apply_rotary,
        )
        context_key_value = context_io[1:3]
    mask = build_lingbot_block_causal_mask(
        video_chunk_ids=video_chunk_ids.to(noisy_video_hidden.device),
        action_chunk_ids=action_chunk_ids.to(noisy_video_hidden.device),
        window_size=window_size,
    )
    attended = lingbot_four_stream_attention(
        noisy_video_qkv=noisy_video_io[:3],
        clean_video_qkv=clean_video_io[:3],
        noisy_action_qkv=noisy_action_io[:3],
        clean_action_qkv=clean_action_io[:3],
        attention_mask=mask,
        context_key_value=context_key_value,
    )

    def finish_video(io: tuple[torch.Tensor, ...], value: torch.Tensor) -> torch.Tensor:
        return h3_post_attention(
            h3_block,
            attended=value,
            residual=io[3],
            gate_attn=io[4],
            shift_ffn=io[5],
            scale_ffn=io[6],
            gate_ffn=io[7],
            adaln_indices=(
                noisy_h3_adaln_indices if io is noisy_video_io
                else clean_h3_adaln_indices
            ),
        )

    def finish_action(io: tuple[torch.Tensor, ...], value: torch.Tensor) -> torch.Tensor:
        return action_block.post_attention(
            residual=io[3],
            attended=value,
            gate_attn=io[4],
            shift_ffn=io[5],
            scale_ffn=io[6],
            gate_ffn=io[7],
            context=action_context,
            context_mask=action_context_mask,
        )

    return (
        finish_video(noisy_video_io, attended[0]),
        finish_video(clean_video_io, attended[1]),
        finish_action(noisy_action_io, attended[2]),
        finish_action(clean_action_io, attended[3]),
    )


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
