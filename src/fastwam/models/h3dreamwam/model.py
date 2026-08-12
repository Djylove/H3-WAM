"""Minimal RGB/flow + action H3-DreamWAM forward path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .action_expert import H3DreamActionExpert
from .joint_attention import paired_h3_action_layer


def apply_h3_rotary(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotated = hidden[..., :rotary_dim]
    passthrough = hidden[..., rotary_dim:]
    cos = cos.to(hidden.dtype)[None, :, None]
    sin = sin.to(hidden.dtype)[None, :, None]
    first, second = rotated.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return torch.cat((rotated * cos + rotated_half * sin, passthrough), dim=-1).contiguous()


@dataclass
class H3DreamWAMOutput:
    rgb_velocity_rows: torch.Tensor
    flow_velocity_rows: torch.Tensor
    action_velocity: torch.Tensor
    audio_velocity_rows: torch.Tensor


class H3DreamPairedLayer(nn.Module):
    """FSDP wrapping unit containing one H3 block and one ActionDiT block."""

    def __init__(self, h3_block: nn.Module, action_block: nn.Module) -> None:
        super().__init__()
        self.h3_block = h3_block
        self.action_block = action_block

    def forward(
        self,
        h3_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        *,
        h3_temb: torch.Tensor,
        h3_adaln_indices: torch.Tensor,
        h3_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        video_indices: torch.Tensor,
        action_time_modulation: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor | None,
        h3_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return paired_h3_action_layer(
            h3_block=self.h3_block,
            action_block=self.action_block,
            h3_hidden=h3_hidden,
            action_hidden=action_hidden,
            h3_temb=h3_temb,
            h3_adaln_indices=h3_adaln_indices,
            h3_rotary_emb=h3_rotary_emb,
            h3_apply_rotary=apply_h3_rotary,
            video_indices=video_indices,
            action_time_modulation=action_time_modulation,
            action_context=action_context,
            action_context_mask=action_context_mask,
            h3_attention_mask=h3_attention_mask,
        )


class H3DreamWAM(nn.Module):
    """Run H3 packed video and an independent ActionDiT as paired layers."""

    def __init__(
        self,
        h3: nn.Module,
        action_expert: H3DreamActionExpert,
        *,
        rgb_patch_width: int,
        use_gradient_checkpointing: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.h3 = h3
        self.action_expert = action_expert
        self.rgb_patch_width = int(rgb_patch_width)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.compute_dtype = compute_dtype
        if self.rgb_patch_width <= 0:
            raise ValueError("rgb_patch_width must be positive")
        if len(self.h3.transformer_blocks) != len(self.action_expert.blocks):
            raise ValueError("H3 and ActionDiT must have the same number of layers")
        h3_blocks = list(self.h3.transformer_blocks)
        action_blocks = list(self.action_expert.blocks)
        # Paired layers own the blocks so FSDP can gather/shard both experts in
        # one forward. The remaining H3/Action modules stay as lightweight I/O.
        self.h3.transformer_blocks = nn.ModuleList()
        self.action_expert.blocks = nn.ModuleList()
        self.paired_layers = nn.ModuleList(
            [
                H3DreamPairedLayer(h3_block, action_block)
                for h3_block, action_block in zip(
                    h3_blocks, action_blocks, strict=True
                )
            ]
        )
        projected_width = int(self.h3.proj_out.out_features)
        if projected_width != 2 * self.rgb_patch_width:
            raise ValueError(
                f"expanded H3 output width must be {2 * self.rgb_patch_width}, "
                f"got {projected_width}"
            )

    @staticmethod
    def _validate_layout(
        *,
        sequence_length: int,
        token_tags: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> None:
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (
            sequence_length,
        ):
            raise ValueError("token_tags/timestep_indices must match packed sequence")
        rows = torch.cat((video_indices, audio_indices, text_indices)).long()
        if rows.numel() != sequence_length or torch.unique(rows).numel() != sequence_length:
            raise ValueError("video/audio/text indices must partition the packed sequence")
        if int(rows.min()) != 0 or int(rows.max()) != sequence_length - 1:
            raise ValueError("packed modality indices must cover [0,sequence_length)")

    def forward(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        state: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action_video_indices: torch.Tensor | None = None,
        h3_attention_mask: torch.Tensor | None = None,
    ) -> H3DreamWAMOutput:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must be [packed_sequence,3]")
        sequence_length = position_ids.shape[0]
        self._validate_layout(
            sequence_length=sequence_length,
            token_tags=token_tags,
            timestep_indices=timestep_indices,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
        )
        if video_rows.shape[-1] != self.h3.proj_in.in_features:
            raise ValueError("video row width does not match expanded H3 proj_in")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        if action_video_indices is None:
            action_video_indices = video_indices

        # DreamWAM exposes proprioception to both experts through one shared
        # context token. Appending it only inside ActionDiT leaves the world
        # backbone unable to condition motion on the robot's current state.
        context, context_mask = self.action_expert.append_state_to_context(
            context=context,
            context_mask=context_mask,
            state=state,
        )
        if context.shape[1] != text_indices.numel():
            raise ValueError(
                "packed H3 layout must reserve one text row for proprio: "
                f"context={context.shape[1]}, text_indices={text_indices.numel()}"
            )

        rotary_emb = self.h3.rope(position_ids)
        video_embeds = self.h3.proj_in(video_rows.to(self.h3.proj_in.weight.dtype))
        audio_embeds = self.h3.audio_proj_in(
            audio_rows.to(self.h3.audio_proj_in.weight.dtype)
        )
        text_embeds = self.h3.context_embedder(
            context.to(self.h3.context_embedder.weight.dtype)
        )
        text_embeds = self.h3.token_refiner(text_embeds)
        hidden = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        hidden = hidden.index_copy(1, text_indices, text_embeds)
        hidden = hidden.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden = hidden.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.h3.time_proj(timestep)
        temb = self.h3.time_embedder(
            temb.to(self.h3.time_embedder.linear_1.weight.dtype)
        )
        adaln_indices = timestep_indices * 3 + token_tags
        action_state = self.action_expert.prepare(
            noisy_actions=noisy_actions,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            state=state,
            append_state=False,
        )
        action_hidden = action_state["tokens"]

        for paired_layer in self.paired_layers:
            def layer(
                current_h3: torch.Tensor,
                current_action: torch.Tensor,
                paired_layer: nn.Module = paired_layer,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if self.compute_dtype is not None:
                    current_h3 = current_h3.to(self.compute_dtype)
                    current_action = current_action.to(self.compute_dtype)
                return paired_layer(
                    current_h3,
                    current_action,
                    h3_temb=temb,
                    h3_adaln_indices=adaln_indices,
                    h3_rotary_emb=rotary_emb,
                    video_indices=action_video_indices,
                    action_time_modulation=action_state["time_modulation"],
                    action_context=action_state["context"],
                    action_context_mask=action_state["context_mask"],
                    h3_attention_mask=h3_attention_mask,
                )

            if self.training and self.use_gradient_checkpointing:
                hidden, action_hidden = checkpoint(
                    layer, hidden, action_hidden, use_reentrant=False
                )
            else:
                hidden, action_hidden = layer(hidden, action_hidden)

        output_hidden = self.h3.norm_out(hidden, temb, timestep_indices)
        output_hidden = output_hidden.to(self.h3.proj_out.weight.dtype)
        video_output = self.h3.proj_out(output_hidden).index_select(1, video_indices)
        audio_output = self.h3.audio_proj_out(output_hidden).index_select(1, audio_indices)
        rgb, flow = video_output.split(self.rgb_patch_width, dim=-1)
        return H3DreamWAMOutput(
            rgb_velocity_rows=rgb,
            flow_velocity_rows=flow,
            action_velocity=self.action_expert.decode(action_hidden),
            audio_velocity_rows=audio_output,
        )
