"""MiniMax-H3 instantiation of Faster-WAM's Dock-of-Transformer design."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .action_expert import _attention
from .docking import H3DoTActionHead, H3DoTKVFusion
from .joint_attention import h3_attention_input, h3_post_attention
from .model import apply_h3_rotary


@dataclass
class H3DoTWAMOutput:
    rgb_velocity_rows: torch.Tensor
    flow_velocity_rows: torch.Tensor
    action_velocity: torch.Tensor
    audio_velocity_rows: torch.Tensor
    docked_keys: torch.Tensor | None = None
    docked_values: torch.Tensor | None = None


class H3DoTHubLayer(nn.Module):
    """FSDP wrapping unit for one H3 layer and its exported K/V cache."""

    def __init__(self, h3_block: nn.Module) -> None:
        super().__init__()
        self.h3_block = h3_block

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        condition_video_indices: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h3_io = h3_attention_input(
            self.h3_block,
            hidden,
            temb=temb,
            adaln_indices=adaln_indices,
            rotary_emb=rotary_emb,
            apply_rotary=apply_h3_rotary,
        )
        attended = _attention(h3_io[0], h3_io[1], h3_io[2], attention_mask)
        next_hidden = h3_post_attention(
            self.h3_block,
            attended=attended,
            residual=h3_io[3],
            gate_attn=h3_io[4],
            shift_ffn=h3_io[5],
            scale_ffn=h3_io[6],
            gate_ffn=h3_io[7],
            adaln_indices=adaln_indices,
        )
        return (
            next_hidden,
            h3_io[1].index_select(1, condition_video_indices),
            h3_io[2].index_select(1, condition_video_indices),
        )


class H3DoTWAM(nn.Module):
    """Use H3 as a full-depth hub and dock a shallow action head onto it."""

    def __init__(
        self,
        h3: nn.Module,
        action_head: H3DoTActionHead,
        kv_fusion: H3DoTKVFusion,
        *,
        state_dim: int,
        text_dim: int,
        rgb_patch_width: int,
        use_gradient_checkpointing: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.h3 = h3
        self.action_head = action_head
        self.kv_fusion = kv_fusion
        self.state_dim = int(state_dim)
        self.text_dim = int(text_dim)
        self.rgb_patch_width = int(rgb_patch_width)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.compute_dtype = compute_dtype
        self.state_embedding = nn.Linear(self.state_dim, self.text_dim)
        video_layers = len(self.h3.transformer_blocks)
        if video_layers != self.kv_fusion.video_layers:
            raise ValueError("H3 depth must match KV-Fusion video depth")
        if len(self.action_head.layers) != self.kv_fusion.action_layers:
            raise ValueError("action head depth must match KV-Fusion output depth")
        action_attention = self.action_head.layers[0].attn
        if (
            action_attention.num_heads != self.kv_fusion.action_num_heads
            or action_attention.head_dim != self.kv_fusion.action_head_dim
        ):
            raise ValueError("action head geometry must match KV-Fusion output")
        h3_blocks = list(self.h3.transformer_blocks)
        self.h3.transformer_blocks = nn.ModuleList()
        self.hub_layers = nn.ModuleList(
            [H3DoTHubLayer(block) for block in h3_blocks]
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
        condition_video_indices: torch.Tensor,
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
        condition_video_indices = condition_video_indices.long().reshape(-1)
        if condition_video_indices.numel() == 0:
            raise ValueError("DoT requires conditioning-frame video rows")
        video_set = set(video_indices.long().tolist())
        if any(index not in video_set for index in condition_video_indices.tolist()):
            raise ValueError("conditioning rows must be a subset of video rows")

    def _append_state(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != (context.shape[0], self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        state_token = self.state_embedding(
            state.to(self.state_embedding.weight.dtype)
        ).to(context.dtype)[:, None]
        state_mask = torch.ones(
            (context.shape[0], 1),
            device=context_mask.device,
            dtype=torch.bool,
        )
        return (
            torch.cat((context, state_token), dim=1),
            torch.cat((context_mask, state_mask), dim=1),
        )

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
        condition_video_indices: torch.Tensor,
        noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        state: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        h3_attention_mask: torch.Tensor | None = None,
        docked_video_positions: torch.Tensor | None = None,
        cached_docked_keys: torch.Tensor | None = None,
        cached_docked_values: torch.Tensor | None = None,
    ) -> H3DoTWAMOutput:
        if (cached_docked_keys is None) != (cached_docked_values is None):
            raise ValueError("cached docked keys and values must be provided together")
        if cached_docked_keys is not None:
            action_velocity = self.action_head(
                noisy_actions=noisy_actions,
                timestep=action_timestep,
                docked_keys=cached_docked_keys,
                docked_values=cached_docked_values,
            )
            empty = noisy_actions.new_empty((noisy_actions.shape[0], 0, 0))
            return H3DoTWAMOutput(
                rgb_velocity_rows=empty,
                flow_velocity_rows=empty,
                action_velocity=action_velocity,
                audio_velocity_rows=empty,
                docked_keys=cached_docked_keys,
                docked_values=cached_docked_values,
            )
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
            condition_video_indices=condition_video_indices,
        )
        if video_rows.shape[-1] != self.h3.proj_in.in_features:
            raise ValueError("video row width does not match expanded H3 proj_in")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        context, context_mask = self._append_state(context, context_mask, state)
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
        cached_keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []

        for hub_layer in self.hub_layers:
            def layer(
                current_hidden: torch.Tensor,
                hub_layer: nn.Module = hub_layer,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if self.compute_dtype is not None:
                    current_hidden = current_hidden.to(self.compute_dtype)
                return hub_layer(
                    current_hidden,
                    temb=temb,
                    adaln_indices=adaln_indices,
                    rotary_emb=rotary_emb,
                    condition_video_indices=condition_video_indices,
                    attention_mask=h3_attention_mask,
                )

            if self.training and self.use_gradient_checkpointing:
                hidden, layer_key, layer_value = checkpoint(
                    layer, hidden, use_reentrant=False
                )
            else:
                hidden, layer_key, layer_value = layer(hidden)
            cached_keys.append(layer_key)
            cached_values.append(layer_value)

        video_cos = rotary_emb[0].index_select(0, condition_video_indices)
        video_sin = rotary_emb[1].index_select(0, condition_video_indices)
        docked_keys, docked_values = self.kv_fusion(
            rotated_video_keys=torch.stack(cached_keys, dim=0),
            video_values=torch.stack(cached_values, dim=0),
            video_cos=video_cos,
            video_sin=video_sin,
            action_positions=docked_video_positions,
        )
        action_velocity = self.action_head(
            noisy_actions=noisy_actions,
            timestep=action_timestep,
            docked_keys=docked_keys,
            docked_values=docked_values,
        )

        output_hidden = self.h3.norm_out(hidden, temb, timestep_indices)
        output_hidden = output_hidden.to(self.h3.proj_out.weight.dtype)
        video_output = self.h3.proj_out(output_hidden).index_select(1, video_indices)
        audio_output = self.h3.audio_proj_out(output_hidden).index_select(1, audio_indices)
        rgb, flow = video_output.split(self.rgb_patch_width, dim=-1)
        return H3DoTWAMOutput(
            rgb_velocity_rows=rgb,
            flow_velocity_rows=flow,
            action_velocity=action_velocity,
            audio_velocity_rows=audio_output,
            docked_keys=docked_keys,
            docked_values=docked_values,
        )
