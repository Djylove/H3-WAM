"""LingBot-VA four-stream training with one shared MiniMax-H3 backbone."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .joint_attention import shared_h3_four_stream_layer
from .model import apply_h3_rotary


@dataclass
class H3LingBotSharedOutput:
    video_velocity_rows: torch.Tensor
    action_velocity: torch.Tensor


class H3LingBotSharedLayer(nn.Module):
    """FSDP boundary containing one H3 block shared by all four streams."""

    def __init__(self, h3_block: nn.Module) -> None:
        super().__init__()
        self.h3_block = h3_block

    def forward(self, *args, **kwargs):
        return shared_h3_four_stream_layer(
            h3_block=self.h3_block,
            *args,
            **kwargs,
        )


class H3LingBotActionAdapters(nn.Module):
    """Only the action-specific modules present in upstream LingBot-VA."""

    def __init__(
        self,
        h3: nn.Module,
        *,
        action_dim: int,
        state_dim: int,
        text_dim: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(h3.proj_in.out_features)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.action_embedding = nn.Linear(self.action_dim, hidden_dim)
        self.state_embedding = nn.Linear(self.state_dim, text_dim)
        # LingBot-VA deep-copies the video time-conditioning module for
        # actions. Keep that separation while initializing from H3's learned
        # curve; action schedules may then adapt without changing video time.
        self.time_proj = deepcopy(h3.time_proj)
        self.time_embedder = deepcopy(h3.time_embedder)
        self.output = nn.Linear(hidden_dim, self.action_dim)

    def append_state(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        if state.shape != (batch, self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context mask must be [B,L]")
        state_token = self.state_embedding(
            state.to(self.state_embedding.weight.dtype)
        ).to(context.dtype).unsqueeze(1)
        state_mask = torch.ones(batch, 1, device=context.device, dtype=torch.bool)
        return (
            torch.cat((context, state_token), dim=1),
            torch.cat((context_mask.bool(), state_mask), dim=1),
        )

    def time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.numel() == 0:
            raise ValueError("action timesteps must be a non-empty 1D tensor")
        projected = self.time_proj(timestep)
        return self.time_embedder(
            projected.to(self.time_embedder.linear_1.weight.dtype)
        )


class H3LingBotSharedWAM(nn.Module):
    """Port LingBot's shared four-stream block stack to MiniMax-H3.

    Upstream LingBot-VA adds action input/time/output projections to Wan and
    sends noisy/clean video/action tokens through the same Transformer blocks.
    This class preserves that executable contract. H3 has only video/text/
    audio AdaLN tags, so the initial port intentionally assigns actions to the
    otherwise unused audio tag. That choice is an explicit backbone-port
    deviation rather than an assertion of semantic equivalence.
    """

    def __init__(
        self,
        h3: nn.Module,
        *,
        action_dim: int = 7,
        state_dim: int = 8,
        text_dim: int = 5120,
        action_modality_id: int = 2,
        use_gradient_checkpointing: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if action_modality_id not in (0, 1, 2):
            raise ValueError("H3 action modality id must be 0, 1 or 2")
        self.h3 = h3
        self.action_adapters = H3LingBotActionAdapters(
            h3,
            action_dim=action_dim,
            state_dim=state_dim,
            text_dim=text_dim,
        )
        self.action_modality_id = int(action_modality_id)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.compute_dtype = compute_dtype
        h3_blocks = list(h3.transformer_blocks)
        if not h3_blocks:
            raise ValueError("H3 must contain at least one Transformer block")
        h3.transformer_blocks = nn.ModuleList()
        self.shared_layers = nn.ModuleList(
            H3LingBotSharedLayer(block) for block in h3_blocks
        )

    def _video_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.numel() == 0:
            raise ValueError("video timesteps must be a non-empty 1D tensor")
        projected = self.h3.time_proj(timestep)
        return self.h3.time_embedder(
            projected.to(self.h3.time_embedder.linear_1.weight.dtype)
        )

    @staticmethod
    def _indices(
        *,
        length: int,
        modality_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(
            (length,), int(modality_id), device=device, dtype=torch.long
        )

    def forward(
        self,
        *,
        noisy_video_rows: torch.Tensor,
        clean_video_rows: torch.Tensor,
        video_position_ids: torch.Tensor,
        video_chunk_ids: torch.Tensor,
        noisy_video_timestep: torch.Tensor,
        clean_video_timestep: torch.Tensor,
        noisy_video_timestep_indices: torch.Tensor | None = None,
        clean_video_timestep_indices: torch.Tensor | None = None,
        noisy_actions: torch.Tensor,
        clean_actions: torch.Tensor,
        action_position_ids: torch.Tensor,
        action_chunk_ids: torch.Tensor,
        noisy_action_timestep: torch.Tensor,
        clean_action_timestep: torch.Tensor | None = None,
        context: torch.Tensor,
        context_position_ids: torch.Tensor,
        state: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        window_size: int | None = None,
        clean_video_valid: torch.Tensor | None = None,
        clean_action_valid: torch.Tensor | None = None,
    ) -> H3LingBotSharedOutput:
        if noisy_video_rows.shape != clean_video_rows.shape:
            raise ValueError("noisy/clean video row shapes must match")
        if noisy_video_rows.ndim != 3 or noisy_video_rows.shape[0] != 1:
            raise ValueError("packed video rows must be [1,V,C]")
        if noisy_actions.shape != clean_actions.shape:
            raise ValueError("noisy/clean action shapes must match")
        if noisy_actions.ndim != 3 or noisy_actions.shape[0] != 1:
            raise ValueError("packed actions must be [1,A,D]")
        video_length = noisy_video_rows.shape[1]
        action_length = noisy_actions.shape[1]
        if video_position_ids.shape != (video_length, 3):
            raise ValueError("video position ids must be [V,3]")
        if action_position_ids.shape != (action_length, 3):
            raise ValueError("action position ids must be [A,3]")
        if video_chunk_ids.numel() != video_length:
            raise ValueError("video chunk ids must cover every video row")
        if action_chunk_ids.numel() != action_length:
            raise ValueError("action chunk ids must cover every action token")
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError("packed H3 context must be [1,L,C]")
        if context_position_ids.shape != (context.shape[1] + 1, 3):
            raise ValueError("context position ids must reserve one proprio row")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        context, context_mask = self.action_adapters.append_state(
            context, context_mask, state
        )

        noisy_video = self.h3.proj_in(
            noisy_video_rows.to(self.h3.proj_in.weight.dtype)
        )
        clean_video = self.h3.proj_in(
            clean_video_rows.to(self.h3.proj_in.weight.dtype)
        )
        context_hidden = self.h3.context_embedder(
            context.to(self.h3.context_embedder.weight.dtype)
        )
        context_hidden = self.h3.token_refiner(context_hidden)
        noisy_action = self.action_adapters.action_embedding(
            noisy_actions.to(self.action_adapters.action_embedding.weight.dtype)
        )
        clean_action = self.action_adapters.action_embedding(
            clean_actions.to(self.action_adapters.action_embedding.weight.dtype)
        )

        noisy_video_temb = self._video_time_embedding(noisy_video_timestep)
        clean_video_temb = self._video_time_embedding(clean_video_timestep)
        noisy_action_temb = self.action_adapters.time_embedding(
            noisy_action_timestep
        )
        if clean_action_timestep is None:
            clean_action_timestep = torch.ones_like(noisy_action_timestep)
        clean_action_temb = self.action_adapters.time_embedding(
            clean_action_timestep
        )
        if noisy_video_timestep_indices is None:
            noisy_video_timestep_indices = torch.zeros(
                video_length, device=noisy_video.device, dtype=torch.long
            )
        if clean_video_timestep_indices is None:
            clean_video_timestep_indices = torch.zeros(
                video_length, device=clean_video.device, dtype=torch.long
            )
        noisy_video_timestep_indices = noisy_video_timestep_indices.to(
            device=noisy_video.device, dtype=torch.long
        ).reshape(-1)
        clean_video_timestep_indices = clean_video_timestep_indices.to(
            device=clean_video.device, dtype=torch.long
        ).reshape(-1)
        if noisy_video_timestep_indices.shape != (video_length,):
            raise ValueError("noisy video timestep indices must cover every row")
        if clean_video_timestep_indices.shape != (video_length,):
            raise ValueError("clean video timestep indices must cover every row")
        if (
            int(noisy_video_timestep_indices.min()) < 0
            or int(noisy_video_timestep_indices.max()) >= noisy_video_temb.shape[0]
            or int(clean_video_timestep_indices.min()) < 0
            or int(clean_video_timestep_indices.max()) >= clean_video_temb.shape[0]
        ):
            raise ValueError("video timestep index is outside its embedding table")
        noisy_video_indices = noisy_video_timestep_indices * 3
        clean_video_indices = clean_video_timestep_indices * 3
        noisy_action_indices = self._indices(
            length=action_length,
            modality_id=self.action_modality_id,
            device=noisy_action.device,
        )
        clean_action_indices = self._indices(
            length=action_length,
            modality_id=self.action_modality_id,
            device=clean_action.device,
        )
        context_indices = self._indices(
            length=context_hidden.shape[1],
            modality_id=1,
            device=context_hidden.device,
        )
        video_rotary = self.h3.rope(video_position_ids)
        action_rotary = self.h3.rope(action_position_ids)
        context_rotary = self.h3.rope(context_position_ids)

        for shared_layer in self.shared_layers:
            def layer(
                nv: torch.Tensor,
                cv: torch.Tensor,
                na: torch.Tensor,
                ca: torch.Tensor,
                ctx: torch.Tensor,
                shared_layer: nn.Module = shared_layer,
            ) -> tuple[torch.Tensor, ...]:
                tensors = (
                    nv,
                    cv,
                    na,
                    ca,
                    ctx,
                    noisy_video_temb,
                    clean_video_temb,
                    noisy_action_temb,
                    clean_action_temb,
                )
                if self.compute_dtype is not None:
                    tensors = tuple(value.to(self.compute_dtype) for value in tensors)
                nv, cv, na, ca, ctx, nvt, cvt, nat, cat = tensors
                return shared_layer(
                    noisy_video_hidden=nv,
                    clean_video_hidden=cv,
                    noisy_action_hidden=na,
                    clean_action_hidden=ca,
                    noisy_video_temb=nvt,
                    clean_video_temb=cvt,
                    noisy_action_temb=nat,
                    clean_action_temb=cat,
                    noisy_video_adaln_indices=noisy_video_indices,
                    clean_video_adaln_indices=clean_video_indices,
                    noisy_action_adaln_indices=noisy_action_indices,
                    clean_action_adaln_indices=clean_action_indices,
                    video_rotary_emb=video_rotary,
                    action_rotary_emb=action_rotary,
                    h3_apply_rotary=apply_h3_rotary,
                    video_chunk_ids=video_chunk_ids,
                    action_chunk_ids=action_chunk_ids,
                    window_size=window_size,
                    clean_video_valid=clean_video_valid,
                    clean_action_valid=clean_action_valid,
                    context_hidden=ctx,
                    context_temb=cvt,
                    context_adaln_indices=context_indices,
                    context_rotary_emb=context_rotary,
                )

            if self.training and self.use_gradient_checkpointing:
                outputs = checkpoint(
                    layer,
                    noisy_video,
                    clean_video,
                    noisy_action,
                    clean_action,
                    context_hidden,
                    use_reentrant=False,
                )
            else:
                outputs = layer(
                    noisy_video,
                    clean_video,
                    noisy_action,
                    clean_action,
                    context_hidden,
                )
            noisy_video, clean_video, noisy_action, clean_action, context_hidden = (
                outputs
            )

        video_hidden = self.h3.norm_out(
            noisy_video, noisy_video_temb, noisy_video_timestep_indices
        )
        video_velocity = self.h3.proj_out(
            video_hidden.to(self.h3.proj_out.weight.dtype)
        )
        # H3's final norm selects a timestep row, unlike per-block AdaLN which
        # indexes timestep*3+modality. There is one action timestep here.
        action_final_timestep_indices = torch.zeros(
            action_length, device=noisy_action.device, dtype=torch.long
        )
        action_hidden = self.h3.norm_out(
            noisy_action, noisy_action_temb, action_final_timestep_indices
        )
        action_velocity = self.action_adapters.output(
            action_hidden.to(self.action_adapters.output.weight.dtype)
        )
        return H3LingBotSharedOutput(
            video_velocity_rows=video_velocity,
            action_velocity=action_velocity,
        )
