"""Full H3 backbone port of LingBot-VA's four-stream training graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .action_expert import H3DreamActionExpert
from .joint_attention import four_stream_h3_action_layer
from .model import apply_h3_rotary


@dataclass
class H3LingBotWAMOutput:
    video_velocity_rows: torch.Tensor
    action_velocity: torch.Tensor


class H3LingBotPairedLayer(nn.Module):
    """One FSDP wrapping unit containing the two modality experts."""

    def __init__(self, h3_block: nn.Module, action_block: nn.Module) -> None:
        super().__init__()
        self.h3_block = h3_block
        self.action_block = action_block

    def forward(self, *args, **kwargs):
        return four_stream_h3_action_layer(
            h3_block=self.h3_block,
            action_block=self.action_block,
            *args,
            **kwargs,
        )


class H3LingBotWAM(nn.Module):
    """Run H3/ActionDiT as direct block-causal four-stream experts.

    H3 keeps ownership of video and text/proprio representations. ActionDiT
    owns action representations. Their Q/K/V tensors meet only inside the
    official-order causal attention operation; no tail gate is used.
    """

    def __init__(
        self,
        h3: nn.Module,
        action_expert: H3DreamActionExpert,
        *,
        use_gradient_checkpointing: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(h3.transformer_blocks) != len(action_expert.blocks):
            raise ValueError("H3 and ActionDiT must have the same number of layers")
        self.h3 = h3
        self.action_expert = action_expert
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.compute_dtype = compute_dtype
        h3_blocks = list(h3.transformer_blocks)
        action_blocks = list(action_expert.blocks)
        h3.transformer_blocks = nn.ModuleList()
        action_expert.blocks = nn.ModuleList()
        self.paired_layers = nn.ModuleList(
            [
                H3LingBotPairedLayer(h3_block, action_block)
                for h3_block, action_block in zip(
                    h3_blocks, action_blocks, strict=True
                )
            ]
        )

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.numel() == 0:
            raise ValueError("H3 timesteps must be a non-empty 1D tensor")
        projected = self.h3.time_proj(timestep)
        return self.h3.time_embedder(
            projected.to(self.h3.time_embedder.linear_1.weight.dtype)
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
        action_chunk_ids: torch.Tensor,
        noisy_action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_position_ids: torch.Tensor,
        state: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        window_size: int | None = None,
    ) -> H3LingBotWAMOutput:
        if noisy_video_rows.shape != clean_video_rows.shape:
            raise ValueError("noisy/clean video row shapes must match")
        if noisy_video_rows.ndim != 3 or noisy_video_rows.shape[0] != 1:
            raise ValueError("packed video rows must be [1,V,C]")
        if noisy_actions.shape != clean_actions.shape:
            raise ValueError("noisy/clean action shapes must match")
        if noisy_actions.ndim != 3 or noisy_actions.shape[0] != 1:
            raise ValueError("packed actions must be [1,A,D]")
        if video_position_ids.shape != (noisy_video_rows.shape[1], 3):
            raise ValueError("video position ids must be [V,3]")
        if video_chunk_ids.numel() != noisy_video_rows.shape[1]:
            raise ValueError("video chunk ids must cover every video row")
        if action_chunk_ids.numel() != noisy_actions.shape[1]:
            raise ValueError("action chunk ids must cover every action token")
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError("packed H3 context must be [1,L,C]")
        if context_position_ids.shape != (context.shape[1] + 1, 3):
            raise ValueError("context position ids must reserve one proprio row")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        context, context_mask = self.action_expert.append_state_to_context(
            context=context,
            context_mask=context_mask,
            state=state,
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
        noisy_video_temb = self._time_embedding(noisy_video_timestep)
        clean_video_temb = self._time_embedding(clean_video_timestep)
        if noisy_video_timestep_indices is None:
            noisy_video_timestep_indices = torch.zeros(
                noisy_video.shape[1], device=noisy_video.device, dtype=torch.long
            )
        if clean_video_timestep_indices is None:
            clean_video_timestep_indices = torch.zeros(
                clean_video.shape[1], device=clean_video.device, dtype=torch.long
            )
        noisy_video_timestep_indices = noisy_video_timestep_indices.to(
            device=noisy_video.device, dtype=torch.long
        ).reshape(-1)
        clean_video_timestep_indices = clean_video_timestep_indices.to(
            device=clean_video.device, dtype=torch.long
        ).reshape(-1)
        if noisy_video_timestep_indices.shape != (noisy_video.shape[1],):
            raise ValueError("noisy video timestep indices must cover every row")
        if clean_video_timestep_indices.shape != (clean_video.shape[1],):
            raise ValueError("clean video timestep indices must cover every row")
        if (
            int(noisy_video_timestep_indices.min()) < 0
            or int(noisy_video_timestep_indices.max()) >= noisy_video_temb.shape[0]
            or int(clean_video_timestep_indices.min()) < 0
            or int(clean_video_timestep_indices.max()) >= clean_video_temb.shape[0]
        ):
            raise ValueError("video timestep index is outside its embedding table")
        noisy_video_adaln_indices = noisy_video_timestep_indices * 3
        clean_video_adaln_indices = clean_video_timestep_indices * 3
        # H3 modality id 1 is text; both language and appended proprio use the
        # pretrained text modulation row.
        context_adaln_indices = torch.ones(
            context_hidden.shape[1], device=context_hidden.device, dtype=torch.long
        )
        video_rotary = self.h3.rope(video_position_ids)
        context_rotary = self.h3.rope(context_position_ids)
        noisy_action_state = self.action_expert.prepare(
            noisy_actions=noisy_actions,
            timestep=noisy_action_timestep,
            context=context,
            context_mask=context_mask,
            state=state,
            append_state=False,
        )
        clean_action_state = self.action_expert.prepare(
            noisy_actions=clean_actions,
            timestep=torch.zeros_like(noisy_action_timestep),
            context=context,
            context_mask=context_mask,
            state=state,
            append_state=False,
        )
        noisy_action = noisy_action_state["tokens"]
        clean_action = clean_action_state["tokens"]

        for paired_layer in self.paired_layers:
            def layer(
                nv: torch.Tensor,
                cv: torch.Tensor,
                na: torch.Tensor,
                ca: torch.Tensor,
                ctx: torch.Tensor,
                paired_layer: nn.Module = paired_layer,
            ) -> tuple[torch.Tensor, ...]:
                noisy_temb = noisy_video_temb
                clean_temb = clean_video_temb
                noisy_action_modulation = noisy_action_state["time_modulation"]
                clean_action_modulation = clean_action_state["time_modulation"]
                action_context = noisy_action_state["context"]
                if self.compute_dtype is not None:
                    nv = nv.to(self.compute_dtype)
                    cv = cv.to(self.compute_dtype)
                    na = na.to(self.compute_dtype)
                    ca = ca.to(self.compute_dtype)
                    ctx = ctx.to(self.compute_dtype)
                    # These tensors are captured by the checkpoint closure,
                    # so FSDP's cast_forward_inputs does not reliably revisit
                    # them during backward recomputation. Cast explicitly to
                    # keep first forward and recompute numerically identical.
                    noisy_temb = noisy_temb.to(self.compute_dtype)
                    clean_temb = clean_temb.to(self.compute_dtype)
                    noisy_action_modulation = noisy_action_modulation.to(
                        self.compute_dtype
                    )
                    clean_action_modulation = clean_action_modulation.to(
                        self.compute_dtype
                    )
                    action_context = action_context.to(self.compute_dtype)
                return paired_layer(
                    noisy_video_hidden=nv,
                    clean_video_hidden=cv,
                    noisy_action_hidden=na,
                    clean_action_hidden=ca,
                    noisy_h3_temb=noisy_temb,
                    clean_h3_temb=clean_temb,
                    noisy_h3_adaln_indices=noisy_video_adaln_indices,
                    clean_h3_adaln_indices=clean_video_adaln_indices,
                    h3_rotary_emb=video_rotary,
                    h3_apply_rotary=apply_h3_rotary,
                    noisy_action_time_modulation=noisy_action_modulation,
                    clean_action_time_modulation=clean_action_modulation,
                    action_context=action_context,
                    action_context_mask=noisy_action_state["context_mask"],
                    video_chunk_ids=video_chunk_ids,
                    action_chunk_ids=action_chunk_ids,
                    window_size=window_size,
                    h3_context_hidden=ctx,
                    h3_context_temb=clean_temb,
                    h3_context_adaln_indices=context_adaln_indices,
                    h3_context_rotary_emb=context_rotary,
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
        action_velocity = self.action_expert.decode(noisy_action)
        return H3LingBotWAMOutput(
            video_velocity_rows=video_velocity,
            action_velocity=action_velocity,
        )
