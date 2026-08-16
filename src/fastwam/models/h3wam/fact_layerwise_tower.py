"""C56b: FACT tracks inside the shared 30-layer H3-to-Action tower.

Unlike the C56a mechanical precursor, this module does not append an
independent Transformer.  At every official FastWAM ActionDiT block, the
layer-aligned H3 K/V prefix (P) and the predicted-action (A), clean-action
teacher condition (G), future-state/value (V), and future-H3 (I) tracks take
part in one masked attention operation.  Thus future losses update exactly the
same thirty blocks that generate actions, while A is causally unable to read
G/V/I.

The H3 prefix remains a frozen, layer-wise K/V carrier rather than trainable Wan
tokens.  That is the explicit H3 backbone-port deviation; the world/action
tracks after the prefix are not separated into an auxiliary trunk.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch
from torch import nn

from .fact_backbone_port import FACTTokenLayout, build_fact_teacher_forcing_mask
from .fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
)


def _fact_vector_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Official FACT's robot-vector encoder topology."""

    return nn.Sequential(
        nn.Linear(int(input_dim), 128),
        nn.GELU(),
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, int(hidden_dim)),
    )


def _fact_vector_decoder(hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Official FACT's robot-vector decoder topology."""

    return nn.Sequential(
        nn.Linear(int(hidden_dim), 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Linear(128, int(output_dim)),
    )


class H3FACTLayerwiseTowerPolicy(nn.Module):
    """FACT P/A/G/V/I tracks sharing all C58b ActionDiT blocks."""

    def __init__(
        self,
        tower: H3FastWAMFullTowerPolicy,
        *,
        future_state_dim: int = 8,
        future_representation_dim: int = 5376,
    ) -> None:
        super().__init__()
        if not tower.enabled or tower.action_expert is None:
            raise ValueError("C56b requires an enabled C58b tower")
        if tuple(tower.action_block_to_h3_layer) != LAYERWISE_H3_50_TO_ACTION_30:
            raise ValueError("C56b requires the exact C58b H3-50 to ActionDiT-30 mapping")
        if tuple(tower.carrier_layers) != LAYERWISE_H3_50_TO_ACTION_30:
            raise ValueError("C56b requires all 30 layer-wise H3 K/V cache entries")
        if future_state_dim <= 0 or future_representation_dim <= 0:
            raise ValueError("C56b future dimensions must be positive")

        expert = tower.action_expert
        self.tower = tower
        self.hidden_dim = int(expert.hidden_dim)
        self.action_dim = int(expert.action_dim)
        self.future_state_dim = int(future_state_dim)
        self.future_representation_dim = int(future_representation_dim)
        self.future_state_encoder = _fact_vector_encoder(
            self.future_state_dim, self.hidden_dim
        )
        self.value_encoder = _fact_vector_encoder(1, self.hidden_dim)
        self.future_representation_encoder = _fact_vector_encoder(
            self.future_representation_dim, self.hidden_dim
        )
        self.future_state_decoder = _fact_vector_decoder(
            self.hidden_dim, self.future_state_dim
        )
        self.value_decoder = _fact_vector_decoder(self.hidden_dim, 1)
        self.future_representation_decoder = _fact_vector_decoder(
            self.hidden_dim, self.future_representation_dim
        )

    @property
    def shared_blocks(self) -> nn.ModuleList:
        assert self.tower.action_expert is not None
        return self.tower.action_expert.blocks

    @staticmethod
    def _vector_tokens(value: torch.Tensor, width: int, *, name: str) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[-1] != width:
            raise ValueError(f"{name} must be [B,D] or [B,T,D] with D={width}")
        return value

    def _token_time_mod(self, timestep: torch.Tensor, token_count: int) -> torch.Tensor:
        """Produce FACT per-token modulation: caller supplies clean/noisy times."""

        expert = self.tower.action_expert
        assert expert is not None and self.tower._upstream is not None
        if timestep.ndim != 2 or timestep.shape[1] != token_count:
            raise ValueError("per-token timestep shape mismatch")
        batch = timestep.shape[0]
        flat = timestep.reshape(-1)
        embedding = self.tower._upstream.sinusoidal_embedding_1d(
            expert.freq_dim, flat
        )
        time = expert.time_embedding(embedding)
        return expert.time_projection(time).reshape(
            batch, token_count, 6, self.hidden_dim
        )

    def _shared_block_forward(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        token_time_mod: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One shared C58b block with FACT's P/A/G/V/I visibility mask."""

        upstream = self.tower._upstream
        assert upstream is not None
        values = (
            block.modulation.to(
                device=token_time_mod.device, dtype=token_time_mod.dtype
            ).unsqueeze(0)
            + token_time_mod
        ).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            item.squeeze(2) for item in values
        )
        attention_input = upstream.modulate(
            block.norm1(tokens), shift_msa, scale_msa
        )
        query = upstream.rope_apply(
            block.self_attn.norm_q(block.self_attn.q(attention_input)),
            freqs,
            block.num_heads,
        )
        key = upstream.rope_apply(
            block.self_attn.norm_k(block.self_attn.k(attention_input)),
            freqs,
            block.num_heads,
        )
        value = block.self_attn.v(attention_input)
        mixed = upstream.flash_attention(
            q=query,
            k=torch.cat((prefix_key, key), dim=1),
            v=torch.cat((prefix_value, value), dim=1),
            num_heads=self.tower.num_heads,
            ctx_mask=attention_mask,
        )
        tokens = block.gate(tokens, gate_msa, block.self_attn.o(mixed))
        cross_mask = (
            context_mask.unsqueeze(1)
            if context_mask.ndim == 3
            else context_mask
        )
        tokens = tokens + block.cross_attn(
            block.norm3(tokens), context, ctx_mask=cross_mask
        )
        ffn_input = upstream.modulate(block.norm2(tokens), shift_mlp, scale_mlp)
        return block.gate(tokens, gate_mlp, block.ffn(ffn_input))

    def _joint_tokens(
        self,
        *,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        clean_actions: torch.Tensor,
        noisy_future_state: torch.Tensor,
        noisy_value: torch.Tensor,
        noisy_future_representation: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor | None,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, FACTTokenLayout]:
        expert = self.tower.action_expert
        assert expert is not None
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("noisy_actions must be [B,T,action_dim]")
        if clean_actions.shape != noisy_actions.shape:
            raise ValueError("clean_actions must match noisy_actions")
        batch = noisy_actions.shape[0]
        if timestep.ndim != 1 or timestep.shape[0] != batch:
            raise ValueError("timestep must be [B]")
        future_state = self._vector_tokens(
            noisy_future_state, self.future_state_dim, name="noisy_future_state"
        )
        value = self._vector_tokens(noisy_value, 1, name="noisy_value")
        future_representation = self._vector_tokens(
            noisy_future_representation,
            self.future_representation_dim,
            name="noisy_future_representation",
        )
        if any(item.shape[0] != batch for item in (future_state, value, future_representation)):
            raise ValueError("C56b track batch mismatch")

        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], dtype=torch.bool, device=text_context.device
            )
        carrier = self.tower._resolve_carrier_cache(video_kv_cache, batch=batch)
        prefix_tokens = next(iter(carrier.values()))["k"].shape[1]

        parameter = self.tower.proprio_encoder.weight
        proprio_token = self.tower.proprio_encoder(
            proprio.to(device=parameter.device, dtype=parameter.dtype)
        ).unsqueeze(1)
        raw_context = torch.cat(
            (
                text_context.to(
                    device=proprio_token.device, dtype=proprio_token.dtype
                ),
                proprio_token,
            ),
            dim=1,
        )
        raw_context_mask = torch.cat(
            (
                text_mask.to(device=raw_context.device, dtype=torch.bool),
                torch.ones((batch, 1), dtype=torch.bool, device=raw_context.device),
            ),
            dim=1,
        )
        action_state = expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=timestep,
            context=raw_context,
            context_mask=raw_context_mask,
        )
        pred_tokens = action_state["tokens"]
        clean_tokens = expert.action_encoder(clean_actions)
        future_state_tokens = self.future_state_encoder(future_state)
        value_tokens = self.value_encoder(value)
        future_representation_tokens = self.future_representation_encoder(
            future_representation
        )
        tokens = torch.cat(
            (
                pred_tokens,
                clean_tokens,
                future_state_tokens,
                value_tokens,
                future_representation_tokens,
            ),
            dim=1,
        )
        layout = FACTTokenLayout(
            state_tokens=0,
            ref_tokens=int(prefix_tokens),
            pred_action_tokens=int(pred_tokens.shape[1]),
            clean_action_tokens=int(clean_tokens.shape[1]),
            future_state_tokens=int(future_state_tokens.shape[1]),
            value_tokens=int(value_tokens.shape[1]),
            future_representation_tokens=int(future_representation_tokens.shape[1]),
        )
        if tokens.shape[1] > expert.freqs.shape[0]:
            raise ValueError("C56b joint token count exceeds ActionDiT RoPE cache")
        freqs = expert.freqs[: tokens.shape[1]].view(
            tokens.shape[1], 1, -1
        ).to(tokens.device)

        noisy_time = timestep[:, None]
        clean_time = torch.zeros_like(noisy_time)
        per_token_time = torch.cat(
            (
                noisy_time.expand(-1, layout.pred_action_tokens),
                clean_time.expand(-1, layout.clean_action_tokens),
                noisy_time.expand(-1, layout.future_state_tokens),
                noisy_time.expand(-1, layout.value_tokens),
                noisy_time.expand(-1, layout.future_representation_tokens),
            ),
            dim=1,
        )
        token_time_mod = self._token_time_mod(per_token_time, tokens.shape[1])
        context_mask = raw_context_mask.unsqueeze(1).expand(
            -1, tokens.shape[1], -1
        )
        full_mask = build_fact_teacher_forcing_mask(
            layout, device=tokens.device, dtype=tokens.dtype
        )
        attention_mask = full_mask[layout.prefix_end :, :]

        for block_index, block in enumerate(expert.blocks):
            layer = self.tower.action_block_to_h3_layer[block_index]
            prefix_key = carrier[layer]["k"].to(
                device=tokens.device, dtype=tokens.dtype
            )
            prefix_value = carrier[layer]["v"].to(
                device=tokens.device, dtype=tokens.dtype
            )
            context_manager = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if tokens.device.type == "cuda"
                else nullcontext()
            )
            with context_manager:
                if self.tower.use_gradient_checkpointing and self.training:
                    tokens = torch.utils.checkpoint.checkpoint(
                        lambda x, k, v, tm, fr, ctx, cm, am, owner=block: self._shared_block_forward(
                            owner, x, k, v, tm, fr, ctx, cm, am
                        ),
                        tokens,
                        prefix_key,
                        prefix_value,
                        token_time_mod,
                        freqs,
                        action_state["context"],
                        context_mask,
                        attention_mask,
                        use_reentrant=False,
                    )
                else:
                    tokens = self._shared_block_forward(
                        block,
                        tokens,
                        prefix_key,
                        prefix_value,
                        token_time_mod,
                        freqs,
                        action_state["context"],
                        context_mask,
                        attention_mask,
                    )
        return tokens, layout

    def forward_action(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FACT Stage 1; exact C58b action path with no auxiliary modules."""

        return self.tower(
            noisy_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        clean_actions: torch.Tensor,
        noisy_future_state: torch.Tensor,
        noisy_value: torch.Tensor,
        noisy_future_representation: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        tokens, layout = self._joint_tokens(
            noisy_actions=noisy_actions,
            timestep=timestep,
            clean_actions=clean_actions,
            noisy_future_state=noisy_future_state,
            noisy_value=noisy_value,
            noisy_future_representation=noisy_future_representation,
            text_context=text_context,
            text_mask=text_mask,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
        )
        expert = self.tower.action_expert
        assert expert is not None
        pred_start = 0
        pred_end = layout.pred_action_tokens
        clean_end = pred_end + layout.clean_action_tokens
        future_state_end = clean_end + layout.future_state_tokens
        value_end = future_state_end + layout.value_tokens
        return {
            "action": expert.head(tokens[:, pred_start:pred_end]),
            "future_state": self.future_state_decoder(
                tokens[:, clean_end:future_state_end]
            ),
            "value": self.value_decoder(tokens[:, future_state_end:value_end]),
            "future_representation": self.future_representation_decoder(
                tokens[:, value_end:]
            ),
            "layout": layout,
        }

    def forward_consequence(
        self,
        *,
        clean_actions: torch.Tensor,
        timestep: torch.Tensor,
        noisy_future_state: torch.Tensor,
        noisy_value: torch.Tensor,
        noisy_future_representation: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """FACT Stage 2 with a dummy A slot preserving the training layout."""

        result = self.forward(
            torch.zeros_like(clean_actions),
            timestep,
            clean_actions=clean_actions,
            noisy_future_state=noisy_future_state,
            noisy_value=noisy_value,
            noisy_future_representation=noisy_future_representation,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        return {
            key: value
            for key, value in result.items()
            if key != "action"
        }
