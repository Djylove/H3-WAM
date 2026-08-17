"""LingBot committed observation/action K/V inside the promoted C58 blocks.

Unlike C62/C64, this route does not learn a post-hoc transform of frozen H3
K/V.  Historical observations and clean executed actions are keys/values read
by each C58 ActionDiT self-attention block.  The action K/V is produced by that
same block at timestep zero, matching LingBot's committed-action lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .c64_miniworld_framewise_context import H3TemporalKeyReindex
from .fastwam_full_tower import H3FastWAMFullTowerPolicy
from .lingbot_persistent_kv import (
    LingBotPersistentKVState,
    merge_observation_kv_sequence,
)


def reindex_h3_observation_kv(
    observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    temporal_inv_freq: torch.Tensor,
    frame_delta: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Reindex one separately captured H3 observation before sequence merge.

    Disk-cache H3 observations were all encoded as local first frames.  C57
    merged them without changing their temporal phase, while rollout encoded
    them with an absolute frame offset.  This helper closes that train/deploy
    gap.  It accepts cache tensors with or without a batch dimension.
    """

    reindex = H3TemporalKeyReindex(temporal_inv_freq)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for raw_layer, item in observation_kv.items():
        layer = int(raw_layer)
        if set(item) != {"k", "v"}:
            raise ValueError(f"observation layer {layer} must contain k/v exactly")
        key = item["k"]
        value = item["v"]
        squeezed = False
        if key.ndim == 3 and value.ndim == 3:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeezed = True
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("H3 observation K/V must match [B,S,H,D] or [S,H,D]")
        shifted = reindex(key, int(frame_delta))
        result[layer] = {
            "k": shifted.squeeze(0) if squeezed else shifted,
            "v": value.clone().squeeze(0) if squeezed else value.clone(),
        }
    return result


def prepare_committed_observation_sequence(
    sequence: Sequence[Mapping[int, Mapping[str, torch.Tensor]]],
    *,
    layers: Sequence[int],
    temporal_inv_freq: torch.Tensor,
    frame_start: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Reindex consecutive real frames, then merge them for one commit."""

    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    prepared = [
        reindex_h3_observation_kv(
            item,
            temporal_inv_freq=temporal_inv_freq,
            frame_delta=frame_start + offset,
        )
        for offset, item in enumerate(sequence)
    ]
    return merge_observation_kv_sequence(prepared, layers=layers)


class H3FastWAMLingBotPersistentPolicy(H3FastWAMFullTowerPolicy):
    """C58 with opt-in committed LingBot observation/action block K/V.

    This scoped candidate intentionally omits LingBot's predicted-video cache:
    H3 does not expose an equivalent video denoise token stream in this action
    policy.  Only real feedback is persistent.  No new learned parameters are
    introduced, so a disabled instance strict-loads and exactly executes C58.
    """

    def __init__(
        self,
        *,
        persistent_enabled: bool = False,
        persistent_window_frames: int = 15,
        observation_tokens_per_frame: int = 32,
        action_tokens_per_frame: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        dimensions = (
            persistent_window_frames,
            observation_tokens_per_frame,
            action_tokens_per_frame,
        )
        if min(dimensions) <= 0:
            raise ValueError("persistent window dimensions must be positive")
        self.persistent_enabled = bool(persistent_enabled)
        self.persistent_window_frames = int(persistent_window_frames)
        self.observation_tokens_per_frame = int(observation_tokens_per_frame)
        self.action_tokens_per_frame = int(action_tokens_per_frame)

    @property
    def persistent_token_capacity(self) -> int:
        return self.persistent_window_frames * (
            self.observation_tokens_per_frame + self.action_tokens_per_frame
        )

    def new_persistent_state(self, episode_key: str) -> LingBotPersistentKVState:
        return LingBotPersistentKVState(
            layers=self.carrier_layers,
            token_capacity=self.persistent_token_capacity,
            episode_key=episode_key,
        )

    def _context_state(
        self,
        *,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.action_expert is None or self.proprio_encoder is None:
            raise RuntimeError("C58 action carrier is disabled")
        batch = int(actions.shape[0])
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must be [B,T,action_dim]")
        if timestep.shape != (batch,):
            raise ValueError("timestep must be [B]")
        if text_context.ndim != 3 or tuple(text_context.shape[::2]) != (
            batch,
            self.context_dim,
        ):
            raise ValueError("text_context must be [B,L,context_dim]")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError("proprio must be [B,proprio_dim]")
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], device=text_context.device, dtype=torch.bool
            )
        if tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text_mask must cover text_context")
        parameter = self.proprio_encoder.weight
        proprio_token = self.proprio_encoder(
            proprio.to(device=parameter.device, dtype=parameter.dtype)
        ).unsqueeze(1)
        context = torch.cat(
            (
                text_context.to(device=proprio_token.device, dtype=proprio_token.dtype),
                proprio_token,
            ),
            dim=1,
        )
        context_mask = torch.cat(
            (
                text_mask.to(device=context.device, dtype=torch.bool),
                torch.ones((batch, 1), device=context.device, dtype=torch.bool),
            ),
            dim=1,
        )
        # FastWAM deliberately keeps continuous timesteps in FP32 so values
        # near 1000 are not rounded to the zero-weight endpoint.  Its released
        # ``pre_dit`` produces the sinusoidal embedding in that dtype and
        # relies on the training autocast boundary for the following BF16/FP16
        # Linear.  Feedback commit is a public runtime API and may be called
        # outside such a boundary, so establish the same mixed-precision
        # contract locally without downcasting ``timestep`` itself.
        parameter_dtype = self.action_expert.time_embedding[0].weight.dtype
        device_type = actions.device.type
        autocast_enabled = parameter_dtype in (torch.bfloat16, torch.float16) and (
            device_type == "cuda"
            or (device_type == "cpu" and parameter_dtype == torch.bfloat16)
        )
        with torch.autocast(
            device_type=device_type,
            dtype=parameter_dtype,
            enabled=autocast_enabled,
        ):
            return self.action_expert.pre_dit(
                action_tokens=actions,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )

    def _absolute_action_freqs(
        self, *, action_start: int, action_count: int, device: torch.device
    ) -> torch.Tensor:
        if self.action_expert is None:
            raise RuntimeError("C58 action carrier is disabled")
        stop = int(action_start) + int(action_count)
        if action_start < 0 or action_count <= 0 or stop > self.action_expert.freqs.shape[0]:
            raise ValueError("persistent action position exceeds the C58 RoPE cache")
        return self.action_expert.freqs[action_start:stop].to(device).view(
            action_count, 1, -1
        )

    def _persistent_action_pass(
        self,
        *,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None,
        state: LingBotPersistentKVState,
        initial_observation_kv: Mapping[int, Mapping[str, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.action_expert is None or self._upstream is None:
            raise RuntimeError("C58 action carrier is disabled")
        if state.layers != self.carrier_layers:
            raise ValueError("persistent state layers differ from C58 carrier layers")
        if state.has_predicted:
            raise RuntimeError(
                "predicted K/V is pending; real feedback must replace it before reuse"
            )
        action_state = self._context_state(
            actions=actions,
            timestep=timestep,
            text_context=text_context,
            proprio=proprio,
            text_mask=text_mask,
        )
        action_state["freqs"] = self._absolute_action_freqs(
            action_start=state.action_st_id,
            action_count=int(actions.shape[1]),
            device=actions.device,
        )
        initial = (
            None
            if initial_observation_kv is None
            else self._resolve_carrier_cache(
                initial_observation_kv, batch=int(actions.shape[0])
            )
        )
        tokens = action_state["tokens"]
        generated: list[tuple[torch.Tensor, torch.Tensor]] = []
        upstream = self._upstream
        for block_index, block in enumerate(self.action_expert.blocks):
            layer = self.action_block_to_h3_layer[block_index]
            values = (
                block.modulation.to(
                    device=action_state["t_mod"].device,
                    dtype=action_state["t_mod"].dtype,
                )
                + action_state["t_mod"]
            ).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
            attention_input = upstream.modulate(
                block.norm1(tokens), shift_msa, scale_msa
            )
            query = upstream.rope_apply(
                block.self_attn.norm_q(block.self_attn.q(attention_input)),
                action_state["freqs"],
                block.num_heads,
            )
            key = upstream.rope_apply(
                block.self_attn.norm_k(block.self_attn.k(attention_input)),
                action_state["freqs"],
                block.num_heads,
            )
            value = block.self_attn.v(attention_input)
            keys: list[torch.Tensor] = []
            values_v: list[torch.Tensor] = []
            prefix = state.materialize(layer, include_predicted=False)
            if prefix is not None:
                keys.append(prefix[0].to(key))
                values_v.append(prefix[1].to(value))
            if initial is not None:
                keys.append(initial[layer]["k"].to(key))
                values_v.append(initial[layer]["v"].to(value))
            history_key = torch.cat(keys, dim=1)
            history_value = torch.cat(values_v, dim=1)

            def finish_block(
                current_tokens: torch.Tensor,
                current_query: torch.Tensor,
                current_key: torch.Tensor,
                current_value: torch.Tensor,
                prefix_key: torch.Tensor,
                prefix_value: torch.Tensor,
                owner=block,
                gate_attention=gate_msa,
                gate_feedforward=gate_mlp,
                shift_feedforward=shift_mlp,
                scale_feedforward=scale_mlp,
                cross_context=action_state["context"],
                cross_mask_value=action_state["context_mask"],
            ) -> torch.Tensor:
                mixed = upstream.flash_attention(
                    q=current_query,
                    k=torch.cat((prefix_key, current_key), dim=1),
                    v=torch.cat((prefix_value, current_value), dim=1),
                    num_heads=self.num_heads,
                    ctx_mask=None,
                )
                output = owner.gate(
                    current_tokens, gate_attention, owner.self_attn.o(mixed)
                )
                cross_mask = cross_mask_value
                if cross_mask.ndim == 3:
                    cross_mask = cross_mask.unsqueeze(1)
                output = output + owner.cross_attn(
                    owner.norm3(output),
                    cross_context,
                    ctx_mask=cross_mask,
                )
                ffn_input = upstream.modulate(
                    owner.norm2(output), shift_feedforward, scale_feedforward
                )
                return owner.gate(
                    output, gate_feedforward, owner.ffn(ffn_input)
                )

            if self.use_gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    finish_block,
                    tokens,
                    query,
                    key,
                    value,
                    history_key,
                    history_value,
                    use_reentrant=False,
                )
            else:
                tokens = finish_block(
                    tokens, query, key, value, history_key, history_value
                )
            generated.append((key, value))
        return self.action_expert.post_dit(tokens, action_state), generated

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
        persistent_state: LingBotPersistentKVState | None = None,
    ) -> torch.Tensor:
        if not self.persistent_enabled:
            if persistent_state is not None:
                raise ValueError("persistent state requires persistent_enabled=True")
            return super().forward(
                noisy_actions,
                timestep,
                text_context=text_context,
                proprio=proprio,
                video_kv_cache=video_kv_cache,
                text_mask=text_mask,
            )
        if persistent_state is None:
            raise ValueError("persistent_state is required")
        # Official LingBot commits the post-execution observation before the
        # next predict.  Once history exists, adding video_kv_cache again would
        # duplicate that same current observation (the C57 train/rollout gap).
        initial = video_kv_cache if persistent_state.frame_st_id == 0 else None
        prediction, _ = self._persistent_action_pass(
            actions=noisy_actions,
            timestep=timestep,
            text_context=text_context,
            proprio=proprio,
            text_mask=text_mask,
            state=persistent_state,
            initial_observation_kv=initial,
        )
        return prediction

    def commit_executed_feedback(
        self,
        state: LingBotPersistentKVState,
        *,
        observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        observed_frame_count: int,
        executed_actions: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> None:
        """Atomically append real observations, then clean executed-action K/V."""

        if not self.persistent_enabled:
            raise RuntimeError("cannot commit when persistent K/V is disabled")
        if observed_frame_count <= 0:
            raise ValueError("feedback must contain a real observation")
        if executed_actions.ndim != 3 or executed_actions.shape[-1] != self.action_dim:
            raise ValueError("executed_actions must be [B,T,action_dim]")
        trial = state.clone(detach=False)
        trial.clear_predicted()
        resolved = self._resolve_carrier_cache(
            observation_kv, batch=int(executed_actions.shape[0])
        )
        expected_observation_tokens = (
            int(observed_frame_count) * self.observation_tokens_per_frame
        )
        actual_observation_tokens = int(
            resolved[self.carrier_layers[0]]["k"].shape[1]
        )
        if actual_observation_tokens != expected_observation_tokens:
            raise ValueError(
                "committed observation K/V token count does not match real frames: "
                f"{actual_observation_tokens} != {expected_observation_tokens}"
            )
        observation_update = trial.next_update_id
        for layer in self.carrier_layers:
            trial.append_layer(
                layer,
                kind="observation",
                key=resolved[layer]["k"],
                value=resolved[layer]["v"],
                update_id=observation_update,
                frame_start=trial.frame_st_id,
                frame_count=observed_frame_count,
            )
        trial.next_update_id += 1
        _, clean_kv = self._persistent_action_pass(
            actions=executed_actions,
            timestep=torch.zeros(
                (executed_actions.shape[0],),
                device=executed_actions.device,
                dtype=torch.float32,
            ),
            text_context=text_context,
            proprio=proprio,
            text_mask=text_mask,
            state=trial,
            initial_observation_kv=None,
        )
        action_update = trial.next_update_id
        for layer, (key, value) in zip(self.carrier_layers, clean_kv, strict=True):
            trial.append_layer(
                layer,
                kind="action",
                key=key,
                value=value,
                update_id=action_update,
                action_start=trial.action_st_id,
                action_count=int(executed_actions.shape[1]),
            )
        trial.next_update_id += 1
        trial.frame_st_id += int(observed_frame_count)
        trial.action_st_id += int(executed_actions.shape[1])
        state.replace_from(trial)


__all__ = [
    "H3FastWAMLingBotPersistentPolicy",
    "prepare_committed_observation_sequence",
    "reindex_h3_observation_kv",
]
