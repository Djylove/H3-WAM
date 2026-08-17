"""Framewise MiniWorld context bridge for the frozen C58 action policy.

C64 corrects two coupled temporal-contract defects in retired C62: one H3
observation is committed every four executed actions, and every committed H3
key is re-indexed along H3's temporal RoPE axis before it joins the rolling
cache.  C58 remains the action generator; this is not a MiniWorld video-policy
checkpoint reproduction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .c62_miniworld_context import MiniWorldRollingContextState, _clone_kv
from .fastwam_full_tower import H3FastWAMFullTowerPolicy


class H3TemporalKeyReindex(nn.Module):
    """Shift only H3's temporal rotary pairs on already-rotated pooled keys.

    H3 allocates three 16-wide frequency bands to time/height/width in each
    rotary half.  All 98 first-frame condition rows share one time coordinate,
    so adaptive spatial pooling commutes with this temporal-only rotation.
    """

    AXIS_WIDTH = 16
    ROTARY_HALF = 48
    MIN_HEAD_DIM = 96

    def __init__(self, temporal_inv_freq: torch.Tensor) -> None:
        super().__init__()
        value = temporal_inv_freq.detach().float().flatten()
        if tuple(value.shape) != (self.AXIS_WIDTH,) or not torch.isfinite(value).all():
            raise ValueError("H3 temporal inv_freq must be finite shape [16]")
        self.register_buffer("temporal_inv_freq", value.clone())

    def forward(self, key: torch.Tensor, frame_delta: int) -> torch.Tensor:
        if key.ndim != 4 or key.shape[-1] < self.MIN_HEAD_DIM:
            raise ValueError("H3 key must be [B,S,H,D] with D>=96")
        delta = int(frame_delta)
        if delta == 0:
            return key
        angle = self.temporal_inv_freq.to(device=key.device) * float(delta)
        cos = angle.cos().to(key.dtype).view(1, 1, 1, -1)
        sin = angle.sin().to(key.dtype).view(1, 1, 1, -1)
        first = key[..., : self.AXIS_WIDTH]
        second = key[
            ...,
            self.ROTARY_HALF : self.ROTARY_HALF + self.AXIS_WIDTH,
        ]
        result = key.clone()
        result[..., : self.AXIS_WIDTH] = first * cos - second * sin
        result[
            ...,
            self.ROTARY_HALF : self.ROTARY_HALF + self.AXIS_WIDTH,
        ] = second * cos + first * sin
        return result


class MiniWorldFramewiseActionKVModulator(nn.Module):
    """One four-action condition per observed frame, without temporal mean."""

    def __init__(
        self,
        *,
        layers: Sequence[int],
        action_dim: int,
        head_dim: int,
        hidden_dim: int = 256,
        lora_rank: int = 32,
    ) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.action_dim = int(action_dim)
        self.head_dim = int(head_dim)
        self.null_action = nn.Parameter(torch.zeros(1, 4, self.action_dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(4 * self.action_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.shared_modulation = nn.Linear(hidden_dim, 4 * self.head_dim)
        self.layer_refiners = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, lora_rank, bias=False),
                    nn.Linear(lora_rank, 4 * self.head_dim, bias=False),
                )
                for layer in self.layers
            }
        )
        nn.init.zeros_(self.shared_modulation.weight)
        nn.init.zeros_(self.shared_modulation.bias)
        for refiner in self.layer_refiners.values():
            nn.init.zeros_(refiner[-1].weight)

    def _embedding(
        self, actions: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor:
        if actions is None:
            actions = self.null_action.expand(reference.shape[0], -1, -1)
        if tuple(actions.shape[1:]) != (4, self.action_dim):
            raise ValueError("C64 requires exactly four actions per observed frame")
        actions = actions.to(device=reference.device, dtype=reference.dtype)
        return self.action_encoder(actions.flatten(1))

    def forward(
        self,
        layer: int,
        item: Mapping[str, torch.Tensor],
        actions: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        encoded = self._embedding(actions, item["k"])
        modulation = self.shared_modulation(encoded) + self.layer_refiners[str(layer)](
            encoded
        )
        shift_k, scale_k, shift_v, scale_v = modulation.chunk(4, dim=-1)
        broadcast = lambda value: value[:, None, None, :]
        return {
            "k": item["k"] * (1 + broadcast(scale_k)) + broadcast(shift_k),
            "v": item["v"] * (1 + broadcast(scale_v)) + broadcast(shift_v),
        }


class C64MiniWorldFramewiseContextPolicy(nn.Module):
    """C58 with opt-in framewise action alignment and H3 temporal reindex."""

    def __init__(
        self,
        parent: H3FastWAMFullTowerPolicy,
        *,
        temporal_inv_freq: torch.Tensor,
        context_enabled: bool = False,
        max_cache_frames: int = 15,
        modulation_hidden_dim: int = 256,
        modulation_lora_rank: int = 32,
    ) -> None:
        super().__init__()
        if not parent.enabled or parent.action_expert is None:
            raise ValueError("C64 requires an enabled C58 parent")
        self.parent = parent
        self.context_enabled = bool(context_enabled)
        self.max_cache_frames = int(max_cache_frames)
        self.reindex = H3TemporalKeyReindex(temporal_inv_freq)
        self.modulator = MiniWorldFramewiseActionKVModulator(
            layers=parent.carrier_layers,
            action_dim=parent.action_dim,
            head_dim=parent.attn_head_dim,
            hidden_dim=modulation_hidden_dim,
            lora_rank=modulation_lora_rank,
        )

    @property
    def carrier_layers(self) -> tuple[int, ...]:
        return tuple(self.parent.carrier_layers)

    def new_context_state(self, episode_key: str) -> MiniWorldRollingContextState:
        return MiniWorldRollingContextState(
            layers=self.carrier_layers,
            max_cache_chunks=self.max_cache_frames,
            sink_chunks=1,
            episode_key=episode_key,
        )

    def commit_real_observation(
        self,
        state: MiniWorldRollingContextState,
        *,
        observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        four_actions_before_observation: torch.Tensor | None,
    ) -> None:
        if four_actions_before_observation is not None and tuple(
            four_actions_before_observation.shape[1:]
        ) != (4, self.parent.action_dim):
            raise ValueError("C64 feedback must contain exactly four executed actions")
        state.append(observation_kv, four_actions_before_observation)

    def _frame_item(
        self,
        layer: int,
        item: Mapping[str, torch.Tensor],
        actions: torch.Tensor | None,
        logical_position: int,
    ) -> dict[str, torch.Tensor]:
        modulated = self.modulator(layer, item, actions)
        return {
            "k": self.reindex(modulated["k"], logical_position),
            "v": modulated["v"],
        }

    def _rolling_carrier(
        self,
        state: MiniWorldRollingContextState,
        current_observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        four_actions_before_current: torch.Tensor | None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if state.layers != self.carrier_layers:
            raise ValueError("rolling state layers differ from C58 parent")
        current = _clone_kv(current_observation_kv, detach=True)
        result: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.carrier_layers:
            frames = [
                self._frame_item(
                    layer,
                    entry.observation_kv[layer],
                    entry.actions_before_observation,
                    logical_position,
                )
                for logical_position, entry in enumerate(state.entries)
            ]
            frames.append(
                self._frame_item(
                    layer,
                    current[layer],
                    four_actions_before_current,
                    len(state.entries),
                )
            )
            result[layer] = {
                name: torch.cat([frame[name] for frame in frames], dim=1)
                for name in ("k", "v")
            }
        return result

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
        context_state: MiniWorldRollingContextState | None = None,
        four_actions_before_current: torch.Tensor | None = None,
        use_context: bool | None = None,
    ) -> torch.Tensor:
        enabled = self.context_enabled if use_context is None else bool(use_context)
        if not enabled:
            if context_state is not None or four_actions_before_current is not None:
                raise ValueError("framewise arguments require context to be enabled")
            return self.parent(
                noisy_actions,
                timestep,
                text_context=text_context,
                proprio=proprio,
                video_kv_cache=video_kv_cache,
                text_mask=text_mask,
            )
        if context_state is None:
            raise ValueError("context_state is required for C64 framewise context")
        rolling = self._rolling_carrier(
            context_state, video_kv_cache, four_actions_before_current
        )
        return self.parent(
            noisy_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=rolling,
            text_mask=text_mask,
        )


__all__ = [
    "C64MiniWorldFramewiseContextPolicy",
    "H3TemporalKeyReindex",
    "MiniWorldFramewiseActionKVModulator",
]
