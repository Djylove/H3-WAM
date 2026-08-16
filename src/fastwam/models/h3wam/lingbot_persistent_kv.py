"""LingBot-VA persistent observation/action K/V port for the D0 action carrier.

The upstream LingBot server keeps predicted K/V only until real robot feedback
arrives, then replaces it with K/V from the observed frames and executed
actions.  This module ports that lifecycle without modifying the pinned
LingBot or DreamWAM source trees.  It is opt-in: with ``persistent_enabled``
false, the subclass delegates directly to D0 and retains its exact state dict
and forward function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from .dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy


CacheKind = Literal[
    "observation", "action", "predicted_observation", "predicted_action"
]


@dataclass
class LingBotKVEntry:
    """One chronological update in a layer-local rolling attention cache."""

    kind: CacheKind
    key: torch.Tensor
    value: torch.Tensor
    update_id: int
    frame_start: int
    frame_count: int
    action_start: int
    action_count: int
    predicted: bool

    @property
    def token_count(self) -> int:
        return int(self.key.shape[1])

    def clone(self, *, detach: bool = False) -> "LingBotKVEntry":
        key = self.key.detach() if detach else self.key
        value = self.value.detach() if detach else self.value
        return LingBotKVEntry(
            kind=self.kind,
            key=key.clone(),
            value=value.clone(),
            update_id=self.update_id,
            frame_start=self.frame_start,
            frame_count=self.frame_count,
            action_start=self.action_start,
            action_count=self.action_count,
            predicted=self.predicted,
        )


class LingBotPersistentKVState:
    """Per-episode rolling cache with LingBot's predicted/committed boundary.

    Entries are kept chronologically.  LingBot's upstream tensor pool may reuse
    physical slots, but RoPE makes attention invariant to their physical order;
    chronological entries are easier to audit and give the same key/value set.
    """

    SNAPSHOT_SCHEMA = 1

    def __init__(
        self,
        *,
        layers: Sequence[int],
        token_capacity: int,
        episode_key: str,
    ) -> None:
        normalized_layers = tuple(int(layer) for layer in layers)
        if not normalized_layers or len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("persistent layers must be non-empty and unique")
        if token_capacity <= 0:
            raise ValueError("persistent token capacity must be positive")
        if not episode_key:
            raise ValueError("episode_key must not be empty")
        self.layers = normalized_layers
        self.token_capacity = int(token_capacity)
        self.episode_key = str(episode_key)
        self.frame_st_id = 0
        self.action_st_id = 0
        self.next_update_id = 0
        self._entries: dict[int, list[LingBotKVEntry]] = {
            layer: [] for layer in self.layers
        }

    def clone(self, *, detach: bool = False) -> "LingBotPersistentKVState":
        result = LingBotPersistentKVState(
            layers=self.layers,
            token_capacity=self.token_capacity,
            episode_key=self.episode_key,
        )
        result.frame_st_id = self.frame_st_id
        result.action_st_id = self.action_st_id
        result.next_update_id = self.next_update_id
        result._entries = {
            layer: [entry.clone(detach=detach) for entry in entries]
            for layer, entries in self._entries.items()
        }
        return result

    def replace_from(self, other: "LingBotPersistentKVState") -> None:
        if self.layers != other.layers or self.token_capacity != other.token_capacity:
            raise ValueError("cannot replace a persistent state with another contract")
        if self.episode_key != other.episode_key:
            raise ValueError("cannot cross episode boundaries while replacing state")
        self.frame_st_id = other.frame_st_id
        self.action_st_id = other.action_st_id
        self.next_update_id = other.next_update_id
        self._entries = other._entries

    @property
    def has_predicted(self) -> bool:
        return any(
            entry.predicted
            for entries in self._entries.values()
            for entry in entries
        )

    def clear_predicted(self) -> None:
        for layer in self.layers:
            self._entries[layer] = [
                entry for entry in self._entries[layer] if not entry.predicted
            ]

    def _evict(self, layer: int) -> None:
        entries = self._entries[layer]
        while sum(entry.token_count for entry in entries) > self.token_capacity:
            entries.pop(0)

    def append_layer(
        self,
        layer: int,
        *,
        kind: CacheKind,
        key: torch.Tensor,
        value: torch.Tensor,
        update_id: int,
        frame_start: int = 0,
        frame_count: int = 0,
        action_start: int = 0,
        action_count: int = 0,
        predicted: bool = False,
    ) -> None:
        if layer not in self._entries:
            raise KeyError(f"layer {layer} is outside the persistent contract")
        if key.ndim != 3 or value.ndim != 3 or key.shape != value.shape:
            raise ValueError("persistent K/V must have one matching [B,S,W] shape")
        if key.shape[1] <= 0 or key.shape[1] > self.token_capacity:
            raise ValueError("one K/V update must fit in the rolling token capacity")
        if min(frame_start, frame_count, action_start, action_count, update_id) < 0:
            raise ValueError("persistent cache coordinates must be non-negative")
        if predicted != kind.startswith("predicted_"):
            raise ValueError("predicted status must match a predicted cache kind")
        self._entries[layer].append(
            LingBotKVEntry(
                kind=kind,
                key=key.clone(),
                value=value.clone(),
                update_id=int(update_id),
                frame_start=int(frame_start),
                frame_count=int(frame_count),
                action_start=int(action_start),
                action_count=int(action_count),
                predicted=bool(predicted),
            )
        )
        self._evict(layer)

    def materialize(
        self, layer: int, *, include_predicted: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if layer not in self._entries:
            raise KeyError(layer)
        entries = [
            entry
            for entry in self._entries[layer]
            if include_predicted or not entry.predicted
        ]
        if not entries:
            return None
        return (
            torch.cat([entry.key for entry in entries], dim=1),
            torch.cat([entry.value for entry in entries], dim=1),
        )

    def audit(self) -> dict[str, object]:
        per_layer = {}
        for layer, entries in self._entries.items():
            per_layer[str(layer)] = {
                "tokens": sum(entry.token_count for entry in entries),
                "entries": len(entries),
                "predicted_entries": sum(entry.predicted for entry in entries),
                "kinds": [entry.kind for entry in entries],
                "update_ids": [entry.update_id for entry in entries],
            }
        return {
            "episode_key": self.episode_key,
            "frame_st_id": self.frame_st_id,
            "action_st_id": self.action_st_id,
            "next_update_id": self.next_update_id,
            "token_capacity": self.token_capacity,
            "has_predicted": self.has_predicted,
            "layers": per_layer,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.SNAPSHOT_SCHEMA,
            "layers": self.layers,
            "token_capacity": self.token_capacity,
            "episode_key": self.episode_key,
            "frame_st_id": self.frame_st_id,
            "action_st_id": self.action_st_id,
            "next_update_id": self.next_update_id,
            "entries": {
                layer: [
                    {
                        "kind": entry.kind,
                        "key": entry.key.detach().cpu().clone(),
                        "value": entry.value.detach().cpu().clone(),
                        "update_id": entry.update_id,
                        "frame_start": entry.frame_start,
                        "frame_count": entry.frame_count,
                        "action_start": entry.action_start,
                        "action_count": entry.action_count,
                        "predicted": entry.predicted,
                    }
                    for entry in entries
                ]
                for layer, entries in self._entries.items()
            },
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LingBotPersistentKVState":
        if snapshot.get("schema_version") != cls.SNAPSHOT_SCHEMA:
            raise ValueError("unsupported persistent K/V snapshot schema")
        result = cls(
            layers=tuple(snapshot["layers"]),
            token_capacity=int(snapshot["token_capacity"]),
            episode_key=str(snapshot["episode_key"]),
        )
        result.frame_st_id = int(snapshot["frame_st_id"])
        result.action_st_id = int(snapshot["action_st_id"])
        result.next_update_id = int(snapshot["next_update_id"])
        entries = snapshot["entries"]
        if not isinstance(entries, Mapping) or {int(key) for key in entries} != set(result.layers):
            raise ValueError("persistent snapshot layer set differs from its contract")
        for raw_layer, raw_entries in entries.items():
            layer = int(raw_layer)
            for raw in raw_entries:
                key = raw["key"].to(device=device, dtype=dtype)
                value = raw["value"].to(device=device, dtype=dtype)
                result.append_layer(
                    layer,
                    kind=raw["kind"],
                    key=key,
                    value=value,
                    update_id=int(raw["update_id"]),
                    frame_start=int(raw["frame_start"]),
                    frame_count=int(raw["frame_count"]),
                    action_start=int(raw["action_start"]),
                    action_count=int(raw["action_count"]),
                    predicted=bool(raw["predicted"]),
                )
        return result


def merge_observation_kv_sequence(
    sequence: Sequence[Mapping[int, Mapping[str, torch.Tensor]]],
    *,
    layers: Sequence[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Concatenate separately encoded real observation frames in time order."""

    if not sequence:
        raise ValueError("at least one real observation K/V item is required")
    expected = set(int(layer) for layer in layers)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        for item in sequence:
            if set(item) != expected or set(item[layer]) != {"k", "v"}:
                raise ValueError("observation K/V sequence has an inconsistent schema")
        result[int(layer)] = {}
        for name in ("k", "v"):
            tensors = [item[layer][name] for item in sequence]
            ranks = {tensor.ndim for tensor in tensors}
            if ranks == {3}:  # frozen disk cache: [S,H,D]
                sequence_dimension = 0
            elif ranks == {4}:  # live/training batch: [B,S,H,D]
                if len({int(tensor.shape[0]) for tensor in tensors}) != 1:
                    raise ValueError("observation K/V sequence batch sizes differ")
                sequence_dimension = 1
            else:
                raise ValueError("observation K/V must be consistently 3D or 4D")
            result[int(layer)][name] = torch.cat(
                tensors, dim=sequence_dimension
            )
    return result


class H3LingBotPersistentKVPolicy(H3DreamWAMKVCarrierPolicy):
    """D0 carrier with an opt-in LingBot observation/action rolling prefix."""

    def __init__(
        self,
        *,
        persistent_enabled: bool = False,
        persistent_window_chunks: int = 15,
        observation_tokens_per_chunk: int = 32,
        action_tokens_per_chunk: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if min(
            persistent_window_chunks,
            observation_tokens_per_chunk,
            action_tokens_per_chunk,
        ) <= 0:
            raise ValueError("persistent window dimensions must be positive")
        self.persistent_enabled = bool(persistent_enabled)
        self.persistent_window_chunks = int(persistent_window_chunks)
        self.observation_tokens_per_chunk = int(observation_tokens_per_chunk)
        self.action_tokens_per_chunk = int(action_tokens_per_chunk)

    @property
    def persistent_token_capacity(self) -> int:
        return self.persistent_window_chunks * (
            self.observation_tokens_per_chunk + self.action_tokens_per_chunk
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
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.action_expert is None or self.proprio_encoder is None:
            raise RuntimeError("D0 action carrier is disabled")
        batch = int(noisy_actions.shape[0])
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("actions must be [B,T,action_dim]")
        if timestep.shape != (batch,):
            raise ValueError("timestep must be [B]")
        if text_context.ndim != 3 or text_context.shape[0] != batch:
            raise ValueError("text_context must be [B,L,context_dim]")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError("proprio must be [B,proprio_dim]")
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], device=text_context.device, dtype=torch.bool
            )
        if tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text_mask must cover text_context")
        proprio_token = self.proprio_encoder(
            proprio.to(
                device=self.proprio_encoder.weight.device,
                dtype=self.proprio_encoder.weight.dtype,
            )
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
        return self.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )

    def _absolute_action_freqs(
        self, *, action_start: int, action_count: int, device: torch.device
    ) -> torch.Tensor:
        if self.action_expert is None:
            raise RuntimeError("D0 action carrier is disabled")
        if action_start < 0 or action_count <= 0:
            raise ValueError("absolute action coordinates are invalid")
        stop = action_start + action_count
        if stop > int(self.action_expert.freqs.shape[0]):
            raise ValueError(
                f"persistent action position {stop} exceeds RoPE cache "
                f"{self.action_expert.freqs.shape[0]}"
            )
        return self.action_expert.freqs[action_start:stop].to(device).view(
            action_count, 1, -1
        )

    @staticmethod
    def _append_prefix(
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        prefix: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if prefix is not None:
            keys.append(prefix[0].to(device=device, dtype=dtype))
            values.append(prefix[1].to(device=device, dtype=dtype))

    def _persistent_action_pass(
        self,
        *,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: torch.Tensor | None,
        state: LingBotPersistentKVState,
        current_observation_kv: Mapping[int, Mapping[str, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if state.layers != self.carrier_layers:
            raise ValueError("persistent state layers differ from D0 carrier layers")
        if state.has_predicted:
            raise RuntimeError(
                "predicted K/V is still pending; commit real feedback before another pass"
            )
        action_state = self._context_state(
            noisy_actions=actions,
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
        current_cache = (
            None
            if current_observation_kv is None
            else self._resolve_carrier_cache(
                current_observation_kv, batch=int(actions.shape[0])
            )
        )
        JointMoT = self._joint_mot_type
        from dreamwam.layers import scaled_dot_product_attention

        tokens = action_state["tokens"]
        predicted_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.carrier_layers):
            block = self.action_expert.blocks[index]
            action_io = JointMoT._attention_input(
                block,
                tokens,
                action_state["freqs"],
                action_state["time_modulation"],
            )
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            self._append_prefix(
                keys,
                values,
                state.materialize(layer, include_predicted=False),
                device=action_io[1].device,
                dtype=action_io[1].dtype,
            )
            if current_cache is not None:
                keys.append(current_cache[index]["k"].to(action_io[1]))
                values.append(current_cache[index]["v"].to(action_io[2]))
            keys.append(action_io[1])
            values.append(action_io[2])
            mixed = scaled_dot_product_attention(
                action_io[0],
                torch.cat(keys, dim=1),
                torch.cat(values, dim=1),
                self.num_heads,
                None,
            )
            tokens = JointMoT._post_attention(
                block,
                action_io[3],
                mixed,
                *action_io[4:],
                action_state["context"],
                action_state["context_mask"],
            )
            predicted_kv.append((action_io[1], action_io[2]))
        return self.action_expert.post_dit(tokens), predicted_kv

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
        executed_action_history: torch.Tensor | None = None,
        executed_action_history_valid: torch.Tensor | None = None,
        persistent_state: LingBotPersistentKVState | None = None,
        stage_prediction: bool = False,
    ) -> torch.Tensor:
        if not self.persistent_enabled:
            if persistent_state is not None or stage_prediction:
                raise ValueError("persistent arguments require persistent_enabled=True")
            return super().forward(
                noisy_actions,
                timestep,
                text_context=text_context,
                proprio=proprio,
                video_kv_cache=video_kv_cache,
                text_mask=text_mask,
                executed_action_history=executed_action_history,
                executed_action_history_valid=executed_action_history_valid,
            )
        if self.history_action_steps:
            raise ValueError("persistent KV and the legacy history adapter are exclusive")
        if executed_action_history is not None or executed_action_history_valid is not None:
            raise ValueError("persistent KV does not accept flattened executed history")
        if persistent_state is None:
            raise ValueError("persistent_state is required when persistent KV is enabled")
        prediction, predicted_kv = self._persistent_action_pass(
            actions=noisy_actions,
            timestep=timestep,
            text_context=text_context,
            proprio=proprio,
            text_mask=text_mask,
            state=persistent_state,
            current_observation_kv=video_kv_cache,
        )
        if stage_prediction:
            # Upstream commits the final video denoise K/V before the final
            # action denoise K/V.  D0's frozen H3 carrier supplies the current
            # visual K/V directly, so this is a backbone-port representation of
            # that predicted-video entry; both entries are still rolled back
            # together when real feedback arrives.
            predicted_observation = self._resolve_carrier_cache(
                video_kv_cache, batch=int(noisy_actions.shape[0])
            )
            observation_update = persistent_state.next_update_id
            for layer, cache in zip(
                self.carrier_layers, predicted_observation, strict=True
            ):
                persistent_state.append_layer(
                    layer,
                    kind="predicted_observation",
                    key=cache["k"],
                    value=cache["v"],
                    update_id=observation_update,
                    frame_start=persistent_state.frame_st_id,
                    frame_count=1,
                    predicted=True,
                )
            persistent_state.next_update_id += 1
            update_id = persistent_state.next_update_id
            for layer, (key, value) in zip(
                self.carrier_layers, predicted_kv, strict=True
            ):
                persistent_state.append_layer(
                    layer,
                    kind="predicted_action",
                    key=key,
                    value=value,
                    update_id=update_id,
                    action_start=persistent_state.action_st_id,
                    action_count=int(noisy_actions.shape[1]),
                    predicted=True,
                )
            persistent_state.next_update_id += 1
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
        """Atomically replace predicted entries with real observation/action K/V."""

        if not self.persistent_enabled:
            raise RuntimeError("cannot commit feedback when persistent KV is disabled")
        if observed_frame_count <= 0:
            raise ValueError("feedback must contain at least one real observation frame")
        if executed_actions.ndim != 3 or executed_actions.shape[-1] != self.action_dim:
            raise ValueError("executed_actions must be [B,T,action_dim]")
        trial = state.clone(detach=False)
        trial.clear_predicted()
        resolved_observation = self._resolve_carrier_cache(
            observation_kv, batch=int(executed_actions.shape[0])
        )
        observation_update = trial.next_update_id
        for layer, cache in zip(
            self.carrier_layers, resolved_observation, strict=True
        ):
            trial.append_layer(
                layer,
                kind="observation",
                key=cache["k"],
                value=cache["v"],
                update_id=observation_update,
                frame_start=trial.frame_st_id,
                frame_count=observed_frame_count,
            )
        trial.next_update_id += 1
        clean_prediction, clean_kv = self._persistent_action_pass(
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
            current_observation_kv=None,
        )
        del clean_prediction
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
