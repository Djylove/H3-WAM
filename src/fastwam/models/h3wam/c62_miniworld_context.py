"""MiniWorld-style rolling world context for the C58 carrier champion.

MiniWorld is an action-conditioned *video* world model, not an action policy.
This module therefore keeps C58 as the action policy and ports only the
source-backed context lifecycle: action-aligned world chunks, a persistent
sink, and bounded FIFO history.  The bridge is opt-in and its disabled path is
the unmodified C58 forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .fastwam_full_tower import H3FastWAMFullTowerPolicy


MINIWORLD_COMMIT = "e484206bbd4360ae56ed8abad51c83f2457ac092"
MINIWORLD_SOURCE_SHA256 = {
    "miniworld/miniworld.py": "0469446f4c54b51440842d7984e073fcb4ba90b4127bf9264403a807be0cfe92",
    "miniworld/denoiser.py": "1dca51f52bce35f71018130ccea56714dd98a61a9abad3b85b6804bde5e7287e",
    "miniworld/train.py": "d04775fab8059bacaaba18562a981089a94e5cf41f07f7ebb15ac28a5dab3c2b",
    "miniworld/data/droid.py": "50ddd9b0a753a09dea211c13fded4010bcd58574f628d4ae29621d9165195248",
    "miniworld/conditioning/actions.py": "fc5829e96faa754b62fdc541961c833315ac24ce0096e6c85c66a6d03af2ff37",
    "scripts/train_droid.sh": "e305f5744fe49f46915b306d25f9c573d835bbe52d264c046e44bcec25656e64",
}


def verify_miniworld_execution_source(source_root: Path | str) -> dict[str, Any]:
    """Verify and import the fixed official execution source.

    The imported functions are exercised here so a byte-correct but unusable
    vendor checkout fails before any training allocation.
    """

    root = Path(source_root).resolve()
    actual: dict[str, str] = {}
    for relative, expected in MINIWORLD_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"MiniWorld source hash mismatch for {relative}: {digest}"
            )
        actual[relative] = digest

    # The fixed project runtime does not need MiniWorld's FlashAttention CUDA
    # kernel for these source probes: the block-causal path uses PyTorch SDPA.
    # Keep the substitution explicit and temporary instead of mutating vendor
    # source or pretending the full MiniWorld training environment is present.
    flash_attn_shim = False
    previous_flash_attn = sys.modules.get("flash_attn")
    if previous_flash_attn is None:
        flash_module = types.ModuleType("flash_attn")

        def _sdpa_flash(q, k, v, causal=False):
            qh, kh, vh = (value.transpose(1, 2) for value in (q, k, v))
            return torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=causal
            ).transpose(1, 2)

        flash_module.flash_attn_func = _sdpa_flash
        sys.modules["flash_attn"] = flash_module
        flash_attn_shim = True
    try:
        actions_module = importlib.import_module("miniworld.conditioning.actions")
        model_module = importlib.import_module("miniworld.miniworld")
    finally:
        if flash_attn_shim:
            sys.modules.pop("flash_attn", None)
    imported_actions = Path(actions_module.__file__).resolve()
    imported_model = Path(model_module.__file__).resolve()
    if imported_actions != root / "miniworld/conditioning/actions.py":
        raise RuntimeError(f"MiniWorld action import escaped source root: {imported_actions}")
    if imported_model != root / "miniworld/miniworld.py":
        raise RuntimeError(f"MiniWorld model import escaped source root: {imported_model}")

    grouped = actions_module.build_cond_seq_from_actions(torch.zeros(1, 8, 7))
    if tuple(grouped.shape) != (1, 3, 28) or not torch.equal(
        grouped[:, 0], torch.zeros_like(grouped[:, 0])
    ):
        raise RuntimeError("MiniWorld action-to-latent alignment probe failed")
    mask = model_module._build_temporal_chunkwise_attn_mask(
        seq_len=8,
        tokens_per_frame=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        chunk_size=2,
    )
    if tuple(mask.shape) != (1, 1, 8, 8):
        raise RuntimeError("MiniWorld block-causal mask shape probe failed")
    # Chunk 0 cannot read chunk 1; tokens within one chunk are bidirectional.
    if not torch.isneginf(mask[0, 0, 0, 4]) or mask[0, 0, 0, 3] != 0:
        raise RuntimeError("MiniWorld block-causal mask semantics probe failed")
    return {
        "commit": MINIWORLD_COMMIT,
        "source_root": str(root),
        "sha256": actual,
        "action_alignment_shape": list(grouped.shape),
        "block_causal_mask_shape": list(mask.shape),
        "flash_attn_import_shim": flash_attn_shim,
    }


def _clone_kv(
    value: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    detach: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer, item in value.items():
        if set(item) != {"k", "v"}:
            raise ValueError(f"layer {layer} must contain k and v exactly")
        result[int(layer)] = {}
        for name in ("k", "v"):
            tensor = item[name]
            if tensor.ndim != 4:
                raise ValueError("rolling observation K/V must be [B,S,H,D]")
            tensor = tensor.detach() if detach else tensor
            result[int(layer)][name] = tensor.to(device=device, dtype=dtype).clone()
    return result


@dataclass
class MiniWorldContextEntry:
    observation_kv: dict[int, dict[str, torch.Tensor]]
    actions_before_observation: torch.Tensor | None
    update_id: int


class MiniWorldRollingContextState:
    """Episode-local sink + FIFO context; it never contains future feedback."""

    SNAPSHOT_SCHEMA = 1

    def __init__(
        self,
        *,
        layers: Sequence[int],
        max_cache_chunks: int,
        sink_chunks: int = 1,
        episode_key: str,
    ) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.max_cache_chunks = int(max_cache_chunks)
        self.sink_chunks = int(sink_chunks)
        self.episode_key = str(episode_key)
        if self.max_cache_chunks <= 0:
            raise ValueError("max_cache_chunks must be positive")
        if self.sink_chunks != 1:
            raise ValueError("C62 fixes the MiniWorld persistent sink to one chunk")
        self.next_update_id = 0
        self.entries: list[MiniWorldContextEntry] = []

    def append(
        self,
        observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        actions_before_observation: torch.Tensor | None,
    ) -> None:
        if set(observation_kv) != set(self.layers):
            raise ValueError("observation K/V layers differ from context contract")
        if actions_before_observation is not None:
            if actions_before_observation.ndim != 3:
                raise ValueError("executed actions must be [B,T,A]")
            if actions_before_observation.shape[1] % 4:
                raise ValueError("MiniWorld alignment requires 4n executed actions")
        self.entries.append(
            MiniWorldContextEntry(
                observation_kv=_clone_kv(observation_kv, detach=False),
                actions_before_observation=(
                    None
                    if actions_before_observation is None
                    else actions_before_observation.clone()
                ),
                update_id=self.next_update_id,
            )
        )
        self.next_update_id += 1
        if len(self.entries) > self.max_cache_chunks:
            # Preserve the first real observation as MiniWorld's attention sink.
            self.entries = [self.entries[0], *self.entries[-(self.max_cache_chunks - 1) :]]
            if self.max_cache_chunks == 1:
                self.entries = self.entries[:1]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SNAPSHOT_SCHEMA,
            "layers": self.layers,
            "max_cache_chunks": self.max_cache_chunks,
            "sink_chunks": self.sink_chunks,
            "episode_key": self.episode_key,
            "next_update_id": self.next_update_id,
            "entries": [
                {
                    "observation_kv": _clone_kv(entry.observation_kv, detach=True, device="cpu"),
                    "actions_before_observation": (
                        None
                        if entry.actions_before_observation is None
                        else entry.actions_before_observation.detach().cpu().clone()
                    ),
                    "update_id": entry.update_id,
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MiniWorldRollingContextState":
        if snapshot.get("schema_version") != cls.SNAPSHOT_SCHEMA:
            raise ValueError("unsupported C62 rolling-context snapshot")
        result = cls(
            layers=tuple(snapshot["layers"]),
            max_cache_chunks=int(snapshot["max_cache_chunks"]),
            sink_chunks=int(snapshot["sink_chunks"]),
            episode_key=str(snapshot["episode_key"]),
        )
        for raw in snapshot["entries"]:
            result.entries.append(
                MiniWorldContextEntry(
                    observation_kv=_clone_kv(
                        raw["observation_kv"], detach=False, device=device, dtype=dtype
                    ),
                    actions_before_observation=(
                        None
                        if raw["actions_before_observation"] is None
                        else raw["actions_before_observation"].to(
                            device=device, dtype=dtype
                        ).clone()
                    ),
                    update_id=int(raw["update_id"]),
                )
            )
        result.next_update_id = int(snapshot["next_update_id"])
        if result.entries and [entry.update_id for entry in result.entries] != sorted(
            entry.update_id for entry in result.entries
        ):
            raise ValueError("rolling-context update order is not causal")
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "episode_key": self.episode_key,
            "max_cache_chunks": self.max_cache_chunks,
            "sink_chunks": self.sink_chunks,
            "entries": len(self.entries),
            "update_ids": [entry.update_id for entry in self.entries],
            "action_lengths": [
                0
                if entry.actions_before_observation is None
                else int(entry.actions_before_observation.shape[1])
                for entry in self.entries
            ],
        }


class MiniWorldActionKVModulator(nn.Module):
    """Shared action modulation plus layer-local low-rank refinements.

    This follows MiniWorld's shared AdaLN + per-block LoRA organization.  It
    modulates already-extracted H3 world K/V and does not claim to reproduce
    MiniWorld's video DiT weights.
    """

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
        self.action_encoder = nn.Sequential(
            nn.Linear(4 * self.action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.shared_modulation = nn.Linear(hidden_dim, 4 * self.head_dim)
        self.layer_refiners = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Linear(hidden_dim, lora_rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(lora_rank, 4 * self.head_dim, bias=False),
                )
                for layer in self.layers
            }
        )
        nn.init.zeros_(self.shared_modulation.weight)
        nn.init.zeros_(self.shared_modulation.bias)
        for refiner in self.layer_refiners.values():
            nn.init.zeros_(refiner[-1].weight)

    def _action_embedding(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must be [B,4n,action_dim]")
        if actions.shape[1] <= 0 or actions.shape[1] % 4:
            raise ValueError("MiniWorld action conditioning requires 4n actions")
        groups = actions.reshape(actions.shape[0], -1, 4 * self.action_dim)
        return self.action_encoder(groups).mean(dim=1)

    def forward(
        self,
        layer: int,
        item: Mapping[str, torch.Tensor],
        actions: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if actions is None:
            return {name: item[name] for name in ("k", "v")}
        encoded = self._action_embedding(actions.to(item["k"]))
        modulation = self.shared_modulation(encoded) + self.layer_refiners[str(layer)](
            encoded
        )
        shift_k, scale_k, shift_v, scale_v = modulation.chunk(4, dim=-1)
        broadcast = lambda value: value[:, None, None, :]
        return {
            "k": item["k"] * (1 + broadcast(scale_k)) + broadcast(shift_k),
            "v": item["v"] * (1 + broadcast(scale_v)) + broadcast(shift_v),
        }


class C62MiniWorldRollingContextPolicy(nn.Module):
    """C58 action policy with opt-in MiniWorld-style real-world context."""

    def __init__(
        self,
        parent: H3FastWAMFullTowerPolicy,
        *,
        context_enabled: bool = False,
        max_cache_chunks: int = 15,
        sink_chunks: int = 1,
        modulation_hidden_dim: int = 256,
        modulation_lora_rank: int = 32,
    ) -> None:
        super().__init__()
        if not parent.enabled or parent.action_expert is None:
            raise ValueError("C62 requires an enabled C58 parent")
        self.parent = parent
        self.context_enabled = bool(context_enabled)
        self.max_cache_chunks = int(max_cache_chunks)
        self.sink_chunks = int(sink_chunks)
        self.modulator = MiniWorldActionKVModulator(
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
            max_cache_chunks=self.max_cache_chunks,
            sink_chunks=self.sink_chunks,
            episode_key=episode_key,
        )

    def commit_real_observation(
        self,
        state: MiniWorldRollingContextState,
        *,
        observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        actions_before_observation: torch.Tensor | None,
    ) -> None:
        if state.layers != self.carrier_layers:
            raise ValueError("rolling state layers differ from C58 parent")
        state.append(observation_kv, actions_before_observation)

    def _rolling_carrier(
        self,
        state: MiniWorldRollingContextState,
        current_observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        actions_before_current: torch.Tensor | None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if state.layers != self.carrier_layers:
            raise ValueError("rolling state layers differ from C58 parent")
        current = _clone_kv(current_observation_kv, detach=False)
        result: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.carrier_layers:
            entries = [
                self.modulator(
                    layer, entry.observation_kv[layer], entry.actions_before_observation
                )
                for entry in state.entries
            ]
            entries.append(
                self.modulator(layer, current[layer], actions_before_current)
            )
            result[layer] = {
                name: torch.cat([entry[name] for entry in entries], dim=1)
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
        actions_before_current: torch.Tensor | None = None,
        use_context: bool | None = None,
    ) -> torch.Tensor:
        enabled = self.context_enabled if use_context is None else bool(use_context)
        if not enabled:
            if context_state is not None or actions_before_current is not None:
                raise ValueError("rolling arguments require context to be enabled")
            return self.parent(
                noisy_actions,
                timestep,
                text_context=text_context,
                proprio=proprio,
                video_kv_cache=video_kv_cache,
                text_mask=text_mask,
            )
        if context_state is None:
            raise ValueError("context_state is required for C62 rolling context")
        rolling = self._rolling_carrier(
            context_state, video_kv_cache, actions_before_current
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
    "C62MiniWorldRollingContextPolicy",
    "MINIWORLD_COMMIT",
    "MINIWORLD_SOURCE_SHA256",
    "MiniWorldRollingContextState",
    "verify_miniworld_execution_source",
]
