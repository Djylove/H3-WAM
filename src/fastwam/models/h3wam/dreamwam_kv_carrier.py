"""DreamWAM-style action carrier over frozen MiniMax-H3 layer K/V caches.

This is an opt-in research module.  It reuses the byte-verified ActionDiT and
JointMoT block helpers from the pinned DreamWAM repository, but consumes real
K/V tensors captured from selected frozen H3 layers instead of constructing a
second trainable video expert.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn


DREAMWAM_COMMIT = "6e989facc0c452fd3488d75f60bc36411005558c"
DREAMWAM_LAYERS_SHA256 = (
    "3cd38ad24eff05e748d9353af3f39200e93b16b6d07d22f153ccef0f36becd96"
)
DREAMWAM_EXPERTS_SHA256 = (
    "9ba51dbb15b8df8e4ff01c5a08acf443a950c422544e4497d13e0e2658bd489c"
)
DREAMWAM_MOT_SHA256 = (
    "5467d135287a6e77074cb653fc3d72218490fcfa40ac486b61d5cc5975ab6c01"
)
DEFAULT_H3_CARRIER_LAYERS = (9, 19, 29, 39, 49)
_PINNED_CARRIER_TYPES = None


def _verified_source(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"pinned DreamWAM source hash mismatch for {path}: {actual}")
    return path


def _load_pinned_dreamwam_carrier():
    global _PINNED_CARRIER_TYPES
    if _PINNED_CARRIER_TYPES is not None:
        return _PINNED_CARRIER_TYPES

    repo_root = Path(__file__).resolve().parents[4]
    source_root = repo_root / "third_party/DreamWAM/dreamwam"
    sources = (
        ("dreamwam.layers", "layers.py", DREAMWAM_LAYERS_SHA256),
        ("dreamwam.experts", "experts.py", DREAMWAM_EXPERTS_SHA256),
        ("dreamwam.mot", "mot.py", DREAMWAM_MOT_SHA256),
    )
    package = sys.modules.setdefault("dreamwam", types.ModuleType("dreamwam"))
    package.__path__ = [str(source_root)]
    for module_name, filename, digest in sources:
        source = _verified_source(source_root / filename, digest)
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load pinned DreamWAM module {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(package, module_name.rsplit(".", 1)[-1], module)
    _PINNED_CARRIER_TYPES = (
        sys.modules["dreamwam.experts"].ActionDiT,
        sys.modules["dreamwam.mot"].JointMoT,
    )
    return _PINNED_CARRIER_TYPES


def h3_kv_cache_bytes(
    *,
    layers: int,
    tokens: int,
    heads: int = 56,
    head_dim: int = 128,
    element_size: int = 2,
) -> int:
    """Exact uncompressed K+V storage for one sample."""

    if min(layers, tokens, heads, head_dim, element_size) <= 0:
        raise ValueError("cache dimensions and element_size must be positive")
    return layers * tokens * heads * head_dim * 2 * element_size


def _storage_signature(tensor: torch.Tensor) -> int:
    """Identify the underlying storage, including differently offset views."""

    return tensor.untyped_storage().data_ptr()


class H3DreamWAMKVCarrierPolicy(nn.Module):
    """ActionDiT whose per-block self-attention carries distinct H3 video K/V."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        carrier_layers: Sequence[int] = DEFAULT_H3_CARRIER_LAYERS,
        action_dim: int = 7,
        proprio_dim: int = 8,
        context_dim: int = 5120,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 56,
        attn_head_dim: int = 128,
        freq_dim: int = 256,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.carrier_layers = tuple(int(index) for index in carrier_layers)
        self.action_block_to_h3_layer = self.carrier_layers
        if not self.carrier_layers:
            raise ValueError("carrier_layers must not be empty")
        if tuple(sorted(set(self.carrier_layers))) != self.carrier_layers:
            raise ValueError("carrier_layers must be strictly increasing and unique")
        if self.carrier_layers[0] < 0 or self.carrier_layers[-1] >= 50:
            raise ValueError("carrier_layers must select H3 blocks in [0,49]")
        dimensions = (
            action_dim,
            proprio_dim,
            context_dim,
            hidden_dim,
            num_heads,
            attn_head_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_width = self.num_heads * self.attn_head_dim
        self.context_dim = int(context_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_expert: nn.Module | None = None
        self.proprio_encoder: nn.Module | None = None
        self._joint_mot_type = None
        if self.enabled:
            ActionDiT, JointMoT = _load_pinned_dreamwam_carrier()
            self.action_expert = ActionDiT(
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                text_dim=context_dim,
                freq_dim=freq_dim,
                num_heads=num_heads,
                attn_head_dim=attn_head_dim,
                num_layers=len(self.carrier_layers),
                eps=eps,
            )
            self.proprio_encoder = nn.Linear(proprio_dim, context_dim)
            self._joint_mot_type = JointMoT

    def _flatten_cache_tensor(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        batch: int,
    ) -> torch.Tensor:
        if tensor.ndim == 4:
            if tuple(tensor.shape[2:]) != (self.num_heads, self.attn_head_dim):
                raise ValueError(
                    f"{name} head shape must be "
                    f"[{self.num_heads},{self.attn_head_dim}]"
                )
            tensor = tensor.flatten(2, 3)
        if tensor.ndim != 3 or tensor.shape[0] != batch:
            raise ValueError(f"{name} must be [B,S,H,D] or [B,S,H*D]")
        if tensor.shape[-1] != self.attention_width:
            raise ValueError(
                f"{name} width must equal {self.attention_width}, got {tensor.shape[-1]}"
            )
        return tensor

    def _validate_cache(
        self,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        batch: int,
    ) -> list[dict[str, torch.Tensor]]:
        if set(video_kv_cache) != set(self.carrier_layers):
            raise ValueError(
                "video_kv_cache layers must exactly match carrier_layers: "
                f"{sorted(video_kv_cache)} vs {list(self.carrier_layers)}"
            )
        flattened = []
        signatures = set()
        sequence_length = None
        for layer in self.carrier_layers:
            item = video_kv_cache[layer]
            if set(item) != {"k", "v"}:
                raise ValueError(f"video_kv_cache[{layer}] must contain k and v exactly")
            tensors = {}
            for name in ("k", "v"):
                source = item[name]
                signature = _storage_signature(source)
                if signature in signatures:
                    raise ValueError(
                        "carrier cache tensors must be layer-specific; repeated/aliased "
                        f"storage detected at H3 layer {layer} {name}"
                    )
                signatures.add(signature)
                tensor = self._flatten_cache_tensor(
                    source, name=f"layer {layer} {name}", batch=batch
                )
                if sequence_length is None:
                    sequence_length = int(tensor.shape[1])
                elif tensor.shape[1] != sequence_length:
                    raise ValueError("all carrier K/V tensors must share sequence length")
                tensors[name] = tensor
            flattened.append(tensors)
        return flattened

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.action_expert is None
            or self.proprio_encoder is None
        ):
            raise RuntimeError("DreamWAM H3 K/V carrier is disabled")
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must be [B,T,D]")
        batch = int(noisy_actions.shape[0])
        if text_context.ndim != 3 or text_context.shape[0] != batch:
            raise ValueError("text_context must be [B,L,context_dim]")
        if text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context width differs from context_dim")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError("proprio must be [B,proprio_dim]")
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], device=text_context.device, dtype=torch.bool
            )
        elif tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text_mask must match text_context tokens")

        flat_cache = self._validate_cache(video_kv_cache, batch=batch)
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
        action_state = self.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = action_state["tokens"]
        JointMoT = self._joint_mot_type
        from dreamwam.layers import scaled_dot_product_attention

        for carrier_index, cache in enumerate(flat_cache):
            block = self.action_expert.blocks[carrier_index]
            action_io = JointMoT._attention_input(
                block,
                action_tokens,
                action_state["freqs"],
                action_state["time_modulation"],
            )
            video_key = cache["k"].to(
                device=action_io[1].device, dtype=action_io[1].dtype
            )
            video_value = cache["v"].to(
                device=action_io[2].device, dtype=action_io[2].dtype
            )
            mixed_attention = scaled_dot_product_attention(
                action_io[0],
                torch.cat((video_key, action_io[1]), dim=1),
                torch.cat((video_value, action_io[2]), dim=1),
                self.num_heads,
                None,
            )
            action_tokens = JointMoT._post_attention(
                block,
                action_io[3],
                mixed_attention,
                *action_io[4:],
                action_state["context"],
                action_state["context_mask"],
            )
        return self.action_expert.post_dit(action_tokens)
