"""Full-depth FastWAM ActionDiT port over the frozen H3 layer-49 carrier.

This module is a ``backbone_port`` of the ActionDiT execution path from the
official FastWAM source at :data:`FASTWAM_COMMIT`.  It does not import the
dirty vendor package or construct FastWAM's Wan video expert.  Instead, the
official ActionDiT blocks consume the already-audited, frozen H3 layer-49 K/V
carrier used by D0.

The primary initialization is deliberately *not* random.  Five trained D0
blocks are placed at uniformly-spaced depth anchors in the 30-layer tower.
All other blocks are cloned from the nearest D0 block and made exact residual
identities by zeroing their three residual output projections.  This preserves
the D0 function at step zero while giving every added block trainable capacity.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import re
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
FASTWAM_ACTION_DIT_SHA256 = (
    "1301d9224149de43bb701f620a5d41858ecc63c6b19a573ec32edd45a3bdb0a2"
)
FASTWAM_VIDEO_DIT_SHA256 = (
    "d098ad77665feeefa81634f31f5bb1d5771c4556d1a67859135f0ed35f9eb6c2"
)
FASTWAM_GRADIENT_SHA256 = (
    "ba5d8f7272eb029dc6cd2849ca99b70f6ad5abb838d21c818beb0590620dc793"
)
FASTWAM_MOT_SHA256 = (
    "9f3f09cd73bf6e4547955336cb8a9055c5345f9dbabf06361ebf139b8d8accb5"
)

DEFAULT_H3_CARRIER_LAYERS = (9, 19, 29, 39, 49)
DEFAULT_SOURCE_LAYERS = 5
DEFAULT_TARGET_LAYERS = 30
H3_BACKBONE_LAYERS = 50
LAYERWISE_H3_50_TO_ACTION_30 = tuple(
    round(index * (H3_BACKBONE_LAYERS - 1) / (DEFAULT_TARGET_LAYERS - 1))
    for index in range(DEFAULT_TARGET_LAYERS)
)
_UPSTREAM_TYPES: tuple[type[nn.Module], Any] | None = None


def _verified_source(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"pinned FastWAM source hash mismatch for {path}: {actual}"
        )
    return path


def _load_pinned_fastwam_action_dit() -> tuple[type[nn.Module], Any]:
    """Load only the byte-pinned official ActionDiT execution dependencies."""

    global _UPSTREAM_TYPES
    if _UPSTREAM_TYPES is not None:
        return _UPSTREAM_TYPES

    repo_root = Path(__file__).resolve().parents[4]
    source_override = os.environ.get("H3WAM_FASTWAM_SOURCE_ROOT")
    source_root = (
        Path(source_override).resolve()
        if source_override
        else repo_root / "third_party/FastWAM/src/fastwam/models/wan22"
    )
    package_name = "_h3wam_fastwam_45d8e145"
    package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
    package.__path__ = [str(source_root)]
    helpers_name = f"{package_name}.helpers"
    helpers = sys.modules.setdefault(helpers_name, types.ModuleType(helpers_name))
    helpers.__path__ = [str(source_root / "helpers")]
    setattr(package, "helpers", helpers)

    sources = (
        (
            f"{helpers_name}.gradient",
            source_root / "helpers/gradient.py",
            FASTWAM_GRADIENT_SHA256,
        ),
        (
            f"{package_name}.wan_video_dit",
            source_root / "wan_video_dit.py",
            FASTWAM_VIDEO_DIT_SHA256,
        ),
        (
            f"{package_name}.action_dit",
            source_root / "action_dit.py",
            FASTWAM_ACTION_DIT_SHA256,
        ),
    )
    # Upstream uses the project logger only for informational messages.  Its
    # package ``fastwam.utils.__init__`` eagerly imports video I/O dependencies
    # that are intentionally absent from the lean cached-feature runtime.  A
    # temporary, behavior-equivalent logger shim keeps source bytes untouched.
    utility_names = ("fastwam.utils", "fastwam.utils.logging_config")
    saved_utilities = {name: sys.modules.get(name) for name in utility_names}
    utils_stub = types.ModuleType("fastwam.utils")
    utils_stub.__path__ = []
    logging_stub = types.ModuleType("fastwam.utils.logging_config")
    logging_stub.get_logger = logging.getLogger
    sys.modules["fastwam.utils"] = utils_stub
    sys.modules["fastwam.utils.logging_config"] = logging_stub
    try:
        for module_name, path, digest in sources:
            source = _verified_source(path, digest)
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load pinned FastWAM module {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            owner = helpers if ".helpers." in module_name else package
            setattr(owner, module_name.rsplit(".", 1)[-1], module)
    finally:
        for name, previous in saved_utilities.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    action_module = sys.modules[f"{package_name}.action_dit"]
    video_module = sys.modules[f"{package_name}.wan_video_dit"]
    _UPSTREAM_TYPES = (action_module.ActionDiT, video_module)
    return _UPSTREAM_TYPES


def depth_anchor_indices(source_layers: int, target_layers: int) -> tuple[int, ...]:
    """Uniformly place all source blocks while retaining the two endpoints."""

    if source_layers <= 0 or target_layers < source_layers:
        raise ValueError("target_layers must be >= positive source_layers")
    if source_layers == 1:
        return (target_layers - 1,)
    anchors = tuple(
        round(index * (target_layers - 1) / (source_layers - 1))
        for index in range(source_layers)
    )
    if len(set(anchors)) != source_layers:
        raise RuntimeError("depth interpolation produced duplicate anchors")
    return anchors


def nearest_source_indices(source_layers: int, target_layers: int) -> tuple[int, ...]:
    if source_layers <= 0 or target_layers <= 0:
        raise ValueError("layer counts must be positive")
    if target_layers == 1:
        return (0,)
    return tuple(
        round(index * (source_layers - 1) / (target_layers - 1))
        for index in range(target_layers)
    )


def _storage_signature(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


@dataclass(frozen=True)
class D0ExpansionReport:
    source_layers: int
    target_layers: int
    anchor_target_indices: tuple[int, ...]
    nearest_source_indices: tuple[int, ...]
    identity_target_indices: tuple[int, ...]
    source_prefix: str
    target_prefix: str
    alpha_scaling_applied: bool
    width_interpolation_applied: bool
    initialization_contract: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class H3FastWAMFullTowerPolicy(nn.Module):
    """Official FastWAM 30-layer ActionDiT with repeated frozen H3 layer49 K/V."""

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
        num_layers: int = DEFAULT_TARGET_LAYERS,
        eps: float = 1.0e-6,
        use_gradient_checkpointing: bool = False,
        action_block_to_h3_layer: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.carrier_layers = tuple(int(layer) for layer in carrier_layers)
        if num_layers != DEFAULT_TARGET_LAYERS:
            raise ValueError(f"C58 requires the official full {DEFAULT_TARGET_LAYERS} layers")
        if action_block_to_h3_layer is None:
            if self.carrier_layers != DEFAULT_H3_CARRIER_LAYERS:
                raise ValueError(
                    f"C58 requires audited H3 cache layers {DEFAULT_H3_CARRIER_LAYERS}"
                )
            block_mapping = (49,) * num_layers
        else:
            block_mapping = tuple(int(layer) for layer in action_block_to_h3_layer)
            if block_mapping != LAYERWISE_H3_50_TO_ACTION_30:
                raise ValueError(
                    "C58b requires the exact uniform H3-50 to ActionDiT-30 mapping"
                )
            if self.carrier_layers != block_mapping:
                raise ValueError(
                    "C58b carrier_layers must exactly equal the 30-layer mapping"
                )
        dimensions = (
            action_dim,
            proprio_dim,
            context_dim,
            hidden_dim,
            ffn_dim,
            num_heads,
            attn_head_dim,
            freq_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        if attn_head_dim % 2:
            raise ValueError("attn_head_dim must be even for RoPE")

        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_width = self.num_heads * self.attn_head_dim
        self.num_layers = int(num_layers)
        self.action_block_to_h3_layer = block_mapping
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.action_expert: nn.Module | None = None
        self.proprio_encoder: nn.Module | None = None
        self._upstream = None
        if self.enabled:
            ActionDiT, upstream = _load_pinned_fastwam_action_dit()
            self.action_expert = ActionDiT(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                ffn_dim=ffn_dim,
                text_dim=context_dim,
                freq_dim=freq_dim,
                eps=eps,
                num_heads=num_heads,
                attn_head_dim=attn_head_dim,
                num_layers=num_layers,
                use_gradient_checkpointing=False,
            )
            self.proprio_encoder = nn.Linear(proprio_dim, context_dim)
            self._upstream = upstream

    def _flatten_cache_tensor(
        self, tensor: torch.Tensor, *, name: str, batch: int
    ) -> torch.Tensor:
        if tensor.ndim == 4:
            if tuple(tensor.shape[2:]) != (self.num_heads, self.attn_head_dim):
                raise ValueError(
                    f"{name} head shape must be [{self.num_heads},{self.attn_head_dim}]"
                )
            tensor = tensor.flatten(2, 3)
        if tensor.ndim != 3 or tensor.shape[0] != batch:
            raise ValueError(f"{name} must be [B,S,H,D] or [B,S,H*D]")
        if tensor.shape[-1] != self.attention_width:
            raise ValueError(
                f"{name} width must equal {self.attention_width}, got {tensor.shape[-1]}"
            )
        return tensor

    def _resolve_carrier_cache(
        self,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        batch: int,
    ) -> dict[int, dict[str, torch.Tensor]]:
        if set(video_kv_cache) != set(self.carrier_layers):
            raise ValueError(
                "video_kv_cache layers must exactly match the carrier contract"
            )
        signatures: set[int] = set()
        sequence_length: int | None = None
        result: dict[int, dict[str, torch.Tensor]] = {}
        for layer in self.carrier_layers:
            item = video_kv_cache[layer]
            if set(item) != {"k", "v"}:
                raise ValueError(f"video_kv_cache[{layer}] must contain k and v exactly")
            flattened: dict[str, torch.Tensor] = {}
            for name in ("k", "v"):
                source = item[name]
                signature = _storage_signature(source)
                if signature in signatures:
                    raise ValueError("carrier cache tensors must use independent storage")
                signatures.add(signature)
                value = self._flatten_cache_tensor(
                    source, name=f"layer {layer} {name}", batch=batch
                )
                if sequence_length is None:
                    sequence_length = int(value.shape[1])
                elif int(value.shape[1]) != sequence_length:
                    raise ValueError("all carrier K/V tensors must share sequence length")
                flattened[name] = value
            result[layer] = flattened
        return result

    def _forward_block(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        video_key: torch.Tensor,
        video_value: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self._upstream is not None
        upstream = self._upstream
        values = (
            block.modulation.to(device=t_mod.device, dtype=t_mod.dtype) + t_mod
        ).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
        attention_input = upstream.modulate(block.norm1(tokens), shift_msa, scale_msa)
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
            k=torch.cat((video_key, key), dim=1),
            v=torch.cat((video_value, value), dim=1),
            num_heads=self.num_heads,
            ctx_mask=None,
        )
        tokens = block.gate(tokens, gate_msa, block.self_attn.o(mixed))
        cross_mask = context_mask.unsqueeze(1) if context_mask.ndim == 3 else context_mask
        tokens = tokens + block.cross_attn(
            block.norm3(tokens), context, ctx_mask=cross_mask
        )
        ffn_input = upstream.modulate(block.norm2(tokens), shift_mlp, scale_mlp)
        return block.gate(tokens, gate_mlp, block.ffn(ffn_input))

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
        if not self.enabled or self.action_expert is None or self.proprio_encoder is None:
            raise RuntimeError("FastWAM full tower is disabled")
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("noisy_actions must be [B,T,action_dim]")
        batch = int(noisy_actions.shape[0])
        if text_context.ndim != 3 or tuple(text_context.shape[::2]) != (
            batch,
            self.context_dim,
        ):
            raise ValueError("text_context must be [B,L,context_dim]")
        if tuple(proprio.shape) != (batch, self.proprio_dim):
            raise ValueError("proprio must be [B,proprio_dim]")
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], dtype=torch.bool, device=text_context.device
            )
        elif tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text_mask must match text_context tokens")

        carrier = self._resolve_carrier_cache(video_kv_cache, batch=batch)
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
                torch.ones((batch, 1), dtype=torch.bool, device=context.device),
            ),
            dim=1,
        )
        state = self.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        tokens = state["tokens"]
        for block_index, block in enumerate(self.action_expert.blocks):
            layer = self.action_block_to_h3_layer[block_index]
            video_key = carrier[layer]["k"].to(
                device=tokens.device, dtype=tokens.dtype
            )
            video_value = carrier[layer]["v"].to(
                device=tokens.device, dtype=tokens.dtype
            )
            if self.use_gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    lambda x, k, v, tm, fr, ctx, mask, owner=block: self._forward_block(
                        owner, x, k, v, tm, fr, ctx, mask
                    ),
                    tokens,
                    video_key,
                    video_value,
                    state["t_mod"],
                    state["freqs"],
                    state["context"],
                    state["context_mask"],
                    use_reentrant=False,
                )
            else:
                tokens = self._forward_block(
                    block,
                    tokens,
                    video_key,
                    video_value,
                    state["t_mod"],
                    state["freqs"],
                    state["context"],
                    state["context_mask"],
                )
        return self.action_expert.post_dit(tokens, state)


def _source_block_count(state: Mapping[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}action_expert\.blocks\.(\d+)\.")
    indices = {int(match.group(1)) for key in state if (match := pattern.match(key))}
    if not indices or indices != set(range(max(indices) + 1)):
        raise ValueError("D0 source blocks must be contiguous from zero")
    return len(indices)


def _zero_residual_outputs(block: nn.Module) -> None:
    modules = (block.self_attn.o, block.cross_attn.o, block.ffn[2])
    with torch.no_grad():
        for module in modules:
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()


def initialize_full_tower_from_d0(
    policy: H3FastWAMFullTowerPolicy,
    d0_state: Mapping[str, torch.Tensor],
    *,
    source_prefix: str = "",
) -> D0ExpansionReport:
    """Strictly migrate a five-block DreamWAM D0 policy to FastWAM full30.

    No tensor width changes are allowed here.  Therefore FastWAM's official
    width interpolation and ``sqrt(d_video/d_action)`` alpha correction are
    audited upstream behaviors but are intentionally not applied to this
    function-preserving parent migration.
    """

    if not policy.enabled or policy.action_expert is None or policy.proprio_encoder is None:
        raise RuntimeError("cannot initialize a disabled full tower")
    source_layers = _source_block_count(d0_state, source_prefix)
    if source_layers != DEFAULT_SOURCE_LAYERS:
        raise ValueError(f"C58 parent must contain exactly {DEFAULT_SOURCE_LAYERS} blocks")
    target_layers = len(policy.action_expert.blocks)
    anchors = depth_anchor_indices(source_layers, target_layers)
    source_for_target = nearest_source_indices(source_layers, target_layers)
    target_state = policy.state_dict()

    shared_mapping = {
        "action_expert.action_encoder.weight": "action_expert.action_embedding.weight",
        "action_expert.action_encoder.bias": "action_expert.action_embedding.bias",
        "action_expert.text_embedding.0.weight": "action_expert.text_embedding.0.weight",
        "action_expert.text_embedding.0.bias": "action_expert.text_embedding.0.bias",
        "action_expert.text_embedding.2.weight": "action_expert.text_embedding.2.weight",
        "action_expert.text_embedding.2.bias": "action_expert.text_embedding.2.bias",
        "action_expert.time_embedding.0.weight": "action_expert.time_embedding.0.weight",
        "action_expert.time_embedding.0.bias": "action_expert.time_embedding.0.bias",
        "action_expert.time_embedding.2.weight": "action_expert.time_embedding.2.weight",
        "action_expert.time_embedding.2.bias": "action_expert.time_embedding.2.bias",
        "action_expert.time_projection.1.weight": "action_expert.time_projection.1.weight",
        "action_expert.time_projection.1.bias": "action_expert.time_projection.1.bias",
        "action_expert.head.weight": "action_expert.head.weight",
        "action_expert.head.bias": "action_expert.head.bias",
        "proprio_encoder.weight": "proprio_encoder.weight",
        "proprio_encoder.bias": "proprio_encoder.bias",
    }
    migrated: dict[str, torch.Tensor] = {}
    for target_key, source_key in shared_mapping.items():
        full_source_key = f"{source_prefix}{source_key}"
        if full_source_key not in d0_state:
            raise ValueError(f"D0 parent is missing {full_source_key}")
        source = d0_state[full_source_key]
        if tuple(source.shape) != tuple(target_state[target_key].shape):
            raise ValueError(
                f"function-preserving migration forbids resize: {full_source_key} "
                f"{tuple(source.shape)} != {tuple(target_state[target_key].shape)}"
            )
        migrated[target_key] = source.detach().to(target_state[target_key])

    block_suffixes = sorted(
        key.split("action_expert.blocks.0.", 1)[1]
        for key in d0_state
        if key.startswith(f"{source_prefix}action_expert.blocks.0.")
    )
    expected_target_suffixes = sorted(
        key.split("action_expert.blocks.0.", 1)[1]
        for key in target_state
        if key.startswith("action_expert.blocks.0.")
    )
    if block_suffixes != expected_target_suffixes:
        raise ValueError("D0 and official FastWAM block parameter schemas differ")
    for target_index, source_index in enumerate(source_for_target):
        for suffix in block_suffixes:
            target_key = f"action_expert.blocks.{target_index}.{suffix}"
            source_key = (
                f"{source_prefix}action_expert.blocks.{source_index}.{suffix}"
            )
            source = d0_state[source_key]
            if tuple(source.shape) != tuple(target_state[target_key].shape):
                raise ValueError(
                    f"function-preserving migration forbids block resize: {source_key}"
                )
            migrated[target_key] = source.detach().to(target_state[target_key])

    missing, unexpected = policy.load_state_dict(migrated, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"full-tower migration schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    anchor_set = set(anchors)
    for target_index, block in enumerate(policy.action_expert.blocks):
        if target_index not in anchor_set:
            _zero_residual_outputs(block)

    return D0ExpansionReport(
        source_layers=source_layers,
        target_layers=target_layers,
        anchor_target_indices=anchors,
        nearest_source_indices=source_for_target,
        identity_target_indices=tuple(
            index for index in range(target_layers) if index not in anchor_set
        ),
        source_prefix=source_prefix,
        target_prefix="",
        alpha_scaling_applied=False,
        width_interpolation_applied=False,
        initialization_contract="exact_d0_function_preserving_residual_identity_v1",
    )


def fastwam_official_alpha(source_width: int, target_width: int) -> float:
    """The exact width-correction scalar used by the official preprocessing CLI."""

    if source_width <= 0 or target_width <= 0:
        raise ValueError("widths must be positive")
    return math.sqrt(float(source_width) / float(target_width))
