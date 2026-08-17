"""Light-WAM state-fusion action decoder port over frozen H3 K/V states.

This module byte-loads the official ``StateFusionActionExpert`` implementation
from the pinned Light-WAM repository.  The H3 port deliberately keeps the
backbone frozen and maps Light-WAM's three adapter taps (8/16/24 of 30 Wan
blocks) to H3 blocks 14/27/41.  H3 value states stand in for the official
adapted hidden states; backbone LoRA and prefix-only loss weighting are not
part of this architecture port and must be tested as separate variables.

Light-WAM's state-fusion expert is a direct action regressor.  It therefore
does not consume noisy actions or an action diffusion timestep.  Those inputs
remain in the forward signature only so the policy can share the audited H3
feature-provider contract with the existing flow policies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


LIGHTWAM_COMMIT = "b2785f66e13fd9987e94ae1ecc1c441d5059c9ae"
LIGHTWAM_STATE_FUSION_SHA256 = (
    "f0f1f73e947e9cc5342b3e661a7e6d32a0a445488591c5c2d5b030c515de7b58"
)
LIGHTWAM_WAN_ADAPTER_LAYERS = (8, 16, 24)
LIGHTWAM_H3_CARRIER_LAYERS = (14, 27, 41)
LIGHTWAM_FEATURE_SOURCE = "backbone"
_PINNED_STATE_FUSION_TYPE: type[nn.Module] | None = None


def _verified_source(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"pinned Light-WAM source hash mismatch for {path}: {actual}"
        )
    return path


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Byte-equivalent helper required by the isolated upstream module."""

    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(
                dim // 2,
                dtype=torch.float64,
                device=position.device,
            ).div(dim // 2),
        ),
    )
    output = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return output.to(position.dtype)


def _load_pinned_lightwam_state_fusion() -> type[nn.Module]:
    """Load the byte-pinned official class without importing Light-WAM runtime."""

    global _PINNED_STATE_FUSION_TYPE
    if _PINNED_STATE_FUSION_TYPE is not None:
        return _PINNED_STATE_FUSION_TYPE

    repo_root = Path(__file__).resolve().parents[4]
    source_override = os.environ.get("H3WAM_LIGHTWAM_SOURCE_ROOT")
    source_root = (
        Path(source_override).resolve()
        if source_override
        else repo_root / "third_party/Light-WAM/src/lightwam/models/wan22"
    )
    source = _verified_source(
        source_root / "state_fusion_action_expert.py",
        LIGHTWAM_STATE_FUSION_SHA256,
    )

    package_name = "_h3wam_lightwam_b2785f66"
    package = sys.modules.setdefault(package_name, types.ModuleType(package_name))
    package.__path__ = [str(source_root)]
    helper_name = f"{package_name}.wan_video_dit"
    helper = types.ModuleType(helper_name)
    helper.sinusoidal_embedding_1d = _sinusoidal_embedding_1d
    sys.modules[helper_name] = helper
    setattr(package, "wan_video_dit", helper)

    module_name = f"{package_name}.state_fusion_action_expert"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned Light-WAM module {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    setattr(package, "state_fusion_action_expert", module)
    _PINNED_STATE_FUSION_TYPE = module.StateFusionActionExpert
    return _PINNED_STATE_FUSION_TYPE


def _storage_signature(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class H3LightWAMStateFusionPolicy(nn.Module):
    """Official Light-WAM shallow decoder consuming selected frozen H3 V states."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        carrier_layers: Sequence[int] = LIGHTWAM_H3_CARRIER_LAYERS,
        action_dim: int = 7,
        proprio_dim: int = 8,
        num_heads: int = 56,
        attn_head_dim: int = 128,
        per_layer_dim: int = 4608,
        trunk_dim: int = 6144,
        num_trunk_blocks: int = 1,
        step_pos_dim: int = 256,
        token_pooling_num_queries: int = 16,
        token_pooling_num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.carrier_layers = tuple(int(layer) for layer in carrier_layers)
        if len(self.carrier_layers) != len(LIGHTWAM_WAN_ADAPTER_LAYERS):
            raise ValueError("Light-WAM H3 port requires exactly three carrier layers")
        if tuple(sorted(set(self.carrier_layers))) != self.carrier_layers:
            raise ValueError("carrier_layers must be strictly increasing and unique")
        if self.carrier_layers[0] < 0 or self.carrier_layers[-1] >= 50:
            raise ValueError("carrier_layers must select H3 blocks in [0,49]")
        dimensions = (
            action_dim,
            proprio_dim,
            num_heads,
            attn_head_dim,
            per_layer_dim,
            trunk_dim,
            step_pos_dim,
            token_pooling_num_queries,
            token_pooling_num_heads,
        )
        if min(dimensions) <= 0:
            raise ValueError("model dimensions must be positive")
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attention_width = self.num_heads * self.attn_head_dim
        self.direct_action_regression = True
        self.state_fusion_action_expert: nn.Module | None = None
        self.proprio_encoder: nn.Module | None = None
        if self.enabled:
            StateFusionActionExpert = _load_pinned_lightwam_state_fusion()
            self.state_fusion_action_expert = StateFusionActionExpert(
                video_hidden_dim=self.attention_width,
                action_dim=self.action_dim,
                num_fusion_layers=len(self.carrier_layers),
                per_layer_dim=per_layer_dim,
                trunk_dim=trunk_dim,
                num_trunk_blocks=num_trunk_blocks,
                step_pos_dim=step_pos_dim,
                token_pooling_type="learned_query",
                token_pooling_num_queries=token_pooling_num_queries,
                token_pooling_num_heads=token_pooling_num_heads,
                feature_sources=(LIGHTWAM_FEATURE_SOURCE,),
            )
            # Official Light-WAM injects proprio as a text-context token before
            # the Wan backbone.  Frozen H3 has no robot-state input, so the
            # port appends an explicitly trainable proprio token to every H3
            # layer state before the unmodified upstream query pooler.
            self.proprio_encoder = nn.Linear(
                self.proprio_dim,
                self.attention_width,
                bias=False,
            )

    def _resolve_layer_states(
        self,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        batch: int,
        proprio: torch.Tensor,
    ) -> list[dict[str, Any]]:
        if set(video_kv_cache) != set(self.carrier_layers):
            raise ValueError(
                "video_kv_cache layers must exactly match the Light-WAM carrier contract"
            )
        if self.proprio_encoder is None:
            raise RuntimeError("enabled policy is missing proprio_encoder")
        parameter = self.proprio_encoder.weight
        proprio_token = self.proprio_encoder(
            proprio.to(device=parameter.device, dtype=parameter.dtype)
        ).unsqueeze(1)
        signatures: set[int] = set()
        sequence_length: int | None = None
        states: list[dict[str, Any]] = []
        for layer in self.carrier_layers:
            item = video_kv_cache[layer]
            if set(item) != {"k", "v"}:
                raise ValueError(f"video_kv_cache[{layer}] must contain k and v exactly")
            for name in ("k", "v"):
                source = item[name]
                signature = _storage_signature(source)
                if signature in signatures:
                    raise ValueError("carrier cache tensors must use independent storage")
                signatures.add(signature)
            value = item["v"]
            if value.ndim == 4:
                if tuple(value.shape[2:]) != (self.num_heads, self.attn_head_dim):
                    raise ValueError(
                        f"video_kv_cache[{layer}].v head shape must be "
                        f"[{self.num_heads},{self.attn_head_dim}]"
                    )
                value = value.flatten(2, 3)
            if value.ndim != 3 or value.shape[0] != batch:
                raise ValueError(
                    f"video_kv_cache[{layer}].v must be [B,S,H,D] or [B,S,H*D]"
                )
            if value.shape[-1] != self.attention_width:
                raise ValueError(
                    f"video_kv_cache[{layer}].v width must be {self.attention_width}"
                )
            if sequence_length is None:
                sequence_length = int(value.shape[1])
            elif int(value.shape[1]) != sequence_length:
                raise ValueError("all Light-WAM carrier layers must share sequence length")
            value = value.to(device=parameter.device, dtype=parameter.dtype)
            states.append(
                {
                    LIGHTWAM_FEATURE_SOURCE: torch.cat(
                        [value, proprio_token],
                        dim=1,
                    )
                }
            )
        return states

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
        if not self.enabled or self.state_fusion_action_expert is None:
            raise RuntimeError("H3 Light-WAM state-fusion policy is disabled")
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("noisy_actions must be [B,T,action_dim]")
        batch, action_horizon, _ = noisy_actions.shape
        if timestep.ndim != 1 or timestep.shape[0] not in {1, batch}:
            raise ValueError("timestep must be [1] or [B]")
        if text_context.ndim != 3 or text_context.shape[0] != batch:
            raise ValueError("text_context must be [B,S,D]")
        if text_mask is not None and tuple(text_mask.shape[:1]) != (batch,):
            raise ValueError("text_mask batch must match noisy_actions")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError(
                f"proprio must be [{batch},{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        layer_states = self._resolve_layer_states(
            video_kv_cache,
            batch=batch,
            proprio=proprio,
        )
        return self.state_fusion_action_expert(
            layer_states,
            action_horizon=action_horizon,
        )

