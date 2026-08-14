"""Project-native MiniMax-H3 pruned INT8 feature backbone.

The public Diffusers MiniMax-H3 call contract is retained, while the released
pruned checkpoint's fused QKV, curve AdaLN and ConvRot INT8 layout are loaded
directly. No ComfyUI package, server, node registry or workflow is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from .int8_linear import ConvRotInt8Linear


HIDDEN_SIZE = 5376
HEADS = 56
HEAD_DIM = 128
MODALITY_COUNT = 3
NORM_EPS = 1e-5


class FrozenLinear(nn.Module):
    """Checkpoint linear stored only as frozen buffers."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("linear weight must be two-dimensional")
        if bias is not None and tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("linear bias shape mismatch")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("bias", None if bias is None else bias.contiguous())

    def forward(self, x: torch.Tensor, *, input_act: str | None = None) -> torch.Tensor:
        if input_act == "swiglu":
            if x.shape[-1] != self.in_features * 2:
                raise ValueError("SwiGLU input must be twice the linear input width")
            gate, up = x.chunk(2, dim=-1)
            x = F.silu(gate) * up
        elif input_act is not None:
            raise ValueError(f"unsupported input activation {input_act!r}")
        if x.shape[-1] != self.in_features:
            raise ValueError("linear input width mismatch")
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


class FrozenRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = NORM_EPS) -> None:
        super().__init__()
        if weight.ndim != 1:
            raise ValueError("RMSNorm weight must be one-dimensional")
        self.eps = float(eps)
        self.register_buffer("weight", weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, (self.weight.numel(),), self.weight.to(x.dtype), self.eps
        )


class _TensorReader:
    def __init__(self, checkpoint) -> None:
        self.checkpoint = checkpoint
        self.keys = set(checkpoint.keys())
        self.consumed: set[str] = set()

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            raise KeyError(f"H3 INT8 checkpoint is missing {key!r}")
        self.consumed.add(key)
        return self.checkpoint.get_tensor(key)

    def optional(self, key: str) -> torch.Tensor | None:
        return self.tensor(key) if key in self.keys else None

    def linear(self, prefix: str) -> nn.Module:
        if f"{prefix}.comfy_quant" in self.keys:
            return ConvRotInt8Linear.from_checkpoint_tensors(
                weight=self.tensor(f"{prefix}.weight"),
                weight_scale=self.tensor(f"{prefix}.weight_scale"),
                marker=self.tensor(f"{prefix}.comfy_quant"),
                bias=self.optional(f"{prefix}.bias"),
            )
        return FrozenLinear(
            self.tensor(f"{prefix}.weight"), self.optional(f"{prefix}.bias")
        )

    def norm(self, prefix: str) -> FrozenRMSNorm:
        return FrozenRMSNorm(self.tensor(f"{prefix}.weight"))


def _rotary_cos_sin(
    position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = position_ids.to(device=inv_freq.device, dtype=torch.float32)
    per_axis = pos.unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    t_freq, h_freq, w_freq = per_axis.unbind(dim=1)
    half = torch.cat((t_freq, h_freq, w_freq), dim=-1)
    angles = torch.cat((half, half), dim=-1)
    return angles.cos(), angles.sin()


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = cos.to(x.dtype)[None, :, None]
    sin = sin.to(x.dtype)[None, :, None]
    return torch.cat((rotary * cos + rotated * sin, tail), dim=-1)


class H3Int8Attention(nn.Module):
    def __init__(self, reader: _TensorReader, prefix: str) -> None:
        super().__init__()
        self.qkv_proj = reader.linear(f"{prefix}.qkv_proj")
        self.out_proj = reader.linear(f"{prefix}.out_proj")
        self.q_norm = reader.norm(f"{prefix}.q_norm")
        self.k_norm = reader.norm(f"{prefix}.k_norm")

    def project_qkv(
        self,
        x: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = x.shape
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        query = self.q_norm(query.view(batch, sequence, HEADS, HEAD_DIM))
        key = self.k_norm(key.view(batch, sequence, HEADS, HEAD_DIM))
        value = value.view(batch, sequence, HEADS, HEAD_DIM)
        if rotary is not None:
            query = _apply_rotary(query, *rotary)
            key = _apply_rotary(key, *rotary)
        return query, key, value

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence = query.shape[:2]
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, sequence, HEADS * HEAD_DIM
        )
        return self.out_proj(attended)

    def forward(
        self,
        x: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query, key, value = self.project_qkv(x, rotary)
        output = self.attend_projected(query, key, value)
        if return_kv:
            return output, {"k": key, "v": value}
        return output


class H3Int8MLP(nn.Module):
    def __init__(self, reader: _TensorReader, prefix: str) -> None:
        super().__init__()
        self.fc1 = reader.linear(f"{prefix}.fc1")
        self.fc2 = reader.linear(f"{prefix}.fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x), input_act="swiglu")


class H3Int8RefinerBlock(nn.Module):
    def __init__(self, reader: _TensorReader, prefix: str) -> None:
        super().__init__()
        self.norm1 = reader.norm(f"{prefix}.norm1")
        self.attn = H3Int8Attention(reader, f"{prefix}.attn")
        self.norm2 = reader.norm(f"{prefix}.norm2")
        self.mlp = H3Int8MLP(reader, f"{prefix}.mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), None)
        return x + self.mlp(self.norm2(x))


class H3Int8TokenRefiner(nn.Module):
    def __init__(self, reader: _TensorReader) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                H3Int8RefinerBlock(reader, f"token_refiner.blocks.{index}")
                for index in range(2)
            ]
        )
        self.final_norm = reader.norm("token_refiner.final_norm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class H3Int8Block(nn.Module):
    def __init__(self, reader: _TensorReader, index: int) -> None:
        super().__init__()
        prefix = f"blocks.{index}"
        self.norm1 = reader.norm(f"{prefix}.norm1")
        self.attn = H3Int8Attention(reader, f"{prefix}.attn")
        self.norm2 = reader.norm(f"{prefix}.norm2")
        self.mlp = H3Int8MLP(reader, f"{prefix}.mlp")
        self.adaln = reader.linear(f"{prefix}.adaln_proj.linear")

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        modulation = self.adaln(temb).view(-1, 6, HIDDEN_SIZE)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.unbind(1)
        shift_a = shift_a.index_select(0, adaln_indices).to(x.dtype)
        scale_a = scale_a.index_select(0, adaln_indices).to(x.dtype)
        gate_a = gate_a.index_select(0, adaln_indices).to(x.dtype)
        normed = self.norm1(x) * (1 + scale_a) + shift_a
        attention = self.attn(normed, rotary, return_kv=return_kv)
        if return_kv:
            attention_output, layer_kv = attention
        else:
            attention_output = attention
            layer_kv = None
        x = x + gate_a * attention_output
        shift_m = shift_m.index_select(0, adaln_indices).to(x.dtype)
        scale_m = scale_m.index_select(0, adaln_indices).to(x.dtype)
        gate_m = gate_m.index_select(0, adaln_indices).to(x.dtype)
        normed = self.norm2(x) * (1 + scale_m) + shift_m
        output = x + gate_m * self.mlp(normed)
        if return_kv:
            return output, layer_kv
        return output


@dataclass
class H3Int8FeatureOutput:
    hidden_states: torch.Tensor
    captured_features: dict[int, torch.Tensor]
    captured_kv: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


def _prepare_text_hidden_states(
    encoder_hidden_states: torch.Tensor,
    condition_proj: nn.Module,
    token_refiner: nn.Module,
) -> torch.Tensor:
    """Normalize raw Qwen or already-refined H3 context for sequence packing.

    Historical ComfyUI feature caches store the output of ``condition_proj``
    and ``token_refiner`` at width 5376.  The native cache path instead stores
    raw Qwen embeddings at the projection input width.  Accepting both keeps
    the two contracts explicit and, crucially, never refines historical
    context for a second time.
    """

    if encoder_hidden_states.ndim != 3:
        raise ValueError("encoder_hidden_states must be [batch,tokens,width]")
    width = int(encoder_hidden_states.shape[-1])
    if width == HIDDEN_SIZE:
        return encoder_hidden_states
    input_width = int(condition_proj.in_features)
    if width != input_width:
        raise ValueError(
            "encoder context width must be raw Qwen "
            f"({input_width}) or refined H3 ({HIDDEN_SIZE}), got {width}"
        )
    return token_refiner(condition_proj(encoder_hidden_states))


class H3Int8FeatureBackbone(nn.Module):
    """Frozen H3 backbone exposing selected packed-sequence block features."""

    def __init__(self, reader: _TensorReader) -> None:
        super().__init__()
        self.video_patch_proj = reader.linear("video_patch_proj")
        self.audio_patch_proj = reader.linear("audio_patch_proj")
        self.condition_proj = reader.linear("condition_proj")
        self.token_refiner = H3Int8TokenRefiner(reader)
        self.blocks = nn.ModuleList([H3Int8Block(reader, i) for i in range(50)])
        self.register_buffer(
            "adaln_t_table", reader.tensor("adaln_t_table").contiguous()
        )
        self.register_buffer(
            "rope_inv_freq", reader.tensor("rope.inv_freq").contiguous()
        )
        self.loaded_checkpoint_keys = frozenset(reader.consumed)
        self.ignored_checkpoint_keys = frozenset(reader.keys - reader.consumed)
        unexpected = {
            key for key in self.ignored_checkpoint_keys if not key.startswith("final_layer.")
        }
        if unexpected:
            raise ValueError(
                "H3 feature backbone left unexpected checkpoint tensors unmapped: "
                f"{sorted(unexpected)[:8]}"
            )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "H3Int8FeatureBackbone":
        with safe_open(Path(path).resolve(), framework="pt", device="cpu") as checkpoint:
            model = cls(_TensorReader(checkpoint))
        model.requires_grad_(False)
        return model

    def _curve_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        table = self.adaln_t_table
        position = timestep.float().clamp(0, 1) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(
            table.index_select(0, lower),
            table.index_select(0, lower + 1),
            (position - lower).unsqueeze(1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        *,
        capture_layers: Collection[int] = (9, 19, 29, 39, 49),
        capture_indices: torch.Tensor | None = None,
        capture_kv_layers: Collection[int] = (),
        kv_capture_indices: torch.Tensor | None = None,
    ) -> H3Int8FeatureOutput:
        if hidden_states.shape[0] != 1:
            raise ValueError("the pruned H3 checkpoint supports one packed sequence")
        sequence_length = int(position_ids.shape[0])
        if tuple(position_ids.shape) != (sequence_length, 3):
            raise ValueError("position_ids must be [sequence,3]")
        if tuple(timestep_indices.shape) != (sequence_length,):
            raise ValueError("timestep_indices must match the packed sequence")
        if tuple(token_tags.shape) != (sequence_length,):
            raise ValueError("token_tags must match the packed sequence")
        selected = {int(index) for index in capture_layers}
        kv_selected = {int(index) for index in capture_kv_layers}
        if selected and (min(selected) < 0 or max(selected) >= 50):
            raise ValueError("capture_layers must select H3 blocks in [0,49]")
        if kv_selected and (min(kv_selected) < 0 or max(kv_selected) >= 50):
            raise ValueError("capture_kv_layers must select H3 blocks in [0,49]")
        if not selected and not kv_selected:
            raise ValueError("at least one feature or K/V capture layer is required")

        video = self.video_patch_proj(hidden_states)
        audio = self.audio_patch_proj(audio_hidden_states)
        text = _prepare_text_hidden_states(
            encoder_hidden_states, self.condition_proj, self.token_refiner
        )
        if text.shape[1] != text_indices.numel():
            raise ValueError(
                "encoder context token count does not match packed text indices"
            )
        packed = text.new_zeros((1, sequence_length, HIDDEN_SIZE))
        text_indices = text_indices.to(device=packed.device, dtype=torch.long)
        video_indices = video_indices.to(device=packed.device, dtype=torch.long)
        audio_indices = audio_indices.to(device=packed.device, dtype=torch.long)
        packed.index_copy_(1, text_indices, text)
        packed.index_copy_(1, video_indices, video.to(text.dtype))
        packed.index_copy_(1, audio_indices, audio.to(text.dtype))

        temb = self._curve_embedding(timestep.to(self.adaln_t_table.device))
        adaln_indices = (
            timestep_indices.to(device=packed.device, dtype=torch.long) * MODALITY_COUNT
            + token_tags.to(device=packed.device, dtype=torch.long)
        )
        rotary = _rotary_cos_sin(position_ids, self.rope_inv_freq)
        captures: dict[int, torch.Tensor] = {}
        kv_captures: dict[int, dict[str, torch.Tensor]] = {}
        for index, block in enumerate(self.blocks):
            block_output = block(
                packed,
                temb,
                adaln_indices,
                rotary,
                return_kv=index in kv_selected,
            )
            if index in kv_selected:
                packed, layer_kv = block_output
                kv_indices = (
                    video_indices
                    if kv_capture_indices is None
                    else kv_capture_indices.to(device=packed.device, dtype=torch.long)
                )
                kv_captures[index] = {
                    name: value.index_select(1, kv_indices)
                    for name, value in layer_kv.items()
                }
            else:
                packed = block_output
            if index in selected:
                captures[index] = (
                    packed
                    if capture_indices is None
                    else packed.index_select(
                        1, capture_indices.to(device=packed.device, dtype=torch.long)
                    )
                )
        return H3Int8FeatureOutput(packed, captures, kv_captures)
