"""Faster-WAM style Dock-of-Transformer modules for MiniMax-H3.

The implementation follows arXiv:2608.02365: cache video K/V from every H3
layer, undo the H3 rotary basis, remap channels, mix layers independently per
attention head, and express the fused video keys in the action head's 1D RoPE
basis.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .action_expert import H3DreamActionAttention, _attention, sinusoidal_embedding
from .model import apply_h3_rotary


def inverse_h3_rotary(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Undo H3's orthogonal 3D rotary transform."""

    return apply_h3_rotary(hidden, cos, -sin)


def action_rope_at_positions(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply action-side 1D RoPE at explicit positions.

    ``tensor`` is ``[B,S,H,D]`` and ``positions`` is ``[S]``.  Explicit
    positions are needed because docked video tokens and action tokens occupy
    different sequences while sharing the same rotary basis.
    """

    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError("action RoPE expects [B,S,H,even_head_dim]")
    if positions.shape != (tensor.shape[1],):
        raise ValueError("positions must match the token sequence")
    head_dim = tensor.shape[-1]
    frequency = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=tensor.device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )
    phase = positions.to(device=tensor.device, dtype=torch.float32)[:, None]
    phase = phase * frequency[None]
    cos = phase.cos()[None, :, None]
    sin = phase.sin()[None, :, None]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    output = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    )
    return output.flatten(-2).to(tensor.dtype)


class H3DoTKVFusion(nn.Module):
    """Channel-remap and head-wise layer fusion from Faster-WAM."""

    def __init__(
        self,
        *,
        video_layers: int,
        action_layers: int,
        video_num_heads: int,
        video_head_dim: int,
        action_num_heads: int,
        action_head_dim: int,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if min(
            video_layers,
            action_layers,
            video_num_heads,
            video_head_dim,
            action_num_heads,
            action_head_dim,
        ) <= 0:
            raise ValueError("DoT dimensions must be positive")
        if video_head_dim != action_head_dim:
            raise ValueError(
                "video/action head dimensions must match for RoPE realignment"
            )
        self.video_layers = int(video_layers)
        self.action_layers = int(action_layers)
        self.video_num_heads = int(video_num_heads)
        self.video_head_dim = int(video_head_dim)
        self.action_num_heads = int(action_num_heads)
        self.action_head_dim = int(action_head_dim)
        self.video_inner_dim = self.video_num_heads * self.video_head_dim
        self.action_inner_dim = self.action_num_heads * self.action_head_dim
        # Equations (2): separate full-width channel maps for K and V.
        # This map is deliberately rectangular when the hub and head widths
        # differ. Faster-WAM uses a 3072-wide action head even though the hub
        # has its own attention width; matching H3's 7168 width here would
        # defeat the purpose of the docking interface.
        self.key_channel = nn.Linear(
            self.video_inner_dim, self.action_inner_dim, bias=False
        )
        self.value_channel = nn.Linear(
            self.video_inner_dim, self.action_inner_dim, bias=False
        )
        # Equation (3): one cross-layer signal per action layer and head.
        # The paper defines an unconstrained aggregation matrix A_h rather
        # than normalized attention weights. A uniform mean is a stable H3
        # initialization while preserving the exact linear parameterization.
        self.layer_mix = nn.Parameter(
            torch.full(
                (self.action_layers, self.action_num_heads, self.video_layers),
                1.0 / self.video_layers,
            )
        )
        self.key_norm = nn.RMSNorm(self.action_head_dim, eps=eps)

    def forward(
        self,
        *,
        rotated_video_keys: torch.Tensor,
        video_values: torch.Tensor,
        video_cos: torch.Tensor,
        video_sin: torch.Tensor,
        action_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse caches into ``[La,B,S,H,D]`` action-side K/V tensors."""

        expected_rank = 5
        if rotated_video_keys.ndim != expected_rank:
            raise ValueError("video keys must be [Lv,B,S,H,D]")
        if video_values.shape != rotated_video_keys.shape:
            raise ValueError("video K/V cache shapes must match")
        layers, batch, sequence, heads, head_dim = rotated_video_keys.shape
        if (layers, heads, head_dim) != (
            self.video_layers,
            self.video_num_heads,
            self.video_head_dim,
        ):
            raise ValueError("video cache does not match DoT configuration")
        if video_cos.shape[0] != sequence or video_sin.shape != video_cos.shape:
            raise ValueError("video rotary cache must match the video token sequence")

        # Equations (18-19): unrotate before channel/layer fusion.
        canonical_keys = torch.stack(
            [
                inverse_h3_rotary(layer, video_cos, video_sin)
                for layer in rotated_video_keys.unbind(0)
            ],
            dim=0,
        )
        action_shape = (
            layers,
            batch,
            sequence,
            self.action_num_heads,
            self.action_head_dim,
        )
        mixed_keys = self.key_channel(canonical_keys.flatten(-2)).reshape(action_shape)
        mixed_values = self.value_channel(video_values.flatten(-2)).reshape(action_shape)
        fused_keys = torch.einsum("ahl,lbshd->abshd", self.layer_mix, mixed_keys)
        fused_values = torch.einsum("ahl,lbshd->abshd", self.layer_mix, mixed_values)
        fused_keys = self.key_norm(fused_keys)

        if action_positions is None:
            # Conditioning-frame tokens form a position-zero prefix. Spatial
            # identity remains encoded in their H3 representation; assigning
            # a shared zero prevents an arbitrary flattened spatial index from
            # masquerading as action time.
            action_positions = torch.zeros(
                sequence,
                device=fused_keys.device,
                dtype=torch.long,
            )
        if action_positions.shape != (sequence,):
            raise ValueError("docked video positions must be [S]")
        fused_keys = torch.stack(
            [
                action_rope_at_positions(layer, action_positions)
                for layer in fused_keys.unbind(0)
            ],
            dim=0,
        )
        return fused_keys, fused_values


class H3DoTActionLayer(nn.Module):
    """A single lightweight action DiT layer without text cross-attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1.0e-5,
        full_width_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.attn = H3DreamActionAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            full_width_rmsnorm=full_width_rmsnorm,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / math.sqrt(hidden_dim)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        time_modulation: torch.Tensor,
        docked_key: torch.Tensor,
        docked_value: torch.Tensor,
    ) -> torch.Tensor:
        if time_modulation.shape[1:] != (6, self.hidden_dim):
            raise ValueError("time modulation must be [B,6,hidden_dim]")
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            self.modulation + time_modulation
        ).chunk(6, dim=1)
        normalized = self.norm1(tokens) * (1.0 + scale_a) + shift_a
        action_query, action_key, action_value = self.attn.qkv(normalized)
        if docked_key.shape[:1] != tokens.shape[:1]:
            raise ValueError("docked KV batch does not match action tokens")
        if docked_key.shape != docked_value.shape:
            raise ValueError("docked K/V shapes must match")
        mixed_key = torch.cat((action_key, docked_key), dim=1)
        mixed_value = torch.cat((action_value, docked_value), dim=1)
        attended = _attention(action_query, mixed_key, mixed_value)
        tokens = tokens + gate_a * self.attn.to_out(attended.flatten(2))
        normalized = self.norm2(tokens) * (1.0 + scale_f) + shift_f
        return tokens + gate_f * self.ffn(normalized)


class H3DoTActionHead(nn.Module):
    """Faster-WAM action head: normally one Transformer layer, no text path."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 56,
        head_dim: int = 128,
        num_layers: int = 1,
        frequency_dim: int = 256,
        eps: float = 1.0e-5,
        full_width_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        if min(action_dim, hidden_dim, ffn_dim, num_layers) <= 0:
            raise ValueError("action head dimensions must be positive")
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.frequency_dim = int(frequency_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                H3DoTActionLayer(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    eps=eps,
                    full_width_rmsnorm=full_width_rmsnorm,
                )
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        *,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        docked_keys: torch.Tensor,
        docked_values: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError("noisy actions must be [B,H,action_dim]")
        if timestep.shape != (noisy_actions.shape[0],):
            raise ValueError("action timestep must be [B]")
        if docked_keys.shape[0] != len(self.layers):
            raise ValueError("one docked cache is required per action layer")
        time_features = sinusoidal_embedding(self.frequency_dim, timestep).to(
            self.time_embedding[0].weight.dtype
        )
        time_hidden = self.time_embedding(time_features)
        time_modulation = self.time_projection(time_hidden).reshape(
            noisy_actions.shape[0], 6, self.hidden_dim
        )
        tokens = self.action_embedding(noisy_actions)
        for index, layer in enumerate(self.layers):
            tokens = layer(
                tokens,
                time_modulation=time_modulation,
                docked_key=docked_keys[index],
                docked_value=docked_values[index],
            )
        return self.output(tokens)
