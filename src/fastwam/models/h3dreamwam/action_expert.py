"""DreamWAM-style ActionDiT aligned to MiniMax-H3 attention geometry."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(dim: int, timestep: torch.Tensor) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError("sinusoidal embedding dimension must be positive and even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    phase = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)


def action_rope(
    tensor: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply 1D RoPE to `[B,T,H,D]` action query/key tensors."""

    if tensor.ndim != 4 or tensor.shape[-1] % 2:
        raise ValueError("action RoPE expects [B,T,H,even_head_dim]")
    sequence = tensor.shape[1]
    head_dim = tensor.shape[-1]
    frequency = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32)
            / head_dim
        )
    )
    phase = torch.outer(
        torch.arange(sequence, device=tensor.device, dtype=torch.float32),
        frequency,
    )
    cos = phase.cos()[None, :, None, :]
    sin = phase.sin()[None, :, None, :]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(tensor.dtype)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2)


class H3DreamActionAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        full_width_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.full_width_rmsnorm = bool(full_width_rmsnorm)
        self.to_q = nn.Linear(hidden_dim, self.inner_dim)
        self.to_k = nn.Linear(hidden_dim, self.inner_dim)
        self.to_v = nn.Linear(hidden_dim, self.inner_dim)
        norm_dim = self.inner_dim if self.full_width_rmsnorm else self.head_dim
        self.norm_q = nn.RMSNorm(norm_dim, eps=eps)
        self.norm_k = nn.RMSNorm(norm_dim, eps=eps)
        self.to_out = nn.Linear(self.inner_dim, hidden_dim)

    def qkv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        query = self.to_q(hidden)
        key = self.to_k(hidden)
        if self.full_width_rmsnorm:
            query = self.norm_q(query).reshape(shape)
            key = self.norm_k(key).reshape(shape)
        else:
            query = self.norm_q(query.reshape(shape))
            key = self.norm_k(key.reshape(shape))
        value = self.to_v(hidden).reshape(shape)
        return action_rope(query), action_rope(key), value


class H3DreamCrossAttention(H3DreamActionAttention):
    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, action_length, _ = hidden.shape
        context_length = context.shape[1]
        query = self.to_q(hidden)
        key = self.to_k(context)
        if self.full_width_rmsnorm:
            query = self.norm_q(query)
            key = self.norm_k(key)
        query = query.reshape(batch, action_length, self.num_heads, self.head_dim)
        key = key.reshape(batch, context_length, self.num_heads, self.head_dim)
        if not self.full_width_rmsnorm:
            query = self.norm_q(query)
            key = self.norm_k(key)
        value = self.to_v(context).reshape(
            batch, context_length, self.num_heads, self.head_dim
        )
        mask = context_mask
        if mask is not None:
            if mask.shape == (batch, context_length):
                mask = mask[:, None, None, :]
            elif mask.shape == (batch, action_length, context_length):
                mask = mask[:, None]
            else:
                raise ValueError("context mask has an incompatible shape")
        attended = _attention(query, key, value, mask)
        return self.to_out(attended.flatten(2))


class H3DreamActionBlock(nn.Module):
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
        self.cross_attn = H3DreamCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            full_width_rmsnorm=full_width_rmsnorm,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / math.sqrt(hidden_dim)
        )
        # MiniWorld-style zero-initialized, layer-local control residual. The
        # existing joint video/action attention remains the checkpoint-
        # compatible base path, so this route can be introduced without
        # perturbing the inherited function at step zero.
        self.video_residual_gate = nn.Parameter(
            torch.zeros(1, 1, self.attn.num_heads, 1)
        )
        # MiniWorld-style low-rank conditioning route. The output projection
        # is zero-initialized, so adding this capacity cannot perturb an
        # inherited ActionDiT function before optimization.
        adapter_rank = 16
        self.video_residual_adapter = nn.Sequential(
            nn.Linear(self.attn.inner_dim, adapter_rank, bias=False),
            nn.SiLU(),
            nn.Linear(adapter_rank, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.video_residual_adapter[-1].weight)

    def attention_input(
        self,
        tokens: torch.Tensor,
        time_modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if time_modulation.ndim != 3 or time_modulation.shape[1:] != (
            6,
            self.hidden_dim,
        ):
            raise ValueError("action time modulation must be [B,6,hidden_dim]")
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation + time_modulation
        ).chunk(6, dim=1)
        attention_hidden = self.norm1(tokens) * (1.0 + scale_attn) + shift_attn
        query, key, value = self.attn.qkv(attention_hidden)
        return (
            query,
            key,
            value,
            tokens,
            gate_attn,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        )

    def post_attention(
        self,
        *,
        residual: torch.Tensor,
        attended: torch.Tensor,
        gate_attn: torch.Tensor,
        shift_ffn: torch.Tensor,
        scale_ffn: torch.Tensor,
        gate_ffn: torch.Tensor,
        video_residual: torch.Tensor | None = None,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = residual + gate_attn * self.attn.to_out(attended.flatten(2))
        if video_residual is not None:
            tokens = tokens + video_residual
        tokens = tokens + self.cross_attn(self.norm3(tokens), context, context_mask)
        ffn_hidden = self.norm2(tokens) * (1.0 + scale_ffn) + shift_ffn
        return tokens + gate_ffn * self.ffn(ffn_hidden)


def load_action_block_state(
    block: H3DreamActionBlock,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """Load a staged block and migrate checkpoints predating video gates.

    Returns ``True`` when a legacy checkpoint was upgraded by inserting the
    zero-initialized residual gate. All other architecture drift remains a
    hard error.
    """

    adapter_keys = (
        "video_residual_adapter.0.weight",
        "video_residual_adapter.2.weight",
    )
    migrated = "video_residual_gate" not in state_dict or any(
        key not in state_dict for key in adapter_keys
    )
    compatible = dict(state_dict)
    if "video_residual_gate" not in compatible:
        compatible["video_residual_gate"] = torch.zeros_like(
            block.video_residual_gate,
            device="cpu",
        )
    current = block.state_dict()
    for key in adapter_keys:
        if key not in compatible:
            compatible[key] = current[key].detach().cpu().clone()
    block.load_state_dict(compatible, strict=True)
    return migrated


class H3DreamActionExpert(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int = 7,
        state_dim: int = 8,
        text_dim: int = 5120,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 56,
        head_dim: int = 128,
        num_layers: int = 50,
        frequency_dim: int = 256,
        eps: float = 1.0e-5,
        full_width_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        if min(action_dim, text_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("ActionDiT dimensions and layer count must be positive")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.frequency_dim = int(frequency_dim)
        self.full_width_rmsnorm = bool(full_width_rmsnorm)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, text_dim)
        self.context_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                H3DreamActionBlock(
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

    def prepare(
        self,
        *,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state: torch.Tensor,
        append_state: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch, horizon, action_dim = noisy_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}")
        if timestep.shape != (batch,):
            raise ValueError("action timestep must be [B]")
        if state.shape != (batch, self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("context must be [B,L,text_dim]")
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context mask must be [B,L]")
        if append_state:
            context, context_mask = self.append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=state,
            )
        time_features = sinusoidal_embedding(self.frequency_dim, timestep).to(
            self.time_embedding[0].weight.dtype
        )
        time_hidden = self.time_embedding(time_features)
        return {
            "tokens": self.action_embedding(noisy_actions),
            "time_modulation": self.time_projection(time_hidden).reshape(
                batch, 6, self.hidden_dim
            ),
            "context": self.context_embedding(context),
            "context_mask": context_mask,
        }

    def append_state_to_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append proprioception once for both H3 and ActionDiT to consume."""

        batch = context.shape[0]
        if state.shape != (batch, self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context mask must be [B,L]")
        state_token = self.state_embedding(
            state.to(self.state_embedding.weight.dtype)
        ).to(context.dtype).unsqueeze(1)
        state_mask = torch.ones(batch, 1, device=context.device, dtype=torch.bool)
        return (
            torch.cat((context, state_token), dim=1),
            torch.cat((context_mask.bool(), state_mask), dim=1),
        )

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(tokens)
