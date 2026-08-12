"""Runtime compatibility fixes for backpropagating through ComfyUI's H3."""

from __future__ import annotations

import importlib
from types import ModuleType

import torch
from torch.utils.checkpoint import checkpoint


def enable_comfy_h3_autograd(*, checkpoint_blocks: bool = True) -> ModuleType:
    """Replace H3 inference-only residual mutations with autograd-safe ops.

    The current ComfyUI H3 implementation updates the transformer residual in
    place to reduce inference memory.  A frozen backbone still needs input
    gradients when training an action adapter, and those mutations invalidate
    PyTorch's saved tensors.  This process-local patch leaves checkpoints and
    the ComfyUI installation untouched.
    """

    module = importlib.import_module("comfy.ldm.minimax.model")
    if getattr(module, "_fastwam_autograd_enabled", False):
        return module

    def mod_scale_shift(
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        segments: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        return torch.cat(
            [
                hidden[start:stop] * (1.0 + scale[row].to(hidden.dtype))
                + shift[row].to(hidden.dtype)
                for start, stop, row in segments
            ],
            dim=0,
        )

    def mod_gate(
        residual: torch.Tensor,
        gate: torch.Tensor,
        update: torch.Tensor,
        segments: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        return torch.cat(
            [
                residual[start:stop] + update[start:stop] * gate[row].to(residual.dtype)
                for start, stop, row in segments
            ],
            dim=0,
        )

    def refiner_forward(self, hidden, transformer_options={}):
        attention = self.attn(self.norm1(hidden), transformer_options=transformer_options)
        hidden = hidden + attention
        return hidden + self.mlp(self.norm2(hidden))

    def apply_split_half_rope(hidden: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        rotation_dim = rotation.shape[-3] * 2
        rotated, passthrough = hidden[..., :rotation_dim], hidden[..., rotation_dim:]
        pairs = rotated.reshape(*rotated.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        rotated = rotation[..., 0] * pairs[..., 0] + rotation[..., 1] * pairs[..., 1]
        rotated = rotated.movedim(-1, -2).reshape(*hidden.shape[:-1], rotation_dim)
        return torch.cat((rotated.to(hidden.dtype), passthrough), dim=-1)

    def attention_forward(self, hidden, rope_freqs=None, transformer_options={}):
        sequence = hidden.shape[0]
        inner = self.heads * self.head_dim
        query, key, value = self.qkv_proj(hidden).split(inner, dim=-1)
        query = query.view(1, sequence, self.heads, self.head_dim)
        key = key.view(1, sequence, self.heads, self.head_dim)
        value = value.view(sequence, self.heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if rope_freqs is not None:
            query = apply_split_half_rope(query, rope_freqs)
            key = apply_split_half_rope(key, rope_freqs)
        query = query[0].transpose(0, 1).unsqueeze(0)
        key = key[0].transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        output = module.optimized_attention(
            query,
            key,
            value,
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(output.squeeze(0))

    module._mod_scale_shift = mod_scale_shift
    module._mod_gate = mod_gate
    module.RefinerBlock.forward = refiner_forward
    module.Attention.forward = attention_forward
    if checkpoint_blocks:
        block_forward = module.DiTBlock.forward

        def checkpointed_block_forward(
            self,
            hidden,
            time_embedding,
            segments,
            rope_freqs,
            transformer_options={},
        ):
            if not torch.is_grad_enabled():
                return block_forward(
                    self,
                    hidden,
                    time_embedding,
                    segments,
                    rope_freqs,
                    transformer_options=transformer_options,
                )

            def run_block(block_input, block_time_embedding):
                return block_forward(
                    self,
                    block_input,
                    block_time_embedding,
                    segments,
                    rope_freqs,
                    transformer_options=transformer_options,
                )

            return checkpoint(
                run_block,
                hidden,
                time_embedding,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        module.DiTBlock.forward = checkpointed_block_forward
    module._fastwam_autograd_enabled = True
    return module
