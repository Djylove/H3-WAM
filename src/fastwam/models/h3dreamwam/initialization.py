"""Initialize H3-DreamWAM ActionDiT from the pretrained H3 backbone."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .action_expert import H3DreamActionExpert
from .docking import H3DoTActionHead


@dataclass(frozen=True)
class H3ActionInitializationReport:
    layers: int
    copied_tensors: int
    resized_tensors: int
    zeroed_biases: int
    residual_gate_init: float
    residual_output_scale: float


@dataclass(frozen=True)
class H3DoTInitializationReport:
    action_layers: int
    h3_layers: int
    source_layer_indices: tuple[int, ...]
    copied_tensors: int
    resized_tensors: int
    zeroed_biases: int
    residual_output_scale: float
    alpha_scaling: bool


def resize_tensor(source: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Sequentially linearly interpolate every mismatched tensor dimension."""

    if tuple(source.shape) == target_shape:
        return source
    output = source.detach().float()
    if output.ndim != len(target_shape):
        raise ValueError(
            f"cannot resize rank-{output.ndim} tensor to rank-{len(target_shape)}"
        )
    for dimension, size in enumerate(target_shape):
        if output.shape[dimension] == size:
            continue
        permutation = [index for index in range(output.ndim) if index != dimension]
        permutation.append(dimension)
        inverse = [0] * output.ndim
        for index, original in enumerate(permutation):
            inverse[original] = index
        output = output.permute(*permutation).contiguous()
        prefix = output.shape[:-1]
        output = F.interpolate(
            output.reshape(-1, 1, output.shape[-1]),
            size=size,
            mode="linear",
            align_corners=True,
        ).reshape(*prefix, size)
        output = output.permute(*inverse).contiguous()
    if tuple(output.shape) != target_shape:
        raise RuntimeError(f"resize produced {tuple(output.shape)}, expected {target_shape}")
    return output


def _copy_resized(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    alpha_scaling: bool = False,
) -> bool:
    resized = tuple(target.shape) != tuple(source.shape)
    value = resize_tensor(source, tuple(target.shape))
    if (
        alpha_scaling
        and source.ndim >= 2
        and source.shape[-1] != target.shape[-1]
    ):
        value = value.float() * math.sqrt(source.shape[-1] / target.shape[-1])
    target.copy_(value.to(device=target.device, dtype=target.dtype))
    return resized


def _zero_bias(module: nn.Module) -> int:
    bias = getattr(module, "bias", None)
    if bias is None:
        return 0
    bias.zero_()
    return 1


def initialize_action_expert_from_h3(
    action_expert: H3DreamActionExpert,
    h3: nn.Module,
    *,
    residual_gate_init: float = 1.0,
    residual_output_scale: float = 0.01,
    alpha_scaling: bool = False,
) -> H3ActionInitializationReport:
    """Interpolate H3 attention/FFN tensors into the independent ActionDiT."""

    if not 0.0 <= residual_gate_init <= 1.0:
        raise ValueError("residual_gate_init must be in [0,1]")
    if not 0.0 < residual_output_scale <= 1.0:
        raise ValueError("residual_output_scale must be in (0,1]")
    h3_blocks = list(h3.transformer_blocks)
    action_blocks = list(action_expert.blocks)
    if len(h3_blocks) != len(action_blocks):
        raise ValueError("H3 and ActionDiT layer counts must match for initialization")
    copied = 0
    resized = 0
    zeroed = 0
    with torch.no_grad():
        # H3 has one text projection. Use it for the first ActionDiT text layer
        # and make the second layer an identity-preserving refinement.
        resized += int(
            _copy_resized(
                action_expert.context_embedding[0].weight,
                h3.context_embedder.weight,
                alpha_scaling=alpha_scaling,
            )
        )
        copied += 1
        zeroed += _zero_bias(action_expert.context_embedding[0])
        nn.init.eye_(action_expert.context_embedding[2].weight)
        zeroed += _zero_bias(action_expert.context_embedding[2])

        resized += int(
            _copy_resized(
                action_expert.time_embedding[0].weight,
                h3.time_embedder.linear_1.weight,
                alpha_scaling=alpha_scaling,
            )
        )
        copied += 1
        zeroed += _zero_bias(action_expert.time_embedding[0])
        resized += int(
            _copy_resized(
                action_expert.time_embedding[2].weight,
                h3.time_embedder.linear_2.weight,
                alpha_scaling=alpha_scaling,
            )
        )
        copied += 1
        zeroed += _zero_bias(action_expert.time_embedding[2])
        # H3 has a separate AdaLN projection in every block, whereas the
        # DreamWAM ActionDiT has one global time projection plus a per-block
        # constant modulation. Directly resizing H3 AdaLN makes 50 residual
        # layers explode, so the new global time route starts as a zero delta.
        action_expert.time_projection[-1].weight.zero_()
        action_expert.time_projection[-1].bias.zero_()
        zeroed += 1

        for h3_block, action_block in zip(h3_blocks, action_blocks, strict=True):
            for attribute, source in (
                ("to_q", h3_block.attn.to_q.weight),
                ("to_k", h3_block.attn.to_k.weight),
                ("to_v", h3_block.attn.to_v.weight),
                ("to_out", h3_block.attn.to_out[0].weight),
            ):
                targets = (
                    getattr(action_block.attn, attribute).weight,
                    getattr(action_block.cross_attn, attribute).weight,
                )
                value = resize_tensor(source, tuple(targets[0].shape))
                if (
                    alpha_scaling
                    and source.ndim >= 2
                    and source.shape[-1] != targets[0].shape[-1]
                ):
                    value = value.float() * math.sqrt(
                        source.shape[-1] / targets[0].shape[-1]
                    )
                for target in targets:
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                copied += 2
                resized += 2 * int(tuple(targets[0].shape) != tuple(source.shape))
            for attribute, source in (
                ("norm_q", h3_block.attn.norm_q.weight),
                ("norm_k", h3_block.attn.norm_k.weight),
            ):
                targets = (
                    getattr(action_block.attn, attribute).weight,
                    getattr(action_block.cross_attn, attribute).weight,
                )
                value = resize_tensor(source, tuple(targets[0].shape))
                for target in targets:
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                copied += 2
                resized += 2 * int(tuple(targets[0].shape) != tuple(source.shape))
            for action_attention in (action_block.attn, action_block.cross_attn):
                zeroed += _zero_bias(action_attention.to_q)
                zeroed += _zero_bias(action_attention.to_k)
                zeroed += _zero_bias(action_attention.to_v)
                zeroed += _zero_bias(action_attention.to_out)
            # Damp the copied residual branch itself, not only its AdaLN gate.
            # With a tiny gate the forward pass is quiet but dL/d(gate) still
            # sees the full H3 activation, which produced a 1.99e6 first-step
            # gradient norm on the real 50-layer model. Scaling the output
            # projection keeps both the forward residual and gate derivative
            # bounded while retaining the interpolated attention basis.
            action_block.attn.to_out.weight.mul_(residual_output_scale)
            # H3 has no cross-attention module. Keep the resized Q/K/V as a
            # useful starting basis, but introduce this new route at zero.
            action_block.cross_attn.to_out.weight.zero_()
            action_block.cross_attn.to_out.bias.zero_()

            swiglu = h3_block.ff.net[0].proj.weight
            gate, value = swiglu.chunk(2, dim=0)
            merged_input = 0.5 * (gate.float() + value.float())
            resized += int(
                _copy_resized(
                    action_block.ffn[0].weight,
                    merged_input,
                    alpha_scaling=alpha_scaling,
                )
            )
            copied += 1
            zeroed += _zero_bias(action_block.ffn[0])
            resized += int(
                _copy_resized(
                    action_block.ffn[2].weight,
                    h3_block.ff.net[2].weight,
                    alpha_scaling=alpha_scaling,
                )
            )
            copied += 1
            zeroed += _zero_bias(action_block.ffn[2])
            action_block.ffn[2].weight.mul_(residual_output_scale)

            action_block.norm3.weight.fill_(1.0)
            action_block.norm3.bias.zero_()
            action_block.video_residual_gate.zero_()
            action_block.video_residual_adapter[-1].weight.zero_()
            action_block.modulation.zero_()
            action_block.modulation[:, 2].fill_(residual_gate_init)
            action_block.modulation[:, 5].fill_(residual_gate_init)

    return H3ActionInitializationReport(
        layers=len(h3_blocks),
        copied_tensors=copied,
        resized_tensors=resized,
        zeroed_biases=zeroed,
        residual_gate_init=float(residual_gate_init),
        residual_output_scale=float(residual_output_scale),
    )


def _uniform_source_layer_indices(
    source_layers: int,
    target_layers: int,
) -> tuple[int, ...]:
    """Select depth-covering source blocks for a shallower action carrier."""

    if source_layers <= 0 or target_layers <= 0:
        raise ValueError("source and target layer counts must be positive")
    if target_layers > source_layers:
        raise ValueError("DoT initialization cannot upsample the H3 block depth")
    if target_layers == 1:
        return (source_layers - 1,)
    denominator = target_layers - 1
    return tuple(
        (index * (source_layers - 1) + denominator // 2) // denominator
        for index in range(target_layers)
    )


def initialize_dot_action_head_from_h3(
    action_head: H3DoTActionHead,
    h3: nn.Module,
    *,
    residual_output_scale: float = 0.01,
    alpha_scaling: bool = True,
) -> H3DoTInitializationReport:
    """Initialize a shallow DoT carrier from depth-spanning H3 blocks.

    FastWAM initializes its ActionDiT backbone by linearly resizing the video
    DiT tensors while leaving the action encoder and prediction head random.
    H3 has 50 blocks whereas a diagnostic DoT carrier may be shallower, so we
    deterministically sample blocks across the full H3 depth and apply the same
    sequential interpolation/alpha-scaling rule. H3's SwiGLU input is averaged
    into the DoT GELU input, matching the existing H3 ActionDiT port.
    """

    if not 0.0 < residual_output_scale <= 1.0:
        raise ValueError("residual_output_scale must be in (0,1]")
    h3_blocks = list(h3.transformer_blocks)
    action_layers = list(action_head.layers)
    source_indices = _uniform_source_layer_indices(
        len(h3_blocks), len(action_layers)
    )
    copied = 0
    resized = 0
    zeroed = 0
    with torch.no_grad():
        resized += int(
            _copy_resized(
                action_head.time_embedding[0].weight,
                h3.time_embedder.linear_1.weight,
                alpha_scaling=alpha_scaling,
            )
        )
        copied += 1
        zeroed += _zero_bias(action_head.time_embedding[0])
        resized += int(
            _copy_resized(
                action_head.time_embedding[2].weight,
                h3.time_embedder.linear_2.weight,
                alpha_scaling=alpha_scaling,
            )
        )
        copied += 1
        zeroed += _zero_bias(action_head.time_embedding[2])
        # H3 has one AdaLN projection per source block but DoT shares a time
        # projection across its layers. Keep that unmatched route at zero and
        # let the copied residual bases enter through a stable constant gate,
        # as in the existing 50-layer H3 ActionDiT initialization.
        action_head.time_projection[-1].weight.zero_()
        action_head.time_projection[-1].bias.zero_()
        zeroed += 1
        for action_layer, source_index in zip(
            action_layers, source_indices, strict=True
        ):
            h3_block = h3_blocks[source_index]
            for attribute, source in (
                ("to_q", h3_block.attn.to_q.weight),
                ("to_k", h3_block.attn.to_k.weight),
                ("to_v", h3_block.attn.to_v.weight),
                ("to_out", h3_block.attn.to_out[0].weight),
            ):
                target_module = getattr(action_layer.attn, attribute)
                resized += int(
                    _copy_resized(
                        target_module.weight,
                        source,
                        alpha_scaling=alpha_scaling,
                    )
                )
                copied += 1
                zeroed += _zero_bias(target_module)
            for attribute, source in (
                ("norm_q", h3_block.attn.norm_q.weight),
                ("norm_k", h3_block.attn.norm_k.weight),
            ):
                target = getattr(action_layer.attn, attribute).weight
                resized += int(_copy_resized(target, source))
                copied += 1

            swiglu = h3_block.ff.net[0].proj.weight
            gate, value = swiglu.chunk(2, dim=0)
            merged_input = 0.5 * (gate.float() + value.float())
            resized += int(
                _copy_resized(
                    action_layer.ffn[0].weight,
                    merged_input,
                    alpha_scaling=alpha_scaling,
                )
            )
            copied += 1
            zeroed += _zero_bias(action_layer.ffn[0])
            resized += int(
                _copy_resized(
                    action_layer.ffn[2].weight,
                    h3_block.ff.net[2].weight,
                    alpha_scaling=alpha_scaling,
                )
            )
            copied += 1
            zeroed += _zero_bias(action_layer.ffn[2])

            # H3-to-1024 interpolation produces large residual activations.
            # Damping both output routes retained finite first-step gradients
            # in the existing 50-layer H3 ActionDiT port. Keep it explicit in
            # checkpoint metadata so an undamped variant remains separable.
            action_layer.attn.to_out.weight.mul_(residual_output_scale)
            action_layer.ffn[2].weight.mul_(residual_output_scale)
            action_layer.modulation.zero_()
            action_layer.modulation[:, 2].fill_(1.0)
            action_layer.modulation[:, 5].fill_(1.0)

    return H3DoTInitializationReport(
        action_layers=len(action_layers),
        h3_layers=len(h3_blocks),
        source_layer_indices=source_indices,
        copied_tensors=copied,
        resized_tensors=resized,
        zeroed_biases=zeroed,
        residual_output_scale=float(residual_output_scale),
        alpha_scaling=bool(alpha_scaling),
    )
