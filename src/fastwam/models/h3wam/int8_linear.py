"""Standalone ConvRot INT8 linear layers for the cloud H3 runtime.

This module deliberately depends only on PyTorch and the independent
``comfy-kitchen`` kernel package.  It does not import ComfyUI, its model
management, server, nodes, or workflow runtime.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn


INT8_TENSORWISE_FORMAT = "int8_tensorwise"


def parse_int8_marker(marker: torch.Tensor) -> dict[str, Any]:
    if marker.dtype != torch.uint8 or marker.ndim != 1:
        raise ValueError("comfy_quant marker must be a one-dimensional uint8 tensor")
    try:
        payload = json.loads(bytes(marker.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid comfy_quant JSON marker") from error
    if payload.get("format") != INT8_TENSORWISE_FORMAT:
        raise ValueError(
            f"unsupported quantized format {payload.get('format')!r}; "
            f"expected {INT8_TENSORWISE_FORMAT!r}"
        )
    group_size = int(payload.get("convrot_groupsize", 256))
    if group_size <= 0:
        raise ValueError("convrot_groupsize must be positive")
    return {
        "format": INT8_TENSORWISE_FORMAT,
        "convrot": bool(payload.get("convrot", False)),
        "convrot_groupsize": group_size,
    }


class ConvRotInt8Linear(nn.Module):
    """Frozen W8A8 Linear backed by comfy-kitchen's public INT8 kernel."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        convrot: bool = True,
        convrot_groupsize: int = 256,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("INT8 linear weight must be a two-dimensional int8 tensor")
        if weight_scale.dtype != torch.float32:
            raise ValueError("INT8 weight scale must use float32")
        if tuple(weight_scale.shape) not in {(weight.shape[0], 1), (1,)}:
            raise ValueError(
                "weight_scale must be scalar or one value per output channel"
            )
        if bias is not None and tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("bias shape does not match INT8 output dimension")
        if convrot_groupsize <= 0:
            raise ValueError("convrot_groupsize must be positive")
        if convrot and weight.shape[1] % convrot_groupsize:
            raise ValueError("ConvRot input dimension must divide the group size")

        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.convrot = bool(convrot)
        self.convrot_groupsize = int(convrot_groupsize)
        self.out_dtype = out_dtype
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_buffer(
            "bias",
            None if bias is None else bias.contiguous(),
        )

    @classmethod
    def from_checkpoint_tensors(
        cls,
        *,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        marker: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> "ConvRotInt8Linear":
        metadata = parse_int8_marker(marker)
        return cls(
            weight,
            weight_scale,
            bias,
            convrot=metadata["convrot"],
            convrot_groupsize=metadata["convrot_groupsize"],
            out_dtype=out_dtype,
        )

    def forward(
        self, input_tensor: torch.Tensor, *, input_act: str | None = None
    ) -> torch.Tensor:
        expected_input_width = (
            self.in_features * 2 if input_act == "swiglu" else self.in_features
        )
        if input_tensor.shape[-1] != expected_input_width:
            raise ValueError(
                f"input width {input_tensor.shape[-1]} does not match "
                f"INT8 layer width {expected_input_width} for input_act={input_act!r}"
            )
        try:
            import comfy_kitchen
        except ImportError as error:
            raise RuntimeError(
                "ConvRotInt8Linear requires the standalone comfy-kitchen package"
            ) from error
        return comfy_kitchen.int8_linear(
            input_tensor,
            self.weight,
            self.weight_scale,
            bias=self.bias,
            out_dtype=self.out_dtype,
            convrot=self.convrot,
            convrot_groupsize=self.convrot_groupsize,
            input_act=input_act,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, convrot={self.convrot}, "
            f"convrot_groupsize={self.convrot_groupsize}, out_dtype={self.out_dtype}"
        )
