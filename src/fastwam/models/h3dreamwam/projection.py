"""Identity-preserving RGB/flow channel expansion for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class H3RGBFlowProjectionReport:
    rgb_channels: int
    flow_channels: int
    patch_volume: int
    old_patch_width: int
    new_patch_width: int
    flow_input_init_scale: float
    flow_output_init_scale: float


def _replace_linear(
    source: nn.Linear,
    *,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    target = nn.Linear(
        in_features,
        out_features,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    target.train(source.training)
    target.weight.requires_grad_(source.weight.requires_grad)
    if target.bias is not None and source.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)
    return target


def _update_config_in_channels(model: nn.Module, in_channels: int) -> None:
    register = getattr(model, "register_to_config", None)
    if callable(register):
        register(in_channels=int(in_channels))
        return
    config = getattr(model, "config", None)
    if config is None:
        raise AttributeError("H3 model must expose config or register_to_config")
    if isinstance(config, dict):
        config["in_channels"] = int(in_channels)
    else:
        setattr(config, "in_channels", int(in_channels))


def expand_h3_rgb_flow_projections(
    model: nn.Module,
    *,
    flow_channels: int | None = None,
    flow_input_init_scale: float = 0.1,
    flow_output_init_scale: float = 0.1,
    generator: torch.Generator | None = None,
) -> H3RGBFlowProjectionReport:
    """Expand H3's patch projections while preserving the pretrained RGB path.

    The first output group remains bitwise copied from H3. New flow input
    columns and flow output rows use the small Gaussian initialization from
    DreamWAM. A zero-flow input still preserves the pretrained RGB path because
    the original input columns and RGB output rows are copied exactly.
    """

    if flow_input_init_scale < 0:
        raise ValueError("flow_input_init_scale must be non-negative")
    if flow_output_init_scale < 0:
        raise ValueError("flow_output_init_scale must be non-negative")
    config: Any = getattr(model, "config", None)
    if config is None:
        raise AttributeError("H3 model must expose config")
    rgb_channels = int(
        config["in_channels"] if isinstance(config, dict) else config.in_channels
    )
    flow_channels = rgb_channels if flow_channels is None else int(flow_channels)
    if rgb_channels <= 0 or flow_channels <= 0:
        raise ValueError("RGB and flow channels must be positive")
    patch_size = tuple(
        int(value)
        for value in (
            config["patch_size"] if isinstance(config, dict) else config.patch_size
        )
    )
    patch_volume = math.prod(patch_size)
    old_width = rgb_channels * patch_volume
    flow_width = flow_channels * patch_volume
    new_width = old_width + flow_width

    old_in = getattr(model, "proj_in", None)
    old_out = getattr(model, "proj_out", None)
    if not isinstance(old_in, nn.Linear) or not isinstance(old_out, nn.Linear):
        raise TypeError("H3 proj_in and proj_out must both be nn.Linear")
    if old_in.in_features != old_width:
        raise ValueError(
            f"proj_in width {old_in.in_features} does not match "
            f"RGB channels x patch volume {old_width}"
        )
    if old_out.out_features != old_width:
        raise ValueError(
            f"proj_out width {old_out.out_features} does not match "
            f"RGB channels x patch volume {old_width}"
        )
    if old_in.out_features != old_out.in_features:
        raise ValueError("H3 input/output projection hidden widths differ")

    new_in = _replace_linear(
        old_in,
        in_features=new_width,
        out_features=old_in.out_features,
    )
    new_out = _replace_linear(
        old_out,
        in_features=old_out.in_features,
        out_features=new_width,
    )
    with torch.no_grad():
        new_in.weight[:, :old_width].copy_(old_in.weight)
        source_std = old_in.weight.float().std().clamp_min(1.0e-8)
        if flow_input_init_scale == 0:
            new_in.weight[:, old_width:].zero_()
        else:
            initialized = torch.randn(
                new_in.weight[:, old_width:].shape,
                device=new_in.weight.device,
                dtype=torch.float32,
                generator=generator,
            )
            initialized.mul_(source_std * float(flow_input_init_scale))
            new_in.weight[:, old_width:].copy_(initialized.to(new_in.weight.dtype))
        if old_in.bias is not None:
            new_in.bias.copy_(old_in.bias)

        new_out.weight[:old_width].copy_(old_out.weight)
        if flow_output_init_scale == 0:
            new_out.weight[old_width:].zero_()
        else:
            output_std = old_out.weight.float().std().clamp_min(1.0e-8)
            initialized = torch.randn(
                new_out.weight[old_width:].shape,
                device=new_out.weight.device,
                dtype=torch.float32,
                generator=generator,
            )
            initialized.mul_(output_std * float(flow_output_init_scale))
            new_out.weight[old_width:].copy_(initialized.to(new_out.weight.dtype))
        if old_out.bias is not None:
            new_out.bias[:old_width].copy_(old_out.bias)
            new_out.bias[old_width:].zero_()

    model.proj_in = new_in
    model.proj_out = new_out
    _update_config_in_channels(model, rgb_channels + flow_channels)
    return H3RGBFlowProjectionReport(
        rgb_channels=rgb_channels,
        flow_channels=flow_channels,
        patch_volume=patch_volume,
        old_patch_width=old_width,
        new_patch_width=new_width,
        flow_input_init_scale=float(flow_input_init_scale),
        flow_output_init_scale=float(flow_output_init_scale),
    )
