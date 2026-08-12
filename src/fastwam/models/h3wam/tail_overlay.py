"""Deploy BF16 H3 tail updates as dense deltas over a quantized Comfy base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


class H3DenseDeltaLinear(nn.Module):
    """Keep a quantized base linear intact and add one frozen dense BF16 delta."""

    def __init__(
        self,
        base: nn.Module,
        delta_weight: torch.Tensor,
        delta_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.in_features = int(getattr(base, "in_features"))
        self.out_features = int(getattr(base, "out_features"))
        expected = (self.out_features, self.in_features)
        if tuple(delta_weight.shape) != expected:
            raise ValueError(
                f"dense delta shape {tuple(delta_weight.shape)} does not match {expected}"
            )
        if delta_bias is not None and tuple(delta_bias.shape) != (self.out_features,):
            raise ValueError("dense delta bias has the wrong shape")
        self.delta_weight = nn.Parameter(
            delta_weight.detach().contiguous(), requires_grad=False
        )
        self.delta_bias = (
            None
            if delta_bias is None
            else nn.Parameter(delta_bias.detach().contiguous(), requires_grad=False)
        )

    @property
    def weight(self):
        """Expose the base weight for Comfy fused-dispatch compatibility checks."""

        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden)
        delta = torch.nn.functional.linear(
            hidden.to(dtype=self.delta_weight.dtype),
            self.delta_weight,
            self.delta_bias,
        )
        return base_output + delta.to(dtype=base_output.dtype)


@dataclass(frozen=True)
class H3TailDeltaReport:
    source_step: int
    blocks: tuple[int, ...]
    linear_modules: int
    norm_modules: int
    parameters: int


def _resolve_owner(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    owner = model
    for part in parts[:-1]:
        if part.isdigit():
            owner = owner[int(part)]
        else:
            owner = getattr(owner, part)
    return owner, parts[-1]


def load_h3_comfy_feature_delta(
    model: nn.Module, checkpoint: str | Path
) -> H3TailDeltaReport:
    """Apply a feature-only tail delta produced by the H3 FSDP exporter.

    Attention/MLP/AdaLN linears stay as the original quantized module plus a
    dense BF16 residual branch. Norm deltas are added directly to their small
    BF16 weights. The overlay is calibrated only for the feature extraction
    timestep contract recorded in the safetensors metadata; it is not a video
    generation checkpoint.
    """

    checkpoint = Path(checkpoint).resolve()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        checkpoint_format = metadata.get("format")
        if checkpoint_format not in {
            "h3wam_comfy_feature_delta_v1",
            "h3wam_comfy_feature_delta_int8_v1",
        }:
            raise ValueError("checkpoint is not an H3-WAM Comfy feature delta")
        keys = set(handle.keys())
        quantized = checkpoint_format == "h3wam_comfy_feature_delta_int8_v1"

        def load_delta(key: str) -> torch.Tensor:
            value = handle.get_tensor(key)
            if quantized:
                value = value.float() * handle.get_tensor(key + ".scale").float()
            return value

        blocks = tuple(int(value) for value in json.loads(metadata["blocks"]))
        linear_paths = []
        norm_paths = []
        for key in sorted(keys):
            if not key.endswith(".weight"):
                continue
            path = key[: -len(".weight")]
            if path.endswith(("norm1", "norm2", "q_norm", "k_norm")):
                norm_paths.append(path)
            else:
                linear_paths.append(path)

        parameter_count = 0
        for path in linear_paths:
            owner, name = _resolve_owner(model, path)
            base = getattr(owner, name)
            if isinstance(base, H3DenseDeltaLinear):
                raise ValueError(f"H3 dense delta already installed at {path}")
            weight = load_delta(path + ".weight").to(
                device=base.weight.device, dtype=torch.bfloat16
            )
            bias_key = path + ".bias"
            bias = (
                load_delta(bias_key).to(
                    device=base.weight.device, dtype=torch.bfloat16
                )
                if bias_key in keys
                else None
            )
            wrapped = H3DenseDeltaLinear(base, weight, bias)
            setattr(owner, name, wrapped)
            parameter_count += weight.numel() + (0 if bias is None else bias.numel())

        with torch.no_grad():
            for path in norm_paths:
                owner, name = _resolve_owner(model, path)
                norm = getattr(owner, name)
                delta = load_delta(path + ".weight").to(
                    device=norm.weight.device, dtype=norm.weight.dtype
                )
                if tuple(delta.shape) != tuple(norm.weight.shape):
                    raise ValueError(f"norm delta shape mismatch at {path}")
                norm.weight.add_(delta)
                parameter_count += delta.numel()

    return H3TailDeltaReport(
        source_step=int(metadata["source_step"]),
        blocks=blocks,
        linear_modules=len(linear_paths),
        norm_modules=len(norm_paths),
        parameters=parameter_count,
    )
