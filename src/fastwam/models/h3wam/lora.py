"""Small trainable LoRA branches around a frozen MiniMax H3 transformer."""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_OFFICIAL_H3_LORA_TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


class H3LoRALinear(nn.Module):
    """Wrap any linear-like module while keeping its original weights frozen."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        in_features = int(getattr(base, "in_features"))
        out_features = int(getattr(base, "out_features"))
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_a = nn.Linear(in_features, self.rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(self.rank, out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self):
        """Expose the frozen weight for fused-linear dispatch checks."""

        return self.base.weight

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def forward(
        self, hidden: torch.Tensor, *, input_act: str | None = None
    ) -> torch.Tensor:
        base_output = (
            self.base(hidden)
            if input_act is None
            else self.base(hidden, input_act=input_act)
        )
        if not self.enabled:
            return base_output
        lora_input = hidden.float()
        if input_act == "swiglu":
            gate, up = lora_input.chunk(2, dim=-1)
            lora_input = F.silu(gate) * up
        elif input_act is not None:
            raise ValueError(f"unsupported LoRA input activation {input_act!r}")
        delta = self.lora_b(self.lora_a(self.dropout(lora_input))) * self.scaling
        return base_output + delta.to(dtype=base_output.dtype)


@dataclass(frozen=True)
class H3LoRAReport:
    blocks: int
    modules: int
    parameters: int


def inject_h3_attention_lora(
    h3_model: nn.Module,
    *,
    rank: int = 4,
    alpha: float | None = None,
    dropout: float = 0.0,
    last_n_blocks: int | None = None,
    include_mlp: bool = False,
) -> H3LoRAReport:
    """Add LoRA branches to selected Comfy H3 projections."""

    blocks = list(getattr(h3_model, "blocks"))
    if last_n_blocks is not None:
        if last_n_blocks <= 0:
            raise ValueError("last_n_blocks must be positive")
        blocks = blocks[-last_n_blocks:]
    alpha = float(rank if alpha is None else alpha)
    module_count = 0
    for block in blocks:
        targets = [(block.attn, "qkv_proj"), (block.attn, "out_proj")]
        if include_mlp:
            targets.extend([(block.mlp, "fc1"), (block.mlp, "fc2")])
        for owner, name in targets:
            current = getattr(owner, name)
            if isinstance(current, H3LoRALinear):
                raise ValueError(f"{name} already has an H3 LoRA wrapper")
            wrapped = H3LoRALinear(
                current, rank=rank, alpha=alpha, dropout=dropout
            ).to(device=current.weight.device)
            setattr(owner, name, wrapped)
            module_count += 1
    parameters = sum(parameter.numel() for parameter in h3_lora_parameters(h3_model))
    return H3LoRAReport(len(blocks), module_count, parameters)


def inject_official_h3_lora(
    h3_model: nn.Module,
    *,
    last_n_blocks: int,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_OFFICIAL_H3_LORA_TARGETS,
) -> H3LoRAReport:
    """Add FP32 LoRA branches to Diffusers MiniMax-H3 tail projections."""

    blocks = list(getattr(h3_model, "transformer_blocks"))
    if not 1 <= last_n_blocks <= len(blocks):
        raise ValueError("last_n_blocks lies outside official H3")
    alpha = float(rank if alpha is None else alpha)
    target_names = tuple(dict.fromkeys(str(value) for value in targets))
    first = len(blocks) - last_n_blocks
    module_count = 0
    for block_index in range(first, len(blocks)):
        block = blocks[block_index]
        for target in target_names:
            parts = target.split(".")
            parent = block
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            leaf = parts[-1]
            current = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
            if isinstance(current, H3LoRALinear):
                raise ValueError(f"official H3 target {target!r} already has LoRA")
            if not isinstance(current, nn.Linear):
                raise TypeError(f"official H3 target {target!r} is not nn.Linear")
            wrapped = H3LoRALinear(
                current, rank=rank, alpha=alpha, dropout=dropout
            ).to(device=current.weight.device)
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            module_count += 1
    parameters = sum(parameter.numel() for parameter in h3_lora_parameters(h3_model))
    return H3LoRAReport(last_n_blocks, module_count, parameters)


def set_h3_lora_enabled(h3_model: nn.Module, enabled: bool) -> int:
    count = 0
    for module in h3_model.modules():
        if isinstance(module, H3LoRALinear):
            module.enabled = bool(enabled)
            count += 1
    return count


@contextmanager
def h3_lora_disabled(h3_model: nn.Module) -> Iterator[None]:
    modules = [module for module in h3_model.modules() if isinstance(module, H3LoRALinear)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled


def h3_lora_parameters(h3_model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in h3_model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    ]


def h3_lora_state_dict(h3_model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name.replace("._fsdp_wrapped_module", ""): value.detach().cpu().clone()
        for name, value in h3_model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_h3_lora_state_dict(
    h3_model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> int:
    canonical_state = {
        name.replace("._fsdp_wrapped_module", ""): value
        for name, value in state_dict.items()
    }
    if len(canonical_state) != len(state_dict):
        raise ValueError("LoRA key canonicalization produced duplicates")
    parameters = {
        name.replace("._fsdp_wrapped_module", ""): parameter
        for name, parameter in h3_model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }
    expected, received = set(parameters), set(canonical_state)
    missing, unexpected = sorted(expected - received), sorted(received - expected)
    if missing or unexpected:
        raise ValueError(
            "LoRA checkpoint keys do not match injected modules: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = canonical_state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"LoRA tensor shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return len(parameters)
