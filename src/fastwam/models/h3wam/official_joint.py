"""Joint-training utilities for the official Diffusers MiniMax-H3 backbone."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def build_h3_observation_attention_mask(
    *,
    sequence_length: int,
    text_indices: torch.Tensor,
    condition_video_indices: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Keep observation features causal with respect to target video/audio rows.

    H3 normally applies full packed-sequence self-attention.  During WAM
    co-training that would let the current-frame features used by the action
    head read noisy future rows.  Text and conditioning-video queries are
    therefore restricted to text and conditioning-video keys.  Target queries
    remain fully connected so the original video-flow objective can still use
    all context and target rows.

    The returned boolean mask follows PyTorch SDPA semantics: ``True`` means a
    query/key pair is allowed.  Leading singleton batch/head dimensions make
    it broadcastable to every H3 attention head.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    text_indices = text_indices.to(device=device, dtype=torch.long).reshape(-1)
    condition_video_indices = condition_video_indices.to(
        device=device, dtype=torch.long
    ).reshape(-1)
    observable = torch.unique(torch.cat((text_indices, condition_video_indices)))
    if observable.numel() == 0:
        raise ValueError("at least one observable H3 row is required")
    if int(observable.min()) < 0 or int(observable.max()) >= sequence_length:
        raise ValueError("observable H3 row index is outside the packed sequence")

    allowed = torch.ones(
        (sequence_length, sequence_length), device=device, dtype=torch.bool
    )
    target_columns = torch.ones(sequence_length, device=device, dtype=torch.bool)
    target_columns[observable] = False
    target_indices = target_columns.nonzero(as_tuple=False).reshape(-1)
    allowed[observable[:, None], target_indices[None, :]] = False
    return allowed.unsqueeze(0).unsqueeze(0)


class H3OfficialFeatureCapture:
    """Capture current-frame rows after selected official H3 blocks."""

    def __init__(
        self,
        blocks: Iterable[nn.Module],
        layer_indices: Iterable[int],
        condition_video_indices: torch.Tensor,
    ) -> None:
        self.blocks = tuple(blocks)
        self.layer_indices = tuple(sorted({int(index) for index in layer_indices}))
        if not self.layer_indices:
            raise ValueError("at least one H3 layer must be captured")
        if self.layer_indices[0] < 0 or self.layer_indices[-1] >= len(self.blocks):
            raise ValueError("captured H3 layer is outside the backbone")
        self.condition_video_indices = condition_video_indices.reshape(-1).long()
        if self.condition_video_indices.numel() == 0:
            raise ValueError("condition_video_indices cannot be empty")
        self.features: dict[int, torch.Tensor] = {}
        self._handles = [
            self.blocks[index].register_forward_hook(self._hook(index))
            for index in self.layer_indices
        ]

    def _hook(self, layer_index: int):
        def capture(_module, _args, output: torch.Tensor) -> None:
            if output.ndim != 3:
                raise ValueError(
                    f"official H3 block output must be [B,S,D], got {tuple(output.shape)}"
                )
            indices = self.condition_video_indices.to(output.device)
            self.features[layer_index] = output.index_select(1, indices)

        return capture

    def clear(self) -> None:
        self.features.clear()

    def set_condition_video_indices(self, indices: torch.Tensor) -> None:
        indices = indices.reshape(-1).long()
        if indices.numel() == 0:
            raise ValueError("condition_video_indices cannot be empty")
        self.condition_video_indices = indices
        self.clear()

    def stacked(self) -> torch.Tensor:
        missing = [index for index in self.layer_indices if index not in self.features]
        if missing:
            raise RuntimeError(f"official H3 feature capture missed layers {missing}")
        return torch.stack(
            [self.features[index] for index in self.layer_indices], dim=1
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class H3BlockAttentionMask:
    """Install one broadcast attention mask on every official H3 block."""

    def __init__(self, blocks: Iterable[nn.Module]) -> None:
        self.mask: torch.Tensor | None = None
        self._handles = [
            block.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            for block in blocks
        ]

    def _pre_hook(self, _module, args: tuple, kwargs: dict):
        if self.mask is None:
            return args, kwargs
        if kwargs.get("attention_mask") is not None:
            raise RuntimeError("H3 block already received an attention mask")
        kwargs["attention_mask"] = self.mask
        return args, kwargs

    def set(self, mask: torch.Tensor | None) -> None:
        self.mask = mask

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
