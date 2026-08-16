"""No-disk-K/V online training boundary for C58b and its child branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .fact_online_data import OnlineH3FACTDemoDataset
from .fastwam_full_tower import LAYERWISE_H3_50_TO_ACTION_30
from .int8_backbone import H3Int8FeatureBackbone
from .int8_online import (
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    SEQUENCE_KV_POOL,
)


class C58OnlineFrozenH3Dataset(OnlineH3FACTDemoDataset):
    """Dense action windows ending exactly before frozen H3 execution."""

    def __init__(
        self,
        manifest: Path | str,
        source_manifest: Path | str,
        cache_root: Path | str,
        h3_checkpoint: Path | str,
        *,
        action_horizon: int = 32,
        sample_offset: int = 0,
        limit: int = 0,
    ) -> None:
        super().__init__(
            manifest,
            source_manifest,
            cache_root,
            split="train",
            action_horizon=action_horizon,
            sample_offset=sample_offset,
            limit=limit,
        )
        self.first_checkpoint_path = Path(h3_checkpoint).resolve()
        if not self.first_checkpoint_path.is_file():
            raise FileNotFoundError(self.first_checkpoint_path)
        self.manifest_items = len(
            [
                line
                for line in self.manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        self.source_manifest_items = len(
            [
                line
                for line in self.source_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        # Preserve D0/C58's exact deployment-bounded normalization contract.
        item["actions"] = item["actions"].clamp(-5.0, 5.0)
        item["proprio"] = item["proprio"].clamp(-5.0, 5.0)
        return item


def collate_c58_online(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("the pruned H3 online runtime requires per-rank batch size 1")
    item = items[0]
    return {
        "sample_ids": [str(item["sample_id"])],
        "actions": item["actions"].unsqueeze(0),
        "proprio": item["proprio"].unsqueeze(0),
        "action_is_pad": item["action_is_pad"].unsqueeze(0),
        "text_context": item["text_context"].unsqueeze(0),
        "text_mask": torch.ones(
            (1, item["text_context"].shape[0]), dtype=torch.bool
        ),
        "text_token_tags": item["text_token_tags"].unsqueeze(0),
        "current_h3_input": item["current_h3_input"].unsqueeze(0),
    }


def move_c58_online_batch(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    result = dict(batch)
    for key in ("actions", "proprio", "text_context"):
        result[key] = batch[key].to(device=device, dtype=dtype, non_blocking=True)
    result["current_h3_input"] = batch["current_h3_input"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    result["text_token_tags"] = batch["text_token_tags"].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    result["action_is_pad"] = batch["action_is_pad"].to(
        device=device, non_blocking=True
    )
    result["text_mask"] = batch["text_mask"].to(device=device, non_blocking=True)
    return result


def materialize_frozen_kv(
    inference_kv: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Make exact ordinary tensors checkpointing may save for ActionDiT."""

    result = {
        layer: {name: tensor.clone() for name, tensor in item.items()}
        for layer, item in inference_kv.items()
    }
    if any(
        torch.is_inference(tensor)
        for item in result.values()
        for tensor in item.values()
    ):
        raise RuntimeError("online H3 K/V remained an inference tensor")
    return result


class C58OnlineFrozenH3Provider(nn.Module):
    """Each DDP rank owns one frozen INT8 H3 and emits ordinary BF16 K/V."""

    def __init__(
        self,
        h3_checkpoint: Path | str,
        *,
        layers: tuple[int, ...] = LAYERWISE_H3_50_TO_ACTION_30,
    ) -> None:
        super().__init__()
        self.h3_checkpoint = Path(h3_checkpoint).resolve()
        backbone = H3Int8FeatureBackbone.from_checkpoint(self.h3_checkpoint)
        backbone.requires_grad_(False)
        self.provider = H3Int8OnlineKVProvider(
            backbone,
            H3Int8OnlineKVContract(
                layers=layers,
                action_horizon=32,
                target_latent_frames=12,
                video_timestep=1.0,
                condition_video_timestep=1.0,
                capture_token_count=32,
                pool_strategy=SEQUENCE_KV_POOL,
            ),
        )
        self.requires_grad_(False)

    def forward(self, batch: dict[str, Any]) -> dict[int, dict[str, torch.Tensor]]:
        if batch["current_h3_input"].shape[0] != 1:
            raise ValueError("online H3 provider requires per-rank batch size 1")
        inference_kv = self.provider(
            batch["current_h3_input"],
            batch["text_context"].float(),
            batch["text_token_tags"][0],
        )
        return materialize_frozen_kv(inference_kv)


def attach_online_h3_kv(
    batch: dict[str, Any], provider: C58OnlineFrozenH3Provider
) -> dict[str, Any]:
    result = dict(batch)
    result["video_kv_cache"] = provider(batch)
    return result


__all__ = [
    "C58OnlineFrozenH3Dataset",
    "C58OnlineFrozenH3Provider",
    "attach_online_h3_kv",
    "collate_c58_online",
    "materialize_frozen_kv",
    "move_c58_online_batch",
]
