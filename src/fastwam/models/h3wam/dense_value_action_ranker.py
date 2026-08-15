"""Frozen C51 dense value expert for online best-of-N action selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .dense_future_value import DenseTemporalFutureValueModel


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenDenseValueActionRanker(nn.Module):
    """Return negative predicted cost so the existing selector can use argmax."""

    def __init__(
        self,
        checkpoint_path: Path,
        final_report_path: Path,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path).resolve()
        final_report_path = Path(final_report_path).resolve()
        report = json.loads(final_report_path.read_text())
        if report.get("status") != "PASS_C51_DENSE_VALUE_FINAL":
            raise ValueError("dense value final report is not PASS C51")
        if report.get("permission") != "GO_FRESH_COUNTERFACTUAL_VALUE_RANKING":
            raise ValueError("C51 did not authorize counterfactual value ranking")
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if report.get("sources", {}).get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("dense value checkpoint identity differs from C51")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "h3wam-c50-dense-future-value-expert-v1":
            raise ValueError("dense value checkpoint format mismatch")
        model = DenseTemporalFutureValueModel(**payload["model_kwargs"])
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False).eval()
        self.model = model
        self.register_buffer("state_mean", payload["normalization"]["state_mean"].float())
        self.register_buffer("state_std", payload["normalization"]["state_std"].float())
        self.ranker_checkpoint_sha256 = checkpoint_sha256
        self.final_report_sha256 = _sha256_file(final_report_path)
        self.consequence_checkpoint_sha256: list[str] = []
        self.to(device)
        self.requires_grad_(False).eval()

    @torch.inference_mode()
    def score(
        self,
        raw_proprio: torch.Tensor,
        h3_features: torch.Tensor,
        environment_actions: torch.Tensor,
    ) -> torch.Tensor:
        if raw_proprio.numel() != 8:
            raise ValueError("raw_proprio must contain exactly eight LIBERO values")
        if h3_features.ndim == 2:
            h3_features = h3_features.unsqueeze(0)
        if h3_features.ndim != 3 or tuple(h3_features.shape[-2:]) != (32, 5376):
            raise ValueError("h3_features must be [1,32,5376]")
        if h3_features.shape[0] != 1:
            raise ValueError("online ranking supports one observation at a time")
        if environment_actions.ndim != 3 or tuple(environment_actions.shape[1:]) != (32, 7):
            raise ValueError("environment_actions must be [N,32,7]")
        candidates = environment_actions.shape[0]
        normalized_state = (
            (raw_proprio.reshape(1, 8).to(self.state_mean).float() - self.state_mean)
            / self.state_std
        ).expand(candidates, -1)
        current = self.model.project_features(h3_features.to(self.state_mean))
        outputs = self.model.forward_projected(
            normalized_state,
            current.expand(candidates, -1),
            environment_actions.to(self.state_mean).float(),
        )
        return -outputs["value"]


__all__ = ["FrozenDenseValueActionRanker"]
