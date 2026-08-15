"""Frozen H3 task-progress probe used for diagnostic shadow rollout only."""

from __future__ import annotations

from pathlib import Path

import torch


PROGRESS_PROBE_FORMAT = "h3wam-frozen-h3-progress-ridge-v1"
TIMEBLIND_PROGRESS_PROBE_FORMAT = "h3wam-frozen-h3-timeblind-progress-ridge-v1"
PROGRESS_DESIGN_CONTRACT = (
    "context onehot + absolute_step/400 + layer49 K/V 512D compact feature"
)
PROGRESS_FEATURE_CONTRACT = "concat(mean_k,std_k,mean_v,std_v) over token/head"
TIMEBLIND_PROGRESS_DESIGN_CONTRACT = (
    "context onehot + layer49 K/V 512D compact feature"
)


def compact_h3_kv_progress_feature(layer_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reduce one 32-token H3 K/V layer to the exact 512D probe feature."""

    values = []
    for name in ("k", "v"):
        if name not in layer_cache:
            raise ValueError(f"H3 progress layer cache is missing {name!r}")
        tensor = layer_cache[name]
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tuple(tensor.shape) != (32, 56, 128):
            raise ValueError(f"unexpected H3 K/V shape: {tuple(tensor.shape)}")
        tensor = tensor.float()
        values.extend(
            (tensor.mean(dim=(0, 1)), tensor.std(dim=(0, 1), unbiased=False))
        )
    return torch.cat(values)


class FrozenH3ProgressProbe:
    """Strict, CPU-only ridge probe that cannot affect action generation."""

    def __init__(self, payload: dict) -> None:
        if payload.get("format") not in (
            PROGRESS_PROBE_FORMAT,
            TIMEBLIND_PROGRESS_PROBE_FORMAT,
        ):
            raise ValueError("unsupported H3 progress probe format")
        self.format = str(payload["format"])
        if payload.get("validation_status") not in (
            "PASS_PROGRESS_FEATURE_GATE",
            "PASS_TIMEBLIND_PROGRESS_FEATURE_GATE",
        ):
            raise ValueError("H3 progress probe did not pass its validation gate")
        design_contract = payload.get("design_contract")
        if design_contract not in (
            PROGRESS_DESIGN_CONTRACT,
            TIMEBLIND_PROGRESS_DESIGN_CONTRACT,
        ):
            raise ValueError("H3 progress probe design contract mismatch")
        if payload.get("feature_contract") != PROGRESS_FEATURE_CONTRACT:
            raise ValueError("H3 progress probe feature contract mismatch")
        contexts = tuple(str(value) for value in payload.get("contexts", ()))
        if len(contexts) != 40 or len(set(contexts)) != len(contexts):
            raise ValueError("H3 progress probe requires 40 unique task contexts")
        self.contexts = contexts
        self.context_indices = {value: index for index, value in enumerate(contexts)}
        self.include_absolute_step = design_contract == PROGRESS_DESIGN_CONTRACT
        design_width = 553 if self.include_absolute_step else 552
        self.mean = self._tensor(payload, "mean", (design_width,))
        self.std = self._tensor(payload, "std", (design_width,))
        self.weights = self._tensor(payload, "weights", (design_width + 1,))
        if torch.any(self.std <= 0):
            raise ValueError("H3 progress probe normalization std must be positive")

    @staticmethod
    def _tensor(payload: dict, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"H3 progress probe {name} must have shape {shape}")
        value = value.detach().to(device="cpu", dtype=torch.float64).clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"H3 progress probe {name} is not finite")
        return value

    @classmethod
    def load(cls, path: Path | str) -> "FrozenH3ProgressProbe":
        payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("H3 progress probe payload must be a dictionary")
        return cls(payload)

    def predict(
        self,
        *,
        context_id: str,
        absolute_step: int,
        layer_cache: dict[str, torch.Tensor],
    ) -> float:
        if context_id not in self.context_indices:
            raise ValueError(f"context is absent from H3 progress probe: {context_id}")
        if absolute_step < 0:
            raise ValueError("absolute step must be non-negative")
        context = torch.zeros(len(self.contexts), dtype=torch.float64)
        context[self.context_indices[context_id]] = 1.0
        feature = compact_h3_kv_progress_feature(layer_cache).double().cpu()
        columns = [context]
        if self.include_absolute_step:
            columns.append(
                torch.tensor([absolute_step / 400.0], dtype=torch.float64)
            )
        columns.append(feature)
        design = torch.cat(columns, dim=0)
        normalized = (design - self.mean) / self.std
        prediction = torch.dot(
            torch.cat((torch.ones(1, dtype=torch.float64), normalized)), self.weights
        )
        return float(prediction.clamp(0.0, 1.0))
