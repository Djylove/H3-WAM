"""Frozen C44 consequence ranker for online best-of-N action selection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .fact_lite_consequence import TemporalFutureH3ConsequenceModel


C44_RANKER_FORMAT = "h3wam-c44-powered-consequence-value-ranker-v1"
C38_TRAIN_DATASET_SHA256 = (
    "2a6c9252b8e77975f58920425bc18110fa8ea63bdc12c4c15571cfffeb9f7459"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenConsequenceActionRanker(nn.Module):
    """Reproduce the frozen C44 consequence-ensemble score online.

    Inputs deliberately use raw LIBERO proprioception and decoded environment
    actions.  This keeps the online contract identical to the C43/C44 causal
    branches rather than silently ranking the action head's normalized space.
    """

    def __init__(
        self,
        ranker_checkpoint: Path,
        consequence_checkpoints: list[Path] | tuple[Path, ...],
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        ranker_checkpoint = Path(ranker_checkpoint).resolve()
        paths = [Path(path).resolve() for path in consequence_checkpoints]
        if len(paths) != 4:
            raise ValueError("C44 online ranker requires exactly four C38 members")
        payload = torch.load(ranker_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != C44_RANKER_FORMAT:
            raise ValueError("C44 ranker format mismatch")
        if payload.get("status") != "PASS_C44_POWERED_CONSEQUENCE_VALUE_RANKING":
            raise ValueError("C44 ranker did not pass its frozen offline gate")
        gate = payload.get("gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            raise ValueError("C44 ranker gate is not PASS")
        expected_hashes = payload.get("ensemble", {}).get("checkpoint_sha256")
        actual_hashes = [_sha256_file(path) for path in paths]
        if expected_hashes != actual_hashes:
            raise ValueError("C44 consequence ensemble identity mismatch")
        if payload.get("ensemble", {}).get("training_dataset_sha256") != C38_TRAIN_DATASET_SHA256:
            raise ValueError("C44 ranker consequence-training identity mismatch")

        normalization = payload["normalization"]
        expected_shapes = {
            "action_mean": (224,), "action_std": (224,),
            "delta_mean": (256,), "delta_std": (256,),
            "action_projection": (224, 32), "consequence_projection": (256, 32),
        }
        for name, shape in expected_shapes.items():
            value = normalization.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"C44 ranker {name} has unexpected shape")
            self.register_buffer(name, value.float().clone())
        weights = payload.get("weights", {}).get("consequence_ensemble")
        if not isinstance(weights, torch.Tensor) or tuple(weights.shape) != (64,):
            raise ValueError("C44 consequence-ensemble weights have unexpected shape")
        self.register_buffer("ranker_weights", weights.float().clone())

        models = []
        state_mean = state_std = None
        for path in paths:
            member = torch.load(path, map_location="cpu", weights_only=False)
            if member.get("model_variant") != "temporal":
                raise ValueError("C44 online ensemble requires temporal C38 members")
            if member.get("contract", {}).get("dataset_sha256") != C38_TRAIN_DATASET_SHA256:
                raise ValueError("C38 member training-dataset identity mismatch")
            model = TemporalFutureH3ConsequenceModel(**member["model_kwargs"])
            model.load_state_dict(member["models"]["conditioned"], strict=True)
            model.requires_grad_(False).eval()
            models.append(model)
            this_mean = member["normalization"]["state_mean"].float()
            this_std = member["normalization"]["state_std"].float()
            if state_mean is None:
                state_mean, state_std = this_mean, this_std
            elif not torch.equal(state_mean, this_mean) or not torch.equal(state_std, this_std):
                raise ValueError("C38 ensemble state normalization differs across members")
        self.models = nn.ModuleList(models)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.ranker_checkpoint_sha256 = _sha256_file(ranker_checkpoint)
        self.consequence_checkpoint_sha256 = actual_hashes
        self.to(device)
        self.requires_grad_(False).eval()

    @torch.inference_mode()
    def score(
        self,
        raw_proprio: torch.Tensor,
        h3_features: torch.Tensor,
        environment_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return one frozen C44 score per candidate action chunk."""

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
        actions = environment_actions.to(self.ranker_weights).float()
        proprio = raw_proprio.reshape(1, 8).to(self.state_mean).float()
        normalized = ((proprio - self.state_mean) / self.state_std).expand(candidates, -1)
        hidden = h3_features.to(self.ranker_weights)

        current = None
        predictions = []
        for model in self.models:
            this_current = model.project_features(hidden)
            if current is None:
                current = this_current
            elif not torch.equal(current, this_current):
                raise RuntimeError("C38 fixed projections differ at runtime")
            predictions.append(model.forward_projected(
                normalized, this_current.expand(candidates, -1), actions
            ))
        predicted = torch.stack(predictions).mean(0)
        action_flat = actions.flatten(1)
        action_design = ((action_flat - self.action_mean) / self.action_std) @ self.action_projection
        delta = predicted - current.expand(candidates, -1)
        consequence_design = ((delta - self.delta_mean) / self.delta_std) @ self.consequence_projection
        design = torch.cat((action_design, consequence_design), dim=1)
        return design @ self.ranker_weights


__all__ = ["C44_RANKER_FORMAT", "FrozenConsequenceActionRanker"]
