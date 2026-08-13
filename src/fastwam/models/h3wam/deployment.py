"""Deployment helpers shared by the H3-WAM LIBERO rollout processes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


class ActionEnsembler:
    """Fuse overlapping action chunks at their absolute execution timestamp.

    This mirrors the temporal ensemble used by the FastWAM LIBERO evaluator:
    every replan contributes a prediction for each future timestamp and the
    executor averages all predictions that target the same timestamp.  Keeping
    this in the H3 deployment module makes the rollout behavior independently
    testable without importing the Hydra-based FastWAM evaluator.
    """

    def __init__(self) -> None:
        self._action_cache: dict[int, list[np.ndarray]] = {}

    def reset(self) -> None:
        self._action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int) -> None:
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"action_chunk must have shape [T, D], got {actions.shape}")
        if start_timestamp < 0:
            raise ValueError("start_timestamp must be non-negative")
        for offset, action in enumerate(actions):
            self._action_cache.setdefault(start_timestamp + offset, []).append(
                action.copy()
            )

    def get_action(self, timestamp: int) -> np.ndarray:
        predictions = self._action_cache.get(timestamp)
        if not predictions:
            raise ValueError(f"no actions cached for timestamp {timestamp}")
        return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float32)

    def prediction_count(self, timestamp: int) -> int:
        return len(self._action_cache.get(timestamp, ()))

    def cleanup(self, current_timestamp: int) -> None:
        for timestamp in tuple(self._action_cache):
            if timestamp < current_timestamp:
                del self._action_cache[timestamp]


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert a LIBERO/robosuite ``(x, y, z, w)`` quaternion."""

    quaternion = np.asarray(quaternion, dtype=np.float32).reshape(4)
    scalar = float(np.clip(quaternion[3], -1.0, 1.0))
    denominator = math.sqrt(max(1.0 - scalar * scalar, 0.0))
    if math.isclose(denominator, 0.0, abs_tol=1e-8):
        return np.zeros(3, dtype=np.float32)
    return (
        quaternion[:3] * np.float32(2.0 * math.acos(scalar) / denominator)
    ).astype(np.float32)


def libero_observation_state(observation: dict) -> torch.Tensor:
    """Build the 8-D state used by the converted FastWAM LIBERO data."""

    state = np.concatenate(
        (
            np.asarray(observation["eef_pos"], dtype=np.float32).reshape(3),
            quaternion_to_axis_angle(observation["eef_quat"]),
            np.asarray(observation["gripper_qpos"], dtype=np.float32).reshape(2),
        )
    )
    return torch.from_numpy(state)


def preprocess_libero_cameras(
    agentview: np.ndarray,
    wristview: np.ndarray,
    *,
    camera_height: int = 224,
    camera_width: int = 224,
) -> torch.Tensor:
    """Rotate, resize and horizontally concatenate two LIBERO RGB cameras.

    Returns one H3 VAE input frame in ``[N, H, W, C]`` and ``[0, 1]``.
    """

    cameras = []
    for name, image in (("agentview", agentview), ("wristview", wristview)):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{name} must be an HWC RGB image, got {array.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(array[::-1, ::-1])).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).float().div_(255.0)
        tensor = functional.interpolate(
            tensor,
            size=(camera_height, camera_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        cameras.append(tensor)
    combined = torch.cat(cameras, dim=-1)
    return combined.permute(0, 2, 3, 1).contiguous()


def minmax_normalize(
    value: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    scale = (maximum.float() - minimum.float()).clamp_min(1e-6)
    return 2.0 * (value.float() - minimum.float()) / scale - 1.0


def minmax_denormalize(
    value: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    return (value.float() + 1.0) * 0.5 * (maximum.float() - minimum.float()) + minimum.float()


def action_denormalization_bounds(
    action_normalization: str,
    cache_stats: dict,
    quantile_stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the exact action bounds carried by a training checkpoint."""

    if action_normalization == "minmax":
        low = torch.as_tensor(cache_stats["action_min"], dtype=torch.float32)
        high = torch.as_tensor(cache_stats["action_max"], dtype=torch.float32)
    elif action_normalization == "quantile":
        if quantile_stats is None:
            raise ValueError("quantile checkpoint is missing action_quantile_stats")
        low = torch.as_tensor(quantile_stats.get("q01", []), dtype=torch.float32)
        high = torch.as_tensor(quantile_stats.get("q99", []), dtype=torch.float32)
    else:
        raise ValueError(f"unsupported action normalization: {action_normalization}")
    if low.shape != (7,) or high.shape != (7,):
        raise ValueError(
            f"action denormalization bounds must be seven-dimensional, got {low.shape}/{high.shape}"
        )
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        raise ValueError("action denormalization bounds must be finite")
    if torch.any(high <= low):
        raise ValueError("action denormalization upper bounds must exceed lower bounds")
    return low, high


def libero_environment_actions(
    normalized_actions: torch.Tensor,
    action_minimum: torch.Tensor,
    action_maximum: torch.Tensor,
    *,
    binarize_gripper: bool = False,
    temporal_median_window: int = 1,
) -> np.ndarray:
    """Undo dataset normalization and convert its [0,1] gripper to LIBERO."""

    actions = minmax_denormalize(
        normalized_actions.detach().cpu(),
        action_minimum.cpu(),
        action_maximum.cpu(),
    ).numpy()
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"expected actions [T, 7], got {actions.shape}")
    # The FastWAM dataset uses 1=open and 0=close. LIBERO uses -1=open,
    # +1=close, hence -(2*x-1).
    actions[..., -1] = -(2.0 * actions[..., -1] - 1.0)
    if temporal_median_window <= 0 or temporal_median_window % 2 == 0:
        raise ValueError("temporal_median_window must be a positive odd integer")
    if temporal_median_window > 1:
        radius = temporal_median_window // 2
        motion = actions[:, :6].copy()
        padded = np.pad(motion, ((radius, radius), (0, 0)), mode="edge")
        actions[:, :6] = np.stack(
            [
                np.median(padded[index : index + temporal_median_window], axis=0)
                for index in range(len(motion))
            ]
        )
    if binarize_gripper:
        actions[..., -1] = np.sign(actions[..., -1])
    if not np.isfinite(actions).all():
        raise FloatingPointError("policy produced non-finite LIBERO actions")
    return np.clip(actions, -1.0, 1.0).astype(np.float32)


def libero_dataset_action(environment_action: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert one LIBERO environment action to the dataset gripper convention."""

    action = torch.as_tensor(environment_action, dtype=torch.float32).clone()
    if action.shape != (7,):
        raise ValueError(f"expected one action [7], got {tuple(action.shape)}")
    action[-1] = (1.0 - action[-1]) * 0.5
    return action


def load_cached_task_context(
    cache_root: Path, task: str, context_id: str | None = None
) -> dict:
    """Load one fixed Qwen/H3 context for a task from the training cache."""

    cache_root = Path(cache_root)
    selected_id = context_id
    if selected_id is None:
        with (cache_root / "manifest.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("task") == task:
                    selected_id = item["id"]
                    break
    if selected_id is None:
        raise KeyError(f"task is absent from the cached manifest: {task!r}")
    context = torch.load(
        cache_root / "refined_contexts" / f"{selected_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {
        "id": selected_id,
        "task": task,
        "context": context["context"],
        "token_tags": context["token_tags"],
    }
