"""Lightweight temporal action ensembling without model-runtime imports."""

from __future__ import annotations

import numpy as np


class ActionEnsembler:
    """Average action chunks that predict the same absolute timestep."""

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
