"""Fail-closed LIBERO wire for C57 persistent observation/action feedback.

This is deliberately a transaction protocol, not an ``executed history``
feature.  A policy prediction opens one replan transaction.  The simulator
then reports the actions actually accepted by ``env.step`` and real
observations exactly after actions 4 and 8.  Only the action-8 report commits
the transaction into the model's persistent K/V state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def _observation_wire(obs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.ascontiguousarray(obs[name], dtype=np.uint8)
        result[f"{name}_bytes"] = image.tobytes()
        result[f"{name}_shape"] = tuple(int(value) for value in image.shape)
    for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        result[name] = np.asarray(obs[name], dtype=np.float32).tolist()
    return result


def observation_digest(obs: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    agent = obs["agentview_image"]
    wrist = obs.get("robot0_eye_in_hand_image", obs.get("wristview_image"))
    if wrist is None:
        raise KeyError("robot0_eye_in_hand_image")
    for image in (agent, wrist):
        digest.update(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
    return digest.hexdigest()


@dataclass
class C57TraceCounters:
    episode_key: str
    transaction_id: int = 0
    action_count: int = 0
    prediction_open: bool = False


class C57PersistentFeedbackClient:
    """Simulator-side reset/predict/obs4/commit8 lifecycle."""

    def __init__(self, connection, *, episode_key: str, task: str) -> None:
        self.connection = connection
        self.task = str(task)
        self.counters = C57TraceCounters(str(episode_key))
        self._actions: list[list[float]] = []

    def _roundtrip(self, message: dict[str, Any]) -> dict[str, Any]:
        self.connection.send(message)
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "C57 feedback server failed"))
        return response

    def reset(self) -> dict[str, Any]:
        self._actions.clear()
        self.counters.action_count = 0
        self.counters.prediction_open = False
        return self._roundtrip(
            {
                "command": "c57_reset",
                "episode_key": self.counters.episode_key,
                "task": self.task,
            }
        )

    def mark_prediction(self) -> None:
        if self.counters.prediction_open:
            raise RuntimeError("C57 prediction already awaits executed feedback")
        self.counters.prediction_open = True
        self.counters.action_count = 0
        self._actions.clear()

    def record_step(self, action: np.ndarray, obs_after: dict[str, Any]) -> None:
        if not self.counters.prediction_open:
            raise RuntimeError("executed feedback arrived before a C57 prediction")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (7,) or not np.isfinite(action_array).all():
            raise ValueError("one finite executed LIBERO action must have shape [7]")
        self._actions.append(action_array.tolist())
        self.counters.action_count += 1
        if self.counters.action_count not in (4, 8):
            return
        response = self._roundtrip(
            {
                "command": "c57_feedback",
                "episode_key": self.counters.episode_key,
                "task": self.task,
                "transaction_id": self.counters.transaction_id,
                "action_count": self.counters.action_count,
                "executed_environment_actions": list(self._actions),
                "observation_sha256": observation_digest(obs_after),
                **_observation_wire(obs_after),
            }
        )
        if int(response["action_count"]) != self.counters.action_count:
            raise RuntimeError("C57 feedback acknowledgement changed action count")
        self._actions.clear()
        if self.counters.action_count == 8:
            if response.get("committed") is not True:
                raise RuntimeError("C57 action-8 feedback was not committed")
            self.counters.prediction_open = False
            self.counters.transaction_id += 1

    def discard_terminal_tail(self) -> None:
        """An episode may terminate before eight actions; reset owns rollback."""

        self._actions.clear()


class C57FeedbackEnv:
    """Transparent LIBERO env wrapper that reports *post-step* observations."""

    def __init__(self, env, client: C57PersistentFeedbackClient) -> None:
        self._env = env
        self._client = client

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action):
        result = self._env.step(action)
        obs = result[0]
        self._client.record_step(np.asarray(action, dtype=np.float32), obs)
        return result
