"""C57 rollout-side executed-action/observation feedback wire.

The simulator records every action actually accepted by ``env.step`` and an
H3 K/V encoding after actions 4 and 8.  Only a complete replan-8 transaction
may be committed, preventing planned actions from masquerading as execution.
"""

from __future__ import annotations

from typing import Mapping

import torch

from .c57_lingbot_interfaces import LingBotPersistentRolloutSession


class C57ExecutedFeedbackWire:
    def __init__(self, *, replan: int = 8, observe_every: int = 4) -> None:
        if replan <= 0 or observe_every <= 0 or replan % observe_every:
            raise ValueError("replan must be a positive multiple of observe_every")
        self.replan = int(replan)
        self.observe_every = int(observe_every)
        self._actions: list[torch.Tensor] = []
        self._observations: list[Mapping[int, Mapping[str, torch.Tensor]]] = []

    @property
    def pending_actions(self) -> int:
        return len(self._actions)

    @property
    def ready(self) -> bool:
        return len(self._actions) == self.replan

    def record_executed_action(
        self,
        action: torch.Tensor,
        *,
        observation_after_action: Mapping[int, Mapping[str, torch.Tensor]] | None,
    ) -> None:
        if self.ready:
            raise RuntimeError("commit the completed feedback transaction first")
        if action.ndim != 1 or action.shape[-1] != 7:
            raise ValueError("one executed normalized action must have shape [7]")
        next_count = len(self._actions) + 1
        observation_due = next_count % self.observe_every == 0
        if observation_due != (observation_after_action is not None):
            raise ValueError(
                "post-action observation K/V is required exactly at observation cadence"
            )
        self._actions.append(action)
        if observation_after_action is not None:
            self._observations.append(observation_after_action)

    def commit(
        self,
        session: LingBotPersistentRolloutSession,
        *,
        text_context: torch.Tensor,
        proprio_at_decision: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> None:
        if not self.ready:
            raise RuntimeError("feedback transaction is incomplete")
        executed = torch.stack(self._actions, dim=0).unsqueeze(0)
        session.commit_real_feedback(
            observed_after_execution=list(self._observations),
            executed_actions=executed,
            text_context=text_context,
            proprio_at_decision=proprio_at_decision,
            text_mask=text_mask,
        )
        self._actions.clear()
        self._observations.clear()

    def discard_on_reset(self) -> None:
        """Discard only an uncommitted episode tail during an explicit reset."""

        self._actions.clear()
        self._observations.clear()
