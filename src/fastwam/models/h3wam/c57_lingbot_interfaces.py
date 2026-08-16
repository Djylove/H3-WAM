"""Training and rollout interfaces for the C57 LingBot persistent-KV port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .int8_online import H3Int8LayoutFunctions
from .lingbot_persistent_kv import (
    H3LingBotPersistentKVPolicy,
    LingBotPersistentKVState,
    merge_observation_kv_sequence,
)


@dataclass(frozen=True)
class LingBotTeacherForcedFeedback:
    """Real feedback available before one teacher-forced current decision."""

    observation_kv: Mapping[int, Mapping[str, torch.Tensor]]
    observed_frame_count: int
    executed_actions: torch.Tensor
    proprio: torch.Tensor


def forward_teacher_forced_history(
    model: H3LingBotPersistentKVPolicy,
    *,
    episode_key: str,
    history: Sequence[LingBotTeacherForcedFeedback],
    noisy_actions: torch.Tensor,
    timestep: torch.Tensor,
    text_context: torch.Tensor,
    proprio: torch.Tensor,
    current_observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
    text_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LingBotPersistentKVState]:
    """Replay exactly the same real-feedback commits used by deployment.

    A fresh state per training example keeps episode boundaries explicit.  No
    tensors are detached, so current action loss can train the clean executed-
    action K/V path through every ActionDiT layer.
    """

    if not model.persistent_enabled:
        raise ValueError("teacher-forced persistent history requires an enabled model")
    state = model.new_persistent_state(episode_key)
    for feedback in history:
        model.commit_executed_feedback(
            state,
            observation_kv=feedback.observation_kv,
            observed_frame_count=feedback.observed_frame_count,
            executed_actions=feedback.executed_actions,
            text_context=text_context,
            proprio=feedback.proprio,
            text_mask=text_mask,
        )
    prediction = model(
        noisy_actions,
        timestep,
        text_context=text_context,
        proprio=proprio,
        video_kv_cache=current_observation_kv,
        text_mask=text_mask,
        persistent_state=state,
    )
    return prediction, state


class LingBotPersistentRolloutSession:
    """Fail-closed episode state machine matching LingBot client/server order."""

    def __init__(
        self, model: H3LingBotPersistentKVPolicy, *, episode_key: str
    ) -> None:
        if not model.persistent_enabled:
            raise ValueError("rollout session requires persistent_enabled=True")
        self.model = model
        self.state = model.new_persistent_state(episode_key)
        self._initial_observation: Mapping[int, Mapping[str, torch.Tensor]] | None = None
        self._awaiting_feedback = False

    @property
    def next_frame_st_id(self) -> int:
        return self.state.frame_st_id

    @property
    def next_action_st_id(self) -> int:
        return self.state.action_st_id

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        current_observation_kv: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
        final_denoise_step: bool,
    ) -> torch.Tensor:
        if self._awaiting_feedback:
            raise RuntimeError("real executed feedback is required before the next prediction")
        if self.state.frame_st_id == 0 and self._initial_observation is None:
            self._initial_observation = current_observation_kv
        prediction = self.model(
            noisy_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=current_observation_kv,
            text_mask=text_mask,
            persistent_state=self.state,
            stage_prediction=final_denoise_step,
        )
        if final_denoise_step:
            self._awaiting_feedback = True
        return prediction

    def commit_real_feedback(
        self,
        *,
        observed_after_execution: Sequence[
            Mapping[int, Mapping[str, torch.Tensor]]
        ],
        executed_actions: torch.Tensor,
        text_context: torch.Tensor,
        proprio_at_decision: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> None:
        if not self._awaiting_feedback or not self.state.has_predicted:
            raise RuntimeError("cannot commit feedback without one staged prediction")
        observations = list(observed_after_execution)
        if self.state.frame_st_id == 0:
            if self._initial_observation is None:
                raise RuntimeError("first feedback lost the initial observation")
            observations.insert(0, self._initial_observation)
        if not observations:
            raise ValueError("feedback must carry at least one observed frame")
        merged = merge_observation_kv_sequence(
            observations, layers=self.model.carrier_layers
        )
        self.model.commit_executed_feedback(
            self.state,
            observation_kv=merged,
            observed_frame_count=len(observations),
            executed_actions=executed_actions,
            text_context=text_context,
            proprio=proprio_at_decision,
            text_mask=text_mask,
        )
        self._initial_observation = None
        self._awaiting_feedback = False

    def snapshot(self) -> dict[str, object]:
        if self._awaiting_feedback:
            raise RuntimeError("do not checkpoint an episode between prediction and feedback")
        return self.state.snapshot()


def offset_h3_layout_functions(
    base: H3Int8LayoutFunctions,
    *,
    frame_st_id: int,
) -> H3Int8LayoutFunctions:
    """Give live H3 video/audio rows an absolute streaming temporal position."""

    if frame_st_id < 0:
        raise ValueError("frame_st_id must be non-negative")

    def build_packed_sequence(**kwargs):
        packed = list(base.build_packed_sequence(**kwargs))
        position_ids = packed[0].clone()
        video_indices = packed[2].long()
        audio_indices = packed[3].long()
        non_text = torch.cat((video_indices, audio_indices))
        position_ids[non_text, 0] += frame_st_id
        packed[0] = position_ids
        return tuple(packed)

    return H3Int8LayoutFunctions(
        build_packed_sequence=build_packed_sequence,
        build_row_timesteps=base.build_row_timesteps,
        patchify_video_latents=base.patchify_video_latents,
    )
