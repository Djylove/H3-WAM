from __future__ import annotations

import copy
import unittest

import torch

from fastwam.models.h3wam.c57_lingbot_interfaces import (
    LingBotPersistentRolloutSession,
    LingBotTeacherForcedFeedback,
    forward_teacher_forced_history,
    offset_h3_layout_functions,
)
from fastwam.models.h3wam.c57_lingbot_rollout_wire import C57ExecutedFeedbackWire
from fastwam.models.h3wam.dreamwam_kv_carrier import (
    H3DreamWAMKVCarrierPolicy,
    REPEAT_LAYER49_CARRIER_SOURCE,
)
from fastwam.models.h3wam.int8_online import H3Int8LayoutFunctions
from fastwam.models.h3wam.lingbot_persistent_kv import (
    H3LingBotPersistentKVPolicy,
    LingBotPersistentKVState,
    merge_observation_kv_sequence,
)


def model_kwargs() -> dict:
    return {
        "enabled": True,
        "carrier_layers": (49,),
        "carrier_source_mode": REPEAT_LAYER49_CARRIER_SOURCE,
        "action_dim": 2,
        "proprio_dim": 3,
        "context_dim": 12,
        "hidden_dim": 16,
        "ffn_dim": 32,
        "num_heads": 2,
        "attn_head_dim": 4,
        "freq_dim": 8,
    }


def make_inputs(seed: int = 3) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "actions": torch.randn(1, 2, 2, generator=generator),
        "timestep": torch.tensor([0.75]),
        "text": torch.randn(1, 3, 12, generator=generator),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "proprio": torch.randn(1, 3, generator=generator),
        "cache": {
            49: {
                "k": torch.randn(1, 3, 2, 4, generator=generator),
                "v": torch.randn(1, 3, 2, 4, generator=generator),
            }
        },
    }


def call_model(model, inputs: dict, **extra) -> torch.Tensor:
    return model(
        inputs["actions"],
        inputs["timestep"],
        text_context=inputs["text"],
        proprio=inputs["proprio"],
        video_kv_cache=inputs["cache"],
        text_mask=inputs["text_mask"],
        **extra,
    )


class C57LingBotPersistentKVTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_disabled_is_bit_exact_and_state_dict_exact(self) -> None:
        parent = H3DreamWAMKVCarrierPolicy(**model_kwargs()).eval()
        candidate = H3LingBotPersistentKVPolicy(
            persistent_enabled=False,
            persistent_window_chunks=2,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).eval()
        candidate.load_state_dict(parent.state_dict(), strict=True)
        self.assertEqual(set(parent.state_dict()), set(candidate.state_dict()))
        inputs = make_inputs()
        with torch.no_grad():
            expected = call_model(parent, inputs)
            actual = call_model(candidate, inputs)
        self.assertTrue(torch.equal(expected, actual))

    def test_predicted_cache_is_replaced_by_real_feedback(self) -> None:
        model = H3LingBotPersistentKVPolicy(
            persistent_enabled=True,
            persistent_window_chunks=2,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).eval()
        inputs = make_inputs()
        state = model.new_persistent_state("episode-0")
        call_model(model, inputs, persistent_state=state, stage_prediction=True)
        self.assertTrue(state.has_predicted)
        self.assertEqual(
            state.audit()["layers"]["49"]["kinds"],
            ["predicted_observation", "predicted_action"],
        )

        model.commit_executed_feedback(
            state,
            observation_kv=inputs["cache"],
            observed_frame_count=1,
            executed_actions=inputs["actions"],
            text_context=inputs["text"],
            proprio=inputs["proprio"],
            text_mask=inputs["text_mask"],
        )
        audit = state.audit()
        self.assertFalse(audit["has_predicted"])
        self.assertEqual(audit["frame_st_id"], 1)
        self.assertEqual(audit["action_st_id"], 2)
        self.assertEqual(audit["layers"]["49"]["kinds"], ["observation", "action"])

        fresh = model.new_persistent_state("fresh")
        with torch.no_grad():
            with_history = call_model(model, inputs, persistent_state=state)
            without_history = call_model(model, inputs, persistent_state=fresh)
        self.assertFalse(torch.equal(with_history, without_history))

    def test_rolling_cache_evicts_oldest_update(self) -> None:
        state = LingBotPersistentKVState(
            layers=(49,), token_capacity=4, episode_key="evict-0"
        )
        for update_id in range(3):
            value = torch.full((1, 2, 8), float(update_id))
            state.append_layer(
                49,
                kind="action",
                key=value,
                value=value + 10,
                update_id=update_id,
                action_start=update_id * 2,
                action_count=2,
            )
        audit = state.audit()["layers"]["49"]
        self.assertEqual(audit["tokens"], 4)
        self.assertEqual(audit["update_ids"], [1, 2])
        key, value = state.materialize(49)
        self.assertTrue(torch.equal(key[:, :2], torch.ones(1, 2, 8)))
        self.assertTrue(torch.equal(value[:, 2:], torch.full((1, 2, 8), 12.0)))

    def test_teacher_forced_history_reaches_clean_action_kv_projection(self) -> None:
        model = H3LingBotPersistentKVPolicy(
            persistent_enabled=True,
            persistent_window_chunks=2,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).train()
        inputs = make_inputs()
        feedback = LingBotTeacherForcedFeedback(
            observation_kv=inputs["cache"],
            observed_frame_count=1,
            executed_actions=inputs["actions"].clone(),
            proprio=inputs["proprio"].clone(),
        )
        prediction, state = forward_teacher_forced_history(
            model,
            episode_key="train-0",
            history=[feedback],
            noisy_actions=inputs["actions"] * 0.7,
            timestep=inputs["timestep"],
            text_context=inputs["text"],
            proprio=inputs["proprio"],
            current_observation_kv=inputs["cache"],
            text_mask=inputs["text_mask"],
        )
        prediction.square().mean().backward()
        gradient = model.action_expert.blocks[0].self_attn.k.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(state.action_st_id, 2)

    def test_model_and_runtime_snapshot_restore_exactly(self) -> None:
        model = H3LingBotPersistentKVPolicy(
            persistent_enabled=True,
            persistent_window_chunks=2,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).eval()
        inputs = make_inputs()
        state = model.new_persistent_state("restore-0")
        model.commit_executed_feedback(
            state,
            observation_kv=inputs["cache"],
            observed_frame_count=1,
            executed_actions=inputs["actions"],
            text_context=inputs["text"],
            proprio=inputs["proprio"],
            text_mask=inputs["text_mask"],
        )
        restored_model = H3LingBotPersistentKVPolicy(
            persistent_enabled=True,
            persistent_window_chunks=2,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).eval()
        restored_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        restored_state = LingBotPersistentKVState.from_snapshot(state.snapshot())
        with torch.no_grad():
            expected = call_model(model, inputs, persistent_state=state)
            actual = call_model(restored_model, inputs, persistent_state=restored_state)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(state.audit(), restored_state.audit())

    def test_rollout_requires_feedback_and_first_commit_includes_initial_frame(self) -> None:
        model = H3LingBotPersistentKVPolicy(
            persistent_enabled=True,
            persistent_window_chunks=3,
            observation_tokens_per_chunk=3,
            action_tokens_per_chunk=2,
            **model_kwargs(),
        ).eval()
        inputs = make_inputs()
        post = make_inputs(seed=7)["cache"]
        session = LingBotPersistentRolloutSession(model, episode_key="rollout-0")
        session.predict_velocity(
            inputs["actions"],
            inputs["timestep"],
            text_context=inputs["text"],
            proprio=inputs["proprio"],
            current_observation_kv=inputs["cache"],
            text_mask=inputs["text_mask"],
            final_denoise_step=True,
        )
        with self.assertRaisesRegex(RuntimeError, "feedback"):
            session.predict_velocity(
                inputs["actions"],
                inputs["timestep"],
                text_context=inputs["text"],
                proprio=inputs["proprio"],
                current_observation_kv=inputs["cache"],
                text_mask=inputs["text_mask"],
                final_denoise_step=False,
            )
        session.commit_real_feedback(
            observed_after_execution=[post],
            executed_actions=inputs["actions"],
            text_context=inputs["text"],
            proprio_at_decision=inputs["proprio"],
            text_mask=inputs["text_mask"],
        )
        self.assertEqual(session.next_frame_st_id, 2)
        self.assertEqual(session.next_action_st_id, 2)
        self.assertFalse(session.state.has_predicted)
        self.assertEqual(session.snapshot()["schema_version"], 1)

    def test_h3_layout_temporal_offset_excludes_text(self) -> None:
        def build_packed_sequence(**kwargs):
            del kwargs
            return (
                torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
                torch.tensor([0, 2, 1]),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([2]),
                1,
                0,
            )

        base = H3Int8LayoutFunctions(
            build_packed_sequence=build_packed_sequence,
            build_row_timesteps=lambda **kwargs: kwargs,
            patchify_video_latents=lambda value, patch: (value, patch),
        )
        offset = offset_h3_layout_functions(base, frame_st_id=5)
        packed = offset.build_packed_sequence()
        self.assertTrue(
            torch.equal(
                packed[0],
                torch.tensor([[5, 0, 0], [6, 0, 0], [0, 0, 0]]),
            )
        )

    def test_rollout_wire_commits_only_executed_replan8_at_cadence(self) -> None:
        class Session:
            def __init__(self):
                self.calls = []

            def commit_real_feedback(self, **kwargs):
                self.calls.append(kwargs)

        session = Session()
        wire = C57ExecutedFeedbackWire(replan=8, observe_every=4)
        observation = {49: {"k": torch.ones(1, 1, 1, 1), "v": torch.ones(1, 1, 1, 1)}}
        for index in range(8):
            wire.record_executed_action(
                torch.full((7,), float(index)),
                observation_after_action=observation if index in (3, 7) else None,
            )
        self.assertTrue(wire.ready)
        wire.commit(
            session,
            text_context=torch.zeros(1, 1, 1),
            proprio_at_decision=torch.zeros(1, 8),
        )
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(tuple(session.calls[0]["executed_actions"].shape), (1, 8, 7))
        self.assertEqual(len(session.calls[0]["observed_after_execution"]), 2)
        self.assertEqual(wire.pending_actions, 0)

    def test_observation_merge_uses_sequence_dimension_before_and_after_collate(self) -> None:
        for batched, expected in ((False, (6, 2, 4)), (True, (1, 6, 2, 4))):
            shape = (1, 3, 2, 4) if batched else (3, 2, 4)
            item = {49: {"k": torch.ones(shape), "v": torch.ones(shape)}}
            merged = merge_observation_kv_sequence([item, item], layers=(49,))
            self.assertEqual(tuple(merged[49]["k"].shape), expected)


if __name__ == "__main__":
    unittest.main()
