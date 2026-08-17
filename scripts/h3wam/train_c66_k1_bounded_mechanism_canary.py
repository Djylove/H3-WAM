#!/usr/bin/env python3
"""Strict single-variable C66-k1 mechanism canary.

This entry point deliberately reuses the reviewed C66 full-history trainer.
The only training/evaluation mechanism change is that one sample commits the
latest complete feedback chunk instead of all seven chunks.  The full source
row is still materialized, so H3/data I/O and all optimizer/budget code remain
identical to C66.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_full_trainer():
    path = Path(__file__).with_name("train_c66_lingbot_c58_persistent_canary.py")
    spec = importlib.util.spec_from_file_location("_c66_k1_full_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reviewed C66 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C66 = _load_full_trainer()

SOURCE_HISTORY_CHUNKS = 7
SOURCE_OBSERVATION_FRAMES = 15
SOURCE_EXECUTED_ACTIONS = 56
K1_HISTORY_CHUNKS = 1
K1_OBSERVATION_FRAMES = 2
K1_EXECUTED_ACTIONS = 8
K1_FRAME_PREFIX = 13
K1_ACTION_PREFIX = 48
K1_UPDATE_PREFIX = 12


def keep_latest_complete_chunk(sequence: dict[str, Any]) -> dict[str, Any]:
    """Return the same materialized sample with only its final full chunk.

    The preceding full chunk supplies the deterministic history-action-shuffle
    control.  It is never used by the clean arm and is never committed in
    addition to k1.
    """

    history = sequence.get("history")
    if not isinstance(history, list) or len(history) != SOURCE_HISTORY_CHUNKS:
        raise ValueError("C66-k1 requires one frozen seven-chunk source row")
    frame_counts = [int(item["observed_frame_count"]) for item in history]
    action_counts = [int(item["executed_actions"].shape[1]) for item in history]
    if (
        sum(frame_counts) != SOURCE_OBSERVATION_FRAMES
        or sum(action_counts) != SOURCE_EXECUTED_ACTIONS
        or frame_counts[-1] != K1_OBSERVATION_FRAMES
        or action_counts[-1] != K1_EXECUTED_ACTIONS
        or sum(frame_counts[:-1]) != K1_FRAME_PREFIX
        or sum(action_counts[:-1]) != K1_ACTION_PREFIX
    ):
        raise ValueError("C66-k1 source history differs from its frozen contract")
    latest = dict(history[-1])
    donor = history[-2]["executed_actions"]
    if tuple(donor.shape) != tuple(latest["executed_actions"].shape):
        raise ValueError("C66-k1 shuffle donor shape differs from latest action")
    result = dict(sequence)
    result["history"] = [latest]
    result["k1_shuffle_actions"] = donor
    result["k1_absolute_prefix"] = {
        "frame_st_id": K1_FRAME_PREFIX,
        "action_st_id": K1_ACTION_PREFIX,
        "next_update_id": K1_UPDATE_PREFIX,
    }
    return result


_full_materialize_sequence = C66.materialize_sequence


def materialize_sequence(provider, raw, inv_freq, device, dtype):
    # Preserve the exact full C66 H3 materialization path, then bound the
    # committed mechanism.  This avoids introducing a data/cache throughput
    # variable into the canary.
    return keep_latest_complete_chunk(
        _full_materialize_sequence(provider, raw, inv_freq, device, dtype)
    )


def predict_context(
    policy,
    sequence: dict[str, Any],
    noisy,
    timesteps,
    *,
    shuffle_actions: bool,
):
    history = sequence.get("history")
    prefix = sequence.get("k1_absolute_prefix")
    if not isinstance(history, list) or len(history) != K1_HISTORY_CHUNKS:
        raise ValueError("C66-k1 prediction accepts exactly one complete chunk")
    if prefix != {
        "frame_st_id": K1_FRAME_PREFIX,
        "action_st_id": K1_ACTION_PREFIX,
        "next_update_id": K1_UPDATE_PREFIX,
    }:
        raise ValueError("C66-k1 absolute coordinate prefix mismatch")
    state = policy.new_persistent_state(f"{sequence['suite']}:{sequence['episode']}")
    state.frame_st_id = K1_FRAME_PREFIX
    state.action_st_id = K1_ACTION_PREFIX
    state.next_update_id = K1_UPDATE_PREFIX
    feedback = history[0]
    executed = (
        sequence["k1_shuffle_actions"]
        if shuffle_actions
        else feedback["executed_actions"]
    )
    current = sequence["current"]
    policy.commit_executed_feedback(
        state,
        observation_kv=feedback["observation_kv"],
        observed_frame_count=feedback["observed_frame_count"],
        executed_actions=executed,
        text_context=current["text_context"],
        proprio=feedback["proprio"],
        text_mask=current["text_mask"],
    )
    prediction = policy(
        noisy,
        timesteps,
        text_context=current["text_context"],
        proprio=current["proprio"],
        video_kv_cache=sequence["current_kv"],
        text_mask=current["text_mask"],
        persistent_state=state,
    )
    return prediction, state


def rewrite_checkpoint(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    contract = dict(value["contract"])
    contract.update(
        {
            "candidate": "C66_K1_BOUNDED_MECHANISM",
            "classification": (
                "full30_actiondit_on_frozen_int8_h3_with_latest_complete_chunk_only"
            ),
            "source_history_chunks": SOURCE_HISTORY_CHUNKS,
            "source_history_observation_frames": SOURCE_OBSERVATION_FRAMES,
            "source_history_executed_actions": SOURCE_EXECUTED_ACTIONS,
            "history_chunks": K1_HISTORY_CHUNKS,
            "history_observation_frames": K1_OBSERVATION_FRAMES,
            "history_executed_actions": K1_EXECUTED_ACTIONS,
            "absolute_frame_prefix": K1_FRAME_PREFIX,
            "absolute_action_prefix": K1_ACTION_PREFIX,
            "absolute_update_prefix": K1_UPDATE_PREFIX,
            "shuffle_control": "previous_complete_chunk_action_into_latest_chunk",
            "permission_boundary": "mechanism_only_no_long_or_rollout",
        }
    )
    value["contract"] = contract
    return value


def rewrite_report(value: dict[str, Any]) -> dict[str, Any]:
    value = dict(value)
    original_pass = value.get("status") == "PASS_C66_PAIRED_CANARY"
    value.update(
        {
            "event": "h3_c66_k1_bounded_mechanism_canary",
            "status": (
                "PASS_C66_K1_BOUNDED_MECHANISM"
                if original_pass
                else "FAIL_C66_K1_BOUNDED_MECHANISM"
            ),
            "permission": (
                "MECHANISM_SIGNAL_ONLY_NO_LONG_OR_ROLLOUT"
                if original_pass
                else "NO_GO_C66_K1_LONG_OR_ROLLOUT"
            ),
            "effect_status": "NOT_LIBERO_EVIDENCE",
            "committed_history_chunks": K1_HISTORY_CHUNKS,
            "committed_history_observation_frames": K1_OBSERVATION_FRAMES,
            "committed_history_executed_actions": K1_EXECUTED_ACTIONS,
            "source_history_chunks": SOURCE_HISTORY_CHUNKS,
            "shuffle_control": "previous_complete_chunk_action_into_latest_chunk",
            "boundary": (
                "Fresh-parent fixed-k1 offline paired MSE is mechanism evidence only; "
                "it cannot authorize long training or rollout."
            ),
        }
    )
    gates = dict(value["gates"])
    gates.update(
        {
            "k1_exactly_one_complete_chunk": True,
            "k1_absolute_coordinates_preserved": True,
            "k1_shuffle_donor_is_distinct_previous_chunk": True,
            "no_long_or_rollout_permission": True,
        }
    )
    value["gates"] = gates
    return value


_atomic_torch = C66.atomic_torch
_atomic_json = C66.atomic_json


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    _atomic_torch(path, rewrite_checkpoint(value))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_json(path, rewrite_report(value))


def main() -> None:
    # The base module resolves these globals at call time, so optimizer, data,
    # seed, DDP, evaluation and gates remain the reviewed implementation.
    C66.materialize_sequence = materialize_sequence
    C66.predict_context = predict_context
    C66.atomic_torch = atomic_torch
    C66.atomic_json = atomic_json
    C66.main()


if __name__ == "__main__":
    main()
