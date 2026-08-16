#!/usr/bin/env python3
"""Standalone INT8-H3 server for the C57 persistent-KV action policy."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam.c57_lingbot_interfaces import (  # noqa: E402
    LingBotPersistentRolloutSession,
    offset_h3_layout_functions,
)
from fastwam.models.h3wam.c57_lingbot_libero_trace import (  # noqa: E402
    observation_digest,
)
from fastwam.models.h3wam.c57_lingbot_rollout_wire import (  # noqa: E402
    C57ExecutedFeedbackWire,
)
from fastwam.models.h3wam.deployment import (  # noqa: E402
    libero_environment_actions,
    libero_observation_state,
    minmax_normalize,
    normalize_libero_environment_action_history,
    preprocess_libero_cameras,
)
from fastwam.models.h3wam.int8_online import _official_layout_functions  # noqa: E402
from fastwam.models.h3wam.lingbot_persistent_kv import (  # noqa: E402
    H3LingBotPersistentKVPolicy,
)


def _load_base():
    path = Path(__file__).with_name("serve_rollout_policy.py")
    spec = importlib.util.spec_from_file_location("_c57_policy_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load rollout server: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _decode_observation(request: dict) -> None:
    def take(primary: str, legacy: str):
        if primary in request:
            return request.pop(primary)
        return request.pop(legacy)

    request["agentview_image"] = np.frombuffer(
        take("agentview_image_bytes", "agentview_bytes"), dtype=np.uint8
    ).reshape(take("agentview_image_shape", "agentview_shape"))
    request["wristview_image"] = np.frombuffer(
        take("robot0_eye_in_hand_image_bytes", "wristview_bytes"), dtype=np.uint8
    ).reshape(take("robot0_eye_in_hand_image_shape", "wristview_shape"))
    for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        legacy = {
            "robot0_eef_pos": "eef_pos",
            "robot0_eef_quat": "eef_quat",
            "robot0_gripper_qpos": "gripper_qpos",
        }[name]
        request[name] = np.asarray(request.get(name, request.get(legacy)), dtype=np.float32)


@dataclass
class Episode:
    task: str
    session: LingBotPersistentRolloutSession
    wire: C57ExecutedFeedbackWire
    text_context: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    proprio_at_decision: torch.Tensor | None = None
    transaction_id: int = 0


class C57Policy(BASE.H3DreamWAMKVInt8Policy):
    """Reuse D0's audited H3/VAE bridge, replacing only its action lifecycle."""

    def __init__(self, args: argparse.Namespace) -> None:
        c57_path = args.checkpoint.resolve()
        payload = torch.load(c57_path, map_location="cpu", weights_only=False)
        contract = payload.get("contract", {})
        if payload.get("schema_version") != 1 or contract.get("candidate") != "C57":
            raise ValueError("C57 server requires a schema-1 Candidate C57 checkpoint")
        if contract.get("method") != "lingbot_persistent_observation_action_kv":
            raise ValueError("checkpoint is not the persistent LingBot lifecycle port")
        if int(contract.get("replan", -1)) != 8 or int(
            contract.get("observe_every", -1)
        ) != 4:
            raise ValueError("C57 checkpoint must declare replan8/observe4")
        parent_args = copy.copy(args)
        parent_args.checkpoint = Path(contract["source_checkpoint"])
        parent_args.sample_ensemble_size = 1
        parent_args.consequence_best_of_n = 1
        parent_args.consequence_ranker_checkpoint = None
        parent_args.consequence_model_checkpoint = []
        parent_args.dense_value_checkpoint = None
        parent_args.dense_value_final_report = None
        parent_args.progress_probe = None
        super().__init__(parent_args)
        spec = torch.load(
            parent_args.checkpoint, map_location="cpu", weights_only=False
        )["contract"]["model_spec"]
        self.action_model = H3LingBotPersistentKVPolicy(
            enabled=True,
            persistent_enabled=True,
            persistent_window_chunks=15,
            observation_tokens_per_chunk=32,
            action_tokens_per_chunk=4,
            carrier_layers=self.carrier_layers,
            carrier_source_mode=self.carrier_source_mode,
            action_dim=int(spec["action_dim"]),
            proprio_dim=int(spec["proprio_dim"]),
            context_dim=int(spec["context_dim"]),
            hidden_dim=int(spec["hidden_dim"]),
            ffn_dim=int(spec["ffn_dim"]),
            num_heads=int(spec["num_heads"]),
            attn_head_dim=int(spec["attn_head_dim"]),
            freq_dim=int(spec["freq_dim"]),
        ).to(device=self.device, dtype=self.dtype)
        self.action_model.load_state_dict(payload["model"], strict=True)
        self.action_model.eval()
        self.completed_steps = int(payload["completed_steps"])
        self.c57_checkpoint = c57_path
        self._layout = _official_layout_functions()
        self.episodes: dict[str, Episode] = {}

    def reset(self, request: dict) -> dict:
        key = str(request["episode_key"])
        task = str(request["task"])
        old = self.episodes.pop(key, None)
        if old is not None:
            old.wire.discard_on_reset()
        self.episodes[key] = Episode(
            task=task,
            session=LingBotPersistentRolloutSession(self.action_model, episode_key=key),
            wire=C57ExecutedFeedbackWire(replan=8, observe_every=4),
        )
        return {"ok": True, "event": "reset", "episode_key": key}

    def _episode(self, request: dict) -> Episode:
        key = str(request["episode_key"])
        if key not in self.episodes:
            raise RuntimeError("C57 episode must be explicitly reset before use")
        episode = self.episodes[key]
        if str(request.get("task", episode.task)) != episode.task:
            raise RuntimeError("task changed inside a persistent episode")
        return episode

    def _encode_observation(self, request: dict, *, frame_position: int):
        context = self._task_context(str(request["task"]))
        pixels = preprocess_libero_cameras(
            request["agentview_image"], request["wristview_image"]
        )
        video = (
            pixels.mul(255.0).round().to(torch.uint8).permute(0, 3, 1, 2)
            .unsqueeze(2).to(self.device)
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            latents = self._encode_vae_condition(
                self.video_vae,
                video,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ).to(device=self.device, dtype=torch.float32)
        self.int8_kv_provider.layout_functions = offset_h3_layout_functions(
            self._layout, frame_st_id=frame_position
        )
        live = self.int8_kv_provider(
            latents, context["context"], context["token_tags"]
        )
        layer49 = live[49]
        cache = {
            layer: {"k": layer49["k"].clone(), "v": layer49["v"].clone()}
            for layer in self.carrier_layers
        }
        state = minmax_normalize(
            libero_observation_state(request),
            self.stats["state_min"],
            self.stats["state_max"],
        ).clamp(-5.0, 5.0).reshape(1, 8).to(self.device, self.dtype)
        mask = torch.ones(
            context["context"].shape[:2], device=self.device, dtype=torch.bool
        )
        return cache, state, context, mask

    @torch.inference_mode()
    def predict(self, request: dict):
        episode = self._episode(request)
        if episode.wire.pending_actions:
            raise RuntimeError("C57 feedback transaction is incomplete")
        frame_position = max(episode.session.next_frame_st_id - 1, 0)
        cache, state, context, mask = self._encode_observation(
            request, frame_position=frame_position
        )
        generator = torch.Generator(device=self.device).manual_seed(int(request["seed"]))
        actions = torch.randn(
            (1, self.horizon, 7), device=self.device, dtype=self.dtype,
            generator=generator,
        )
        timesteps, deltas = self.flow_scheduler.build_inference_schedule(
            self.inference_steps, self.device, self.dtype
        )
        started = time.perf_counter()
        for index, (timestep, delta) in enumerate(zip(timesteps, deltas, strict=True)):
            velocity = episode.session.predict_velocity(
                actions,
                timestep.float().expand(1),
                text_context=context["context"],
                proprio=state,
                current_observation_kv=cache,
                text_mask=mask,
                final_denoise_step=index + 1 == len(timesteps),
            )
            actions = self.flow_scheduler.step(velocity, delta, actions)
        episode.text_context = context["context"]
        episode.text_mask = mask
        episode.proprio_at_decision = state
        environment, report = libero_environment_actions(
            actions[0], self.stats["action_min"], self.stats["action_max"],
            binarize_gripper=self.binarize_gripper,
            temporal_median_window=self.action_median_window,
            normalized_action_pre_clamp=self.normalized_action_pre_clamp,
            return_decode_report=True,
        )
        environment[:, :6] = np.clip(environment[:, :6] * self.action_scale, -1, 1)
        return environment, {
            "context_id": context["id"],
            "first_environment_action": environment[0].tolist(),
            "environment_action_chunk": environment.tolist(),
            "candidate": "C57",
            "persistent_lifecycle": "reset_predict_obs4_commit8",
            "checkpoint_completed_steps": self.completed_steps,
            "frame_st_id": episode.session.next_frame_st_id,
            "action_st_id": episode.session.next_action_st_id,
            "inference_seconds": time.perf_counter() - started,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": report,
        }

    @torch.inference_mode()
    def feedback(self, request: dict) -> dict:
        episode = self._episode(request)
        if observation_digest(request) != str(request["observation_sha256"]):
            raise RuntimeError("C57 post-action observation wire hash mismatch")
        total = int(request["action_count"])
        if int(request["transaction_id"]) != episode.transaction_id:
            raise RuntimeError("C57 feedback transaction ID mismatch")
        expected_total = episode.wire.pending_actions + 4
        if total != expected_total or total not in (4, 8):
            raise RuntimeError("C57 feedback must arrive as cumulative action4/action8")
        raw = np.asarray(request["executed_environment_actions"], dtype=np.float32)
        if raw.shape != (4, 7):
            raise ValueError("each feedback message must carry exactly four actions")
        normalized = normalize_libero_environment_action_history(
            raw, np.ones(4, dtype=bool), self.stats["action_min"],
            self.stats["action_max"], clip=5.0,
        ).to(device=self.device, dtype=self.dtype)
        frame_position = episode.session.next_frame_st_id
        if episode.session.next_frame_st_id == 0:
            frame_position += 1 if total == 4 else 2
        elif total == 8:
            frame_position += 1
        cache, _, _, _ = self._encode_observation(
            request, frame_position=frame_position
        )
        for index, action in enumerate(normalized):
            episode.wire.record_executed_action(
                action,
                observation_after_action=cache if index == 3 else None,
            )
        committed = total == 8
        if committed:
            if any(
                value is None
                for value in (
                    episode.text_context,
                    episode.text_mask,
                    episode.proprio_at_decision,
                )
            ):
                raise RuntimeError("C57 prediction context was not staged")
            episode.wire.commit(
                episode.session,
                text_context=episode.text_context,
                text_mask=episode.text_mask,
                proprio_at_decision=episode.proprio_at_decision,
            )
            episode.transaction_id += 1
        return {
            "ok": True,
            "event": "commit" if committed else "observation",
            "action_count": total,
            "committed": committed,
            "frame_st_id": episode.session.next_frame_st_id,
            "action_st_id": episode.session.next_action_st_id,
        }


def main() -> None:
    args = BASE.parse_args()
    if args.policy != "h3_dreamwam_kv_int8":
        raise ValueError("C57 server uses the audited INT8 DreamWAM bridge")
    if args.model_evaluations != 10 or args.action_horizon != 32:
        raise ValueError("C57 requires 10 Euler steps and horizon32")
    args.ready_file = args.ready_file.resolve()
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.unlink(missing_ok=True)
    policy = C57Policy(args)
    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    args.ready_file.write_text(json.dumps({"ready": True, "candidate": "C57"}))
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    command = request.get("command")
                    if command == "close":
                        connection.send({"ok": True})
                        return
                    try:
                        if command == "c57_reset":
                            response = policy.reset(request)
                        elif command in ("predict", "c57_feedback"):
                            _decode_observation(request)
                            if command == "predict":
                                actions, metadata = policy.predict(request)
                                response = {
                                    "ok": True, "actions": actions.tolist(),
                                    "metadata": metadata,
                                }
                            else:
                                response = policy.feedback(request)
                        else:
                            raise ValueError(f"unknown C57 command: {command}")
                        connection.send(response)
                        trace = {
                            "event": "c57_persistent_trace",
                            "command": command,
                            "episode_key": request.get("episode_key"),
                            "transaction_id": request.get("transaction_id"),
                            "action_count": request.get("action_count"),
                            "committed": response.get("committed"),
                            "frame_st_id": response.get("frame_st_id"),
                            "action_st_id": response.get("action_st_id"),
                        }
                        if command == "predict":
                            trace.update(
                                {
                                    "frame_st_id": response["metadata"].get("frame_st_id"),
                                    "action_st_id": response["metadata"].get("action_st_id"),
                                    "lifecycle": response["metadata"].get(
                                        "persistent_lifecycle"
                                    ),
                                }
                            )
                        print(json.dumps(trace, sort_keys=True), flush=True)
                    except Exception:
                        connection.send({"ok": False, "error": traceback.format_exc()})
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        args.ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
