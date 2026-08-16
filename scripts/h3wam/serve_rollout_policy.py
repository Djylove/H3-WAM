#!/usr/bin/env python3
"""Serve an H3-WAM or small-baseline policy to a LIBERO simulator process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
VENDORED_STARWAM_ROOT = REPO_ROOT / "third_party" / "StarWAM"
if VENDORED_STARWAM_ROOT.is_dir():
    sys.path.insert(0, str(VENDORED_STARWAM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=(
            "h3",
            "h3_feature",
            "h3_feature_int8",
            "h3_starwam_int8",
            "h3_dreamwam_kv_int8",
            "h3_fastwam_online_int8",
            "baseline",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--h3-feature-ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Additional single-mode regression action heads to average.",
    )
    parser.add_argument(
        "--h3-feature-ensemble-mode",
        choices=("mean", "switch", "disagreement_switch", "learned_switch"),
        default="mean",
    )
    parser.add_argument("--h3-feature-switch-step", type=int)
    parser.add_argument("--h3-feature-disagreement-threshold", type=float)
    parser.add_argument(
        "--h3-feature-switch-consecutive",
        type=int,
        default=1,
        help="Consecutive threshold crossings required before latching recovery head.",
    )
    parser.add_argument("--h3-feature-switch-gate-checkpoint", type=Path)
    parser.add_argument("--h3-feature-gate-threshold", type=float)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", default="h3wam-local-rollout")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--model-evaluations", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--h3-checkpoint", type=Path)
    parser.add_argument(
        "--h3-model",
        type=Path,
        help="Official MiniMax-H3 model root; INT8 live rollout loads its VAE.",
    )
    parser.add_argument(
        "--starwam-source-manifest",
        type=Path,
        help="Frozen full cache manifest used to resolve R1 task text contexts.",
    )
    parser.add_argument(
        "--dreamwam-source-manifest",
        type=Path,
        help="Frozen full cache manifest used to resolve Candidate D0 text contexts.",
    )
    parser.add_argument(
        "--c58b-balanced80-ready",
        type=Path,
        help="Required offline gate artifact for h3_fastwam_online_int8.",
    )
    parser.add_argument(
        "--progress-probe",
        type=Path,
        help="Validated frozen H3 progress ridge for shadow diagnostics only.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-latent-frames", type=int, default=12)
    parser.add_argument(
        "--h3-feature-audio-horizon",
        type=int,
        help="Packed H3 audio length when it differs from the action-head horizon.",
    )
    parser.add_argument(
        "--h3-tail-delta",
        type=Path,
        help="Feature-only BF16 tail delta applied over the quantized Comfy H3 base.",
    )
    parser.add_argument(
        "--h3-video-lora-checkpoint",
        type=Path,
        help="Optional video-objective H3 LoRA used with an H3 feature action head.",
    )
    parser.add_argument("--video-vae", type=Path)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-last-blocks", type=int, default=50)
    parser.add_argument(
        "--context-mode",
        choices=("cached", "online_episode", "online_replan"),
        default="cached",
    )
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument(
        "--binarize-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--action-median-window", type=int, default=1)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale the six LIBERO motion dimensions after denormalization.",
    )
    parser.add_argument(
        "--normalized-action-pre-clamp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clamp normalized action predictions to [-1,1] before "
            "denormalization. Disabled by default for historical compatibility."
        ),
    )
    parser.add_argument("--sample-ensemble-size", type=int, default=1)
    parser.add_argument(
        "--consequence-ranker-checkpoint",
        type=Path,
        help="Frozen PASS C44 ranker used only for best-of-N selection.",
    )
    parser.add_argument(
        "--consequence-model-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="One frozen C38 temporal consequence member; pass exactly four.",
    )
    parser.add_argument(
        "--dense-value-checkpoint",
        type=Path,
        help="Frozen C51 dense value expert used for best-of-N selection.",
    )
    parser.add_argument(
        "--dense-value-final-report",
        type=Path,
        help="PASS C51 report that pins the dense value checkpoint.",
    )
    parser.add_argument(
        "--consequence-best-of-n",
        type=int,
        default=1,
        help="Generate N independent action chunks and select by the C44 ranker.",
    )
    parser.add_argument(
        "--consequence-candidate-seed-offset",
        type=int,
        action="append",
        default=[],
        help="Explicit frozen seed offset per best-of-N candidate.",
    )
    parser.add_argument(
        "--consequence-selection-min-step",
        type=int,
        default=0,
        help="Rank only requests at or after this environment step.",
    )
    parser.add_argument(
        "--consequence-selection-max-step",
        type=int,
        help="Rank only requests with environment step below this exclusive bound.",
    )
    parser.add_argument(
        "--h3-feature-ablation",
        choices=("none", "zero"),
        default="none",
        help="Diagnostic ablation for the H3 feature-action policy.",
    )
    parser.add_argument(
        "--h3-action-mode",
        type=int,
        help=(
            "Force one mixture action mode. Without this flag, use the checkpoint's "
            "recommended_action_mode when present, otherwise use the learned gate."
        ),
    )
    parser.add_argument(
        "--lock-h3-action-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Choose a mixture mode on the first replan and keep it for the episode.",
    )
    return parser.parse_args()


class CachedContextMixin:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root.resolve()
        self._contexts: dict[str, dict] = {}

    def task_context(self, task: str, device: torch.device) -> dict:
        from fastwam.models.h3wam import load_cached_task_context

        if task not in self._contexts:
            self._contexts[task] = load_cached_task_context(
                self.cache_root,
                task,
                context_id=getattr(self, "context_id_override", None),
            )
        cached = self._contexts[task]
        return {
            "id": cached["id"],
            "context": cached["context"].to(device=device, dtype=torch.bfloat16),
            "token_tags": cached["token_tags"],
        }


class BaselinePolicy(CachedContextMixin):
    def __init__(self, args: argparse.Namespace) -> None:
        from fastwam.models.h3wam import H3ActionFlowScheduler, SmallActionFlowTransformer

        super().__init__(args.cache_root)
        if args.context_mode != "cached":
            raise ValueError("the small baseline currently supports only cached context")
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        self.stats = checkpoint["normalization"]
        self.device = torch.device("cuda")
        self.state_dim = int(checkpoint.get("state_dim", 8))
        self.include_phase = bool(checkpoint.get("include_phase", False))
        self.phase_only = bool(checkpoint.get("phase_only", False))
        self.phase_sequence = bool(checkpoint.get("phase_sequence", False))
        self.phase_length = int(checkpoint.get("phase_length", 1))
        self.model = SmallActionFlowTransformer(action_dim=7, state_dim=self.state_dim).to(
            self.device
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.scheduler = H3ActionFlowScheduler()
        self.evaluations = args.model_evaluations
        self.horizon = args.action_horizon
        self.binarize_gripper = args.binarize_gripper
        self.action_median_window = args.action_median_window
        self.action_scale = float(args.action_scale)
        self.normalized_action_pre_clamp = bool(args.normalized_action_pre_clamp)
        if self.action_scale <= 0:
            raise ValueError("action-scale must be positive")
        self.sample_ensemble_size = args.sample_ensemble_size
        self.objective = checkpoint.get("objective", "flow")

    @torch.inference_mode()
    def predict(self, request: dict) -> tuple[torch.Tensor, dict]:
        from fastwam.models.h3wam import (
            libero_environment_actions,
            libero_observation_state,
            minmax_normalize,
        )

        task_context = self.task_context(request["task"], self.device)
        state = minmax_normalize(
            libero_observation_state(request),
            self.stats["state_min"],
            self.stats["state_max"],
        )
        if self.phase_only:
            state = torch.zeros_like(state)
        if self.include_phase:
            phase = 2.0 * float(request.get("step", 0)) / max(self.phase_length - 1, 1) - 1.0
            state = torch.cat(
                (state, torch.tensor([min(phase, 1.0)], dtype=torch.float32))
            )
        state = state.unsqueeze(0).to(self.device)
        started = time.perf_counter()
        if self.objective == "regression":
            actions = self.model(
                torch.zeros((1, self.horizon, 7), device=self.device),
                state=state,
                context=task_context["context"],
                video_sigma=torch.zeros(1, device=self.device),
            )
        else:
            generator = torch.Generator(device=self.device).manual_seed(int(request["seed"]))
            actions = torch.randn(
                (1, self.horizon, 7),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            sigmas, deltas = self.scheduler.inference_schedule(
                self.evaluations, device=self.device
            )
            for sigma, delta in zip(sigmas, deltas):
                prediction = self.model(
                    actions,
                    state=state,
                    context=task_context["context"],
                    video_sigma=sigma.reshape(1),
                )
                action_delta = self.scheduler.action_inference_delta(sigma, delta)
                actions = actions + prediction / self.scheduler.action_slope(
                    sigma
                ) * action_delta
        torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - started
        environment_actions, decode_report = libero_environment_actions(
            actions[0],
            self.stats["action_min"],
            self.stats["action_max"],
            binarize_gripper=self.binarize_gripper,
            temporal_median_window=self.action_median_window,
            normalized_action_pre_clamp=self.normalized_action_pre_clamp,
            return_decode_report=True,
        )
        environment_actions[:, :6] = np.clip(
            environment_actions[:, :6] * self.action_scale, -1.0, 1.0
        )
        return environment_actions, {
            "context_id": task_context["id"],
            "first_environment_action": environment_actions[0].tolist(),
            "environment_action_chunk": environment_actions.tolist(),
            "inference_seconds": inference_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": decode_report,
        }


class H3Policy(CachedContextMixin):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.comfy_root is None or args.h3_checkpoint is None or args.video_vae is None:
            raise ValueError("H3 policy requires --comfy-root, --h3-checkpoint and --video-vae")
        sys.path.insert(0, str(args.comfy_root.resolve()))
        import comfy.model_management as model_management
        import comfy.sd
        import comfy.utils

        from fastwam.models.h3wam import (
            H3ActionAdapter,
            H3ActionBridge,
            H3ActionFlowScheduler,
            inject_h3_attention_lora,
            load_h3_lora_state_dict,
        )

        super().__init__(args.cache_root)
        self.model_management = model_management
        model_management.in_training = False
        self.device = model_management.get_torch_device()
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        self.stats = checkpoint["normalization"]
        self.objective = checkpoint.get("objective", "flow")
        self.phase_only = bool(checkpoint.get("phase_only", False))
        self.phase_length = int(checkpoint.get("phase_length", 1))
        self.fixed_context = bool(checkpoint.get("fixed_context", False))
        self.phase_sequence = bool(checkpoint.get("phase_sequence", False))
        self.fixed_first_frame = bool(checkpoint.get("fixed_first_frame", False))
        if self.fixed_context and args.context_mode != "cached":
            raise ValueError("a fixed-context H3 checkpoint requires cached context mode")
        self._fixed_visuals: dict[str, dict] = {}

        self.h3_patcher = comfy.sd.load_diffusion_model(
            str(args.h3_checkpoint.resolve())
        )
        model_management.load_models_gpu([self.h3_patcher])
        h3_model = self.h3_patcher.model.diffusion_model
        self.h3_model = h3_model
        adapter = H3ActionAdapter(
            action_dim=int(checkpoint["action_dim"]),
            state_dim=int(checkpoint["state_dim"]),
            direct_conditioning=bool(
                checkpoint.get("direct_action_conditioning", False)
            ),
            direct_action_residual=bool(
                checkpoint.get("direct_action_residual", False)
            ),
        ).to(device=self.device, dtype=torch.float32)
        self.bridge = H3ActionBridge(h3_model, adapter, freeze_h3=True)
        if args.lora_rank > 0:
            inject_h3_attention_lora(
                h3_model,
                rank=args.lora_rank,
                last_n_blocks=args.lora_last_blocks,
            )
            load_h3_lora_state_dict(h3_model, checkpoint["h3_lora"])
        adapter.load_state_dict(checkpoint["adapter"])
        self.bridge.eval()
        self.scheduler = H3ActionFlowScheduler(
            video_shift=float(h3_model.sigma_shift_video),
            action_shift=float(h3_model.sigma_shift_audio),
        )

        vae_state = comfy.utils.load_torch_file(str(args.video_vae.resolve()))
        self.video_vae = comfy.sd.VAE(sd=vae_state)
        del vae_state
        self.evaluations = args.model_evaluations
        self.horizon = args.action_horizon
        self.binarize_gripper = args.binarize_gripper
        self.action_median_window = args.action_median_window
        self.normalized_action_pre_clamp = bool(args.normalized_action_pre_clamp)
        self.sample_ensemble_size = args.sample_ensemble_size
        self.context_mode = args.context_mode
        self._episode_contexts: dict[str, dict] = {}
        self.clip = None
        if self.context_mode in ("online_episode", "online_replan"):
            if args.text_encoder is None:
                raise ValueError("online_episode context requires --text-encoder")
            self.clip = comfy.sd.load_clip(
                ckpt_paths=[str(args.text_encoder.resolve())],
                clip_type=comfy.sd.CLIPType.MINIMAX,
            )

    @torch.no_grad()
    def online_episode_context(
        self,
        request: dict,
        pixels: torch.Tensor,
    ) -> tuple[dict, float]:
        episode_key = str(request["episode_key"])
        if self.context_mode == "online_episode" and episode_key in self._episode_contexts:
            cached = self._episode_contexts[episode_key]
            return {
                "id": cached["id"],
                "context": cached["context"].to(
                    device=self.device, dtype=torch.bfloat16
                ),
                "token_tags": cached["token_tags"],
            }, 0.0
        started = time.perf_counter()
        tokens = self.clip.tokenize(request["task"], images=[pixels])
        conditioning = self.clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
        raw_context, metadata = conditioning[0]
        token_tags = metadata.get("minimax_token_tags")
        if token_tags is None:
            raise RuntimeError("MiniMax online context did not return token tags")
        self.model_management.load_models_gpu([self.h3_patcher])
        refined = self.h3_model.preprocess_text_embeds(
            raw_context.to(device=self.device, dtype=torch.bfloat16)
        )
        torch.cuda.synchronize(self.device)
        result = {
            "id": f"online:{episode_key}",
            "context": refined.detach().cpu(),
            "token_tags": token_tags.detach().cpu(),
        }
        if self.context_mode == "online_episode":
            self._episode_contexts[episode_key] = result
        return {
            "id": result["id"],
            "context": result["context"].to(
                device=self.device, dtype=torch.bfloat16
            ),
            "token_tags": result["token_tags"],
        }, time.perf_counter() - started

    @torch.no_grad()
    def predict(self, request: dict) -> tuple[torch.Tensor, dict]:
        from fastwam.models.h3wam import (
            libero_environment_actions,
            libero_observation_state,
            make_first_frame_payload,
            minmax_normalize,
            preprocess_libero_cameras,
            sample_h3wam_actions,
        )

        if self.phase_only:
            if self.phase_sequence:
                state = torch.zeros(
                    self.horizon,
                    libero_observation_state(request).numel(),
                    dtype=torch.float32,
                )
                phase_steps = torch.arange(self.horizon, dtype=torch.float32)
                phase_steps.add_(int(request.get("step", 0))).clamp_max_(
                    self.phase_length - 1
                )
                state[:, -1] = (
                    2.0 * phase_steps / max(self.phase_length - 1, 1) - 1.0
                )
            else:
                state = torch.zeros_like(libero_observation_state(request))
                phase = 2.0 * float(request.get("step", 0)) / max(
                    self.phase_length - 1, 1
                ) - 1.0
                state[-1] = min(phase, 1.0)
        else:
            state = minmax_normalize(
                libero_observation_state(request),
                self.stats["state_min"],
                self.stats["state_max"],
            )
        state = state.unsqueeze(0).to(self.device)
        pixels = None
        if not self.fixed_first_frame or self.context_mode in (
            "online_episode",
            "online_replan",
        ):
            pixels = preprocess_libero_cameras(
                request["agentview_image"], request["wristview_image"]
            )
        if self.context_mode in ("online_episode", "online_replan"):
            assert pixels is not None
            task_context, context_encode_seconds = self.online_episode_context(
                request, pixels
            )
        else:
            task_context = self.task_context(request["task"], self.device)
            context_encode_seconds = 0.0
        encode_started = time.perf_counter()
        fixed_visual = None
        if self.fixed_first_frame:
            context_id = task_context["id"]
            if context_id not in self._fixed_visuals:
                self._fixed_visuals[context_id] = torch.load(
                    self.cache_root / "windows" / f"{context_id}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            fixed_visual = self._fixed_visuals[context_id]
            first_frame_latents = fixed_visual["first_frame_latents"].to(
                self.device, dtype=torch.bfloat16
            )
        else:
            assert pixels is not None
            first_frame_latents = self.video_vae.encode(pixels).to(
                self.device, dtype=torch.bfloat16
            )
        # The VAE may evict H3 when VRAM is tight; explicitly make H3 resident
        # again before sampling.
        self.model_management.load_models_gpu([self.h3_patcher])
        torch.cuda.synchronize(self.device)
        encode_seconds = time.perf_counter() - encode_started

        frame_count = 39 if fixed_visual is None else int(fixed_visual["h3_frame_count"])
        payload = make_first_frame_payload(first_frame_latents, frame_count=frame_count)
        payload["text_token_tags"] = task_context["token_tags"]
        sample_started = time.perf_counter()
        if self.objective == "regression":
            if fixed_visual is None:
                video_shape = (
                    1,
                    24,
                    12,
                    int(first_frame_latents.shape[-2]),
                    int(first_frame_latents.shape[-1]),
                )
            else:
                video_shape = tuple(fixed_visual["video_latents"].shape)
            output = self.bridge(
                video_latents=torch.zeros(
                    video_shape, device=self.device, dtype=torch.bfloat16
                ),
                noisy_actions=torch.zeros(
                    (1, self.horizon, 7), device=self.device, dtype=torch.float32
                ),
                timestep=torch.zeros(1, device=self.device),
                context=task_context["context"],
                state=state,
                minimax_payload=payload,
            )
            normalized_actions = output.action_velocity
        else:
            action_samples = []
            for sample_index in range(self.sample_ensemble_size):
                generator = torch.Generator(device=self.device).manual_seed(
                    int(request["seed"]) + sample_index
                )
                sample = sample_h3wam_actions(
                    self.bridge,
                    context=task_context["context"],
                    state=state,
                    scheduler=self.scheduler,
                    action_shape=(1, self.horizon, 7),
                    video_shape=(
                        1,
                        24,
                        12,
                        int(first_frame_latents.shape[-2]),
                        int(first_frame_latents.shape[-1]),
                    ),
                    model_evaluations=self.evaluations,
                    minimax_payload=payload,
                    generator=generator,
                )
                action_samples.append(sample.actions)
            normalized_actions = torch.stack(action_samples).mean(dim=0)
        torch.cuda.synchronize(self.device)
        sample_seconds = time.perf_counter() - sample_started
        environment_actions, decode_report = libero_environment_actions(
            normalized_actions[0],
            self.stats["action_min"],
            self.stats["action_max"],
            binarize_gripper=self.binarize_gripper,
            temporal_median_window=self.action_median_window,
            normalized_action_pre_clamp=self.normalized_action_pre_clamp,
            return_decode_report=True,
        )
        return environment_actions, {
            "context_id": task_context["id"],
            "first_environment_action": environment_actions[0].tolist(),
            "environment_action_chunk": environment_actions.tolist(),
            "context_encode_seconds": context_encode_seconds,
            "vae_encode_seconds": encode_seconds,
            "inference_seconds": sample_seconds,
            "sample_ensemble_size": self.sample_ensemble_size,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": decode_report,
        }


class H3FeaturePolicy(CachedContextMixin):
    """Frozen H3 video expert plus an independent cross-attention action expert."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.feature_runtime = (
            "int8" if args.policy == "h3_feature_int8" else "comfy"
        )
        if args.h3_checkpoint is None:
            raise ValueError("H3 feature policy requires --h3-checkpoint")
        if self.feature_runtime == "comfy" and (
            args.comfy_root is None or args.video_vae is None
        ):
            raise ValueError(
                "H3 feature policy requires --comfy-root, --h3-checkpoint and --video-vae"
            )
        if self.feature_runtime == "int8" and args.h3_model is None:
            raise ValueError("standalone INT8 H3 feature policy requires --h3-model")
        if args.context_mode != "cached":
            raise ValueError("H3 feature policy currently requires cached context mode")
        if self.feature_runtime == "comfy":
            sys.path.insert(0, str(args.comfy_root.resolve()))
            import comfy.model_management as model_management
            import comfy.sd
            import comfy.utils

        from fastwam.models.h3wam import (
            H3FeatureActionTransformer,
            inject_h3_attention_lora,
            load_h3_comfy_feature_delta,
            load_h3_lora_state_dict,
        )
        from fastwam.models.wan22.schedulers.scheduler_continuous import (
            WanContinuousFlowMatchScheduler,
        )

        super().__init__(args.cache_root)
        self.model_management = None
        if self.feature_runtime == "comfy":
            self.model_management = model_management
            model_management.in_training = False
            self.device = model_management.get_torch_device()
        else:
            self.device = torch.device(args.device)
            if self.device.type != "cuda":
                raise ValueError("standalone INT8 online rollout requires a CUDA device")
            # comfy_kitchen's DLPack CUDA backend uses the process current
            # device as well as each tensor's device.  Keep both contracts in
            # sync when parallel canaries bind policy servers to cuda:1+.
            torch.cuda.set_device(self.device)
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        if checkpoint.get("policy_type") != "h3_feature_action":
            raise ValueError("checkpoint is not an H3 feature-action policy")
        self.training_tasks = set(checkpoint.get("training_tasks", ()))
        self.stats = checkpoint["normalization"]
        self.phase_length = int(checkpoint["phase_length"])
        self.use_proprio = bool(checkpoint.get("use_proprio", False))
        self.use_previous_action = bool(
            checkpoint.get("use_previous_action", False)
        )
        self.include_phase = bool(checkpoint.get("include_phase", True))
        self.state_dim = int(checkpoint["state_dim"])
        self.objective = str(checkpoint.get("objective", "regression"))
        self.feature_timestep = float(checkpoint.get("feature_timestep", 1000.0))
        self.context_id_override = checkpoint.get("feature_context_id")
        self.flow_scheduler = WanContinuousFlowMatchScheduler(
            shift=float(checkpoint.get("flow_shift", 5.0))
        )
        self.flow_inference_steps = int(
            args.model_evaluations
            if self.objective == "flow"
            else checkpoint.get("flow_inference_steps", args.model_evaluations)
        )
        self.horizon = args.action_horizon
        trained_horizon = int(checkpoint.get("action_horizon", 32))
        if self.horizon > trained_horizon:
            raise ValueError(
                f"requested action horizon {self.horizon} exceeds checkpoint "
                f"training horizon {trained_horizon}"
            )
        feature_shape = tuple(checkpoint["feature_shape"])
        if self.horizon > 64:
            raise ValueError("H3 feature action horizon cannot exceed 64")
        def build_action_model(member_checkpoint: dict):
            model = H3FeatureActionTransformer(
                action_dim=int(member_checkpoint["action_dim"]),
                state_dim=int(member_checkpoint["state_dim"]),
                h3_feature_dim=tuple(member_checkpoint["feature_shape"])[-1],
                hidden_dim=int(member_checkpoint["hidden_dim"]),
                num_layers=int(member_checkpoint["num_layers"]),
                num_heads=int(member_checkpoint["num_heads"]),
                ffn_dim=int(member_checkpoint["ffn_dim"]),
                num_action_modes=int(member_checkpoint.get("num_action_modes", 1)),
            ).to(self.device)
            model.load_state_dict(member_checkpoint["model"])
            return model.eval()

        self.action_model = build_action_model(checkpoint)
        self.feature_layers = tuple(int(index) for index in checkpoint["feature_layers"])
        self.ensemble_members = []
        self.ensemble_mode = args.h3_feature_ensemble_mode
        self.ensemble_switch_step = args.h3_feature_switch_step
        self.ensemble_disagreement_threshold = (
            args.h3_feature_disagreement_threshold
        )
        self.ensemble_switch_consecutive = args.h3_feature_switch_consecutive
        self._episode_action_heads: dict[str, int] = {}
        self._episode_disagreement_streaks: dict[str, int] = {}
        self.switch_gate = None
        self.switch_gate_threshold = args.h3_feature_gate_threshold
        if args.h3_feature_ensemble_checkpoint:
            if self.objective != "regression" or int(
                checkpoint.get("num_action_modes", 1)
            ) != 1:
                raise ValueError(
                    "H3 feature ensembles require single-mode regression checkpoints"
                )
            for path in args.h3_feature_ensemble_checkpoint:
                member = torch.load(
                    path.resolve(), map_location="cpu", weights_only=False
                )
                if member.get("policy_type") != "h3_feature_action":
                    raise ValueError(f"ensemble checkpoint {path} has the wrong policy type")
                if str(member.get("objective", "regression")) != "regression" or int(
                    member.get("num_action_modes", 1)
                ) != 1:
                    raise ValueError(
                        "H3 feature ensembles require single-mode regression checkpoints"
                    )
                if tuple(member["feature_shape"]) != feature_shape or tuple(
                    int(index) for index in member["feature_layers"]
                ) != self.feature_layers:
                    raise ValueError("ensemble checkpoint H3 feature contract differs")
                if self.horizon > int(member.get("action_horizon", 32)):
                    raise ValueError("ensemble checkpoint was trained for a shorter horizon")
                member_tasks = set(member.get("training_tasks", ()))
                if self.training_tasks and member_tasks and member_tasks != self.training_tasks:
                    raise ValueError("ensemble checkpoints have different training tasks")
                self.training_tasks.update(member_tasks)
                self.ensemble_members.append(
                    {
                        "model": build_action_model(member),
                        "use_proprio": bool(member.get("use_proprio", False)),
                        "use_previous_action": bool(
                            member.get("use_previous_action", False)
                        ),
                        "include_phase": bool(member.get("include_phase", True)),
                        "state_dim": int(member["state_dim"]),
                        "phase_length": int(member["phase_length"]),
                    }
                )
        if self.ensemble_mode in (
            "switch",
            "disagreement_switch",
            "learned_switch",
        ):
            if len(self.ensemble_members) != 1:
                raise ValueError(
                    f"{self.ensemble_mode} ensemble mode requires exactly one "
                    "extra checkpoint"
                )
        if self.ensemble_mode == "switch":
            if self.ensemble_switch_step is None or self.ensemble_switch_step <= 0:
                raise ValueError("switch ensemble mode requires a positive switch step")
        if self.ensemble_mode == "disagreement_switch":
            if (
                self.ensemble_disagreement_threshold is None
                or self.ensemble_disagreement_threshold <= 0
            ):
                raise ValueError(
                    "disagreement_switch requires a positive disagreement threshold"
                )
            if self.ensemble_switch_consecutive <= 0:
                raise ValueError("switch-consecutive must be positive")
        if self.ensemble_mode == "learned_switch":
            if args.h3_feature_switch_gate_checkpoint is None:
                raise ValueError(
                    "learned_switch requires a switch-gate checkpoint"
                )
            from fastwam.models.h3wam import H3FeatureSwitchGate

            gate_checkpoint = torch.load(
                args.h3_feature_switch_gate_checkpoint.resolve(),
                map_location="cpu",
                weights_only=False,
            )
            if gate_checkpoint.get("policy_type") != "h3_feature_switch_gate":
                raise ValueError("switch-gate checkpoint has the wrong policy type")
            if int(gate_checkpoint["h3_feature_dim"]) != feature_shape[-1]:
                raise ValueError("switch-gate H3 feature dimension differs")
            if tuple(gate_checkpoint["feature_layers"]) != self.feature_layers:
                raise ValueError("switch-gate H3 feature layers differ")
            self.switch_gate = H3FeatureSwitchGate(
                h3_feature_dim=int(gate_checkpoint["h3_feature_dim"]),
                state_dim=int(gate_checkpoint["state_dim"]),
                hidden_dim=int(gate_checkpoint["hidden_dim"]),
            ).to(self.device)
            self.switch_gate.load_state_dict(gate_checkpoint["model"])
            self.switch_gate.eval()
            if self.switch_gate_threshold is None:
                self.switch_gate_threshold = float(
                    gate_checkpoint.get("threshold", 0.5)
                )
            if not 0.0 < self.switch_gate_threshold < 1.0:
                raise ValueError("feature-gate threshold must be between zero and one")

        self.h3_patcher = None
        self.int8_feature_provider = None
        self._encode_vae_condition = None
        if self.feature_runtime == "comfy":
            self.h3_patcher = comfy.sd.load_diffusion_model(
                str(args.h3_checkpoint.resolve())
            )
            model_management.load_models_gpu([self.h3_patcher])
            self.h3_model = self.h3_patcher.model.diffusion_model
        else:
            from fastwam.models.h3wam import H3Int8FeatureBackbone

            self.h3_model = H3Int8FeatureBackbone.from_checkpoint(
                args.h3_checkpoint.resolve()
            ).to(self.device)
        self.h3_model.requires_grad_(False).eval()
        self.tail_delta_report = None
        if args.h3_tail_delta is not None:
            if self.feature_runtime != "comfy":
                raise ValueError("H3 tail delta is not supported by the INT8 runtime")
            if args.h3_video_lora_checkpoint is not None or checkpoint.get("h3_lora"):
                raise ValueError("use either an H3 tail delta or H3 LoRA, not both")
            self.tail_delta_report = load_h3_comfy_feature_delta(
                self.h3_model, args.h3_tail_delta
            )
            model_management.in_training = True
        external_video_lora = None
        if args.h3_video_lora_checkpoint is not None:
            external_video_lora = torch.load(
                args.h3_video_lora_checkpoint.resolve(),
                map_location="cpu",
                weights_only=False,
            )
            if external_video_lora.get("policy_type") != "h3_video_lora":
                raise ValueError("external H3 video LoRA checkpoint has the wrong policy type")
            if checkpoint.get("h3_lora"):
                raise ValueError(
                    "cannot combine an action-trained H3 LoRA and an external video LoRA"
                )
        lora_checkpoint = external_video_lora or checkpoint
        h3_lora = lora_checkpoint.get("h3_lora")
        if h3_lora:
            lora_rank = int(lora_checkpoint["h3_lora_rank"])
            lora_last_blocks = int(lora_checkpoint["h3_lora_last_blocks"])
            include_mlp_lora = bool(
                lora_checkpoint.get("h3_lora_include_mlp", False)
            )
            # ComfyUI's inference-only fused SwiGLU down-projection bypasses a
            # wrapped module. Its training dispatch calls the wrapper normally,
            # which is required for fc2 LoRA while still running under
            # torch.inference_mode here.
            if include_mlp_lora and self.feature_runtime == "comfy":
                model_management.in_training = True
            inject_h3_attention_lora(
                self.h3_model,
                rank=lora_rank,
                alpha=float(lora_checkpoint.get("h3_lora_alpha", lora_rank)),
                last_n_blocks=lora_last_blocks,
                include_mlp=include_mlp_lora,
            )
            load_h3_lora_state_dict(self.h3_model, h3_lora)
            self.h3_model.eval()
        if self.feature_runtime == "comfy":
            vae_state = comfy.utils.load_torch_file(str(args.video_vae.resolve()))
            self.video_vae = comfy.sd.VAE(sd=vae_state)
            del vae_state
        else:
            from diffusers import AutoencoderKLMiniMaxH3
            from fastwam.models.h3wam import (
                H3Int8OnlineFeatureContract,
                H3Int8OnlineFeatureProvider,
                encode_h3_vae_condition_standalone,
            )

            feature_audio_horizon = (
                self.horizon
                if args.h3_feature_audio_horizon is None
                else int(args.h3_feature_audio_horizon)
            )
            if feature_audio_horizon <= 0:
                raise ValueError("h3-feature-audio-horizon must be positive")
            self.int8_feature_provider = H3Int8OnlineFeatureProvider(
                self.h3_model,
                H3Int8OnlineFeatureContract(
                    layers=self.feature_layers,
                    action_horizon=feature_audio_horizon,
                    target_latent_frames=args.target_latent_frames,
                    video_timestep=0.0,
                    condition_video_timestep=0.999,
                    capture_compatibility="comfy_alias_v1",
                ),
            )
            self.video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
                args.h3_model.resolve(),
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
            self._encode_vae_condition = encode_h3_vae_condition_standalone
        self.binarize_gripper = args.binarize_gripper
        self.action_median_window = args.action_median_window
        self.action_scale = float(args.action_scale)
        self.normalized_action_pre_clamp = bool(args.normalized_action_pre_clamp)
        if self.action_scale <= 0:
            raise ValueError("action-scale must be positive")
        self.feature_ablation = args.h3_feature_ablation
        self.num_action_modes = int(checkpoint.get("num_action_modes", 1))
        recommended_action_mode = checkpoint.get("recommended_action_mode")
        self.forced_action_mode = (
            args.h3_action_mode
            if args.h3_action_mode is not None
            else recommended_action_mode
        )
        self.forced_action_mode_source = (
            "cli"
            if args.h3_action_mode is not None
            else "checkpoint"
            if recommended_action_mode is not None
            else None
        )
        self.lock_action_mode = args.lock_h3_action_mode
        self._episode_action_modes: dict[str, int] = {}
        if self.forced_action_mode is not None and not (
            0 <= self.forced_action_mode < self.num_action_modes
        ):
            raise ValueError("forced H3 action mode is outside checkpoint mode range")

    @torch.inference_mode()
    def predict(self, request: dict) -> tuple[torch.Tensor, dict]:
        from fastwam.models.h3wam import (
            H3BlockFeatureCapture,
            libero_environment_actions,
            libero_observation_state,
            make_first_frame_payload,
            minmax_normalize,
            preprocess_libero_cameras,
            H3MixtureActionOutput,
        )

        if self.training_tasks and request["task"] not in self.training_tasks:
            raise ValueError(
                f"rollout task {request['task']!r} is not among checkpoint "
                f"training tasks {sorted(self.training_tasks)!r}"
            )
        task_context = self.task_context(request["task"], self.device)
        pixels = preprocess_libero_cameras(
            request["agentview_image"], request["wristview_image"]
        )
        encode_started = time.perf_counter()
        if self.feature_runtime == "comfy":
            first_frame = self.video_vae.encode(pixels).to(
                device=self.device, dtype=torch.bfloat16
            )
            assert self.model_management is not None
            assert self.h3_patcher is not None
            self.model_management.load_models_gpu([self.h3_patcher])
        else:
            assert self._encode_vae_condition is not None
            video = (
                pixels.mul(255.0)
                .round()
                .to(torch.uint8)
                .permute(0, 3, 1, 2)
                .unsqueeze(2)
                .to(self.device)
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                first_frame = self._encode_vae_condition(
                    self.video_vae,
                    video,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ).to(device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)
        encode_seconds = time.perf_counter() - encode_started

        h3_started = time.perf_counter()
        if self.feature_runtime == "comfy":
            text_len = int(task_context["context"].shape[1])
            frame_rows = int(
                first_frame.shape[2]
                * (first_frame.shape[3] // 2)
                * (first_frame.shape[4] // 2)
            )
            capture = H3BlockFeatureCapture(
                self.feature_layers,
                token_start=text_len,
                token_stop=text_len + frame_rows,
            )
            payload = make_first_frame_payload(first_frame, frame_count=39)
            payload["text_token_tags"] = task_context["token_tags"].to(self.device)
            self.h3_model(
                [
                    torch.zeros(
                        (1, 24, 12, first_frame.shape[-2], first_frame.shape[-1]),
                        device=self.device,
                        dtype=torch.bfloat16,
                    ),
                    torch.zeros(
                        (1, 32, 2, self.horizon),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ],
                torch.tensor([self.feature_timestep], device=self.device),
                task_context["context"],
                transformer_options=capture.transformer_options(),
                minimax_payload=payload,
            )
            features = capture.stacked().unsqueeze(0)
        else:
            assert self.int8_feature_provider is not None
            features = self.int8_feature_provider(
                first_frame,
                task_context["context"],
                task_context["token_tags"].to(self.device),
            )
        if self.feature_ablation == "zero":
            features = torch.zeros_like(features)
        torch.cuda.synchronize(self.device)
        h3_seconds = time.perf_counter() - h3_started

        def policy_state(
            use_proprio: bool,
            use_previous_action: bool,
            include_phase: bool,
            state_dim: int,
            phase_length: int,
        ) -> torch.Tensor:
            parts = []
            if use_proprio:
                proprio = minmax_normalize(
                    libero_observation_state(request),
                    self.stats["state_min"],
                    self.stats["state_max"],
                ).to(self.device)
                parts.append(
                    proprio.reshape(1, 1, 8).expand(
                        1, self.horizon, 8
                    ).clone()
                )
            if use_previous_action:
                from fastwam.models.h3wam import libero_dataset_action

                previous = libero_dataset_action(
                    request["previous_environment_action"]
                )
                previous = minmax_normalize(
                    previous,
                    self.stats["action_min"],
                    self.stats["action_max"],
                ).to(self.device)
                parts.append(
                    previous.reshape(1, 1, 7).expand(
                        1, self.horizon, 7
                    ).clone()
                )
            if include_phase:
                phase_steps = torch.arange(
                    self.horizon, device=self.device, dtype=torch.float32
                )
                phase_steps.add_(int(request.get("step", 0))).clamp_max_(
                    phase_length - 1
                )
                phase = torch.zeros(
                    (1, self.horizon, 1),
                    device=self.device,
                    dtype=torch.float32,
                )
                phase[0, :, 0] = (
                    2.0 * phase_steps / max(phase_length - 1, 1) - 1.0
                )
                parts.append(phase)
            if not parts:
                parts.append(
                    torch.zeros(
                        (1, self.horizon, 1),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
            result = torch.cat(parts, dim=-1)
            if result.shape[-1] != state_dim:
                raise RuntimeError(
                    f"constructed state dim {result.shape[-1]} does not match {state_dim}"
                )
            return result

        state = policy_state(
            self.use_proprio,
            self.use_previous_action,
            self.include_phase,
            self.state_dim,
            self.phase_length,
        )
        action_started = time.perf_counter()
        ensemble_disagreement = None
        ensemble_gripper_disagreement = None
        selected_action_head = 0
        switch_gate_probability = None
        if self.objective == "flow":
            generator = torch.Generator(device=self.device).manual_seed(
                int(request["seed"])
            )
            action_output = torch.randn(
                (1, self.horizon, 7),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            timesteps, deltas = self.flow_scheduler.build_inference_schedule(
                self.flow_inference_steps,
                device=self.device,
                dtype=action_output.dtype,
            )
            for timestep, delta in zip(timesteps, deltas):
                velocity = self.action_model(
                    action_output,
                    state=state,
                    h3_features=features,
                    video_sigma=timestep.reshape(1),
                )
                if not isinstance(velocity, torch.Tensor):
                    raise RuntimeError("flow policy must have exactly one action mode")
                action_output = self.flow_scheduler.step(
                    velocity, delta, action_output
                )
        else:
            action_output = self.action_model(
                torch.zeros(
                    (1, self.horizon, 7), device=self.device, dtype=torch.float32
                ),
                state=state,
                h3_features=features,
                video_sigma=torch.zeros(1, device=self.device),
            )
            if self.ensemble_members:
                if not isinstance(action_output, torch.Tensor):
                    raise RuntimeError("ensemble base model returned action modes")
                ensemble_outputs = [action_output]
                for member in self.ensemble_members:
                    member_state = policy_state(
                        member["use_proprio"],
                        member["use_previous_action"],
                        member["include_phase"],
                        member["state_dim"],
                        member["phase_length"],
                    )
                    member_output = member["model"](
                        torch.zeros(
                            (1, self.horizon, 7),
                            device=self.device,
                            dtype=torch.float32,
                        ),
                        state=member_state,
                        h3_features=features,
                        video_sigma=torch.zeros(1, device=self.device),
                    )
                    if not isinstance(member_output, torch.Tensor):
                        raise RuntimeError("ensemble member returned action modes")
                    ensemble_outputs.append(member_output)
                if len(ensemble_outputs) == 2:
                    difference = ensemble_outputs[0] - ensemble_outputs[1]
                    ensemble_disagreement = float(
                        torch.linalg.vector_norm(
                            difference[..., :6], dim=-1
                        ).mean().item()
                    )
                    ensemble_gripper_disagreement = float(
                        difference[..., -1].abs().mean().item()
                    )
                if self.ensemble_mode == "mean":
                    action_output = torch.stack(ensemble_outputs).mean(dim=0)
                    selected_action_head = -1
                elif self.ensemble_mode == "switch":
                    assert self.ensemble_switch_step is not None
                    selected_action_head = int(
                        int(request.get("step", 0)) >= self.ensemble_switch_step
                    )
                    action_output = ensemble_outputs[selected_action_head]
                elif self.ensemble_mode == "disagreement_switch":
                    assert ensemble_disagreement is not None
                    assert self.ensemble_disagreement_threshold is not None
                    episode_key = str(request["episode_key"])
                    if self._episode_action_heads.get(episode_key, 0) == 1:
                        selected_action_head = 1
                    else:
                        streak = self._episode_disagreement_streaks.get(
                            episode_key, 0
                        )
                        if (
                            ensemble_disagreement
                            >= self.ensemble_disagreement_threshold
                        ):
                            streak += 1
                        else:
                            streak = 0
                        self._episode_disagreement_streaks[episode_key] = streak
                        selected_action_head = int(
                            streak >= self.ensemble_switch_consecutive
                        )
                        if selected_action_head:
                            self._episode_action_heads[episode_key] = 1
                    action_output = ensemble_outputs[selected_action_head]
                else:
                    assert self.switch_gate is not None
                    assert self.switch_gate_threshold is not None
                    episode_key = str(request["episode_key"])
                    gate_state = minmax_normalize(
                        libero_observation_state(request),
                        self.stats["state_min"],
                        self.stats["state_max"],
                    ).to(self.device).reshape(1, 8)
                    switch_gate_probability = float(
                        self.switch_gate(features, gate_state).sigmoid().item()
                    )
                    if self._episode_action_heads.get(episode_key, 0) == 1:
                        selected_action_head = 1
                    else:
                        selected_action_head = int(
                            switch_gate_probability >= self.switch_gate_threshold
                        )
                        if selected_action_head:
                            self._episode_action_heads[episode_key] = 1
                    action_output = ensemble_outputs[selected_action_head]
        selected_mode = None
        selected_mode_source = None
        mode_probabilities = None
        if isinstance(action_output, H3MixtureActionOutput):
            probabilities = action_output.mode_logits.softmax(dim=-1)
            episode_key = str(request["episode_key"])
            if self.forced_action_mode is not None:
                selected_mode = self.forced_action_mode
                selected_mode_source = self.forced_action_mode_source
            elif self.lock_action_mode and episode_key in self._episode_action_modes:
                selected_mode = self._episode_action_modes[episode_key]
                selected_mode_source = "episode_lock"
            else:
                selected_mode = int(probabilities.argmax(dim=-1).item())
                selected_mode_source = "learned_gate"
                if self.lock_action_mode:
                    self._episode_action_modes[episode_key] = selected_mode
            normalized_actions = action_output.actions[:, selected_mode]
            mode_probabilities = probabilities[0].tolist()
        else:
            normalized_actions = action_output
        torch.cuda.synchronize(self.device)
        action_seconds = time.perf_counter() - action_started
        environment_actions, decode_report = libero_environment_actions(
            normalized_actions[0],
            self.stats["action_min"],
            self.stats["action_max"],
            binarize_gripper=self.binarize_gripper,
            temporal_median_window=self.action_median_window,
            normalized_action_pre_clamp=self.normalized_action_pre_clamp,
            return_decode_report=True,
        )
        environment_actions[:, :6] = np.clip(
            environment_actions[:, :6] * self.action_scale, -1.0, 1.0
        )
        return environment_actions, {
            "context_id": task_context["id"],
            "first_environment_action": environment_actions[0].tolist(),
            "environment_action_chunk": environment_actions.tolist(),
            "context_encode_seconds": 0.0,
            "vae_encode_seconds": encode_seconds,
            "h3_feature_seconds": h3_seconds,
            "action_model_seconds": action_seconds,
            "h3_feature_ablation": self.feature_ablation,
            "h3_feature_runtime": self.feature_runtime,
            "h3_feature_timestep": (
                self.feature_timestep if self.feature_runtime == "comfy" else 0.0
            ),
            "h3_condition_video_timestep": (
                None if self.feature_runtime == "comfy" else 0.999
            ),
            "h3_capture_compatibility": (
                None if self.feature_runtime == "comfy" else "comfy_alias_v1"
            ),
            "h3_tail_delta_source_step": (
                None
                if self.tail_delta_report is None
                else self.tail_delta_report.source_step
            ),
            "action_objective": self.objective,
            "action_flow_steps": self.flow_inference_steps if self.objective == "flow" else 0,
            "selected_action_mode": selected_mode,
            "selected_action_mode_source": selected_mode_source,
            "action_mode_probabilities": mode_probabilities,
            "action_mode_locked": self.lock_action_mode,
            "action_scale": self.action_scale,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": decode_report,
            "action_head_ensemble_size": 1 + len(self.ensemble_members),
            "action_head_ensemble_mode": self.ensemble_mode,
            "action_head_switch_step": self.ensemble_switch_step,
            "action_head_disagreement_threshold": self.ensemble_disagreement_threshold,
            "action_head_switch_consecutive": self.ensemble_switch_consecutive,
            "action_head_switch_gate_probability": switch_gate_probability,
            "action_head_switch_gate_threshold": self.switch_gate_threshold,
            "action_head_selected_index": selected_action_head,
            "action_head_motion_disagreement": ensemble_disagreement,
            "action_head_gripper_disagreement": ensemble_gripper_disagreement,
            "inference_seconds": h3_seconds + action_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }


class H3StarWAMInt8Policy:
    """Online LIBERO adapter for schema-2 H3 + pinned StarWAM checkpoints.

    This is deliberately separate from ``H3FeaturePolicy``: the latter restores
    the historical shallow ``h3_feature_action`` schema, while R1 restores the
    byte-pinned 30-layer StarWAM ActionDiT and its shift-5 Euler sampler.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        if args.h3_checkpoint is None or args.h3_model is None:
            raise ValueError(
                "h3_starwam_int8 requires --h3-checkpoint and --h3-model"
            )
        if args.starwam_source_manifest is None:
            raise ValueError(
                "h3_starwam_int8 requires --starwam-source-manifest"
            )
        if args.context_mode != "cached":
            raise ValueError("h3_starwam_int8 requires cached text contexts")
        if args.model_evaluations != 10:
            raise ValueError("R1 StarWAM rollout is fixed to 10 Euler steps")
        if args.h3_video_lora_checkpoint is not None or args.h3_tail_delta is not None:
            raise ValueError("R1 standalone INT8 rollout does not accept H3 deltas")

        self.device = torch.device(args.device)
        if self.device.type != "cuda":
            raise ValueError("h3_starwam_int8 requires a CUDA device")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16
        self.cache_root = args.cache_root.resolve()
        self.source_manifest = args.starwam_source_manifest.resolve()
        self.horizon = int(args.action_horizon)
        self.inference_steps = int(args.model_evaluations)
        self.binarize_gripper = bool(args.binarize_gripper)
        self.action_median_window = int(args.action_median_window)
        self.action_scale = float(args.action_scale)
        self.normalized_action_pre_clamp = bool(
            args.normalized_action_pre_clamp
        )
        self.sample_ensemble_size = int(args.sample_ensemble_size)
        self.feature_ablation = str(args.h3_feature_ablation)
        if self.action_scale <= 0:
            raise ValueError("action-scale must be positive")

        payload = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=True
        )
        if payload.get("schema_version") != 2:
            raise ValueError("R1 rollout requires a schema-2 checkpoint")
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("R1 checkpoint contract is missing")
        if contract.get("feature_strategy") != "starwam_adaptive_avg_pool1d_v1":
            raise ValueError("R1 checkpoint feature strategy mismatch")
        if tuple(contract.get("feature_layers", ())) != (49,):
            raise ValueError("R1 rollout currently requires last32 layer 49")
        if int(contract.get("feature_tokens", -1)) != 32:
            raise ValueError("R1 rollout requires 32 pooled visual tokens")
        if float(contract.get("feature_timestep", -1.0)) != 1.0:
            raise ValueError("R1 H3 feature timestep must be 1.0")
        if float(contract.get("action_shift", -1.0)) != 5.0:
            raise ValueError("R1 action shift must be 5")
        trained_horizon = int(contract.get("action_horizon", -1))
        if self.horizon <= 0 or self.horizon > trained_horizon:
            raise ValueError("requested horizon exceeds the R1 training horizon")
        model_spec = contract.get("model_spec")
        if not isinstance(model_spec, dict):
            raise ValueError("R1 checkpoint model_spec is missing")

        from fastwam.models.h3wam import (
            H3Int8FeatureBackbone,
            H3Int8OnlineFeatureContract,
            H3Int8OnlineFeatureProvider,
            H3StarWAMFeatureActionPolicy,
            encode_h3_vae_condition_standalone,
        )
        from diffusers import AutoencoderKLMiniMaxH3

        self.action_model = H3StarWAMFeatureActionPolicy(
            action_dim=int(model_spec["action_dim"]),
            proprio_dim=int(model_spec["proprio_dim"]),
            h3_feature_dim=int(model_spec["h3_feature_dim"]),
            context_dim=int(model_spec["context_dim"]),
            hidden_dim=int(model_spec["hidden_dim"]),
            ffn_dim=int(model_spec["ffn_dim"]),
            num_heads=int(model_spec["num_heads"]),
            attn_head_dim=int(model_spec["attn_head_dim"]),
            num_layers=int(model_spec["action_layers"]),
            freq_dim=int(model_spec["freq_dim"]),
            max_seq_len=int(model_spec["max_seq_len"]),
            use_gradient_checkpointing=bool(model_spec["gradient_checkpointing"]),
            include_feature_timestep=bool(model_spec["include_feature_timestep"]),
            feature_timestep=float(model_spec["feature_timestep"]),
            feature_input_scale=float(model_spec["feature_input_scale"]),
        ).to(device=self.device, dtype=self.dtype)
        self.action_model.load_state_dict(payload["model"], strict=True)
        self.action_model.eval()
        from starwam.modules.scheduler import FlowMatchScheduler

        self.flow_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=5.0
        )
        self.h3_model = H3Int8FeatureBackbone.from_checkpoint(
            args.h3_checkpoint.resolve()
        ).to(self.device)
        self.h3_model.requires_grad_(False).eval()
        feature_audio_horizon = (
            self.horizon
            if args.h3_feature_audio_horizon is None
            else int(args.h3_feature_audio_horizon)
        )
        self.int8_feature_provider = H3Int8OnlineFeatureProvider(
            self.h3_model,
            H3Int8OnlineFeatureContract(
                layers=(49,),
                action_horizon=feature_audio_horizon,
                target_latent_frames=int(args.target_latent_frames),
                video_timestep=0.0,
                condition_video_timestep=0.999,
                capture_compatibility="comfy_alias_v1",
            ),
        )
        self.video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.h3_model.resolve(),
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self._encode_vae_condition = encode_h3_vae_condition_standalone

        self.stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        if tuple(self.stats["action_min"].shape) != (7,) or tuple(
            self.stats["state_min"].shape
        ) != (8,):
            raise ValueError("R1 cache stats have an unexpected action/state shape")
        self.task_context_ids = self._load_task_context_ids()
        self._contexts: dict[str, dict] = {}
        self.completed_steps = int(payload["completed_steps"])

    def _load_task_context_ids(self) -> dict[str, str]:
        task_context_ids: dict[str, set[str]] = {}
        with self.source_manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task_context_ids.setdefault(str(row["task"]), set()).add(
                    str(row["context_id"])
                )
        ambiguous = {
            task: sorted(context_ids)
            for task, context_ids in task_context_ids.items()
            if len(context_ids) != 1
        }
        if ambiguous:
            raise ValueError(f"R1 task has ambiguous context IDs: {ambiguous}")
        if not task_context_ids:
            raise ValueError("R1 source manifest is empty")
        return {
            task: next(iter(context_ids))
            for task, context_ids in task_context_ids.items()
        }

    def _task_context(self, task: str) -> dict:
        if task not in self.task_context_ids:
            raise ValueError(f"task is absent from the R1 source manifest: {task!r}")
        if task not in self._contexts:
            context_id = self.task_context_ids[task]
            payload = torch.load(
                self.cache_root / "contexts" / f"{context_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("text_only") is not True:
                raise ValueError(f"R1 context is not text-only: {context_id}")
            context = payload["context"]
            if context.ndim != 3 or context.shape[0] != 1:
                raise ValueError(f"unexpected R1 context shape: {context.shape}")
            token_tags = payload.get("token_tags")
            if token_tags is None or torch.any(token_tags != 1):
                raise ValueError(f"R1 context has non-text token tags: {context_id}")
            self._contexts[task] = {
                "id": context_id,
                "context": context.to(device=self.device, dtype=self.dtype),
                "token_tags": token_tags.to(self.device),
            }
        return self._contexts[task]

    @torch.inference_mode()
    def predict(self, request: dict) -> tuple[np.ndarray, dict]:
        from fastwam.models.h3wam import (
            libero_environment_actions,
            libero_observation_state,
            minmax_normalize,
            preprocess_libero_cameras,
        )

        task_context = self._task_context(str(request["task"]))
        pixels = preprocess_libero_cameras(
            request["agentview_image"], request["wristview_image"]
        )
        video = (
            pixels.mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
            .to(self.device)
        )
        vae_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            first_frame = self._encode_vae_condition(
                self.video_vae,
                video,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ).to(device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)
        vae_seconds = time.perf_counter() - vae_started

        h3_started = time.perf_counter()
        features = self.int8_feature_provider(
            first_frame,
            task_context["context"],
            task_context["token_tags"],
        )
        if self.feature_ablation == "zero":
            features = torch.zeros_like(features)
        torch.cuda.synchronize(self.device)
        h3_seconds = time.perf_counter() - h3_started

        state = minmax_normalize(
            libero_observation_state(request),
            self.stats["state_min"],
            self.stats["state_max"],
        ).clamp(-5.0, 5.0).reshape(1, 8).to(
            device=self.device, dtype=self.dtype
        )
        text_mask = torch.ones(
            task_context["context"].shape[:2],
            device=self.device,
            dtype=torch.bool,
        )
        action_started = time.perf_counter()
        action_samples = []
        for sample_index in range(self.sample_ensemble_size):
            generator = torch.Generator(device=self.device).manual_seed(
                int(request["seed"]) + sample_index
            )
            actions = torch.randn(
                (1, self.horizon, 7),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            timesteps, deltas = self.flow_scheduler.build_inference_schedule(
                self.inference_steps, self.device, self.dtype
            )
            for timestep, delta in zip(timesteps, deltas, strict=True):
                velocity = self.action_model(
                    actions,
                    timestep.expand(1),
                    text_context=task_context["context"],
                    h3_features=features,
                    proprio=state,
                    text_mask=text_mask,
                )
                actions = self.flow_scheduler.step(velocity, delta, actions)
            action_samples.append(actions)
        normalized_actions = torch.stack(action_samples).mean(dim=0)
        torch.cuda.synchronize(self.device)
        action_seconds = time.perf_counter() - action_started

        environment_actions, decode_report = libero_environment_actions(
            normalized_actions[0],
            self.stats["action_min"],
            self.stats["action_max"],
            binarize_gripper=self.binarize_gripper,
            temporal_median_window=self.action_median_window,
            normalized_action_pre_clamp=self.normalized_action_pre_clamp,
            return_decode_report=True,
        )
        environment_actions[:, :6] = np.clip(
            environment_actions[:, :6] * self.action_scale, -1.0, 1.0
        )
        return environment_actions, {
            "context_id": task_context["id"],
            "first_environment_action": environment_actions[0].tolist(),
            "environment_action_chunk": environment_actions.tolist(),
            "vae_encode_seconds": vae_seconds,
            "h3_feature_seconds": h3_seconds,
            "action_model_seconds": action_seconds,
            "inference_seconds": h3_seconds + action_seconds,
            "h3_feature_runtime": "int8",
            "h3_feature_ablation": self.feature_ablation,
            "action_objective": "pinned_starwam_weighted_masked_flow",
            "action_flow_steps": self.inference_steps,
            "checkpoint_completed_steps": self.completed_steps,
            "sample_ensemble_size": self.sample_ensemble_size,
            "action_scale": self.action_scale,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": decode_report,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class H3DreamWAMKVInt8Policy:
    """Online LIBERO adapter for Candidate D/D0 schema-2 checkpoints."""

    @staticmethod
    def _resolve_candidate_source_mode(contract: dict) -> tuple[str, str]:
        candidate = str(contract.get("candidate"))
        expected_source_modes = {
            "D": "aligned_5layer",
            "D0": "repeat_layer49",
        }
        if candidate not in expected_source_modes:
            raise ValueError("online rollout only supports paired Candidate D/D0")
        carrier_source_mode = str(contract.get("carrier_source_mode"))
        if carrier_source_mode != expected_source_modes[candidate]:
            raise ValueError(
                f"Candidate {candidate} rollout source mode mismatch: "
                f"{carrier_source_mode} != {expected_source_modes[candidate]}"
            )
        return candidate, carrier_source_mode

    def __init__(self, args: argparse.Namespace) -> None:
        fastwam_online = args.policy == "h3_fastwam_online_int8"
        policy_name = "h3_fastwam_online_int8" if fastwam_online else "h3_dreamwam_kv_int8"
        if args.h3_checkpoint is None or args.h3_model is None:
            raise ValueError(
                f"{policy_name} requires --h3-checkpoint and --h3-model"
            )
        if args.dreamwam_source_manifest is None:
            raise ValueError(
                f"{policy_name} requires --dreamwam-source-manifest"
            )
        if args.context_mode != "cached":
            raise ValueError(f"{policy_name} requires cached text contexts")
        if args.model_evaluations != 10:
            raise ValueError(f"{policy_name} rollout is fixed to 10 Euler steps")
        if args.h3_video_lora_checkpoint is not None or args.h3_tail_delta is not None:
            raise ValueError(f"{policy_name} rollout does not accept H3 deltas")
        if fastwam_online:
            if args.c58b_balanced80_ready is None:
                raise ValueError(
                    "h3_fastwam_online_int8 requires --c58b-balanced80-ready"
                )
            if (
                args.progress_probe is not None
                or args.consequence_ranker_checkpoint is not None
                or args.dense_value_checkpoint is not None
            ):
                raise ValueError("C58b fresh canary forbids auxiliary rankers/probes")

        self.device = torch.device(args.device)
        if self.device.type != "cuda":
            raise ValueError(f"{policy_name} requires a CUDA device")
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16
        self.cache_root = args.cache_root.resolve()
        self.source_manifest = args.dreamwam_source_manifest.resolve()
        self.horizon = int(args.action_horizon)
        self.inference_steps = int(args.model_evaluations)
        self.binarize_gripper = bool(args.binarize_gripper)
        self.action_median_window = int(args.action_median_window)
        self.action_scale = float(args.action_scale)
        self.normalized_action_pre_clamp = bool(args.normalized_action_pre_clamp)
        self.sample_ensemble_size = int(args.sample_ensemble_size)
        self.feature_ablation = str(args.h3_feature_ablation)
        self.consequence_best_of_n = int(args.consequence_best_of_n)
        if self.consequence_best_of_n <= 0:
            raise ValueError("consequence-best-of-n must be positive")
        c44_ranker_requested = args.consequence_ranker_checkpoint is not None
        if c44_ranker_requested != bool(args.consequence_model_checkpoint):
            raise ValueError(
                "C44 ranker and its consequence checkpoints must be passed together"
            )
        dense_ranker_requested = (
            args.dense_value_checkpoint is not None
            or args.dense_value_final_report is not None
        )
        if dense_ranker_requested and (
            args.dense_value_checkpoint is None
            or args.dense_value_final_report is None
        ):
            raise ValueError("dense value checkpoint and C51 report must be passed together")
        if c44_ranker_requested and dense_ranker_requested:
            raise ValueError("C44 and C51 online rankers are mutually exclusive")
        ranker_requested = c44_ranker_requested or dense_ranker_requested
        if ranker_requested:
            if self.consequence_best_of_n < 2:
                raise ValueError("C44 online selection requires consequence-best-of-n >= 2")
            if self.sample_ensemble_size != 1:
                raise ValueError("C44 best-of-N cannot be combined with action averaging")
            if self.horizon != 32:
                raise ValueError("C44 ranker was trained only for a 32-action horizon")
            if self.feature_ablation != "none":
                raise ValueError("C44 ranker does not support H3 feature ablation")
            offsets = (
                list(args.consequence_candidate_seed_offset)
                if args.consequence_candidate_seed_offset
                else list(range(self.consequence_best_of_n))
            )
            if len(offsets) != self.consequence_best_of_n:
                raise ValueError("candidate seed offsets must match consequence-best-of-n")
            if offsets[0] != 0 or len(set(offsets)) != len(offsets) or min(offsets) < 0:
                raise ValueError("candidate seed offsets must be unique, nonnegative and start at 0")
            self.consequence_candidate_seed_offsets = tuple(offsets)
            self.consequence_selection_min_step = int(
                args.consequence_selection_min_step
            )
            self.consequence_selection_max_step = args.consequence_selection_max_step
            if self.consequence_selection_min_step < 0:
                raise ValueError("consequence-selection-min-step must be nonnegative")
            if (
                self.consequence_selection_max_step is not None
                and self.consequence_selection_max_step <= 0
            ):
                raise ValueError("consequence-selection-max-step must be positive")
            if (
                self.consequence_selection_max_step is not None
                and self.consequence_selection_min_step
                >= self.consequence_selection_max_step
            ):
                raise ValueError("consequence selection step interval is empty")
        elif self.consequence_best_of_n != 1:
            raise ValueError("consequence-best-of-n > 1 requires a C44 ranker")
        else:
            if args.consequence_candidate_seed_offset:
                raise ValueError("candidate seed offsets require a C44 ranker")
            if args.consequence_selection_max_step is not None:
                raise ValueError("selection max step requires a C44 ranker")
            self.consequence_candidate_seed_offsets = (0,)
            self.consequence_selection_min_step = 0
            self.consequence_selection_max_step = None
        if self.action_scale <= 0:
            raise ValueError("action-scale must be positive")

        payload = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        expected_schema = 1 if fastwam_online else 2
        if payload.get("schema_version") != expected_schema:
            raise ValueError(f"{policy_name} requires a schema-{expected_schema} checkpoint")
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{policy_name} checkpoint contract is missing")
        if fastwam_online:
            candidate = str(contract.get("candidate"))
            carrier_source_mode = str(contract.get("carrier_source_mode"))
            if candidate != "C58B_FASTWAM_FULL30_H3_LAYERWISE":
                raise ValueError("C58b candidate identity mismatch")
            if carrier_source_mode != "uniform_h3_50_to_action30":
                raise ValueError("C58b carrier source mode mismatch")
        else:
            candidate, carrier_source_mode = self._resolve_candidate_source_mode(contract)
        if contract.get("kv_schema") != "h3_dreamwam_kv_v1":
            raise ValueError(f"{policy_name} K/V schema mismatch")
        carrier_layers = tuple(int(layer) for layer in contract.get("kv_layers", ()))
        expected_carrier_layers = (
            (0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
             25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49)
            if fastwam_online else (9, 19, 29, 39, 49)
        )
        if carrier_layers != expected_carrier_layers:
            raise ValueError(f"{policy_name} requires audited carrier layers")
        if int(contract.get("kv_tokens", -1)) != 32:
            raise ValueError("Candidate D0 rollout requires 32 pooled K/V tokens")
        kv_pool_strategy = str(contract.get("kv_strategy", ""))
        if kv_pool_strategy not in (
            "adaptive_avg_pool1d_sequence_v1",
            "dual_view_spatial_grid_4x4_each_v1",
        ):
            raise ValueError("Candidate D0 rollout K/V pool strategy is unsupported")
        if not fastwam_online and int(contract.get("kv_num_heads", -1)) != 56:
            raise ValueError("Candidate D0 rollout requires 56 H3 attention heads")
        if not fastwam_online and int(contract.get("kv_attn_head_dim", -1)) != 128:
            raise ValueError("Candidate D0 rollout requires H3 head dimension 128")
        if float(contract.get("action_shift", -1.0)) != 5.0:
            raise ValueError("Candidate D0 action shift must be 5")
        trained_horizon = int(contract.get("action_horizon", -1))
        if self.horizon <= 0 or self.horizon > trained_horizon:
            raise ValueError("requested horizon exceeds Candidate D0 training horizon")
        expected_h3_sha256 = str(contract.get("h3_checkpoint_sha256", ""))
        if contract.get("verify_h3_checkpoint_sha256") is not True:
            raise ValueError("Candidate D0 checkpoint did not verify its H3 weights")
        actual_h3_sha256 = _sha256_file(args.h3_checkpoint.resolve())
        if actual_h3_sha256 != expected_h3_sha256:
            raise ValueError(
                "Candidate D0 online H3 checkpoint SHA256 mismatch: "
                f"{actual_h3_sha256} != {expected_h3_sha256}"
            )
        model_spec = contract.get("model_spec")
        if not isinstance(model_spec, dict):
            raise ValueError("Candidate D0 checkpoint model_spec is missing")
        if tuple(model_spec.get("carrier_layers", ())) != carrier_layers:
            raise ValueError("Candidate D0 model and cache carrier layers differ")
        if model_spec.get("carrier_source_mode") != carrier_source_mode:
            raise ValueError(
                f"Candidate {candidate} model_spec source mode mismatch"
            )
        if fastwam_online:
            if (
                int(payload.get("completed_steps", -1)) != 10_000
                or int(model_spec.get("action_layers", -1)) != 30
                or tuple(contract.get("action_block_to_h3_layer", ()))
                != expected_carrier_layers
                or contract.get("h3_execution") != "online_frozen_int8_per_rank_v1"
                or contract.get("disk_kv_training_input") is not False
                or contract.get("kv_subdir") is not None
            ):
                raise ValueError("C58b online full30 deployment contract mismatch")
            ready_path = args.c58b_balanced80_ready.resolve()
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            checkpoint_sha256 = _sha256_file(args.checkpoint.resolve())
            if (
                ready.get("permission") != "GO_FRESH_LIBERO"
                or Path(ready.get("checkpoint", "")).resolve()
                != args.checkpoint.resolve()
                or ready.get("checkpoint_sha256") != checkpoint_sha256
            ):
                raise ValueError("C58b balanced80 gate/checkpoint identity mismatch")

        from fastwam.models.h3wam import (
            FrozenConsequenceActionRanker,
            FrozenDenseValueActionRanker,
            FrozenH3ProgressProbe,
            H3DreamWAMKVCarrierPolicy,
            H3Int8FeatureBackbone,
            H3Int8OnlineFeatureContract,
            H3Int8OnlineFeatureProvider,
            H3Int8OnlineKVContract,
            H3Int8OnlineKVProvider,
            encode_h3_vae_condition_standalone,
        )
        from fastwam.models.h3wam.fastwam_full_tower import (
            H3FastWAMFullTowerPolicy,
        )
        from diffusers import AutoencoderKLMiniMaxH3
        from fastwam.models.h3wam.starwam_feature_action import (
            _load_pinned_starwam_action_dit,
        )

        # Install only the byte-verified lightweight StarWAM module namespace.
        # Importing StarWAM's package root would pull its PyArrow dataset stack
        # into the isolated policy runtime.
        _load_pinned_starwam_action_dit()
        from starwam.modules.scheduler import FlowMatchScheduler

        self.carrier_layers = carrier_layers
        self.candidate = candidate
        self.carrier_source_mode = carrier_source_mode
        self.captured_h3_layers = (
            carrier_layers
            if fastwam_online or carrier_source_mode == "aligned_5layer"
            else (49,)
        )
        if fastwam_online:
            self.action_model = H3FastWAMFullTowerPolicy(
                enabled=True,
                carrier_layers=carrier_layers,
                action_dim=int(model_spec["action_dim"]),
                proprio_dim=int(model_spec["proprio_dim"]),
                context_dim=int(model_spec["context_dim"]),
                hidden_dim=int(model_spec["hidden_dim"]),
                ffn_dim=int(model_spec["ffn_dim"]),
                num_heads=int(model_spec["num_heads"]),
                attn_head_dim=int(model_spec["attn_head_dim"]),
                freq_dim=int(model_spec["freq_dim"]),
                num_layers=30,
                use_gradient_checkpointing=False,
                action_block_to_h3_layer=expected_carrier_layers,
            ).to(device=self.device, dtype=self.dtype)
        else:
            self.action_model = H3DreamWAMKVCarrierPolicy(
                enabled=True,
                carrier_layers=carrier_layers,
                carrier_source_mode=carrier_source_mode,
                action_dim=int(model_spec["action_dim"]),
                proprio_dim=int(model_spec["proprio_dim"]),
                context_dim=int(model_spec["context_dim"]),
                hidden_dim=int(model_spec["hidden_dim"]),
                ffn_dim=int(model_spec["ffn_dim"]),
                num_heads=int(model_spec["num_heads"]),
                attn_head_dim=int(model_spec["attn_head_dim"]),
                freq_dim=int(model_spec["freq_dim"]),
                history_action_steps=int(model_spec.get("history_action_steps", 0)),
            ).to(device=self.device, dtype=self.dtype)
        self.action_model.load_state_dict(payload["model"], strict=True)
        self.action_model.eval()
        self.flow_scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=5.0
        )
        self.h3_model = H3Int8FeatureBackbone.from_checkpoint(
            args.h3_checkpoint.resolve()
        ).to(self.device)
        self.h3_model.requires_grad_(False).eval()
        feature_audio_horizon = (
            self.horizon
            if args.h3_feature_audio_horizon is None
            else int(args.h3_feature_audio_horizon)
        )
        self.int8_kv_provider = H3Int8OnlineKVProvider(
            self.h3_model,
            H3Int8OnlineKVContract(
                layers=self.captured_h3_layers,
                action_horizon=feature_audio_horizon,
                target_latent_frames=int(args.target_latent_frames),
                video_timestep=1.0,
                condition_video_timestep=1.0,
                capture_token_count=32,
                pool_strategy=kv_pool_strategy,
            ),
        )
        if ranker_requested:
            self.int8_hidden_provider = H3Int8OnlineFeatureProvider(
                self.h3_model,
                H3Int8OnlineFeatureContract(
                    layers=(49,),
                    action_horizon=32,
                    target_latent_frames=12,
                    video_timestep=1.0,
                    condition_video_timestep=1.0,
                    capture_compatibility="none",
                ),
            )
            if c44_ranker_requested:
                self.consequence_ranker = FrozenConsequenceActionRanker(
                    args.consequence_ranker_checkpoint,
                    args.consequence_model_checkpoint,
                    device=self.device,
                )
                self.action_ranker_type = "c44_consequence"
            else:
                self.consequence_ranker = FrozenDenseValueActionRanker(
                    args.dense_value_checkpoint,
                    args.dense_value_final_report,
                    device=self.device,
                )
                self.action_ranker_type = "c51_dense_value"
        else:
            self.int8_hidden_provider = None
            self.consequence_ranker = None
            self.action_ranker_type = None
        if args.progress_probe is not None:
            if kv_pool_strategy != "adaptive_avg_pool1d_sequence_v1":
                raise ValueError(
                    "H3 progress probe was trained only on sequence-pooled layer49 K/V"
                )
            if self.feature_ablation != "none":
                raise ValueError("H3 progress shadow does not support feature ablation")
            self.progress_probe = FrozenH3ProgressProbe.load(args.progress_probe)
        else:
            self.progress_probe = None
        self.video_vae = AutoencoderKLMiniMaxH3.from_pretrained(
            args.h3_model.resolve(),
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self._encode_vae_condition = encode_h3_vae_condition_standalone
        self.stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        if tuple(self.stats["action_min"].shape) != (7,) or tuple(
            self.stats["state_min"].shape
        ) != (8,):
            raise ValueError("Candidate D0 cache stats have unexpected shapes")
        self.context_width = int(model_spec["context_dim"])
        self.task_context_ids = self._load_task_context_ids()
        self._contexts: dict[str, dict] = {}
        self.completed_steps = int(payload["completed_steps"])
        self.history_action_steps = int(model_spec.get("history_action_steps", 0))
        if not fastwam_online and self.history_action_steps != int(contract.get("history_action_steps", 0)):
            raise ValueError("Candidate D0 history-action contract mismatch")
        if fastwam_online and self.history_action_steps != 0:
            raise ValueError("C58b fresh canary requires zero action history")
        self.h3_checkpoint_sha256 = actual_h3_sha256
        self.fastwam_online = fastwam_online

    def _load_task_context_ids(self) -> dict[str, str]:
        task_context_ids: dict[str, set[str]] = {}
        with self.source_manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task_context_ids.setdefault(str(row["task"]), set()).add(
                    str(row["context_id"])
                )
        ambiguous = {
            task: sorted(context_ids)
            for task, context_ids in task_context_ids.items()
            if len(context_ids) != 1
        }
        if ambiguous:
            raise ValueError(f"Candidate D0 task has ambiguous context IDs: {ambiguous}")
        if not task_context_ids:
            raise ValueError("Candidate D0 source manifest is empty")
        return {
            task: next(iter(context_ids))
            for task, context_ids in task_context_ids.items()
        }

    def _task_context(self, task: str) -> dict:
        if task not in self.task_context_ids:
            raise ValueError(
                f"task is absent from Candidate D0 source manifest: {task!r}"
            )
        if task not in self._contexts:
            context_id = self.task_context_ids[task]
            payload = torch.load(
                self.cache_root / "contexts" / f"{context_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if payload.get("text_only") is not True:
                raise ValueError(f"Candidate D0 context is not text-only: {context_id}")
            context = payload["context"]
            if tuple(context.shape[:1]) != (1,) or context.ndim != 3:
                raise ValueError(f"unexpected Candidate D0 context shape: {context.shape}")
            if int(context.shape[-1]) != self.context_width:
                raise ValueError("Candidate D0 context width differs from training")
            token_tags = payload.get("token_tags")
            if token_tags is None or torch.any(token_tags != 1):
                raise ValueError(f"Candidate D0 context has non-text tags: {context_id}")
            self._contexts[task] = {
                "id": context_id,
                "context": context.to(device=self.device, dtype=self.dtype),
                "token_tags": token_tags.to(self.device),
            }
        return self._contexts[task]

    @torch.inference_mode()
    def predict(self, request: dict) -> tuple[np.ndarray, dict]:
        from fastwam.models.h3wam import (
            libero_environment_actions,
            libero_observation_state,
            minmax_normalize,
            preprocess_libero_cameras,
        )

        task_context = self._task_context(str(request["task"]))
        rank_this_request = self.consequence_ranker is not None and (
            int(request.get("step", 0)) >= self.consequence_selection_min_step
            and (
                self.consequence_selection_max_step is None
                or int(request.get("step", 0)) < self.consequence_selection_max_step
            )
        )
        pixels = preprocess_libero_cameras(
            request["agentview_image"], request["wristview_image"]
        )
        video = (
            pixels.mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
            .to(self.device)
        )
        vae_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            first_frame = self._encode_vae_condition(
                self.video_vae,
                video,
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ).to(device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)
        vae_seconds = time.perf_counter() - vae_started

        h3_started = time.perf_counter()
        live_cache = self.int8_kv_provider(
            first_frame,
            task_context["context"],
            task_context["token_tags"],
        )
        consequence_hidden = None
        if rank_this_request:
            consequence_hidden = self.int8_hidden_provider(
                first_frame,
                task_context["context"],
                task_context["token_tags"],
            )[0]
            if consequence_hidden.shape[1] > 32:
                consequence_hidden = torch.nn.functional.adaptive_avg_pool1d(
                    consequence_hidden.transpose(1, 2), 32
                ).transpose(1, 2)
            if tuple(consequence_hidden.shape) != (1, 32, 5376):
                raise RuntimeError(
                    "C44 online H3 feature shape mismatch: "
                    f"{tuple(consequence_hidden.shape)}"
                )
            consequence_hidden = consequence_hidden.to(torch.bfloat16)
        progress_value = None
        if self.progress_probe is not None:
            progress_value = self.progress_probe.predict(
                context_id=task_context["id"],
                absolute_step=int(request.get("step", 0)),
                layer_cache=live_cache[49],
            )
        if self.feature_ablation == "zero":
            live_cache = {
                layer: {
                    name: torch.zeros_like(value)
                    for name, value in layer_cache.items()
                }
                for layer, layer_cache in live_cache.items()
            }
        if self.carrier_source_mode == "repeat_layer49":
            layer49 = live_cache[49]
            video_kv_cache = {
                layer: {
                    "k": layer49["k"].clone(),
                    "v": layer49["v"].clone(),
                }
                for layer in self.carrier_layers
            }
        else:
            video_kv_cache = {
                layer: {
                    "k": live_cache[layer]["k"],
                    "v": live_cache[layer]["v"],
                }
                for layer in self.carrier_layers
            }
        torch.cuda.synchronize(self.device)
        h3_seconds = time.perf_counter() - h3_started

        state = minmax_normalize(
            libero_observation_state(request),
            self.stats["state_min"],
            self.stats["state_max"],
        ).clamp(-5.0, 5.0).reshape(1, 8).to(
            device=self.device, dtype=self.dtype
        )
        text_mask = torch.ones(
            task_context["context"].shape[:2],
            device=self.device,
            dtype=torch.bool,
        )
        normalized_history = None
        history_valid = None
        if self.history_action_steps:
            from fastwam.models.h3wam import normalize_libero_environment_action_history

            raw_history = np.asarray(
                request.get("executed_action_history", []), dtype=np.float32
            ).reshape(-1, 7)
            if len(raw_history) > self.history_action_steps:
                raw_history = raw_history[-self.history_action_steps :]
            missing = self.history_action_steps - len(raw_history)
            padded = np.zeros((self.history_action_steps, 7), dtype=np.float32)
            valid = np.zeros(self.history_action_steps, dtype=bool)
            if len(raw_history):
                padded[missing:] = raw_history
                valid[missing:] = True
            normalized_history = normalize_libero_environment_action_history(
                padded,
                valid,
                self.stats["action_min"],
                self.stats["action_max"],
                clip=5.0,
            ).reshape(1, self.history_action_steps, 7).to(
                device=self.device, dtype=self.dtype
            )
            history_valid = torch.as_tensor(
                valid, device=self.device, dtype=torch.bool
            ).reshape(1, self.history_action_steps)
        action_started = time.perf_counter()
        action_samples = []
        generated_samples = (
            self.consequence_best_of_n
            if rank_this_request
            else self.sample_ensemble_size
        )
        for sample_index in range(generated_samples):
            generator = torch.Generator(device=self.device).manual_seed(
                int(request["seed"])
                + (
                    self.consequence_candidate_seed_offsets[sample_index]
                    if rank_this_request
                    else sample_index
                )
            )
            actions = torch.randn(
                (1, self.horizon, 7),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            timesteps, deltas = self.flow_scheduler.build_inference_schedule(
                self.inference_steps, self.device, self.dtype
            )
            for timestep, delta in zip(timesteps, deltas, strict=True):
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    action_kwargs = {
                        "text_context": task_context["context"],
                        "proprio": state,
                        "video_kv_cache": video_kv_cache,
                        "text_mask": text_mask,
                    }
                    if not self.fastwam_online:
                        action_kwargs.update({
                            "executed_action_history": normalized_history,
                            "executed_action_history_valid": history_valid,
                        })
                    velocity = self.action_model(
                        actions, timestep.float().expand(1), **action_kwargs
                    )
                actions = self.flow_scheduler.step(velocity, delta, actions)
            action_samples.append(actions)
        normalized_actions = (
            action_samples[0]
            if rank_this_request
            else torch.stack(action_samples).mean(dim=0)
        )
        torch.cuda.synchronize(self.device)
        action_seconds = time.perf_counter() - action_started

        candidate_environment_actions = []
        candidate_decode_reports = []
        candidates_to_decode = (
            action_samples if rank_this_request else [normalized_actions]
        )
        for candidate in candidates_to_decode:
            decoded, report = libero_environment_actions(
                candidate[0],
                self.stats["action_min"],
                self.stats["action_max"],
                binarize_gripper=self.binarize_gripper,
                temporal_median_window=self.action_median_window,
                normalized_action_pre_clamp=self.normalized_action_pre_clamp,
                return_decode_report=True,
            )
            decoded[:, :6] = np.clip(decoded[:, :6] * self.action_scale, -1.0, 1.0)
            candidate_environment_actions.append(decoded)
            candidate_decode_reports.append(report)
        selected_index = 0
        candidate_scores = None
        if rank_this_request:
            raw_state = torch.as_tensor(
                libero_observation_state(request), device=self.device, dtype=torch.float32
            )
            action_tensor = torch.as_tensor(
                np.stack(candidate_environment_actions),
                device=self.device,
                dtype=torch.float32,
            )
            scores = self.consequence_ranker.score(
                raw_state, consequence_hidden, action_tensor
            )
            selected_index = int(scores.argmax().item())
            candidate_scores = scores.float().cpu().tolist()
        environment_actions = candidate_environment_actions[selected_index]
        decode_report = candidate_decode_reports[selected_index]
        return environment_actions, {
            "context_id": task_context["id"],
            "first_environment_action": environment_actions[0].tolist(),
            "environment_action_chunk": environment_actions.tolist(),
            "vae_encode_seconds": vae_seconds,
            "h3_kv_seconds": h3_seconds,
            "action_model_seconds": action_seconds,
            "inference_seconds": h3_seconds + action_seconds,
            "h3_feature_runtime": "int8_live_kv",
            "h3_feature_ablation": self.feature_ablation,
            "h3_checkpoint_sha256": self.h3_checkpoint_sha256,
            "progress_value": progress_value,
            "progress_probe_format": (
                None
                if self.progress_probe is None
                else self.progress_probe.format
            ),
            "progress_shadow_only": self.progress_probe is not None,
            "candidate": self.candidate,
            "carrier_source_mode": self.carrier_source_mode,
            "carrier_layers": list(self.carrier_layers),
            "captured_h3_layers": list(self.captured_h3_layers),
            "kv_tokens": 32,
            "action_objective": (
                "fastwam_full30_layerwise_h3_shift5_flow"
                if self.fastwam_online else "dreamwam_kv_shift5_flow"
            ),
            "action_flow_steps": self.inference_steps,
            "checkpoint_completed_steps": self.completed_steps,
            "history_action_steps": self.history_action_steps,
            "history_valid_steps": (
                0 if history_valid is None else int(history_valid.sum().item())
            ),
            "sample_ensemble_size": self.sample_ensemble_size,
            "consequence_best_of_n": self.consequence_best_of_n,
            "consequence_candidate_seeds": (
                None
                if not rank_this_request
                else [
                    int(request["seed"]) + offset
                    for offset in self.consequence_candidate_seed_offsets
                ]
            ),
            "consequence_candidate_scores": candidate_scores,
            "consequence_candidate0_first_environment_action": (
                None
                if not rank_this_request
                else candidate_environment_actions[0][0].tolist()
            ),
            "consequence_candidate0_environment_action_chunk": (
                None
                if not rank_this_request
                else candidate_environment_actions[0].tolist()
            ),
            "consequence_selected_index": (
                None if not rank_this_request else selected_index
            ),
            "consequence_selected_seed": (
                None
                if not rank_this_request
                else int(request["seed"])
                + self.consequence_candidate_seed_offsets[selected_index]
            ),
            "consequence_score_range": (
                None
                if candidate_scores is None
                else max(candidate_scores) - min(candidate_scores)
            ),
            "consequence_ranker_sha256": (
                None
                if self.consequence_ranker is None
                else self.consequence_ranker.ranker_checkpoint_sha256
            ),
            "action_ranker_type": self.action_ranker_type,
            "consequence_candidate_seed_offsets": (
                None
                if self.consequence_ranker is None
                else list(self.consequence_candidate_seed_offsets)
            ),
            "consequence_selection_max_step": self.consequence_selection_max_step,
            "consequence_selection_min_step": self.consequence_selection_min_step,
            "consequence_checkpoint_sha256": (
                None
                if self.consequence_ranker is None
                else self.consequence_ranker.consequence_checkpoint_sha256
            ),
            "action_scale": self.action_scale,
            "normalized_action_pre_clamp": self.normalized_action_pre_clamp,
            "normalized_action_decode": decode_report,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
        }


def main() -> None:
    args = parse_args()
    if (
        args.model_evaluations <= 0
        or args.action_horizon <= 0
        or args.sample_ensemble_size <= 0
    ):
        raise ValueError(
            "model-evaluations, action-horizon and sample-ensemble-size must be positive"
        )
    args.ready_file = args.ready_file.resolve()
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.unlink(missing_ok=True)
    torch.manual_seed(args.seed)
    if args.policy == "h3":
        policy = H3Policy(args)
    elif args.policy in ("h3_feature", "h3_feature_int8"):
        policy = H3FeaturePolicy(args)
    elif args.policy == "h3_starwam_int8":
        policy = H3StarWAMInt8Policy(args)
    elif args.policy in ("h3_dreamwam_kv_int8", "h3_fastwam_online_int8"):
        policy = H3DreamWAMKVInt8Policy(args)
    else:
        policy = BaselinePolicy(args)
    listener = Listener((args.host, args.port), authkey=args.authkey.encode("utf-8"))
    args.ready_file.write_text(
        json.dumps(
            {
                "ready": True,
                "policy": args.policy,
                "host": args.host,
                "port": args.port,
                "checkpoint": str(args.checkpoint.resolve()),
                "normalized_action_pre_clamp": args.normalized_action_pre_clamp,
            },
            indent=2,
        )
    )
    print(json.dumps({"stage": "ready", "policy": args.policy, "port": args.port}), flush=True)
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
                    if command != "predict":
                        connection.send({"ok": False, "error": f"unknown command: {command}"})
                        continue
                    try:
                        request["agentview_image"] = np.frombuffer(
                            request.pop("agentview_bytes"), dtype=np.uint8
                        ).reshape(request.pop("agentview_shape"))
                        request["wristview_image"] = np.frombuffer(
                            request.pop("wristview_bytes"), dtype=np.uint8
                        ).reshape(request.pop("wristview_shape"))
                        for key in ("eef_pos", "eef_quat", "gripper_qpos"):
                            request[key] = np.asarray(request[key], dtype=np.float32)
                        actions, metadata = policy.predict(request)
                        connection.send(
                            {
                                "ok": True,
                                "actions": actions.tolist(),
                                "metadata": metadata,
                            }
                        )
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
