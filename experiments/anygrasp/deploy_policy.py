from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}; expected no/fp16/bf16.")


def _optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    out = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not out.is_absolute():
        out = base / out
    return out.resolve()


def compose_task_config(task: str, overrides: Optional[list[str]] = None) -> DictConfig:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    hydra_overrides = [f"task={task}"]
    if overrides:
        hydra_overrides.extend(str(item) for item in overrides)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(config_name="train", overrides=hydra_overrides)


def infer_dataset_stats_path(checkpoint_path: Path, explicit: Optional[str | Path] = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None and not _is_none_like(explicit):
        candidates.append(_resolve_path(explicit))
    for parent in list(checkpoint_path.parents)[:5]:
        candidates.append(parent / "dataset_stats.json")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find dataset_stats.json. Pass --dataset-stats-path or place it near the checkpoint."
    )


class FastWAMAnyGraspPolicy:
    """GR00T-style policy wrapper for real AnyGrasp FastWAM checkpoints.

    Expected observation format:
        {
            "video": {"top": np.ndarray},       # uint8, HWC / THWC / BTHWC
            "state": {"default": np.ndarray},   # float32, D / TD / BTD
            "language": {"task": [[str]]},      # or a plain string via "instruction"
        }

    Convenience aliases are also accepted: observation["image"], observation["state"],
    and observation["instruction"].
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        checkpoint_path: str | Path,
        dataset_stats_path: str | Path,
        device: str = "cuda:0",
        mixed_precision: str = "bf16",
        action_horizon: Optional[int] = None,
        replan_steps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        high_video_inference_steps: Optional[int] = None,
        low_video_inference_steps: Optional[int] = None,
        high_denoise_step: Optional[int] = None,
        low_denoise_step: Optional[int] = None,
        high_reuse_step: Optional[int] = None,
        low_reuse_step: Optional[int] = None,
        action_inference_steps: Optional[int] = None,
        joint_denoise: Optional[bool] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_hierarchical: bool = False,
        compile_cudagraphs: bool = True,
        optimize_denoise_static: bool = True,
        inference_backend: str = "inductor",
        warmup: bool = True,
        rtc_warmup: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.", flush=True)
            self.device = "cpu"
        self.model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None

        self.model = instantiate(model_cfg, model_dtype=self.model_dtype, device=self.device)
        self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(self.device).eval()

        self.is_hierarchical_model = callable(getattr(self.model, "infer_hierarchical", None))

        self.processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        configured_horizon = _optional_int(cfg.get("action_horizon"))
        if configured_horizon is None:
            configured_horizon = _optional_int(cfg.model.get("hierarchical_action_horizon"))
        if configured_horizon is None:
            configured_horizon = int(cfg.data.train.num_frames) - 1
        self.action_horizon = int(action_horizon if action_horizon is not None else configured_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")

        configured_replan_steps = _optional_int(cfg.get("replan_steps"))
        if configured_replan_steps is None:
            configured_replan_steps = self.action_horizon
        requested_replan_steps = int(replan_steps if replan_steps is not None else configured_replan_steps)
        if requested_replan_steps <= 0:
            raise ValueError(f"`replan_steps` must be positive, got {requested_replan_steps}.")
        self.replan_steps = min(requested_replan_steps, self.action_horizon)

        configured_inference_steps = _optional_int(cfg.get("num_inference_steps"))
        if configured_inference_steps is None:
            configured_inference_steps = int(cfg.eval_num_inference_steps)
        self.num_inference_steps = int(
            num_inference_steps if num_inference_steps is not None else configured_inference_steps
        )
        if self.num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {self.num_inference_steps}.")
        self.high_video_inference_steps = high_video_inference_steps if high_video_inference_steps is not None else _optional_int(cfg.get("high_video_inference_steps"))
        self.low_video_inference_steps = low_video_inference_steps if low_video_inference_steps is not None else _optional_int(cfg.get("low_video_inference_steps"))
        self.high_denoise_step = high_denoise_step if high_denoise_step is not None else _optional_int(cfg.get("high_denoise_step"))
        self.low_denoise_step = low_denoise_step if low_denoise_step is not None else _optional_int(cfg.get("low_denoise_step"))
        self.high_reuse_step = high_reuse_step if high_reuse_step is not None else _optional_int(cfg.get("high_reuse_step"))
        self.low_reuse_step = low_reuse_step if low_reuse_step is not None else _optional_int(cfg.get("low_reuse_step"))
        self.action_inference_steps = action_inference_steps if action_inference_steps is not None else _optional_int(cfg.get("action_inference_steps"))
        self.joint_denoise = bool(cfg.get("joint_denoise", False) if joint_denoise is None else joint_denoise)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.compile_hierarchical = bool(compile_hierarchical)
        self.compile_cudagraphs = bool(compile_cudagraphs)
        self.optimize_denoise_static = bool(optimize_denoise_static)
        self.inference_backend = str(inference_backend).strip().lower()
        self.rtc_warmup = bool(rtc_warmup)
        if self.inference_backend not in {"inductor", "tensorrt"}:
            raise ValueError("inference_backend must be 'inductor' or 'tensorrt'.")

        self.image_key = str(self.processor.shape_meta["images"][0]["key"])
        self.state_key = str(self.processor.shape_meta["state"][0]["key"])
        self.action_key = str(self.processor.shape_meta["action"][0]["key"])
        self.input_h = int(self.processor.shape_meta["images"][0]["shape"][1])
        self.input_w = int(self.processor.shape_meta["images"][0]["shape"][2])
        self.raw_state_dim = int(self.processor.shape_meta["state"][0]["raw_shape"])
        self.state_dim = int(self.processor.shape_meta["state"][0]["shape"])
        self.raw_action_dim = int(self.processor.shape_meta["action"][0]["raw_shape"])
        self.action_dim = int(self.processor.shape_meta["action"][0]["shape"])
        self.num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

        self._frame_history: list[torch.Tensor] = []
        self._last_frame_action_index: Optional[int] = None
        self._max_keyframe_history = 9
        self._cached_prompt: Optional[str] = None
        self._cached_context: Optional[torch.Tensor] = None
        self._cached_context_mask: Optional[torch.Tensor] = None
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0
        self._warmup_info: Optional[dict[str, Any]] = None

        if warmup:
            info = self.warmup()
            print(f"FastWAM AnyGrasp warmup completed in {info['warmup_s']:.3f}s", flush=True)

    def reset(self, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        del options
        self._frame_history.clear()
        self._last_frame_action_index = None
        clear_reuse_cache = getattr(self.model, "clear_hierarchical_reuse_cache", None)
        if callable(clear_reuse_cache):
            clear_reuse_cache()
        return {"status": "ok"}

    def warmup(self, instruction: str = "warm up") -> dict[str, Any]:
        observation = {
            "image": np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8),
            "state": np.zeros((self.state_dim,), dtype=np.float32),
            "instruction": instruction,
        }
        start = time.perf_counter()
        action, inference_info = self.get_action(observation)
        rtc_info = None
        if self.rtc_warmup:
            prefix_steps = min(16, self.action_horizon - 1)
            if prefix_steps <= 0:
                raise ValueError("RTC warmup requires action_horizon > 1.")
            _, rtc_info = self.get_action(
                observation,
                options={
                    "rtc_prev_action_chunk": action[self.action_key][0, :prefix_steps],
                    "rtc_inference_delay": min(10, prefix_steps),
                    "rtc_prefix_horizon": prefix_steps,
                    "rtc_prefix_attention_schedule": "exp",
                    "rtc_max_guidance_weight": 5.0,
                },
            )
        info: dict[str, Any] = {
            "status": "ok",
            "warmup_s": float(time.perf_counter() - start),
            "inference": inference_info,
        }
        if rtc_info is not None:
            info["rtc_inference"] = rtc_info
        compile_status = getattr(self.model, "_compiled_hierarchical_status", None)
        if self.compile_hierarchical and callable(compile_status):
            info["compile"] = compile_status()
        if self.inference_backend == "tensorrt":
            validation = info.get("compile", {}).get("validation", {})
            trt_validation = [value for key, value in validation.items() if key.startswith("tensorrt:")]
            if not trt_validation or not all(bool(value.get("passed")) for value in trt_validation):
                raise RuntimeError(
                    "TensorRT warmup did not complete all output validation steps: "
                    f"{info.get('compile', {})}"
                )
        self.reset()
        self._warmup_info = info
        return info

    def _resolve_prompt_inputs(
        self,
        *,
        prompt: str,
        signature: dict[str, inspect.Parameter],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        encode_prompt = getattr(self.model, "encode_prompt", None)
        enabled = "context" in signature and "context_mask" in signature and callable(encode_prompt)
        cache_info = {
            "enabled": enabled,
            "hit": False,
            "hits": self._prompt_cache_hits,
            "misses": self._prompt_cache_misses,
            "encode_s": 0.0,
        }
        if not enabled:
            return {"prompt": prompt}, cache_info

        if (
            self._cached_prompt == prompt
            and self._cached_context is not None
            and self._cached_context_mask is not None
        ):
            self._prompt_cache_hits += 1
            cache_info["hit"] = True
        else:
            start = time.perf_counter()
            with torch.no_grad():
                context, context_mask = encode_prompt(prompt)
            cache_info["encode_s"] = time.perf_counter() - start
            self._cached_prompt = prompt
            self._cached_context = context.detach()
            self._cached_context_mask = context_mask.detach()
            self._prompt_cache_misses += 1

        cache_info["hits"] = self._prompt_cache_hits
        cache_info["misses"] = self._prompt_cache_misses
        return {
            "prompt": None,
            "context": self._cached_context,
            "context_mask": self._cached_context_mask,
        }, cache_info

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "video": {
                "modality_keys": [self.image_key],
                "shape": [self.input_h, self.input_w, 3],
                "dtype": "uint8",
                "accepted_shapes": ["HWC", "THWC", "BTHWC", "CHW", "TCHW", "BTCHW"],
            },
            "state": {
                "modality_keys": [self.state_key],
                "raw_dim": self.raw_state_dim,
                "model_dim": self.state_dim,
                "dtype": "float32",
                "accepted_shapes": ["D", "TD", "BTD"],
            },
            "language": {
                "modality_keys": ["task"],
                "accepted_shapes": ["str", "[[str]]"],
            },
            "action": {
                "modality_keys": [self.action_key],
                "raw_dim": self.raw_action_dim,
                "model_dim": self.action_dim,
                "horizon": self.action_horizon,
                "default_action_space": "selected",
                "dtype": "float32",
            },
        }

    def _extract_image(self, observation: dict[str, Any]) -> np.ndarray:
        if "video" in observation:
            video = observation["video"]
            if not isinstance(video, dict):
                raise TypeError("observation['video'] must be a dict of camera arrays.")
            if self.image_key in video:
                image = video[self.image_key]
            elif len(video) == 1:
                image = next(iter(video.values()))
            else:
                raise KeyError(f"Expected video key {self.image_key!r}; got keys={list(video)}.")
        elif "image" in observation:
            image = observation["image"]
        else:
            raise KeyError("Observation must include observation['video'][camera] or observation['image'].")

        arr = np.asarray(image)
        if arr.ndim == 5:
            if int(arr.shape[0]) != 1:
                raise ValueError(f"Only batch size 1 is supported for deploy, got image shape {arr.shape}.")
            arr = arr[0, -1]
        elif arr.ndim == 4:
            arr = arr[-1]
        if arr.ndim != 3:
            raise ValueError(f"Image must be HWC/CHW or temporal/batched variant, got shape {arr.shape}.")
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:
            raise ValueError(f"Image must have 3 RGB channels, got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            if arr_f.max(initial=0.0) <= 1.5:
                arr_f = arr_f * 255.0
            arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
        image_pil = Image.fromarray(arr, mode="RGB").resize((self.input_w, self.input_h), resample=Image.BILINEAR)
        return np.array(image_pil, dtype=np.uint8, copy=True)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=self.model.torch_dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def _extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        state_obj = observation.get("state")
        if isinstance(state_obj, dict):
            if self.state_key in state_obj:
                state = state_obj[self.state_key]
            elif len(state_obj) == 1:
                state = next(iter(state_obj.values()))
            else:
                raise KeyError(f"Expected state key {self.state_key!r}; got keys={list(state_obj)}.")
        elif state_obj is not None:
            state = state_obj
        elif "proprio" in observation:
            state = observation["proprio"]
        else:
            raise KeyError("Observation must include observation['state'][key], observation['state'], or observation['proprio'].")

        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 3:
            if int(arr.shape[0]) != 1:
                raise ValueError(f"Only batch size 1 is supported for deploy, got state shape {arr.shape}.")
            arr = arr[0, -1]
        elif arr.ndim == 2:
            arr = arr[-1]
        if arr.ndim != 1:
            raise ValueError(f"State must be D/TD/BTD, got shape {arr.shape}.")
        if arr.shape[0] not in {self.raw_state_dim, self.state_dim}:
            raise ValueError(
                f"State dim must be raw_dim={self.raw_state_dim} or model_dim={self.state_dim}, got {arr.shape[0]}."
            )
        return arr

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if int(state_tensor.shape[-1]) == self.raw_state_dim:
            batch = {"state": {self.state_key: state_tensor.unsqueeze(0)}}
            batch = self.processor.action_state_transform(batch)
            batch = self.processor.normalizer.forward(batch)
            return batch["state"][self.state_key]
        batch = {"state": {self.state_key: state_tensor.unsqueeze(0)}}
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self.state_key]

    def _extract_instruction(self, observation: dict[str, Any], options: dict[str, Any]) -> str:
        if not _is_none_like(options.get("instruction")):
            return str(options["instruction"])
        if not _is_none_like(observation.get("instruction")):
            return str(observation["instruction"])
        language = observation.get("language")
        if isinstance(language, dict):
            task = language.get("task")
            if isinstance(task, str):
                return task
            if isinstance(task, (list, tuple)):
                value: Any = task
                while isinstance(value, (list, tuple)) and len(value) > 0:
                    value = value[0]
                if isinstance(value, str):
                    return value
        raise KeyError("Instruction missing. Provide observation['language']['task'] or observation['instruction'].")

    def _denormalize_selected_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}.")
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _normalize_selected_action(self, action: Any) -> torch.Tensor:
        tensor = torch.as_tensor(action, dtype=torch.float32, device="cpu")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise ValueError(
                f"RTC action prefix must have shape [K,D] or [1,K,D], got {tuple(tensor.shape)}."
            )
        if int(tensor.shape[-1]) != self.action_dim:
            raise ValueError(
                f"RTC selected action dim must be {self.action_dim}, got {tensor.shape[-1]}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("RTC action prefix contains non-finite values.")
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        return normalizer.forward(tensor).to(device=self.device, dtype=self.model_dtype)

    def _selected_to_raw_action(self, selected_action: np.ndarray) -> np.ndarray:
        data = {"action": {self.action_key: torch.as_tensor(selected_action, dtype=torch.float32)}}
        for transform in reversed(self.processor.action_state_transforms or []):
            data = transform.backward(data)
        return data["action"][self.action_key].numpy()

    def _get_padded_frame_history(
        self,
        current_frame: torch.Tensor,
        action_index: Optional[int] = None,
    ) -> list[torch.Tensor]:
        if action_index is None or action_index != self._last_frame_action_index:
            self._frame_history.append(current_frame.detach())
            self._frame_history = self._frame_history[-self._max_keyframe_history:]
            self._last_frame_action_index = action_index
        frames = list(self._frame_history)
        if len(frames) < self._max_keyframe_history:
            frames = [frames[0]] * (self._max_keyframe_history - len(frames)) + frames
        return frames

    def get_action(
        self,
        observation: dict[str, Any],
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = {} if options is None else dict(options)
        image_np = self._extract_image(observation)
        image_tensor = self._image_to_tensor(image_np)
        state = self._extract_state(observation)
        proprio = self._normalize_state(state)
        instruction = self._extract_instruction(observation, options)
        prompt = DEFAULT_PROMPT.format(task=instruction)
        compile_hierarchical = bool(options.get("compile_hierarchical", self.compile_hierarchical))
        compile_cudagraphs = bool(options.get("compile_cudagraphs", self.compile_cudagraphs))
        optimize_denoise_static = bool(options.get("optimize_denoise_static", self.optimize_denoise_static))
        inference_backend = str(options.get("inference_backend", self.inference_backend)).strip().lower()
        rtc_prefix_raw = options.get("rtc_prev_action_chunk")
        rtc_active = not _is_none_like(rtc_prefix_raw)
        rtc_prefix = None
        rtc_inference_delay = 0
        rtc_prefix_horizon = 0
        rtc_schedule = str(options.get("rtc_prefix_attention_schedule", "exp")).lower()
        rtc_max_guidance_weight = float(options.get("rtc_max_guidance_weight", 5.0))
        if rtc_active:
            rtc_prefix = self._normalize_selected_action(rtc_prefix_raw)
            rtc_prefix_horizon = int(
                options.get("rtc_prefix_horizon", int(rtc_prefix.shape[1]))
            )
            rtc_inference_delay = int(options.get("rtc_inference_delay", 0))
            action_horizon = int(options.get("action_horizon", self.action_horizon))
            if not 0 < rtc_prefix_horizon <= int(rtc_prefix.shape[1]) <= action_horizon:
                raise ValueError(
                    "RTC requires 0 < prefix_horizon <= prefix length <= action_horizon."
                )
            if rtc_inference_delay < 0:
                raise ValueError("rtc_inference_delay must be non-negative.")
            if rtc_schedule not in {"exp", "linear", "ones", "zeros"}:
                raise ValueError(
                    "rtc_prefix_attention_schedule must be exp/linear/ones/zeros."
                )
            if rtc_max_guidance_weight < 0:
                raise ValueError("rtc_max_guidance_weight must be non-negative.")
            if compile_cudagraphs:
                raise ValueError(
                    "RTC gradient guidance is incompatible with CUDA Graph capture; "
                    "start the server with --no-compile-cudagraphs."
                )
            if inference_backend == "tensorrt":
                raise ValueError(
                    "RTC gradient guidance requires autograd and cannot use TensorRT."
                )

        infer_action = getattr(self.model, "infer_action")
        signature = inspect.signature(infer_action).parameters
        prompt_kwargs, cache_info = self._resolve_prompt_inputs(
            prompt=prompt,
            signature=signature,
        )
        infer_kwargs: dict[str, Any] = {
            **prompt_kwargs,
            "input_image": image_tensor,
            "action_horizon": int(options.get("action_horizon", self.action_horizon)),
            "proprio": proprio,
            "negative_prompt": str(options.get("negative_prompt", "")),
            "text_cfg_scale": float(options.get("text_cfg_scale", 1.0)),
            "num_inference_steps": int(options.get("num_inference_steps", self.num_inference_steps)),
            "sigma_shift": _optional_float(options.get("sigma_shift", self.sigma_shift)),
            "seed": None if _is_none_like(options.get("seed", self.seed)) else int(options.get("seed", self.seed)),
            "rand_device": str(options.get("rand_device", self.rand_device)),
            "tiled": bool(options.get("tiled", self.tiled)),
        }
        optional_kwargs = {
            "num_video_frames": self.num_video_frames,
            "high_video_inference_steps": options.get("high_video_inference_steps", self.high_video_inference_steps),
            "low_video_inference_steps": options.get("low_video_inference_steps", self.low_video_inference_steps),
            "high_denoise_step": options.get("high_denoise_step", self.high_denoise_step),
            "low_denoise_step": options.get("low_denoise_step", self.low_denoise_step),
            "high_reuse_step": options.get("high_reuse_step", self.high_reuse_step),
            "low_reuse_step": options.get("low_reuse_step", self.low_reuse_step),
            "action_inference_steps": options.get("action_inference_steps", self.action_inference_steps),
            "joint_denoise": bool(options.get("joint_denoise", self.joint_denoise)),
            "compile_hierarchical": compile_hierarchical,
            "compile_cudagraphs": compile_cudagraphs,
            "optimize_denoise_static": optimize_denoise_static,
            "inference_backend": inference_backend,
            "rtc_prev_action_chunk": rtc_prefix,
            "rtc_inference_delay": rtc_inference_delay,
            "rtc_prefix_horizon": rtc_prefix_horizon,
            "rtc_prefix_attention_schedule": rtc_schedule,
            "rtc_max_guidance_weight": rtc_max_guidance_weight,
        }
        for key, value in optional_kwargs.items():
            if key in signature:
                infer_kwargs[key] = value
        if "observed_chunk_videos" in signature:
            observation_action_index = options.get("observation_action_index")
            infer_kwargs["observed_chunk_videos"] = self._get_padded_frame_history(
                image_tensor[0],
                None
                if _is_none_like(observation_action_index)
                else int(observation_action_index),
            )

        start = time.perf_counter()
        with torch.no_grad():
            pred = infer_action(**infer_kwargs)
        infer_s = time.perf_counter() - start

        selected = self._denormalize_selected_action(pred["action"])[0].astype(np.float32)
        action_space = str(options.get("action_space", "selected")).lower()
        action_dict: dict[str, np.ndarray]
        raw = None
        if action_space == "raw":
            raw = self._selected_to_raw_action(selected[np.newaxis, ...])[0].astype(np.float32)
            action_dict = {self.action_key: raw[np.newaxis, ...]}
        elif action_space == "selected":
            action_dict = {self.action_key: selected[np.newaxis, ...]}
            if bool(options.get("return_raw_action", False)):
                raw = self._selected_to_raw_action(selected[np.newaxis, ...])[0].astype(np.float32)
                action_dict[f"{self.action_key}_raw"] = raw[np.newaxis, ...]
        else:
            raise ValueError("options['action_space'] must be 'selected' or 'raw'.")

        info = {
            "instruction": instruction,
            "prompt": prompt,
            "inference_s": float(infer_s),
            "action_space": action_space,
            "action_key": self.action_key,
            "selected_action_dim": self.action_dim,
            "raw_action_dim": self.raw_action_dim,
            "action_horizon": int(selected.shape[0]),
            "replan_steps": self.replan_steps,
            "model_variant": "hierarchical" if self.is_hierarchical_model else "native",
            "image_key": self.image_key,
            "state_key": self.state_key,
            "prompt_cache": cache_info,
            "compile_hierarchical": compile_hierarchical,
            "compile_cudagraphs": compile_cudagraphs,
            "optimize_denoise_static": optimize_denoise_static,
            "inference_backend": inference_backend,
            "rtc": {
                "enabled": rtc_active,
                "prefix_steps": 0 if rtc_prefix is None else int(rtc_prefix.shape[1]),
                "inference_delay_steps": rtc_inference_delay,
                "prefix_horizon": rtc_prefix_horizon,
                "prefix_attention_schedule": rtc_schedule,
                "max_guidance_weight": rtc_max_guidance_weight,
                "guidance_phase": "action_only",
            },
        }
        compile_status = getattr(self.model, "_compiled_hierarchical_status", None)
        if compile_hierarchical and callable(compile_status):
            info["compile"] = compile_status()
        return action_dict, info
