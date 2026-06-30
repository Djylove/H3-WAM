import logging
import atexit
import ast
import json
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _parse_optional_int_list(value: Any) -> Optional[list[int]]:
    if _is_none_like(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            value = ast.literal_eval(text)
        else:
            value = [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    return [int(value)]


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
    sim_cfg_overrides: Optional[Any] = None,
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    if not _is_none_like(sim_cfg_overrides):
        if isinstance(sim_cfg_overrides, (list, tuple)):
            overrides.extend(str(item) for item in sim_cfg_overrides)
        else:
            overrides.append(str(sim_cfg_overrides))

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _safe_name(text: str, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return (cleaned or "item")[:max_len]


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        high_video_inference_steps: Optional[int],
        low_video_inference_steps: Optional[int],
        high_denoise_step: Optional[int],
        low_denoise_step: Optional[int],
        high_reuse_step: Optional[int],
        low_reuse_step: Optional[int],
        action_inference_steps: Optional[int],
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        joint_denoise: bool,
        timing_enabled: bool,
        num_video_frames: int,
        attention_viz_enabled: bool = False,
        attention_viz_output_dir: Optional[Path] = None,
        attention_viz_steps: Optional[list[int]] = None,
        attention_viz_layers: Optional[list[int]] = None,
        attention_viz_max_plans: int = 1,
        attention_viz_max_records: int = 128,
        attention_viz_query_chunk_size: int = 256,
        attention_viz_alpha: float = 0.55,
        attention_viz_video_fps: int = 4,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self._num_video_frames = int(num_video_frames)
        self._is_hierarchical_model = callable(getattr(self.model, "infer_hierarchical", None))
        # Sparse post-action keyframes sampled from the current action horizon.
        self._pending_keyframes: list[torch.Tensor] = []
        self._keyframe_history: list[torch.Tensor] = []
        # Observation frame at the start of the current action horizon.
        self._horizon_start_frame: Optional[torch.Tensor] = None
        self._keyframe_stride = 16
        self._max_keyframe_history: int = 9
        self._last_action_executed: bool = False
        self._horizon_executed_actions: int = 0

        self.action_horizon = int(action_horizon)
        if self._is_hierarchical_model:
            model_horizon = int(getattr(self.model, "hierarchical_action_horizon", self.action_horizon))
            if self.action_horizon != model_horizon:
                logger.warning(
                    "Hierarchical policy requires action_horizon=%d, but got %d. Overriding.",
                    model_horizon,
                    self.action_horizon,
                )
                self.action_horizon = model_horizon
        self.replan_steps = int(max(1, min(replan_steps, self.action_horizon)))
        if self._is_hierarchical_model and self.replan_steps != self.action_horizon:
            logger.warning(
                "Hierarchical policy expects full action-horizon execution with replan_steps=action_horizon=%d, but got %d. Overriding.",
                self.action_horizon,
                self.replan_steps,
            )
            self.replan_steps = self.action_horizon
        self.num_inference_steps = int(num_inference_steps)
        self.high_video_inference_steps = high_video_inference_steps
        self.low_video_inference_steps = low_video_inference_steps
        self.high_denoise_step = high_denoise_step
        self.low_denoise_step = low_denoise_step
        self.high_reuse_step = high_reuse_step
        self.low_reuse_step = low_reuse_step
        self.action_inference_steps = action_inference_steps
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.joint_denoise = bool(joint_denoise)
        self.timing_enabled = bool(timing_enabled)
        self.attention_viz_enabled = bool(attention_viz_enabled)
        self.attention_viz_output_dir = attention_viz_output_dir
        if self.attention_viz_enabled:
            if self.attention_viz_output_dir is None:
                self.attention_viz_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin_attention_viz"
            self.attention_viz_output_dir.mkdir(parents=True, exist_ok=True)
        self.attention_viz_steps = attention_viz_steps
        self.attention_viz_layers = attention_viz_layers
        self.attention_viz_max_plans = int(attention_viz_max_plans)
        self.attention_viz_max_records = max(1, int(attention_viz_max_records))
        self.attention_viz_query_chunk_size = max(1, int(attention_viz_query_chunk_size))
        self.attention_viz_alpha = float(attention_viz_alpha)
        self.attention_viz_video_fps = max(1, int(attention_viz_video_fps))
        self._attention_viz_plan_count = 0
        self._attention_rollout_frames: list[Image.Image] = []
        self._attention_rollout_records: list[dict[str, Any]] = []
        self._active_attention_heatmaps: dict[str, tuple[np.ndarray, dict[str, Any]]] | None = None
        if self.attention_viz_enabled:
            atexit.register(self._flush_attention_rollout_video)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )
        logger.info(
            "FastWAM hierarchical cfg | mask_high=%s | mask_low=%s | joint_denoise=%s | high_reuse_step=%s | low_reuse_step=%s",
            getattr(self.model, "hierarchical_mask_high_predict", None),
            getattr(self.model, "hierarchical_mask_low_predict", None),
            self.joint_denoise,
            self.high_reuse_step,
            self.low_reuse_step,
        )
        if self.attention_viz_enabled:
            logger.info(
                "Attention visualization enabled | output=%s | steps=%s | layers=%s | max_plans=%d | video_fps=%d",
                self.attention_viz_output_dir,
                self.attention_viz_steps,
                self.attention_viz_layers,
                self.attention_viz_max_plans,
                self.attention_viz_video_fps,
            )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_np(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

    def _image_np_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self._image_np_to_tensor(self._build_robotwin_image_np(observation))

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        low = float(np.percentile(heatmap, 1.0))
        high = float(np.percentile(heatmap, 99.0))
        if high <= low:
            high = float(heatmap.max())
            low = float(heatmap.min())
        if high <= low:
            return np.zeros_like(heatmap, dtype=np.float32)
        return np.clip((heatmap - low) / (high - low), 0.0, 1.0)

    @staticmethod
    def _attention_colormap(norm: np.ndarray) -> np.ndarray:
        norm = np.asarray(norm, dtype=np.float32).clip(0.0, 1.0)
        stops = np.asarray(
            [
                [48, 34, 121],
                [38, 109, 157],
                [42, 176, 127],
                [253, 231, 37],
                [190, 0, 38],
            ],
            dtype=np.float32,
        )
        scaled = norm * (len(stops) - 1)
        idx = np.floor(scaled).astype(np.int32).clip(0, len(stops) - 2)
        frac = (scaled - idx)[..., None]
        rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
        return rgb.astype(np.uint8)

    def _attention_overlay_image(
        self,
        *,
        base_image: np.ndarray,
        heatmap: np.ndarray,
        title: str,
    ) -> Image.Image:
        norm = self._normalize_heatmap(heatmap)
        rgba = np.zeros((*norm.shape, 4), dtype=np.uint8)
        rgba[..., :3] = self._attention_colormap(norm)
        rgba[..., 3] = ((0.18 + 0.72 * norm) * 255.0 * self.attention_viz_alpha).clip(0, 255).astype(np.uint8)

        base = Image.fromarray(base_image.astype(np.uint8), mode="RGB").convert("RGBA")
        heat = Image.fromarray(rgba, mode="RGBA").resize(base.size, resample=Image.BILINEAR)
        out = Image.alpha_composite(base, heat)
        draw = ImageDraw.Draw(out)
        draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0, 160))
        draw.text((6, 4), title, fill=(255, 255, 255, 255))
        return out.convert("RGB")

    @staticmethod
    def _plain_image_with_title(base_image: np.ndarray, title: str) -> Image.Image:
        out = Image.fromarray(base_image.astype(np.uint8), mode="RGB").copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0))
        draw.text((6, 4), title, fill=(255, 255, 255))
        return out

    @staticmethod
    def _aggregate_attention_record(record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
        attention = record.get("attention")
        if isinstance(attention, torch.Tensor):
            attention_np = attention.detach().float().cpu().numpy()
        else:
            attention_np = np.asarray(attention, dtype=np.float32)
        if attention_np.ndim != 2:
            return None
        grid_h, grid_w = [int(v) for v in record.get("grid_size", (0, 0))]
        if grid_h <= 0 or grid_w <= 0 or attention_np.shape[1] != grid_h * grid_w:
            return None
        heatmap = attention_np.max(axis=0).reshape(grid_h, grid_w)
        meta = {
            "relation": str(record.get("relation", "attention")),
            "target_kind": str(record.get("target_kind", "")),
            "phase": str(record.get("phase", "")),
            "step_idx": int(record.get("step_idx", -1)),
            "total_steps": int(record.get("total_steps", -1)),
            "layer_idx": int(record.get("layer_idx", -1)),
            "attention_score_sum": float(record.get("attention_score_sum", float(attention_np.sum()))),
            "score_aggregation": str(record.get("score_aggregation", "max_over_action_queries_heads")),
            "grid_size": [grid_h, grid_w],
            "num_frames": int(attention_np.shape[0]),
            "frame_indices": [int(v) for v in list(record.get("frame_indices", []))],
            "aggregation": "max_over_target_frames",
        }
        return heatmap, meta

    def _attention_plan_allowed(self) -> bool:
        return self.attention_viz_max_plans < 0 or self._attention_viz_plan_count < self.attention_viz_max_plans

    def _select_rollout_attention_heatmaps(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
        selected: dict[str, tuple[tuple[int, int, int], np.ndarray, dict[str, Any]]] = {}
        for record in records[: self.attention_viz_max_records]:
            aggregated = self._aggregate_attention_record(record)
            if aggregated is None:
                continue
            heatmap, meta = aggregated
            relation = str(meta["relation"])
            # Prefer later denoising steps and later layers for the rollout video.
            rank = (int(meta["step_idx"]), int(meta["layer_idx"]), len(selected))
            prev = selected.get(relation)
            if prev is None or rank >= prev[0]:
                selected[relation] = (rank, heatmap, meta)
        return {relation: (heatmap, meta) for relation, (_rank, heatmap, meta) in selected.items()}

    def _make_attention_rollout_frame(
        self,
        *,
        base_image: np.ndarray,
        heatmaps: dict[str, tuple[np.ndarray, dict[str, Any]]],
        source: str,
    ) -> Image.Image:
        tile_specs = [
            ("Image", None),
            ("Action -> High", "action_to_high"),
            ("Action -> Low", "action_to_low"),
            ("Low -> High", "low_to_high"),
        ]

        tiles: list[Image.Image] = []
        for label, relation in tile_specs:
            if relation is None:
                image = self._plain_image_with_title(base_image, f"{label} | {source}")
            elif relation in heatmaps:
                heatmap, meta = heatmaps[relation]
                image = self._attention_overlay_image(
                    base_image=base_image,
                    heatmap=heatmap,
                    title=f"{label} | s{int(meta['step_idx'])} l{int(meta['layer_idx'])}",
                )
            else:
                image = self._plain_image_with_title(base_image, f"{label} | no map")
            tiles.append(image)

        w, h = tiles[0].size
        panel = Image.new("RGB", (w * 2, h * 2), color=(255, 255, 255))
        panel.paste(tiles[0], (0, 0))
        panel.paste(tiles[1], (w, 0))
        panel.paste(tiles[2], (0, h))
        panel.paste(tiles[3], (w, h))
        return panel

    def _append_attention_rollout_frame(
        self,
        *,
        base_image: np.ndarray,
        source: str,
        heatmaps: dict[str, tuple[np.ndarray, dict[str, Any]]] | None = None,
    ) -> None:
        if not self.attention_viz_enabled:
            return
        if heatmaps is None:
            heatmaps = self._active_attention_heatmaps
        if not heatmaps:
            return
        frame = self._make_attention_rollout_frame(
            base_image=base_image,
            heatmaps=heatmaps,
            source=source,
        )
        self._attention_rollout_frames.append(frame)
        self._attention_rollout_records.append(
            {
                "episode": int(self.episode_count),
                "step": int(self.step_count),
                "source": str(source),
                "relations": {
                    relation: {
                        key: value
                        for key, value in meta.items()
                        if key != "attention"
                    }
                    for relation, (_heatmap, meta) in heatmaps.items()
                },
            }
        )

    def _flush_attention_rollout_video(self) -> None:
        if not self.attention_viz_enabled or not self._attention_rollout_frames:
            return
        if self.attention_viz_output_dir is None:
            return
        episode_dir = self.attention_viz_output_dir / f"episode_{self.episode_count:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_path = episode_dir / "attention_rollout_2x2.mp4"
        save_mp4(self._attention_rollout_frames, str(video_path), fps=self.attention_viz_video_fps)
        metadata_path = episode_dir / "attention_rollout_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "episode": int(self.episode_count),
                    "fps": int(self.attention_viz_video_fps),
                    "num_frames": len(self._attention_rollout_frames),
                    "layout": [
                        "Image",
                        "Action -> High",
                        "Action -> Low",
                        "Low -> High",
                    ],
                    "records": self._attention_rollout_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _save_attention_panel(*, rows: list[tuple[str, Image.Image]], title: str, output_path: Path) -> None:
        if not rows:
            return
        label_w = 118
        title_h = 30
        gap = 4
        image_w, image_h = rows[0][1].size
        panel_w = label_w + image_w
        panel_h = title_h + len(rows) * image_h + (len(rows) - 1) * gap
        panel = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_w, title_h), fill=(255, 255, 255))
        draw.text((8, 7), title, fill=(20, 50, 90))
        y = title_h
        for label, image in rows:
            draw.rectangle((0, y, label_w, y + image_h), fill=(245, 248, 252))
            draw.text((10, y + max(8, image_h // 2 - 8)), label, fill=(20, 50, 90))
            panel.paste(image, (label_w, y))
            y += image_h + gap
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(output_path)

    def _save_attention_visualizations(
        self,
        *,
        records: list[dict[str, Any]],
        base_image: np.ndarray,
        instruction: str,
    ) -> None:
        if not self.attention_viz_enabled or not records:
            return
        if not self._attention_plan_allowed():
            return
        if self.attention_viz_output_dir is None:
            return

        plan_dir = (
            self.attention_viz_output_dir
            / f"episode_{self.episode_count:03d}"
            / f"plan_{self._attention_viz_plan_count:04d}_step_{self.step_count:05d}"
        )
        plan_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[tuple[str, int, int], dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
        for record in records[: self.attention_viz_max_records]:
            aggregated = self._aggregate_attention_record(record)
            if aggregated is None:
                continue
            heatmap, meta = aggregated
            key = (str(meta["phase"]), int(meta["step_idx"]), int(meta["layer_idx"]))
            grouped.setdefault(key, {})[str(meta["relation"])] = (heatmap, meta)

        metadata = []
        relation_rows = [
            ("action_to_high", "Action -> High"),
            ("action_to_low", "Action -> Low"),
            ("low_to_high", "Low -> High"),
        ]
        for panel_idx, ((phase, step_idx, layer_idx), relation_map) in enumerate(sorted(grouped.items())):
            title = f"{phase} | denoise step {step_idx} | layer {layer_idx}"
            rows: list[tuple[str, Image.Image]] = [
                ("Image", self._plain_image_with_title(base_image, "Image")),
            ]
            panel_records = []
            for relation, label in relation_rows:
                item = relation_map.get(relation)
                if item is None:
                    continue
                heatmap, meta = item
                rows.append(
                    (
                        label,
                        self._attention_overlay_image(
                            base_image=base_image,
                            heatmap=heatmap,
                            title=f"{label} | score={float(meta['attention_score_sum']):.4f}",
                        ),
                    )
                )
                panel_records.append(meta)
            if len(rows) <= 1:
                continue
            panel_path = plan_dir / f"panel_{panel_idx:03d}_{_safe_name(phase)}_s{step_idx:03d}_l{layer_idx:02d}.png"
            self._save_attention_panel(rows=rows, title=title, output_path=panel_path)
            metadata.append(
                {
                    "panel_path": panel_path.name,
                    "phase": phase,
                    "step_idx": step_idx,
                    "layer_idx": layer_idx,
                    "records": panel_records,
                }
            )
        (plan_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "instruction": instruction,
                    "episode": self.episode_count,
                    "step": self.step_count,
                    "records": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._attention_viz_plan_count += 1

    def _get_padded_keyframe_history(self) -> list[torch.Tensor]:
        frames = list(self._keyframe_history[-self._max_keyframe_history:])
        if len(frames) == 0:
            if self._horizon_start_frame is None:
                return []
            frames = [self._horizon_start_frame]
        if len(frames) < self._max_keyframe_history:
            frames = [frames[0]] * (self._max_keyframe_history - len(frames)) + frames
        return frames

    def _infer_action_horizon(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_np = self._build_robotwin_image_np(observation)
        image_tensor = self._image_np_to_tensor(image_np)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_action_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }

        infer_action_fn = getattr(self.model, "infer_action", None)
        if not callable(infer_action_fn):
            raise AttributeError("Model must provide `infer_action` for policy inference.")
        infer_action_sig = inspect.signature(infer_action_fn).parameters
        if "num_video_frames" in infer_action_sig:
            infer_action_kwargs["num_video_frames"] = int(self._num_video_frames)
        if "high_video_inference_steps" in infer_action_sig:
            infer_action_kwargs["high_video_inference_steps"] = self.high_video_inference_steps
        if "low_video_inference_steps" in infer_action_sig:
            infer_action_kwargs["low_video_inference_steps"] = self.low_video_inference_steps
        if "high_denoise_step" in infer_action_sig:
            infer_action_kwargs["high_denoise_step"] = self.high_denoise_step
        if "low_denoise_step" in infer_action_sig:
            infer_action_kwargs["low_denoise_step"] = self.low_denoise_step
        if "high_reuse_step" in infer_action_sig:
            infer_action_kwargs["high_reuse_step"] = self.high_reuse_step
        if "low_reuse_step" in infer_action_sig:
            infer_action_kwargs["low_reuse_step"] = self.low_reuse_step
        if "action_inference_steps" in infer_action_sig:
            infer_action_kwargs["action_inference_steps"] = self.action_inference_steps
        if "joint_denoise" in infer_action_sig:
            infer_action_kwargs["joint_denoise"] = self.joint_denoise
        if (
            "attention_viz" in infer_action_sig
            and self.attention_viz_enabled
            and self._attention_plan_allowed()
        ):
            infer_action_kwargs["attention_viz"] = {
                "enabled": True,
                "steps": self.attention_viz_steps,
                "layers": self.attention_viz_layers,
                "max_records": self.attention_viz_max_records,
                "query_chunk_size": self.attention_viz_query_chunk_size,
            }
        if self._is_hierarchical_model and "observed_chunk_videos" in infer_action_sig:
            keyframe_history = self._get_padded_keyframe_history()
            if len(keyframe_history) > 0:
                infer_action_kwargs["observed_chunk_videos"] = keyframe_history

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = infer_action_fn(**infer_action_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        attention_records = list(pred.get("attention_maps", []))
        rollout_plan_idx = self._attention_viz_plan_count
        self._save_attention_visualizations(
            records=attention_records,
            base_image=image_np,
            instruction=instruction,
        )
        heatmaps = self._select_rollout_attention_heatmaps(attention_records)
        if heatmaps:
            self._active_attention_heatmaps = heatmaps
            self._append_attention_rollout_frame(
                base_image=image_np,
                source=f"replan_{rollout_plan_idx:04d}",
                heatmaps=heatmaps,
            )

        # Start a new action-horizon rollout window for sparse post-action keyframes.
        self._pending_keyframes.clear()
        self._horizon_executed_actions = 0

        action_tensor = pred["action"]  # [T, D]
        action_horizon_pred = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_horizon_pred

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_horizon_pred = self._infer_action_horizon(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_horizon_pred.shape[0])
        if self._is_hierarchical_model:
            logger.info(
                "Hierarchical rollout | memory_frames=%d | exec_steps=%d",
                len(self._keyframe_history),
                n_exec,
            )
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_horizon_pred[i], dtype=np.float32))

    def _finalize_keyframe_history(self, current_frame: torch.Tensor) -> None:
        if not self._is_hierarchical_model:
            self._pending_keyframes.clear()
            return
        if self._horizon_start_frame is None or len(self._pending_keyframes) == 0:
            self._pending_keyframes.clear()
            self._horizon_executed_actions = 0
            return

        if self._horizon_executed_actions != self.action_horizon:
            logger.warning(
                "Expected %d executed actions per horizon, got %d.",
                self.action_horizon,
                self._horizon_executed_actions,
            )

        if len(self._keyframe_history) == 0:
            self._keyframe_history.append(self._horizon_start_frame)
        for frame in self._pending_keyframes:
            if not torch.equal(self._keyframe_history[-1], frame):
                self._keyframe_history.append(frame)
        self._keyframe_history = self._keyframe_history[-self._max_keyframe_history:]

        self._pending_keyframes.clear()
        self._horizon_executed_actions = 0
        self._horizon_start_frame = current_frame

    def should_request_observation(self) -> bool:
        if self._is_hierarchical_model:
            if not self.pending_actions:
                return True
            return bool(
                self._last_action_executed
                and self._horizon_executed_actions > 0
                and self._horizon_executed_actions % self._keyframe_stride == 0
            )
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        frame: Optional[torch.Tensor] = None
        image_np: Optional[np.ndarray] = None
        if self._is_hierarchical_model and observation is not None:
            image_np = self._build_robotwin_image_np(observation)
            frame = self._image_np_to_tensor(image_np)[0].detach()

            # Observation at this step is the post-action frame of previous step.
            if self._last_action_executed:
                if (
                    self._horizon_executed_actions > 0
                    and self._horizon_executed_actions % self._keyframe_stride == 0
                ):
                    self._pending_keyframes.append(frame)
                self._last_action_executed = False
            if self.pending_actions and image_np is not None:
                self._append_attention_rollout_frame(
                    base_image=image_np,
                    source=f"chunk_step_{self.step_count:05d}",
                )

        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            if self._is_hierarchical_model and self.step_count > 0:
                if frame is None:
                    raise ValueError("Hierarchical policy requires current frame when finalizing keyframe history.")
                self._finalize_keyframe_history(current_frame=frame)
            if self._is_hierarchical_model:
                if frame is None:
                    raise ValueError("Hierarchical policy requires current frame for planning.")
                if self._horizon_start_frame is None:
                    self._horizon_start_frame = frame
                if len(self._keyframe_history) == 0:
                    self._keyframe_history.append(self._horizon_start_frame)
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self._is_hierarchical_model:
            self._last_action_executed = True
            self._horizon_executed_actions += 1
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self._flush_attention_rollout_video()
        self.pending_actions.clear()
        self._pending_keyframes.clear()
        self._keyframe_history.clear()
        clear_reuse_cache = getattr(self.model, "clear_hierarchical_reuse_cache", None)
        if callable(clear_reuse_cache):
            clear_reuse_cache()
        self._horizon_start_frame = None
        self._last_action_executed = False
        self._horizon_executed_actions = 0
        self.episode_count += 1
        self._attention_viz_plan_count = 0
        self._attention_rollout_frames.clear()
        self._attention_rollout_records.clear()
        self._active_attention_heatmaps = None
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    sim_cfg_overrides = usr_args.get("sim_cfg_overrides")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
        sim_cfg_overrides=sim_cfg_overrides,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    high_video_inference_steps = _parse_optional_int(usr_args.get("high_video_inference_steps"))
    if high_video_inference_steps is None:
        high_video_inference_steps = _parse_optional_int(cfg.EVALUATION.get("high_video_inference_steps"))
    low_video_inference_steps = _parse_optional_int(usr_args.get("low_video_inference_steps"))
    if low_video_inference_steps is None:
        low_video_inference_steps = _parse_optional_int(cfg.EVALUATION.get("low_video_inference_steps"))
    high_denoise_step = _parse_optional_int(usr_args.get("high_denoise_step"))
    if high_denoise_step is None:
        high_denoise_step = _parse_optional_int(cfg.EVALUATION.get("high_denoise_step"))
    low_denoise_step = _parse_optional_int(usr_args.get("low_denoise_step"))
    if low_denoise_step is None:
        low_denoise_step = _parse_optional_int(cfg.EVALUATION.get("low_denoise_step"))
    high_reuse_step = _parse_optional_int(usr_args.get("high_reuse_step"))
    if high_reuse_step is None:
        high_reuse_step = _parse_optional_int(cfg.EVALUATION.get("high_reuse_step"))
    low_reuse_step = _parse_optional_int(usr_args.get("low_reuse_step"))
    if low_reuse_step is None:
        low_reuse_step = _parse_optional_int(cfg.EVALUATION.get("low_reuse_step"))
    action_inference_steps = _parse_optional_int(usr_args.get("action_inference_steps"))
    if action_inference_steps is None:
        action_inference_steps = _parse_optional_int(cfg.EVALUATION.get("action_inference_steps"))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    joint_denoise = _parse_bool(usr_args.get("joint_denoise", cfg.EVALUATION.get("joint_denoise", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    attention_viz_enabled = _parse_bool(
        usr_args.get("attention_viz_enabled", cfg.EVALUATION.get("attention_viz_enabled", False))
    )
    eval_output_dir = usr_args.get("eval_output_dir", None)
    attention_viz_output = usr_args.get(
        "attention_viz_output_dir",
        cfg.EVALUATION.get("attention_viz_output_dir", None),
    )
    if _is_none_like(attention_viz_output) and not _is_none_like(eval_output_dir):
        attention_viz_output = str(Path(str(eval_output_dir)).expanduser().resolve() / "attention_viz")
    attention_viz_output_dir = None if _is_none_like(attention_viz_output) else Path(str(attention_viz_output)).expanduser().resolve()
    attention_viz_steps = _parse_optional_int_list(
        usr_args.get("attention_viz_steps", cfg.EVALUATION.get("attention_viz_steps", None))
    )
    attention_viz_layers = _parse_optional_int_list(
        usr_args.get("attention_viz_layers", cfg.EVALUATION.get("attention_viz_layers", [-1]))
    )
    attention_viz_max_plans = int(
        usr_args.get("attention_viz_max_plans", cfg.EVALUATION.get("attention_viz_max_plans", 1))
    )
    attention_viz_max_records = int(
        usr_args.get("attention_viz_max_records", cfg.EVALUATION.get("attention_viz_max_records", 128))
    )
    attention_viz_query_chunk_size = int(
        usr_args.get("attention_viz_query_chunk_size", cfg.EVALUATION.get("attention_viz_query_chunk_size", 256))
    )
    attention_viz_alpha = float(
        usr_args.get("attention_viz_alpha", cfg.EVALUATION.get("attention_viz_alpha", 0.55))
    )
    attention_viz_video_fps = int(
        usr_args.get("attention_viz_video_fps", cfg.EVALUATION.get("attention_viz_video_fps", 4))
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        high_video_inference_steps=high_video_inference_steps,
        low_video_inference_steps=low_video_inference_steps,
        high_denoise_step=high_denoise_step,
        low_denoise_step=low_denoise_step,
        high_reuse_step=high_reuse_step,
        low_reuse_step=low_reuse_step,
        action_inference_steps=action_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        joint_denoise=joint_denoise,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        attention_viz_enabled=attention_viz_enabled,
        attention_viz_output_dir=attention_viz_output_dir,
        attention_viz_steps=attention_viz_steps,
        attention_viz_layers=attention_viz_layers,
        attention_viz_max_plans=attention_viz_max_plans,
        attention_viz_max_records=attention_viz_max_records,
        attention_viz_query_chunk_size=attention_viz_query_chunk_size,
        attention_viz_alpha=attention_viz_alpha,
        attention_viz_video_fps=attention_viz_video_fps,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
