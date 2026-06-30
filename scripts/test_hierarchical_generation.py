import html
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils import misc
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

register_default_resolvers()
logger = get_logger(__name__)


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(v) for v in value]
    return [int(value)]


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    path = cfg.test.get("checkpoint_path")
    if not path:
        raise ValueError("Please provide `test.checkpoint_path=/path/to/checkpoint.pt`.")
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _infer_stats_path(cfg: DictConfig, checkpoint_path: str) -> str | None:
    explicit = cfg.test.get("dataset_stats_path")
    if explicit not in (None, "", "null"):
        return str(explicit)

    val_stats = cfg.data.val.get("pretrained_norm_stats") if cfg.data.get("val") is not None else None
    if val_stats not in (None, "", "null"):
        return str(val_stats)

    train_stats = cfg.data.train.get("pretrained_norm_stats") if cfg.data.get("train") is not None else None
    if train_stats not in (None, "", "null"):
        return str(train_stats)

    ckpt = Path(checkpoint_path).resolve()
    for parent in [ckpt.parent, *ckpt.parents]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return str(candidate)
    return None


def _save_eval_style_video(tensors: list[torch.Tensor], path: str, fps: int):
    if not tensors:
        raise ValueError("Expected at least one video tensor to save.")
    base_shape = tuple(tensors[0].shape)
    for tensor in tensors:
        if tuple(tensor.shape) != base_shape:
            raise ValueError(
                "Eval-style visualization shape mismatch: "
                f"got {tuple(tensor.shape)}, expected {base_shape}"
            )
    stitched = torch.cat([tensor.detach().float().cpu() for tensor in tensors], dim=2).contiguous()
    frames = []
    for t in range(stitched.shape[1]):
        frame = (
            stitched[:, t]
            .permute(1, 2, 0)
            .clamp(0.0, 1.0)
            .numpy()
            * 255.0
        ).astype(np.uint8)
        frames.append(Image.fromarray(frame))
    save_mp4(frames, path, fps=fps)


def _expected_video_hw(cfg: DictConfig) -> tuple[int, int] | None:
    size = cfg.data.val.get("video_size")
    if size in (None, "", "null"):
        return None
    if len(size) != 2:
        raise ValueError(f"Expected data.val.video_size=[H,W], got {size}")
    return int(size[0]), int(size[1])


def _assert_dataset_video_shape(sample: dict[str, Any], expected_hw: tuple[int, int] | None):
    if expected_hw is None:
        return
    video = sample.get("video")
    if not isinstance(video, torch.Tensor):
        return
    actual_hw = tuple(int(v) for v in video.shape[-2:])
    if actual_hw != tuple(expected_hw):
        raise ValueError(
            "Dataset preprocessing output shape does not match data.val.video_size: "
            f"got HxW={actual_hw}, expected HxW={expected_hw}. "
            "Check that this script is using the same data config/transforms as training evaluate."
        )


def _sample_indices(dataset_len: int, cfg: DictConfig) -> list[int]:
    explicit = _as_int_list(cfg.test.get("sample_indices"), default=[])
    if explicit:
        for idx in explicit:
            if idx < 0 or idx >= dataset_len:
                raise IndexError(f"sample index {idx} out of bounds for val dataset length {dataset_len}")
        return explicit[: int(cfg.test.num_samples)]

    rng = np.random.default_rng(int(cfg.test.seed))
    count = min(int(cfg.test.num_samples), dataset_len)
    if bool(cfg.test.get("random_samples", True)):
        return [int(v) for v in rng.choice(dataset_len, size=count, replace=False)]
    return list(range(count))


def _get_eval_inputs(sample: dict, model) -> dict[str, Any]:
    prompt = sample["prompt"]
    video = sample["video"]
    keyframe = sample.get("keyframe")
    action = sample.get("action")
    proprio = sample.get("proprio")

    input_image = video[:, 0].unsqueeze(0)
    _, num_frames, _, _ = video.shape
    is_hierarchical_model = callable(getattr(model, "infer_hierarchical", None))
    if proprio is not None and not is_hierarchical_model:
        proprio = proprio[0]

    infer_kwargs = {
        "input_image": input_image,
        "num_frames": int(num_frames),
        "action": action,
        "action_horizon": int(action.shape[0]) if action is not None else None,
        "proprio": proprio,
        "text_cfg_scale": 1.0,
        "action_cfg_scale": 1.0,
        "seed": 42,
        "tiled": False,
    }
    if sample.get("context") is not None:
        infer_kwargs["prompt"] = None
        infer_kwargs["context"] = sample["context"]
        infer_kwargs["context_mask"] = sample["context_mask"]
    else:
        infer_kwargs["prompt"] = prompt

    if is_hierarchical_model:
        infer_kwargs["gt_video"] = video
        if keyframe is not None:
            infer_kwargs["gt_keyframe"] = keyframe
        infer_kwargs["high_denoise_step"] = None
        infer_kwargs["low_denoise_step"] = None
    if "return_high_level_video" in inspect.signature(model.infer).parameters:
        infer_kwargs["return_high_level_video"] = True
    return infer_kwargs


def _evaluate_prediction(pred_frames: list[Image.Image], gt_video: torch.Tensor) -> dict[str, float]:
    pred_tensor = pil_frames_to_video_tensor(pred_frames)
    gt_tensor = ((gt_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    return {
        "psnr": video_psnr(pred_tensor, gt_tensor),
        "ssim": video_ssim(pred_tensor, gt_tensor),
    }


def _write_html(results: list[dict[str, Any]], output_dir: Path):
    rows = []
    for item in results:
        high_cell = ""
        if item.get("high_video_path"):
            high_cell = f'<video controls loop muted src="{html.escape(item["high_video_path"])}"></video>'
        rows.append(
            "<tr>"
            f"<td>{item['sample_index']}</td>"
            f"<td>{item['high_steps']}</td>"
            f"<td>{item['low_steps']}</td>"
            f"<td>{item.get('psnr', ''):.4f}</td>"
            f"<td>{item.get('ssim', ''):.4f}</td>"
            f'<td><video controls loop muted src="{html.escape(item["low_video_path"])}"></video></td>'
            f"<td>{high_cell}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hierarchical Generation Test</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    video {{ max-width: 420px; }}
  </style>
</head>
<body>
  <h1>Hierarchical Generation Test</h1>
  <table>
    <thead>
      <tr><th>sample</th><th>high steps</th><th>low steps</th><th>PSNR</th><th>SSIM</th><th>low pred/gt</th><th>high pred/gt</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)

    checkpoint_path = _resolve_checkpoint_path(cfg)
    output_dir = Path(str(cfg.test.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))

    stats_path = _infer_stats_path(cfg, checkpoint_path)
    if stats_path is None:
        raise ValueError(
            "Could not infer dataset stats. Pass `test.dataset_stats_path=...` "
            "or `data.val.pretrained_norm_stats=...`."
        )
    OmegaConf.update(cfg, "data.val.pretrained_norm_stats", stats_path, merge=False, force_add=True)

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = str(cfg.test.device)

    logger.info("Loading model on %s with dtype=%s", device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model.load_checkpoint(checkpoint_path)
    model.eval()

    logger.info("Building validation/test dataset with stats: %s", stats_path)
    dataset = instantiate(cfg.data.val, pretrained_norm_stats=stats_path)
    indices = _sample_indices(len(dataset), cfg)
    expected_hw = _expected_video_hw(cfg)
    logger.info(
        "Using training-evaluate preprocessing from data.val; expected video HxW=%s",
        expected_hw,
    )
    high_steps_list = _as_int_list(cfg.test.high_video_inference_steps, default=[10])
    low_steps_list = _as_int_list(cfg.test.low_video_inference_steps, default=[10])
    num_inference_steps = int(cfg.test.num_inference_steps)
    fps = int(cfg.test.fps)
    if cfg.test.get("action_inference_steps") not in (None, "", "null"):
        logger.warning(
            "`test.action_inference_steps` is ignored in this script. "
            "To match training evaluate, action_inference_steps is tied to low_video_inference_steps."
        )

    OmegaConf.save(config=cfg, f=str(output_dir / "config.yaml"), resolve=True)

    results = []
    with torch.no_grad():
        for sample_index in indices:
            raw_sample = dataset[int(sample_index)]
            _assert_dataset_video_shape(raw_sample, expected_hw)
            prompt_path = output_dir / f"sample_{sample_index:06d}_prompt.txt"
            prompt_path.write_text(str(raw_sample.get("prompt", "")), encoding="utf-8")

            sample = {
                key: value.to(device=model.device, dtype=model.torch_dtype) if isinstance(value, torch.Tensor) and key in {"video", "keyframe", "action", "proprio", "context"} else value
                for key, value in raw_sample.items()
            }
            if isinstance(raw_sample.get("context_mask"), torch.Tensor):
                sample["context_mask"] = raw_sample["context_mask"].to(device=model.device, dtype=torch.bool)

            gt_video_tensor = ((raw_sample["video"].detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
            gt_keyframe_tensor = (
                ((raw_sample["keyframe"].detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
                if raw_sample.get("keyframe") is not None
                else None
            )

            gt_video_batch = sample["video"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            vae_latents = model._encode_video_latents(gt_video_batch, tiled=bool(cfg.test.tiled))
            vae_recon_video = model._decode_latents(vae_latents, tiled=bool(cfg.test.tiled))
            vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

            for high_steps in high_steps_list:
                for low_steps in low_steps_list:
                    infer_kwargs = _get_eval_inputs(sample, model)
                    infer_kwargs["num_inference_steps"] = num_inference_steps
                    infer_kwargs["high_video_inference_steps"] = int(high_steps)
                    infer_kwargs["low_video_inference_steps"] = int(low_steps)
                    # Match Trainer.evaluate: infer() uses full denoising, and low-level
                    # video steps are tied to action denoising steps. No joint_denoise path
                    # is used by model.infer().
                    infer_kwargs["action_inference_steps"] = int(low_steps)
                    infer_kwargs["seed"] = int(cfg.test.seed)
                    infer_kwargs["rand_device"] = str(cfg.test.rand_device)
                    infer_kwargs["tiled"] = bool(cfg.test.tiled)

                    logger.info(
                        "Running sample=%d high_steps=%d low_steps=%d action_steps=%d",
                        sample_index,
                        high_steps,
                        low_steps,
                        infer_kwargs["action_inference_steps"],
                    )
                    pred = model.infer(**infer_kwargs)
                    pred_video = pred["video"]
                    pred_high = pred.get("video_high")
                    pred_video_tensor = pil_frames_to_video_tensor(pred_video)
                    if tuple(pred_video_tensor.shape) != tuple(gt_video_tensor.shape):
                        raise ValueError(
                            "Infer prediction/GT shape mismatch: "
                            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                        )
                    if tuple(vae_video_tensor.shape) != tuple(gt_video_tensor.shape):
                        raise ValueError(
                            "VAE reconstruction/GT shape mismatch: "
                            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                        )

                    stem = f"sample_{sample_index:06d}_h{high_steps}_l{low_steps}"
                    low_path = output_dir / f"{stem}_low_pred_vae_gt.mp4"
                    _save_eval_style_video(
                        tensors=[pred_video_tensor, vae_video_tensor, gt_video_tensor],
                        path=str(low_path),
                        fps=fps,
                    )

                    high_path = None
                    if pred_high is not None and gt_keyframe_tensor is not None:
                        pred_high_tensor = pil_frames_to_video_tensor(pred_high)
                        if tuple(pred_high_tensor.shape) != tuple(gt_keyframe_tensor.shape):
                            raise ValueError(
                                "High-level keyframe/GT keyframe shape mismatch: "
                                f"pred={tuple(pred_high_tensor.shape)} vs gt={tuple(gt_keyframe_tensor.shape)}"
                            )
                        high_path = output_dir / f"{stem}_high_pred_gt.mp4"
                        _save_eval_style_video(
                            tensors=[pred_high_tensor, gt_keyframe_tensor],
                            path=str(high_path),
                            fps=fps,
                        )

                    metrics = _evaluate_prediction(pred_video, raw_sample["video"])
                    result = {
                        "sample_index": int(sample_index),
                        "high_steps": int(high_steps),
                        "low_steps": int(low_steps),
                        "action_steps": int(infer_kwargs["action_inference_steps"]),
                        "prompt_path": prompt_path.name,
                        "low_video_path": low_path.name,
                        "high_video_path": None if high_path is None else high_path.name,
                        **metrics,
                    }
                    results.append(result)

    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(results, output_dir)
    logger.info("Saved %d results to %s", len(results), output_dir)


if __name__ == "__main__":
    main()
