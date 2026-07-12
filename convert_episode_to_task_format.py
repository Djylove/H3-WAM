#!/usr/bin/env python3
"""Convert qnexo episode directories into erobot v3.0 task/batch layout.

The source episode layout stores each stream in its own parquet file:
metadata.json, action.parquet, action.base.parquet, observation.state.parquet,
observation.base_state.parquet, observation.images.camera_top.parquet, and
optionally observation.images.camera_top_depth.parquet.

The output layout follows the erobot v3.0 structure:
batch_000000/data/chunk-000/file-000.parquet
batch_000000/videos/observation.images.top/chunk-000/file-000.mp4
batch_000000/meta/...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import imageio.v2 as imageio
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
except ImportError as exc:  # pragma: no cover - only used for CLI diagnostics
    raise SystemExit(
        "Missing dependency: "
        f"{exc}. Install with: python3 -m pip install pyarrow pandas numpy "
        "pillow imageio imageio-ffmpeg"
    ) from exc


FPS = 30
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832
CONVERTER_VERSION = "v1.7"
SUBTASK_ANNOTATOR_VERSION = "v1.0"
EXPAND_ANNOTATOR_VERSION = "v1.2"

OBS_STATE_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "base_height",
    "base_pitch",
]

ACTION_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "vel_height",
    "vel_pitch",
    "base_yaw",
    "vel_x",
    "vel_y",
    "vel_yaw",
]

OBS_POSE_NAMES = [
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
    "left_hand_ortho6d_1",
    "left_hand_ortho6d_2",
    "left_hand_ortho6d_3",
    "left_hand_ortho6d_4",
    "left_hand_ortho6d_5",
    "left_hand_ortho6d_6",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "right_hand_ortho6d_1",
    "right_hand_ortho6d_2",
    "right_hand_ortho6d_3",
    "right_hand_ortho6d_4",
    "right_hand_ortho6d_5",
    "right_hand_ortho6d_6",
    "head_ortho6d_1",
    "head_ortho6d_2",
    "head_ortho6d_3",
    "head_ortho6d_4",
    "head_ortho6d_5",
    "head_ortho6d_6",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "pose_base_height",
    "pose_base_pitch",
]

ACTION_POSE_NAMES = [
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
    "left_hand_ortho6d_1",
    "left_hand_ortho6d_2",
    "left_hand_ortho6d_3",
    "left_hand_ortho6d_4",
    "left_hand_ortho6d_5",
    "left_hand_ortho6d_6",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "right_hand_ortho6d_1",
    "right_hand_ortho6d_2",
    "right_hand_ortho6d_3",
    "right_hand_ortho6d_4",
    "right_hand_ortho6d_5",
    "right_hand_ortho6d_6",
    "head_ortho6d_1",
    "head_ortho6d_2",
    "head_ortho6d_3",
    "head_ortho6d_4",
    "head_ortho6d_5",
    "head_ortho6d_6",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "pose_base_vel_height",
    "pose_base_vel_pitch",
    "pose_base_yaw",
    "pose_base_vel_x",
    "pose_base_vel_y",
    "pose_base_vel_yaw",
]

POSE_BASE_ALIASES = {
    "pose_base_height": "base_height",
    "pose_base_pitch": "base_pitch",
    "pose_base_vel_height": "vel_height",
    "pose_base_vel_pitch": "vel_pitch",
    "pose_base_yaw": "base_yaw",
    "pose_base_vel_x": "vel_x",
    "pose_base_vel_y": "vel_y",
    "pose_base_vel_yaw": "vel_yaw",
}

DATA_COLUMNS = [
    "observation.state.pose",
    "action.pose",
    "observation.state",
    "action",
    "timestamp",
    "timestamp_ns",
    "state_ts_ns",
    "base_state_ts_ns",
    "action_ts_ns",
    "action_base_ts_ns",
    "video_camera_top_ts_ns",
    "depth_camera_top_ts_ns",
    "next.done",
    "progress",
    "frame_index",
    "index",
    "episode_index",
    "task_index",
    "subtask_index",
    "expand_task_index",
]

DATA_COLUMNS_WITH_DEPTH_INDEX = [
    "observation.state.pose",
    "action.pose",
    "observation.state",
    "action",
    "timestamp",
    "timestamp_ns",
    "state_ts_ns",
    "base_state_ts_ns",
    "action_ts_ns",
    "action_base_ts_ns",
    "video_camera_top_ts_ns",
    "depth_camera_top_ts_ns",
    "depth_index",
    "next.done",
    "progress",
    "frame_index",
    "index",
    "episode_index",
    "task_index",
    "subtask_index",
    "expand_task_index",
]

STATS_JSON_ORDER = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "observation.state",
    "action",
    "observation.state.pose",
    "action.pose",
    "progress",
    "next.done",
    "timestamp_ns",
    "state_ts_ns",
    "base_state_ts_ns",
    "action_ts_ns",
    "action_base_ts_ns",
    "video_camera_top_ts_ns",
    "depth_camera_top_ts_ns",
    "observation.depth.top",
    "subtask_index",
    "expand_task_index",
    "observation.images.top",
]

STATS_JSON_ORDER_WITH_DEPTH_INDEX = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "observation.state",
    "action",
    "observation.state.pose",
    "action.pose",
    "progress",
    "next.done",
    "timestamp_ns",
    "state_ts_ns",
    "action_ts_ns",
    "video_camera_top_ts_ns",
    "action_base_ts_ns",
    "base_state_ts_ns",
    "depth_camera_top_ts_ns",
    "depth_index",
    "observation.depth.top",
    "subtask_index",
    "expand_task_index",
    "observation.images.top",
]

EPISODE_META_STATS_BEFORE_META = [
    "observation.state",
    "action",
    "observation.state.pose",
    "action.pose",
    "progress",
    "next.done",
    "timestamp_ns",
    "state_ts_ns",
    "base_state_ts_ns",
    "action_ts_ns",
    "action_base_ts_ns",
    "video_camera_top_ts_ns",
    "depth_camera_top_ts_ns",
]

EPISODE_META_STATS_BEFORE_META_WITH_DEPTH_INDEX = [
    "observation.state",
    "action",
    "observation.state.pose",
    "action.pose",
    "progress",
    "next.done",
    "timestamp_ns",
    "state_ts_ns",
    "action_ts_ns",
    "video_camera_top_ts_ns",
    "action_base_ts_ns",
    "base_state_ts_ns",
    "depth_camera_top_ts_ns",
    "depth_index",
]

EPISODE_META_STATS_AFTER_META = [
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


@dataclass
class TimedNamedStream:
    timestamps_ns: np.ndarray
    values: list[dict[str, float]]

    def nearest(self, timestamp_ns: int) -> tuple[int, dict[str, float]]:
        idx = nearest_index(self.timestamps_ns, timestamp_ns)
        return int(self.timestamps_ns[idx]), self.values[idx]


@dataclass
class TimedRawStream:
    timestamps_ns: np.ndarray
    values: list[Any]
    parameters: list[str]

    def nearest_index(self, timestamp_ns: int) -> int:
        return nearest_index(self.timestamps_ns, timestamp_ns)

    def nearest_timestamp(self, timestamp_ns: int) -> int:
        idx = nearest_index(self.timestamps_ns, timestamp_ns)
        return int(self.timestamps_ns[idx])


@dataclass
class EpisodeBuild:
    source_dir: Path
    metadata: dict[str, Any]
    df: pd.DataFrame
    top_images: TimedRawStream
    depth_images: TimedRawStream | None
    task_text: str
    expand_task: str


def nearest_index(timestamps_ns: np.ndarray, query_ns: int) -> int:
    if len(timestamps_ns) == 0:
        raise ValueError("Cannot align against an empty timestamp stream")
    pos = int(np.searchsorted(timestamps_ns, query_ns))
    if pos <= 0:
        return 0
    if pos >= len(timestamps_ns):
        return len(timestamps_ns) - 1
    before = timestamps_ns[pos - 1]
    after = timestamps_ns[pos]
    return pos - 1 if abs(query_ns - before) <= abs(after - query_ns) else pos


def timestamp_series_to_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).astype("int64").to_numpy()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_named_stream(path: Path, value_col: str) -> TimedNamedStream:
    df = pq.read_table(path).to_pandas()
    timestamps_ns = timestamp_series_to_ns(df["timestamp_utc"])
    values = [named_items_to_dict(items) for items in df[value_col]]
    return TimedNamedStream(timestamps_ns=timestamps_ns, values=values)


def read_raw_stream(path: Path, value_col: str) -> TimedRawStream:
    df = pq.read_table(path).to_pandas()
    timestamps_ns = timestamp_series_to_ns(df["timestamp_utc"])
    return TimedRawStream(
        timestamps_ns=timestamps_ns,
        values=list(df[value_col]),
        parameters=list(df.get("parameters", ["{}"] * len(df))),
    )


def named_items_to_dict(items: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        name = str(item["name"])
        result[name] = float(item["value"])
    return result


def base_state_values(base_state_items: Any, mode: str) -> dict[str, float]:
    if mode == "zero" or not base_state_items:
        return {"base_height": 0.0, "base_pitch": 0.0}
    first = base_state_items[0]
    base = first.get("base") or {}
    position = base.get("position") or []
    rpy = base.get("rpy") or []
    return {
        "base_height": float(position[2]) if len(position) > 2 else 0.0,
        "base_pitch": float(rpy[1]) if len(rpy) > 1 else 0.0,
    }


def vector_from_names(
    names: list[str],
    sources: dict[str, float],
    aliases: dict[str, str] | None = None,
) -> list[float]:
    vector: list[float] = []
    aliases = aliases or {}
    for name in names:
        source_name = aliases.get(name, name)
        vector.append(float(sources.get(source_name, 0.0)))
    return vector


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)


def numbered_batch_name(base_name: str, offset: int) -> str:
    prefix, sep, suffix = base_name.rpartition("_")
    if sep and suffix.isdigit():
        return f"{prefix}_{int(suffix) + offset:0{len(suffix)}d}"
    return f"{base_name}_{offset:06d}"


def episode_batches(
    episodes: list["EpisodeBuild"],
    episodes_per_batch: int,
) -> list[list["EpisodeBuild"]]:
    if episodes_per_batch <= 0:
        return [episodes]
    return [
        episodes[index : index + episodes_per_batch]
        for index in range(0, len(episodes), episodes_per_batch)
    ]


def build_rows(
    episode_dir: Path,
    episode_index: int,
    task_index: int,
    expand_task_index: int,
    fps: int,
    base_state_mode: str,
    include_depth_index: bool,
) -> tuple[pd.DataFrame, TimedRawStream, TimedRawStream | None]:
    action = read_named_stream(episode_dir / "action.parquet", "action")
    action_base = read_named_stream(episode_dir / "action.base.parquet", "action.base")
    state = read_named_stream(episode_dir / "observation.state.parquet", "observation.state")

    base_state_df = pq.read_table(episode_dir / "observation.base_state.parquet").to_pandas()
    base_state_ts = timestamp_series_to_ns(base_state_df["timestamp_utc"])
    base_state_values_raw = list(base_state_df["observation.base_state"])

    top_images = read_raw_stream(
        episode_dir / "observation.images.camera_top.parquet",
        "observation.images.camera_top",
    )
    depth_path = episode_dir / "observation.images.camera_top_depth.parquet"
    depth_images = (
        read_raw_stream(depth_path, "observation.images.camera_top_depth")
        if depth_path.exists()
        else None
    )

    if len(top_images.timestamps_ns) == 0:
        raise ValueError(f"No top camera frames found in {episode_dir}")

    rows: list[dict[str, Any]] = []
    start_ns = int(top_images.timestamps_ns[0])
    step_ns = int(round(1_000_000_000 / fps))
    last_frame = len(top_images.timestamps_ns) - 1

    for frame_index, video_ts_ns in enumerate(top_images.timestamps_ns):
        video_ts_ns = int(video_ts_ns)
        state_ts_ns, state_values = state.nearest(video_ts_ns)
        action_ts_ns, action_values = action.nearest(video_ts_ns)
        action_base_ts_ns, action_base_values = action_base.nearest(video_ts_ns)

        base_idx = nearest_index(base_state_ts, video_ts_ns)
        base_state_ts_ns = int(base_state_ts[base_idx])
        base_values = base_state_values(base_state_values_raw[base_idx], base_state_mode)

        if depth_images is not None and len(depth_images.timestamps_ns) > 0:
            depth_idx = depth_images.nearest_index(video_ts_ns)
            depth_ts_ns = int(depth_images.timestamps_ns[depth_idx])
        else:
            depth_idx = -1
            depth_ts_ns = 0

        obs_sources = {**state_values, **base_values}
        action_sources = {**action_values, **action_base_values}

        row = {
            "observation.state.pose": vector_from_names(
                OBS_POSE_NAMES, obs_sources, POSE_BASE_ALIASES
            ),
            "action.pose": vector_from_names(
                ACTION_POSE_NAMES, action_sources, POSE_BASE_ALIASES
            ),
            "observation.state": vector_from_names(OBS_STATE_NAMES, obs_sources),
            "action": vector_from_names(ACTION_NAMES, action_sources),
            "timestamp": float(frame_index / fps),
            "timestamp_ns": int(start_ns + frame_index * step_ns),
            "state_ts_ns": state_ts_ns,
            "base_state_ts_ns": base_state_ts_ns,
            "action_ts_ns": action_ts_ns,
            "action_base_ts_ns": action_base_ts_ns,
            "video_camera_top_ts_ns": video_ts_ns,
            "depth_camera_top_ts_ns": int(depth_ts_ns),
            "next.done": frame_index == last_frame,
            "progress": float(frame_index / last_frame) if last_frame > 0 else 1.0,
            "frame_index": int(frame_index),
            "index": int(frame_index),
            "episode_index": int(episode_index),
            "task_index": int(task_index),
            "subtask_index": 0,
            "expand_task_index": int(expand_task_index),
        }
        if include_depth_index:
            row["depth_index"] = int(depth_idx)
        rows.append(row)

    columns = DATA_COLUMNS_WITH_DEPTH_INDEX if include_depth_index else DATA_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = df["timestamp"].astype("float32")
    return df, top_images, depth_images


def decode_image(raw: Any) -> Image.Image:
    if isinstance(raw, np.ndarray):
        payload = raw.astype(np.uint8, copy=False).tobytes()
    elif isinstance(raw, (bytes, bytearray)):
        payload = bytes(raw)
    else:
        payload = bytes(raw)
    return Image.open(io.BytesIO(payload)).convert("RGB")


def write_video(
    images: TimedRawStream,
    output_path: Path,
    fps: int,
    no_video: bool,
    video_codec: str,
    with_quantiles: bool,
    fast_stats: bool,
) -> dict[str, list[Any]]:
    writer = None
    if not no_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_codec = "libsvtav1" if video_codec == "av1" else "libx264"
        ffmpeg_params = ["-preset", "12", "-crf", "35"] if video_codec == "av1" else []
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=ffmpeg_codec,
            pixelformat="yuv420p",
            ffmpeg_params=ffmpeg_params,
            ffmpeg_log_level="error",
            macro_block_size=16,
        )

    count = 0
    channel_min = np.full(3, np.inf, dtype=np.float64)
    channel_max = np.full(3, -np.inf, dtype=np.float64)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    channel_hist = np.zeros((3, 256), dtype=np.int64) if with_quantiles else None
    pixel_count = 0

    try:
        for raw in images.values:
            frame = decode_image(raw).resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
            arr = np.asarray(frame, dtype=np.uint8)
            if writer is not None:
                writer.append_data(arr)
            if fast_stats:
                count += 1
                continue

            normalized = arr.astype(np.float64) / 255.0
            flat = normalized.reshape(-1, 3)
            channel_min = np.minimum(channel_min, flat.min(axis=0))
            channel_max = np.maximum(channel_max, flat.max(axis=0))
            channel_sum += flat.sum(axis=0)
            channel_sumsq += (flat * flat).sum(axis=0)
            if channel_hist is not None:
                for channel in range(3):
                    channel_hist[channel] += np.bincount(
                        arr[:, :, channel].reshape(-1), minlength=256
                    )
            pixel_count += flat.shape[0]
            count += 1
    finally:
        if writer is not None:
            writer.close()

    if count == 0 or pixel_count == 0:
        stats = image_stats_placeholder(with_quantiles)
        stats["count"] = [int(count)]
        return stats

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sumsq / pixel_count - mean * mean, 0.0)
    std = np.sqrt(variance)
    stats = {
        "min": nested_channel_stats(channel_min),
        "max": nested_channel_stats(channel_max),
        "mean": nested_channel_stats(mean),
        "std": nested_channel_stats(std),
        "count": [int(count)],
    }
    if channel_hist is not None:
        quantiles = histogram_quantiles(channel_hist, [0.01, 0.10, 0.50, 0.90, 0.99]) / 255.0
        for idx, key in enumerate(["q01", "q10", "q50", "q90", "q99"]):
            stats[key] = nested_channel_stats(quantiles[:, idx])
    return stats


def configure_ffmpeg(ffmpeg_exe: Path | None) -> None:
    if ffmpeg_exe is None:
        return
    resolved = ffmpeg_exe.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"ffmpeg executable does not exist: {resolved}")
    os.environ["IMAGEIO_FFMPEG_EXE"] = str(resolved)


def nested_channel_stats(values: np.ndarray) -> list[list[list[float]]]:
    return [[[round(float(value), 6)]] for value in values.tolist()]


def image_stats_placeholder(with_quantiles: bool = False) -> dict[str, list[Any]]:
    stats = {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.0]], [[0.0]], [[0.0]]],
        "std": [[[0.0]], [[0.0]], [[0.0]]],
        "count": [0],
    }
    if with_quantiles:
        for key in ["q01", "q10", "q50", "q90", "q99"]:
            stats[key] = [[[0.0]], [[0.0]], [[0.0]]]
    return stats


def histogram_quantiles(hist: np.ndarray, quantiles: list[float]) -> np.ndarray:
    if hist.ndim == 1:
        hist_2d = hist.reshape(1, -1)
    else:
        hist_2d = hist
    result = np.zeros((hist_2d.shape[0], len(quantiles)), dtype=np.float64)
    for row_idx, row in enumerate(hist_2d):
        total = int(row.sum())
        if total == 0:
            continue
        cumulative = np.cumsum(row)
        for q_idx, quantile in enumerate(quantiles):
            target = max(int(np.ceil(quantile * total)), 1)
            result[row_idx, q_idx] = float(np.searchsorted(cumulative, target, side="left"))
    return result


def decode_depth(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        payload = raw.astype(np.uint8, copy=False).tobytes()
    elif isinstance(raw, (bytes, bytearray)):
        payload = bytes(raw)
    else:
        payload = bytes(raw)
    return np.asarray(Image.open(io.BytesIO(payload)))


def compute_depth_stats(
    depth_images: TimedRawStream | None,
    with_quantiles: bool,
    fast_stats: bool,
) -> dict[str, list[Any]]:
    if depth_images is None or not depth_images.values:
        return depth_stats_placeholder(with_quantiles)
    if fast_stats:
        stats = depth_stats_placeholder(with_quantiles)
        stats["count"] = [len(depth_images.values)]
        return stats

    count = 0
    value_min = np.inf
    value_max = -np.inf
    value_sum = 0.0
    value_sumsq = 0.0
    pixel_count = 0
    hist = np.zeros(65536, dtype=np.int64) if with_quantiles else None

    for raw in depth_images.values:
        arr = decode_depth(raw)
        if arr.ndim > 2:
            arr = arr[:, :, 0]
        arr64 = arr.astype(np.float64, copy=False)
        value_min = min(value_min, float(arr64.min()))
        value_max = max(value_max, float(arr64.max()))
        value_sum += float(arr64.sum())
        value_sumsq += float((arr64 * arr64).sum())
        pixel_count += int(arr64.size)
        count += 1
        if hist is not None:
            clipped = np.clip(arr.astype(np.int64, copy=False).reshape(-1), 0, 65535)
            hist += np.bincount(clipped, minlength=65536)

    if count == 0 or pixel_count == 0:
        return depth_stats_placeholder(with_quantiles)

    mean = value_sum / pixel_count
    variance = max(value_sumsq / pixel_count - mean * mean, 0.0)
    stats = {
        "min": [float(value_min)],
        "max": [float(value_max)],
        "mean": [float(mean)],
        "std": [float(np.sqrt(variance))],
        "count": [int(count)],
    }
    if hist is not None:
        quantile_values = histogram_quantiles(hist, [0.01, 0.10, 0.50, 0.90, 0.99]).reshape(-1)
        for value, key in zip(quantile_values.tolist(), ["q01", "q10", "q50", "q90", "q99"]):
            stats[key] = [float(value)]
    return stats


def depth_stats_placeholder(with_quantiles: bool = False) -> dict[str, list[Any]]:
    stats = {
        "min": [0.0],
        "max": [0.0],
        "mean": [0.0],
        "std": [0.0],
        "count": [0],
    }
    if with_quantiles:
        for key in ["q01", "q10", "q50", "q90", "q99"]:
            stats[key] = [0.0]
    return stats


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_data_parquet(episode_dfs: list[pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for df in episode_dfs:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def large_string_series(values: list[str]) -> pd.Series:
    return pd.Series(values, dtype=pd.ArrowDtype(pa.large_string()))


def combine_raw_streams(streams: list[TimedRawStream | None]) -> TimedRawStream | None:
    present = [stream for stream in streams if stream is not None]
    if not present:
        return None
    timestamps = np.concatenate([stream.timestamps_ns for stream in present])
    values: list[Any] = []
    parameters: list[str] = []
    for stream in present:
        values.extend(stream.values)
        parameters.extend(stream.parameters)
    return TimedRawStream(timestamps_ns=timestamps, values=values, parameters=parameters)


def numeric_matrix(values: pd.Series) -> np.ndarray:
    first = values.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.asarray(values.to_list(), dtype=np.float64)
    return values.astype(np.float64).to_numpy().reshape(-1, 1)


def basic_stats(values: pd.Series) -> dict[str, list[Any]]:
    matrix = numeric_matrix(values)
    count = matrix.shape[0]
    return {
        "min": matrix.min(axis=0).astype(float).tolist(),
        "max": matrix.max(axis=0).astype(float).tolist(),
        "mean": matrix.mean(axis=0).astype(float).tolist(),
        "std": matrix.std(axis=0, ddof=0).astype(float).tolist(),
        "count": [int(count)],
    }


def stats_with_quantiles(values: pd.Series) -> dict[str, list[Any]]:
    stats = basic_stats(values)
    matrix = numeric_matrix(values)
    for key, q in [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]:
        stats[key] = np.quantile(matrix, q, axis=0).astype(float).tolist()
    return stats


def bool_stats(values: pd.Series, with_quantiles: bool = False) -> dict[str, list[Any]]:
    numeric = values.astype(bool).astype(np.int64)
    stats = basic_stats(numeric)
    stats["min"] = [int(stats["min"][0])]
    stats["max"] = [int(stats["max"][0])]
    if with_quantiles:
        matrix = numeric.to_numpy().reshape(-1, 1)
        for key, q in [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]:
            stats[key] = np.quantile(matrix, q, axis=0).astype(float).tolist()
    return stats


def depth_index_episode_stats(values: pd.Series) -> dict[str, list[Any]]:
    numeric = values.astype(np.int64).to_numpy()
    return {
        "min": [int(numeric.min())],
        "max": [int(numeric.max())],
        "mean": [float(numeric.mean())],
        "std": [float(numeric.std(ddof=0))],
        "count": [int(numeric.shape[0])],
    }


def build_stats_json(
    df: pd.DataFrame,
    image_stats: dict[str, list[Any]],
    depth_stats: dict[str, list[Any]],
    include_depth_index: bool,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    order = STATS_JSON_ORDER_WITH_DEPTH_INDEX if include_depth_index else STATS_JSON_ORDER
    for col in order:
        if col == "observation.images.top":
            stats[col] = image_stats
            continue
        if col == "observation.depth.top":
            stats[col] = depth_stats
            continue
        if col == "next.done":
            stats[col] = bool_stats(df[col], with_quantiles=True)
        else:
            stats[col] = stats_with_quantiles(df[col])
    return stats


def build_episode_meta_row(
    df: pd.DataFrame,
    task_text: str,
    expand_task: str,
    image_stats: dict[str, list[Any]],
    fps: int,
    include_depth_index: bool,
) -> dict[str, Any]:
    dataset_from_index = int(df["index"].min())
    dataset_to_index = int(df["index"].max() + 1)
    row: dict[str, Any] = {
        "episode_index": int(df["episode_index"].iloc[0]),
        "tasks": [task_text],
        "length": int(len(df)),
        "data/chunk_index": 0,
        "data/file_index": 0,
        "videos/observation.images.top/chunk_index": 0,
        "videos/observation.images.top/file_index": 0,
        "videos/observation.images.top/from_timestamp": float(dataset_from_index / fps),
        "videos/observation.images.top/to_timestamp": float(dataset_to_index / fps),
    }
    before_meta = (
        EPISODE_META_STATS_BEFORE_META_WITH_DEPTH_INDEX
        if include_depth_index
        else EPISODE_META_STATS_BEFORE_META
    )
    for col in before_meta:
        if col == "next.done":
            stats = bool_stats(df[col])
        elif col == "depth_index":
            local_depth_index = df[col].astype(np.int64) - int(df[col].min())
            stats = depth_index_episode_stats(local_depth_index)
        else:
            stats = basic_stats(df[col])
        for stat_name, value in stats.items():
            row[f"stats/{col}/{stat_name}"] = value

    row["meta/episodes/chunk_index"] = 0
    row["meta/episodes/file_index"] = 0
    row["dataset_from_index"] = dataset_from_index
    row["dataset_to_index"] = dataset_to_index
    if include_depth_index:
        row["videos/observation_images_top_depth/from_timestamp"] = int(df["depth_index"].min())
        row["videos/observation_images_top_depth/to_timestamp"] = int(df["depth_index"].max())

    for col in EPISODE_META_STATS_AFTER_META:
        stats = basic_stats(df[col])
        for stat_name, value in stats.items():
            row[f"stats/{col}/{stat_name}"] = value

    for stat_name in ["min", "max", "mean", "std", "count"]:
        row[f"stats/observation.images.top/{stat_name}"] = image_stats[stat_name]
    row["expand_task"] = expand_task
    return row


def episode_meta_columns(include_depth_index: bool) -> list[str]:
    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "videos/observation.images.top/chunk_index",
        "videos/observation.images.top/file_index",
        "videos/observation.images.top/from_timestamp",
        "videos/observation.images.top/to_timestamp",
    ]
    before_meta = (
        EPISODE_META_STATS_BEFORE_META_WITH_DEPTH_INDEX
        if include_depth_index
        else EPISODE_META_STATS_BEFORE_META
    )
    for col in before_meta:
        for stat_name in ["min", "max", "mean", "std", "count"]:
            columns.append(f"stats/{col}/{stat_name}")
    columns.extend(
        [
            "meta/episodes/chunk_index",
            "meta/episodes/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
    )
    if include_depth_index:
        columns.extend(
            [
                "videos/observation_images_top_depth/from_timestamp",
                "videos/observation_images_top_depth/to_timestamp",
            ]
        )
    for col in EPISODE_META_STATS_AFTER_META:
        for stat_name in ["min", "max", "mean", "std", "count"]:
            columns.append(f"stats/{col}/{stat_name}")
    for stat_name in ["min", "max", "mean", "std", "count"]:
        columns.append(f"stats/observation.images.top/{stat_name}")
    columns.append("expand_task")
    return columns


def build_info_json(
    total_episodes: int,
    total_frames: int,
    task_count: int,
    fps: int,
    include_depth_index: bool,
    video_codec: str,
) -> dict[str, Any]:
    features = {
        "observation.state": feature("float32", [len(OBS_STATE_NAMES)], OBS_STATE_NAMES, fps),
        "action": feature("float32", [len(ACTION_NAMES)], ACTION_NAMES, fps),
        "observation.state.pose": feature("float32", [len(OBS_POSE_NAMES)], OBS_POSE_NAMES, fps),
        "action.pose": feature("float32", [len(ACTION_POSE_NAMES)], ACTION_POSE_NAMES, fps),
        "timestamp": feature("float32", [1], None, fps),
        "progress": feature("float32", [1], None, fps),
        "frame_index": feature("int64", [1], None, fps),
        "episode_index": feature("int64", [1], None, fps),
        "index": feature("int64", [1], None, fps),
        "task_index": feature("int64", [1], None, fps),
        "next.done": feature("bool", [1], None, fps),
        "timestamp_ns": feature("int64", [1], None, fps),
        "state_ts_ns": feature("int64", [1], None, fps),
    }
    if include_depth_index:
        features.update(
            {
                "action_ts_ns": feature("int64", [1], None, fps),
                "observation.images.top": video_feature(fps, video_codec),
                "video_camera_top_ts_ns": feature("int64", [1], None, fps),
                "action_base_ts_ns": feature("int64", [1], None, fps),
                "base_state_ts_ns": feature("int64", [1], None, fps),
                "depth_camera_top_ts_ns": feature("int64", [1], None, fps),
                "depth_index": feature("int64", [1], None, fps),
                "observation.depth.top": depth_feature(fps),
                "subtask_index": feature("int64", [1], None, fps),
                "expand_task_index": feature("int64", [1], None, fps),
            }
        )
    else:
        features.update(
            {
                "base_state_ts_ns": feature("int64", [1], None, fps),
                "action_ts_ns": feature("int64", [1], None, fps),
                "action_base_ts_ns": feature("int64", [1], None, fps),
                "observation.images.top": video_feature(fps, video_codec),
                "video_camera_top_ts_ns": feature("int64", [1], None, fps),
                "depth_camera_top_ts_ns": feature("int64", [1], None, fps),
                "observation.depth.top": depth_feature(fps),
                "subtask_index": feature("int64", [1], None, fps),
                "expand_task_index": feature("int64", [1], None, fps),
            }
        )
    return {
        "codebase_version": "v3.0",
        "robot_type": "gr3qnexo",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(task_count),
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
    }


def feature(dtype: str, shape: list[int], names: list[str] | None, fps: int) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names, "fps": int(fps)}


def video_feature(fps: int, video_codec: str) -> dict[str, Any]:
    return {
        "dtype": "video",
        "copy": True,
        "shape": [3, VIDEO_HEIGHT, VIDEO_WIDTH],
        "names": ["channel", "height", "width"],
        "info": {
            "video.height": VIDEO_HEIGHT,
            "video.width": VIDEO_WIDTH,
            "video.codec": video_codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }


def depth_feature(fps: int) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [600, 960], "fps": int(fps)}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_metadata_files(
    output_root: Path,
    batch_dir: Path,
    all_df: pd.DataFrame,
    episodes: list[EpisodeBuild],
    task_texts_by_index: dict[int, str],
    task_ids_by_index: dict[int, str],
    image_stats: dict[str, list[Any]],
    depth_stats: dict[str, list[Any]],
    fps: int,
    include_depth_index: bool,
    video_codec: str,
    subtask_text: str,
    atomic_skill: str,
    root_all_df: pd.DataFrame | None = None,
    root_episodes: list[EpisodeBuild] | None = None,
) -> None:
    summary_df = root_all_df if root_all_df is not None else all_df
    summary_episodes = root_episodes if root_episodes is not None else episodes
    meta_dir = batch_dir / "meta"
    write_json(
        meta_dir / "info.json",
        build_info_json(
            len(episodes),
            len(all_df),
            len(task_texts_by_index),
            fps,
            include_depth_index,
            video_codec,
        ),
    )
    write_json(
        meta_dir / "stats.json",
        build_stats_json(all_df, image_stats, depth_stats, include_depth_index),
    )
    now = time.time()
    write_json(
        meta_dir / "data_lineage.json",
        {
            "version": "1.0",
            "created_at": now,
            "repo_id": output_root.name,
            "total_episodes": len(episodes),
            "converter_version": CONVERTER_VERSION,
            "subtask_annotator_version": SUBTASK_ANNOTATOR_VERSION,
            "episodes": [
                {
                    "episode_index": int(episode.df["episode_index"].iloc[0]),
                    "sample_id": sample_id_from_metadata(episode.metadata),
                    "converted_at": now,
                }
                for episode in episodes
            ],
            "expand_annotator_version": EXPAND_ANNOTATOR_VERSION,
        },
    )

    tasks_df = pd.DataFrame(
        {"task_index": list(task_texts_by_index.keys())},
        index=pd.Index(large_string_series(list(task_texts_by_index.values())), name=None),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    subtask_df = pd.DataFrame(
        [
            {
                "subtask_index": 0,
                "atomic_skill": atomic_skill,
                "subtask": subtask_text,
                "has_regrasp": False,
            }
        ]
    )
    subtask_df["atomic_skill"] = large_string_series([atomic_skill])
    subtask_df["subtask"] = large_string_series([subtask_text])
    write_parquet(subtask_df, meta_dir / "subtask.parquet")

    expand_df = pd.DataFrame(
        [
            {
                "index": int(episode.df["expand_task_index"].iloc[0]),
                "expand_task": episode.expand_task,
            }
            for episode in episodes
        ]
    )
    expand_df["expand_task"] = large_string_series(expand_df["expand_task"].astype(str).tolist())
    write_parquet(expand_df, meta_dir / "expand_annotation/expand_task_annotation.parquet")

    episode_rows = [
        build_episode_meta_row(
            episode.df,
            episode.task_text,
            episode.expand_task,
            image_stats,
            fps,
            include_depth_index,
        )
        for episode in episodes
    ]
    episode_df = pd.DataFrame(episode_rows, columns=episode_meta_columns(include_depth_index))
    episode_df["expand_task"] = large_string_series(episode_df["expand_task"].astype(str).tolist())
    write_parquet(
        episode_df,
        meta_dir / "episodes/chunk-000/file-000.parquet",
    )

    duration_seconds = len(summary_df) / fps
    task_counts: dict[str, int] = {task_id: 0 for task_id in task_ids_by_index.values()}
    for episode in summary_episodes:
        task_id = str(episode.metadata.get("task_id", "0"))
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    task_stats = {
        "summary": {
            "total_episodes": len(summary_episodes),
            "total_duration_hours": round(duration_seconds / 3600.0, 6),
            "total_tasks": len(task_counts),
            "episodes_per_task": task_counts,
            "total_atomic_skills": 1 if atomic_skill else 0,
            "total_items": 0,
        },
        "task_ids": [task_ids_by_index[index] for index in sorted(task_ids_by_index)],
        "tasks": {
            str(task_id): {
                "id": int(task_id) if str(task_id).isdigit() else task_id,
                "guidences": [],
                "descriptions": {"zh": [task_texts_by_index[task_index]], "en": []},
                "items": [],
                "atomic_skills": [atomic_skill] if atomic_skill else [],
            }
            for task_index, task_id in task_ids_by_index.items()
        },
        "atomic_skills": {
            atomic_skill: {
                "atomic_skill_id": 1,
                "episode_count": len(summary_episodes),
                "total_duration_hours": round(duration_seconds / 3600.0, 6),
            }
        }
        if atomic_skill
        else {},
        "items": {},
        "episodes": {
            sample_id_from_metadata(episode.metadata): {
                "id": episode.metadata.get("episode_index", 0),
                "uniq_id": sample_id_from_metadata(episode.metadata),
                "task_id": str(episode.metadata.get("task_id", "0")),
                "pilot": str(episode.metadata.get("pilot", "")),
                "operator": str(episode.metadata.get("operator", "")),
                "machine_id": episode.metadata.get("machine_id", ""),
                "equipment": episode.metadata.get("equipment", ""),
                "trajectory_start": normalize_time(episode.metadata.get("start_time")),
                "trajectory_length": int(len(episode.df)),
                "trajectory_duration": round(len(episode.df) / fps, 6),
            }
            for episode in summary_episodes
        },
        "generate_time": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "task_statistics.json", task_stats)
    write_json(
        output_root / "quality_checker.json",
        {
            "status": "not_run",
            "summary": "Generated by conversion script; no external quality checks were run.",
            "total_episodes": len(summary_episodes),
            "total_frames": int(len(summary_df)),
        },
    )


def sample_id_from_metadata(metadata: dict[str, Any]) -> str:
    session_id = str(metadata.get("session_id", "session"))
    episode_index = metadata.get("episode_index", 0)
    return f"{session_id}_episode_{int(episode_index):09d}"


def normalize_time(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    return text[:-1] if text.endswith("Z") else text


def default_text(metadata: dict[str, Any], fallback: str) -> str:
    notes = str(metadata.get("notes") or "").strip()
    task_id = metadata.get("task_id", "")
    if notes:
        return notes
    if task_id != "":
        return f"Task {task_id}"
    return fallback


def is_episode_dir(path: Path) -> bool:
    return (path / "metadata.json").exists() and (path / "action.parquet").exists()


def episode_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.rsplit("_", 1)[-1]), path.name
    except ValueError:
        return sys.maxsize, path.name


def discover_episode_dirs(input_path: Path) -> list[Path]:
    if is_episode_dir(input_path):
        return [input_path]
    episode_dirs = sorted(
        [path for path in input_path.glob("episode_*") if path.is_dir() and is_episode_dir(path)],
        key=episode_sort_key,
    )
    if not episode_dirs:
        raise FileNotFoundError(
            f"No episode directories found under {input_path}. "
            "Pass either an episode_* directory or a folder containing episode_* directories."
        )
    return episode_dirs


def build_task_maps(
    metadatas: list[dict[str, Any]],
    task_text_override: str | None,
) -> tuple[dict[str, int], dict[int, str], dict[int, str]]:
    task_ids: list[str] = []
    representative: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        task_id = str(metadata.get("task_id", "0"))
        if task_id not in representative:
            representative[task_id] = metadata
            task_ids.append(task_id)

    multiple_tasks = len(task_ids) > 1
    task_index_by_id: dict[str, int] = {}
    task_texts_by_index: dict[int, str] = {}
    task_ids_by_index: dict[int, str] = {}
    for task_index, task_id in enumerate(task_ids):
        metadata = representative[task_id]
        if task_text_override:
            text = task_text_override if not multiple_tasks else f"{task_text_override} / task {task_id}"
        else:
            text = default_text(metadata, f"Task {task_id}")
            if multiple_tasks:
                text = f"{text} / task {task_id}"
        task_index_by_id[task_id] = task_index
        task_texts_by_index[task_index] = text
        task_ids_by_index[task_index] = task_id
    return task_index_by_id, task_texts_by_index, task_ids_by_index


def load_expand_task_overrides(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = read_json(path.expanduser().resolve())
    values = payload.get("expand_tasks") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(
            "--expand-task-file must contain a JSON list of strings or "
            "an object with an expand_tasks list"
        )
    return values


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert qnexo episode parquet streams to erobot v3.0 layout."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source episode directory")
    parser.add_argument("--output", required=True, type=Path, help="Output task directory")
    parser.add_argument("--batch-name", default="batch_000000", help="Output batch directory name")
    parser.add_argument(
        "--episodes-per-batch",
        default=0,
        type=int,
        help=(
            "Split output into numbered batch folders with this many episodes each. "
            "0 keeps all episodes in --batch-name."
        ),
    )
    parser.add_argument("--fps", default=FPS, type=int, help="Output sampling FPS")
    parser.add_argument("--episode-index", default=0, type=int, help="Target episode index")
    parser.add_argument("--task-index", default=0, type=int, help="Target task index")
    parser.add_argument(
        "--task-id-override",
        default=None,
        help="Override metadata task_id in generated task_statistics/task mapping",
    )
    parser.add_argument("--task-text", default=None, help="Task text written to meta/tasks.parquet")
    parser.add_argument("--expand-task", default=None, help="Expanded task annotation")
    parser.add_argument(
        "--expand-task-file",
        default=None,
        type=Path,
        help="JSON list of per-episode expanded task annotations",
    )
    parser.add_argument(
        "--subtask-text",
        default="Converted episode segment.",
        help="Subtask text written to meta/subtask.parquet",
    )
    parser.add_argument(
        "--atomic-skill",
        default="Unknown",
        help="Atomic skill written to meta/subtask.parquet and task_statistics.json",
    )
    parser.add_argument(
        "--base-state-mode",
        choices=["zero", "source"],
        default="zero",
        help="Fill base_height/base_pitch with zeros or source base position/rpy values",
    )
    parser.add_argument(
        "--no-depth-index",
        action="store_true",
        help="Write the older v3.0-compatible layout without the depth_index column",
    )
    parser.add_argument(
        "--video-codec",
        choices=["av1", "h264"],
        default="av1",
        help="Video codec declared in info.json and used for mp4 encoding",
    )
    parser.add_argument(
        "--ffmpeg-exe",
        default=None,
        type=Path,
        help="Explicit ffmpeg executable for imageio, useful when AV1 is installed in conda",
    )
    parser.add_argument(
        "--fast-stats",
        action="store_true",
        help="Keep stats fields but skip expensive per-pixel image/depth statistics",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip mp4 writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    configure_ffmpeg(args.ffmpeg_exe)

    episode_dirs = discover_episode_dirs(input_path)
    metadatas = [read_json(episode_dir / "metadata.json") for episode_dir in episode_dirs]
    if args.task_id_override is not None:
        overridden = []
        for metadata in metadatas:
            copied = dict(metadata)
            copied["source_task_id"] = metadata.get("task_id", "")
            copied["task_id"] = args.task_id_override
            overridden.append(copied)
        metadatas = overridden
    task_index_by_id, task_texts_by_index, task_ids_by_index = build_task_maps(
        metadatas, args.task_text
    )
    expand_task_overrides = load_expand_task_overrides(args.expand_task_file)
    if expand_task_overrides is not None and len(expand_task_overrides) != len(episode_dirs):
        raise ValueError(
            f"--expand-task-file contains {len(expand_task_overrides)} item(s), "
            f"but input contains {len(episode_dirs)} episode(s)"
        )

    ensure_output_root(output_root, args.overwrite)
    include_depth_index = not args.no_depth_index
    episodes: list[EpisodeBuild] = []
    global_index = 0
    skipped_episodes: list[tuple[Path, str]] = []

    for offset, (episode_dir, metadata) in enumerate(zip(episode_dirs, metadatas)):
        # Keep output indices contiguous even when an unreadable source episode is skipped.
        output_episode_index = args.episode_index + len(episodes)
        task_id = str(metadata.get("task_id", "0"))
        task_index = task_index_by_id.get(task_id, args.task_index)
        task_text = task_texts_by_index.get(task_index, default_text(metadata, f"Task {task_id}"))
        if expand_task_overrides is not None:
            expand_task = expand_task_overrides[offset]
        else:
            expand_task = args.expand_task or task_text

        try:
            df, top_images, depth_images = build_rows(
                episode_dir=episode_dir,
                episode_index=output_episode_index,
                task_index=task_index,
                expand_task_index=output_episode_index,
                fps=args.fps,
                base_state_mode=args.base_state_mode,
                include_depth_index=include_depth_index,
            )
        except Exception as exc:
            skipped_episodes.append((episode_dir, f"{type(exc).__name__}: {exc}"))
            print(f"Skipping unreadable episode {episode_dir}: {type(exc).__name__}: {exc}")
            continue
        df["index"] = df["index"].astype(np.int64) + int(global_index)
        if include_depth_index:
            local_depth = df["depth_index"].astype(np.int64)
            df["depth_index"] = local_depth.where(local_depth < 0, local_depth + int(global_index))

        episodes.append(
            EpisodeBuild(
                source_dir=episode_dir,
                metadata=metadata,
                df=df,
                top_images=top_images,
                depth_images=depth_images,
                task_text=task_text,
                expand_task=expand_task,
            )
        )
        global_index += len(df)

    if skipped_episodes:
        print(f"Skipped {len(skipped_episodes)} unreadable episode(s) during conversion.")
    if not episodes:
        raise ValueError("No readable episodes remain after conversion-time validation")

    all_df = pd.concat([episode.df for episode in episodes], ignore_index=True)
    batches = episode_batches(episodes, args.episodes_per_batch)

    for batch_index, batch_episodes in enumerate(batches):
        batch_name = (
            args.batch_name
            if args.episodes_per_batch <= 0
            else numbered_batch_name(args.batch_name, batch_index)
        )
        batch_dir = output_root / batch_name
        batch_df = pd.concat([episode.df for episode in batch_episodes], ignore_index=True)

        data_path = batch_dir / "data/chunk-000/file-000.parquet"
        write_data_parquet([episode.df for episode in batch_episodes], data_path)

        video_path = batch_dir / "videos/observation.images.top/chunk-000/file-000.mp4"
        combined_top_images = combine_raw_streams(
            [episode.top_images for episode in batch_episodes]
        )
        if combined_top_images is None:
            raise ValueError("No top camera images found in input episodes")
        image_stats = write_video(
            combined_top_images,
            video_path,
            args.fps,
            args.no_video,
            args.video_codec,
            include_depth_index,
            args.fast_stats,
        )
        combined_depth_images = combine_raw_streams(
            [episode.depth_images for episode in batch_episodes]
        )
        depth_stats = compute_depth_stats(
            combined_depth_images,
            include_depth_index,
            args.fast_stats,
        )

        write_metadata_files(
            output_root=output_root,
            batch_dir=batch_dir,
            all_df=batch_df,
            episodes=batch_episodes,
            task_texts_by_index=task_texts_by_index,
            task_ids_by_index=task_ids_by_index,
            image_stats=image_stats,
            depth_stats=depth_stats,
            fps=args.fps,
            include_depth_index=include_depth_index,
            video_codec=args.video_codec,
            subtask_text=args.subtask_text,
            atomic_skill=args.atomic_skill,
            root_all_df=all_df,
            root_episodes=episodes,
        )

        print(
            f"Converted batch {batch_name}: {len(batch_episodes)} episode(s), "
            f"{len(batch_df)} frame(s)"
        )
        print(f"Data: {data_path}")
        if not args.no_video:
            print(f"Video: {video_path}")
        print(f"Meta: {batch_dir / 'meta'}")

    print(f"Converted {len(episodes)} episode(s) from {input_path} -> {output_root}")
    print(f"Frames: {len(all_df)}")
    print(f"Batches: {len(batches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
