"""No-K/V-cache input pipeline for the online frozen-H3 FACT port.

The datasets in this module deliberately stop before H3.  They return either
already-audited H3 VAE first-frame latents (dense expert demonstrations) or
raw LIBERO camera pixels (policy rollouts).  The training process owns frozen
VAE/H3 execution and must produce the thirty layer-wise K/V bundles in memory.
No item reads or writes a precomputed H3 feature/K/V artifact.

This mirrors FACT's executable data boundary: expert episodes retain action
imitation, failure episodes mask imitation while keeping future/value targets,
and dataset selection is episode-balanced before frame selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from .deployment import minmax_normalize, preprocess_libero_cameras
from .fact_backbone_port import (
    C59FailureOverlay,
    C60CausalFailureLabels,
    C59_VALUE_CONTRACTS,
)


ONLINE_FACT_SAMPLE_FORMAT = "h3wam-c56b-online-fact-sample-v1"
FACT_CODE_VALUE_CONTRACT = "fact_code_remaining_plus_penalty"
SUPPORTED_ROLLOUT_FORMATS = {
    "h3wam-c48-fact-dense-value-dataset-v1",
    "h3wam-c60-counterfactual-failure-dataset-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _task_context_ids(
    source_manifest: Path, tasks: set[str]
) -> dict[str, str]:
    matches: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(source_manifest):
        task = str(row["task"])
        if task in tasks:
            matches[task].add(str(row["context_id"]))
    missing = sorted(tasks - set(matches))
    ambiguous = {task: sorted(ids) for task, ids in matches.items() if len(ids) != 1}
    if missing or ambiguous:
        raise ValueError(
            f"task/context mapping invalid: missing={missing}, ambiguous={ambiguous}"
        )
    return {task: next(iter(ids)) for task, ids in matches.items()}


def _load_text_context(cache_root: Path, context_id: str) -> dict[str, torch.Tensor]:
    payload = torch.load(
        cache_root / "contexts" / f"{context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    context = payload.get("context")
    tags = payload.get("token_tags")
    if (
        payload.get("text_only") is not True
        or not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or context.shape[0] != 1
        or context.shape[-1] not in (5120, 5376)
        or not isinstance(tags, torch.Tensor)
        or tags.ndim != 1
        or tags.numel() != context.shape[1]
    ):
        raise ValueError(f"invalid H3 text-only context: {context_id}")
    return {"context": context[0].float(), "token_tags": tags.long()}


def environment_actions_to_dataset(actions: torch.Tensor) -> torch.Tensor:
    """Invert the LIBERO deployment gripper conversion exactly."""

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("environment actions must be [T,7]")
    result = actions.float().clone()
    result[:, -1] = (1.0 - result[:, -1]) / 2.0
    return result


def dataset_actions_to_environment(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("dataset actions must be [T,7]")
    result = actions.float().clone()
    result[:, -1] = -(2.0 * result[:, -1] - 1.0)
    return result


class OnlineH3FACTDemoDataset(Dataset):
    """Dense expert windows with VAE latents but without any H3 cache."""

    stream_name = "expert_demo"
    input_mode = "vae_latents"

    def __init__(
        self,
        manifest: Path | str,
        source_manifest: Path | str,
        cache_root: Path | str,
        *,
        split: str = "train",
        action_horizon: int = 32,
        sample_offset: int = 0,
        limit: int = 0,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if action_horizon <= 0 or sample_offset < 0 or limit < 0:
            raise ValueError("invalid demo slice/action horizon")
        self.manifest = Path(manifest).resolve()
        self.source_manifest = Path(source_manifest).resolve()
        self.cache_root = Path(cache_root).resolve()
        split_rows = _read_jsonl(self.manifest)
        source_rows = _read_jsonl(self.source_manifest)
        source_by_id = {str(row["id"]): row for row in source_rows}
        if len(source_by_id) != len(source_rows):
            raise ValueError("source manifest contains duplicate ids")
        if any(source_by_id.get(str(row["id"])) != row for row in split_rows):
            raise ValueError("demo split is not byte-identical source provenance")
        rows = [row for row in split_rows if str(row.get("split")) == split]
        if sample_offset >= len(rows):
            raise ValueError("sample_offset does not select a demo row")
        rows = rows[sample_offset:]
        if limit:
            rows = rows[:limit]
        if not rows:
            raise ValueError("selected demo dataset is empty")
        self.rows = rows
        self.action_horizon = int(action_horizon)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.source_manifest_sha256 = sha256_file(self.source_manifest)
        stats_path = self.cache_root / "stats.pt"
        self.stats_sha256 = sha256_file(stats_path)
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        self._future_by_key = {
            (str(row["suite"]), int(row["episode"]), int(row["start"])): row
            for row in source_rows
        }
        self.episode_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.episode_to_indices[(str(row["suite"]), int(row["episode"]))].append(
                index
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _window(self, sample_id: str) -> dict[str, Any]:
        return torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        window = self._window(sample_id)
        actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"demo action shape mismatch: {sample_id}")
        current_latents = window["first_frame_latents"].float()
        video_latents = window["video_latents"].float()
        if (
            current_latents.ndim != 5
            or tuple(current_latents.shape[:3]) != (1, 24, 1)
            or video_latents.ndim != 5
            or tuple(video_latents.shape[:2]) != (1, 24)
        ):
            raise ValueError(f"demo H3 VAE latent shape mismatch: {sample_id}")
        # The last latent frame is the observed future visual target from this
        # exact dense window. H3 itself still runs online on this tensor.
        future_latents = video_latents[:, :, -1:].contiguous()
        future_key = (
            str(row["suite"]),
            int(row["episode"]),
            int(row["start"]) + self.action_horizon,
        )
        future_row = self._future_by_key.get(future_key)
        if future_row is None:
            future_state = torch.zeros_like(window["state"].float())
            future_state_loss_mask = 0.0
        else:
            future_state = self._window(str(future_row["id"]))["state"].float()
            future_state_loss_mask = 1.0
        context = _load_text_context(self.cache_root, str(row["context_id"]))
        return {
            "format": ONLINE_FACT_SAMPLE_FORMAT,
            "stream": self.stream_name,
            "sample_id": sample_id,
            "episode_id": f"{row['suite']}:{int(row['episode'])}",
            "input_mode": self.input_mode,
            "current_h3_input": current_latents[0].clone(),
            "future_h3_input": future_latents[0].clone(),
            "text_context": context["context"],
            "text_token_tags": context["token_tags"],
            "actions": minmax_normalize(
                actions, self.action_min, self.action_max
            ),
            "action_is_pad": action_is_pad,
            "proprio": minmax_normalize(
                window["state"].float(), self.state_min, self.state_max
            ),
            "future_state": minmax_normalize(
                future_state, self.state_min, self.state_max
            ) if future_state_loss_mask else torch.zeros_like(future_state),
            "value": torch.zeros((), dtype=torch.float32),
            "action_loss_mask": torch.tensor(1.0),
            "future_representation_loss_mask": torch.tensor(1.0),
            "future_state_loss_mask": torch.tensor(future_state_loss_mask),
            # Value for expert demonstrations is deliberately not fabricated;
            # C48/C59/C60 carry the audited FACT value contract.
            "value_loss_mask": torch.tensor(0.0),
            "failure_active_mask": torch.tensor(0.0),
        }


class OnlineH3FACTRolloutDataset(Dataset):
    """C48/C59 or C60 rollout rows decoded to RGB for online VAE/H3."""

    input_mode = "pixels"

    def __init__(
        self,
        dataset: Path | str,
        observations: Path | str,
        source_manifest: Path | str,
        demo_cache_root: Path | str,
        *,
        split: str = "train",
        value_contract: str = FACT_CODE_VALUE_CONTRACT,
        c59_overlay_root: Path | str | None = None,
        expected_dataset_sha256: str | None = None,
        expected_observations_sha256: str | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if value_contract not in C59_VALUE_CONTRACTS:
            raise ValueError("unsupported FACT value contract")
        if value_contract != FACT_CODE_VALUE_CONTRACT:
            raise ValueError(
                "the first online C56b port permits only the pinned FACT code value contract"
            )
        self.dataset_path = Path(dataset).resolve()
        self.observations_path = Path(observations).resolve()
        self.source_manifest = Path(source_manifest).resolve()
        self.demo_cache_root = Path(demo_cache_root).resolve()
        self.dataset_sha256 = sha256_file(self.dataset_path)
        self.observations_sha256 = sha256_file(self.observations_path)
        if (
            expected_dataset_sha256 is not None
            and self.dataset_sha256 != expected_dataset_sha256
        ):
            raise ValueError("rollout dataset SHA256 mismatch")
        if (
            expected_observations_sha256 is not None
            and self.observations_sha256 != expected_observations_sha256
        ):
            raise ValueError("rollout observations SHA256 mismatch")
        payload = torch.load(
            self.dataset_path, map_location="cpu", weights_only=False
        )
        self.dataset_format = str(payload.get("format"))
        if self.dataset_format not in SUPPORTED_ROLLOUT_FORMATS:
            raise ValueError("unsupported FACT rollout dataset")
        self.rows = [row for row in payload["samples"] if str(row["split"]) == split]
        if not self.rows:
            raise ValueError("selected rollout split is empty")
        observation_rows = _read_jsonl(self.observations_path)
        self.observation_by_id = {
            int(row["observation_id"]): row for row in observation_rows
        }
        if len(self.observation_by_id) != len(observation_rows):
            raise ValueError("duplicate rollout observation ids")
        required_ids = {
            int(row[name])
            for row in self.rows
            for name in ("current_observation_id", "future_observation_id")
        }
        if not required_ids <= set(self.observation_by_id):
            raise ValueError("rollout observation provenance is incomplete")
        tasks = {
            str(self.observation_by_id[value]["task_language"])
            for value in required_ids
        }
        self.context_by_task = _task_context_ids(self.source_manifest, tasks)
        stats_path = self.demo_cache_root / "stats.pt"
        self.stats_sha256 = sha256_file(stats_path)
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        self.value_contract = value_contract
        if self.dataset_format == "h3wam-c60-counterfactual-failure-dataset-v1":
            if c59_overlay_root is not None:
                raise ValueError("C60 uses embedded causal labels, not a C59 overlay")
            self.labels: C59FailureOverlay | C60CausalFailureLabels = (
                C60CausalFailureLabels(
                    self.dataset_path,
                    expected_sha256=expected_dataset_sha256,
                    value_contract=value_contract,
                )
            )
            self.stream_name = "c60_causal_failure"
        else:
            if c59_overlay_root is None:
                raise ValueError("C48 requires the audited C59 failure overlay")
            self.labels = C59FailureOverlay(
                c59_overlay_root,
                source_dataset=self.dataset_path,
                value_contract=value_contract,
            )
            self.stream_name = "c48_c59_observational"
        self.episode_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.episode_to_indices[int(row["episode_id"])].append(index)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pixels(row: dict[str, Any]) -> torch.Tensor:
        with np.load(str(row["trajectory"]), allow_pickle=False) as archive:
            kind = str(row.get("kind"))
            if kind == "row":
                index = int(row["row_index"])
                agentview = archive["agentview_image"][index]
                wristview = archive["wristview_image"][index]
            elif kind == "terminal" and row.get("row_index") is None:
                agentview = archive["terminal_agentview_image"]
                wristview = archive["terminal_wristview_image"]
            else:
                raise ValueError("online C56b observation kind is not row/terminal")
            frame = preprocess_libero_cameras(
                agentview,
                wristview,
            )
        return (
            frame.mul(255).round().clamp_(0, 255).to(torch.uint8)
            .permute(0, 3, 1, 2).unsqueeze(2)[0]
            .contiguous()
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        current_observation = self.observation_by_id[
            int(row["current_observation_id"])
        ]
        future_observation = self.observation_by_id[
            int(row["future_observation_id"])
        ]
        if (
            int(current_observation["episode_id"]) != int(row["episode_id"])
            or int(future_observation["episode_id"]) != int(row["episode_id"])
            or str(current_observation["split"]) != str(row["split"])
            or str(future_observation["split"]) != str(row["split"])
        ):
            raise ValueError("rollout sample/observation identity mismatch")
        task = str(current_observation["task_language"])
        if str(future_observation["task_language"]) != task:
            raise ValueError("rollout future task language changed")
        context = _load_text_context(
            self.demo_cache_root, self.context_by_task[task]
        )
        environment_actions = row["executed_actions"].float()
        dataset_actions = environment_actions_to_dataset(environment_actions)
        valid = ~row["action_is_pad"].bool()
        roundtrip = dataset_actions_to_environment(dataset_actions)
        if valid.any() and not torch.allclose(
            roundtrip[valid], environment_actions[valid], rtol=0.0, atol=1e-7
        ):
            raise ValueError("LIBERO action/gripper round-trip failed")
        if isinstance(self.labels, C60CausalFailureLabels):
            target = self.labels.target_for(row)
        else:
            target = self.labels.for_sample(int(row["sample_id"]))
        # FACT code uses min/max 0/2, so its normalized scalar is raw - 1.
        value = torch.tensor(float(target["value_target"]) - 1.0)
        return {
            "format": ONLINE_FACT_SAMPLE_FORMAT,
            "stream": self.stream_name,
            "sample_id": f"{self.stream_name}_{int(row['sample_id']):06d}",
            "episode_id": int(row["episode_id"]),
            "input_mode": self.input_mode,
            "current_h3_input": self._pixels(current_observation),
            "future_h3_input": self._pixels(future_observation),
            "text_context": context["context"],
            "text_token_tags": context["token_tags"],
            "actions": minmax_normalize(
                dataset_actions, self.action_min, self.action_max
            ),
            "action_is_pad": row["action_is_pad"].bool(),
            "proprio": minmax_normalize(
                row["current_proprio"].float(), self.state_min, self.state_max
            ),
            "future_state": minmax_normalize(
                row["future_proprio"].float(), self.state_min, self.state_max
            ),
            "value": value,
            "action_loss_mask": torch.tensor(float(target["action_loss_mask"])),
            "future_representation_loss_mask": torch.tensor(
                float(target["future_loss_mask"])
            ),
            "future_state_loss_mask": torch.tensor(float(target["future_loss_mask"])),
            "value_loss_mask": torch.tensor(float(target["value_loss_mask"])),
            "failure_active_mask": torch.tensor(
                float(target.get("failure_active_mask", 0))
            ),
        }


def collate_online_fact(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate one homogeneous online-H3 microbatch without creating K/V."""

    if not batch:
        raise ValueError("online FACT batch is empty")
    if len({str(item["input_mode"]) for item in batch}) != 1:
        raise ValueError("pixels and VAE latents cannot share one H3 microbatch")
    if len({str(item["stream"]) for item in batch}) != 1:
        raise ValueError("FACT streams must be scheduled as separate microbatches")
    forbidden = {"video_kv_cache", "h3_features", "future_h3_target"}
    if any(forbidden & set(item) for item in batch):
        raise ValueError("online FACT samples must not contain cached H3 tensors")
    token_count = max(int(item["text_context"].shape[0]) for item in batch)
    context_width = int(batch[0]["text_context"].shape[1])
    contexts = torch.zeros(len(batch), token_count, context_width, dtype=torch.float32)
    tags = torch.zeros(len(batch), token_count, dtype=torch.long)
    text_mask = torch.zeros(len(batch), token_count, dtype=torch.bool)
    for index, item in enumerate(batch):
        context = item["text_context"].float()
        item_tags = item["text_token_tags"].long()
        if context.ndim != 2 or item_tags.shape != (context.shape[0],):
            raise ValueError("H3 text context/token-tag shape mismatch")
        if context.shape[1] != context_width:
            raise ValueError("H3 context widths differ inside a microbatch")
        contexts[index, : context.shape[0]] = context
        tags[index, : context.shape[0]] = item_tags
        text_mask[index, : context.shape[0]] = True
    tensor_keys = (
        "current_h3_input",
        "future_h3_input",
        "actions",
        "action_is_pad",
        "proprio",
        "future_state",
        "value",
        "action_loss_mask",
        "future_representation_loss_mask",
        "future_state_loss_mask",
        "value_loss_mask",
        "failure_active_mask",
    )
    result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    result.update(
        {
            "format": ONLINE_FACT_SAMPLE_FORMAT,
            "stream": str(batch[0]["stream"]),
            "input_mode": str(batch[0]["input_mode"]),
            "sample_ids": [str(item["sample_id"]) for item in batch],
            "episode_ids": [item["episode_id"] for item in batch],
            "text_context": contexts,
            "text_token_tags": tags,
            "text_mask": text_mask,
        }
    )
    return result


class OnlineFACTEpisodeMixtureSampler(Sampler[int]):
    """Official-FACT-style dataset→episode→frame mixture over online streams."""

    def __init__(
        self,
        dataset: ConcatDataset,
        *,
        dataset_weights: Sequence[float] | None = None,
        samples_per_epoch: int,
        seed: int = 6666,
        infinite: bool = True,
    ) -> None:
        if not isinstance(dataset, ConcatDataset) or not dataset.datasets:
            raise TypeError("online FACT mixture requires a non-empty ConcatDataset")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        weights = list(dataset_weights or (1.0,) * len(dataset.datasets))
        if len(weights) != len(dataset.datasets) or any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in weights
        ):
            raise ValueError("dataset weights must be finite, positive and aligned")
        self.dataset = dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.infinite = bool(infinite)
        self.epoch = 0
        self._offsets: list[int] = []
        self._episodes: list[list[list[int]]] = []
        offset = 0
        episode_counts = []
        for child in dataset.datasets:
            mapping = getattr(child, "episode_to_indices", None)
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("every online FACT child needs episode_to_indices")
            episodes = [list(indices) for indices in mapping.values()]
            if any(not values for values in episodes):
                raise ValueError("online FACT episode has no sample indices")
            self._offsets.append(offset)
            self._episodes.append(episodes)
            episode_counts.append(len(episodes))
            offset += len(child)
        effective = np.asarray(weights, dtype=np.float64) * np.asarray(
            episode_counts, dtype=np.float64
        )
        self._probabilities = effective / effective.sum()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        while True:
            generator = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            for _ in range(self.samples_per_epoch):
                dataset_index = int(
                    generator.choice(len(self._episodes), p=self._probabilities)
                )
                episodes = self._episodes[dataset_index]
                episode = episodes[int(generator.integers(len(episodes)))]
                local_index = episode[int(generator.integers(len(episode)))]
                yield self._offsets[dataset_index] + local_index
            if not self.infinite:
                return


__all__ = [
    "FACT_CODE_VALUE_CONTRACT",
    "ONLINE_FACT_SAMPLE_FORMAT",
    "OnlineFACTEpisodeMixtureSampler",
    "OnlineH3FACTDemoDataset",
    "OnlineH3FACTRolloutDataset",
    "collate_online_fact",
    "dataset_actions_to_environment",
    "environment_actions_to_dataset",
]
