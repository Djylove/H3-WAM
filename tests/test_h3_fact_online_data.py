import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

from fastwam.models.h3wam.fact_online_data import (
    OnlineFACTEpisodeMixtureSampler,
    OnlineH3FACTDemoDataset,
    OnlineH3FACTRolloutDataset,
    collate_online_fact,
    dataset_actions_to_environment,
    environment_actions_to_dataset,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_common(root: Path, task: str = "pick object") -> tuple[Path, Path]:
    cache = root / "cache"
    (cache / "windows").mkdir(parents=True)
    (cache / "contexts").mkdir(parents=True)
    torch.save(
        {
            "action_min": torch.zeros(7),
            "action_max": torch.ones(7),
            "state_min": torch.zeros(8),
            "state_max": torch.ones(8),
        },
        cache / "stats.pt",
    )
    torch.save(
        {
            "context": torch.arange(3 * 5120, dtype=torch.float32).reshape(1, 3, 5120),
            "token_tags": torch.ones(3, dtype=torch.long),
            "text_only": True,
            "task": task,
        },
        cache / "contexts/task_ctx.pt",
    )
    source = root / "source.jsonl"
    return cache, source


def test_demo_dataset_runs_to_online_h3_boundary_without_kv(tmp_path: Path):
    cache, source = write_common(tmp_path)
    rows = [
        {
            "id": f"demo_{start}",
            "suite": "libero_goal",
            "episode": 7,
            "start": start,
            "split": "train",
            "task": "pick object",
            "context_id": "task_ctx",
        }
        for start in (0, 32)
    ]
    write_jsonl(source, rows)
    split = tmp_path / "split.jsonl"
    write_jsonl(split, rows)
    for position, row in enumerate(rows):
        torch.save(
            {
                "first_frame_latents": torch.full((1, 24, 1, 2, 4), position + 1.0),
                "video_latents": torch.stack(
                    [torch.full((1, 24, 2, 4), float(index)) for index in range(12)],
                    dim=2,
                ),
                "actions": torch.full((32, 7), 0.25),
                "action_is_pad": torch.zeros(32, dtype=torch.bool),
                "state": torch.full((8,), float(position)),
            },
            cache / "windows" / f"{row['id']}.pt",
        )

    dataset = OnlineH3FACTDemoDataset(split, source, cache)
    item = dataset[0]
    assert item["stream"] == "expert_demo"
    assert item["input_mode"] == "vae_latents"
    assert tuple(item["current_h3_input"].shape) == (24, 1, 2, 4)
    assert tuple(item["future_h3_input"].shape) == (24, 1, 2, 4)
    torch.testing.assert_close(item["future_h3_input"], torch.full_like(item["future_h3_input"], 11.0))
    torch.testing.assert_close(item["future_state"], torch.ones(8))
    assert float(item["future_state_loss_mask"]) == 1.0
    assert float(item["value_loss_mask"]) == 0.0
    assert "video_kv_cache" not in item
    assert "h3_features" not in item

    batch = collate_online_fact([item])
    assert tuple(batch["current_h3_input"].shape) == (1, 24, 1, 2, 4)
    assert tuple(batch["text_context"].shape) == (1, 3, 5120)
    assert batch["text_mask"].all()
    assert "video_kv_cache" not in batch


def make_c60(root: Path, cache: Path, source: Path) -> tuple[Path, Path]:
    trajectory = root / "trajectory.npz"
    frames = np.zeros((2, 12, 16, 3), dtype=np.uint8)
    frames[1] = 255
    np.savez(
        trajectory,
        agentview_image=frames,
        wristview_image=frames,
        terminal_agentview_image=frames[1],
        terminal_wristview_image=frames[1],
    )
    write_jsonl(
        source,
        [
            {
                "id": "context_source",
                "suite": "libero_goal",
                "episode": 0,
                "start": 0,
                "split": "train",
                "task": "pick object",
                "context_id": "task_ctx",
            }
        ],
    )
    dataset_path = root / "c60.pt"
    observations_path = root / "observations.jsonl"
    write_jsonl(
        observations_path,
        [
            {
                "observation_id": index,
                "episode_id": 0,
                "split": "train",
                "trajectory": str(trajectory),
                "kind": "row" if index == 0 else "terminal",
                "row_index": index if index == 0 else None,
                "step": index * 32,
                "task_language": "pick object",
            }
            for index in (0, 1)
        ],
    )
    sample = {
        "sample_id": 0,
        "episode_id": 0,
        "split": "train",
        "success": False,
        "current_observation_id": 0,
        "future_observation_id": 1,
        "current_step": 0,
        "future_step": 32,
        "failure_active_from_step": 0,
        "failure_active": True,
        "executed_actions": torch.zeros(32, 7),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "current_proprio": torch.full((8,), 0.25),
        "future_proprio": torch.full((8,), 0.75),
        "action_loss_mask": 0.0,
        "future_loss_mask": 1.0,
        "value_loss_mask": 1.0,
        "fact_code_value_raw": 1.5,
        "fact_paper_progress_target": 0.0,
    }
    torch.save(
        {
            "format": "h3wam-c60-counterfactual-failure-dataset-v1",
            "action_contract": "all branch actions masked from imitation",
            "counts": {"train": {"samples": 1}},
            "episodes": [
                {
                    "episode_index": 0,
                    "source_ordinal": 10,
                    "split": "train",
                    "failure_episode": True,
                    "failure_active_from_frame": 0,
                    "annotation_source": "state_aligned_counterfactual_action_intervention",
                }
            ],
            "samples": [sample],
        },
        dataset_path,
    )
    return dataset_path, observations_path


def test_c60_dataset_decodes_rgb_and_keeps_failure_masks(tmp_path: Path):
    cache, source = write_common(tmp_path)
    dataset_path, observations = make_c60(tmp_path, cache, source)
    dataset = OnlineH3FACTRolloutDataset(
        dataset_path,
        observations,
        source,
        cache,
    )
    item = dataset[0]
    assert item["stream"] == "c60_causal_failure"
    assert item["input_mode"] == "pixels"
    assert tuple(item["current_h3_input"].shape) == (3, 1, 224, 448)
    assert item["current_h3_input"].dtype == torch.uint8
    assert int(item["current_h3_input"].max()) == 0
    assert int(item["future_h3_input"].min()) == 255
    assert float(item["action_loss_mask"]) == 0.0
    assert float(item["future_representation_loss_mask"]) == 1.0
    assert float(item["future_state_loss_mask"]) == 1.0
    assert float(item["value_loss_mask"]) == 1.0
    assert float(item["value"]) == pytest.approx(0.5)
    assert "video_kv_cache" not in item

    batch = collate_online_fact([item])
    assert tuple(batch["current_h3_input"].shape) == (1, 3, 1, 224, 448)
    assert batch["stream"] == "c60_causal_failure"


def test_action_roundtrip_and_mixed_input_rejection(tmp_path: Path):
    actions = torch.randn(32, 7)
    torch.testing.assert_close(
        dataset_actions_to_environment(environment_actions_to_dataset(actions)),
        actions,
        rtol=0,
        atol=2e-7,
    )
    cache, source = write_common(tmp_path)
    dataset_path, observations = make_c60(tmp_path, cache, source)
    rollout = OnlineH3FACTRolloutDataset(
        dataset_path, observations, source, cache
    )[0]
    latent = dict(rollout)
    latent["stream"] = "expert_demo"
    latent["input_mode"] = "vae_latents"
    latent["current_h3_input"] = torch.zeros(24, 1, 2, 4)
    latent["future_h3_input"] = torch.zeros(24, 1, 2, 4)
    with pytest.raises(ValueError, match="pixels and VAE latents"):
        collate_online_fact([rollout, latent])


class _EpisodeDataset(Dataset):
    def __init__(self, episodes: list[list[int]]):
        self.values = [value for episode in episodes for value in episode]
        self.episode_to_indices = {}
        offset = 0
        for episode_id, episode in enumerate(episodes):
            self.episode_to_indices[episode_id] = list(
                range(offset, offset + len(episode))
            )
            offset += len(episode)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def test_episode_mixture_is_deterministic_and_uses_concat_offsets():
    children = [_EpisodeDataset([[0, 1], [2]]), _EpisodeDataset([[10, 11]])]
    dataset = ConcatDataset(children)
    first = OnlineFACTEpisodeMixtureSampler(
        dataset,
        dataset_weights=(1.0, 4.0),
        samples_per_epoch=32,
        seed=123,
        infinite=False,
    )
    second = OnlineFACTEpisodeMixtureSampler(
        dataset,
        dataset_weights=(1.0, 4.0),
        samples_per_epoch=32,
        seed=123,
        infinite=False,
    )
    left = list(first)
    right = list(second)
    assert left == right
    assert len(left) == 32
    assert all(0 <= index < len(dataset) for index in left)
    # Child 1 begins at concat index 3 and has the larger declared weight.
    assert sum(index >= 3 for index in left) > sum(index < 3 for index in left)


def test_cache_launchers_are_retired_and_online_launcher_is_commit_gated():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "launch_c60_fact_cache_8gpu.sh",
        "launch_c60_fact_layerwise30_kv_8gpu.sh",
    ):
        result = subprocess.run(
            ["bash", str(root / "scripts/h3wam" / name)],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 64
        assert "No cache process was started" in result.stderr
    environment = dict(os.environ)
    environment.pop("C58_ONLINE_INTERFACE_COMMIT", None)
    result = subprocess.run(
        ["bash", str(root / "scripts/h3wam/launch_c56b_fact_online_8gpu.sh")],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 65
    assert "no GPU process started" in result.stderr
