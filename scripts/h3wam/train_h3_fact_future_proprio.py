#!/usr/bin/env python3
"""Run the isolated H3-FACT F0/F1 future-proprio consequence canary.

The experiment trains three parameter-identical consequence experts on the
same batches: correct action chunks, a no-self-map shuffled-action control and
an action-independent (zero action) control.  H3 remains a frozen cache and no
action-policy code is imported or modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.fact_lite_consequence import (  # noqa: E402
    FutureProprioConsequenceModel,
    actions_for_arm,
    future_proprio_mse,
)


FACT_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
H3_INT8_CHECKPOINT_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
)
CACHE_STRATEGY = "starwam_adaptive_avg_pool1d_v1"
CACHE_BACKBONE = "H3Int8FeatureBackbone"
CACHE_QUANTIZATION = "int8_tensorwise_convrot"
ARMS = ("conditioned", "shuffled", "independent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--feature-subdir", default="h3_int8_starwam_last32_dense_v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-h3-checkpoint-sha256", action="store_true")
    parser.add_argument(
        "--expected-h3-checkpoint-sha256", default=H3_INT8_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--feature-input-scale", type=float, default=0.009606920816877307
    )
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.01,
        help="Required conditioned MSE gain over each control.",
    )
    parser.add_argument(
        "--minimum-shuffle-degradation",
        type=float,
        default=0.01,
        help="Required relative MSE degradation when val actions are shuffled.",
    )
    parser.add_argument(
        "--selection-salt", default="h3-fact-future-proprio-f1-v1"
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate ids: {path}")
    return rows


def select_rows(
    rows: list[dict[str, Any]], *, limit: int, salt: str
) -> list[dict[str, Any]]:
    if limit <= 0 or limit > len(rows):
        raise ValueError(f"limit must be in [1,{len(rows)}], got {limit}")

    def score(row: dict[str, Any]) -> tuple[str, str]:
        sample_id = str(row["id"])
        digest = hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).hexdigest()
        return digest, sample_id

    return sorted(rows, key=score)[:limit]


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(Path(row["dataset_root"]).resolve()),
        str(row["suite"]),
        int(row["episode"]),
    )


def audit_manifests(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {str(row["id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source manifest contains duplicate ids")
    for split_name, rows in (("train", train_rows), ("validation", val_rows)):
        for row in rows:
            if source_by_id.get(str(row["id"])) != row:
                raise ValueError(
                    f"{split_name} row {row['id']} is not byte-equivalent source provenance"
                )
    train_episodes = {episode_key(row) for row in train_rows}
    val_episodes = {episode_key(row) for row in val_rows}
    overlap = train_episodes & val_episodes
    if overlap:
        raise ValueError(f"train/validation episode leakage: {sorted(overlap)[:5]}")
    id_overlap = {str(row["id"]) for row in train_rows} & {
        str(row["id"]) for row in val_rows
    }
    if id_overlap:
        raise ValueError(f"train/validation window leakage: {sorted(id_overlap)[:5]}")
    return {
        "episode_key": ["dataset_root", "suite", "episode"],
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "episode_overlap": 0,
        "window_overlap": 0,
    }


def normalize_minmax(
    values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    value_range = (upper.to(values) - lower.to(values)).clamp_min(1.0e-6)
    return (2.0 * (values - lower.to(values)) / value_range - 1.0).clamp(-5.0, 5.0)


class LeRobotParquetStateReader:
    """Resolve and cache one LeRobot proprio table per episode."""

    def __init__(self) -> None:
        self._metadata: dict[Path, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
        self._states: dict[tuple[Path, int], torch.Tensor] = {}

    def _root_metadata(
        self, root: Path
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        root = root.resolve()
        if root not in self._metadata:
            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            episodes = {
                int(row["episode_index"]): row
                for row in (
                    json.loads(line)
                    for line in (root / "meta/episodes.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                )
            }
            self._metadata[root] = info, episodes
        return self._metadata[root]

    def parquet_path(self, root: Path, episode: int) -> Path:
        root = root.resolve()
        info, episodes = self._root_metadata(root)
        if episode not in episodes:
            raise ValueError(f"episode {episode} is absent from {root}/meta/episodes.jsonl")
        chunks_size = int(info.get("chunks_size", 1000))
        episode_chunk = episode // chunks_size
        metadata = episodes[episode]
        relative = str(
            info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        ).format(
            episode_chunk=episode_chunk,
            episode_index=episode,
            chunk_index=int(metadata.get("data/chunk_index", episode_chunk)),
            file_index=int(metadata.get("data/file_index", episode)),
        )
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing LeRobot parquet: {path}")
        return path

    def states(self, root: Path, episode: int) -> torch.Tensor:
        root = root.resolve()
        key = (root, int(episode))
        if key not in self._states:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError(
                    "future-proprio extraction requires pandas plus a parquet engine"
                ) from exc
            table = pd.read_parquet(
                self.parquet_path(root, int(episode)),
                columns=["observation.state"],
            )
            values = torch.stack(
                [
                    torch.from_numpy(value.copy()).float()
                    for value in table["observation.state"]
                ]
            )
            if values.ndim != 2 or values.shape[1] != 8:
                raise ValueError(
                    f"unexpected observation.state table shape for episode {episode}: "
                    f"{tuple(values.shape)}"
                )
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"non-finite observation.state in episode {episode}")
            self._states[key] = values
        return self._states[key]


class FutureProprioDataset(Dataset):
    """Strict v7 cache reader with future state joined from raw parquet."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        cache_root: Path,
        feature_subdir: str,
        source_manifest_items: int,
        action_horizon: int = 32,
        state_reader: LeRobotParquetStateReader | None = None,
    ) -> None:
        if not rows:
            raise ValueError("selected future-proprio rows are empty")
        self.rows = rows
        self.cache_root = cache_root.resolve()
        self.feature_root = self.cache_root / feature_subdir
        self.source_manifest_items = int(source_manifest_items)
        self.action_horizon = int(action_horizon)
        self.state_reader = state_reader or LeRobotParquetStateReader()
        stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.action_min = stats["action_min"].float()
        self.action_max = stats["action_max"].float()
        self.state_min = stats["state_min"].float()
        self.state_max = stats["state_max"].float()
        if tuple(self.action_min.shape) != (7,) or tuple(self.state_min.shape) != (8,):
            raise ValueError("v7 normalization stats have unexpected shapes")
        first_payload = torch.load(
            self.feature_root / f"{self.rows[0]['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not first_payload.get("checkpoint"):
            raise ValueError("H3 feature cache is missing its checkpoint identity")
        self.h3_checkpoint_path = Path(first_payload["checkpoint"]).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["id"])
        start = int(row["start"])
        future_index = start + self.action_horizon
        if bool(row.get("padded_tail", False)) or future_index >= int(row["length"]):
            raise ValueError(f"future target is outside the dense window: {sample_id}")

        feature_payload = torch.load(
            self.feature_root / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        expected = {
            "layers": (49,),
            "context_id": str(row["context_id"]),
            "action_horizon": self.action_horizon,
            "capture_token_count": 32,
            "capture_token_strategy": CACHE_STRATEGY,
            "backbone": CACHE_BACKBONE,
            "quantization": CACHE_QUANTIZATION,
            "manifest_items": self.source_manifest_items,
        }
        for key, expected_value in expected.items():
            actual = feature_payload.get(key)
            if key == "layers" and actual is not None:
                actual = tuple(actual)
            if actual != expected_value:
                raise ValueError(
                    f"feature cache contract mismatch for {sample_id}: "
                    f"{key}={actual!r}, expected {expected_value!r}"
                )
        if not math.isclose(float(feature_payload.get("timestep", -1.0)), 1.0):
            raise ValueError(f"feature cache timestep must be 1.0 for {sample_id}")
        if Path(feature_payload.get("checkpoint", "")).resolve() != self.h3_checkpoint_path:
            raise ValueError(f"mixed H3 checkpoint identities for {sample_id}")
        features = feature_payload["features"]
        if tuple(features.shape) != (1, 32, 5376):
            raise ValueError(f"unexpected H3 feature shape for {sample_id}: {features.shape}")
        if not bool(torch.isfinite(features.float()).all()):
            raise ValueError(f"non-finite H3 feature for {sample_id}")

        window = torch.load(
            self.cache_root / "windows" / f"{sample_id}.pt",
            map_location="cpu",
            weights_only=False,
        )
        actions = window["actions"][: self.action_horizon].float()
        action_is_pad = window.get(
            "action_is_pad", torch.zeros(self.action_horizon, dtype=torch.bool)
        )[: self.action_horizon].bool()
        if tuple(actions.shape) != (self.action_horizon, 7):
            raise ValueError(f"unexpected action shape for {sample_id}: {actions.shape}")
        if bool(action_is_pad.any()):
            raise ValueError(f"F1 requires an unpadded dense action chunk: {sample_id}")

        raw_states = self.state_reader.states(
            Path(row["dataset_root"]), int(row["episode"])
        )
        if future_index >= raw_states.shape[0]:
            raise ValueError(f"future parquet index is out of range for {sample_id}")
        cached_current = window["state"].float()
        raw_current = raw_states[start].float()
        current_parity_max_abs = float((cached_current - raw_current).abs().max())
        if current_parity_max_abs > 1.0e-6:
            raise ValueError(
                f"cache/parquet current-state mismatch for {sample_id}: "
                f"max_abs={current_parity_max_abs}"
            )
        future = raw_states[future_index].float()
        return {
            "sample_id": sample_id,
            "features": features,
            "actions": normalize_minmax(actions, self.action_min, self.action_max),
            "current_proprio": normalize_minmax(
                cached_current, self.state_min, self.state_max
            ),
            "future_proprio": normalize_minmax(
                future, self.state_min, self.state_max
            ),
            "current_parity_max_abs": current_parity_max_abs,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "features": torch.stack([item["features"] for item in items]),
        "actions": torch.stack([item["actions"] for item in items]),
        "current_proprio": torch.stack(
            [item["current_proprio"] for item in items]
        ),
        "future_proprio": torch.stack([item["future_proprio"] for item in items]),
        "current_parity_max_abs": max(
            float(item["current_parity_max_abs"]) for item in items
        ),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        **{
            key: batch[key].to(device=device, non_blocking=True)
            for key in ("features", "actions", "current_proprio", "future_proprio")
        },
    }


@torch.inference_mode()
def evaluate(
    models: dict[str, FutureProprioConsequenceModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    for model in models.values():
        model.eval()
    sums = Counter()
    state_dims = {int(model.state_dim) for model in models.values()}
    if len(state_dims) != 1:
        raise ValueError("all consequence arms must use the same state dimension")
    state_dim = next(iter(state_dims))
    per_dim = {
        name: torch.zeros(state_dim, dtype=torch.float64)
        for name in (
            "conditioned_true",
            "conditioned_shuffled",
            "shuffled_train_true",
            "independent",
        )
    }
    items = 0
    max_parity = 0.0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        inputs = {
            "conditioned_true": (
                models["conditioned"],
                actions_for_arm(batch["actions"], "conditioned"),
            ),
            "conditioned_shuffled": (
                models["conditioned"],
                actions_for_arm(batch["actions"], "shuffled"),
            ),
            "shuffled_train_true": (
                models["shuffled"],
                actions_for_arm(batch["actions"], "conditioned"),
            ),
            "independent": (
                models["independent"],
                actions_for_arm(batch["actions"], "independent"),
            ),
        }
        for name, (model, actions) in inputs.items():
            prediction = model(batch["current_proprio"], batch["features"], actions)
            squared = (
                prediction.float() - batch["future_proprio"].float()
            ).square()
            sums[name] += float(squared.sum())
            per_dim[name] += squared.sum(dim=0).double().cpu()
        items += int(batch["actions"].shape[0])
        max_parity = max(max_parity, float(batch["current_parity_max_abs"]))
    elements = items * state_dim
    return {
        name: {
            "normalized_mse": sums[name] / elements,
            "per_dimension_mse": (per_dim[name] / items).tolist(),
        }
        for name in per_dim
    } | {
        "samples": items,
        "cache_parquet_current_state_max_abs": max_parity,
    }


def train(
    models: dict[str, FutureProprioConsequenceModel],
    loader: DataLoader,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        for name, model in models.items()
    }
    iterator = iter(loader)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, device)
        item: dict[str, Any] = {
            "step": step,
            "sample_ids": batch["sample_ids"],
            "loss": {},
            "grad_norm": {},
        }
        for name in ARMS:
            model = models[name]
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            _, loss = future_proprio_mse(
                model,
                current_proprio=batch["current_proprio"],
                h3_features=batch["features"],
                candidate_actions=actions_for_arm(batch["actions"], name),
                future_proprio=batch["future_proprio"],
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {name} loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0.0:
                raise FloatingPointError(
                    f"non-finite/zero {name} gradient at step {step}: "
                    f"{float(grad_norm)}"
                )
            optimizer.step()
            item["loss"][name] = float(loss.detach())
            item["grad_norm"][name] = float(grad_norm.detach())
        if step == 1 or step == steps or step % 25 == 0:
            history.append(item)
            print(json.dumps(item), flush=True)
    return history


def fresh_restore_probe(
    models: dict[str, FutureProprioConsequenceModel],
    batch: dict[str, Any],
    *,
    model_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    batch = move_batch(batch, device)
    states = {
        name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    max_abs = 0.0
    with torch.inference_mode():
        for name, model in models.items():
            expected = model(
                batch["current_proprio"],
                batch["features"],
                actions_for_arm(batch["actions"], name),
            ).detach().cpu()
            restored = FutureProprioConsequenceModel(**model_kwargs).to(device)
            restored.load_state_dict(states[name], strict=True)
            actual = restored(
                batch["current_proprio"],
                batch["features"],
                actions_for_arm(batch["actions"], name),
            ).detach().cpu()
            max_abs = max(max_abs, float((expected - actual).abs().max()))
    return states, max_abs


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(abs(reference), 1.0e-12)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("refusing to overwrite F1 report or checkpoint")
    if min(
        args.steps,
        args.train_limit,
        args.val_limit,
        args.batch_size,
        args.action_horizon,
        args.hidden_dim,
    ) <= 0:
        raise ValueError("steps, limits, batch size and dimensions must be positive")
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2 for a no-self-map shuffled arm")
    if args.train_limit % args.batch_size or args.val_limit % args.batch_size:
        raise ValueError("train-limit and val-limit must be divisible by batch-size")
    if args.minimum_relative_improvement < 0 or args.minimum_shuffle_degradation < 0:
        raise ValueError("mechanism thresholds must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    source_manifest = args.source_manifest.resolve()
    cache_root = args.cache_root.resolve()
    all_train_rows = read_jsonl(train_manifest)
    all_val_rows = read_jsonl(val_manifest)
    source_rows = read_jsonl(source_manifest)
    split_audit = audit_manifests(all_train_rows, all_val_rows, source_rows)
    train_rows = select_rows(
        all_train_rows, limit=args.train_limit, salt=f"{args.selection_salt}|train"
    )
    val_rows = select_rows(
        all_val_rows, limit=args.val_limit, salt=f"{args.selection_salt}|validation"
    )
    reader = LeRobotParquetStateReader()
    dataset_kwargs = {
        "cache_root": cache_root,
        "feature_subdir": args.feature_subdir,
        "source_manifest_items": len(source_rows),
        "action_horizon": args.action_horizon,
        "state_reader": reader,
    }
    train_dataset = FutureProprioDataset(train_rows, **dataset_kwargs)
    val_dataset = FutureProprioDataset(val_rows, **dataset_kwargs)
    if train_dataset.h3_checkpoint_path != val_dataset.h3_checkpoint_path:
        raise ValueError("train/validation caches use different H3 checkpoints")
    actual_h3_checkpoint_sha256 = None
    if args.verify_h3_checkpoint_sha256:
        actual_h3_checkpoint_sha256 = sha256_file(train_dataset.h3_checkpoint_path)
        if actual_h3_checkpoint_sha256 != args.expected_h3_checkpoint_sha256:
            raise ValueError(
                "H3 checkpoint SHA256 mismatch: "
                f"{actual_h3_checkpoint_sha256} != {args.expected_h3_checkpoint_sha256}"
            )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model_kwargs = {
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": args.action_horizon,
        "h3_feature_dim": 5376,
        "hidden_dim": args.hidden_dim,
        "feature_input_scale": args.feature_input_scale,
    }
    initial = FutureProprioConsequenceModel(**model_kwargs)
    models = {
        name: copy.deepcopy(initial).to(device)
        for name in ARMS
    }
    initial_metrics = evaluate(models, val_loader, device)
    started = time.perf_counter()
    history = train(
        models,
        train_loader,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )
    final_metrics = evaluate(models, val_loader, device)
    probe_batch = next(iter(val_loader))
    state_dicts, restore_max_abs = fresh_restore_probe(
        models, probe_batch, model_kwargs=model_kwargs, device=device
    )

    conditioned = float(final_metrics["conditioned_true"]["normalized_mse"])
    independent = float(final_metrics["independent"]["normalized_mse"])
    shuffled_train = float(final_metrics["shuffled_train_true"]["normalized_mse"])
    shuffled_eval = float(final_metrics["conditioned_shuffled"]["normalized_mse"])
    independent_gain = relative_gain(independent, conditioned)
    shuffled_train_gain = relative_gain(shuffled_train, conditioned)
    shuffle_degradation = (shuffled_eval - conditioned) / max(conditioned, 1.0e-12)
    finite = all(
        math.isfinite(float(final_metrics[name]["normalized_mse"]))
        for name in (
            "conditioned_true",
            "conditioned_shuffled",
            "shuffled_train_true",
            "independent",
        )
    )
    mechanism_pass = (
        finite
        and restore_max_abs == 0.0
        and independent_gain >= args.minimum_relative_improvement
        and shuffled_train_gain >= args.minimum_relative_improvement
        and shuffle_degradation >= args.minimum_shuffle_degradation
    )

    checkpoint_payload = {
        "schema_version": 1,
        "classification": "novel_composition_fact_lite_consequence_only",
        "completed_steps": args.steps,
        "model_kwargs": model_kwargs,
        "models": state_dicts,
        "contract": {
            "fact_commit": FACT_COMMIT,
            "h3_checkpoint_path": str(train_dataset.h3_checkpoint_path),
            "h3_checkpoint_sha256": (
                actual_h3_checkpoint_sha256 or args.expected_h3_checkpoint_sha256
            ),
            "h3_checkpoint_sha256_verified": args.verify_h3_checkpoint_sha256,
            "future_target": "raw_lerobot_observation.state[start+32]",
            "current_state_parity": "cache_state_equals_raw_parquet_state[start]",
            "action_boundary": "candidate_actions_detached_inside_consequence_forward",
            "action_generator_present": False,
            "arms": list(ARMS),
        },
    }
    atomic_torch_save(checkpoint_payload, args.checkpoint.resolve())
    report = {
        "experiment_id": "h3_fact_lite_future_proprio_f1_v1",
        "classification": "novel_composition",
        "status": (
            "PASS_MECHANISM_GATE" if mechanism_pass else "FAIL_MECHANISM_GATE"
        ),
        "claim_boundary": (
            "Future-proprio consequence mechanism only; no value ranking, "
            "failure-aware learning, action-policy improvement or LIBERO success claim."
        ),
        "source": {
            "fact_commit": FACT_COMMIT,
            "h3_checkpoint_path": str(train_dataset.h3_checkpoint_path),
            "h3_checkpoint_sha256": (
                actual_h3_checkpoint_sha256 or args.expected_h3_checkpoint_sha256
            ),
            "h3_checkpoint_sha256_verified": args.verify_h3_checkpoint_sha256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest": str(val_manifest),
            "validation_manifest_sha256": sha256_file(val_manifest),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "cache_root": str(cache_root),
            "stats_sha256": sha256_file(cache_root / "stats.pt"),
            "feature_subdir": args.feature_subdir,
        },
        "data": {
            **split_audit,
            "train_selected": len(train_rows),
            "validation_selected": len(val_rows),
            "train_selected_ids_sha256": sha256_ids(
                [str(row["id"]) for row in train_rows]
            ),
            "validation_selected_ids_sha256": sha256_ids(
                [str(row["id"]) for row in val_rows]
            ),
            "train_tasks": len({str(row["task"]) for row in train_rows}),
            "validation_tasks": len({str(row["task"]) for row in val_rows}),
            "train_suites": dict(Counter(str(row["suite"]) for row in train_rows)),
            "validation_suites": dict(Counter(str(row["suite"]) for row in val_rows)),
            "action_horizon": args.action_horizon,
            "future_offset": args.action_horizon,
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
            "device": str(device),
            "duration_seconds": time.perf_counter() - started,
            "history": history,
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mechanism": {
            "conditioned_gain_over_independent": independent_gain,
            "conditioned_gain_over_shuffled_train": shuffled_train_gain,
            "conditioned_shuffle_degradation": shuffle_degradation,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "minimum_shuffle_degradation": args.minimum_shuffle_degradation,
            "all_metrics_finite": finite,
            "fresh_restore_max_abs": restore_max_abs,
            "future_to_action_generator_gradient_possible": False,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": None,
        "next": (
            "If PASS, repeat at 500 steps and add future-H3/value only as separate "
            "children. Failure rollout ranking and best-of-N remain blocked."
            if mechanism_pass
            else "Do not add failure-aware value or best-of-N; inspect mapping and action dependence."
        ),
    }
    report["checkpoint_sha256"] = sha256_file(args.checkpoint.resolve())
    atomic_json_save(report, args.output.resolve())
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
