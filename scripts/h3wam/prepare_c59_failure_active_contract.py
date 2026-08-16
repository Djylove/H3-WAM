#!/usr/bin/env python3
"""Build an auditable FACT failure-rollout overlay without copying C48 tensors.

FACT masks action imitation for an entire failed episode, while its value penalty
starts only after an annotated failure onset.  The public FACT repository expects
``meta/failure_rollouts.jsonl`` but does not publish an onset annotator.  This
script therefore refuses to infer onset from terminal failure or object motion.
Unannotated failures remain valid future-consequence data and receive an onset
sentinel after the final local frame, so their bad actions are not imitated and
their value target is not spuriously penalized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT = "h3wam-c59-fact-failure-active-overlay-v1"
SOURCE_FORMAT = "h3wam-c48-fact-dense-value-dataset-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def index_annotations(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id in indexed:
            raise ValueError(f"duplicate failure annotation for episode {episode_id}")
        onset = row.get("failure_active_from_step")
        if onset is None:
            raise ValueError(f"episode {episode_id} annotation has no onset step")
        if int(onset) < 0:
            raise ValueError(f"episode {episode_id} onset must be non-negative")
        if not str(row.get("annotation_source", "")).strip():
            raise ValueError(f"episode {episode_id} annotation source is required")
        if not str(row.get("evidence", "")).strip():
            raise ValueError(f"episode {episode_id} annotation evidence is required")
        indexed[episode_id] = {**row, "failure_active_from_step": int(onset)}
    return indexed


def derive_episode_overlay(
    episode_id: int,
    samples: list[dict[str, Any]],
    annotation: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one official-loader row and sample-aligned audit labels."""

    ordered = sorted(samples, key=lambda row: (int(row["current_step"]), int(row["sample_id"])))
    if not ordered:
        raise ValueError(f"episode {episode_id} has no samples")
    success_values = {bool(row["success"]) for row in ordered}
    if len(success_values) != 1:
        raise ValueError(f"episode {episode_id} mixes outcomes")
    success = success_values.pop()
    terminal_steps = {int(row["terminal_step"]) for row in ordered}
    if len(terminal_steps) != 1:
        raise ValueError(f"episode {episode_id} mixes terminal steps")
    terminal_step = terminal_steps.pop()
    if success and annotation is not None:
        raise ValueError(f"successful episode {episode_id} cannot have failure onset")

    failure_episode = not success
    onset_step = None if annotation is None else int(annotation["failure_active_from_step"])
    if onset_step is not None and onset_step > terminal_step:
        raise ValueError(
            f"episode {episode_id} onset {onset_step} exceeds terminal {terminal_step}"
        )
    if failure_episode and onset_step is not None:
        active_positions = [
            position
            for position, row in enumerate(ordered)
            if int(row["future_step"]) >= onset_step
        ]
        active_from_frame = active_positions[0] if active_positions else len(ordered)
    else:
        # FACT's loader treats a listed episode as a failure for action masking.
        # A one-past-end sentinel prevents an invented value penalty.
        active_from_frame = len(ordered)

    labels = []
    for position, row in enumerate(ordered):
        active = failure_episode and onset_step is not None and position >= active_from_frame
        future_step = int(row["future_step"])
        remaining = max(0.0, float(terminal_step - future_step) / max(float(terminal_step), 1.0))
        progress = min(1.0, max(0.0, float(future_step) / max(float(terminal_step), 1.0)))
        labels.append(
            {
                "sample_id": int(row["sample_id"]),
                "episode_id": episode_id,
                "local_frame_index": position,
                "failure_episode_mask": int(failure_episode),
                "failure_active_mask": int(active),
                "action_loss_mask": int(success),
                "future_loss_mask": 1,
                # This is the exact orientation used by the pinned FACT code:
                # remaining-time plus a positive failure penalty, normalized later.
                "fact_code_value_raw": remaining + (1.0 if active else 0.0),
                # Kept separately because arXiv Eq. 6 uses increasing progress and
                # subtracts the penalty.  Never silently conflate the two contracts.
                "fact_paper_progress_target": max(0.0, progress - (1.0 if active else 0.0)),
                "source_c48_value_target": float(row["value_target"]),
            }
        )

    loader_row = {
        "episode_index": episode_id,
        "failure_episode": failure_episode,
        "failure_active_from_frame": active_from_frame,
        "failure_active_from_step": onset_step,
        "failure_onset_available": onset_step is not None,
        "episode_length": len(ordered),
        "terminal_step": terminal_step,
        "annotation_source": None if annotation is None else str(annotation["annotation_source"]),
        "evidence": None if annotation is None else str(annotation["evidence"]),
    }
    return loader_row, labels


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c48-dataset", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = args.c48_dataset.resolve()
    annotations_path = None if args.annotations is None else args.annotations.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite C59 output: {output_root}")

    import torch

    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if payload.get("format") != SOURCE_FORMAT:
        raise ValueError(f"C59 requires {SOURCE_FORMAT}")
    samples = list(payload["samples"])
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        by_episode[int(row["episode_id"])].append(row)
    annotations = index_annotations(load_jsonl(annotations_path))
    unknown = sorted(set(annotations) - set(by_episode))
    if unknown:
        raise ValueError(f"annotations refer to unknown episodes: {unknown[:10]}")

    loader_rows = []
    sample_labels = []
    review_queue = []
    for episode_id in sorted(by_episode):
        row, labels = derive_episode_overlay(
            episode_id, by_episode[episode_id], annotations.get(episode_id)
        )
        loader_rows.append(row)
        sample_labels.extend(labels)
        if row["failure_episode"] and not row["failure_onset_available"]:
            first = min(by_episode[episode_id], key=lambda item: int(item["sample_id"]))
            review_queue.append(
                {
                    "episode_id": episode_id,
                    "suite": str(first["suite"]),
                    "task": int(first["task"]),
                    "trial": int(first["trial"]),
                    "terminal_step": int(first["terminal_step"]),
                    "required_output": {
                        "episode_id": episode_id,
                        "failure_active_from_step": "integer simulator step",
                        "annotation_source": "human_review|explicit_intervention|simulator_irreversible_event",
                        "evidence": "trace/video/event locator",
                    },
                }
            )

    output_root.mkdir(parents=True)
    failure_path = output_root / "failure_rollouts.jsonl"
    labels_path = output_root / "sample_labels.jsonl"
    review_path = output_root / "failure_onset_review_queue.jsonl"
    atomic_write(failure_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in loader_rows))
    atomic_write(labels_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_labels))
    atomic_write(review_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in review_queue))
    failures = sum(bool(row["failure_episode"]) for row in loader_rows)
    annotated = sum(
        bool(row["failure_episode"] and row["failure_onset_available"])
        for row in loader_rows
    )
    report = {
        "format": FORMAT,
        "status": "PASS_C59_FAILURE_OVERLAY_CONTRACT",
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "annotations": None if annotations_path is None else str(annotations_path),
        "annotations_sha256": None if annotations_path is None else sha256_file(annotations_path),
        "episodes": len(loader_rows),
        "failure_episodes": failures,
        "annotated_failure_onsets": annotated,
        "unannotated_failure_onsets": failures - annotated,
        "samples": len(sample_labels),
        "action_contract": "all failure episode actions masked; successful demonstration actions supervised",
        "future_contract": "all observed success and failure futures supervised",
        "value_contract": (
            "pinned FACT code remaining-time orientation and paper Eq.6 progress orientation "
            "are stored as separate fields; value penalty activates only after explicit onset"
        ),
        "prohibited_inference": (
            "terminal failure, joint displacement, contact, timeout, or heuristic classification "
            "alone must not be converted into failure_active_from_step"
        ),
        "files": {
            "failure_rollouts": {"path": str(failure_path), "sha256": sha256_file(failure_path)},
            "sample_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "review_queue": {"path": str(review_path), "sha256": sha256_file(review_path)},
        },
    }
    atomic_write(output_root / "COMPLETED.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
