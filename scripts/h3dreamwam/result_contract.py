"""Fail-closed helpers for consuming H3-WAM rollout result files."""

from __future__ import annotations

import json
from pathlib import Path


def completed_rollout(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"rollout result is not complete: {path}")
    expected_tasks = int(payload.get("expected_tasks", -1))
    expected_episodes = int(payload.get("expected_episodes", -1))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_tasks:
        raise ValueError(f"rollout result has an incomplete task count: {path}")
    episodes = sum(len(task.get("episodes", ())) for task in tasks)
    if episodes != expected_episodes or payload.get("finished_episodes") != episodes:
        raise ValueError(f"rollout result has an incomplete episode count: {path}")
    return payload


def is_completed_rollout(path: Path) -> bool:
    try:
        completed_rollout(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True
