from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fastwam.models.h3wam.c57_lingbot_libero_trace import (
    C57FeedbackEnv,
    C57PersistentFeedbackClient,
)


def observation(value: int) -> dict:
    return {
        "agentview_image": np.full((4, 5, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((3, 4, 3), value + 1, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


class Connection:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def recv(self):
        message = self.messages[-1]
        if message["command"] == "c57_reset":
            return {"ok": True}
        total = message["action_count"]
        return {"ok": True, "action_count": total, "committed": total == 8}


class Env:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return observation(self.steps), 0.0, False, {}


def test_real_libero_trace_is_reset_obs4_commit8() -> None:
    connection = Connection()
    client = C57PersistentFeedbackClient(
        connection, episode_key="suite/task/trial0", task="pick object"
    )
    client.reset()
    client.mark_prediction()
    env = C57FeedbackEnv(Env(), client)
    for index in range(8):
        env.step(np.asarray([index, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    commands = [message["command"] for message in connection.messages]
    assert commands == ["c57_reset", "c57_feedback", "c57_feedback"]
    assert [message.get("action_count") for message in connection.messages[1:]] == [4, 8]
    assert all(len(message["executed_environment_actions"]) == 4 for message in connection.messages[1:])
    assert connection.messages[1]["agentview_image_shape"] == (4, 5, 3)
    assert not client.counters.prediction_open
    assert client.counters.transaction_id == 1


def test_feedback_is_fail_closed() -> None:
    client = C57PersistentFeedbackClient(Connection(), episode_key="ep", task="task")
    with pytest.raises(RuntimeError, match="before a C57 prediction"):
        client.record_step(np.zeros(7, dtype=np.float32), observation(0))
    client.reset()
    client.mark_prediction()
    with pytest.raises(RuntimeError, match="already awaits"):
        client.mark_prediction()


def test_terminal_tail_never_commits() -> None:
    connection = Connection()
    client = C57PersistentFeedbackClient(connection, episode_key="ep", task="task")
    client.reset()
    client.mark_prediction()
    for _ in range(3):
        client.record_step(np.zeros(7, dtype=np.float32), observation(0))
    client.discard_terminal_tail()
    assert [message["command"] for message in connection.messages] == ["c57_reset"]


def test_heldout_plan_is_episode_disjoint_and_seed_frozen(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    sequence = tmp_path / "sequence.jsonl"
    selected = tmp_path / "selected.jsonl"
    plan = tmp_path / "plan.json"
    train.write_text(
        json.dumps({"id": "train0", "suite": "goal", "episode": 0}) + "\n"
    )
    heldout_rows = [
        {"id": f"val{episode}", "suite": "goal", "episode": episode}
        for episode in (1, 2)
    ]
    heldout.write_text("".join(json.dumps(row) + "\n" for row in heldout_rows))
    sequence.write_text(
        "".join(
            json.dumps(
                {
                    "sequence_schema": "c57_lingbot_replan8_v1",
                    "current_id": row["id"],
                    "history": [{"action_source_id": "unused"}],
                }
            )
            + "\n"
            for row in heldout_rows
        )
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/h3wam/freeze_c57_heldout_eval_plan.py",
            "--train-source-manifest", str(train),
            "--heldout-source-manifest", str(heldout),
            "--heldout-sequence-manifest", str(sequence),
            "--selected-manifest", str(selected),
            "--plan", str(plan),
            "--per-suite", "2",
        ],
        check=True,
    )
    frozen = json.loads(plan.read_text())
    chosen = [json.loads(line) for line in selected.read_text().splitlines()]
    assert frozen["checkpoint_milestones"] == list(range(200, 5001, 200))
    assert frozen["promotion_checkpoint"] == 5000
    assert len({row["eval_flow_seed"] for row in chosen}) == 2


def test_sequence_capacity_failure_does_not_publish_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "sequence.jsonl"
    audit = tmp_path / "audit.json"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"sample{start}",
                    "suite": "goal",
                    "episode": 1,
                    "start": start,
                }
            )
            + "\n"
            for start in range(130)
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/h3wam/build_c57_lingbot_sequence_manifest.py",
            str(source),
            str(output),
            "--audit", str(audit),
            "--max-history-chunks", "15",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "persistent token capacity exceeded" in result.stderr
    assert not output.exists()
    assert not audit.exists()
