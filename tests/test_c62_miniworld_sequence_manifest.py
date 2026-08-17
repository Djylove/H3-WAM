import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h3wam/build_c62_miniworld_sequence_manifest.py"
SPEC = importlib.util.spec_from_file_location("_c62_sequence_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rows():
    rows = []
    for suite in ("a", "b"):
        for episode in range(6):
            for start in range(64):
                rows.append(
                    {
                        "id": f"{suite}-{episode}-{start}",
                        "suite": suite,
                        "episode": episode,
                        "start": start,
                        "split": "train",
                    }
                )
    return rows


def test_real_history_alignment_and_no_future_leakage():
    rows = MODULE.build_sequence_rows(_rows(), history_chunks=3)
    row = next(item for item in rows if item["start"] == 40)
    assert [item["observation_start"] for item in row["history"]] == [16, 24, 32]
    assert row["history"][0]["actions_before_observation_id"] is None
    assert row["history"][1]["action_indices"] == list(range(16, 24))
    assert row["history"][2]["action_indices"] == list(range(24, 32))
    assert row["actions_before_current_indices"] == list(range(32, 40))
    assert all(
        max(item["action_indices"], default=-1) < item["observation_start"]
        for item in row["history"]
    )


def test_episode_disjoint_balanced_selection_is_deterministic():
    rows = MODULE.build_sequence_rows(_rows(), history_chunks=3)
    first = MODULE.select_episode_disjoint(
        rows,
        train_per_suite=8,
        heldout_per_suite=4,
        heldout_episodes_per_suite=2,
        seed=7,
    )
    second = MODULE.select_episode_disjoint(
        rows,
        train_per_suite=8,
        heldout_per_suite=4,
        heldout_episodes_per_suite=2,
        seed=7,
    )
    assert [[row["id"] for row in split] for split in first] == [
        [row["id"] for row in split] for split in second
    ]
    train, heldout = first
    train_episodes = {(row["suite"], row["episode"]) for row in train}
    heldout_episodes = {(row["suite"], row["episode"]) for row in heldout}
    assert not train_episodes & heldout_episodes
    assert len(train) == 16
    assert len(heldout) == 8
    assert len(heldout_episodes) == 4
