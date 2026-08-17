import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py"
    spec = importlib.util.spec_from_file_location("_test_c66_k1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K1 = _load()


def _full_sequence():
    history = []
    for index, frame_count in enumerate([3, 2, 2, 2, 2, 2, 2]):
        history.append(
            {
                "observation_kv": {0: {"k": torch.tensor([index]), "v": torch.tensor([index])}},
                "observed_frame_count": frame_count,
                "executed_actions": torch.full((1, 8, 2), float(index)),
                "proprio": torch.tensor([[float(index), 0.0, 0.0]]),
            }
        )
    return {
        "id": "sample",
        "suite": "libero_object",
        "episode": 7,
        "current": {
            "text_context": torch.ones(1, 2, 4),
            "text_mask": torch.ones(1, 2, dtype=torch.bool),
            "proprio": torch.ones(1, 3),
        },
        "current_kv": {0: {"k": torch.ones(1), "v": torch.ones(1)}},
        "history": history,
    }


def test_k1_keeps_only_latest_complete_chunk_and_distinct_shuffle_donor():
    result = K1.keep_latest_complete_chunk(_full_sequence())
    assert len(result["history"]) == 1
    assert result["history"][0]["observed_frame_count"] == 2
    assert torch.equal(result["history"][0]["executed_actions"], torch.full((1, 8, 2), 6.0))
    assert torch.equal(result["k1_shuffle_actions"], torch.full((1, 8, 2), 5.0))
    assert not torch.equal(
        result["history"][0]["executed_actions"], result["k1_shuffle_actions"]
    )
    assert result["k1_absolute_prefix"] == {
        "frame_st_id": 13,
        "action_st_id": 48,
        "next_update_id": 12,
    }


class _State:
    def __init__(self):
        self.frame_st_id = 0
        self.action_st_id = 0
        self.next_update_id = 0


class _Policy:
    def __init__(self):
        self.executed = None
        self.state = None

    def new_persistent_state(self, episode_key):
        assert episode_key == "libero_object:7"
        self.state = _State()
        return self.state

    def commit_executed_feedback(self, state, **kwargs):
        self.executed = kwargs["executed_actions"].clone()
        state.frame_st_id += kwargs["observed_frame_count"]
        state.action_st_id += int(kwargs["executed_actions"].shape[1])
        state.next_update_id += 2

    def __call__(self, noisy, timesteps, **kwargs):
        assert kwargs["persistent_state"] is self.state
        return noisy + self.executed[:, : noisy.shape[1]]


def test_predict_k1_preserves_absolute_rollout_coordinates_and_control_is_effective():
    sequence = K1.keep_latest_complete_chunk(_full_sequence())
    noisy = torch.zeros(1, 4, 2)
    timestep = torch.tensor([500.0])
    clean_policy = _Policy()
    clean, clean_state = K1.predict_context(
        clean_policy, sequence, noisy, timestep, shuffle_actions=False
    )
    shuffle_policy = _Policy()
    shuffled, shuffled_state = K1.predict_context(
        shuffle_policy, sequence, noisy, timestep, shuffle_actions=True
    )
    assert (clean_state.frame_st_id, clean_state.action_st_id, clean_state.next_update_id) == (15, 56, 14)
    assert (shuffled_state.frame_st_id, shuffled_state.action_st_id, shuffled_state.next_update_id) == (15, 56, 14)
    assert torch.equal(clean_policy.executed, torch.full((1, 8, 2), 6.0))
    assert torch.equal(shuffle_policy.executed, torch.full((1, 8, 2), 5.0))
    assert not torch.equal(clean, shuffled)


def test_artifact_rewrite_is_fail_closed_for_long_and_rollout():
    checkpoint = K1.rewrite_checkpoint(
        {
            "contract": {
                "candidate": "C66_LINGBOT_C58_BLOCK_PERSISTENT",
                "history_chunks": 7,
                "history_observation_frames": 15,
                "history_executed_actions": 56,
            }
        }
    )
    assert checkpoint["contract"]["candidate"] == "C66_K1_BOUNDED_MECHANISM"
    assert checkpoint["contract"]["history_chunks"] == 1
    assert checkpoint["contract"]["source_history_chunks"] == 7
    report = K1.rewrite_report(
        {
            "status": "PASS_C66_PAIRED_CANARY",
            "permission": "GO_C66_LIBERO_CANARY",
            "gates": {},
        }
    )
    assert report["status"] == "PASS_C66_K1_BOUNDED_MECHANISM"
    assert report["permission"] == "MECHANISM_SIGNAL_ONLY_NO_LONG_OR_ROLLOUT"
    assert "GO" not in report["permission"]
    assert report["gates"]["no_long_or_rollout_permission"] is True


def test_wrapper_delegates_budget_optimizer_and_data_to_reviewed_c66():
    source = (ROOT / "scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py").read_text()
    assert "C66.main()" in source
    assert "torch.optim" not in source
    assert "DataLoader" not in source
    assert "SOURCE_HISTORY_CHUNKS = 7" in source
    assert "K1_HISTORY_CHUNKS = 1" in source


def test_restore_diagnostic_is_read_only_and_compares_snapshot_tensors_exactly():
    path = ROOT / "scripts/h3wam/diagnose_c66_k1_restore_exact.py"
    spec = importlib.util.spec_from_file_location("_test_c66_k1_restore", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    snapshot = {
        "schema_version": 1,
        "layers": (1,),
        "token_capacity": 12,
        "episode_key": "episode",
        "frame_st_id": 15,
        "action_st_id": 56,
        "next_update_id": 14,
        "entries": {
            1: [
                {
                    "kind": "observation",
                    "key": torch.ones(1, 2, 3),
                    "value": torch.ones(1, 2, 3),
                    "update_id": 12,
                    "frame_start": 13,
                    "frame_count": 2,
                    "action_start": 0,
                    "action_count": 0,
                    "predicted": False,
                }
            ]
        },
    }
    exact, maximum = module.snapshot_exact(snapshot, snapshot)
    assert exact is True
    assert maximum == 0
    changed = {
        **snapshot,
        "entries": {
            1: [{**snapshot["entries"][1][0], "key": torch.zeros(1, 2, 3)}]
        },
    }
    exact, maximum = module.snapshot_exact(snapshot, changed)
    assert exact is False
    assert maximum == 1
    source = path.read_text()
    assert "torch.optim" not in source
    assert "optimizer_steps\": 0" in source
    assert "training_checkpoints_written\": 0" in source
