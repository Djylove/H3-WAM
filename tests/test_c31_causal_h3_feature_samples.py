import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "precompute_c31_feature_samples_test",
    ROOT / "scripts/h3wam/precompute_c26_causal_h3_features.py",
)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_c31_feature_samples_order_current_before_future_targets():
    image = torch.zeros(4, 4, 3, dtype=torch.uint8)
    dataset = {
        "format": "h3wam-c31-action-conditioned-consequence-dataset-v1",
        "states": [
            {"group_id": 0, "task_language": "task a", "agentview_image": image, "wristview_image": image},
            {"group_id": 1, "task_language": "task b", "agentview_image": image + 1, "wristview_image": image + 1},
        ],
        "branches": [
            {"ordinal": 0, "group_id": 1, "future_agentview_image": image + 2, "future_wristview_image": image + 2},
            {"ordinal": 1, "group_id": 0, "future_agentview_image": image + 3, "future_wristview_image": image + 3},
        ],
    }
    samples = MODULE.feature_samples(dataset)
    assert [(row["kind"], row["index"]) for row in samples] == [
        ("current", 0), ("current", 1), ("future", 0), ("future", 1)
    ]
    assert samples[2]["task_language"] == "task b"
    assert samples[3]["task_language"] == "task a"


def test_legacy_feature_samples_remain_group_ordered():
    image = torch.zeros(4, 4, 3, dtype=torch.uint8)
    dataset = {
        "format": "h3wam-c27-causal-critic-dataset-v1",
        "states": [
            {"group_id": 0, "task_language": "task", "agentview_image": image, "wristview_image": image}
        ],
    }
    assert [(row["kind"], row["index"]) for row in MODULE.feature_samples(dataset)] == [("current", 0)]
