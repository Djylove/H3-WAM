import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C26 = load_module(
    "train_c26_causal_action_critic_test_module",
    ROOT / "scripts/h3wam/train_c26_causal_action_critic.py",
)
C27 = load_module(
    "prepare_c27_expanded_causal_dataset_test_module",
    ROOT / "scripts/h3wam/prepare_c27_expanded_causal_dataset.py",
)
C28 = load_module(
    "train_c28_fresh_causal_action_critic_test_module",
    ROOT / "scripts/h3wam/train_c28_fresh_causal_action_critic.py",
)


def test_pairwise_metrics_and_exact_permutation_use_only_within_group_order():
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    scores = labels * 2.0 + torch.arange(8) * 0.001
    success, failure = C26.pair_indices(group_ids, labels, [0, 1])
    assert len(success) == len(failure) == 8
    metrics = C26.ranking_metrics(scores, group_ids, labels, [0, 1])
    assert metrics["pairwise_correct"] == 8
    assert metrics["top1_successes"] == 2
    pvalue, permutations = C26.exact_label_permutation_pvalue(
        scores, group_ids, labels, [0, 1], metrics["pairwise_correct"]
    )
    assert permutations == 36
    assert 0.0 < pvalue < 0.2


def test_pairwise_linear_fits_separable_training_pairs_without_state_labels():
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    design = torch.stack((labels * 2.0 - 1.0, torch.arange(8).float()), dim=1)
    fitted = C26.fit_pairwise_linear(
        design, group_ids, labels, [0, 1], steps=200,
        learning_rate=0.03, weight_decay=0.03, device=torch.device("cpu"),
    )
    scores = design @ fitted["weights"]
    metrics = C26.ranking_metrics(scores, group_ids, labels, [0, 1])
    assert metrics["pairwise_accuracy"] == 1.0
    assert fitted["examples_seen"] == 200 * 16


def test_shuffle_control_preserves_each_group_label_count():
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    shuffled = C26.shuffled_train_labels(labels, group_ids, [0, 1], 71)
    assert shuffled.shape == labels.shape
    for group_id in (0, 1):
        members = group_ids == group_id
        assert shuffled[members].sum() == labels[members].sum()


def test_c27_split_is_deterministic_stratified_and_source_isolated():
    records = []
    for suite, count in (("libero_goal", 5), ("libero_object", 8), ("libero_spatial", 4)):
        for index in range(count):
            records.append({
                "suite": suite,
                "source_episode": f"{suite}:task{index}:trial0",
            })
    first = C27.deterministic_split(records)
    second = C27.deterministic_split(list(reversed(records)))
    assert first == second
    for suite, expected_validation in (
        ("libero_goal", 1), ("libero_object", 2), ("libero_spatial", 1)
    ):
        rows = [row for row in records if row["suite"] == suite]
        assert sum(first[row["source_episode"]] == "val" for row in rows) == expected_validation
        assert all(first[row["source_episode"]] in ("train", "val") for row in rows)


def test_c28_combination_consumes_old_validation_as_train_but_keeps_c27_fresh():
    c25 = {
        "format": "h3wam-c26-causal-critic-dataset-v1",
        "states": [
            {"group_id": 0, "split": "train", "source_episode": "old-train"},
            {"group_id": 1, "split": "val", "source_episode": "old-val"},
        ],
        "branches": [
            {"group_id": 0, "split": "train", "source_episode": "old-train"},
            {"group_id": 1, "split": "val", "source_episode": "old-val"},
        ],
    }
    c27 = {
        "format": "h3wam-c27-causal-critic-dataset-v1",
        "states": [
            {"group_id": 0, "split": "train", "source_episode": "fresh-train"},
            {"group_id": 1, "split": "val", "source_episode": "fresh-val"},
        ],
        "branches": [
            {"group_id": 0, "split": "train", "source_episode": "fresh-train"},
            {"group_id": 1, "split": "val", "source_episode": "fresh-val"},
        ],
    }
    combined = C28.combine_datasets(c25, c27)
    assert [row["group_id"] for row in combined["states"]] == [0, 1, 2, 3]
    assert [row["split"] for row in combined["states"]] == [
        "train", "train", "train", "val"
    ]


def test_c28_permutation_pvalue_uses_exact_small_space():
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    scores = labels * 2.0
    metric = C26.ranking_metrics(scores, group_ids, labels, [0, 1])
    result = C28.permutation_pvalue(
        scores, group_ids, labels, [0, 1], metric["pairwise_correct"]
    )
    assert result["mode"] == "exact"
    assert result["space"] == 36
    assert result["value"] < 0.2
