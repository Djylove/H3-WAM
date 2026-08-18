import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_c74_repeats_the_fixed_rank_schedule_across_four_nodes():
    trainer = load(
        ROOT / "scripts/h3wam/train_c56b_fact_online.py", "_test_c74_trainer"
    )
    expected = list(trainer.RANK_CATEGORIES) * 4
    assert [trainer.rank_category(trainer.DEFAULT_RANK_SCHEDULE, rank, 1) for rank in range(32)] == expected
    contract = trainer.rank_schedule_contract(trainer.DEFAULT_RANK_SCHEDULE, 32)
    assert contract["rank_categories"] == expected
    assert contract["rank_schedule"] == {
        "name": "c67_4_2_1_1", "group_size": 8, "group_repetitions": 4,
    }


def test_c74_launchers_bind_real_four_node_torchrun_and_scaled_optimizer():
    canary = (ROOT / "scripts/h3wam/launch_c74_action_only_multinode_three_expert_epoch_canary_32gpu.sh").read_text()
    long = (ROOT / "scripts/h3wam/launch_c74_action_only_multinode_three_expert_epoch_32647_32gpu.sh").read_text()
    for source in (canary, long):
        assert '--nnodes "${nnodes}"' in source
        assert '--node-rank "${node_rank}"' in source
        assert '--master-addr "${master_addr}"' in source
        assert "--nproc-per-node 8" in source
        assert "--base-lr 8e-5 --action-lr 8e-4 --warmup-steps 125" in source
        assert "--scheduler-horizon 32647" in source
        assert "--objective-mode action_only" in source
    assert "milestones=($(seq 1000 1000 7000) 7549 $(seq 8000 1000 32000) 32647)" in long
    assert "restore-check-only" in long


def test_c74_finalizer_and_budget_cover_three_expert_epochs():
    finalizer = load(
        ROOT / "scripts/h3wam/finalize_c74_action_only_multinode_three_expert_epoch.py",
        "_test_c74_finalizer",
    )
    assert finalizer.MILESTONES[:9] == (1000, 2000, 3000, 4000, 5000, 6000, 7000, 7549, 8000)
    assert finalizer.MILESTONES[-2:] == (32000, 32647)
    assert len(finalizer.MILESTONES) == 34
    assert finalizer.previous_milestone(8000) == 7549
    assert finalizer.expected_lr_factor(32647) == 0.0
    dossier = json.loads((
        ROOT / "experiments/dossiers/h3_c74_action_only_multinode_three_expert_epoch_v1.json"
    ).read_text())
    budget = dossier["budget"]
    assert budget["training_samples"] == 32 * 32647
    assert budget["expert_training_samples"] == 16 * 32647
    assert math.isclose(
        budget["cumulative_expert_effective_epochs_with_c58_parent"],
        (80000 + 16 * 32647) / 200779,
        rel_tol=1e-6,
    )
    assert dossier["decision"]["status"] == "GO_CANARY"
