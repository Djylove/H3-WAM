from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_online_trainer_preserves_full_fact_contract():
    source = (ROOT / "scripts/h3wam/train_c56b_fact_online.py").read_text()
    assert "RANK_CATEGORIES" in source
    assert source.count('"expert_demo"') >= 4
    assert '"success_rollout"' in source
    assert '"observational_failure"' in source
    assert '"causal_failure"' in source
    assert "fact_backbone_port_losses" in source
    assert "DistributedDataParallel" in source
    assert "materialize_kv_for_autograd_consumer" in source
    assert "future_state_loss_mask" in source
    assert "optimizer.step()" in source
    assert "load_state_dict(loaded[\"model\"], strict=True)" in source
    assert "CachedDreamWAMKVDataset" not in source
    assert "kv-subdir" not in source


def test_long_launcher_is_milestoned_and_requires_fixed_c58_parent():
    source = (ROOT / "scripts/h3wam/launch_c56b_fact_online_long10000_8gpu.sh").read_text()
    assert "C58_PARENT_CHECKPOINT" in source
    assert "seq 1000 1000 10000" in source
    assert "restore-check-only" in source
    assert "GO_LONG" in source
