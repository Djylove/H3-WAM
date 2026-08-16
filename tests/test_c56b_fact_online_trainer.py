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
    assert "--expected-causal-dataset-sha256" in source
    assert "--expected-causal-observations-sha256" in source
    assert "a.expected_causal_dataset_sha256" in source
    assert '"causal_failure_dataset_sha256"' in source
    assert "optimizer.step()" in source
    assert "load_state_dict(loaded[\"model\"], strict=True)" in source
    assert "CachedDreamWAMKVDataset" not in source
    assert "kv-subdir" not in source


def test_long_launcher_is_milestoned_and_requires_fixed_c58_parent():
    source = (ROOT / "scripts/h3wam/launch_c56b_fact_online_long10000_8gpu.sh").read_text()
    assert "C58_PARENT_CHECKPOINT" in source
    assert "C58_PARENT_READY" in source
    assert "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE" in source
    assert "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL" in source
    assert 'ready.get("checkpoint_sha256")' in source
    assert "seq 1000 1000 10000" in source
    assert "restore-check-only" in source
    assert "GO_LONG" in source
    assert "CAUSAL_FAILURE_DATASET" in source
    assert "EXPECTED_CAUSAL_DATASET_SHA256" in source


def test_formal_parent_must_be_the_fixed_s10000_checkpoint():
    source = (ROOT / "scripts/h3wam/train_c56b_fact_online.py").read_text()
    assert 'int(c58_parent.get("completed_steps", -1)) != 10000' in source
    assert "fixed online C58b s10000 layerwise arm" in source


def test_c56_watcher_is_fail_closed_and_waits_for_all_eight_gpus():
    source = (ROOT / "scripts/h3wam/watch_and_launch_c56b_after_c58b_final.sh").read_text()
    assert "c56_canary_gate" in source
    assert "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE" in source
    assert "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL" in source
    assert 'ready.get("checkpoint_sha256")' in source
    assert "refusing duplicate launch" in source
    assert "--query-compute-apps=pid" in source
    assert 'gpu_count' in source
    assert "custom causal data requires CAUSAL_FAILURE_READY" in source
    assert "PASS_C61_FINALIZED_FACT_FAILURE_DATASET" in source
    assert "dataset_sha256" in source
    assert "PASS_C61_MATCHED_DATA_GATE" in source
    assert "C61_DATA_READY.json" in source


def test_c61_matched_arm_reuses_the_exact_c56_launcher():
    source = (ROOT / "scripts/h3wam/watch_and_launch_c56b_c61_matched.sh").read_text()
    assert "watch_and_launch_c56b_after_c58b_final.sh" in source
    assert "CAUSAL_FAILURE_READY" in source
    assert "online-long10000-c61-matched-v1" in source
    assert "base-lr" not in source
    assert "action-lr" not in source
