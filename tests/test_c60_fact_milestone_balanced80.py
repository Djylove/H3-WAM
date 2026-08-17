from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_freezes_samples_noise_solver_and_normalization():
    source = (
        ROOT / "scripts/h3wam/evaluate_c60_fact_milestone_balanced80.py"
    ).read_text()
    assert "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42" not in source
    assert "SELECTED_IDS_SHA256 = PAIRED.SELECTED_IDS_SHA256" in source
    assert "config.seed != 42 or config.inference_steps != 10" in source
    assert 'shift=5.0' in source
    assert "OnlineC58bValidationDataset" in source
    assert "--c56b-paired-ready" not in source


def test_queue_covers_all_ten_without_checkpoint_selection():
    source = (
        ROOT / "scripts/h3wam/launch_c60_fact_milestone_balanced80_queue.sh"
    ).read_text()
    assert "seq 1000 1000 10000" in source
    assert "(step / 1000 - 1) % 8 == gpu" in source
    assert "for gpu in 0 1 2 3 4 5 6 7" in source
    assert "rollout_libero" not in source
    assert "RESULTS.json" in source


def test_aggregator_preregisters_late_budget_gates():
    source = (
        ROOT / "scripts/h3wam/aggregate_c60_fact_milestone_balanced80.py"
    ).read_text()
    for gate in (
        "late_window_physical_not_worse_than_mid",
        "late_window_normalized_not_worse_than_mid",
        "s10_physical_not_worse_than_s5",
        "s10_normalized_not_worse_than_s5",
        "s10_gripper_within_0_005_of_s5",
        "s10_language_preserves_90pct_of_s5",
        "s10_visual_preserves_90pct_of_s5",
    ):
        assert gate in source
    assert "ELIGIBLE_TO_AUTHOR_S20K_DOSSIER" in source
    assert "NO_EVIDENCE_FOR_S20K_CONTINUATION" in source
    assert "closed_loop_was_not_used_to_select_an_existing_milestone" in source
