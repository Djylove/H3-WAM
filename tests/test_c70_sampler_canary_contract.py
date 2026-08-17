from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_c70_launcher_freezes_the_only_training_variable() -> None:
    source = (ROOT / "scripts/h3wam/launch_c70_sampler_coverage_canary_8gpu.sh").read_text()
    assert "--rank-schedule c70_6_1_half_half" in source
    assert "--objective-mode fact_joint" in source
    assert "--scheduler-horizon 20000" in source
    assert "--seed 20260816" in source
    assert "--steps 10" in source
    assert "--steps 1" in source
    assert "C70_PROBE_ONLY" in source
    assert "no_checkpoint_written" in source
    assert "--restore-check-only" in source
    assert "GO_C70_LONG" in source


def test_c70_execution_source_is_in_complete_freeze() -> None:
    freeze = (ROOT / "scripts/h3wam/freeze_c67_rollout_source.py").read_text()
    assert '"scripts/h3wam/launch_c70_sampler_coverage_canary_8gpu.sh"' in freeze


def test_probe_without_save_checkpoint_serializes_json_null() -> None:
    source = (ROOT / "scripts/h3wam/train_c56b_fact_online.py").read_text()
    assert "str(a.save_checkpoint) if a.save_checkpoint is not None else None" in source
    assert '"checkpoint": str(a.save_checkpoint),' not in source
