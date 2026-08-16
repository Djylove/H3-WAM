import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    path = ROOT / "scripts/h3wam/fit_c56b_fact_online_target_norm.py"
    spec = importlib.util.spec_from_file_location("_test_c56b_online_norm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NORM = load_probe()


def test_registered_mixture_is_exact_and_repeated():
    schedule = NORM.registered_stream_schedule(32)
    assert len(schedule) == 32
    assert {name: schedule.count(name) for name in NORM.MIXTURE_COUNTS_PER_16} == {
        name: count * 2 for name, count in NORM.MIXTURE_COUNTS_PER_16.items()
    }
    with pytest.raises(ValueError, match="multiple of 16"):
        NORM.registered_stream_schedule(17)


def test_moment_fit_matches_population_zscore():
    values = torch.tensor([[1.0, 5.0], [3.0, 9.0], [5.0, 13.0]], dtype=torch.float64)
    mean, std = NORM.fit_mean_std_from_moments(
        values.sum(0), values.square().sum(0), len(values)
    )
    torch.testing.assert_close(mean, values.mean(0))
    torch.testing.assert_close(std, values.std(0, unbiased=False))
    normalized = (values - mean) / std
    torch.testing.assert_close(normalized.mean(0), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        normalized.std(0, unbiased=False), torch.ones(2, dtype=torch.float64)
    )


def test_calibration_source_forbids_validation_and_feature_cache():
    source = (ROOT / "scripts/h3wam/fit_c56b_fact_online_target_norm.py").read_text()
    assert 'split="train"' in source
    assert 'split="validation"' not in source
    assert "CachedDreamWAMKVDataset" not in source
    assert "projected_features" not in source
    assert "future_representation_from_online_kv" in source
