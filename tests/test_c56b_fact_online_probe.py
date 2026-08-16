import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    path = ROOT / "scripts/h3wam/probe_c56b_fact_online.py"
    spec = importlib.util.spec_from_file_location("_test_c56b_fact_online_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ONLINE = load_probe()


def test_future_representation_is_unprojected_layer49_value_mean():
    layers = ONLINE.LAYERWISE_H3_50_TO_ACTION_30
    kv = {
        layer: {
            "k": torch.zeros(1, 3, 56, 128),
            "v": torch.full((1, 3, 56, 128), float(layer)),
        }
        for layer in layers
    }
    result = ONLINE.future_representation_from_online_kv(kv)
    assert tuple(result.shape) == (1, 56 * 128)
    torch.testing.assert_close(result, torch.full_like(result, 49.0))
    with pytest.raises(ValueError, match="thirty-layer order"):
        ONLINE.future_representation_from_online_kv({49: kv[49]})


def test_probe_has_no_precomputed_h3_cache_argument_or_dataset():
    source = (ROOT / "scripts/h3wam/probe_c56b_fact_online.py").read_text()
    assert "CachedDreamWAMKVDataset" not in source
    assert "--kv-subdir" not in source
    assert "--cache-root" not in source
    assert "materialize_kv_for_autograd_consumer" in source
    assert "OnlineH3FACTRolloutDataset" in source
