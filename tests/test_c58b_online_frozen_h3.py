import importlib.util
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    path = ROOT / "scripts/h3wam" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C58 = load_script("_test_c58b_offset_parent", "train_h3_fastwam_full_tower.py")
ONLINE = load_script("_test_c58b_online_probe", "probe_c58b_online_frozen_h3.py")


def test_c58_probe_inherits_nonzero_training_slice():
    selection = C58.probe_dataset_selection(Namespace(sample_offset=112000))
    assert selection == {"limit": 1, "sample_offset": 112000}


def test_online_probe_selects_declared_manifest_offset():
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "manifest.jsonl"
        rows = [{"id": f"sample-{index}"} for index in range(3)]
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        assert ONLINE.selected_manifest_row(manifest, 2) == rows[2]
        with pytest.raises(ValueError, match="existing manifest row"):
            ONLINE.selected_manifest_row(manifest, 3)


def test_online_cache_parity_is_exact_and_layer_order_is_strict():
    online = {
        0: {"k": torch.ones(1, 2, 2, 3), "v": torch.zeros(1, 2, 2, 3)},
        2: {"k": torch.full((1, 2, 2, 3), 2.0), "v": torch.ones(1, 2, 2, 3)},
    }
    report = ONLINE.compare_kv_exact(online, online)
    assert all(item["exact"] and item["max_abs"] == 0 for item in report.values())
    mismatched = {
        layer: {name: value.clone() for name, value in item.items()}
        for layer, item in online.items()
    }
    mismatched[2]["v"][0, 0, 0, 0] = 3
    with pytest.raises(RuntimeError, match="layer 2 v"):
        ONLINE.compare_kv_exact(online, mismatched)
    with pytest.raises(RuntimeError, match="layer order"):
        ONLINE.compare_kv_exact(online, {2: online[2], 0: online[0]})


def test_frozen_h3_inference_tensors_cross_into_autograd_without_value_change():
    with torch.inference_mode():
        inference_kv = {0: {"k": torch.randn(1, 2), "v": torch.randn(1, 2)}}
    assert all(torch.is_inference(value) for value in inference_kv[0].values())
    materialized = ONLINE.materialize_kv_for_autograd_consumer(inference_kv)
    assert all(not torch.is_inference(value) for value in materialized[0].values())
    for name in ("k", "v"):
        torch.testing.assert_close(
            materialized[0][name], inference_kv[0][name], rtol=0, atol=0
        )
