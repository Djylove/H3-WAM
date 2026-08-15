from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/h3wam"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "prepare_c31_executed_action_contract_test_module",
    ROOT / "scripts/h3wam/prepare_c31_action_conditioned_consequence_dataset.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_terminal_prefix_masks_actions_the_environment_never_consumed():
    proposed = np.arange(32 * 7, dtype=np.float32).reshape(32, 7)
    executed, is_pad = MODULE.mask_unexecuted_action_tail(
        proposed, executed_steps=5
    )

    np.testing.assert_array_equal(executed[:5], proposed[:5])
    np.testing.assert_array_equal(executed[5:], np.zeros((27, 7), np.float32))
    np.testing.assert_array_equal(is_pad, np.arange(32) >= 5)
    np.testing.assert_array_equal(proposed, np.arange(32 * 7).reshape(32, 7))


def test_full_chunk_has_no_padding_and_invalid_lengths_fail():
    proposed = np.ones((32, 7), dtype=np.float32)
    executed, is_pad = MODULE.mask_unexecuted_action_tail(
        proposed, executed_steps=32
    )
    np.testing.assert_array_equal(executed, proposed)
    assert not is_pad.any()
    with pytest.raises(ValueError):
        MODULE.mask_unexecuted_action_tail(proposed, executed_steps=0)
    with pytest.raises(ValueError):
        MODULE.mask_unexecuted_action_tail(proposed, executed_steps=33)
