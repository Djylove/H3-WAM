from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/h3wam/audit_libero_failure_modes.py"
SPEC = spec_from_file_location("audit_libero_failure_modes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_object_target_normalizes_box_name() -> None:
    assert AUDIT.object_target(
        "pick up the cream cheese box and place it in the basket"
    ) == "cream_cheese_1_joint0"


def test_spatial_target_uses_official_bddl_binding() -> None:
    assert AUDIT.target_keys(
        "spatial",
        0,
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ) == ("akita_black_bowl_1_joint0",)


def test_failure_classes_distinguish_wrong_object_and_completion() -> None:
    base = {
        "success": False,
        "target_joint_keys": ["cream_cheese_1_joint0"],
    }
    wrong = {
        **base,
        "max_object_joint_delta": {
            "cream_cheese_1_joint0": 0.01,
            "tomato_sauce_1_joint0": 0.4,
            "basket_1_joint0": 0.2,
        },
    }
    completion = {
        **base,
        "max_object_joint_delta": {
            "cream_cheese_1_joint0": 0.3,
            "tomato_sauce_1_joint0": 0.0,
        },
    }
    assert AUDIT.classify(wrong) == "FAIL_WRONG_OBJECT_EVIDENCE"
    assert AUDIT.classify(completion) == "FAIL_TARGETS_MOVED_COMPLETION"
