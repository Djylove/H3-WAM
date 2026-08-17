from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/h3wam/audit_c65_third_node_acceleration.py"
SPEC = importlib.util.spec_from_file_location("audit_c65_third", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_legacy_check_then_mkdir_is_not_a_shared_claim() -> None:
    source = '''
if [[ -s "${out}/results.json" ]] && compgen -G "${out}/*trajectory.npz" >/dev/null; then
  continue
fi
[[ ! -e "${out}" ]] || { return 1; }
mkdir -p "${out}"
'''
    report = MODULE.audit_legacy_launcher(source)
    assert report["complete_skip_partial_refuse_then_mkdir"] is True
    assert report["all_workers_honor_atomic_claim"] is False
    assert report["check_then_mkdir_toctou"] is True


def test_active_legacy_workers_make_live_third_node_fail_closed() -> None:
    launcher = {
        "complete_skip_partial_refuse_then_mkdir": True,
        "all_workers_honor_atomic_claim": False,
    }
    inventory = {"uncreated_jobs": 100}
    decision = MODULE.decide(launcher, inventory, active_legacy_launchers=2)
    assert decision["status"].startswith("NO_GO_C65_THIRD_NODE")
    assert decision["permission"] == "DO_NOT_LAUNCH_NODE_30137"


def test_no_active_legacy_worker_still_does_not_auto_grant_permission() -> None:
    launcher = {
        "complete_skip_partial_refuse_then_mkdir": True,
        "all_workers_honor_atomic_claim": False,
    }
    inventory = {"uncreated_jobs": 100}
    decision = MODULE.decide(launcher, inventory, active_legacy_launchers=0)
    assert decision["status"] == "REQUIRES_SEPARATE_REVIEW"
    assert decision["permission"] == "NOT_GRANTED"
