#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C70_SOURCE_SNAPSHOT:?C70 canary requires a complete read-only source snapshot}"
source_freeze_sha="${C70_SOURCE_FREEZE_SHA256:?Set reviewed SOURCE_FREEZE.json SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
probe_only="${C70_PROBE_ONLY:-0}"
if [[ "${probe_only}" == "1" ]]; then
  default_output="${workspace}/outputs/c70-sampler-coverage-v1/probe1-v1"
elif [[ "${probe_only}" == "0" ]]; then
  default_output="${workspace}/outputs/c70-sampler-coverage-v1/canary10-v1"
else
  echo "C70_PROBE_ONLY must be 0 or 1" >&2; exit 2
fi
output_root="${OUTPUT_ROOT:-${default_output}}"
checkpoint="${output_root}/c70_sampler_s10.pt"
c58_parent="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
c58_ready="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json"

for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" \
  "${project}/scripts/h3wam/freeze_c67_rollout_source.py" \
  "${project}/scripts/h3wam/train_c56b_fact_online.py" \
  "${project}/experiments/dossiers/h3_c70_sampler_coverage_v1.json" \
  "${c58_parent}" "${c58_ready}"; do
  [[ -e "${path}" ]] || { echo "missing C70 canary input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${project}/scripts/h3wam/freeze_c67_rollout_source.py" \
  --verify --snapshot "${project}" --expected-manifest-sha256 "${source_freeze_sha}"
"${python_bin}" - "${c58_ready}" "${c58_parent}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

ready_path, parent = map(Path, sys.argv[1:])
ready = json.loads(ready_path.read_text())
expected = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
if (
    ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
    or ready.get("permission") != "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL"
    or ready.get("checkpoint_sha256") != expected
    or Path(ready.get("checkpoint", "")).resolve() != parent.resolve()
    or sha(parent) != expected
):
    raise SystemExit("C70 fixed C58 parent gate failed")
PY

[[ ! -e "${output_root}" ]] || { echo "refusing existing C70 canary output" >&2; exit 2; }
mkdir -p "${output_root}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cu13_lib="$("${python_bin}" - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c70-sampler-canary"
mkdir -p "${TMPDIR}"
cd "${project}"

common=(
  --demo-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
  --demo-cache-root "${workspace}/data/v7_dense_h3_cache"
  --c48-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt"
  --c48-observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl"
  --c59-overlay-root "${workspace}/eval/c59-fact-failure-active-overlay-v1"
  --c60-dataset "${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
  --c60-observations "${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  --h3-model "${workspace}/models/MiniMax-H3"
  --d0-parent-checkpoint "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
  --c58-parent-checkpoint "${c58_parent}"
  --target-norm "${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1/target_norm.pt"
  --base-lr 2e-5 --action-lr 2e-4 --warmup-steps 500 --scheduler-horizon 20000
  --seed 20260816 --gradient-checkpointing --objective-mode fact_joint
  --rank-schedule c70_6_1_half_half
)

if [[ "${probe_only}" == "1" ]]; then
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
    --output "${output_root}/train.json"
  "${python_bin}" - "${output_root}" "${source_freeze_sha}" <<'PY'
import json, math, sys
from pathlib import Path

root, freeze_sha = Path(sys.argv[1]), sys.argv[2]
train = json.loads((root / "train.json").read_text())
history, contract = train["history"], train["contract"]
row = history[0] if len(history) == 1 else {}
schedule = contract.get("rank_schedule", {})
gate = {
    "one_finite_optimizer_step": len(history) == 1 and all(
        math.isfinite(float(row[key])) for key in (
            "loss", "action_loss", "future_representation_loss",
            "future_state_loss", "value_loss",
        )
    ),
    "all_30_shared_gradients": len(row.get("block_gradient_norms_mean_across_ranks", [])) == 30
    and min(row.get("block_gradient_norms_mean_across_ranks", [0])) > 0,
    "future_no_leak": row.get("sum_rank_future_leak_abs") == 0,
    "exact_sampler_contract": schedule.get("name") == "c70_6_1_half_half"
    and schedule.get("mean_streams_per_step") == {
        "expert_demo": 6.0, "success_rollout": 1.0,
        "observational_failure": 0.5, "causal_failure": 0.5,
    },
    "fact_joint_objective": contract.get("objective_mode") == "fact_joint"
    and contract.get("loss_weights") == [10.0, 1.0, 0.4, 0.4],
    "no_checkpoint_written": train.get("checkpoint") is None
    and train.get("checkpoint_bytes") is None,
}
report = {
    "format": "h3wam-c70-sampler-coverage-probe-v1",
    "status": "PASS_C70_PROBE_ONLY" if all(gate.values()) else "FAIL_C70_PROBE_ONLY",
    "permission": "READY_TO_UPDATE_DOSSIER_NOT_TRAINING_PERMISSION",
    "effect_status": "NO_EFFECT_CLAIM",
    "gate": gate,
    "source_freeze_sha256": freeze_sha,
    "claim_boundary": "One non-retained optimizer step only; no candidate checkpoint, training permission or effect claim.",
}
(root / "PROBE.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, sort_keys=True))
if report["status"] != "PASS_C70_PROBE_ONLY":
    raise SystemExit(64)
PY
  exit 0
fi

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 10 \
  --save-checkpoint "${checkpoint}" --output "${output_root}/train.json"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
  --load-checkpoint "${checkpoint}" --restore-check-only \
  --output "${output_root}/restore.json"

"${python_bin}" - "${output_root}" "${source_freeze_sha}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

root, freeze_sha = Path(sys.argv[1]), sys.argv[2]
train = json.loads((root / "train.json").read_text())
restore = json.loads((root / "restore.json").read_text())
history, contract = train["history"], train["contract"]
schedule = contract.get("rank_schedule", {})
gate = {
    "ten_finite_steps": len(history) == 10 and all(
        all(math.isfinite(float(row[key])) for key in (
            "loss", "action_loss", "future_representation_loss",
            "future_state_loss", "value_loss",
        )) for row in history
    ),
    "exact_sampler_contract": contract.get("rank_categories") == [
        "expert_demo", "expert_demo", "expert_demo", "expert_demo",
        "expert_demo", "expert_demo", "success_rollout",
        "alternating_observational_failure_causal_failure",
    ] and schedule.get("name") == "c70_6_1_half_half"
    and schedule.get("mean_streams_per_step") == {
        "expert_demo": 6.0, "success_rollout": 1.0,
        "observational_failure": 0.5, "causal_failure": 0.5,
    },
    "fact_joint_objective": contract.get("objective_mode") == "fact_joint"
    and contract.get("loss_weights") == [10.0, 1.0, 0.4, 0.4],
    "all_30_shared_gradients": all(
        len(row["block_gradient_norms_mean_across_ranks"]) == 30
        and min(row["block_gradient_norms_mean_across_ranks"]) > 0
        for row in history
    ),
    "future_no_leak": max(row["sum_rank_future_leak_abs"] for row in history) == 0,
    "strict_restore": restore.get("restore_max_abs") == 0,
    "frozen_online_h3": contract.get("h3_execution") == "online_frozen_int8_per_rank_v1"
    and contract.get("no_kv_cache") is True,
}
checkpoint = root / "c70_sampler_s10.pt"
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        hasher.update(chunk)
marker = {
    "format": "h3wam-c70-sampler-coverage-canary-v1",
    "status": "GO_C70_LONG" if all(gate.values()) else "NO_GO_C70_LONG",
    "permission": "MECHANICAL_PERMISSION_ONLY",
    "effect_status": "NOT_EVIDENCE_READY",
    "gate": gate,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hasher.hexdigest(),
    "source_freeze_sha256": freeze_sha,
    "claim_boundary": "Sampler/DDP/gradient/restore proof only; no held-out or LIBERO effect claim.",
}
(root / "GO_LONG.json").write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps(marker, sort_keys=True))
if marker["status"] != "GO_C70_LONG":
    raise SystemExit(64)
PY
