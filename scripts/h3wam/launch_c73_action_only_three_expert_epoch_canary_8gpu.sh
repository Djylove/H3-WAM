#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C73_SOURCE_SNAPSHOT:?C73 canary requires a complete read-only source snapshot}"
freeze_sha="${C73_SOURCE_FREEZE_SHA256:?Set reviewed SOURCE_FREEZE.json SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c73-action-only-three-expert-epoch-v1/canary10-v1}"
checkpoint="${output_root}/c73_action_only_s10.pt"
c58_parent="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
c58_ready="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
trainer="${project}/scripts/h3wam/train_c56b_fact_online.py"

for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" "${trainer}" \
  "${project}/experiments/dossiers/h3_c73_action_only_three_expert_epoch_v1.json" \
  "${c58_parent}" "${c58_ready}"; do
  [[ -e "${path}" ]] || { echo "missing C73 canary input: ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${project}/SOURCE_FREEZE.json" | awk '{print $1}')" == "${freeze_sha}" ]] || {
  echo "C73 source freeze SHA mismatch" >&2; exit 2;
}
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"
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
    raise SystemExit("C73 fixed C58b parent gate failed")
PY

[[ ! -e "${output_root}" ]] || { echo "refusing existing C73 canary output" >&2; exit 2; }
mkdir -p "${output_root}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${project}/third_party/FastWAM/src/fastwam/models/wan22"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c73-action-only-canary"
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
  --base-lr 2e-5 --action-lr 2e-4 --warmup-steps 500 --scheduler-horizon 130585
  --seed 20260816 --gradient-checkpointing --objective-mode action_only
)

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 10 \
  --save-checkpoint "${checkpoint}" --output "${output_root}/train.json"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
  --load-checkpoint "${checkpoint}" --restore-check-only \
  --output "${output_root}/restore.json"

"${python_bin}" - "${output_root}" "${freeze_sha}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

root, freeze_sha = Path(sys.argv[1]), sys.argv[2]
train = json.loads((root / "train.json").read_text())
restore = json.loads((root / "restore.json").read_text())
history, contract = train["history"], train["contract"]
gate = {
    "ten_finite_steps": len(history) == 10 and all(
        all(math.isfinite(float(row[key])) for key in (
            "loss", "action_loss", "future_representation_loss",
            "future_state_loss", "value_loss",
        )) for row in history
    ),
    "three_expert_epoch_schedule": contract.get("scheduler_horizon") == 130585,
    "matched_rank_mixture": contract.get("rank_categories") == [
        "expert_demo", "expert_demo", "expert_demo", "expert_demo",
        "success_rollout", "success_rollout", "observational_failure", "causal_failure",
    ],
    "action_only_objective": contract.get("objective_mode") == "action_only"
    and contract.get("loss_weights") == [10.0, 0.0, 0.0, 0.0],
    "auxiliary_heads_frozen": len(contract.get("frozen_auxiliary_parameters", [])) > 0,
    "all_30_global_shared_gradients": all(
        len(row["block_gradient_norms_mean_across_ranks"]) == 30
        and min(row["block_gradient_norms_mean_across_ranks"]) > 0 for row in history
    ),
    "future_no_leak": max(row["sum_rank_future_leak_abs"] for row in history) == 0,
    "strict_restore": restore.get("restore_max_abs") == 0,
    "frozen_online_h3": contract.get("h3_execution") == "online_frozen_int8_per_rank_v1"
    and contract.get("no_kv_cache") is True,
}
checkpoint = root / "c73_action_only_s10.pt"
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        hasher.update(chunk)
digest = hasher.hexdigest()
marker = {
    "format": "h3wam-c73-action-only-three-expert-epoch-canary-v1",
    "status": "GO_C73_LONG" if all(gate.values()) else "NO_GO_C73_LONG",
    "permission": "MECHANICAL_PERMISSION_ONLY",
    "effect_status": "NOT_EVIDENCE_READY",
    "gate": gate,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
    "source_freeze_sha256": freeze_sha,
    "claim_boundary": "Optimizer/DDP/schedule/freeze/restore proof only; no action or LIBERO effect claim.",
}
(root / "GO_LONG.json").write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps(marker, sort_keys=True))
if marker["status"] != "GO_C73_LONG":
    raise SystemExit(64)
PY
