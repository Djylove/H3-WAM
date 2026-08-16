#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/optimizer-canary10-v1}"
checkpoint="${output_root}/c56b_online_s10.pt"

[[ ! -e "${output_root}" ]] || { echo "refusing existing C56b canary output" >&2; exit 2; }
mkdir -p "${output_root}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c56b-online-canary"
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
  --target-norm "${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1/target_norm.pt"
  --base-lr 2e-5 --action-lr 2e-4 --warmup-steps 500 --scheduler-horizon 10000
  --gradient-checkpointing
)

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" \
  --steps 10 --save-checkpoint "${checkpoint}" --output "${output_root}/train.json"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c56b_fact_online.py "${common[@]}" \
  --steps 1 --load-checkpoint "${checkpoint}" --restore-check-only \
  --output "${output_root}/restore.json"
"${python_bin}" - "${output_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
train = json.loads((root / "train.json").read_text())
restore = json.loads((root / "restore.json").read_text())
history = train["history"]
gate = {
    "ten_finite_steps": len(history) == 10 and all(
        all(__import__("math").isfinite(float(row[key])) for key in (
            "loss", "action_loss", "future_representation_loss",
            "future_state_loss", "value_loss",
        )) for row in history
    ),
    "all_30_shared_gradients": all(
        len(row["block_gradient_norms_mean_across_ranks"]) == 30
        and min(row["block_gradient_norms_mean_across_ranks"]) > 0
        for row in history
    ),
    "future_no_leak": max(row["sum_rank_future_leak_abs"] for row in history) == 0,
    "strict_restore": restore["restore_max_abs"] == 0,
    "no_kv_cache": train["contract"]["no_kv_cache"] is True,
}
checkpoint = root / "c56b_online_s10.pt"
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024):
        hasher.update(chunk)
digest = hasher.hexdigest()
marker = {
    "format": "h3wam-c56b-online-go-long-marker-v1",
    "status": "GO_LONG" if all(gate.values()) else "NO_GO",
    "effect_status": "NOT_EVIDENCE_READY",
    "gate": gate,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
    "claim_boundary": "Mechanical optimizer/restore permission only.",
}
(root / "GO_LONG.json").write_text(json.dumps(marker, indent=2) + "\n")
print(json.dumps(marker, sort_keys=True))
if marker["status"] != "GO_LONG":
    raise SystemExit(64)
PY
