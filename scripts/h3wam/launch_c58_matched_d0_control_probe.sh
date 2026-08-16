#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
SOURCE_ROOT="${H3WAM_FASTWAM_SOURCE_ROOT:-/mnt/h3-wam/upstream-readonly/FastWAM-45d8e145/wan22}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58-matched-d0-control-v1/probe10}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
D0_PARENT="${D0_PARENT:-/mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"

for path in "${PYTHON_BIN}" "${MANIFEST}" "${SOURCE_MANIFEST}" "${D0_PARENT}" \
  "${SOURCE_ROOT}/action_dit.py"; do
  [[ -e "${path}" ]] || { echo "missing matched-control probe input: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing existing probe root: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
cuda13_lib="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${SOURCE_ROOT}"
cd "${PROJECT_ROOT}"
common=(
  scripts/h3wam/train_h3_fastwam_full_tower.py "${MANIFEST}"
  --source-manifest "${SOURCE_MANIFEST}"
  --cache-root "${CACHE_ROOT}" --kv-subdir "${KV_SUBDIR}"
  --d0-parent-checkpoint "${D0_PARENT}"
  --matched-d0-control --verify-h3-checkpoint-sha256
  --steps 10 --sample-offset 112000 --limit 80
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
  --learning-rate 1e-4 --weight-decay 0.01
  --warmup-steps 1000 --scheduler-horizon 10000 --min-learning-rate 1e-6
  --action-horizon 32 --action-shift 5
)
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${common[@]}" --save-checkpoint "${OUTPUT_ROOT}/control_s10.pt" \
  --output "${OUTPUT_ROOT}/train_s10.json" >"${OUTPUT_ROOT}/train_s10.log" 2>&1
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${common[@]}" --load-checkpoint "${OUTPUT_ROOT}/control_s10.pt" \
  --restore-check-only --output "${OUTPUT_ROOT}/restore_s10.json" \
  >"${OUTPUT_ROOT}/restore_s10.log" 2>&1
"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json,math,os,sys
from pathlib import Path
root=Path(sys.argv[1]); train=json.loads((root/"train_s10.json").read_text()); restore=json.loads((root/"restore_s10.json").read_text())
contract=train["contract"]
checks={
 "candidate":contract.get("candidate")=="C58_MATCHED_D0_FRESH_OPTIMIZER",
 "fresh_optimizer":contract.get("d0_parent_optimizer_restored") is False,
 "five_layers":contract.get("model_spec",{}).get("action_layers")==5,
 "ten_steps":train.get("completed_steps")==10 and train.get("training_samples")==80,
 "parent_parity":train.get("step0_parent_parity_max_abs")==0.0,
 "all_gradients":len(train.get("history",()))==10 and all(len(item.get("block_gradient_norms",()))==5 and all(value>0 for value in item["block_gradient_norms"]) for item in train.get("history",())),
 "finite_loss":len(train.get("history",()))==10 and all(isinstance(item.get("loss"),(int,float)) and math.isfinite(float(item["loss"])) for item in train.get("history",())),
 "strict_restore":restore.get("restore_probe_max_abs")==0.0,
}
if not all(checks.values()): raise SystemExit(f"matched control canary failed: {checks}")
payload={"status":"PASS_C58_MATCHED_CONTROL_CANARY","checks":checks,"effect_status":"NOT_EFFECT_EVIDENCE"}
out=root/"CANARY_READY.json"; tmp=out.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,out)
PY
echo "[C58 control] ten-step fresh-optimizer canary and strict restore PASS"
