#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "action_only" && "$1" != "joint_aux" ]]; then
  echo "usage: $0 action_only|joint_aux" >&2
  exit 2
fi
arm="$1"

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
root="${OUTPUT_ROOT:-${workspace}/outputs/c55-fact-joint-action-canary-v2/${arm}}"
ready="${workspace}/eval/c55-fact-joint-action-v1/kv-full-v1/READY.json"
parent="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
parent_sha="36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"

[[ -f "${ready}" ]] || { echo "C55 full K/V READY is missing" >&2; exit 2; }
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]] || {
  echo "C55 parent identity mismatch" >&2
  exit 2
}
[[ ! -e "${root}" ]] || { echo "refusing to reuse C55 canary root" >&2; exit 1; }
mkdir -p "${root}"

export PYTHONPATH="${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
pytorch_cu13_lib="$("${python_bin}" - <<'PY'
import sysconfig
from pathlib import Path
path = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
if not (path / "libnvJitLink.so.13").is_file():
    raise SystemExit(path)
print(path)
PY
)"
export LD_LIBRARY_PATH="${pytorch_cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c55-${arm}-canary"
mkdir -p "${TMPDIR}"
cd "${project}"

common=(
  --arm "${arm}"
  --parent-checkpoint "${parent}" --expected-parent-sha256 "${parent_sha}"
  --demo-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
  --demo-source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
  --demo-cache-root "${workspace}/data/v7_dense_h3_cache"
  --demo-kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1
  --rollout-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt"
  --rollout-projected-features "${workspace}/eval/c49-dense-value-h3-features-v1/projected_features.pt"
  --rollout-kv-root "${workspace}/eval/c55-fact-joint-action-v1/kv-full-v1"
  --learning-rate 5e-5 --weight-decay 0.01 --warmup-steps 500
  --scheduler-horizon 6000 --min-learning-rate 1e-6 --seed 20260816
)

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c55_fact_joint_action.py "${common[@]}" \
  --steps 10 --save-checkpoint "${root}/step10.pt" \
  --output "${root}/train.json" >"${root}/train.log" 2>&1

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_c55_fact_joint_action.py "${common[@]}" \
  --steps 1 --load-checkpoint "${root}/step10.pt" --restore-check-only \
  --output "${root}/restore.json" >"${root}/restore.log" 2>&1

"${python_bin}" - "${root}" "${arm}" >"${root}/COMPLETED" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
arm = sys.argv[2]
train = json.loads((root / "train.json").read_text())
restore = json.loads((root / "restore.json").read_text())
if train["status"] != "PASS_MECHANICAL" or restore["status"] != "PASS_CHECKPOINT_RESTORE":
    raise SystemExit("C55 canary gate failed")
print(json.dumps({"arm": arm, "status": "PASS_C55_CANARY", "elapsed_seconds": train["elapsed_seconds"], "restore_probe_max_abs": restore["restore_probe_max_abs"]}, sort_keys=True))
PY
