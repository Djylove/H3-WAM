#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
c58_parent="${C58_PARENT_CHECKPOINT:?Set C58_PARENT_CHECKPOINT to the completed online C58b checkpoint}"
c58_ready="${C58_PARENT_READY:?Set C58_PARENT_READY to the audited C58b final READY.json}"
canary_marker="${CANARY_MARKER:-${workspace}/outputs/c56b-fact-online-v1/optimizer-canary10-v1/GO_LONG.json}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1}"
causal_dataset="${CAUSAL_FAILURE_DATASET:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt}"
causal_observations="${CAUSAL_FAILURE_OBSERVATIONS:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl}"
causal_dataset_sha="${EXPECTED_CAUSAL_DATASET_SHA256:-1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4}"
causal_observations_sha="${EXPECTED_CAUSAL_OBSERVATIONS_SHA256:-b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55}"

[[ -f "${c58_parent}" && -f "${c58_ready}" && -f "${canary_marker}" \
   && -f "${causal_dataset}" && -f "${causal_observations}" ]] || { echo "missing C56b long parent/gate/data" >&2; exit 2; }
"${python_bin}" - "${canary_marker}" "${c58_ready}" "${c58_parent}" <<'PY'
import hashlib, json, sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text()).get("status") != "GO_LONG":
    raise SystemExit("C56b mechanical canary is not GO_LONG")
ready = json.loads(Path(sys.argv[2]).read_text())
parent = Path(sys.argv[3]).resolve()
checks = {
    "status": ready.get("status") == "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
    "permission": ready.get("permission") == "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL",
    "steps": int(ready.get("completed_steps", -1)) == 10000,
    "checkpoint": Path(ready.get("checkpoint", "")).resolve() == parent,
    "size": parent.is_file() and parent.stat().st_size == int(ready.get("checkpoint_size_bytes", -1)),
}
if all(checks.values()):
    digest = hashlib.sha256()
    with parent.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    checks["sha256"] = digest.hexdigest() == ready.get("checkpoint_sha256")
failed = [key for key, value in checks.items() if not value]
if failed:
    raise SystemExit("C56b fixed C58 parent gate failed: " + ",".join(failed))
PY
[[ ! -e "${output_root}" ]] || { echo "refusing existing C56b long output" >&2; exit 2; }
mkdir -p "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/restore"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c56b-online-long"
mkdir -p "${TMPDIR}"
cd "${project}"

common=(
  --demo-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
  --demo-cache-root "${workspace}/data/v7_dense_h3_cache"
  --c48-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt"
  --c48-observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl"
  --c59-overlay-root "${workspace}/eval/c59-fact-failure-active-overlay-v1"
  --c60-dataset "${causal_dataset}"
  --c60-observations "${causal_observations}"
  --expected-causal-dataset-sha256 "${causal_dataset_sha}"
  --expected-causal-observations-sha256 "${causal_observations_sha}"
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  --h3-model "${workspace}/models/MiniMax-H3"
  --d0-parent-checkpoint "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
  --c58-parent-checkpoint "${c58_parent}"
  --target-norm "${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1/target_norm.pt"
  --base-lr 2e-5 --action-lr 2e-4 --warmup-steps 500 --scheduler-horizon 10000
  --gradient-checkpointing
)

previous=""
for milestone in $(seq 1000 1000 10000); do
  checkpoint="${output_root}/checkpoints/c56b_online_s${milestone}.pt"
  train_args=(--steps 1000)
  if [[ -n "${previous}" ]]; then
    train_args+=(--load-checkpoint "${previous}")
  fi
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" "${train_args[@]}" \
    --save-checkpoint "${checkpoint}" --output "${output_root}/reports/train_s${milestone}.json"
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
    --load-checkpoint "${checkpoint}" --restore-check-only \
    --output "${output_root}/restore/restore_s${milestone}.json"
  previous="${checkpoint}"
done
