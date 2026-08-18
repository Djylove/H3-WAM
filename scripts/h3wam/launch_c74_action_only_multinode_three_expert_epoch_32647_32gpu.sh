#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C74_SOURCE_SNAPSHOT:?C74 long training requires a complete read-only source snapshot}"
freeze_sha="${C74_SOURCE_FREEZE_SHA256:?Set reviewed SOURCE_FREEZE.json SHA256}"
release_file="${C74_RELEASE_FILE:?C74 requires a hash-bound long-run release JSON}"
canary_gate="${C74_CANARY_GO_LONG:?C74 requires the passed canary GO_LONG.json}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c74-action-only-multinode-three-expert-epoch-v1/online-long32647-v1}"
c58_parent="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
c58_ready="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json"
trainer="${project}/scripts/h3wam/train_c56b_fact_online.py"
finalizer="${project}/scripts/h3wam/finalize_c74_action_only_multinode_three_expert_epoch.py"
launcher="${project}/scripts/h3wam/launch_c74_action_only_multinode_three_expert_epoch_32647_32gpu.sh"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
nnodes="${C74_NNODES:-4}"
node_rank="${C74_NODE_RANK:?Set C74 node rank 0..3}"
master_addr="${C74_MASTER_ADDR:?Set C74 rendezvous address}"
master_port="${C74_MASTER_PORT:-29674}"
[[ "${nnodes}" == 4 && "${node_rank}" =~ ^[0-3]$ ]] || {
  echo "C74 long run requires four nodes with node rank 0..3" >&2; exit 2;
}

for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${release_file}" \
  "${canary_gate}" "${c58_parent}" "${c58_ready}" "${trainer}" "${finalizer}" \
  "${launcher}" "${verifier}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/sample_labels.jsonl" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl" \
  "${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1/target_norm.pt"; do
  [[ -e "${path}" ]] || { echo "missing C74 long input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

"${python_bin}" - "${release_file}" "${canary_gate}" "${project}/SOURCE_FREEZE.json" \
  "${freeze_sha}" "${output_root}" "${trainer}" "${finalizer}" "${launcher}" \
  "${c58_ready}" "${c58_parent}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/sample_labels.jsonl" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl" \
  "${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1/target_norm.pt" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

(release_path, canary_path, freeze_path, freeze_sha, output, trainer,
 finalizer, launcher, ready_path, parent, *data_paths) = sys.argv[1:]
release = json.loads(Path(release_path).read_text())
canary = json.loads(Path(canary_path).read_text())
ready = json.loads(Path(ready_path).read_text())
expected_parent = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
if (
    canary.get("status") != "GO_C74_LONG"
    or canary.get("permission") != "MECHANICAL_PERMISSION_ONLY"
    or not all(canary.get("gate", {}).values())
):
    raise SystemExit("C74 canary did not authorize long training")
if (
    ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
    or ready.get("permission") != "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL"
    or ready.get("checkpoint_sha256") != expected_parent
    or Path(ready.get("checkpoint", "")).resolve() != Path(parent).resolve()
    or sha(parent) != expected_parent
):
    raise SystemExit("C74 fixed C58b parent gate failed")
keys = (
    "demo_manifest_sha256", "source_manifest_sha256", "demo_stats_sha256",
    "c48_dataset_sha256", "c48_observations_sha256", "c59_completed_sha256",
    "c59_sample_labels_sha256", "c60_dataset_sha256", "c60_observations_sha256",
    "target_norm_sha256",
)
actual_data = dict(zip(keys, (sha(path) for path in data_paths)))
fixed = {
    "format": "h3wam-c74-action-only-multinode-three-expert-epoch-release-v1",
    "status": "GO_C74_ACTION_ONLY_32647",
    "permission": "MANUAL_GPU_RELEASE",
    "optimizer_steps": 32647,
    "scheduler_horizon": 32647,
    "checkpoint_interval": 1000,
    "output_root": str(Path(output).resolve()),
    "source_freeze_sha256": freeze_sha,
    "canary_gate_sha256": sha(canary_path),
    "trainer_sha256": sha(trainer),
    "finalizer_sha256": sha(finalizer),
    "launcher_sha256": sha(launcher),
    "c58_checkpoint_sha256": expected_parent,
    "data_sha256": actual_data,
}
mismatches = [key for key, expected in fixed.items() if release.get(key) != expected]
if mismatches or sha(freeze_path) != freeze_sha:
    raise SystemExit("C74 long release mismatch: " + ",".join(mismatches))
PY

prepared="${output_root}/MULTINODE_PREPARED.json"
if [[ "${node_rank}" == 0 ]]; then
  [[ ! -e "${output_root}" ]] || { echo "refusing existing C74 long output" >&2; exit 2; }
  free_bytes="$(df -PB1 "${workspace}" | awk 'NR==2 {print $4}')"
  [[ "${free_bytes}" =~ ^[0-9]+$ && "${free_bytes}" -ge 644245094400 ]] || {
    echo "C74 requires at least 600 GiB free; found ${free_bytes:-unknown}" >&2; exit 2;
  }
  mkdir -p "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/restore"
  "${python_bin}" - "${prepared}" "${freeze_sha}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "C74_MULTINODE_PREPARED", "source_freeze_sha256": sys.argv[2],
}, indent=2) + "\n")
PY
else
  for _ in $(seq 1 180); do [[ -f "${prepared}" ]] && break; sleep 1; done
  [[ -f "${prepared}" ]] || { echo "timed out waiting for C74 long preparation" >&2; exit 2; }
fi
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
export TMPDIR="${workspace}/tmp/c74-action-only-long-node${node_rank}"
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
  --base-lr 8e-5 --action-lr 8e-4 --warmup-steps 125 --scheduler-horizon 32647
  --seed 20260816 --gradient-checkpointing --objective-mode action_only
)

previous=""
completed=0
milestones=($(seq 1000 1000 7000) 7549 $(seq 8000 1000 32000) 32647)
invocation=0
for milestone in "${milestones[@]}"; do
  delta=$((milestone - completed))
  checkpoint="${output_root}/checkpoints/c74_action_only_s${milestone}.pt"
  train_args=(--steps "${delta}")
  if [[ -n "${previous}" ]]; then train_args+=(--load-checkpoint "${previous}"); fi
  rendezvous_port=$((master_port + invocation)); invocation=$((invocation + 1))
  "${python_bin}" -m torch.distributed.run --nnodes "${nnodes}" --nproc-per-node 8 \
    --node-rank "${node_rank}" --master-addr "${master_addr}" --master-port "${rendezvous_port}" \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" "${train_args[@]}" \
    --save-checkpoint "${checkpoint}" --output "${output_root}/reports/train_s${milestone}.json"
  rendezvous_port=$((master_port + invocation)); invocation=$((invocation + 1))
  "${python_bin}" -m torch.distributed.run --nnodes "${nnodes}" --nproc-per-node 8 \
    --node-rank "${node_rank}" --master-addr "${master_addr}" --master-port "${rendezvous_port}" \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
    --load-checkpoint "${checkpoint}" --restore-check-only \
    --output "${output_root}/restore/restore_s${milestone}.json"
  previous="${checkpoint}"
  completed="${milestone}"
done

if [[ "${node_rank}" == 0 ]]; then
  "${python_bin}" "${finalizer}" --root "${output_root}" \
    --output "${output_root}/TRAINING_COMPLETE.json"
fi
