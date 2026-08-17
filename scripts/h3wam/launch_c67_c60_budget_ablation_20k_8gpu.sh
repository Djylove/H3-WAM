#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
c58_parent="${C58_PARENT_CHECKPOINT:?Set C58_PARENT_CHECKPOINT to fixed C58B s10000}"
c58_ready="${C58_PARENT_READY:?Set C58_PARENT_READY to audited C58B READY.json}"
release_file="${C67_RELEASE_FILE:?C67 requires an independently issued hash-bound manual release JSON}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c67-c60-budget-ablation-v1/online-long20000-v1}"
causal_dataset="${CAUSAL_FAILURE_DATASET:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt}"
causal_observations="${CAUSAL_FAILURE_OBSERVATIONS:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl}"
causal_dataset_sha="1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
causal_observations_sha="b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
dossier="${project}/experiments/dossiers/h3_c67_c60_budget_ablation_v1.json"
trainer="${project}/scripts/h3wam/train_c56b_fact_online.py"
finalizer="${project}/scripts/h3wam/finalize_c67_c60_budget_ablation_20k.py"
launcher="${project}/scripts/h3wam/launch_c67_c60_budget_ablation_20k_8gpu.sh"

for path in "${python_bin}" "${release_file}" "${c58_parent}" "${c58_ready}" \
  "${causal_dataset}" "${causal_observations}" "${dossier}" "${trainer}" \
  "${finalizer}" "${launcher}"; do
  [[ -e "${path}" ]] || { echo "missing C67 release/input/source: ${path}" >&2; exit 2; }
done

"${python_bin}" - "${release_file}" "${output_root}" "${c58_ready}" "${c58_parent}" \
  "${dossier}" "${trainer}" "${finalizer}" "${launcher}" "${project}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

release_path, output, ready_path, parent = map(Path, sys.argv[1:5])
dossier, trainer, finalizer, launcher, project = map(Path, sys.argv[5:10])
release = json.loads(release_path.resolve().read_text())
ready = json.loads(ready_path.resolve().read_text())
fixed = {
    "format": "h3wam-c67-budget-ablation-release-v1",
    "status": "GO_C67_BUDGET_ABLATION_20K",
    "permission": "MANUAL_GPU_RELEASE",
    "optimizer_steps": 20000,
    "scheduler_horizon": 20000,
    "checkpoint_interval": 1000,
    "output_root": str(output.resolve()),
    "c58_checkpoint_sha256": "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541",
}
failed = [name for name, value in fixed.items() if release.get(name) != value]
if failed:
    raise SystemExit("C67 manual release contract failed: " + ",".join(failed))
if (
    ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
    or ready.get("permission") != "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL"
    or ready.get("completed_steps") != 10000
    or ready.get("checkpoint_sha256") != fixed["c58_checkpoint_sha256"]
    or Path(ready.get("checkpoint", "")).resolve() != parent.resolve()
    or sha(parent.resolve()) != fixed["c58_checkpoint_sha256"]
):
    raise SystemExit("C67 fixed C58 parent gate failed")
paths = {
    "dossier": dossier,
    "trainer": trainer,
    "finalizer": finalizer,
    "launcher": launcher,
    "fact_layerwise_tower": project / "src/fastwam/models/h3wam/fact_layerwise_tower.py",
    "fact_online_data": project / "src/fastwam/models/h3wam/fact_online_data.py",
    "c58_online_training": project / "src/fastwam/models/h3wam/c58_online_training.py",
    "fastwam_full_tower": project / "src/fastwam/models/h3wam/fastwam_full_tower.py",
    "fact_upstream": project / "third_party/FACT/world_action_model/trainer/wa_casual_trainer.py",
    "fastwam_upstream": project / "third_party/FastWAM/src/fastwam/models/wan22/action_dit.py",
    "c58_ready": ready_path,
}
declared = release.get("source_sha256")
if not isinstance(declared, dict) or set(declared) != set(paths):
    raise SystemExit("C67 release source manifest keys mismatch")
mismatched = [name for name, path in paths.items() if not path.is_file() or sha(path.resolve()) != declared[name]]
if mismatched:
    raise SystemExit("C67 release source hash mismatch: " + ",".join(mismatched))
PY

[[ ! -e "${output_root}" ]] || { echo "refusing existing C67 output: ${output_root}" >&2; exit 2; }
free_bytes="$(df -PB1 "${workspace}" | awk 'NR==2 {print $4}')"
[[ "${free_bytes}" =~ ^[0-9]+$ && "${free_bytes}" -ge 322122547200 ]] || {
  echo "C67 requires at least 300 GiB free before launch; found ${free_bytes:-unknown}" >&2
  exit 2
}
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
export TMPDIR="${workspace}/tmp/c67-budget-ablation"
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
  --base-lr 2e-5 --action-lr 2e-4 --warmup-steps 500 --scheduler-horizon 20000
  --seed 20260816 --gradient-checkpointing
)

previous=""
for milestone in $(seq 1000 1000 20000); do
  checkpoint="${output_root}/checkpoints/c67_online_s${milestone}.pt"
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

"${python_bin}" scripts/h3wam/finalize_c67_c60_budget_ablation_20k.py \
  --root "${output_root}" --c58-ready "${c58_ready}" \
  --output "${output_root}/TRAINING_COMPLETE.json"
