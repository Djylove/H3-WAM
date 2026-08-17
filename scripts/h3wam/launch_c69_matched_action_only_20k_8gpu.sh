#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C69_SOURCE_SNAPSHOT:?C69 long training requires a complete read-only source snapshot}"
source_freeze_sha="${C69_SOURCE_FREEZE_SHA256:?Set reviewed SOURCE_FREEZE.json SHA256}"
release_file="${C69_RELEASE_FILE:?C69 requires a hash-bound long-run release JSON}"
canary_gate="${C69_CANARY_GO_LONG:?C69 requires the passed canary GO_LONG.json}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c69-matched-action-only-v1/online-long20000-v1}"
c58_parent="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
c58_ready="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json"
trainer="${project}/scripts/h3wam/train_c56b_fact_online.py"
finalizer="${project}/scripts/h3wam/finalize_c69_matched_action_only_20k.py"
launcher="${project}/scripts/h3wam/launch_c69_matched_action_only_20k_8gpu.sh"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"

for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${release_file}" \
  "${canary_gate}" "${c58_parent}" "${c58_ready}" "${trainer}" "${finalizer}" \
  "${launcher}" "${verifier}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/sample_labels.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing C69 long input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${source_freeze_sha}"

"${python_bin}" - "${release_file}" "${canary_gate}" "${project}/SOURCE_FREEZE.json" \
  "${source_freeze_sha}" "${output_root}" "${trainer}" "${finalizer}" "${launcher}" \
  "${c58_ready}" "${c58_parent}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/sample_labels.jsonl" <<'PY'
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
    canary.get("status") != "GO_C69_LONG"
    or canary.get("permission") != "MECHANICAL_PERMISSION_ONLY"
    or not all(canary.get("gate", {}).values())
):
    raise SystemExit("C69 canary did not authorize long training")
if (
    ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
    or ready.get("permission") != "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL"
    or ready.get("checkpoint_sha256") != expected_parent
    or Path(ready.get("checkpoint", "")).resolve() != Path(parent).resolve()
    or sha(parent) != expected_parent
):
    raise SystemExit("C69 fixed C58 parent gate failed")
expected_data = {
    "demo_manifest_sha256": "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
    "source_manifest_sha256": "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
    "demo_stats_sha256": "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
    "c48_dataset_sha256": "d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
    "c48_observations_sha256": "399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
    "c59_completed_sha256": "4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
    "c59_sample_labels_sha256": "f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}
actual_data = dict(zip(expected_data, (sha(path) for path in data_paths)))
if actual_data != expected_data:
    raise SystemExit("C69 immutable seven-data hash gate failed")
fixed = {
    "format": "h3wam-c69-matched-action-only-release-v1",
    "status": "GO_C69_MATCHED_ACTION_ONLY_20K",
    "permission": "MANUAL_GPU_RELEASE",
    "optimizer_steps": 20000,
    "scheduler_horizon": 20000,
    "checkpoint_interval": 1000,
    "output_root": str(Path(output).resolve()),
    "source_freeze_sha256": freeze_sha,
    "canary_gate_sha256": sha(canary_path),
    "trainer_sha256": sha(trainer),
    "finalizer_sha256": sha(finalizer),
    "launcher_sha256": sha(launcher),
    "c58_checkpoint_sha256": expected_parent,
    "data_sha256": expected_data,
}
mismatches = [key for key, expected in fixed.items() if release.get(key) != expected]
if mismatches or sha(freeze_path) != freeze_sha:
    raise SystemExit("C69 long release mismatch: " + ",".join(mismatches))
PY

[[ ! -e "${output_root}" ]] || { echo "refusing existing C69 long output" >&2; exit 2; }
free_bytes="$(df -PB1 "${workspace}" | awk 'NR==2 {print $4}')"
[[ "${free_bytes}" =~ ^[0-9]+$ && "${free_bytes}" -ge 322122547200 ]] || {
  echo "C69 requires at least 300 GiB free; found ${free_bytes:-unknown}" >&2; exit 2;
}
mkdir -p "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/restore"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c69-action-only-long"
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
  --seed 20260816 --gradient-checkpointing --objective-mode action_only
)

previous=""
for milestone in $(seq 1000 1000 20000); do
  checkpoint="${output_root}/checkpoints/c69_action_only_s${milestone}.pt"
  train_args=(--steps 1000)
  if [[ -n "${previous}" ]]; then train_args+=(--load-checkpoint "${previous}"); fi
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" "${train_args[@]}" \
    --save-checkpoint "${checkpoint}" --output "${output_root}/reports/train_s${milestone}.json"
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3wam/train_c56b_fact_online.py "${common[@]}" --steps 1 \
    --load-checkpoint "${checkpoint}" --restore-check-only \
    --output "${output_root}/restore/restore_s${milestone}.json"
  previous="${checkpoint}"
done

"${python_bin}" "${finalizer}" --root "${output_root}" \
  --output "${output_root}/TRAINING_COMPLETE.json"
