#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C67_SOURCE_SNAPSHOT:?C67 training requires the complete read-only source snapshot}"
source_freeze_sha="${C67_SOURCE_FREEZE_SHA256:?Set the independently reviewed SOURCE_FREEZE.json SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
c58_parent="${C58_PARENT_CHECKPOINT:?Set C58_PARENT_CHECKPOINT to fixed C58B s10000}"
c58_ready="${C58_PARENT_READY:?Set C58_PARENT_READY to audited C58B READY.json}"
release_file="${C67_RELEASE_FILE:?C67 requires an independently issued hash-bound manual release JSON}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c67-c60-budget-ablation-v1/online-long20000-v1}"
causal_dataset="${CAUSAL_FAILURE_DATASET:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt}"
causal_observations="${CAUSAL_FAILURE_OBSERVATIONS:-${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl}"
causal_dataset_sha="1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
causal_observations_sha="b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
demo_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
demo_cache_root="${workspace}/data/v7_dense_h3_cache"
c48_dataset="${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt"
c48_observations="${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl"
c59_overlay_root="${workspace}/eval/c59-fact-failure-active-overlay-v1"
source_freeze="${project}/SOURCE_FREEZE.json"
source_verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
dossier="${project}/experiments/dossiers/h3_c67_c60_budget_ablation_v1.json"
trainer="${project}/scripts/h3wam/train_c56b_fact_online.py"
finalizer="${project}/scripts/h3wam/finalize_c67_c60_budget_ablation_20k.py"
launcher="${project}/scripts/h3wam/launch_c67_c60_budget_ablation_20k_8gpu.sh"

for path in "${python_bin}" "${release_file}" "${c58_parent}" "${c58_ready}" \
  "${causal_dataset}" "${causal_observations}" "${dossier}" "${trainer}" \
  "${finalizer}" "${launcher}" "${source_freeze}" "${source_verifier}" \
  "${demo_manifest}" "${source_manifest}" "${demo_cache_root}/stats.pt" \
  "${c48_dataset}" "${c48_observations}" "${c59_overlay_root}/COMPLETED.json" \
  "${c59_overlay_root}/sample_labels.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing C67 release/input/source: ${path}" >&2; exit 2; }
done

"${python_bin}" "${source_verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${source_freeze_sha}"

"${python_bin}" - "${release_file}" "${output_root}" "${c58_ready}" "${c58_parent}" \
  "${dossier}" "${trainer}" "${finalizer}" "${launcher}" "${project}" \
  "${source_freeze_sha}" "${demo_manifest}" "${source_manifest}" \
  "${demo_cache_root}/stats.pt" "${c48_dataset}" "${c48_observations}" \
  "${c59_overlay_root}/COMPLETED.json" "${c59_overlay_root}/sample_labels.jsonl" <<'PY'
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
freeze_sha = sys.argv[10]
data_paths = list(map(Path, sys.argv[11:18]))
release = json.loads(release_path.resolve().read_text())
ready = json.loads(ready_path.resolve().read_text())
fixed = {
    "format": "h3wam-c67-budget-ablation-release-v2",
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
if release.get("c58_ready_sha256") != sha(ready_path.resolve()):
    raise SystemExit("C67 release C58 READY SHA mismatch")
freeze_path = project.resolve() / "SOURCE_FREEZE.json"
if sha(freeze_path) != freeze_sha:
    raise SystemExit("C67 source freeze manifest SHA mismatch")
freeze = json.loads(freeze_path.read_text())
declared_freeze = release.get("source_freeze")
freeze_identity = {
    "snapshot": str(project.resolve()),
    "manifest_sha256": freeze_sha,
    "git_commit": freeze.get("git_commit"),
    "git_tree": freeze.get("git_tree"),
    "repositories": freeze.get("repositories"),
    "directory_sources": freeze.get("directory_sources"),
    "dynamic_execution_sha256": freeze.get("dynamic_execution_sha256"),
}
if declared_freeze != freeze_identity:
    raise SystemExit("C67 release does not bind the complete source freeze")
expected_data = {
    "demo_manifest_sha256": "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
    "source_manifest_sha256": "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
    "demo_stats_sha256": "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
    "c48_dataset_sha256": "d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
    "c48_observations_sha256": "399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
    "c59_completed_sha256": "4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
    "c59_sample_labels_sha256": "f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}
if release.get("historical_c60_data_sha256") != expected_data:
    raise SystemExit("C67 release historical seven-data SHA contract mismatch")
actual_data = dict(zip(expected_data, (sha(path.resolve()) for path in data_paths)))
if actual_data != expected_data:
    raise SystemExit("C67 training historical seven-data SHA fail-close")
if (
    ready.get("status") != "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE"
    or ready.get("permission") != "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL"
    or ready.get("completed_steps") != 10000
    or ready.get("checkpoint_sha256") != fixed["c58_checkpoint_sha256"]
    or Path(ready.get("checkpoint", "")).resolve() != parent.resolve()
    or sha(parent.resolve()) != fixed["c58_checkpoint_sha256"]
):
    raise SystemExit("C67 fixed C58 parent gate failed")
for path in (dossier, trainer, finalizer, launcher):
    if not path.resolve().is_relative_to(project.resolve()):
        raise SystemExit(f"C67 training source escaped snapshot: {path}")
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
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
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

"${python_bin}" - "${project}/third_party/diffusers_h3/src" <<'PY'
import importlib
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
modules = (
    importlib.import_module("diffusers"),
    importlib.import_module("diffusers.modular_pipelines.minimax_h3.before_denoise"),
    importlib.import_module("diffusers.modular_pipelines.minimax_h3.encoders"),
)
for module in modules:
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(expected):
        raise SystemExit(f"C67 diffusers_h3 import escaped frozen snapshot: {origin}")
print("PASS_C67_FROZEN_DIFFUSERS_IMPORT_ORIGIN")
PY

common=(
  --demo-manifest "${demo_manifest}"
  --source-manifest "${source_manifest}"
  --demo-cache-root "${demo_cache_root}"
  --c48-dataset "${c48_dataset}"
  --c48-observations "${c48_observations}"
  --c59-overlay-root "${c59_overlay_root}"
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
