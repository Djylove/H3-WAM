#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C71_SOURCE_SNAPSHOT:?C71 probe requires a complete immutable source snapshot}"
freeze_sha="${C71_SOURCE_FREEZE_SHA256:?Set the reviewed SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
output="${C71_PROBE_OUTPUT:-${workspace}/outputs/c71-lightwam-state-fusion-v1/probe-one-batch-v1/report.json}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
probe="${project}/scripts/h3wam/probe_c71_lightwam_online.py"
source_freeze="${project}/SOURCE_FREEZE.json"

for path in "${python_bin}" "${verifier}" "${probe}" "${source_freeze}" \
  "${project}/third_party/Light-WAM/src/lightwam/models/wan22/state_fusion_action_expert.py" \
  "${project}/third_party/diffusers_h3/src/diffusers/__init__.py" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"; do
  [[ -e "${path}" ]] || { echo "missing C71 probe input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output}" ]] || { echo "refusing existing C71 probe output: ${output}" >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_LIGHTWAM_SOURCE_ROOT="${project}/third_party/Light-WAM/src/lightwam/models/wan22"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c71-lightwam-probe"
mkdir -p "${TMPDIR}"
cd "${project}"

"${python_bin}" - "${project}/third_party/diffusers_h3/src" "${project}/third_party/Light-WAM" <<'PY'
import importlib
from pathlib import Path
import subprocess
import sys

diffusers_root, lightwam_root = map(lambda value: Path(value).resolve(), sys.argv[1:])
module = importlib.import_module("diffusers")
if not Path(module.__file__).resolve().is_relative_to(diffusers_root):
    raise SystemExit(f"C71 diffusers import escaped snapshot: {module.__file__}")
# A frozen archive intentionally has no .git; SOURCE_FREEZE already binds it.
if (lightwam_root / ".git").exists():
    head = subprocess.run(
        ("git", "-C", str(lightwam_root), "rev-parse", "HEAD"),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if head.stdout.strip() != "b2785f66e13fd9987e94ae1ecc1c441d5059c9ae":
        raise SystemExit("C71 Light-WAM source revision mismatch")
print("PASS_C71_FROZEN_IMPORT_ORIGINS")
PY

CUDA_VISIBLE_DEVICES="${C71_CUDA_VISIBLE_DEVICES:-0}" "${python_bin}" "${probe}" \
  --manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --cache-root "${workspace}/data/v7_dense_h3_cache" \
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --source-freeze "${source_freeze}" \
  --expected-source-freeze-sha256 "${freeze_sha}" \
  --sample-offset "${C71_SAMPLE_OFFSET:-0}" \
  --output "${output}"
