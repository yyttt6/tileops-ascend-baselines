#!/bin/bash
set -euo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
family=$1
shift
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# The task uses registry names, not display aliases. Preserve each native stack
# even if an older foreground controller requested the generic template profile.
if [ "$family" = r-template ]; then
 library=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["library"])' "$P/TASKS.json" "$3")
 case "$library" in
  vllm-ascend) family=r-vllm ;;
  sgl-kernel-npu) family=r-sgl ;;
  tilelang) family=r-tilelang-original ;;
  tilelang-ascend) family=r-tla-original ;;
 esac
fi
if [[ "$family" == r-* ]]; then
 profile=${family#r-}
 if [ "$profile" = sgl ] || [ "$profile" = vllm ]; then
  export ORIGINAL_VENDOR_PROFILE=1
  if [ "$1" != prepare ]; then
   exec bash /opt/baseline-over3-20261110/vendor-runtime-repair/vendor_python.sh "$profile" "$P/rollout_bootstrap.py" "$@"
  fi
  export LD_LIBRARY_PATH=/opt/tileops-baseline-cache/workspace/.cache/all-pairs/sgl-ascend-runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  exec /opt/tileops-all-pairs/bin/allpairs-python "$P/rollout_bootstrap.py" "$@"
 fi
 set +u
 source /opt/baseline-over3-20261110/unified-stack-v1/environment.sh
 set -u
 if [ "$profile" = tilelang-original ]; then
  export BASELINE_ORIGINAL_TILELANG_VENDOR
  BASELINE_ORIGINAL_TILELANG_VENDOR=$(readlink -f "$P/workspace/ascend_baselines/tilelang")
  export TILELANG_LIBRARY_PATH="$BASELINE_ORIGINAL_TILELANG_VENDOR/tilelang/lib"
  export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
  export TVM_FFI_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
  export TILELANG_CACHE_DIR="$P/.cache/tilelang-selected-original-v1"
  export PYTHONDONTWRITEBYTECODE=1
  export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
 fi
 if [ "$profile" = tla-original ]; then
  export BASELINE_ORIGINAL_TLA_VENDOR
  BASELINE_ORIGINAL_TLA_VENDOR=$(readlink -f "$P/workspace/ascend_baselines/tilelang-ascend")
  export TILELANG_LIBRARY_PATH="$P/native/tla-original-v1/lib"
  export TVM_LIBRARY_PATH="$TILELANG_LIBRARY_PATH"
  export TVM_IMPORT_PYTHON_PATH="$BASELINE_ORIGINAL_TLA_VENDOR/3rdparty/tvm/python"
  export LD_LIBRARY_PATH="$TILELANG_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONDONTWRITEBYTECODE=1
 fi
 if [ "$profile" != template ] && [ "$profile" != tilelang-original ] && [ "$profile" != tla-original ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
  export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/bin:$PATH
 fi
 exec /opt/tileops-ci/runtime/bin/python -I -S "$P/rollout_bootstrap.py" "$@"
fi
if [ "$family" = checkdeep ]; then
 exec bash /opt/baseline-over3-20261110/vendor-runtime-repair/vendor_python.sh deepgemm "$P/supplement.py" "$@"
fi
if [ "$family" = deepgemm ]; then
 export LD_LIBRARY_PATH=/opt/tileops-baseline-cache/workspace/.cache/all-pairs/sgl-ascend-runtime:/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_host_builds/toolchain/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
 /opt/tileops-ci/runtime/bin/python -I -S /opt/baseline-over3-20261110/vendor-runtime-repair/verify_runtime.py
 exec /opt/tileops-all-pairs/bin/allpairs-python "$P/worker.py" "$@"
else
 set +u
 source /opt/baseline-over3-20261110/unified-stack-v1/environment.sh
 set -u
 if [ "$family" = catlass ] || [ "$family" = checkcatlass ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
  export ASCEND_TOOLKIT_HOME=$ASCEND_HOME_PATH
  export PATH=$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$ASCEND_HOME_PATH/bin:$PATH
 fi
 if [ "$family" = checktemplate ] || [ "$family" = checkcatlass ]; then
  exec /opt/tileops-ci/runtime/bin/python -I -S "$P/supplement_bootstrap.py" "$@"
 fi
 if [ "$family" = candidate ]; then
  exec /opt/tileops-ci/runtime/bin/python -I -S "$P/candidate_bootstrap_robust.py" "$@"
 fi
 exec /opt/tileops-ci/runtime/bin/python -I -S "$P/managed_bootstrap_robust.py" "$@"
fi
