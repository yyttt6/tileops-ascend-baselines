#!/bin/bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
set +u
source /opt/tileops-ci/server2-20261030/config/device-env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TILELANG_CACHE_DIR="$TASK_ROOT/cache/tilelang"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/extensions"
export ASCEND950_REFERENCE_ROOT=/opt/tileops-ci/server2-20261030/references/fla-v1/site-packages
export PYTHONPATH="$ASCEND950_REFERENCE_ROOT"
exec /opt/tileops-ci/server2-20261030/runtime/bin/python "$TASK_ROOT/worker.py" "$@"
