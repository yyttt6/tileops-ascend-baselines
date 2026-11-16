#!/bin/bash
set -euo pipefail
unset BASELINE_ORIGINAL_TLA_VENDOR BASELINE_ORIGINAL_TILELANG_VENDOR TVM_IMPORT_PYTHON_PATH TVM_LIBRARY_PATH TILELANG_LIBRARY_PATH TVM_FFI_LIBRARY_PATH
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
 export LD_LIBRARY_PATH="$(python3 -I -S -c 'import os;print(":".join(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and p!="/opt/baseline-over3-20261110/ordered100-20261112/native/tla-original-v1/lib"))')"
fi
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/runtime-env-formal.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1/private-v1/vendors/originalvar_math
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs:/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/src:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export BASELINE_GRAPH92_ENV_BOUND=1

export ASCEND_CLASSIC_MATH_VENDOR="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1/private-v1/vendors/originalvar_math"
