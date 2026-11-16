#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/runtime-env-formal.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-rope-v1/private-v1/vendors/ropepairs_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs:/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/src:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export BASELINE_GRAPH92_ENV_BOUND=1

RMS_COMPAT_INCLUDE=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include
RMS_HOSTAPI_ROOT=$(readlink -f "$RMS_COMPAT_INCLUDE/ascendc/host_api")
RMS_SDK_INCLUDE=$(dirname "$(dirname "$RMS_HOSTAPI_ROOT")")/include
export CPATH="$RMS_COMPAT_INCLUDE:$RMS_SDK_INCLUDE:${CPATH:-}"

export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-rope-v1/source/posembedding/rotary_position_embedding/op_kernel:$CPATH"
