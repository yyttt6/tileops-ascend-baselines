#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/runtime-env-formal.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/python_binary/fla_npu/opp/vendors/fla_npu_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs:/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/src:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export BASELINE_GRAPH92_ENV_BOUND=1

export ASCEND_CLASSIC_FLA_PACKAGE=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/python_binary/fla_npu
export FLA_NPU_DISABLE_PTH=1

export PYTHONPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/python_binary:$PYTHONPATH"
