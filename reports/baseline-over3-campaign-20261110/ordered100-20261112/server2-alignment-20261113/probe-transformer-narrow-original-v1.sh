#!/bin/bash
set -eo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1/runtime-env-formal.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1/private-v1/vendors/selectionnarrow_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$ASCEND_CUSTOM_OPP_PATH/op_proto/lib/linux/aarch64:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling:$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs:/opt/baseline-over3-20261110/ordered100-20261112/workspace/TileOPs/src:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=0
export NARROW_PROBE_TAG="narrow-probe-$(date -u +%H%M%S%N)"
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1/probe_transformer_narrow_original.py
