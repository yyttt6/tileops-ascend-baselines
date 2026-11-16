#!/bin/bash
set -euo pipefail
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
ROOT=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/complete-private-device-v2
export ASCEND_CUSTOM_OPP_PATH="$ROOT"
export LD_LIBRARY_PATH="$ROOT/op_api/lib:$ROOT/op_proto/lib/linux/aarch64:$ROOT/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH"
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/probe_extra_unary_device_v2.py "$1" "$2"
