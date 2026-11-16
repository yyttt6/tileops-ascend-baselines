#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export ASCEND_HOME_PATH=/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0
export ASCEND_TOOLKIT_HOME=/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export OPS_NN_VENDOR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/graph-private-v1/vendors/allpairs_nn
export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/lib64:$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/lib/linux/aarch64:$OPS_NN_VENDOR/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/python/site-packages:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/python/site-packages:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segmentmin-api-v1/probe_graph92_min_h2.py "$@"
