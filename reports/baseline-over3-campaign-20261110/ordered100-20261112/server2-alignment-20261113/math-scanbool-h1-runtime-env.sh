#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export ASCEND_HOME_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/graph92-complete-profile-v2/sdk
export ASCEND_TOOLKIT_HOME=/opt/baseline-over3-20261110/server2-aligned-stack/native/graph92-complete-profile-v2/sdk
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp
export OPS_NN_VENDOR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/private-v1/vendors/allpairs_math_scanbool_math
export ASCEND_CUSTOM_OPP_PATH="$OPS_NN_VENDOR"
export LD_LIBRARY_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/graph92-complete-profile-v2/sdk/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/lib64:$OPS_NN_VENDOR/op_api/lib:$OPS_NN_VENDOR/op_proto/lib/linux/aarch64:$OPS_NN_VENDOR/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/graph92-python-v1/site-packages:/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/python/site-packages:/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/python/site-packages:${PYTHONPATH:-}"
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
