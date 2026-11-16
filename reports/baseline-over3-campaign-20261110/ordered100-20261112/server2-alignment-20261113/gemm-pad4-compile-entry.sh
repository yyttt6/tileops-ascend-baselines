#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1/compat/include:${CPATH:-}"
export ASCEND_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1/source/build-device-v1/opp
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1/source/build-device-v1/common/stub/op_tiling"
exec python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1/compile_gemm_pad4_original.py
