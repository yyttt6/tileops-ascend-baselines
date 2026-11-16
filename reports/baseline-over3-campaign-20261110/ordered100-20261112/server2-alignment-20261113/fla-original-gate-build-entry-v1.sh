#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export ASCEND_HOME_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/sdk-complete-v2
export ASCEND_TOOLKIT_HOME=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/sdk-complete-v2
export ASCEND_CANN_PACKAGE_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/sdk-complete-v2
export ASCEND_AICPU_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/sdk-complete-v2
export PYTHONDONTWRITEBYTECODE=1
export FLA_NPU_DISABLE_PTH=1
export TMPDIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1/tmp
mkdir -p "$TMPDIR"
cd /opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1/source
bash build.sh --pkg --soc=ascend950 --ops=kda_gate_cumsum --vendor_name=fla_npu -j8 --build-type=Release --ccache off --cann_3rd_lib_path=/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/dependencies-original-v5
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1/stage_fla_gate_aot_v1.py
