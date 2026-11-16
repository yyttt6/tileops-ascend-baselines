#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export CMAKE_ARGS="-DASC_DIR=/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/tikcpp/ascendc_kernel_cmake -DCUSTOM_ASC_CMAKE_PATH=/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/tikcpp/ascendc_kernel_cmake"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-mean-original-v1/tmp"
mkdir -p "$TMPDIR"
exec python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-mean-original-v1/finalize_mean_mixed_v4.py
