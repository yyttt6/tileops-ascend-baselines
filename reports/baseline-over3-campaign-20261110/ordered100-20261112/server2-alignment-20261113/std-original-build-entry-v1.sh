#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export CMAKE_ARGS="-DASC_DIR=/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/tikcpp/ascendc_kernel_cmake -DCUSTOM_ASC_CMAKE_PATH=/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/tikcpp/ascendc_kernel_cmake"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-std-original-v1/tmp"
mkdir -p "$TMPDIR"
cd "/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-std-original-v1/source"
bash build.sh --pkg --soc=ascend950 --ops=reduce_std_v2 --vendor_name=originalstd -j8 --build-type=Release --ccache=off --cann_3rd_lib_path=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/dependencies
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python "/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-std-original-v1/assemble_std_original_package_cpu.py"
