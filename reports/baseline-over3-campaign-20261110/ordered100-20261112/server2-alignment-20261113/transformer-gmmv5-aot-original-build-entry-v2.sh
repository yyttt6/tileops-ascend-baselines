#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2/tmp
mkdir -p "$TMPDIR"
cd /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2/source
bash build.sh --pkg --soc=ascend950 --ops=grouped_matmul --vendor_name=bf16gmm -j8 --build-type=Release --ccache off --cann_3rd_lib_path=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-arithmetic-v1/dependencies

exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2/assemble_transformer_gmmv5_aot_package_cpu.py
