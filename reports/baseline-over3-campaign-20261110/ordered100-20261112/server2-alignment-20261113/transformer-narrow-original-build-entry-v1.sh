#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1/tmp
mkdir -p "$TMPDIR"
cd /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1/source
exec bash build.sh --pkg --jit --soc=ascend950 --ops=moe_gating_top_k_softmax --vendor_name=selectionnarrow -j8 --build-type=Release --ccache off --cann_3rd_lib_path=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1/dependencies
