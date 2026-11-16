#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:$PATH"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-attention-v1/tmp
mkdir -p "$TMPDIR"
cd /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-attention-v1/source
exec bash build.sh --pkg --jit --soc=ascend950 --ops=flash_attention_score --vendor_name=attentionpairs -j8 --build-type=Release --ccache off --cann_3rd_lib_path=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-attention-v1/dependencies
