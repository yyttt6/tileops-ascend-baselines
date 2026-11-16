#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-v2/compat/include:${CPATH:-}"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-v2/source/build-server2-v1 --target cust_opapi --parallel 4
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-v2/source/build-server2-v1 --target package --parallel 4
