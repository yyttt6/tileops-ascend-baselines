#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v2 --target cann-ops-math-allpairs_math_linux-aarch64_ascendc_cust_opapi cann-ops-math-allpairs_math_linux-aarch64_ascendc_cust_opmaster gen_bin_scripts --parallel 4
