#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/asc/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-div-v1/build-host-v1 --target cann-ops-math-allpairs_math_div_linux-aarch64_ascendc_cust_op_proto --parallel 4
