#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/asc/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/build-host-v1 --target cann-ops-math-allpairs_math_scanbool_linux-aarch64_ascendc_cust_opapi cann-ops-math-allpairs_math_scanbool_linux-aarch64_ascendc_cust_opmaster cann-ops-math-allpairs_math_scanbool_linux-aarch64_ascendc_cust_op_proto gen_bin_scripts --parallel 4

cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/build-host-v1 --target cumsum_src_copy cumprod_src_copy logical_or_src_copy not_equal_src_copy gen_kernel_options gen_compile_options_ascend950 gen_opc_info_ascend950 config_compile_ascend950_cumsum config_compile_ascend950_cumprod config_compile_ascend950_logical_or config_compile_ascend950_not_equal --parallel 1
