#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/asc/include:/opt/baseline-over3-20261110/unified-stack-v1/toolchain-candidate/sdk-9.2.0/aarch64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export ASCEND_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/compile-opp
export LD_LIBRARY_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/build-host-v2:${LD_LIBRARY_PATH}"
mkdir -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64
if [ -L /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so ]; then
 test "$(readlink /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so)" = /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/build-host-v2/libcust_opmaster_rt2.0.so
else
 ln -s /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/build-host-v2/libcust_opmaster_rt2.0.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
fi
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1/build-host-v2 --target gen_opc_info_ascend950 config_compile_ascend950_sqrt --parallel 1
