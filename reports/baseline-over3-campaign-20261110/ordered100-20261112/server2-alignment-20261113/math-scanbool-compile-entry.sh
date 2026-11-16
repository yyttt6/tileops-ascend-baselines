#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/asc/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export ASCEND_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/compile-opp
export LD_LIBRARY_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/build-host-v1:${LD_LIBRARY_PATH}"
mkdir -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64
ln -s /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/build-host-v1/libcust_opmaster_rt2.0.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/compile-opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
exec python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1/compile_scanbool_original.py
