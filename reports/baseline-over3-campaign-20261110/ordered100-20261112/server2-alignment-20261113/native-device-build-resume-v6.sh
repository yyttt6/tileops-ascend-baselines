#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:${CPATH:-}"
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
test "$(command -v asc_opc)" = "/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin/asc_opc"
export ASCEND_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/opp
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/common/stub/op_tiling"
mkdir -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64
ln -sf /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/libophost_nn.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
ln -sf /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/libophost_nn.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/libopmaster_ct.so
printf 'os=linux\nos_version=\narch=aarch64\n' > /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/opp/scene.info
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 --target ascendc_impl_gen --parallel 4
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 --target gen_bin_scripts --parallel 4
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 --target prepare_binary_compile_ascend950 --parallel 1
OPC_NUM=$(wc -l < /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4/binary/ascend950/bin/opc_cmd/opc_cmd.sh)
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 -DOPC_NUM_ascend950="$OPC_NUM"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 --target binary --parallel 4
