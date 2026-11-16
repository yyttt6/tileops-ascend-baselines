#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
# Native SDK frontend must match the native SDK/compiler; Triton referenceprofile remains unchanged.
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/bin:${PATH}"
test "$(command -v asc_opc)" = "/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/bin/asc_opc"
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 "-DASC_DIR=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" "-DCUSTOM_ASC_CMAKE_PATH=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" -DCANN_3RD_LIB_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/dependencies -DTILINGAPI_INC_DIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include -DVENDOR_NAME=allpairs -DASCEND_COMPUTE_UNIT=ascend950 -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF -DENABLE_BINARY=OFF -DENABLE_PACKAGE=OFF -DENABLE_CUSTOM=OFF -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=ON -DCMAKE_INSTALL_PREFIX=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/install -UASCEND_OP_NAME -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$CXX"
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/scripts/util/dependency_parser.py --ops foreach_mul_list -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 -DENABLE_CUSTOM=OFF -DENABLE_PACKAGE=OFF -DENABLE_BINARY=ON -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=OFF -DASCEND_OP_NAME=foreach_mul_list -DENABLE_GEN_ACLNN=ON
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 -DENABLE_CUSTOM=OFF -DENABLE_PACKAGE=OFF -DENABLE_BINARY=ON -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=OFF -DASCEND_OP_NAME=foreach_mul_list -DENABLE_GEN_ACLNN=OFF
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/source/build-device-v4 --target ophost_nn --parallel 4
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
