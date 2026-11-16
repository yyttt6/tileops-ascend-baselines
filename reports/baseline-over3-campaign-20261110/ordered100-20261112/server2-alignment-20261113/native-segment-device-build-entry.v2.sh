#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/compat/include:${CPATH:-}"
python3 - <<'BASELINE_BUILD_PATH'
from pathlib import Path
import datetime
r=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1')/'source';p=r/'build';target=Path('build-device-v1')
if p.is_symlink() and p.readlink()==target:pass
else:
 if p.exists() or p.is_symlink():p.rename(r/('build.before-device-alias-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
 p.symlink_to(target)
BASELINE_BUILD_PATH
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 "-DASC_DIR=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" "-DCUSTOM_ASC_CMAKE_PATH=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" -DCANN_3RD_LIB_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/dependencies -DTILINGAPI_INC_DIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/compat/include -DVENDOR_NAME=allpairs -DASCEND_COMPUTE_UNIT=ascend950 -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF -DENABLE_BINARY=OFF -DENABLE_PACKAGE=OFF -DENABLE_CUSTOM=OFF -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=ON -DCMAKE_INSTALL_PREFIX=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/install -UASCEND_OP_NAME -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$CXX"
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/scripts/util/dependency_parser.py --ops "unsorted_segment_sum;unsorted_segment_prod;unsorted_segment_min" -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 -DENABLE_CUSTOM=OFF -DENABLE_PACKAGE=OFF -DENABLE_BINARY=ON -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=OFF "-DASCEND_OP_NAME=unsorted_segment_sum;unsorted_segment_prod;unsorted_segment_min" -DENABLE_GEN_ACLNN=ON
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 -DENABLE_CUSTOM=OFF -DENABLE_PACKAGE=OFF -DENABLE_BINARY=ON -DENABLE_OP_KERNEL=ON -DPREPROCESS_ONLY=OFF "-DASCEND_OP_NAME=unsorted_segment_sum;unsorted_segment_prod;unsorted_segment_min" -DENABLE_GEN_ACLNN=OFF
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target ophost_nn --parallel 4
export ASCEND_OPP_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/opp
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/common/stub/op_tiling"
mkdir -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64
ln -sf /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/libophost_nn.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
ln -sf /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/libophost_nn.so /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/aarch64/libopmaster_ct.so
printf 'os=linux\nos_version=\narch=aarch64\n' > /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/opp/scene.info
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target ascendc_impl_gen --parallel 4
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/scripts/util/ascendc_impl_build.py /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/autogen/exc/aic-ascend950-ops-info.ini "" "" /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/tbe/ascendc /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/tbe/dynamic /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/autogen
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target gen_bin_scripts --parallel 4
if [ -d /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/binary/ascend950/bin/opc_cmd ]; then mv /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/binary/ascend950/bin/opc_cmd /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/binary/ascend950/bin/opc_cmd.before-regeneration.$(date -u +%Y%m%dT%H%M%S%N); fi
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target prepare_binary_compile_ascend950 --parallel 1
OPC_NUM=$(wc -l < /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1/binary/ascend950/bin/opc_cmd/opc_cmd.sh)
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 -DOPC_NUM_ascend950="$OPC_NUM"
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target binary --parallel 4
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segment-v1/source/build-device-v1 --target gen_bin_info_config --parallel 1
