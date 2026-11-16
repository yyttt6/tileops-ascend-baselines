#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/asc/include:/opt/tileops-ci/server2-20261030/toolchain-candidate/sdk-9.2.0/aarch64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
test ! -e /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/source/build
ln -s ../build-host-v2 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/source/build
tar -cf /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/generated-before-path-completion.tar -C /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v2 tbe binary
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v2 --target add_src_copy gen_kernel_options gen_compile_options_ascend950 gen_opc_info_ascend950 config_compile_ascend950_add --parallel 1
find /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v2/binary/ascend950/bin/opc_cmd -type f -name '*.sh' -exec sha256sum {} \;
