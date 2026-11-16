#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v1 -DCANN_3RD_LIB_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/dependencies -DASCEND_COMPUTE_UNIT=ascend950 -DASCEND_OP_NAME=add -DVENDOR_NAME=allpairs_math -DENABLE_TEST=OFF -DENABLE_BINARY=OFF -DENABLE_CUSTOM=ON -DENABLE_PACKAGE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/install -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_STANDARD=17
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/build-host-v1 --target help
