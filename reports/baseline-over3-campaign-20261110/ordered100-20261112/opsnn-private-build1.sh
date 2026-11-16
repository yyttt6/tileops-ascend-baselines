#!/bin/bash
set -eo pipefail
source /opt/baseline-over3-20261110/unified-stack-v1/environment.sh
cmake -S /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source -B /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source/build-config3 -DENABLE_CUSTOM=ON -DENABLE_GEN_ACLNN=ON -DPREPROCESS_ONLY=OFF
cmake -S /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source -B /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source/build-config3 -DENABLE_CUSTOM=ON -DENABLE_GEN_ACLNN=OFF -DPREPROCESS_ONLY=OFF
cmake --build /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source/build-config3 --target ascendc_impl_gen --parallel 4
cmake --build /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source/build-config3 --target build_es_nn --parallel 4
cmake --build /opt/baseline-over3-20261110/ordered100-20261112/native-rebuild/ops-nn-v1/source/build-config3 --target package --parallel 4
