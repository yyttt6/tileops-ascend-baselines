#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/compat/include:${CPATH:-}"
python3 - <<'STATIC_ALIAS'
from pathlib import Path
import datetime
r=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source');p=r/'build';assert p.is_symlink();t=p.readlink();assert t in [Path('build-device-v1'),Path('build-static-v1')]
if t!=Path('build-static-v1'):
 p.rename(r/('build.alias-before-static-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')));p.symlink_to('build-static-v1')
STATIC_ALIAS
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 "-DASC_DIR=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" "-DCUSTOM_ASC_CMAKE_PATH=${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake" -DCANN_3RD_LIB_PATH=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/dependencies -DTILINGAPI_INC_DIR=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/compat/include -DVENDOR_NAME=allpairs -DASCEND_COMPUTE_UNIT=ascend950 -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF -DENABLE_BINARY=OFF -DENABLE_PACKAGE=OFF -DENABLE_CUSTOM=OFF -DENABLE_OP_KERNEL=OFF -DPREPROCESS_ONLY=ON -DCMAKE_INSTALL_PREFIX=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/install -UASCEND_OP_NAME -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER="$CXX"
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/scripts/util/dependency_parser.py --ops "foreach_abs;foreach_neg" -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 -DENABLE_CUSTOM=ON -DENABLE_PACKAGE=OFF -DENABLE_GEN_ACLNN=ON -DPREPROCESS_ONLY=OFF "-DASCEND_OP_NAME=foreach_abs;foreach_neg" -DENABLE_STATIC=ON -DENABLE_BINARY=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 -DENABLE_CUSTOM=ON -DENABLE_PACKAGE=OFF -DENABLE_GEN_ACLNN=OFF -DPREPROCESS_ONLY=OFF "-DASCEND_OP_NAME=foreach_abs;foreach_neg" -DENABLE_STATIC=ON -DENABLE_BINARY=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 --target ophost_nn_static opapi_nn_static ops_info_gen_ascend950 --parallel 4
mkdir -p /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1/bin_tmp/ascend950
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/scripts/util/build_opp_kernel_static.py GenStaticOpResourceIni -s ascend950 -b /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/scripts/util/build_opp_kernel_static.py StaticCompile -s ascend950 -b /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 -n=0 -a=aarch64
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source -B /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 -DENABLE_GEN_ACLNN=OFF -DENABLE_STATIC=ON
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1 --target cann_nn_static --parallel 4
python3 - <<'STATIC_VALIDATE'
from pathlib import Path
import json,hashlib
b=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1');archive=b/'bin_tmp/ascend950/libcann_nn_static.a';data=archive.read_bytes();bins=sorted((b/'binary/ascend950/bin/ascend950').rglob('*.o'));assert len(bins)==7 and all(p.read_bytes() in data for p in bins),'All originalkernels mustbe embedded';r={'archive':str(archive),'archive_sha256':hashlib.sha256(data).hexdigest(),'original_device_objects_embedded':len(bins),'no_device_kernel_source_or_precision_changes':True,'API_runtime_selection_notyetqualified':True};(b/'STATIC_ARCHIVE_VERIFICATION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
STATIC_VALIDATE
