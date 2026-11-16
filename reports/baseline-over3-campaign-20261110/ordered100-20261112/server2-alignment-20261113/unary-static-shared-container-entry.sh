#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
B=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-v1/source/build-static-v1
mkdir -p "$B/private-static-api-v1"
"$CXX" -shared -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,-Bsymbolic \
 -Wl,-u,aclnnForeachAbs -Wl,-u,aclnnForeachAbsGetWorkspaceSize -Wl,-u,aclnnForeachNeg -Wl,-u,aclnnForeachNegGetWorkspaceSize \
 -L "$ASCEND_HOME_PATH/aarch64-linux/lib64" -L "$ASCEND_HOME_PATH/lib64" \
 -Wl,--start-group "$B/bin_tmp/ascend950/libcann_nn_static.a" -Wl,--end-group \
 -lgraph -lgraph_base -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog \
 -o "$B/private-static-api-v1/libbaseline_original_unary.so"
nm -D --defined-only "$B/private-static-api-v1/libbaseline_original_unary.so" | awk '$3 ~ /^aclnnForeach(Abs|Neg)/ {print}'
