#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
cmake --build /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-adaptivemaxpool-v1/source/build-server2-v1 --target cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_opapi cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_opmaster cann-ops-nn-allpairs_linux-aarch64_ascendc_cust_op_proto --parallel 4
