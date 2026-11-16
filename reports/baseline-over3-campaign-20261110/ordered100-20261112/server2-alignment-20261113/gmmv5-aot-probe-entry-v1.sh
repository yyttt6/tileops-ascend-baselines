#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-gmmv5-aot-original-v2/runtime-aot-probe-v1.sh
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/ordered100-20261112/probe_gmmv5_original_aot_v1.py gmmv5-original-aot-probe-v1 774
