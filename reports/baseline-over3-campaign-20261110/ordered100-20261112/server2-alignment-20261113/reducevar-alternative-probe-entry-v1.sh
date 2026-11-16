#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1/runtime-alternative-v1.sh
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python /opt/baseline-over3-20261110/ordered100-20261112/probe_reducevar_original_alternative_v1.py reducevar-alternative-original-probe-v1 683 684
