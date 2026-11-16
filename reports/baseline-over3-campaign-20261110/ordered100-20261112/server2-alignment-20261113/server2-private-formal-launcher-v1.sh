#!/bin/bash
set -euo pipefail
P=$(cd -- "$(dirname -- "$0")" && pwd)
family=$1
shift
[ "$family" = r-template ] || exit 64
case "$1" in prepare|ops-transformer-softmax-expansion) ;; *) exit 64 ;; esac
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python "$P/rollout_bootstrap.py" "$@"
