#!/bin/bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
set +u
source /opt/tileops-ci/server2-20261030/config/device-env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec /opt/tileops-ci/server2-20261030/runtime/bin/python \
  /opt/tileops-ci/server2-20261030/config/check-toolchain-v1.py \
  --lock /opt/tileops-ci/server2-20261030/config/toolchain-v1.lock.json \
  --output "$TASK_ROOT/TOOLCHAIN.json"
