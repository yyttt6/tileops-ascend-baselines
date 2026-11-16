#!/bin/bash
set -euo pipefail
source /opt/baseline-over3-20261110/server2-aligned-stack/environment.sh
export PATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/native-frontend92/compiler/bin:${PATH}"
export CPATH="/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/compat/include:${CPATH:-}"
B=/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/source/build-device-v1
mkdir -p "$B/RECOVERY"
tar -czf "$B/RECOVERY/soc950-recovery-20261113T201336Z-before.tar.gz" -C "$B" tbe/dynamic binary/ascend950/bin
python3 /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/source/scripts/util/ascendc_impl_build.py "$B/autogen/aic-ascend950-ops-info.ini" "" "" "$B/tbe/ascendc" "$B/tbe/dynamic" "$B/autogen"
python3 - <<'CHECK_SOURCE'
from pathlib import Path
import re
b=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/source/build-device-v1')
for op in ['foreach_log', 'foreach_log1p', 'foreach_expm1']:
 p=b/'tbe/dynamic'/(op+'.py');s=p.read_text();assert 'arch35/'+op+'.cpp' in s or ('arch35/'+op) in s,(str(p),'950source not selected')
 print('EXACT950SOURCE',op,flush=True)
CHECK_SOURCE
cmake --build "$B" --target gen_bin_scripts --parallel 4
if [ -d "$B/binary/ascend950/bin/opc_cmd" ]; then mv "$B/binary/ascend950/bin/opc_cmd" "$B/RECOVERY/soc950-recovery-20261113T201336Z-opc_cmd"; fi
cmake --build "$B" --target prepare_binary_compile_ascend950 --parallel 1
OPC_NUM=$(wc -l < "$B/binary/ascend950/bin/opc_cmd/opc_cmd.sh")
cmake -S /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-extra-v1/source -B "$B" -DOPC_NUM_ascend950="$OPC_NUM"
cmake --build "$B" --target binary --parallel 4
cmake --build "$B" --target gen_bin_info_config --parallel 1
