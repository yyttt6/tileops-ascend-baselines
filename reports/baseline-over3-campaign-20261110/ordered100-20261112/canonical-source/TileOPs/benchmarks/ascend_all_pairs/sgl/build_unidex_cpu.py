"""Offline compile only: no framework imports, device loading or execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/'.cache/all-pairs/sgl'
EVIDENCE=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl'


def main():
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    receipt=EVIDENCE/f'unidex_cpu_build_{run}.json'
    flags=ROOT/'ascend_baselines/sgl-kernel-npu/build/csrc/CMakeFiles/sgl_kernel_npu.dir/flags.make'
    content=flags.read_text()
    torch=re.search(r'-I(\S+/torch)/include ',content)[1]
    npu=re.search(r'-I(\S+/torch_npu)/include ',content)[1]
    abi=re.search(r'-D_GLIBCXX_USE_CXX11_ABI=(\d)',content)[1]
    directory=CACHE/'unidex';directory.mkdir(parents=True,exist_ok=True)
    tmp=CACHE/'tmp';tmp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FLA_NPU_DISABLE_PTH='1',TMPDIR=str(tmp),XDG_CACHE_HOME=str(CACHE/'xdg'))
    source=Path(__file__).with_name('native_unidex')
    cmds=[['cmake','-S',str(source),'-B',str(directory),'-DSGL_OFFLINE_CONFIG=ON',
           f'-DTORCH_DIR={torch}',f'-DTORCH_NPU_DIR={npu}',f'-DSGL_TORCH_ABI={abi}'],
          ['cmake','--build',str(directory),'-j','2']]
    records=[]
    for i,cmd in enumerate(cmds):
        trace=directory/f'cpu_build_{run}_{i}.strace'
        with (directory/f'cpu_build_{run}_{i}.log').open('w') as log:
            rc=subprocess.call(['strace','-f','-e','trace=%file,ioctl','-o',str(trace),*cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
        text=trace.read_text(errors='replace')
        device_lines=[line for line in text.splitlines() if re.search(r'/dev/(davinci|hisi_hdc|devmm)',line)]
        records.append({'command':cmd,'exit_code':rc,'trace':str(trace.relative_to(ROOT)),
                        'npu_device_accesses':device_lines,'finished_at':datetime.now(timezone.utc).isoformat()})
        record=json.dumps({'mode':'offline compile only; no shared library loading',
            'original_build_flags':str(flags.relative_to(ROOT)),
            'original_build_flags_sha256':hashlib.sha256(flags.read_bytes()).hexdigest(),'steps':records},indent=2)+'\n'
        receipt.write_text(record)
        (EVIDENCE/'unidex_cpu_build.json').write_text(record)
        print(json.dumps({'step':i,'exit_code':rc,'npu_device_accesses':len(device_lines)}),flush=True)
        if device_lines:raise RuntimeError('Unexpected device access in offline compiler')
        if rc:return rc
    vendor=ROOT/'ascend_baselines/sgl-kernel-npu/csrc/sparsity_driven_kv_offload/unidex_copy'
    sources=[source/'CMakeLists.txt',source/'binding.cpp',vendor/'op_host/unidex_copy.cpp',vendor/'op_kernel/unidex_copy_kernel.cpp']
    binary=directory/'libsgl_unidex_compat.so'
    (directory/'build_success.json').write_text(json.dumps({
        'source_fingerprint':hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'mode':'offline compile only','receipt':str(receipt.relative_to(ROOT))},indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
