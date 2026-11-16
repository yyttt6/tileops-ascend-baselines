"""Independent CPU-only JIT source-package build; no kernel execution or torch import."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import sys

WS = Path('/root/dyq/workspace')
BASE = WS / '.cache/all-pairs/ops_nn'
SRC = BASE / 'source'
OPS = '''relu sigmoid hard_sigmoid hard_swish leaky_relu p_relu elu selu mish softplus_v2 gelu_v2 swish swi_glu gelu_mul softmax_v2 log_softmax_v2 layer_norm_v4 group_norm_v2 batch_norm_v3 batch_norm_grad_v3 ada_layer_norm_v2 add_layer_norm adaptive_avg_pool2d avg_pool1_d avg_pool_v2 avg_pool3_d max_pool3d_with_argmax_v2 max_pool_with_argmax_v3 lp_norm_v2 ascend_quant_v2 dequantize dynamic_quant dynamic_block_quant rms_norm add_rms_norm inplace_add_rms_norm adaptive_max_pool2d mat_mul_v3 batch_mat_mul_v3 conv2d_v2 convolution_forward conv3d_v2'''.split()

OPS += ['foreach_'+n for n in 'abs cos div_list erf exp expm1 lerp_list lerp_scalar log log1p maximum_list minimum_list mul_list neg pow_list reciprocal round_off_number sign sin sqrt sub_list tanh'.split()]

def main():
    expanded='--expanded' in sys.argv or '--masked' in sys.argv
    install_name='install3' if '--masked' in sys.argv else 'install2' if expanded else 'install'
    if expanded:OPS.extend(['unsorted_segment_sum','unsorted_segment_prod','unsorted_segment_min','unsorted_segment_max','kth_value','foreach_pow_scalar'])
    if '--masked' in sys.argv:OPS.extend(['masked_scatter','quant_batch_matmul_v3'])
    if not SRC.exists():
        shutil.copytree(WS / 'ascend_baselines/ops-nn', SRC,
                        ignore=shutil.ignore_patterns('.git', 'build', 'build_out', '__pycache__'))
    snapshot = {str(p.relative_to(SRC)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in SRC.rglob('*') if p.is_file() and 'build' not in p.relative_to(SRC).parts
                and 'build_out' not in p.relative_to(SRC).parts and '.git' not in p.relative_to(SRC).parts}
    stamp=str(int(time.time()))
    for old in ('build.log','build_receipt.json','source_snapshot.json'):
        if (BASE/old).exists():shutil.copy2(BASE/old,BASE/(old+'.'+stamp+'.prior'))
    (BASE / 'source_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    record = dict(ops=OPS, started_at=time.time(), mode='CPU host compile and JIT source packaging; no device access',
                  source_snapshot=str(BASE/'source_snapshot.json'),kernel_or_schedule_edits=False,
                  host_dtype_metadata_reused=False)
    cmd = ['bash', '-c', 'source /root/dyq/workspace/env.sh\nexec bash ./build.sh "$@"', 'original-build',
           '--pkg','--jit','--soc=ascend950','--ops='+','.join(OPS), '--vendor_name=allpairs',
           '-j8','--build-type=Release','--ccache=off']
    record['command']=cmd
    env=os.environ.copy();env.update(CMAKE_BUILD_PARALLEL_LEVEL='8',PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(BASE/'tmp'),ASCEND_WORK_PATH=str(BASE/'work'),ASCEND_CACHE_PATH=str(BASE/'kernel_cache'))
    with (BASE/'build.log').open('w') as f:
        record['build_exit_code']=subprocess.run(cmd,cwd=SRC,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if record['build_exit_code']==0:
        with (BASE/'install.log').open('w') as f:
            record['install_exit_code']=subprocess.run(['cmake','--install',str(SRC/'build'),'--prefix',str(BASE/install_name)],stdout=f,stderr=subprocess.STDOUT).returncode
        vendors=BASE/install_name/'packages/vendors'
        if vendors.is_dir():shutil.copytree(vendors,BASE/install_name/'vendors',dirs_exist_ok=True)
    record['finished_at']=time.time()
    (BASE/'build_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record.get('install_exit_code',record['build_exit_code'])
if __name__=='__main__':raise SystemExit(main())
