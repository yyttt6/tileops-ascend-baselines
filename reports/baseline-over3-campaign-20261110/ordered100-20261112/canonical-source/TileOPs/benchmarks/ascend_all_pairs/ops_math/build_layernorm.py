#!/usr/bin/env python3
"""CPU-only compiler/linker; no runtime imports or NPU initialization."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from specs import ROOT,SOURCE,CACHE,EVIDENCE
out=CACHE/'layernorm';out.mkdir(exist_ok=True)
source=SOURCE/'examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc'
compiler=ROOT/'.tools/ascend/compiler/bin/bisheng';cann=ROOT/'.tools/ascend/cann';packages=ROOT/'.venv/lib/python3.12/site-packages'
includes=[packages/'torch/include',packages/'torch/include/torch/csrc/api/include',packages/'torch_npu/include',Path('/usr/include/python3.12'),cann/'include',cann/'compiler/tikcpp/tikcfw',cann/'compiler/ascendc/include/basic_api/impl',cann/'compiler/ascendc/include/basic_api/interface',cann/'compiler/ascendc/include/highlevel_api',cann/'compiler/ascendc/include/highlevel_api/tiling',cann/'asc/include',cann/'asc/impl/c_api',cann/'asc/impl/utils']
command=[str(compiler),'-x','asc','--npu-arch=dav-3510','-std=c++17','-fPIC','-ferror-limit=8','-DEXTENSION_MODULE_NAME=allpairs_math_example',str(source)]
for i in includes:command+=['-I',str(i)]
command+=['-c','-O3','-o',str(out/'layernorm.o')]
env=os.environ.copy();env.update(ASCEND_HOME_PATH=str(cann),ASCEND_TOOLKIT_HOME=str(cann),TMPDIR=str(CACHE/'tmp'));(CACHE/'tmp').mkdir(exist_ok=True)
receipt={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'command':command,'cpu_only':True,'native_source_modified':False};t=time.monotonic()
with (out/'compile.log').open('w') as log:r=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt.update(compile_exit_code=r.returncode,compile_seconds=time.monotonic()-t)
if r.returncode==0:
 cmd=['g++','-shared','-o',str(out/'libops_math_layernorm.so'),str(out/'layernorm.o'),'-Wl,-z,defs']
 for p in (packages/'torch/lib',packages/'torch_npu/lib',cann/'lib64',cann/'x86_64-linux/lib64',Path('/usr/local/lib')):cmd+=['-L',str(p),'-Wl,-rpath,'+str(p)]
 cmd+=['-l'+n for n in ('torch','torch_cpu','c10','torch_npu','ascendcl','platform','register','tiling_api','runtime','ascendc_runtime','dl','mmpa','c_sec','profapi','ascendalog','unified_dlog')]
 with (out/'link.log').open('w') as log:r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
 receipt.update(link_command=cmd,link_exit_code=r.returncode)
 if r.returncode==0:receipt['binary_sha256']=hashlib.sha256((out/'libops_math_layernorm.so').read_bytes()).hexdigest()
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
