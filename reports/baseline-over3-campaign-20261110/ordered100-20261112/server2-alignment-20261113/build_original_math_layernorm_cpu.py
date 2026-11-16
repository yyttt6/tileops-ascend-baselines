"""Pure compiler/linker: original committed fast-launch source; no runtime or NPU import."""
from pathlib import Path
import os,sys,importlib.abc,subprocess,hashlib,json,datetime,sysconfig,importlib.metadata,shutil,fcntl
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU build forbids device import '+name,name=name)
sys.meta_path.insert(0,NoNPU())
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1');B=N.parent/'ops-math-add-v1';K='examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc';assert not N.exists();assert shutil.disk_usage('/').free>=20000000000;N.mkdir();lock=(N/'CPU_BUILD.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();r={'started_at':now(),'pid':os.getpid(),'hostname':os.uname().nodename,'status':'CPU_metadata_preflight','CPU_compiler_linker_only':True,'no_NPU_import_or_initialization_or_execution':True,'source_kernel_modified':False,'original_flags_preserved':True,'not_runtime_or_formal_acceptance':True}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
 t=N/'BUILD_RECEIPT.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(N/'BUILD_RECEIPT.json')
save()
try:
 m=json.loads((B/'SOURCE_MANIFEST.json').read_text());raw=(B/'source'/K).read_bytes();record=m['files'][K];assert sha(B/'source'/K)==record['sha256'];blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+bytes([0])+raw).hexdigest();assert blob==record['git_blob'];assert m['pin']=='64322f6487162c0934ec0364af57ed2f51175a03';source=N/'layer_norm.asc';source.write_bytes(raw)
 import torch
 assert torch.__version__=='2.13.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI is True;assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules
 compiler=Path(shutil.which('bisheng'));cxx=Path(os.environ['CXX']);cann=Path(os.environ['ASCEND_HOME_PATH']);arch=cann/'aarch64-linux';packages=Path(torch.__file__).resolve().parent.parent;npu=next(Path(p)/'torch_npu' for p in sys.path if (Path(p)/'torch_npu/include').is_dir());assert sysconfig.get_path('include');includes=[packages/'torch/include',packages/'torch/include/torch/csrc/api/include',npu/'include',Path(sysconfig.get_path('include')),arch/'include',arch/'asc/include',arch/'asc/impl/c_api',arch/'asc/impl/utils',arch/'tikcpp/tikcfw',arch/'compiler/tikcpp/tikcfw',cann/'compiler/tikcpp/tikcfw',arch/'asc/include/basic_api/impl',arch/'asc/include/basic_api/interface',arch/'asc/include/highlevel_api',arch/'asc/include/highlevel_api/tiling'];includes=[p for p in includes if p.is_dir()];command=[str(compiler),'-x','asc','--npu-arch=dav-3510','-std=c++17','-fPIC','-ferror-limit=8','-DEXTENSION_MODULE_NAME=allpairs_math_example',str(source)]
 for p in includes:command+=['-I',str(p)]
 command+=['-c','-O3','-o',str(N/'layernorm.o')];r.update(status='CPU_compile_original_source',pin=m['pin'],source_sha256=sha(source),source_git_blob=blob,original_build_script='frozen ops_math/build_layernorm.py: -xasc/dav3510/stdc++17/fPIC/ferror-limit8/module/O3 unchanged; only verified canonical paths relocated',torch=torch.__version__,torch_npu_metadata=importlib.metadata.version('torch-npu'),torch_CXX11_ABI=True,compiler=str(compiler),compiler_sha256=sha(compiler),CXX=str(cxx),CXX_sha256=sha(cxx),SDK=str(cann),command=command,header_roots=list(map(str,includes)));save();env=os.environ.copy();env['TMPDIR']=str(N/'tmp');(N/'tmp').mkdir()
 with (N/'compile.log').open('w') as out:rc=subprocess.run(command,cwd=N,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=480).returncode
 r['compile_exit_code']=rc;save();assert rc==0,'Original source compile failed, no automatic retry or source/flag edits'
 libraries=[packages/'torch/lib',npu/'lib',arch/'lib64',arch/'asc/lib64',cann/'lib64',Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64'),Path('/usr/local/lib')];libraries=[p for p in libraries if p.is_dir()];command=[str(cxx),'-shared','-o',str(N/'libops_math_layernorm.so'),str(N/'layernorm.o'),'-Wl,-z,defs']
 for p in libraries:command+=['-L',str(p),'-Wl,-rpath,'+str(p)]
 command+=['-l'+n for n in ('torch','torch_cpu','c10','torch_npu','ascendcl','platform','register','tiling_api','runtime','ascendc_runtime','dl','mmpa','c_sec','profapi','ascendalog','unified_dlog')];r.update(status='CPU_link_original_source',link_command=command);save()
 with (N/'link.log').open('w') as out:rc=subprocess.run(command,cwd=N,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120).returncode
 r['link_exit_code']=rc;save();assert rc==0,'Original host link failed; preserve evidence, no blindretry'
 r.update(status='CPU_build_complete_runtime_qualification_required',object_sha256=sha(N/'layernorm.o'),binary_sha256=sha(N/'libops_math_layernorm.so'),no_NPU_modules_loaded='torch_npu' not in sys.modules and 'acl' not in sys.modules)
except BaseException as e:r.update(status='failed_requires_diagnosis',error=repr(e));raise
finally:r['checked_at']=now();save()
print(json.dumps({k:r[k] for k in ['status','hostname','source_sha256','binary_sha256','no_NPU_modules_loaded']}))
