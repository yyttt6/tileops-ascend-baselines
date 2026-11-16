from pathlib import Path
import os,json,hashlib,time,datetime,subprocess,sysconfig,importlib.util
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-f16-add-v1');source=N/'source/experimental/moe/layernorm_add_add_quant/layernorm_add_add_quant.cpp';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='47facaea7ac9ef49afc3f86ffc911fb813df720593ca082156f96eb19be07d8d'
SDK=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-v1/compat/include/ascendc/host_api').resolve().parents[2];arch=SDK/'aarch64-linux';cc=SDK/'bin/bisheng';gcc=SDK.parents[1]/'compiler/usr/bin/g++-12'
assert sha(cc)=='fb635ef2090ff89f81c7962399b561d617c5a86417260c662d9bc6f84016f3c9';assert sha(gcc)=='cfc5c472d6ec4132cf368e3ed9e045ea9c27f29b3207e0e4e755e2abfe0ccb1b'
torchroot=Path(torch.__file__).parent;npuroot=Path(importlib.util.find_spec('torch_npu').origin).parent
includes=[torchroot/'include',torchroot/'include/torch/csrc/api/include',npuroot/'include',Path(sysconfig.get_paths()['include']),arch/'include',arch/'ascendc/include/basic_api/impl',arch/'ascendc/include/basic_api/interface',arch/'ascendc/include/highlevel_api',arch/'ascendc/include/highlevel_api/tiling',arch/'asc/include',arch/'asc/impl/c_api',arch/'asc/impl/utils',N/'source/experimental/mamba/common']
includes=[p for p in includes if p.is_dir()];obj=N/'original_transformer_f16_add.o';binary=N/'libclassic_transformer_add_original.so';assert not obj.exists() and not binary.exists()
cmd=[str(cc),'-x','asc','--npu-arch=dav-3510','-std=c++17','-fPIC','-DTORCH_MODE','-ferror-limit=8',str(source)]
for p in includes:cmd+=['-I',str(p)]
cmd+=['--cce-auto-sync','-include','kernel_operator.h','-c','-O2','-o',str(obj)]
report={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(source),'compile_command':cmd,'torch':torch.__version__,'torch_CXX11_ABI':torch._C._GLIBCXX_USE_CXX11_ABI,'compiler_sha256':sha(cc),'host_linker_sha256':sha(gcc),'original_source_modified':False,'optimization_auto_sync_preserved_from_original_receipt':True,'device_or_accuracy_qualification':False}
def save():
 (N/'BUILD_RECIPE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
save();start=time.monotonic();rc=subprocess.run(cmd,timeout=240).returncode;report.update(compile_exit_code=rc,compile_seconds=time.monotonic()-start);save();assert rc==0,'Original compile failed; source/flags unmodified, diagnose before retry'
link=[str(gcc),'-shared','-o',str(binary),str(obj),'-Wl,-z,defs']
libdirs=[torchroot/'lib',npuroot/'lib',SDK/'lib64',arch/'lib64',Path('/usr/local/lib')]
for p in libdirs:
 if p.is_dir():link+=['-L',str(p),'-Wl,-rpath,'+str(p)]
link+=['-l'+x for x in ['torch','torch_cpu','c10','torch_npu','ascendcl','platform','register','tiling_api','runtime','ascendc_runtime','dl','mmpa','c_sec','profapi','ascendalog','unified_dlog']]
report['link_command']=link;save();start=time.monotonic();rc=subprocess.run(link,timeout=120).returncode;report.update(link_exit_code=rc,link_seconds=time.monotonic()-start);save();assert rc==0,'Original link failed; no symbolaliases or sharedstack changes allowed'
report.update(object_sha256=sha(obj),binary_sha256=sha(binary),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='original_source_object_and_library_built_native_load_and_checker_pending');save();print(json.dumps({k:report[k] for k in ['status','compile_seconds','link_seconds','object_sha256','binary_sha256']}))
