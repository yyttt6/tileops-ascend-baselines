"""Guard-only originalGraphCall+SDK9.2 diagnostic; not a benchmark result."""
from pathlib import Path
import os,sys,json,hashlib,ast,ctypes as C,datetime,traceback,fcntl
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-segmentmin-api-v1');owner=sys.argv[1];dest=N/sys.argv[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp()
out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','zero_baseline_coverage':True,'profile':'SDK9.2 Graph runtime/compiler+nativefrontend92 Python, originalvendor min source/kernel','no_kernel_or_flags_changes':True}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 source=N/'original_graph_call.py';assert sha(source)=='caab7e464808f00a23269f15f9cf14d7a9a5ea88af6d8134718f7523ef8013ca';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GraphCall');ns={'C':C,'os':os,'json':json,'Path':Path,'WORKSPACE':N};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),ns);binding=ns['GraphCall']('ops-nn');out['GraphCall_source_sha256']=sha(source)
 x=(torch.arange(21).reshape(7,3).float()%11-5)/8;gold=x.amin(0,keepdim=True);a=x.to('npu');ids=torch.zeros(7,device='npu',dtype=torch.int32);number=torch.tensor(1,device='npu',dtype=torch.int32);result=torch.empty((1,3),device='npu',dtype=torch.float32)
 out['status']='before_originalGraphCall';save();binding.call('UnsortedSegmentMin',[a,ids,number],[result],{},constants={2:(C.c_int32*1)(1)});torch.npu.synchronize();actual=result.cpu();torch.save({'input':x,'produced':actual,'expected':gold,'zero_baseline_coverage':True},N/(sys.argv[2]+'.pt'));torch.testing.assert_close(actual,gold,atol=0,rtol=0);assert torch.equal(a.cpu().view(torch.int32),x.view(torch.int32));assert torch.equal(ids.cpu(),torch.zeros(7,dtype=torch.int32)) and number.cpu().item()==1
 out.update(status='originalGraph92_Min_small_functional_passed_diagnostic_only',full_output_and_inputs_verified=True)
except BaseException as exc:out.update(status='failed_requires_diagnosis',error=repr(exc),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);files={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={p:sha(p) for p in files if Path(p).is_file()};save()
print(json.dumps({'status':out['status'],'not_baseline':True}))
