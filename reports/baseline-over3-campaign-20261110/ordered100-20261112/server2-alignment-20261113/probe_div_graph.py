"""Diagnostic original GraphCall Div, no baseline coverage or timing claim."""
from pathlib import Path
import os,sys,json,hashlib,ast,ctypes as C,datetime,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-div-v1');owner=sys.argv[1];dest=N/sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp()
out={'status':'initializing','zero_baseline_coverage':True,'checks':[]}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 m=json.loads((N/'DIV_PACKAGE.json').read_text());root=N/'graph-private-v1'
 for f,r in m['files'].items():assert sha(root/f)==r['sha256'],f
 source=N/'original_graph_call.py';assert sha(source)=='caab7e464808f00a23269f15f9cf14d7a9a5ea88af6d8134718f7523ef8013ca'
 cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GraphCall');ns={'C':C,'os':os,'json':json,'Path':Path,'WORKSPACE':N};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(source),'exec'),ns);binding=ns['GraphCall']('ops-math');out['original_GraphCall_SHA']=sha(source)
 for dtype in [torch.float16,torch.float32]:
  for broadcast in [False,True]:
   x=(torch.arange(51).reshape(3,17).float()%17-8).to(dtype);y=((torch.arange(17 if broadcast else 51).reshape((1,17) if broadcast else (3,17)).float()%7)+1).to(dtype);a=x.to('npu');b=y.to('npu');z=torch.empty((3,17),dtype=dtype,device='npu');gold=x/y
   out['status']='before_Div';save();binding.call('Div',[a,b],[z],{});torch.npu.synchronize();got=z.cpu();torch.testing.assert_close(got,gold,atol=0.001 if dtype==torch.float16 else 1e-6,rtol=0.001 if dtype==torch.float16 else 1e-6)
   assert torch.equal(a.cpu().view(torch.int16 if dtype==torch.float16 else torch.int32),x.view(torch.int16 if dtype==torch.float16 else torch.int32));assert torch.equal(b.cpu().view(torch.int16 if dtype==torch.float16 else torch.int32),y.view(torch.int16 if dtype==torch.float16 else torch.int32));out['checks'].append({'dtype':str(dtype),'broadcast':broadcast,'all_outputs_and_inputs':True})
 # Full original INT32 marker operation must remain executable with the vendor loaded.
 marker=torch.zeros(1024,dtype=torch.int32,device='npu');marker.add_(1);torch.npu.synchronize();assert torch.equal(marker.cpu(),torch.ones(1024,dtype=torch.int32));out['original_INT32_marker_passed']=True
 out.update(status='original_Graph_Div_functional_passed',full_output_and_inputs_verified=True)
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);out['loaded_library_sha256']={line.split()[-1]:sha(line.split()[-1]) for line in maps.splitlines() if len(line.split())>=6 and line.split()[-1].startswith('/') and '.so' in line.split()[-1] and Path(line.split()[-1]).is_file()};save()
