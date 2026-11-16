"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-inplacerms-v1');owner=sys.argv[1];dest=N/sys.argv[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ticks(pid):return int(pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
with pathlib.Path('/run/lock/ascend-npu/device-0.lock').open() as f:
 g=json.load(f);ch=g['child'];assert g['phase']=='running' and g['device']==0 and g['owner']==owner and ch['pid']==os.getpgrp() and ticks(ch['pid'])==ch['start_ticks'] and ticks(g['pid'])==g['start_ticks'] and g['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Hostguard kernel lock not held')
out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','rows':[],'not_baseline_15raw_or_kernel_attribution_acceptance':True,'device_kernel_provenance_pending':True,'guard_owner':owner,'probe_source_sha256':sha(pathlib.Path(__file__))}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from ops_nn_inplacerms_api import InplaceRmsAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_nn_inplacerms_api.py');native=InplaceRmsAPI(torch,pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112/workspace'),N/'INPLACERMS_NATIVE_PACKAGE.json')
 call_index=0
 sys.path[:0]=[str(native.W/'TileOPs/src'),str(native.W/'TileOPs')]
 from types import SimpleNamespace
 from workloads.norm import _norm_reference,norm_verification
 from workloads.numerics import reference_tolerance
 out['frozen_adapter_sha256']=sha(native.source)
 for dtype in [torch.float16,torch.bfloat16]:
  row={'op':'InplaceAddRmsNorm','public_API':'aclnnInplaceAddRmsNorm','dtype':str(dtype),'status':'running'};out['rows'].append(row);save()
  for shape in [(3,4096),(2,8192)]:
   print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':call_index,'op':'InplaceAddRmsNorm','dtype':str(dtype),'shape':shape}),flush=True)
   torch.manual_seed(819+call_index);x=torch.randn(shape,dtype=dtype);residual=torch.randn(shape,dtype=dtype);w=torch.randn(shape[-1],dtype=dtype);params={'eps':1e-6,'normalized_shape':(shape[-1],)};gold=_norm_reference('FusedAddRMSNormFwdOp',(x,residual,w),params);case=SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name='FusedAddRMSNormFwdOp')),params=params);fn=native.prepare(case);a=x.to('npu');r=residual.to('npu');ww=w.to('npu');actual=tuple(z.cpu() for z in fn(a,r,ww));torch.npu.synchronize();torch.save({'input':x,'residual':residual,'weight':w,'produced':actual,'expected':gold,'not_formal_acceptance':True},N/(sys.argv[2]+'.group-'+str(call_index)+'.pt'));tol=norm_verification(dtype).tolerance(reference_tolerance(dtype))
   for got,expected in zip(actual,gold,strict=True):torch.testing.assert_close(got,expected,equal_nan=True,**tol)
   for got,expected in [(a.cpu(),x),(r.cpu(),residual),(ww.cpu(),w)]:assert torch.equal(got.view(torch.int16),expected.view(torch.int16))
   print('BASELINE_PROBE_CALL_END '+json.dumps({'index':call_index,'full_outputs_and_inputs_verified':True,'original_workload_tolerance':tol}),flush=True);call_index+=1
  row.update(status='full_outputs_and_inputs_verified',outputs=2,tensor_outputs_verified=4);save()
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
