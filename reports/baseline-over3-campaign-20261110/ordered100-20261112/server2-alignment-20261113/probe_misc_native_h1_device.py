"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-misc-v1');owner=sys.argv[1];dest=N/sys.argv[2]
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
 from ops_nn_misc_api import MiscNativeAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_nn_misc_api.py');native=MiscNativeAPI(torch,N/'MISC_NATIVE_PACKAGE.v2.json')
 for op in ['ForeachPowScalar','SoftplusV2']:
  for dtype in [torch.float16,torch.bfloat16,torch.float32]:
   row={'op':op,'dtype':str(dtype),'status':'running','public_API':'ForeachPowScalar' if op=='ForeachPowScalar' else 'Softplus'};out['rows'].append(row);save()
   for values in [[[1,-2,.5],[3.25,-4.5,5.75]],[-3,.5,2,4,1,2,.25,8,4,-2,1,1.25,3,.75,4,-8,2]]:
    x=torch.tensor(values,dtype=dtype);x=x.abs()+.5 if op=='ForeachPowScalar' else x;gold=torch.rsqrt(x) if op=='ForeachPowScalar' else torch.nn.functional.softplus(x,beta=1.,threshold=20.);a=x.to('npu');y=native(a.contiguous(),row['public_API'],1.,20.);torch.npu.synchronize();tol={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[dtype];torch.testing.assert_close(y.cpu(),gold,rtol=tol,atol=tol);torch.testing.assert_close(a.cpu(),x,rtol=0,atol=0)
   row.update(status='full_outputs_and_inputs_verified',outputs=2);save()
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
