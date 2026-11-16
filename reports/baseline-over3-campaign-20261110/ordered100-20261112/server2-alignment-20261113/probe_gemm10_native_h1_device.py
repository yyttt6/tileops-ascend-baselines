"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1');owner=sys.argv[1];dest=N/sys.argv[2]
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
 from ops_nn_gemm10_api import Gemm10API
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_nn_gemm10_api.py');native=Gemm10API(torch,N/'GEMM10_NATIVE_PACKAGE.json')
 generator=torch.Generator().manual_seed(7313)
 for index,(rows,k,cols) in enumerate([(3,64,32),(1,128,64)]):
  row={'op':'aclnnMatmul','dtype':'torch.float32','shape':[rows,k,cols],'cubeMathType':'KEEP_DTYPE=0','status':'running'};out['rows'].append(row);save();print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':index,'op':'aclnnMatmul','native_dtype':'torch.float32','shape':[rows,k,cols]}),flush=True)
  x=torch.randn((rows,k),generator=generator,dtype=torch.float32);w=torch.randn((k,cols),generator=generator,dtype=torch.float32);gold=(x.double()@w.double()).float();xa=x.to('npu');nt=w.T.contiguous().to('npu');z=torch.empty_like(gold,device='npu');native.call('MatMulV3',[xa,nt,None,None],[z],{'transpose_x1':False,'transpose_x2':True});torch.npu.synchronize();actual=z.cpu();torch.save({'inputs':[x,w],'produced':actual,'expected_cpu_fp64_cast_fp32':gold,'shape':[rows,k,cols],'cubeMathType':0,'not_formal_or_reference_policy_change':True},N/(sys.argv[2]+'.group-'+str(index)+'.pt'));torch.testing.assert_close(actual,gold,rtol=1e-5,atol=1e-4);torch.testing.assert_close(xa.cpu(),x,rtol=0,atol=0);torch.testing.assert_close(nt.cpu(),w.T.contiguous(),rtol=0,atol=0);row.update(status='full_outputs_and_inputs_verified',outputs=1);save();print('BASELINE_PROBE_CALL_END '+json.dumps({'index':index,'full_verified':True}),flush=True)
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
