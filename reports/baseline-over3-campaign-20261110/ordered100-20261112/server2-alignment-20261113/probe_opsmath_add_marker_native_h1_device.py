"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1');owner=sys.argv[1];dest=N/sys.argv[2]
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
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_math_add_api_v2.py')
 print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':0,'op':'Add','dtype':'torch.int32','scope':'original timer flush_buffer[:1].add_(1) call'}),flush=True)
 x=torch.ones(1,dtype=torch.int32,device='npu');x[:1].add_(1);torch.npu.synchronize();actual=x.cpu();assert torch.equal(actual,torch.tensor([2],dtype=torch.int32));torch.save({'produced':actual,'expected':torch.tensor([2],dtype=torch.int32),'not_formal_acceptance':True},N/(sys.argv[2]+'.group-0.pt'))
 print('BASELINE_PROBE_CALL_END '+json.dumps({'index':0,'original_timer_marker_call_verified':True}),flush=True)
 out['rows']=[{'op':'Add','public_API':'torch original aclnnInplaceAdds marker','dtype':'torch.int32','status':'full_outputs_and_inputs_verified','outputs':1,'tensor_outputs_verified':1}];out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
