"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-adaptivemaxpool-v1');owner=sys.argv[1];dest=N/sys.argv[2]
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
 from ops_nn_adaptivemaxpool_api import AdaptiveMaxPoolAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_nn_adaptivemaxpool_api.py');native=AdaptiveMaxPoolAPI(torch,pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112/workspace'),N/'ADAPTIVEMAXPOOL_NATIVE_PACKAGE.json')
 call_index=0
 for dtype in [torch.float16,torch.bfloat16,torch.float32]:
  row={'op':'MaxPool3DWithArgmaxV2','public_API':'MaxPool3dWithArgmax','dtype':str(dtype),'status':'running'};out['rows'].append(row);save()
  for shape in [(2,3,1,5,7),(1,2,1,1,17)]:
   print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':call_index,'op':'MaxPool3DWithArgmaxV2','dtype':str(dtype),'shape':shape}),flush=True)
   x=((torch.arange(__import__('math').prod(shape)).reshape(shape).float()%7)-3).to(dtype);gold,gi=torch.nn.functional.adaptive_max_pool3d(x,(1,1,1),return_indices=True);gi=gi.to(torch.int32);a=x.to('npu');z=torch.empty_like(gold,device='npu');zi=torch.empty_like(gi,device='npu');native.call('AdaptiveMaxPool3d',[a],[z,zi],{'output_size':[1,1,1],'indices_dtype':3});torch.npu.synchronize();actual=[z.cpu(),zi.cpu()];torch.save({'input':x,'produced':actual,'expected':[gold,gi],'attrs':{'output_size':[1,1,1]},'not_formal_acceptance':True},N/(sys.argv[2]+'.group-'+str(call_index)+'.pt'));torch.testing.assert_close(actual[0],gold,rtol=0,atol=0);torch.testing.assert_close(actual[1],gi,rtol=0,atol=0);torch.testing.assert_close(a.cpu(),x,rtol=0,atol=0);print('BASELINE_PROBE_CALL_END '+json.dumps({'index':call_index,'values_and_indices_verified':True}),flush=True);call_index+=1
  row.update(status='full_outputs_and_inputs_verified',outputs=2,tensor_outputs_verified=4);save()
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
