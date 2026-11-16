"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-layernorm-v1');owner=sys.argv[1];dest=N/sys.argv[2]
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
 from ops_nn_layernorm_bf16_api import LayerNormNativeAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};out['API_binding_sha256']=sha(N/'ops_nn_layernorm_bf16_api.py');native=LayerNormNativeAPI(torch,N/'LAYERNORM_NATIVE_PACKAGE.v2.json')
 call_index=0
 for dtype in [torch.bfloat16]:
  row={'op':'LayerNormV4','public_API':'LayerNorm','dtype':str(dtype),'status':'running'};out['rows'].append(row);save()
  for shape in [(1,4096),(2,16384)]:
   print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':call_index,'op':'LayerNormV4','dtype':str(dtype),'shape':shape}),flush=True)
   width=shape[-1];x=((torch.arange(__import__('math').prod(shape)).reshape(shape).float()%23-11)/8).to(dtype);w=((torch.arange(width).float()%7)/8+.75).to(dtype);b=((torch.arange(width).float()%5-2)/16).to(dtype)
   xf=x.float();gold=torch.nn.functional.layer_norm(xf,(width,),w.float(),b.float(),1e-5).to(dtype);gm=xf.mean(-1,keepdim=True);gr=torch.rsqrt(xf.var(-1,keepdim=True,unbiased=False)+1e-5)
   xa=x.to('npu');wa=w.to('npu');ba=b.to('npu');y,mean,rstd=native(xa.contiguous(),wa.contiguous(),ba.contiguous(),(width,),1e-5);torch.npu.synchronize();torch.save({'inputs':[x,w,b],'produced':[y.cpu(),mean.cpu(),rstd.cpu()],'expected':[gold,gm,gr],'shape':shape,'normalized_shape':[width],'eps':1e-5,'scope':'BF16 separate from failedFP16; diagnostic notperformance'},N/(sys.argv[2]+'.group-'+str(call_index)+'.pt'));tol=1.6e-2;torch.testing.assert_close(y.cpu(),gold,rtol=tol,atol=tol);torch.testing.assert_close(mean.cpu(),gm,rtol=1e-4,atol=1e-4);torch.testing.assert_close(rstd.cpu(),gr,rtol=1e-4,atol=1e-4)
   for actual,original in [(xa,x),(wa,w),(ba,b)]:torch.testing.assert_close(actual.cpu(),original,rtol=0,atol=0)
   print('BASELINE_PROBE_CALL_END '+json.dumps({'index':call_index,'tensor_outputs_verified':3,'inputs_preserved':True}),flush=True);call_index+=1
  row.update(status='full_outputs_and_inputs_verified',outputs=2,output_groups=2,tensor_outputs_verified=6,auxiliary_outputs=['mean_float32','rstd_float32']);save()
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
