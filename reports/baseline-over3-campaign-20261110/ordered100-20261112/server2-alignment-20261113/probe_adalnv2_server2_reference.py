"""Guard-only four complete original cases. Correctness and provenance, not15raw."""
from pathlib import Path
import os,sys,json,pickle,datetime,hashlib,fcntl,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-v2');P=Path('/opt/baseline-over3-20261110/ordered100-20261112');T=P/'workspace/TileOPs';owner=sys.argv[1];dest=N/sys.argv[2];sys.path[:0]=[str(T/'src'),str(T),str(N),str(P)];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with open('/run/lock/ascend-npu/device-0.lock') as f:
 g=json.load(f);assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp()
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('whole foreground hostguard required')
out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','rows':[],'not_formal15raw_or_performance_acceptance':True,'guard_owner':owner,'source_sha256':{str(p):sha(p) for p in [Path(__file__),N/'ops_nn_adalnv2_api.py',N/'ADALNV2_NATIVE_PACKAGE.json']}}
def put():dest.write_text(json.dumps(out,indent=2)+'\n')
put()
try:
 import torch,torch_npu
 from ops_nn_adalnv2_api import AdaLayerNormV2API
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;prop=torch.npu.get_device_properties(0);assert(prop.cube_core_num,prop.vector_core_num)==(32,64);native=AdaLayerNormV2API(torch,N/'ADALNV2_NATIVE_PACKAGE.json');out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 for i,(batch,width) in enumerate([(1,4096),(2,1152),(2,4096)]):
  x=((torch.arange(batch*width).float()%29-14)/8).reshape(batch,1,width);sc=((torch.arange(batch*width).float()%11-5)/8).reshape(batch,width);sh=((torch.arange(batch*width).float()%7-3)/16).reshape(batch,width);eps=1e-5;gold=torch.nn.functional.layer_norm(x,(width,),eps=eps)*(1+sc[:,None,:])+sh[:,None,:];gm=x.mean(-1,keepdim=True);gr=torch.rsqrt(x.var(-1,keepdim=True,unbiased=False)+eps);args=tuple(t.to('npu') for t in (x,sc,sh))
  print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':i,'op':'AdaLayerNormV2','dtype':'torch.float32','shape':list(x.shape)}),flush=True);y,m,r=native(*args,eps);torch.npu.synchronize();torch.testing.assert_close(y.cpu(),gold,rtol=1e-4,atol=1e-4);torch.testing.assert_close(m.cpu(),gm,rtol=1e-4,atol=1e-4);torch.testing.assert_close(r.cpu(),gr,rtol=1e-4,atol=1e-4)
  for actual,original in zip(args,(x,sc,sh),strict=True):assert torch.equal(actual.cpu().contiguous().view(torch.uint8),original.contiguous().view(torch.uint8))
  out['rows'].append({'index':i,'status':'complete_original_API_outputs_verified','native_complete_outputs':3,'auxiliary_mean_and_rstd_checked':True,'input_bits_preserved':True,'shape':list(x.shape),'not_frozen_manifest_acceptance':True});put();print('BASELINE_PROBE_CALL_END '+json.dumps({'index':i,'complete_native_outputs':3}),flush=True)
 out['status']='server2_full_API_functionally_passed_private_kernel_attribution_pending'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);libs={v.split()[-1] for v in maps.splitlines() if len(v.split())>=6 and v.split()[-1].startswith('/') and '.so' in v.split()[-1]};out['loaded_library_sha256']={v:sha(v) for v in libs if Path(v).is_file()};put()
