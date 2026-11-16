"""Guard-only four complete original cases. Correctness and provenance, not15raw."""
from pathlib import Path
import os,sys,json,pickle,datetime,hashlib,fcntl,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-adaln-decode-ln-v1');P=Path('/opt/baseline-over3-20261110/ordered100-20261112');T=P/'workspace/TileOPs';owner=sys.argv[1];dest=N/sys.argv[2];sys.path[:0]=[str(T/'src'),str(T),str(N),str(P)];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with open('/run/lock/ascend-npu/device-0.lock') as f:
 g=json.load(f);assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp()
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('whole foreground hostguard required')
out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','rows':[],'not_formal15raw_or_performance_acceptance':True,'guard_owner':owner,'source_sha256':{str(p):sha(p) for p in [Path(__file__),N/'ops_nn_adaln_decode_ln_api.py',N/'LAYERNORM_BASE_PACKAGE.json',P/'prove_ops_nn_adalnv2_wrapper_cpu.py']}}
def put():dest.write_text(json.dumps(out,indent=2)+'\n')
put()
try:
 import torch,torch_npu
 from workloads.device import set_run_device
 from workloads.numerics import compare_outputs
 from ops_nn_adaln_decode_ln_api import LayerNormNativeAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;prop=torch.npu.get_device_properties(0);assert(prop.cube_core_num,prop.vector_core_num)==(32,64);native=LayerNormNativeAPI(torch,N/'LAYERNORM_BASE_PACKAGE.json');out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 proof=json.loads((P/'OPS_NN_ADALNV2_CPU_PROOF.json').read_text());assert proof['no_NPU_imported'] and len(proof['rows'])==4
 for i in [390]:
  d=P/'prepared-rollout'/str(i);meta=json.loads((d/'prepare.json').read_text());assert sha(d/'payload.pt')==meta['payload_sha256'] and sha(d/'case.pkl')==meta['descriptor_sha256'];assert next(v for v in proof['rows'] if v['ordinal']==i)['prepare_sha256']==sha(d/'prepare.json');set_run_device('cpu');case=pickle.load((d/'case.pkl').open('rb'));xs=torch.load(d/'payload.pt',weights_only=False)['inputs'];case.__dict__['inputs']=xs;expected=case.reference(*xs);eps=case.params.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps;args=tuple(v.to('npu') for v in xs);x,scale,shift=args
  print('BASELINE_PROBE_CALL_BEGIN '+json.dumps({'index':i,'op':'LayerNorm_complete_affine_for_single_row_AdaLN','dtype':str(x.dtype),'shape':list(x.shape)}),flush=True);assert x.shape[0]==1 and tuple(scale.shape)==tuple(shift.shape)==tuple(x.shape);y,m,r=native(x.float().contiguous(),scale.float().reshape(-1).contiguous(),shift.float().reshape(-1).contiguous(),(x.shape[-1],),eps);produced=y.reshape_as(x).to(x.dtype);torch.npu.synchronize();check=compare_outputs(produced.cpu(),expected,case.verification);assert check.checked_outputs==check.total_outputs and not check.unchecked_reason
  xf=xs[0].float();goldm=xf.mean(-1,keepdim=True);goldr=torch.rsqrt(xf.var(-1,keepdim=True,unbiased=False)+eps);torch.testing.assert_close(m.cpu(),goldm,rtol=1e-4,atol=1e-4);torch.testing.assert_close(r.cpu(),goldr,rtol=1e-4,atol=1e-4)
  for actual,original in zip(args,xs,strict=True):assert torch.equal(actual.cpu().contiguous().view(torch.uint8),original.contiguous().view(torch.uint8))
  out['rows'].append({'ordinal':i,'key':meta['key'],'status':'original_full_checker_passed','native_complete_outputs':3,'auxiliary_mean_and_rstd_checked':True,'input_bits_preserved':True,'payload_sha256':meta['payload_sha256'],'prepare_sha256':sha(d/'prepare.json')});put();print('BASELINE_PROBE_CALL_END '+json.dumps({'index':i,'complete_native_outputs':3}),flush=True)
 out['status']='single_original_decode_case_LayerNorm_affine_passed_attribution_pending'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);libs={v.split()[-1] for v in maps.splitlines() if len(v.split())>=6 and v.split()[-1].startswith('/') and '.so' in v.split()[-1]};out['loaded_library_sha256']={v:sha(v) for v in libs if Path(v).is_file()};put()
