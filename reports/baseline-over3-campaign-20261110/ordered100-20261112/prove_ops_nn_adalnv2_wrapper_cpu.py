"""CPU-only existing API argument semantics, original frozen inputs/checker."""
from pathlib import Path
import os,sys,json,pickle,hashlib,datetime,importlib.abc
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU only forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for i in [390,391,392,393]:
 d=P/'prepared-rollout'/str(i);meta=json.loads((d/'prepare.json').read_text());assert sha(d/'payload.pt')==meta['payload_sha256'] and sha(d/'case.pkl')==meta['descriptor_sha256'];case=pickle.load((d/'case.pkl').open('rb'));saved=torch.load(d/'payload.pt',weights_only=False);xs=saved['inputs'];case.__dict__['inputs']=xs;x,scale,shift=xs;before=[(v,v.contiguous().view(torch.uint8).clone()) for v in xs];eps=case.params.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
 # Same API formula, FP32 actual arguments; an extra singleton makes each original row one batch with sequence length1.
 xx=x.float().reshape(x.shape[0],1,x.shape[1]);sc=scale.float()-1.0;sh=shift.float();normalized=torch.nn.functional.layer_norm(xx,(xx.shape[-1],),eps=eps);produced=(normalized*(1.0+sc[:,None,:])+sh[:,None,:]).reshape_as(x).to(x.dtype);expected=case.reference(*xs);check=compare_outputs(produced,expected,case.verification);assert check.checked_outputs==check.total_outputs and not check.unchecked_reason;assert all(torch.equal(v.contiguous().view(torch.uint8),b) for v,b in before)
 bad=torch.zeros_like(produced)
 try:compare_outputs(bad,expected,case.verification)
 except (AssertionError,ValueError):negative=True
 else:negative=False
 assert negative;rows.append({'ordinal':i,'key':meta['key'],'payload_sha256':meta['payload_sha256'],'prepare_sha256':sha(d/'prepare.json'),'original_full_checker_passed':True,'inputs_bit_preserved':True,'negative_zero_output_rejected':True,'original_FP32_widening_retained':True,'API_parameter_conversion':'scale_fp32-1 then original API1+scale; FP32 rounding checked on full originalcase; no precision or tolerance change','batch_shape':[x.shape[0],1,x.shape[1]],'no_native_qualification':True});print('CPU_ORIGINAL_FROZEN_ADALNV2_WRAPPER_CHECKER_PASS',i,flush=True)
assert not any(n.split('.')[0] in ('torch_npu','acl') for n in sys.modules);(P/'OPS_NN_ADALNV2_CPU_PROOF.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'rows':rows,'no_original_or_candidate_measurement':True},indent=2))
