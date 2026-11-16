from pathlib import Path
import os,sys,importlib.abc,json,pickle,hashlib,datetime
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
from torch_qk2_native_rms_rope_support import build
set_run_device('cpu');torch.set_num_threads(8);rows=[]
def cpu_rms(x,w,epsilon):
 assert x.dtype==w.dtype==torch.float32
 rstd=torch.rsqrt(x.square().mean(-1,keepdim=True)+epsilon)
 return x*rstd*w,rstd
def cpu_rotary(x,r1,r2,rotary_mode):
 assert rotary_mode=='interleave' and x.ndim==r1.ndim==r2.ndim==4 and x.dtype==r1.dtype==r2.dtype==torch.float32
 a,b=x[...,0::2],x[...,1::2]
 rotated=torch.stack((-b,a),dim=-1).flatten(-2)
 return x*r1+rotated*r2
for i in [1356,1358]:
 d=P/'prepared-torch-qk2-08f-v2'/str(i);case=pickle.load((d/'case.pkl').open('rb'));saved=torch.load(d/'payload.pt',weights_only=False);xs=saved['inputs'];case.__dict__['inputs']=xs;bits=[x.contiguous().view(torch.uint8).clone() for x in xs];measured,proof=build(case,cpu_rms,cpu_rotary);out=measured(*xs);check=compare_outputs(out,saved['expected'],case.verification);assert check.checked_outputs==check.total_outputs and not check.unchecked_reason;assert all(torch.equal(x.contiguous().view(torch.uint8),b) for x,b in zip(xs,bits));rows.append({'ordinal':i,'original_complete_checker_passed_with_CPU_API_surrogate':True,'inputs_bit_preserved':True,'proof':proof});case._release()
assert not any(x.split('.')[0] in ('torch_npu','acl') for x in sys.modules)
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'no_NPU_imported':True,'no_native_API_qualification_claimed':True,'actual_native_math_and_performance_require_formal_guarded_validation':True,'support_sha256':hashlib.sha256((P/'torch_qk2_native_rms_rope_support.py').read_bytes()).hexdigest()};(P/'TORCH_QK2_NATIVE_RMS_ROPE_CPU_PROOF.json').write_text(json.dumps(result,indent=2));print('CPU_ORIGINAL_QK_FULL_REVERSE_AST_AND_ORIGINAL_CHECKER_PASS',len(rows),flush=True)
