"""CPU-only saved-output diagnosis; no native call, oracle edit or acceptance."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,datetime
P=Path(__file__).resolve().parent;os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnostic forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());sys.path[:0]=[str(P),str(P/'workspace/TileOPs'),str(P/'workspace/TileOPs/src')]
import torch,pickle
from workloads.device import set_run_device
from workloads.numerics import reference_tolerance
set_run_device('cpu');torch.set_num_threads(8)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cp=json.loads((P/'rollout-ops-math-original-norm-309-l1-quiet-v2.CHECKPOINT.json').read_text()) if (P/'rollout-ops-math-original-norm-309-l1-quiet-v2.CHECKPOINT.json').exists() else None
# Tag is passed explicitly from the locally archived, immutable checkpoint.
tag=sys.argv[1];prep=P/'prepared-rollout/309';diag=P/'runs'/tag/'309/CHECKER_DIAGNOSTIC.pt';meta=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==meta['payload_sha256'];saved=torch.load(prep/'payload.pt',weights_only=False,map_location='cpu');d=torch.load(diag,weights_only=False,map_location='cpu');case=pickle.load((prep/'case.pkl').open('rb'));args=saved['inputs'];x,rm,rv,w,b=args;assert rm is None and rv is None
original=case.workload.ref_program(*args);produced=d['produced'];expected=d['expected'];assert isinstance(produced,torch.Tensor) and isinstance(expected,torch.Tensor)
z=x.double();axes=tuple(range(2,z.ndim));mean=z.mean(axes,keepdim=True);variance=(z-mean).square().mean(axes,keepdim=True);shape=(1,x.shape[1])+ (1,)*(x.ndim-2);gold=((z-mean)*torch.rsqrt(variance+case.params.get('eps',1e-5))*w.double().reshape(shape)+b.double().reshape(shape)).to(x.dtype)
tol=reference_tolerance(x.dtype)
def cmp(a,t):
 bad=~torch.isclose(a,t,**tol,equal_nan=True);return {'mismatched':int(bad.sum()),'numel':a.numel(),'max_abs':float((a.double()-t.double()).abs().max()),'rtol':tol['rtol'],'atol':tol['atol']}
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':309,'CPU_only_no_native_call':True,'no_NPU_imported':not any(k.split('.')[0] in ('torch_npu','acl') for k in sys.modules),'no_oracle_or_checker_changed':True,'no_numeric_retry':True,'original_checker_failure_preserved':True,'dtype':str(x.dtype),'shape':list(x.shape),'payload_sha256':sha(prep/'payload.pt'),'saved_diagnostic_sha256':sha(diag),'diagnostic_script_sha256':sha(Path(__file__)),'saved_produced_vs_original':cmp(produced,original),'saved_expected_vs_original':cmp(expected,original),'saved_produced_vs_FP64_math_cast_original_dtype':cmp(produced,gold),'original_vs_FP64_math_cast_original_dtype':cmp(original,gold),'FP64_only_diagnostic_not_acceptance':True}
assert report['no_NPU_imported'];dest=P/'server2-alignment-20261113';dest.mkdir(exist_ok=True);(dest/'ORIGINAL_NORM_309_CPU_DIAGNOSIS.json').write_text(json.dumps(report,indent=2)+'\n');torch.save({'FP64_math_cast_original_dtype':gold,'original_CPU_reference':original,'saved_produced':produced,'saved_expected':expected},dest/'ORIGINAL_NORM_309_CPU_DIAGNOSIS.pt');print(json.dumps(report),flush=True)
