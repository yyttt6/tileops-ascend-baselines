from pathlib import Path
import sys,os,importlib.abc,pickle,json,hashlib,datetime
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';sys.path[:0]=[str(T/'src'),str(T),str(P)];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU only forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
from selective_scan_bare_grouped_device import build
from selective_scan_bare_device import build as original_build
set_run_device('cpu');torch.set_num_threads(8);d=P/'prepared-selective-scan-bare-08f/1499';case=pickle.load((d/'case.pkl').open('rb'));xs=torch.load(d/'payload.pt',weights_only=False)['inputs'];case.__dict__['inputs']=xs;before=[(x,x.contiguous().view(torch.uint8).clone()) for x in xs if isinstance(x,torch.Tensor)];fn,ref,checker,proof=build(case);old,_,_,_=original_build(case);produced=fn(*xs);expected=ref(*xs);check=compare_outputs(produced,expected,checker);assert check.checked_outputs==check.total_outputs==2 and not check.unchecked_reason;assert all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(produced,old(*xs),strict=True));assert all(torch.equal(x.contiguous().view(torch.uint8),b) for x,b in before);assert not any(x.split('.')[0] in ('torch_npu','acl') for x in sys.modules)
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':1499,'no_NPU_imported':True,'both_outputs_bitexact_vs_original_FP32':True,'frozen_FP64_oracle_checker_passed':True,'inputs_bit_preserved':True,'proof':proof,'helper_sha256':hashlib.sha256((P/'grouped_scan_layout.py').read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256((P/'selective_scan_bare_grouped_device.py').read_bytes()).hexdigest()};(P/'GROUPED_SCAN_BARE_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('BARE_LAYOUT_FROZEN_CHECKER_AND_BITEXACT_PASS',flush=True)
