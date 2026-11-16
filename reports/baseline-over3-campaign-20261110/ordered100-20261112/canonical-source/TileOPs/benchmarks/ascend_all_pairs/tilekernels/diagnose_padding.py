from pathlib import Path
import sys,os,json
WS=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(WS/'TileOPs'),str(WS/'TileOPs/src')]
from scripts.ascend_classic_check import require_guard
require_guard()
import torch,torch_npu
x=torch.tensor([[0.,1.]],device='npu',dtype=torch.float32)
value=-torch.finfo(torch.float32).max
rows={}
for name,call in {
    'functional_pad':lambda:torch.nn.functional.pad(x,(0,1),value=value),
    'full_on_device':lambda:torch.full((1,1),value,device=x.device,dtype=torch.float32),
    'host_constant_copy':lambda:torch.tensor([[value]],dtype=torch.float32).to(x.device),
    'finite_65504_pad':lambda:torch.nn.functional.pad(x,(0,1),value=-65504.),
}.items():
    t=call();torch.npu.synchronize()
    rows[name]={'values':t.cpu().tolist(),'int32_bits':t.view(torch.int32).cpu().tolist(),'isfinite':torch.isfinite(t).all().item()}
print(json.dumps(rows))
(WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels/padding_diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
torch.npu.synchronize()
