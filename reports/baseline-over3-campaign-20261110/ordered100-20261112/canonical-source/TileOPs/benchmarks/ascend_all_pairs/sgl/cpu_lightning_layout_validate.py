"""CPU-only layout check; substitutes CPU primitives, never a native receipt."""
import importlib.abc
import json
import os
from pathlib import Path
import sys

if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
    raise RuntimeError('CPU-only startup environment required')
class BlockDevice(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch_npu', 'triton', 'sgl_kernel_npu', 'fla'):
            raise ImportError('Device imports forbidden: '+fullname)
sys.meta_path.insert(0, BlockDevice())
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src')]
import torch
from benchmarks.ascend_all_pairs.sgl.lightning_logits import compose
from workloads.device import set_run_device
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerWorkload
set_run_device('cpu')
torch.set_num_threads(4)
torch.manual_seed(23)
run = compose(torch, lambda x,w:x.float() @ w.float().T,
              lambda x,y:x*y, torch.maximum, lambda x:x.mean(-1), lambda x,y,s:x*s+y)
rows = []
for dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
    for scaled in (False, True):
        b,s,h,d,n,g = 2,33,8,16,1031,2
        q = torch.randn(b,s,h,d).to(dtype)
        k = torch.randn(b,n,g,d).to(dtype)
        w = torch.randn(s,h)
        start = torch.tensor([-3,0,5,1023,1031,1032] * 6, dtype=torch.int32)[:s]
        end = torch.tensor([7,1035,1027,1031,1035,1037] * 6, dtype=torch.int32)[:s]
        scale = torch.rand(b,n,g)+0.25 if scaled else None
        original = FP8LightningIndexerWorkload(b,s,h,d,n,g,True)
        reference = original.ref_program(q,k.float() if scale is None else k.float()*scale.unsqueeze(-1),w,start,end)[0]
        output = run(q,k,w,start,end,scale)
        assert torch.equal(torch.isneginf(output),torch.isneginf(reference))
        mask = torch.isfinite(reference)
        torch.testing.assert_close(output[mask], reference[mask], atol=3e-5, rtol=3e-4)
        rows.append({'dtype':str(dtype),'key_scale':scaled,'shape':list(output.shape),
                     'max_abs_error':(output[mask]-reference[mask]).abs().max().item(),
                     'masked_positions_equal':True,'status':'cpu_layout_passed'})
record={'claim':'Independent output/query/key/group layout and mask check only; no native execution or numerical acceptance',
        'device_backend_imported':any(x in sys.modules for x in ('torch_npu','triton','sgl_kernel_npu','fla')), 'cases':rows}
path=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/continuation_20261021T1415/cpu_lightning_layout_validation.json'
path.write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
