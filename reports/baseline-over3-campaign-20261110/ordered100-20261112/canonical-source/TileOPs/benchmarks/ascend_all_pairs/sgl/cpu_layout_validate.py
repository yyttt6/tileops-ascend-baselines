"""Independent CPU check of frozen pooling layout tables against original references."""
import importlib.abc
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')!='0':raise RuntimeError('Disable device autoload')
class BlockDevice(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch_npu','triton','sgl_kernel_npu','fla'):raise ImportError(fullname)
sys.meta_path.insert(0,BlockDevice())
import torch
from benchmarks.api import cases
from benchmarks.ascend_all_pairs.sgl import pool_layout
from benchmarks.ascend_all_pairs.sgl.cpu_prepare import CACHE,digest
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
torch.set_num_threads(4)
set_run_device('cpu')
results=[]
for file in sorted(CACHE.glob('*Pool*pt')):
    d=torch.load(file,weights_only=False)
    if d['op']=='MeanPoolingFwdOp':continue
    case=next(c for c in cases(d['op']) if c.id==d['case_id'])
    case.__dict__['inputs']=d['inputs']
    x,=case.inputs
    output_shape=case._call.tensors['output'][0]
    groups=pool_layout.load(d['op'],x.shape,output_shape,case.params)
    ndim=2 if 'Adaptive' in d['op'] else int(d['op'].split('Pool')[1][0])
    import math
    volume=math.prod(x.shape[-ndim:]);noutput=math.prod(output_shape[-ndim:])
    flat=x.reshape(-1,volume)
    maximum='Max' in d['op'];indices='Indices' in d['op']
    padded=torch.cat((flat,flat.new_full((flat.shape[0],1),float('-inf') if maximum else 0)),dim=1)
    output=flat.new_empty((flat.shape[0],noutput));idxout=torch.empty_like(output,dtype=torch.int64)
    for (width,divisor),items in groups.items():
        for begin in range(0,len(items),256):
            sub=items[begin:begin+256]
            pos=torch.tensor([r for _,r in sub]);order=[r for r,_ in sub]
            windows=padded[:,pos.remainder(volume+1)]
            if maximum:
                value,selected=windows.max(dim=-1)
                idxout[:,order]=pos[None].expand(flat.shape[0],-1,-1).gather(-1,selected[...,None])[...,0]
            else:value=windows.float().sum(dim=-1).div(divisor).to(x.dtype)
            output[:,order]=value
    got=(output.reshape(output_shape),idxout.reshape(output_shape)) if indices else output.reshape(output_shape)
    row={'op':d['op'],'case_id':d['case_id'],'check':'CPU layout only; no vendor/device validation'}
    try:
        result=compare_outputs(got,d['reference_calls'][digest(torch,d['inputs'])],case.verification)
        row.update(status='cpu_layout_passed',checked_outputs=result.checked_outputs,total_outputs=result.total_outputs)
    except Exception as exc:row.update(status='cpu_layout_failed',error_type=type(exc).__name__,error=str(exc))
    results.append(row)
path=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/continuation_20261021T1415/cpu_pool_layout_validation.json'
path.write_text(json.dumps(results,indent=2))
print(json.dumps({'cases':len(results),'passed':sum(r['status']=='cpu_layout_passed' for r in results),'failed':[r for r in results if r['status']!='cpu_layout_passed'],'torch_npu_imported':'torch_npu' in sys.modules}))
