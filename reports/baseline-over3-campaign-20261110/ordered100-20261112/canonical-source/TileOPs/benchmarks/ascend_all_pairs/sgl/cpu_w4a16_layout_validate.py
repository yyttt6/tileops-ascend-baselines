"""CPU-only exhaustive-byte packing checks, not native numerical acceptance."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
    raise RuntimeError('CPU-only startup environment required')
class BlockDevice(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch_npu','triton','sgl_kernel_npu','fla'):
            raise ImportError('Device imports forbidden: '+fullname)
sys.meta_path.insert(0,BlockDevice())
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src')]
import torch
from benchmarks.ascend_all_pairs.sgl.w4a16 import unpack
from workloads.gemm import repack_w4a16_weight,dequantize_w4a16_weight
torch.set_num_threads(4)
torch.manual_seed(19)
table=torch.tensor([[b&15,b>>4] for b in range(256)],dtype=torch.float32)
checks=[]
for n,k in ((64,128),(128,256),(64,4096)):
    original=torch.arange(n*k//2).to(torch.uint8).reshape(n,k//2)
    packed=repack_w4a16_weight(original)
    q=unpack(torch,packed,table)
    expected=dequantize_w4a16_weight(packed,torch.ones(n,k//128),torch.zeros(n,k//128,dtype=torch.uint8))
    assert torch.equal(q,expected)
    for dtype in (torch.float16,torch.bfloat16):
        scale=torch.rand(n,k//128).to(dtype)
        zero=torch.randint(0,16,(n,k//128),dtype=torch.uint8)
        value=((q-zero.float().repeat_interleave(128,dim=1))*scale.float().repeat_interleave(128,dim=1)).to(dtype)
        expected=dequantize_w4a16_weight(packed,scale,zero).to(dtype)
        assert torch.equal(value,expected)
        checks.append({'shape':[n,k],'dtype':str(dtype),'all_256_byte_values':True,'exact_weight_match':True})
record={'claim':'CPU format/dequantization layout validation only; CPU primitive substitutions, no native/device pass',
        'device_backend_imported':any(x in sys.modules for x in ('torch_npu','triton','sgl_kernel_npu','fla')),
        'checks':checks,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('w4a16.py'),ROOT/'TileOPs/workloads/gemm.py')}}
p=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/continuation_20261021T1415/cpu_w4a16_layout_validation.json'
p.write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
