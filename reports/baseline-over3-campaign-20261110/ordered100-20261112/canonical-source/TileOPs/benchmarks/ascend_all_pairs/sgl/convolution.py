"""Materialize strided complete convolution windows for original SGL GEMV."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import math


def conv_nd(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes,UnsupportedCase
    from benchmarks.ascend_sgl_gemv_cases import sgl_gemv_cases
    x,w,*bias=case.inputs
    if x.dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Original SGL vector LoRA only accepts FP16/BF16 operands')
    ndim=x.ndim-2;p=case.params;groups=p['groups']
    def tup(v):return (v,)*ndim if isinstance(v,int) else tuple(v)
    stride=tup(p['stride']);padding=tup(p['padding']);dilation=tup(p['dilation']);kernel=w.shape[2:]
    outsp=tuple((n+2*pad-d*(k-1)-1)//s+1 for n,pad,d,k,s in zip(x.shape[2:],padding,dilation,kernel,stride))
    cin=x.shape[1]//groups;cout=w.shape[0]//groups;rows=x.shape[0]*math.prod(outsp)
    bias_present=bool(bias and bias[0] is not None)
    width=cin*math.prod(kernel)+int(bias_present)
    # Factories see shapes only; weights and window values are rebuilt per run.
    block=4096
    natives={n:sgl_gemv_cases(SimpleNamespace(inputs=(torch.empty((n,width),device=x.device,dtype=x.dtype),torch.empty((cout,width),device=x.device,dtype=x.dtype)),params={'trans_a':False,'trans_b':True})) for n in {min(rows,block),rows%block} if n}
    def run(x,w,bias=None):
        padded=F.pad(x,tuple(v for pad in reversed(padding) for v in (pad,pad)))
        windows=padded
        for axis,(size,dil,step) in enumerate(zip(kernel,dilation,stride)):
            windows=windows.unfold(axis+2,dil*(size-1)+1,step)
        windows=windows[(slice(None),)*(2+ndim)+tuple(slice(None,None,d) for d in dilation)]
        outputs=[]
        for group in range(groups):
            selected=windows[:,group*cin:(group+1)*cin]
            order=(0,*range(2,2+ndim),1,*range(2+ndim,2+2*ndim))
            matrix=selected.permute(order).contiguous().reshape(rows,-1)
            weight=w[group*cout:(group+1)*cout].contiguous().reshape(cout,-1)
            if bias is not None:
                matrix=torch.cat((matrix,torch.ones((rows,1),device=x.device,dtype=x.dtype)),dim=-1)
                weight=torch.cat((weight,bias[group*cout:(group+1)*cout,None]),dim=-1)
            output=torch.cat([natives[min(block,rows-i)].implementation.run(matrix[i:i+block],weight) for i in range(0,rows,block)],dim=0)
            outputs.append(output.reshape(x.shape[0],*outsp,cout))
        result=torch.cat(outputs,dim=-1)
        return result.permute(0,ndim+1,*range(1,ndim+1)).contiguous()
    hashes=source_hashes(Path(__file__))
    for native in natives.values():hashes.update(native.source_sha256)
    return replace(next(iter(natives.values())),implementation=Implementation(run),source_sha256=hashes,
        adaptation='Padding and strided unfold gather complete channel/spatial reduction windows; flatten weights with identical order, optional constant bias coordinate; original complete SGL GEMV per independent row/group. No partial reduction or external convolution arithmetic.',
        limitations=('All im2col materialization, optional bias-coordinate packing and row/group launches count in timing; every original dtype/shape requires numerical validation.',))


def conv2d(case):
    # Retain the reviewed representative route for its exact original domain;
    # all other kernels use the full-window layout route in this isolated file.
    w=case.inputs[1];p=case.params
    pair=lambda v:(v,v) if isinstance(v,int) else tuple(v)
    if tuple(w.shape[-2:])==(1,1) and pair(p['stride'])==(1,1) and pair(p['padding'])==(0,0) and p['groups']==1:
        from .classics import conv2d as reviewed
        return reviewed(case)
    return conv_nd(case)
