"""Pooling through complete native reduction rows; only window layout is external."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import math


def pool(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes,UnsupportedCase
    x,=case.inputs
    name=case._call.signature.name
    maximum='Max' in name
    adaptive='Adaptive' in name
    indices='Indices' in name
    p=case.params
    ndim=2 if adaptive else int(name.split('Pool')[1][0])
    spatial=tuple(x.shape[-ndim:])
    prefix=tuple(x.shape[:-ndim])
    def tup(v):return (v,)*ndim if isinstance(v,int) else tuple(v)
    output_shape=tuple(case._call.tensors['output'][0])
    outsp=output_shape[-ndim:]
    from . import pool_layout,pool_launch_layout
    parts,layout_path=pool_launch_layout.load(name,tuple(x.shape),output_shape,p)
    host='python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py' if maximum else 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    module='sgl_kernel_npu.sample.argmax_softmax_prob' if maximum else 'sgl_kernel_npu.fla.layernorm_gated'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu',module,'argmax_softmax_prob_fused' if maximum else 'layer_norm_fwd_npu_simd',host)
    mulhost='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    scale=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',mulhost)
    volume=math.prod(spatial)
    prepared=[]
    for cpu_positions,cpu_gathers,cpu_order,width,divisor in parts:
        positions=torch.tensor(cpu_positions,device=x.device,dtype=torch.int64) if indices else None
        gather_positions=torch.tensor(cpu_gathers,device=x.device,dtype=torch.int64)
        order=torch.tensor(cpu_order,device=x.device,dtype=torch.int64)
        prepared.append((positions,gather_positions,order,width,divisor,torch.ones(width,device=x.device,dtype=x.dtype)))
    def run(x):
        flat=x.reshape(-1,volume)
        pad=flat.new_full((flat.shape[0],1),float('-inf') if maximum else 0)
        padded=torch.cat((flat,pad),dim=-1)
        output=x.new_empty((flat.shape[0],math.prod(outsp)))
        idxout=torch.empty(output.shape,device=x.device,dtype=torch.int64) if indices else None
        for positions,gather_positions,order,width,divisor,ones in prepared:
            windows=padded[:,gather_positions].contiguous().reshape(-1,width)
            if maximum:
                selected=fn(windows)[0]
                values=windows.gather(1,selected[:,None]).reshape(flat.shape[0],-1)
                if indices:
                    original=positions[None].expand(flat.shape[0],-1,-1).reshape(-1,width)
                    idxout[:,order]=original.gather(1,selected[:,None]).reshape(flat.shape[0],-1)
            else:
                values=fn(windows,ones,None,1e-5)[1]
                if width!=divisor:
                    values=scale(values.reshape(-1,1),torch.zeros_like(values).reshape(-1,1),width/divisor).reshape(-1)
                values=values.to(x.dtype).reshape(flat.shape[0],-1)
            output[:,order]=values
        output=output.reshape(output_shape)
        return (output,idxout.reshape(output_shape)) if indices else output
    return Candidate(Implementation(run),module+('.argmax_softmax_prob_fused' if maximum else '.layer_norm_fwd_npu_simd()[1]')+' + native mul_add for explicit divisor',
        'Gather complete spatial windows using shape-only index tables; native original argmax/mean reduction, gather original values and absolute indices, layout-scatter outputs. Padding/crop and optional constant divisor use original APIs; no Torch pooling/reduction arithmetic.',
        source_hashes(Path(__file__),Path(pool_layout.__file__),Path(pool_launch_layout.__file__),layout_path,LIBRARIES/'sgl-kernel-npu'/host,LIBRARIES/'sgl-kernel-npu'/mulhost),
        limitations=('Original argmax finite-input contract; all frozen pooling inputs and controls must be validated individually.',))
