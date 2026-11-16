"""Elementwise compositions of complete, unchanged native SIMD APIs."""
from pathlib import Path
from .merge_native import _builder

OPERATIONS={
    'HardsigmoidFwdOp':('vadd','vmax','vmin','vdiv'),
    'HardswishFwdOp':('vadd','vmax','vmin','vdiv','vmul'),
    'LerpScalarFwdOp':('vsub','vmul','vadd'),
    'LerpTensorFwdOp':('vsub','vmul','vadd'),
    'PreluFwdOp':('vmin','vmax','vmul','vadd'),
    'SigmoidFwdOp':('vneg','vexp','vadd','vdiv'),
    'SiluFwdOp':('vneg','vexp','vadd','vdiv','vmul'),
    'SiluAndMulFwdOp':('vneg','vexp','vadd','vdiv','vmul'),
}


def composed(case):
    raise RuntimeError("Original-only boundary: chained SIMD operator composition is retired; original API support remains unresolved.")
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    name=case._call.signature.name
    if case.params.get('inplace',False):raise UnsupportedCase('This composition exposes the current out-of-place manifest contract only.')
    tl,builder,hashes=_builder()
    kernels={op:tl.compile(builder(op,'float32',precision='ftz_false' if op=='vexp' else 'exact' if op=='vdiv' else None),target='ascend') for op in OPERATIONS[name]}
    hashes.update(source_hashes(Path(__file__)))
    def run(x,*operands):
        original_dtype=x.dtype
        if name=='SiluAndMulFwdOp':
            half=x.shape[-1]//2
            operands=(x[...,half:],);x=x[...,:half]
        elif name=='PreluFwdOp':
            weight=operands[0]
            operands=(weight.reshape((1,weight.numel())+(1,)*(x.ndim-2)),)
        shape=tuple(torch.broadcast_shapes(x.shape,*(v.shape for v in operands)))
        n=1
        for extent in shape:n*=extent
        data=[F.pad(t.expand(shape).contiguous().float().flatten(),(0,-n%64)) for t in (x,*operands)]
        output=torch.empty_like(data[0])
        constants={v:torch.full((64,),v,device=x.device,dtype=torch.float32) for v in (0.,1.,3.,6.,float(case.params.get('weight',0.5)) if name=='LerpScalarFwdOp' else 0.5)}
        mask=torch.ones(64,device=x.device,dtype=torch.int32).view(torch.uint32)
        scratch=[torch.empty(64,device=x.device,dtype=torch.float32) for _ in range(8)]
        zero=constants[0.]
        for start in range(0,output.numel(),64):
            index=0
            def call(op,a,b=None):
                nonlocal index
                out=scratch[index];index+=1
                kernels[op](a,zero if b is None else b,zero,mask,out)
                return out
            a=data[0][start:start+64]
            if name in ('HardsigmoidFwdOp','HardswishFwdOp'):
                value=call('vdiv',call('vmin',call('vmax',call('vadd',a,constants[3.]),zero),constants[6.]),constants[6.])
                if name=='HardswishFwdOp':value=call('vmul',a,value)
            elif name in ('LerpScalarFwdOp','LerpTensorFwdOp'):
                weight=constants[float(case.params.get('weight',0.5))] if name=='LerpScalarFwdOp' else data[2][start:start+64]
                value=call('vadd',a,call('vmul',weight,call('vsub',data[1][start:start+64],a)))
            elif name=='PreluFwdOp':
                value=call('vadd',call('vmax',a,zero),call('vmul',call('vmin',a,zero),data[1][start:start+64]))
            else:
                value=call('vdiv',constants[1.],call('vadd',constants[1.],call('vexp',call('vneg',a))))
                if name!='SigmoidFwdOp':value=call('vmul',a,value)
                if name=='SiluAndMulFwdOp':value=call('vmul',value,data[1][start:start+64])
            output[start:start+64].copy_(value)
        return output[:n].reshape(shape).to(original_dtype)
    return Candidate(Implementation(run),'tilelang::original merging_kernel composition '+name,
                     'Only original complete native APIs: '+','.join(OPERATIONS[name])+'. Current operands, broadcast/channel/gate layouts, constants, exact FP32 widening and final storage conversion are materialized inside run. All native bodies/schedules and original manifest tolerances are unchanged. Composition numerical equivalence is case-validated; no Torch arithmetic.',hashes)
