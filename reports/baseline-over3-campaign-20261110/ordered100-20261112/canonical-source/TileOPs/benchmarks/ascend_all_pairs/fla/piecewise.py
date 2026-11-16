"""Finite real piecewise identities using original FLA magnitude/tap APIs."""
from pathlib import Path


def piecewise(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .absolute import absolute
    from .native_affine import make_affine
    magnitude=absolute(case)
    affine,hashes=make_affine()
    name=case._call.signature.name
    params=case.params

    def run(x,weight=None,maximum=None):
        dtype=x.dtype
        x=x.float()
        def constant(value):return torch.full((),value,device=x.device,dtype=torch.float32)
        def clamp(value,lower,upper):
            if lower is not None:value=extreme(value,lower,True)
            if upper is not None:value=extreme(value,upper,False)
            return value
        def extreme(value,bound,is_max):
            bound=bound.float() if isinstance(bound,torch.Tensor) else constant(bound)
            shape=torch.broadcast_shapes(value.shape,bound.shape)
            value=value.expand(shape);bound=bound.expand(shape)
            delta=affine(value,1.0,affine(bound,-1.0))
            center=affine(affine(value,1.0,bound),0.5)
            return affine(magnitude.implementation.run(delta),0.5 if is_max else -0.5,center)
        if name in ('ReluFwdOp','LeakyReluFwdOp','PreluFwdOp'):
            if name=='PreluFwdOp':
                shape=() if weight.numel()==1 else (1,weight.numel(),*(1 for _ in x.shape[2:]))
                weight=weight.float().reshape(shape)
                positive=affine(weight,0.5,constant(0.5))
                negative=affine(weight,-0.5,constant(0.5))
            else:
                slope=params.get('negative_slope',0.0) if name=='LeakyReluFwdOp' else 0.0
                positive=(1.0+slope)/2.0
                negative=(1.0-slope)/2.0
            value=affine(magnitude.implementation.run(x),negative,affine(x,positive))
        elif name in ('MaximumFwdOp','MinimumFwdOp'):
            value=extreme(x,weight,name=='MaximumFwdOp')
        elif name=='ClampScalarFwdOp':
            value=clamp(x,params.get('min'),params.get('max'))
        elif name=='ClampTensorFwdOp':
            value=clamp(x,weight,maximum)
        elif name=='HardtanhFwdOp':
            value=clamp(x,params.get('min_val',-1.0),params.get('max_val',1.0))
        else:
            gate=affine(clamp(affine(x,1.0,constant(3.0)),0.0,6.0),1.0/6.0)
            value=affine(x,gate) if name=='HardswishFwdOp' else gate
        return value.to(dtype)

    return Candidate(Implementation(run),'FLA scalar L2 reciprocal magnitude + original native tap/residual',
        'Use the finite real identities ReLU=(x+abs(x))/2, PReLU=(1+w)*x/2+(1-w)*abs(x)/2, min/max=(x+y +/- abs(x-y))/2, and clamp from two magnitude identities. Hard sigmoid/swish retain their exact piecewise formulas. Current channel weights and every intermediate value are evaluated by original FLA kernels in FP32; final manifest cast only. Original reciprocal-norm overflow/nonfinite limits and numerical error remain subject to unchanged exact-case verification.',
        magnitude.source_sha256|hashes|source_hashes(Path(__file__)))
