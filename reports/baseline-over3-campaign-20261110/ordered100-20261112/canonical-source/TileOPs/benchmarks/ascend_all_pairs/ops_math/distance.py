"""Vector norm and zero count select original Cdist(x, zero, p)."""
from pathlib import Path
from math import prod
from types import SimpleNamespace
from .specs import SOURCE

def make(case,op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .graph import make as graph_make
    p=case.params;shape,dtype=p['x'];out_shape,out_dtype=p['output'];count=op=='CountNonzeroFwdOp'
    if dtype not in ('float16','bfloat16','float32','bool'):
        raise UnsupportedCase('Cdist owns only FP16/BF16/FP32 input math; this input dtype has no audited lossless supported widening.')
    ndim=len(shape);dim=p.get('dim');axes=list(range(ndim)) if dim is None else [d%ndim for d in dim] if isinstance(dim,(tuple,list)) else [dim%ndim];kept=[i for i in range(ndim) if i not in axes];k=prod(shape[i] for i in axes);rows=prod(shape[i] for i in kept)
    if count and k>2**24:raise UnsupportedCase('FP32 count output would not preserve every integer in this reduction range.')
    child=graph_make(SimpleNamespace(params={'output':((rows,1),'float32'),'_cdist':True,'_p':0. if count else float(p['ord'])}),'MulFwdOp')
    def run(x):
        packed=x.permute(*kept,*axes).contiguous().reshape(rows,k).float()
        out=child.implementation.run(packed,torch.zeros((1,k),device=x.device,dtype=torch.float32))
        return out.reshape(out_shape).to(getattr(torch,out_dtype))
    src=[q for part in ('op_host','op_kernel','op_graph') for q in (SOURCE/'math/cdist'/part).rglob('*') if q.is_file() and q.suffix in ('.cpp','.h','.py')]
    return Candidate(Implementation(run),'ops-math::Cdist(x,zero,p)',
      'Pack reduction axes into the distance coordinate; original p=0 counts nonzero, p=1/2/inf computes corresponding norm. Exact input widening and output cast; count range bounded for exact integer conversion.',
      child.source_sha256|source_hashes(Path(__file__),*src),cleanup=child.cleanup)
