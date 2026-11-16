"""Semantically matching original routes; not registered as executable providers.

Used to record current deployment failures without changing vendor math/schedules.
"""
from pathlib import Path
from .original import load

UNARY = {'AbsFwdOp':'abs', 'ExpFwdOp':'exp','LogFwdOp':'log','SinFwdOp':'sin',
         'CosFwdOp':'cos','TanhFwdOp':'tanh','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt',
         'ErfFwdOp':'erf','FloorFwdOp':'floor','CeilFwdOp':'ceil','TruncFwdOp':'trunc',
         'RoundFwdOp':'nearbyint','SigmoidFwdOp':'sigmoid','IsnanFwdOp':'isnan','IsinfFwdOp':'isinf'}

def unary(case):
    import torch
    import torch.nn.functional as F
    import tilelang
    import tilelang.language as T
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    name = UNARY[case._call.signature.name]
    module, hashes = load('testing/python/math/test_math_fast_math.py', ['run_single_arg_mathop_test'],
                          extract={'run_single_arg_mathop_test':'main'})
    x, = case.inputs
    size = x.numel()
    n = ((size+1023)//1024)*32
    native = module.run_single_arg_mathop_test(name, getattr(T,name), M=32, N=n, dtype=T.float32)
    kernel = tilelang.compile(native, out_idx=1, target='ascend', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:False})
    def run(x):
        out = kernel(F.pad(x.float().contiguous().flatten(), (0,32*n-size)).reshape(32,n)).flatten()[:size]
        dtype = torch.bool if name in ('isnan','isinf') else x.dtype
        return out.to(dtype).reshape(x.shape)
    return Candidate(Implementation(run), 'tilelang::run_single_arg_mathop_test/'+name,
                     'Extract unchanged original PrimFunc, original 32x32 blocks and 128 threads, target selection only; exact FP32 widening and pad/crop.', hashes|source_hashes(Path(__file__)))


def layer_norm(case):
    import torch
    from math import prod
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    module, hashes = load('examples/norm/layernorm.py', ['_layernorm_fwd'],
                          jit={'_layernorm_fwd':'tilelang.jit(target="ascend", out_idx=[-3,-2,-1])'})
    x,w,b = case.inputs
    shape = tuple(case.params.get('normalized_shape', (x.shape[-1],)))
    d, m = prod(shape), x.numel()//prod(shape)
    native = module._layernorm_fwd(m,d,eps=case.params.get('eps',1e-5),in_dtype='float32',out_dtype='float32')
    def run(x,w,b):
        gamma = torch.ones(d,device=x.device) if w is None else w.float().reshape(d)
        beta = torch.zeros(d,device=x.device) if b is None else b.float().reshape(d)
        return native(x.float().contiguous().reshape(m,d),gamma,beta)[0].to(x.dtype).reshape(x.shape)
    return Candidate(Implementation(run),'tilelang::_layernorm_fwd','Original LayerNorm body, default schedule, target selection and exact FP32/layout conversion only.',hashes|source_hashes(Path(__file__)))


def softmax(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    module, hashes = load('examples/online_softmax/online_softmax.py',['softmax_kernel'],
                          jit={'softmax_kernel':'tilelang.jit(target="ascend")'})
    x, = case.inputs
    dim = case.params.get('dim',-1)
    def run(x):
        moved=x.movedim(dim,-1)
        module.M, module.N = moved.numel()//moved.shape[-1], moved.shape[-1]
        out=module.softmax_kernel(moved.contiguous().reshape(-1,moved.shape[-1]),dtype=str(x.dtype).split('.')[-1])
        return out.reshape(moved.shape).movedim(-1,dim).contiguous()
    return Candidate(Implementation(run),'tilelang::online_softmax.softmax_kernel','Original online softmax and default schedule; target selection plus reduction-axis layout.',hashes|source_hashes(Path(__file__)))


def amax(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    module, hashes = load('examples/topk/example_topk.py',['tl_topk'],jit={'tl_topk':'tilelang.jit(target="ascend")'})
    def run(x):
        out=module.tl_topk(x.float().contiguous(),1,64)[0].to(x.dtype)
        return out if case.params['keepdim'] else out.squeeze(-1)
    return Candidate(Implementation(run),'tilelang::tl_topk(k=1).values','Original topk value selection, exact input widening; original documented blk_m=64.',hashes|source_hashes(Path(__file__)))

DIAGNOSTIC_PROVIDERS = {op:unary for op in UNARY}
DIAGNOSTIC_PROVIDERS.update({'LayerNormFwdOp':layer_norm,'SoftmaxFwdOp':softmax,'AmaxFwdOp':amax})
