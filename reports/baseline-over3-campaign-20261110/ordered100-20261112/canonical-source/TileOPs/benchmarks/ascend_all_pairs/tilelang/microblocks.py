"""Independent slices through unchanged Ascend instruction test kernels.

These are intentionally unscheduled host loops; replacing them with a new
batched kernel would alter the vendor schedule and is outside this task.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import sys


def _load_builder(relative,outer,inner=None):
    from benchmarks.ascend_baselines import LIBRARIES,WORKSPACE,source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    tl=_runtime()
    source=LIBRARIES/'tilelang'/relative
    tree=ast.parse(source.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==outer)
    fn.decorator_list=[]
    if inner:
        innerfn=next(n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name==inner)
        fn.body=fn.body[:fn.body.index(innerfn)+1]+[ast.Return(ast.Name(inner,ast.Load()))]
    header=ast.parse('import tilelang\nimport tilelang.ascend.language as T\nfrom tilelang.ascend.language import simd as S\n').body
    adapted=ast.unparse(ast.fix_missing_locations(ast.Module(body=header+[fn],type_ignores=[])))+'\n'
    key=hashlib.sha256(adapted.encode()).hexdigest()
    path=WORKSPACE/'.cache/all-pairs/tilelang/sources'/('micro-'+key+'.py');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==adapted
    else:path.write_text(adapted)
    spec=importlib.util.spec_from_file_location('_allpairs_micro_'+key,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return tl,getattr(module,outer),source_hashes(source,path,Path(__file__))|_runtime_hashes(tl)


def bitwise(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    is_not=case._call.signature.name in ('BitwiseNotFwdOp','LogicalNotFwdOp')
    a=case.inputs[0]
    if case._call.signature.name=='LogicalNotFwdOp' and a.dtype!=torch.bool:raise UnsupportedCase('XOR1 is logical inversion only for boolean inputs; original kernel has no general nonzero test.')
    tl,builder,hashes=_load_builder('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel')
    kernel=tl.compile(builder(64),target='ascend')
    def run(a,b=None):
        shape=tuple(a.shape) if b is None else tuple(torch.broadcast_shapes(a.shape,b.shape))
        dt=a.dtype
        def prepare(x):
            x=x.expand(shape).contiguous()
            return x.view(torch.int32).flatten() if dt==torch.int64 else x.to(torch.int32).flatten()
        aa=prepare(a);bb=(torch.ones_like(aa) if dt==torch.bool and is_not else aa) if b is None else prepare(b)
        count=aa.numel()
        import torch.nn.functional as F
        aa=F.pad(aa,(0,-count%64));bb=F.pad(bb,(0,-count%64))
        out=torch.empty_like(aa);scratch=torch.empty(64,device=a.device,dtype=torch.int32)
        for start in range(0,aa.numel(),64):
            if is_not and dt!=torch.bool:kernel(aa[start:start+64],bb[start:start+64],scratch,out[start:start+64])
            else:kernel(aa[start:start+64],bb[start:start+64],out[start:start+64],scratch)
        value=out[:count]
        value=value.view(torch.int64) if dt==torch.int64 else value.to(dt)
        return value.reshape(shape)
    return Candidate(Implementation(run),'tilelang::bitwise_kernel(64) '+('Not' if is_not else 'Xor'),
                     'Original fixed64 SIMD kernel; independent flat slices, broadcast packing; int64 bit-preserving pair-of-int32 view, narrow integer widening/cast-back; no new device schedule.',hashes,
                     limitations=('Many original tiny launches; no performance claim.',))


def clamp(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    tl,builder,hashes=_load_builder('testing/ascend/language/test_tilelang_ascend_clamp.py','test_clamp_nan_and_inverted_bounds','main')
    kernel=tl.compile(builder('float32',128),target='ascend')
    op=case._call.signature.name
    p=case.params
    def run(x,*bounds):
        if op=='MaximumFwdOp':low,high=bounds[0],None
        elif op=='MinimumFwdOp':low,high=None,bounds[0]
        elif op=='ReluFwdOp':low,high=0.,None
        elif op=='ClampTensorFwdOp':low,high=bounds
        elif op=='HardtanhFwdOp':low,high=p.get('min_val',-1.),p.get('max_val',1.)
        else:low,high=p.get('min'),p.get('max')
        tensors=[x]+[v for v in (low,high) if isinstance(v,torch.Tensor)]
        shape=tuple(torch.broadcast_shapes(*(v.shape for v in tensors)))
        size=1
        for v in shape:size*=v
        def prep(v,default):
            if v is None:v=default
            if isinstance(v,torch.Tensor):return F.pad(v.expand(shape).contiguous().float().flatten(),(0,-size%128))
            return torch.full((size+-size%128,),v,device=x.device,dtype=torch.float32)
        xx,ll,hh=prep(x,0),prep(low,-float('inf')),prep(high,float('inf'))
        out=torch.empty_like(xx)
        for start in range(0,xx.numel(),128):kernel(xx[start:start+128],ll[start:start+128],hh[start:start+128],out[start:start+128])
        return out[:size].to(x.dtype).reshape(shape)
    return Candidate(Implementation(run),'tilelang::test_clamp_nan_and_inverted_bounds.main',
                     'Extract unchanged original128-element clamp kernel; scalar/tensor-bound broadcast and exact FP32 widening; independent slices and output crop/cast.',hashes)


def small_arithmetic(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    op=case._call.signature.name
    if case.params.get('rounding_mode') is not None:
        raise UnsupportedCase('Original precise division returns floating quotient; floor/trunc modes require a separate blocked generic rounding API.')
    relative='testing/ascend/language/simdvf/test_simdvf_arithmetic.py'
    if op=='AbsFwdOp':
        tl,builder,hashes=_load_builder(relative,'test_absolute_difference','kernel')
        program=builder('float32')
        configs={}
    else:
        tl,builder,hashes=_load_builder(relative,'test_precise_division','kernel')
        program=builder(False)
        configs={tl.PassConfigKey.TL_ENABLE_FAST_MATH:False}
    kernel=tl.compile(program,target='ascend',pass_configs=configs)
    def run(a,b=None):
        shape=tuple(a.shape) if b is None else tuple(torch.broadcast_shapes(a.shape,b.shape))
        aa=a.expand(shape).contiguous().float().flatten()
        count=aa.numel();aa=F.pad(aa,(0,-count%64))
        if op=='AbsFwdOp':bb=torch.zeros_like(aa)
        elif op=='ReciprocalFwdOp':bb=aa;aa=torch.ones_like(bb)
        else:bb=F.pad(b.expand(shape).contiguous().float().flatten(),(0,-count%64),value=1.)
        out=torch.empty_like(aa)
        for start in range(0,aa.numel(),64):kernel(aa[start:start+64],bb[start:start+64],out[start:start+64])
        return out[:count].to(a.dtype).reshape(shape)
    return Candidate(Implementation(run),'tilelang::'+('test_absolute_difference.kernel' if op=='AbsFwdOp' else 'test_precise_division.kernel'),
                     'Original fixed64 arithmetic; exact FP32 widening and independent slicing. Abs selects zero second operand, reciprocal selects constant-one numerator. Original precise-div pass configuration retained.',hashes)
