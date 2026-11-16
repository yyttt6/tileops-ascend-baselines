"""Original predicated SIMD API; all function/macro/kernel bodies retained."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import sys


def _builder():
    from benchmarks.ascend_baselines import LIBRARIES,WORKSPACE,source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    tl=_runtime()
    source=LIBRARIES/'tilelang/testing/ascend/language/test_tilelang_ascend_simdvf_merging.py'
    tree=ast.parse(source.read_text())
    names={'BINARY_OPS','SCALAR_OPS','UNARY_OPS','SFU_OPS'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names) or (isinstance(n,ast.FunctionDef) and n.name=='merging_kernel')]
    header=ast.parse('import tilelang\nfrom tilelang import tvm\nimport tilelang.ascend.language as T\n').body
    text=ast.unparse(ast.fix_missing_locations(ast.Module(body=header+nodes,type_ignores=[])))+'\n'
    key=hashlib.sha256(text.encode()).hexdigest();path=WORKSPACE/'.cache/all-pairs/tilelang/sources'/('merging-'+key+'.py')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    spec=importlib.util.spec_from_file_location('_allpairs_merging_'+key,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return tl,module.merging_kernel,source_hashes(source,path,Path(__file__))|_runtime_hashes(tl)


def arithmetic(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    name=case._call.signature.name
    operation={'ExpFwdOp':'vexp','LogFwdOp':'vln','SqrtFwdOp':'vsqrt','RsqrtFwdOp':'vsqrt','BitwiseAndFwdOp':'vand','BitwiseOrFwdOp':'vor','MulFwdOp':'vmul','SubFwdOp':'vsub'}[name]
    if name=='SubFwdOp' and case.params.get('alpha',1)!=1:raise UnsupportedCase('Original direct subtraction accepts alpha=1 only.')
    integer=operation in ('vand','vor')
    tl,builder,hashes=_builder()
    kernel=tl.compile(builder(operation,'int32' if integer else 'float32',precision=None if integer else 'ftz_false'),target='ascend')
    reciprocal=tl.compile(builder('vdiv','float32',precision='exact'),target='ascend') if name=='RsqrtFwdOp' else None
    def run(x,y=None):
        shape=tuple(x.shape) if y is None else tuple(torch.broadcast_shapes(x.shape,y.shape))
        dt=x.dtype
        def prep(v):
            a=v.expand(shape).contiguous()
            if integer:a=a.view(torch.int32) if dt==torch.int64 else a.to(torch.int32)
            else:a=a.float()
            return a.flatten()
        a=prep(x);b=a if y is None else prep(y);count=a.numel()
        a=F.pad(a,(0,-count%64));b=F.pad(b,(0,-count%64))
        old=torch.zeros(64,device=x.device,dtype=a.dtype)
        mask=torch.ones(64,device=x.device,dtype=torch.int32).view(torch.uint32)
        out=torch.empty_like(a);ones=torch.ones_like(old) if reciprocal else None
        for start in range(0,a.numel(),64):
            view=out[start:start+64]
            kernel(a[start:start+64],b[start:start+64],old,mask,view)
            if reciprocal:reciprocal(ones,view,old,mask,view)
        result=out[:count]
        if integer and dt==torch.int64:result=result.view(torch.int64)
        return result.to(dt).reshape(shape)
    return Candidate(Implementation(run),'tilelang::merging_kernel('+operation+')',
                     'Unchanged original native SIMD API with all-true mask; independent fixed64 slices, exact widening or int64 limb views. Original ftz_false SFU precision; Rsqrt selects original exact division after sqrt. No Torch arithmetic.',hashes)


def selection(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    op=case._call.signature.name
    tl,builder,hashes=_builder()
    kernel=tl.compile(builder('vxor','int32'),target='ascend')
    def run(*args):
        if op=='WhereFwdOp':mask,a,b=args
        else:
            b,mask=args[:2]
            a=args[2] if len(args)>2 else torch.tensor(case.params['value'],device=b.device,dtype=b.dtype)
        shape=tuple(torch.broadcast_shapes(mask.shape,a.shape,b.shape));dt=a.dtype
        def words(v):
            t=v.expand(shape).contiguous()
            if t.element_size()==8:return t.view(torch.int32).flatten()
            if t.element_size()==4:return t.view(torch.int32).flatten()
            if t.element_size()==2:return t.view(torch.int16).to(torch.int32).flatten()
            return t.view(torch.int8).to(torch.int32).flatten()
        aa,bb=words(a),words(b)
        mm=mask.expand(shape).contiguous().to(torch.int32).flatten()
        if a.element_size()==8:mm=mm.repeat_interleave(2)
        count=aa.numel();aa=F.pad(aa,(0,-count%64));bb=F.pad(bb,(0,-count%64));mm=F.pad(mm,(0,-count%64)).view(torch.uint32)
        out=torch.empty_like(aa);zero=torch.zeros(64,device=a.device,dtype=torch.int32)
        for start in range(0,aa.numel(),64):kernel(aa[start:start+64],zero,bb[start:start+64],mm[start:start+64],out[start:start+64])
        result=out[:count]
        if a.element_size()==2:result=result.to(torch.int16)
        elif a.element_size()==1:result=result.to(torch.int8)
        return result.view(dt).reshape(shape)
    return Candidate(Implementation(run),'tilelang::merging_kernel(vxor,int32) with current mask',
                     'Original masked XOR with constant-zero operand is a bit-exact conditional copy. Current false operand binds Old and true operand binds A; exact integer bit views/widening, mask layout and broadcasting only. No Torch where/masked_fill/arithmetic.',hashes)
