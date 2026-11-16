"""Composition of the original fixed-slope Ascend LeakyReLU API."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import sys


def leaky(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes
    slope=case.params.get('negative_slope',0.01)
    if slope!=0.01 or case.params.get('inplace',False):
        raise UnsupportedCase('Original fixed0.1 API composition covers out-of-place slope0.01 only.')
    tl=_runtime()
    source=LIBRARIES/'tilelang/testing/ascend/language/simdvf/test_simdvf_arithmetic.py'
    tree=ast.parse(source.read_text())
    constant=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='N' for t in n.targets))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='relu_kernel')
    fn.decorator_list=[]
    header=ast.parse('import tilelang\nimport tilelang.ascend.language as T\nfrom tilelang.ascend.language import simd as S\n').body
    text=ast.unparse(ast.fix_missing_locations(ast.Module(body=header+[constant,fn],type_ignores=[])))+'\n'
    digest=hashlib.sha256(text.encode()).hexdigest()
    path=WORKSPACE/'.cache/all-pairs/tilelang/sources'/('leaky-'+digest+'.py')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    spec=importlib.util.spec_from_file_location('_allpairs_leaky_'+digest,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    kernel=tl.compile(module.relu_kernel(),target='ascend')
    assert module.N==64
    def run(x):
        n=x.numel();a=F.pad(x.contiguous().float().flatten(),(0,-n%64))
        out=torch.empty_like(a);temporary=torch.empty(64,device=x.device,dtype=torch.float32);unused=torch.empty_like(temporary)
        for start in range(0,a.numel(),64):
            kernel(a[start:start+64],temporary,unused)
            kernel(temporary,out[start:start+64],unused)
        return out[:n].reshape(x.shape).to(x.dtype)
    return Candidate(Implementation(run),'tilelang::relu_kernel fixed0.1 twice',
                     'Unchanged original N=64 kernel called twice: positive values stay unchanged, negatives receive the original0.1 slope twice for manifest0.01. Exact FP32 input widening and final dtype cast; full manifest tolerances remain unchanged. Unused fixed0.2 output is not substituted for parameterized PReLU.',
                     source_hashes(source,path,Path(__file__))|_runtime_hashes(tl))
