"""Per-tensor FP8 scales mapped to the original dense GEMM alpha parameter."""
from pathlib import Path
import importlib.util
import sys


def _api():
    from benchmarks.ascend_baselines import LIBRARIES,source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    tl=_runtime()
    root=LIBRARIES/'tilelang/examples/ascend/deepgemm'
    name='_allpairs_tilelang_deepgemm'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name], source_hashes(*root.rglob('*.py'),root/'_heuristics.pyx',Path(__file__))|_runtime_hashes(tl)


def fp8_scaled(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    a,b,sa,sb,*extra=case.inputs
    if sa.numel()!=1 or sb.numel()!=1:
        raise UnsupportedCase('Native alpha is scalar; arbitrary per-block FP32 scales cannot be rounded to native UE8M0. Generic blockscaled route remains deployment-blocked.')
    module,hashes=_api()
    odtype=getattr(torch,case.params.get('out_dtype','bfloat16'))
    batched=a.ndim==3
    trans_b=case.params.get('trans_b',False) if batched else True
    def one(a,b,alpha,bias=None):
        lhs=a.to(torch.bfloat16).contiguous()
        rhs=(b if trans_b else b.T).to(torch.bfloat16).contiguous()
        out=torch.empty((lhs.shape[0],rhs.shape[0]),device=a.device,dtype=torch.float32)
        initial=None if bias is None else bias.float().reshape(1,-1).expand_as(out).contiguous()
        module.bf16_gemm_nt(lhs,rhs,out,c=initial,alpha=alpha)
        return out.to(odtype)
    def run(a,b,sa,sb,*extra):
        # Host scalar API binding; native GEMM applies alpha before output conversion.
        alpha=float(sa.item())*float(sb.item())
        if batched:
            return torch.stack([one(a[i],b[i],alpha) for i in range(a.shape[0])])
        return one(a,b,alpha,extra[0] if extra else None)
    return Candidate(Implementation(run),'tilelang::deepgemm.bf16_gemm_nt(alpha,c)',
                     'FP8 values widen exactly to BF16. Current per-tensor scales map to original runtime alpha; optional bias is original FP32 accumulation input. Native FP32 output then contract cast. Host scalar reads, batch calls and all casts counted.',hashes,timing_mode='host_roundtrip')
