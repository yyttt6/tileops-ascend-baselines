"""Read-only paged/dense attention through the original Ascend forward kernel."""
from pathlib import Path
import hashlib
import importlib.util
import sys
from math import gcd


def _builder(dtype):
    from benchmarks.ascend_baselines import LIBRARIES, WORKSPACE, source_hashes, UnsupportedCase
    if dtype != 'bfloat16':
        raise UnsupportedCase('deployment-blocked: Original native attention fixes BF16 input and probability storage. FP16-to-BF16 is not lossless; changing internal probability dtype changes kernel mathematics. The original generic FP16 API requires non-VF scheduling.')
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes
    tl = _runtime()
    original = LIBRARIES / 'tilelang/examples/ascend/flash_attention/core.py'
    source = original.read_text()
    adapted = source  # Entire original kernel, including BF16 probability rounding.
    key = hashlib.sha256(adapted.encode()).hexdigest()
    directory = WORKSPACE / '.cache/all-pairs/tilelang/attention'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key+'.py')
    if path.exists():
        assert path.read_text() == adapted
    else:
        path.write_text(adapted)
    name = '_allpairs_tilelang_attention_'+key
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return tl, sys.modules[name].flash_attention_fwd, source_hashes(original,path,Path(__file__))|_runtime_hashes(tl)


def _attention(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import UnsupportedCase
    from .attention_ir import compile_program
    q,k,v,*_ = case.inputs
    p=case.params
    if q.dtype not in (torch.float16,torch.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise UnsupportedCase('Original FP16/BF16 path does not implement scaled FP8 attention.')
    if q.shape[-1] not in (64,128):
        raise UnsupportedCase('Original SIMD packing is fixed128; only exact feature padding up to128 is used.')
    if p.get('pos_encoding_mode','none') != 'none' or (p.get('softcap') or 0)>0:
        raise UnsupportedCase('Original kernel has no RoPE or score-softcap API.')
    tl,builder,hashes=_builder(str(q.dtype).split('.')[-1])
    dim=q.shape[-1]
    scale=p.get('sm_scale') if p.get('sm_scale') is not None else dim**-0.5
    kernels={}
    def run(q,k,v):
        sq,hq,_=q.shape
        sk,hk,_=k.shape
        if (p.get('is_causal',True) and sq!=1) or ((p.get('window_size_left',-1)>=0 or p.get('window_size_right',-1)>=0) and sq!=1):
            raise UnsupportedCase('Original Ascend attention is noncausal. Multi-query causal/window kernels need original generic CUDA route, currently deployment-blocked.')
        if p.get('window_size_left',-1)>=0:
            k,v=k[max(0,sk-1-p['window_size_left']):],v[max(0,sk-1-p['window_size_left']):]
            sk=k.shape[0]
        if sk==0:
            raise UnsupportedCase('Empty context requires an explicit zero-output API absent from this builder.')
        group=hq//hk
        rows=sq*group
        qpad=rows+-rows%128
        repetitions=128//gcd(sk,128)
        kp=sk*repetitions
        key=(qpad,kp)
        if key not in kernels:
            kernels[key]=compile_program(tl,builder,hashes,qpad,kp,128,out_dtype=str(q.dtype).split('.')[-1],softmax_scale=scale)
        kernel=kernels[key]
        out=torch.empty_like(q)
        for h in range(hk):
            qp=F.pad(q[:,h*group:(h+1)*group].reshape(rows,dim).contiguous(),(0,128-dim,0,qpad-rows))
            # Repeat all keys/values equally so padding adds no relative softmax mass.
            kk=F.pad(k[:,h].contiguous().repeat(repetitions,1),(0,128-dim))
            vv=F.pad(v[:,h].contiguous().repeat(repetitions,1),(0,128-dim))
            out[:,h*group:(h+1)*group].copy_(kernel(qp,kk,vv)[:rows,:dim].reshape(sq,group,dim))
        return out
    return run,hashes


def dense(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if any(t is not None for t in case.inputs[3:]):
        raise UnsupportedCase('Original kernel has no scaling, RoPE or sink tensor inputs.')
    native,hashes=_attention(case)
    def run(q,k,v,*extra):
        return torch.stack([native(q[b],k[b],v[b]) for b in range(q.shape[0])])
    return Candidate(Implementation(run),'tilelang::flash_attention_fwd','Unmodified original BF16 per-KV-head attention; feature/query padding, equal replication for unaligned KV lengths; decode window cropping. All copies in run. FP16 dtype rewriting is excluded because it changes probability rounding.',hashes)


def paged(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase
    if any(t is not None for t in case.inputs[6:]):
        raise UnsupportedCase('Original kernel has no scaling, RoPE or sink tensor inputs.')
    native,hashes=_attention(case)
    def run(q,k,v,table,lens,cuq,*extra):
        lengths=lens.cpu().tolist()
        bounds=cuq.cpu().tolist()
        outs=[]
        for b,length in enumerate(lengths):
            ids=table[b,:(length+k.shape[1]-1)//k.shape[1]].long()
            kk=k.index_select(0,ids).flatten(0,1)[:length]
            vv=v.index_select(0,ids).flatten(0,1)[:length]
            outs.append(native(q[bounds[b]:bounds[b+1]],kk,vv))
        return torch.cat(outs,dim=0)
    return Candidate(Implementation(run),'tilelang::flash_attention_fwd over gathered pages','Per-call page-table gather and logical cache crop; exact dtype/feature layout; original kernel for each request/head. Read current sequence metadata on every invocation.',hashes,timing_mode='host_roundtrip')
