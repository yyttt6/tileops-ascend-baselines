"""SGL-owned GEMV and SwiGLU for supplied, already materialized expert groups."""
from pathlib import Path


def grouped(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes,_owned_native
    p=case.params
    activation=p.get('activation')
    if activation not in (None,'silu_and_mul'):
        raise UnsupportedCase('SGL has SwiGLU but no GELU epilogue for this expert contract.')
    x,w,metadata=case.inputs
    if x.dtype not in (torch.float16,torch.bfloat16) or w.dtype!=x.dtype:
        raise UnsupportedCase('Original SGL vector LoRA accepts matching FP16/BF16 operands only.')
    # Importing this original Python module also loads the package native library.
    host='python/sgl_kernel_npu/sgl_kernel_npu/activation/swiglu_quant.py'
    swiglu=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.activation.swiglu_quant','swiglu_quant',host)
    root=LIBRARIES/'sgl-kernel-npu'
    binary=root/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so'
    _owned_native(binary,['_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l'])
    layout=p['layout']
    if isinstance(layout,dict):
        from types import SimpleNamespace
        kind,fields=next(iter(layout.items()))
        layout=SimpleNamespace(kind=kind,**fields)
    out_dtype=getattr(torch,p['out_dtype']) if p.get('out_dtype') else x.dtype
    def product(a,b):
        m,k=a.shape
        n=b.shape[0]
        pieces=[]
        indices=torch.zeros(1,dtype=torch.int32,device=a.device)
        lengths=torch.tensor([m],dtype=torch.int32,device=a.device)
        scales=torch.ones(1,dtype=torch.float16,device=a.device)
        for start in range(0,n,1024):
            end=min(start+1024,n)
            width=(end-start+15)//16*16
            padded_k=max((k+15)//16*16,(width//16+1)*16)
            aa=F.pad(a.contiguous(),(0,padded_k-k))
            bb=F.pad(b[start:end].contiguous(),(0,padded_k-k,0,width-(end-start))).unsqueeze(0)
            ranks=torch.tensor([width],dtype=torch.int32,device=a.device)
            out=torch.empty((m,width),device=a.device,dtype=torch.float32)
            torch.ops.npu.sgemmv_shrink(aa,bb,indices,lengths,ranks,scales,out,1)
            pieces.append(out[:,:end-start])
        return torch.cat(pieces,dim=1)
    def activate(out):
        gate,up=out.chunk(2,dim=-1)
        size=gate.numel();tail=-size%1024
        g=F.pad(gate.contiguous().reshape(-1),(0,tail)).reshape(-1,1024)
        u=F.pad(up.contiguous().reshape(-1),(0,tail)).reshape(-1,1024)
        counts=torch.tensor([g.shape[0]],device=out.device,dtype=torch.int32)
        y,_=swiglu(torch.cat((g,u),dim=-1),counts,1,need_quant=False)
        return y.reshape(-1)[:size].reshape(gate.shape)
    def run(x,w,metadata):
        n=w.shape[1]//(2 if activation else 1)
        out=torch.zeros((*x.shape[:-1],n),device=x.device,dtype=out_dtype)
        meta=metadata.cpu().tolist()
        begin=0
        for g in range(w.shape[0]):
            if layout.kind=='masked':
                count=meta[g]
                if not count:continue
                index=(g,slice(0,count));a=x[index]
            elif layout.metadata_kind=='per_row':
                index=(metadata==g).nonzero().flatten()
                if not index.numel():continue
                a=x[index]
            else:
                begin=(begin+layout.alignment-1)//layout.alignment*layout.alignment
                end=meta[g];index=slice(begin,end);a=x[index];begin=end
                if not a.shape[0]:continue
            value=product(a,w[g])
            if activation:value=activate(value)
            out[index]=value.to(out_dtype)
        return out
    files=[f for f in (root/'csrc/lora').rglob('*') if f.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink + optional native swiglu_quant(need_quant=False)',
                     'Read supplied expert metadata on every call; gather/view independent groups, original full-K vector GEMV tiles over independent columns; native FP32 output and optional SGL SwiGLU before final contract cast; zero undefined rows.',
                     source_hashes(Path(__file__),root/host,binary,root/'csrc/pytorch_extensions.cpp',*files),timing_mode='host_roundtrip')


def expert_mlp(case):
    from types import SimpleNamespace
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    x,up,down,metadata=case.inputs
    first=grouped(SimpleNamespace(inputs=(x,up,metadata),params=case.params))
    # The second factory only reads dtype/layout; no input-dependent result is built.
    second=grouped(SimpleNamespace(inputs=(x,down,metadata),params=case.params|{'activation':None}))
    def run(x,up,down,metadata):
        hidden=first.implementation.run(x,up,metadata)
        return second.implementation.run(hidden,down,metadata)
    return replace(first,implementation=Implementation(run),api=first.api+' then '+second.api,
                   adaptation='Original SGL grouped up projection and SwiGLU, contract intermediate storage cast, then original SGL down projection; supplied expert layout retained and all native calls included.',
                   source_sha256=first.source_sha256|second.source_sha256|source_hashes(Path(__file__)))
