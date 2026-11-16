"""Complete ungated DeltaNet chunk outputs using original FLA primitives."""
from pathlib import Path


def delta_chunk_forward(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .providers import _native,_triton_module
    from .native_affine import make_affine
    fp16=case.inputs[0].dtype==torch.float16
    if fp16:
        kkt,kkt_source=_triton_module('chunk_scaled_dot_kkt')
        solve,solve_source=_triton_module('solve_tril_fast')
        from triton.language.extra.cann.extension import vec_ops
        # Current Triton exposes the original slice builtins in its CANN
        # extension. Preserve the same operands/offsets/sizes/strides and IR.
        for name in ('insert_slice','extract_slice'):
            if not hasattr(solve.tl,name):setattr(solve.tl,name,getattr(vec_ops,name))
        native=None
        hashes=source_hashes(kkt_source,solve_source,Path(vec_ops.__file__))
    elif case.inputs[0].dtype==torch.bfloat16:
        native,hashes=_native('aclnnChunkGatedDeltaRuleFwdPrepare')
    else:
        raise UnsupportedCase('Original DeltaNet prepare adapters require FP16 or BF16.')
    scan,source=_triton_module('cumsum')
    affine,more=make_affine()
    chunk=int(case.params['chunk_size'])
    if chunk!=64:raise UnsupportedCase('Original complete prepare supports chunk_size=64.')

    def dot(left,right):
        # Batch dimensions match. Native tap products and additive prefixes
        # retain FP32 state operands without narrowing them to BF16 matrix APIs.
        shape=left.shape[:-2];rows,width=left.shape[-2:];columns=right.shape[-1]
        result=[]
        cap=max(1,(16*1024*1024)//max(1,left.numel()//rows*columns))
        for start in range(0,rows,cap):
            stop=min(rows,start+cap)
            a=left[...,start:stop,None,:].float().expand(*shape,stop-start,columns,width)
            b=right.transpose(-1,-2)[...,None,:,:].float()
            products=affine(a,b)
            extent=1<<(width-1).bit_length()
            packed=F.pad(products,(0,extent-width)).reshape(-1,extent,1)
            sums=scan.chunk_local_cumsum(packed,chunk_size=extent,output_dtype=torch.float32)[:,width-1,0]
            result.append(sums.reshape(*shape,stop-start,columns))
        return torch.cat(result,dim=-2)

    def run(q,k,v,beta):
        key,value=q.shape[-1],v.shape[-1]
        if fp16:
            # Original KKT host API consumes BHLD k and BLH beta/g; its
            # default FP32 result is strictly lower triangular. Keep the
            # original solver default FP32 precision and unchanged tiling.
            lower=kkt.chunk_scaled_dot_kkt_fwd(k,g=torch.zeros_like(beta).transpose(1,2).contiguous(),
                beta=beta.transpose(1,2).contiguous(),chunk_size=chunk)
            inverse=solve.solve_tril_npu(lower).transpose(1,2).contiguous()
            A=inverse.to(q.dtype)
            ws=[];us=[]
        else:
            padded=[F.pad(x,(0,128-x.shape[-1])) for x in (q,k,v)]
            prepared=native.npu_chunk_gated_delta_rule_fwd_prepare(*padded,torch.zeros_like(beta),beta,chunk_size=chunk,use_exp2=False)
            w,u,A=prepared[-3:]
            w=w[...,:key].contiguous();u=u[...,:value].contiguous()
        state=torch.zeros((*q.shape[:2],key,value),device=q.device,dtype=torch.float32)
        boundaries=[state];outputs=[]
        mask=torch.tensor([[1.0 if col<=row else 0.0 for col in range(chunk)] for row in range(chunk)],device=q.device,dtype=torch.float32)
        for start in range(0,q.shape[2],chunk):
            sl=slice(start,start+chunk)
            qc,kc=q[:,:,sl],k[:,:,sl]
            if fp16:
                # Preserve full original FP16 values. Native FP32 tap/scan
                # products apply the inverse to beta*k and beta*v; store the
                # two public WY buffers once in their manifest FP16 dtype.
                coeff=inverse[:,:,sl]
                bc=beta[:,:,sl,None].float()
                wc=dot(coeff,affine(kc.float(),bc)).to(q.dtype)
                uc=dot(coeff,affine(v[:,:,sl].float(),bc)).to(q.dtype)
                ws.append(wc);us.append(uc)
            else:
                wc,uc=w[:,:,sl],u[:,:,sl]
            residual=affine(dot(wc,state),-1.0,uc.float())
            causal=affine(dot(qc,kc.transpose(-1,-2)),mask)
            outputs.append(affine(dot(causal,residual),1.0,dot(qc,state)).to(q.dtype))
            state=affine(dot(kc.transpose(-1,-2),residual),1.0,state)
            boundaries.append(state)
        if fp16:
            w=torch.cat(ws,dim=2);u=torch.cat(us,dim=2)
        elif native.BACKENDS.get('npu_chunk_gated_delta_rule_fwd_prepare')!='ctypes':raise RuntimeError('Wrong original prepare backend')
        return torch.cat(outputs,dim=2),torch.stack(boundaries,dim=2),A,A.clone(),w,u

    return Candidate(Implementation(run),('FLA original FP16 KKT + solve_tril_npu + FP32 native tap/scan chunk recurrence' if fp16 else 'FLA original GDN prepare(g=0) + FP32 native tap/scan chunk recurrence'),
        'Return all six original DeltaNet chunk outputs: output, FP32 initial/intermediate/final boundaries, both inverse WY matrices, w and u. BF16 uses original native prepare for inverse/WY tensors. FP16 uses the original default-FP32 Triton KKT and solve_tril APIs and original FP32 tap/scan products for WY, storing all manifest buffers in their original dtypes; no FP16-to-BF16 cast. Original native tap products and additive prefixes evaluate FP32 state products without lossy state narrowing, with the manifest unscaled q contract and constant causal mask. All transforms, arithmetic and copies occur inside run.',
        hashes|more|source_hashes(Path(__file__),source))
