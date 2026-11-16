from pathlib import Path

def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.ascend_all_pairs.fla.providers import _native,_triton_module
    op=case._call.signature.name
    if op=='CumsumFwdOp':
        native,source=_triton_module('cumsum')
        def run(x):
            assert tuple(x.shape)==(2048,4096)
            return native.chunk_local_cumsum(x.reshape(2048,4096,1),chunk_size=4096,output_dtype=x.dtype).reshape_as(x)
        return Candidate(Implementation(run),'chunk_local_cumsum(B2048,T4096,H1,chunk_size4096)','Exactly one original API call; one full-row chunk; no carry, totals or offset addition.',source_hashes(Path(__file__),source))
    if op=='GemmFwdOp':
        from benchmarks.ascend_all_pairs.fla.cube_matmul import gemm
        return gemm(case)
    if op=='GroupedGemmFwdOp':
        from benchmarks.ascend_all_pairs.fla.cube_matmul import make_matmul
        matmul,hashes=make_matmul()
        p=case.params
        def run(a,b,sizes,offsets):
            assert not p['transpose_a']
            columns=b.shape[1] if p['transpose_b'] else b.shape[2]
            result=torch.empty((a.shape[0],columns),device=a.device,dtype=a.dtype)
            for group,(start,size) in enumerate(zip(offsets.cpu().tolist(),sizes.cpu().tolist(),strict=True)):
                if size:
                    right=b[group].t().contiguous() if p['transpose_b'] else b[group]
                    result[start:start+size]=matmul(a[start:start+size],right)
            return result
        return Candidate(Implementation(run),'npu_chunk_fwd_h(W=0,g=0,full final state)','Independent groups, copy-only packing, full contraction in each native call; no unrelated Triton import.',hashes|source_hashes(Path(__file__)))
    if op=='KDAFwdOp':
        from benchmarks.ascend_all_pairs.fla.providers import chunk_kda
        return chunk_kda(case)
    if op=='AddFwdOp':
        native,source=_triton_module('causal_conv1d')
        def run(x,y):
            data=x.unsqueeze(0).contiguous()
            residual=y.expand_as(x).unsqueeze(0).contiguous()
            out,_=native.causal_conv1d_fwd_impl_old(data,torch.ones((1,x.shape[-1]),device=x.device,dtype=x.dtype),bias=None,residual=residual)
            return out[0]
        return Candidate(Implementation(run),'causal_conv1d_fwd_impl_old(W=1)','Standard BTD layout, unit tap, broadcast residual.',source_hashes(Path(__file__),source))
    native,hashes=_native('aclnnChunkGatedDeltaRuleFwd')
    p=case.params
    def run(q,k,v,*args):
        if op=='DeltaNetFwdOp':
            beta,initial,offsets,offsets_cpu=args
            g=torch.zeros((*q.shape[:2],v.shape[2]),device=q.device,dtype=torch.float32)
        elif op=='GLAFwdOp':
            ignored,initial,offsets,offsets_cpu=args
            assert initial is None and q.shape[1]==1
            g=torch.zeros((*q.shape[:2],v.shape[2]),device=q.device,dtype=torch.float32)
            beta=torch.ones_like(g)
        else:
            g,beta,*tail=args
            initial,offsets,offsets_cpu,a_log,dt_bias=(tail+[None]*5)[:5]
        metadata=offsets_cpu if offsets_cpu is not None else offsets
        bounds=None if metadata is None else metadata.cpu().tolist()
        if bounds is not None:
            assert len(set(b-a for a,b in zip(bounds,bounds[1:])))==1
            def pack(t): return t.reshape(len(bounds)-1,bounds[1]-bounds[0],*t.shape[2:]).contiguous()
            q,k,v,g,beta=map(pack,(q,k,v,g,beta))
        q,k,v=[t.transpose(1,2).contiguous() for t in (q,k,v)]
        result=native.npu_chunk_gated_delta_rule_fwd(q,k,v,g.float(),beta.float(),initial_state=None if initial is None else initial.clone(),output_final_state=True,scale=p.get('scale') or q.shape[-1]**-.5,layout='BNSD',state_v_first=p.get('state_v_first',False),use_qk_l2norm_in_kernel=p.get('use_qk_l2norm_in_kernel',False),use_beta_sigmoid_in_kernel=p.get('use_beta_sigmoid_in_kernel',False),allow_neg_eigval=p.get('allow_neg_eigval',False),use_gate_in_kernel=p.get('use_gate_in_kernel',False))
        out,final=result[:2]
        if bounds is not None: out=out.reshape(1,-1,*out.shape[2:])
        return out,final
    return Candidate(Implementation(run),'npu_chunk_gated_delta_rule_fwd','Original full GDN; BNSD, lossless FP32 gate/beta, full state, fixed gates for Delta/first-token GLA.',hashes|source_hashes(Path(__file__)))
