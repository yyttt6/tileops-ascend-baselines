"""Original four-stage GDN backward specialized to ungated DeltaNet."""
from pathlib import Path


def deltanet_chunk_backward(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, LIBRARIES, source_hashes
    from benchmarks.ascend_extra_baselines import _owned_native
    from .providers import _native
    if case.inputs[1].dtype != torch.bfloat16:
        raise UnsupportedCase('Original FLA complete backward and prepare require BF16; an FP16-to-BF16 cast is lossy.')
    public, hashes = _native('aclnnChunkGatedDeltaRuleBwd')
    _, forward_hashes = _native('aclnnChunkFwdH')
    hashes.update(forward_hashes)
    root=LIBRARIES/'flash-linear-attention-npu'
    prepare=root/'fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare'
    hashes.update(source_hashes(Path(__file__),*[p for p in prepare.rglob('*') if p.suffix in ('.cpp','.h','.hpp')]))
    installed=Path(public.__file__).resolve().parents[2]
    symbols=('aclnnChunkGatedDeltaRuleFwdPrepare','aclnnChunkGdnBwdIntra',
             'aclnnChunkFwdH','aclnnChunkGatedDeltaRuleBwdDhu',
             'aclnnChunkGatedDeltaRuleBwdFinalize')
    _owned_native(installed/'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so',
                  [name+suffix for name in symbols for suffix in ('','GetWorkspaceSize')])
    chunk=case.params['chunk_size']

    def run(do,q,k,v,beta,*saved):
        key,value=q.shape[-1],v.shape[-1]
        q,k,v=(F.pad(x,(0,128-x.shape[-1])) for x in (q,k,v))
        gate=torch.zeros_like(beta)
        # This workload explicitly generates random saved buffers and requires
        # regenerating the forward intermediates. Use the original prepare API.
        prepared=public.npu_chunk_gated_delta_rule_fwd_prepare(q,k,v,gate,beta,chunk_size=chunk,use_exp2=False)
        do=F.pad(do,(0,128-value)).contiguous()
        # The original composite's internal transpose exposes stale storage
        # shape to BwdDhu tiling on this runtime (c002: d_o H=256 vs dv H=16).
        # Invoke its identical four original kernels through their public APIs,
        # where every intermediate has a concrete contiguous BNSD descriptor.
        w,u,dv_local=public.npu_chunk_gdn_bwd_intra(q,k,v,gate,beta,prepared[-1],
            do,1.0,chunk,use_exp2=False)
        h,v_new,_=public.npu_chunk_fwd_h(k,w,u,g=gate,chunk_size=chunk,
            output_final_state=False,save_new_value=True,use_exp2=False,state_v_first=False)
        dh,_,dv2=public.npu_chunk_gated_delta_rule_bwd_dhu(q,k,w,do,dv_local,
            1.0,chunk,g=gate,use_exp2=False,transpose_state_layout=False)
        result=public.npu_chunk_gated_delta_rule_bwd_finalize(q,k,v,v_new,do,dv2,
            gate,beta,h,dh,prepared[-1],scale=1.0,chunk_size=chunk,
            use_exp2=False,state_v_first=False)
        for name in ('npu_chunk_gated_delta_rule_fwd_prepare','npu_chunk_gdn_bwd_intra',
                     'npu_chunk_fwd_h','npu_chunk_gated_delta_rule_bwd_dhu',
                     'npu_chunk_gated_delta_rule_bwd_finalize'):
            if public.BACKENDS.get(name)!='ctypes':raise RuntimeError('Wrong original backward backend '+name)
        return (result[0][...,:key].contiguous(),result[1][...,:key].contiguous(),
                result[2][...,:value].contiguous(),result[3].contiguous())

    return Candidate(Implementation(run),'FLA prepare + original Intra/FwdH/BwdDhu/Finalize backward chain(g=0)',
        'Original composite kernel order, scale=1, g=0, BF16 states and original tiling/precision. Public constituent APIs materialize contiguous BNSD intermediate descriptors, avoiding the composite executor internal-transpose storage-shape failure. Regenerate forward buffers inside run as required by the original workload; zero feature padding/cropping only, complete dq/dk/dv/dbeta outputs, no torch data arithmetic or tolerance edits.',hashes)
