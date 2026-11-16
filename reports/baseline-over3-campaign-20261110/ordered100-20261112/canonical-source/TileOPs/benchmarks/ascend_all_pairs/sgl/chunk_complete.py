"""All six declared DeltaNet chunk outputs from unchanged SGL forward stages."""
from pathlib import Path


def build(case):
    import importlib
    import inspect
    import os
    import sys
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
    if case.params['chunk_size'] != 64:
        raise UnsupportedCase('Original SGL complete forward pipeline fixes its chunk size at 64')
    if os.environ.get('GDN_USE_MEGA_GDN','0') != '0':
        raise UnsupportedCase('Require inspected original Triton forward pipeline')
    package = LIBRARIES / 'sgl-kernel-npu/python/sgl_kernel_npu'
    sys.path.insert(0,str(package))
    module = importlib.import_module('sgl_kernel_npu.fla.chunk')
    forward = module.chunk_gated_delta_rule_fwd
    recompute = module.recompute_w_u_fwd
    if Path(inspect.getsourcefile(forward)).resolve() != package/'sgl_kernel_npu/fla/chunk.py':
        raise RuntimeError('Unexpected SGL chunk forward source')
    def run(q,k,v,beta):
        B,H,T,K = q.shape
        V = v.shape[-1]
        qp,kp,vp = [x.transpose(1,2).contiguous().reshape(1,B*T,H,x.shape[-1]) for x in (q,k,v)]
        bp = beta.transpose(1,2).contiguous().reshape(1,B*T,H)
        cu = torch.arange(B+1,device=q.device,dtype=torch.int64)*T
        initial = torch.zeros((B,H,K,V),device=q.device,dtype=torch.float32)
        gate = torch.zeros((1,B*T,H),device=q.device,dtype=torch.float32)
        cumulative,o,A,final,_w,h,_v_new = forward(qp,kp,vp,gate,bp,1.0,initial,True,cu)
        # The public forward returns residual v_new, whereas the manifest asks
        # for uncorrected WY u. Reuse the original complete recompute API.
        w,u = recompute(k=kp,v=vp,beta=bp,A=A,g_cumsum=cumulative,cu_seqlens=cu)
        states = h.reshape(B,T//64,H,K,V).permute(0,2,1,3,4).float()
        states = torch.cat((states,final[:,:,None]),dim=2).contiguous()
        restore = lambda x: x.reshape(B,T,H,x.shape[-1]).transpose(1,2).contiguous()
        inverse = restore(A)
        return restore(o),states,inverse,inverse.clone(),restore(w),restore(u)
    return Candidate(Implementation(run),
        'sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_fwd(g=0,scale=1) + original recompute_w_u_fwd',
        'Pack dense BHTD to BTHD with explicit original sequence offsets. Return all six original declared tensors: output, pre-chunk states plus final state, both equal ungated inverse factors, uncorrected w and u. Widen stored h to required FP32 without changing native arithmetic. No reference values or external recurrence used by candidate.',
        source_hashes(Path(__file__),*sorted((package/'sgl_kernel_npu/fla').glob('*.py'))),
        limitations=('Original stored chunk states use input dtype before required FP32 widening; all six outputs and local strict-lower errors must pass unchanged manifest comparator.',))
