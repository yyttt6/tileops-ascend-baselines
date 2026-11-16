"""SGL's complete delta/KDA recurrence APIs, never a Torch recurrence fallback."""
from pathlib import Path


def _native(case, channel_gate):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes
    q,k,v,g,beta,*_=case.inputs
    p=case.params
    if q.dtype != torch.bfloat16:
        if not channel_gate:
            return _triton_gdn(case)
        raise UnsupportedCase('Native chunk KDA accepts only BF16 Q/K/V; FP16 narrowing to BF16 is not lossless.')
    if q.shape!=k.shape or any(d%16 or d<16 or d>256 for d in (q.shape[-1],v.shape[-1])):
        raise UnsupportedCase('Original native KDA requires matching Q/K with K,V in 16..256 and divisible by16.')
    host='python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_kda_fwd.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.chunk_kda_fwd','chunk_kda_fwd',host)
    normhost='python/sgl_kernel_npu/sgl_kernel_npu/fla/l2norm.py'
    l2=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.l2norm','l2norm_fwd',normhost) if p.get('use_qk_l2norm_in_kernel') else None
    sighost='python/sgl_kernel_npu/sgl_kernel_npu/activation/fused_sigmoid_mul.py'
    mulhost='python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    sigmoid=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.activation.fused_sigmoid_mul','fused_sigmoid_mul',sighost) if p.get('use_beta_sigmoid_in_kernel') else None
    multiply=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.moe.mul_add','mul_add',mulhost) if p.get('allow_neg_eigval') else None
    def run(q,k,v,g,beta,initial,cu,cu_cpu,a_log,dt_bias):
        if sigmoid is not None:
            beta=sigmoid(torch.ones_like(beta,dtype=torch.float32),beta.float())
            if multiply is not None:
                shape=beta.shape
                beta=multiply(beta.reshape(-1,1),torch.zeros_like(beta).reshape(-1,1),2.0).reshape(shape)
        if l2 is not None:
            q,k=l2(q.contiguous(),eps=1e-6),l2(k.contiguous(),eps=1e-6)
        gate=g.contiguous() if channel_gate else g[...,None].expand(*g.shape,q.shape[-1]).contiguous()
        dt=dt_bias
        if not channel_gate and dt is not None:
            dt=dt.reshape(-1,1).expand(-1,q.shape[-1]).contiguous().reshape(-1)
        boundaries=cu if cu is not None else (None if cu_cpu is None else cu_cpu.to(q.device))
        if boundaries is not None:boundaries=boundaries.to(device=q.device,dtype=torch.int64).contiguous()
        out=fn(q.contiguous(),k.contiguous(),v.contiguous(),gate,beta.contiguous(),
               a_log=a_log,dt_bias=dt,initial_state=None if initial is None else initial.clone().contiguous(),
               cu_seqlens=boundaries,layout='BSND',scale=q.shape[-1]**-0.5 if p.get('scale') is None else p['scale'],
               safe_gate=p.get('lower_bound') is not None,lower_bound=-5.0 if p.get('lower_bound') is None else p['lower_bound'],
               use_gate_in_kernel=p.get('use_gate_in_kernel',False),state_v_first=p.get('state_v_first',False),output_final_state=True)
        return out[0],out[1]
    root=LIBRARIES/'sgl-kernel-npu'
    files=[f for f in (root/'csrc/chunk_kda_fwd').rglob('*') if f.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.chunk_kda_fwd.chunk_kda_fwd' + (' + original SGL l2norm_fwd(eps=1e-6)' if l2 is not None else ''),
                     'Original complete KDA pipeline; scalar GDN gate broadcasts across key channels, optional native gate activation, original SGL sigmoid/constant-double beta and L2 APIs; preserve sequence boundaries, cloned initial state, scale and both outputs.',
                     source_hashes(Path(__file__),root/host,root/normhost,root/sighost,root/mulhost,root/'csrc/pytorch_extensions.cpp',root/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so',*files))


def gdn(case):
    return _native(case,False)


def kda(case):
    return _native(case,True)


def _triton_gdn(case):
    import importlib
    import inspect
    import sys
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,source_hashes
    p=case.params
    import os
    if os.environ.get('GDN_USE_MEGA_GDN','0')!='0':
        raise UnsupportedCase('Require the inspected original default full Triton GDN pipeline')
    if p.get('use_gate_in_kernel') or p.get('use_beta_sigmoid_in_kernel') or p.get('allow_neg_eigval'):
        raise UnsupportedCase('Original full Triton GDN API takes activated log gates/beta and has no raw-gate/beta activation flags.')
    root=LIBRARIES/'sgl-kernel-npu/python/sgl_kernel_npu'
    sys.path.insert(0,str(root))
    module=importlib.import_module('sgl_kernel_npu.fla.chunk')
    fn=module.chunk_gated_delta_rule_npu
    if Path(inspect.unwrap(fn).__code__.co_filename).resolve() != root/'sgl_kernel_npu/fla/chunk.py':
        raise RuntimeError('Original SGL full GDN ownership mismatch')
    def run(q,k,v,g,beta,initial,cu,cu_cpu,a_log,dt_bias):
        shape=v.shape
        if cu is None and cu_cpu is not None:cu=cu_cpu.to(q.device)
        if cu is None:
            b,t=q.shape[:2]
            cu=torch.arange(b+1,device=q.device,dtype=torch.int64)*t
            q,k,v=[x.contiguous().reshape(1,b*t,*x.shape[2:]) for x in (q,k,v)]
            g,beta=[x.contiguous().reshape(1,b*t,x.shape[-1]) for x in (g,beta)]
        state=initial
        if state is None:
            state=torch.zeros((cu.numel()-1,v.shape[2],q.shape[3],v.shape[3]),device=q.device,dtype=torch.float32)
        elif p.get('state_v_first'):
            state=state.transpose(-1,-2).contiguous()
        out,final,_=fn(q.contiguous(),k.contiguous(),v.contiguous(),g.contiguous(),beta.contiguous(),scale=p.get('scale'),initial_state=state,
                       output_final_state=True,cu_seqlens=cu,head_first=False,use_qk_l2norm_in_kernel=p.get('use_qk_l2norm_in_kernel',False))
        return out.reshape(shape),final.transpose(-1,-2).contiguous() if p.get('state_v_first') else final
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.chunk.chunk_gated_delta_rule_npu',
                     'Complete original Triton GDN pipeline; flatten dense sequences with explicit offsets, transpose state layout when requested; preserve activated gates/beta.',
                     source_hashes(Path(__file__),*sorted((root/'sgl_kernel_npu/fla').glob('*.py'))))


def delta_recurrent(case):
    from types import SimpleNamespace
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_sgl_delta_cases import sgl_delta_cases
    q,k,v,beta,state=case.inputs
    inputs=(q[:,None],k[:,None],v[:,None],beta[:,None],state.float(),None,None)
    native=sgl_delta_cases(SimpleNamespace(inputs=inputs,params={'scale':1.0,'use_qk_l2norm_in_kernel':False}))
    def run(q,k,v,beta,state):
        out,final=native.implementation.run(q[:,None],k[:,None],v[:,None],beta[:,None],state.float(),None,None)
        return out[:,0],final.to(state.dtype)
    from benchmarks.ascend_baselines import source_hashes
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|source_hashes(Path(__file__)),
                   adaptation='Single-token sequence with unit query scale; widen input state losslessly to FP32, restore manifest output state dtype; original SGL recurrence.')


def delta_chunk(case):
    from types import SimpleNamespace
    from dataclasses import replace
    from benchmarks.api import Implementation
    from benchmarks.ascend_sgl_delta_cases import sgl_delta_cases
    q,k,v,beta=case.inputs
    # Chunk operator is BHTD; inference API is BTHD and includes final state.
    inputs=tuple(x.transpose(1,2).contiguous() for x in (q,k,v,beta))+(None,None,None)
    native=sgl_delta_cases(SimpleNamespace(inputs=inputs,params={'scale':1.0,'use_qk_l2norm_in_kernel':False}))
    def run(q,k,v,beta):
        out,_=native.implementation.run(*(x.transpose(1,2).contiguous() for x in (q,k,v,beta)),None,None,None)
        return out.transpose(1,2).contiguous()
    from benchmarks.ascend_baselines import source_hashes
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|source_hashes(Path(__file__)),
                   adaptation='BHTD to BTHD; unit query scale and zero initial state; native complete recurrence output; discard final state not declared by chunk manifest.')
