"""Original KdaGateCumsum activation extraction; native deployment required."""
from math import log
from pathlib import Path


def gate_activation(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    from .providers import _native
    from .native_affine import make_affine
    name=case._call.signature.name
    if name=='SoftplusFwdOp' and (case.params.get('beta',1.0)!=1.0 or case.params.get('threshold',20.0)!=20.0):
        raise UnsupportedCase('This unchanged native gate implements beta=1; only original default threshold20 has been matched.')
    native,hashes=_native('aclnnKdaGateCumsum')
    affine,affine_hashes=make_affine()

    def run(x):
        flat=x.contiguous().reshape(-1).float();pieces=[]
        a_log=torch.zeros((1,),device=x.device,dtype=torch.float32)
        for start in range(0,flat.numel(),1048576):
            value=flat[start:start+1048576];count=value.numel()
            if name=='TanhFwdOp':value=affine(value,2.0)
            data=F.pad(value,(0,-count%128)).reshape(-1,1,1,128)
            gate=native.npu_kda_gate_cumsum(data,64,A_log=a_log,
                use_gate_in_kernel=True,safe_gate=name!='SoftplusFwdOp',lower_bound=-1.0)
            # This original gate API returns base-two log gates. At T=1 its
            # prefix is the element itself; -ln(2) restores the activation.
            value=affine(gate.reshape(-1)[:count],-log(2.0))
            if name=='TanhFwdOp':
                negative_one=torch.full((),-1.0,device=x.device,dtype=torch.float32)
                value=affine(value,2.0,negative_one)
            pieces.append(value.to(x.dtype))
        if native.BACKENDS.get('npu_kda_gate_cumsum')!='ctypes':raise RuntimeError('Wrong owned gate backend')
        return torch.cat(pieces).reshape(x.shape)

    return Candidate(Implementation(run),'FLA npu_kda_gate_cumsum(T=1, raw/safe gate) + original native scale',
        'Original safe gate with A_log=0/lower_bound=-1 exposes sigmoid; original raw gate exposes softplus. Single-token batches remove prefix interaction; original native constant tap converts the documented base-two gate scale. Tanh uses original native 2*sigmoid(2*x)-1. All allocation/padding/calls inside run, original kernels unchanged.',
        hashes|affine_hashes|source_hashes(Path(__file__)))
