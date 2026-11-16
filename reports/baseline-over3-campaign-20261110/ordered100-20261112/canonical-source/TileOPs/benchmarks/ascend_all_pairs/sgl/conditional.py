"""Select payload bits using original SGL argmax over boolean control rows."""
from pathlib import Path


def select(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,local_function,source_hashes,LIBRARIES
    host='python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',host)
    name=case._call.signature.name
    def run(*inputs):
        if name=='WhereFwdOp':
            mask,yes,no=inputs
        else:
            no,mask,*value=inputs
            yes=value[0] if value else torch.tensor(case.params['value'],device=no.device,dtype=no.dtype)
        shape=torch.broadcast_shapes(mask.shape,yes.shape,no.shape)
        control=mask.expand(shape).to(torch.float32).reshape(-1)
        pairs=torch.stack((torch.zeros_like(control),control),dim=-1)
        chosen=fn(pairs)[0]
        payload=torch.stack((no.expand(shape),yes.expand(shape)),dim=-1).reshape(-1,2)
        return payload.gather(1,chosen[:,None]).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.sample.argmax_softmax_prob.argmax_softmax_prob_fused',
        'Losslessly cast boolean controls to0/1; original first-tie argmax of[0,mask] selects index0/1. Gather unchanged original payload bits, preserving signed zero/nonfinite payloads. Broadcast/layout/constants only outside original selector; no Torch where/masked_fill arithmetic.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
