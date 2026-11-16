"""Finite manifest comparisons from original first-tie native argmax."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
    from .extra_arithmetic import _affine
    if any(x.dtype not in (torch.float16,torch.bfloat16,torch.float32) for x in case.inputs):
        raise UnsupportedCase('The source-reviewed finite comparison route accepts FP16/BF16/FP32 operands only.')
    host='python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    select=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',host)
    affine,hashes=_affine(case)
    name=case._call.signature.name
    def run(x,y):
        shape=torch.broadcast_shapes(x.shape,y.shape)
        xx,yy=x.expand(shape),y.expand(shape)
        def less(a,b):
            # Ties retain column zero. Hence the native index is exactly a<b.
            return select(torch.stack((a,b),dim=-1).reshape(-1,2))[0].float().reshape(shape)
        if name in ('LtFwdOp','GeFwdOp'):
            value=less(xx,yy)
        elif name in ('GtFwdOp','LeFwdOp'):
            value=less(yy,xx)
        else:
            value=affine(less(xx,yy),less(yy,xx),1.0)
        if name in ('GeFwdOp','LeFwdOp','EqFwdOp'):
            value=affine(value,torch.ones_like(value),-1.0)
        return value.to(torch.bool)
    return Candidate(Implementation(run),'sgl_kernel_npu::argmax_softmax_prob_fused first-tie selector + original mul_add',
        'Broadcast each original finite manifest pair into a complete two-column native argmax. The exact 0/1 indices encode strict ordering; original native sum/complement supplies equality and inclusive ordering, followed by declared boolean storage conversion. No external tensor comparison computes the answer.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host)|hashes,
        limitations=('Scoped to the original normal-distributed finite manifest inputs. Native argmax NaN ordering is not asserted to implement general IEEE comparison semantics.',))
