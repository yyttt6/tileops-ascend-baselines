"""Select native reduction outputs, preserving complete reduction dimensions."""
from pathlib import Path
import math


def row_layout(case):
    x = case.inputs[0]
    dim = case.params.get('dim', -1)
    axes = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    axes = tuple(d % x.ndim for d in axes)
    kept = tuple(d for d in range(x.ndim) if d not in axes)
    width = math.prod(x.shape[d] for d in axes)
    shape = tuple(1 if d in axes else n for d,n in enumerate(x.shape)) if case.params.get('keepdim',False) else tuple(x.shape[d] for d in kept)
    def pack(x):
        return x.permute(*(kept+axes)).contiguous().reshape(-1,width)
    return pack,width,shape


def argmax(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    pack,width,shape = row_layout(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',host)
    def run(x):
        return fn(pack(x))[0].reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.sample.argmax_softmax_prob.argmax_softmax_prob_fused()[0]',
                     'Permute reduction axis last, preserve row element order (first-index ties), select native int64 argmax; complete softmax work remains timed.',
                     source_hashes(Path(__file__), LIBRARIES/'sgl-kernel-npu'/host))


def mean(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    pack,width,shape = row_layout(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    ones = torch.ones(width, device=case.inputs[0].device, dtype=case.inputs[0].dtype)
    dtype = getattr(torch,case.params.get('dtype') or str(case.inputs[0].dtype).split('.')[-1])
    def run(x):
        return fn(pack(x),ones,None,1e-5)[1].to(dtype).reshape(shape)
    return Candidate(Implementation(run),'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd()[1]',
                     'Permute and flatten complete reduction axes, select original FP32 mean and cast to declared output dtype; all native norm work counted.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
