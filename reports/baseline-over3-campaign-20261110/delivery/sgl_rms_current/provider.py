from math import prod
from pathlib import Path
def _rms_layout(case):
    import torch
    x, weight = case.inputs
    hidden = prod(case.params['normalized_shape'])
    shape = tuple(x.shape)
    eps = case.params.get('eps')
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    unit_weight = torch.ones(hidden, device=x.device, dtype=x.dtype) if weight is None else None

    def pack(x, weight):
        gamma = unit_weight if weight is None else weight.contiguous().view(hidden)
        return (x.contiguous().view(-1, hidden), gamma)
    return (x, shape, hidden, eps, pack)

def sgl_rms(case):
    import torch
    from benchmarks.api import Implementation
    x, shape, hidden, eps, pack = _rms_layout(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_bias.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.norm.rmsnorm_bias', 'rmsnorm_bias', host)
    bias = torch.zeros(hidden, dtype=x.dtype, device=x.device)

    def run(x, weight):
        x2, gamma = pack(x, weight)
        return fn(x2, gamma, bias, eps).view(shape)
    return Candidate(Implementation(run), 'sgl_kernel_npu.norm.rmsnorm_bias.rmsnorm_bias', '2D view and zero bias; quantization disabled through original optional arguments.', source_hashes(LIBRARIES / 'sgl-kernel-npu' / host, LIBRARIES / 'sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py'))