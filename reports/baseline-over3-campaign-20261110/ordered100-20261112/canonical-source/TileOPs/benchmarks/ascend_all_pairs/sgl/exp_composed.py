"""Complete original GDN gate specialization exposing native exponential."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
    from .additional import maximum, minimum
    from .extra_arithmetic import _affine
    x, = case.inputs
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original gate specialization accepts real floating inputs only')
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_gdn_gating.py'
    gating = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                           'sgl_kernel_npu.fla.fused_gdn_gating', 'fused_gdn_gating_kernel_without_sigmoid', host)
    affine, hashes = _affine(case)
    name = case._call.signature.name
    high, low = maximum(case), minimum(case)

    def exp(value):
        original_shape = value.shape
        flat = value.contiguous().reshape(-1)
        result = torch.empty(flat.shape, device=flat.device, dtype=torch.float32)
        # A_log is per head. Independent head slices avoid enormous launch grids;
        # no reduction is split and no vendor scheduling parameter is modified.
        for start in range(0, flat.numel(), 65536):
            source = flat[start:start+65536]
            ones = torch.ones((1, source.numel()), device=source.device, dtype=torch.float32)
            zeros = torch.zeros(source.numel(), device=source.device, dtype=torch.float32)
            # Public threshold=0 selects the exact x=1 linear Softplus branch.
            # The complete returned gate is therefore -exp(A_log), in FP32.
            negative, _ = gating(source, ones, ones, zeros, beta=1.0, threshold=0.0)
            result[start:start+source.numel()].copy_(affine(negative, torch.zeros_like(negative), -1.0).reshape(-1))
        return result.reshape(original_shape)

    def run(x):
        xx = x.float()
        if name == 'ExpFwdOp':
            result = exp(xx)
        elif name == 'Expm1FwdOp':
            result = affine(exp(xx), -torch.ones_like(xx), 1.0)
        else:
            zero = torch.zeros((), device=xx.device, dtype=xx.dtype)
            positive = high.implementation.run(xx, zero)
            nonpositive = low.implementation.run(xx, zero)
            tail = affine(exp(nonpositive), -torch.ones_like(xx), 1.0)
            alpha = case.params['alpha'] if name == 'EluFwdOp' else 1.6732632423543772848170429916717
            result = affine(tail, positive, alpha)
            if name == 'SeluFwdOp':
                result = affine(result, torch.zeros_like(result), 1.0507009873554804934193349852946)
        return result.to(x.dtype)

    return Candidate(Implementation(run), 'sgl_kernel_npu::fused_gdn_gating_kernel_without_sigmoid(threshold=0) + native mul_add/selectors',
        'Bind A_log to the original input and the complete public gate Softplus branch to exact one (a=1, bias=0, threshold=0). Native gate returns -exp(input); original native sign/affine and selectors implement the requested exponential family. Independent element slicing only, FP32 intermediates and one final public dtype conversion.',
        hashes | high.source_sha256 | low.source_sha256 | source_hashes(Path(__file__), LIBRARIES/'sgl-kernel-npu'/host),
        limitations=('Expm1 and negative activation tails use native exp minus one and may fail the original near-zero tolerance; such failures must remain failures without changing the comparator.',))
