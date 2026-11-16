"""Use SGL's existing chunked TP RMS implementation with one local rank."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, _rms_layout, local_function, source_hashes


def sgl_large_rms_cases(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import sgl_rms

    x, shape, hidden, eps, pack = _rms_layout(case)
    if hidden < 32768:
        return sgl_rms(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/norm/split_qkv_tp_rmsnorm_rope.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                        'sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope',
                        'split_qkv_tp_rmsnorm_rope', host)
    # Native public head_dim selects its existing chunk size; Q/K widths stay
    # equal to the complete manifest normalization axis. No partial sums outside.
    head_dim = 16384 if hidden >= 262144 else 4096
    if hidden % head_dim:
        raise ValueError('This alternate entry is prepared for manifest power-of-two large widths.')
    def run(x, weight):
        value, gamma = pack(x, weight)
        out = torch.empty_like(value)
        for begin in range(0, value.shape[0], 128):
            part = value[begin:begin + 128].contiguous()
            zeros = torch.zeros_like(part)
            packed = torch.cat((part, zeros, zeros), dim=-1)
            # Original RoPE is identity with cos=1/sin=0. The native two-stage
            # global RMS reduction is reused as-is; tp_world=1 skips all-reduce.
            cos = torch.ones((part.shape[0], 2), device=x.device, dtype=x.dtype)
            sin = torch.zeros_like(cos)
            q, _, _ = fn(packed, cos, sin, hidden, hidden, head_dim, eps,
                         gamma, gamma, 2, 1, None)
            out[begin:begin + part.shape[0]].copy_(q)
        return out.reshape(shape)
    root = LIBRARIES / 'sgl-kernel-npu'
    return Candidate(Implementation(run), 'sgl_kernel_npu::split_qkv_tp_rmsnorm_rope(tp_world=1)',
                     'Pack X asQ and zeroK/V; original global RMS with complete hidden width, native head_dim4096/16384 chunking and identityRoPE; independent-row batches128, selectQ; no source/schedule changes.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_baselines.py'),
                                   root / host, root / 'python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py'),
                     limitations=('All packing, identity metadata, extra native K/V work and row-batch launches count in timing.',))
