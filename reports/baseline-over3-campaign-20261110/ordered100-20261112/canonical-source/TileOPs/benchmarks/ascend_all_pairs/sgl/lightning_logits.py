"""Complete weighted-ReLU logits from original SGL matrix and row APIs."""
from pathlib import Path


def compose(torch, matmul, multiply, maximum, row_mean, affine):
    """Independent output tiling only; each native call retains all reduction terms."""
    def run(q, k, weights, starts, ends, key_scale=None):
        batch, queries, heads, dim = q.shape
        keys, groups = k.shape[1:3]
        group_heads = heads//groups
        # Host values determine copying ranges only, never score arithmetic.
        bounds = list(zip(starts.cpu().tolist(), ends.cpu().tolist(), strict=True))
        output = torch.full((batch, queries, keys, groups), float('-inf'), device=q.device, dtype=torch.float32)
        for b in range(batch):
            for g in range(groups):
                for qs in range(0, queries, 32):
                    qe = min(qs+32, queries)
                    qq = q[b, qs:qe, g*group_heads:(g+1)*group_heads].contiguous().reshape(-1, dim)
                    ww = weights[qs:qe, g*group_heads:(g+1)*group_heads].reshape(qe-qs, group_heads, 1)
                    for ks in range(0, keys, 1024):
                        ke = min(ks+1024, keys)
                        score = matmul(qq, k[b, ks:ke, g].contiguous()).reshape(qe-qs, group_heads, ke-ks)
                        if key_scale is not None:
                            score = multiply(score, key_scale[b, ks:ke, g].reshape(1, 1, -1))
                        score = maximum(score, torch.zeros((), device=q.device, dtype=torch.float32))
                        weighted = multiply(score, ww)
                        complete_rows = weighted.transpose(1, 2).contiguous().reshape(-1, group_heads)
                        mean = row_mean(complete_rows)
                        logits = affine(mean, torch.zeros_like(mean), group_heads).reshape(qe-qs, ke-ks)
                        for local_q, (lo, hi) in enumerate(bounds[qs:qe]):
                            begin, end = max(ks, lo, 0), min(ke, hi, keys)
                            if begin < end:
                                output[b, qs+local_q, begin:end, g].copy_(logits[local_q, begin-ks:end-ks])
        return output
    return run


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
    from .native_fp32_gemv import bind
    from .additional import mul, maximum
    from .extra_arithmetic import _affine
    q, k, *_ = case.inputs
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn) or k.dtype != q.dtype:
        raise UnsupportedCase('Full weighted logits use native FP16/BF16 or losslessly widened E4M3 inputs')
    matmul, hashes = bind(q.device)
    product, select = mul(case), maximum(case)
    affine, affine_hashes = _affine(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    norm = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.fla.layernorm_gated', 'layer_norm_fwd_npu_simd', host)
    def row_mean(rows):
        return norm(rows, torch.ones(rows.shape[1], device=rows.device, dtype=rows.dtype), None, 1e-5)[1]
    run = compose(torch, matmul, product.implementation.run, select.implementation.run, row_mean, affine)
    return Candidate(Implementation(run), 'sgl_kernel_npu::sgemmv_shrink + native scale_shift/argmax/complete-row mean/mul_add',
        'Return the complete FP32 weighted-ReLU logits, including every declared masked position. Original full-K SGEMMV, original key scaling, native ReLU selection and head weighting, then one complete native head-row mean and exact integer width scaling. Tile independent query/key/group outputs to bound workspace; preserve every reduction axis. Mask ranges only copy native results into -inf-initialized public output.',
        hashes | product.source_sha256 | select.source_sha256 | affine_hashes | source_hashes(Path(__file__), LIBRARIES/'sgl-kernel-npu'/host),
        timing_mode='host_roundtrip',
        limitations=('Moving per-key scale after the native dot and obtaining complete head sum from native mean changes FP32 rounding; every original case still requires its unchanged normalized-error check. No selected-index API is substituted for full logits.',))
