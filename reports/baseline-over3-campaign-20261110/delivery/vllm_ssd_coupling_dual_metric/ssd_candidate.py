from .omission_geometry import coupling_geometry
def coupling_original(p, matmul, muls_add):
    batch, chunks, groups, q, state = coupling_geometry(p)

    def run(c_mat, b_mat):
        import torch
        c = c_mat.float().reshape(batch, chunks, q, groups, state)
        b = b_mat.float().reshape(batch, chunks, q, groups, state)
        result = torch.empty(tuple(p['cb'][0]), dtype=c_mat.dtype, device=c_mat.device)
        for bi in range(batch):
            for ci in range(chunks):
                for gi in range(groups):
                    full = matmul(c[bi, ci, :, gi].contiguous(), b[bi, ci, :, gi].transpose(0, 1).contiguous())
                    flat = full.reshape(-1, 1)
                    masked = muls_add(flat, torch.full_like(flat, -0.0), 0.0).reshape(q, q)
                    for row in range(q):
                        masked[row, :row + 1].copy_(full[row, :row + 1])
                    result[bi, ci, gi].copy_(masked.to(c_mat.dtype))
        return result
    return run

def coupling_fixed(p, matmul, muls_add):
    batch, chunks, groups, q, state = coupling_geometry(p)

    def run(c_mat, b_mat):
        import torch
        c = c_mat.float().reshape(batch, chunks, q, groups, state)
        b = b_mat.float().reshape(batch, chunks, q, groups, state)
        result = torch.empty(tuple(p['cb'][0]), dtype=c_mat.dtype, device=c_mat.device)
        lower = torch.tril_indices(q, q, device='cpu').to(c_mat.device)
        for bi in range(batch):
            for ci in range(chunks):
                for gi in range(groups):
                    full = matmul(c[bi, ci, :, gi].contiguous(), b[bi, ci, :, gi].transpose(0, 1).contiguous())
                    flat = full.reshape(-1, 1)
                    masked = muls_add(flat, torch.full_like(flat, -0.0), 0.0).reshape(q, q)
                    masked[lower[0], lower[1]] = full[lower[0], lower[1]]
                    result[bi, ci, gi].copy_(masked.to(c_mat.dtype))
        return result
    return run

from .runtime import TRITON_ROOT,candidate,module,validate
from .omissions import _sources
def ssd_original(case):
    validate(case)
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    arithmetic, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
    return candidate(coupling_original(case.params, native.matmul_persistent, arithmetic.muls_add_triton), 'matmul_persistent + muls_add_triton (causal SSD coupling)', 'Exact FP32 widening; whole C_chunk @ B_chunk.T for every batch/chunk/group. Original scalar multiply-add supplies FP32 zero masking, then copy the causal lower triangle and cast once, matching current oracle including nonfinite upper entries. All outputs/packing timed.', [*_sources(), *sources, *extra], timing_mode='host_completed', limitations=('The audit suggested literal zero initialization, but current oracle multiplies by a mask: nonfinite upper-triangle products must remain NaN. This candidate follows the unchanged oracle.',))

def ssd_fixed(case):
    validate(case)
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    arithmetic, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
    return candidate(coupling_fixed(case.params, native.matmul_persistent, arithmetic.muls_add_triton), 'matmul_persistent + muls_add_triton (causal SSD coupling)', 'Exact FP32 widening; whole C_chunk @ B_chunk.T for every batch/chunk/group. Original scalar multiply-add supplies FP32 zero masking, then copy the causal lower triangle and cast once, matching current oracle including nonfinite upper entries. All outputs/packing timed.', [*_sources(), *sources, *extra], timing_mode='host_completed', limitations=('The audit suggested literal zero initialization, but current oracle multiplies by a mask: nonfinite upper-triangle products must remain NaN. This candidate follows the unchanged oracle.',))
