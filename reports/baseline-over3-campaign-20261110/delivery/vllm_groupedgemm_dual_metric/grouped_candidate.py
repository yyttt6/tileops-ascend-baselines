from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def grouped_original(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    ta, tb = (case.params['transpose_a'], case.params['transpose_b'])

    def run(a, b, sizes, offsets):
        counts, starts = (sizes.tolist(), offsets.tolist())
        outputs = []
        for i, (size, start) in enumerate(zip(counts, starts, strict=True)):
            aa = a[start:start + size]
            if ta:
                aa = aa.transpose(0, 1)
                bb = b[:, start:start + size].transpose(0, 1) if tb else b[start:start + size]
            else:
                bb = b[i].transpose(0, 1) if tb else b[i]
            if aa.shape[0] == 0:
                outputs.append(torch.empty((0, bb.shape[1]), dtype=a.dtype, device=a.device))
            else:
                outputs.append(native.matmul_persistent(aa.contiguous(), bb.contiguous()))
        return torch.stack(outputs) if ta else torch.cat(outputs)
    return candidate(run, 'matmul_persistent (segmented)', 'Read current group sizes and offsets inside run; slice each complete GEMM, honor all transpose combinations, concatenate or stack original outputs.', [Path(__file__), *sources], timing_mode='host_completed')

def grouped_fixed(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    ta, tb = (case.params['transpose_a'], case.params['transpose_b'])

    def run(a, b, sizes, offsets):
        counts, starts = (sizes.tolist(), offsets.tolist())
        outputs = []
        for i, (size, start) in enumerate(zip(counts, starts, strict=True)):
            aa = a[start:start + size]
            if ta:
                aa = aa.transpose(0, 1)
                bb = b[:, start:start + size].transpose(0, 1) if tb else b[start:start + size]
            else:
                bb = b[i].transpose(0, 1) if tb else b[i]
            if aa.shape[0] == 0:
                outputs.append(torch.empty((0, bb.shape[1]), dtype=a.dtype, device=a.device))
            else:
                outputs.append(native.linear_persistent(aa.contiguous(), bb.transpose(-1, -2).contiguous()))
        return torch.stack(outputs) if ta else torch.cat(outputs)
    return candidate(run, 'matmul_persistent (segmented)', 'Read current group sizes and offsets inside run; slice each complete GEMM, honor all transpose combinations, concatenate or stack original outputs.', [Path(__file__), *sources], timing_mode='host_completed')
