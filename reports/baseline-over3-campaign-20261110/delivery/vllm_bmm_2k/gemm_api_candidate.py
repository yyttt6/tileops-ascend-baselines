from pathlib import Path
from .runtime import validate,module,TRITON_ROOT,candidate
def gemm_original(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    op, p = (case._call.signature.name, case.params)

    def run(a, b):
        if op == 'BmmFwdOp':
            return torch.stack([native.matmul_persistent(a[i].contiguous(), b[i].contiguous()) for i in range(a.shape[0])])
        aa = a.transpose(-1, -2) if p.get('trans_a', False) else a
        bb = b.transpose(-1, -2) if p.get('trans_b', False) else b
        return native.matmul_persistent(aa.contiguous(), bb.contiguous())
    return candidate(run, 'matmul_persistent', 'Honor transpose flags; contiguous views. BMM dispatches one original complete GEMM per batch and stacks outputs inside run; original FP32 accumulation and fixed tiles unchanged.', [Path(__file__), *sources])
def gemm_fixed(case):
    validate(case)
    import torch
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    op, p = (case._call.signature.name, case.params)

    def run(a, b):
        if op == 'BmmFwdOp':
            return torch.stack([native.linear_persistent(a[i].contiguous(), b[i].transpose(-1, -2).contiguous()) for i in range(a.shape[0])])
        aa = a.transpose(-1, -2) if p.get('trans_a', False) else a
        bb = b.transpose(-1, -2) if p.get('trans_b', False) else b
        return native.matmul_persistent(aa.contiguous(), bb.contiguous())
    return candidate(run, 'matmul_persistent', 'Honor transpose flags; contiguous views. BMM dispatches one original complete GEMM per batch and stacks outputs inside run; original FP32 accumulation and fixed tiles unchanged.', [Path(__file__), *sources])
