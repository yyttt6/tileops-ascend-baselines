"""CPU-reviewable packaging for full ForeachMulList; no arithmetic replacement.

The callback must own a qualified private original native API. This module alone
is not provider qualification. Broadcasting and both contiguous conversions are
performed per call and belong inside the formal measured run.
"""
def pack_and_apply(torch, native_foreach, x, y):
    if x.dtype != y.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('Original native FP16/BF16/FP32 same-dtype contract')
    if x.device != y.device:
        raise ValueError('Inputs must share the measured target device')
    a, b = torch.broadcast_tensors(x, y)
    a, b = a.contiguous(), b.contiguous()
    result = native_foreach([a], [b])
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise ValueError('Complete native TensorList result required')
    out = result[0]
    if out.shape != a.shape or out.dtype != a.dtype or out.device != a.device:
        raise ValueError('Native output contract changed')
    return out
