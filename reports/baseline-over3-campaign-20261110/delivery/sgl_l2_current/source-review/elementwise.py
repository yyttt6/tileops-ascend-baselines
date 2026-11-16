"""Layout/API-only adapters for original SGL elementwise kernels."""
from pathlib import Path


def _gated(case, sigmoid_only):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/activation/fused_sigmoid_mul.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                        'sgl_kernel_npu.activation.fused_sigmoid_mul', 'fused_sigmoid_mul', host)
    def run(x):
        return fn(torch.ones_like(x) if sigmoid_only else x, x)
    return Candidate(Implementation(run), 'sgl_kernel_npu.activation.fused_sigmoid_mul.fused_sigmoid_mul',
                     'Original x*sigmoid(gate); constant-one x for sigmoid, shared input for SiLU. No input-dependent precomputation.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host,
                                   LIBRARIES / 'sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py'))


def sigmoid(case):
    return _gated(case, True)


def silu(case):
    return _gated(case, False)


def sub(case):
    import math
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    x, y = case.inputs
    shape = torch.broadcast_shapes(x.shape, y.shape)
    size = math.prod(shape)
    tail = -size % 4096
    alpha = case.params.get('alpha', 1)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.moe.mul_add', 'mul_add', host)
    def pack(t):
        return F.pad(t.expand(shape).contiguous().reshape(-1), (0, tail)).reshape(-1, 4096)
    def run(x, y):
        return fn(pack(y), pack(x), -alpha).reshape(-1)[:size].reshape(shape)
    return Candidate(Implementation(run), 'sgl_kernel_npu.moe.mul_add.mul_add(other,input,-alpha)',
                     'Native scalar factor=-alpha; broadcast, flatten independent elements, pad/crop tail; all input transformations are per call.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host))


def neg(case):
    from types import SimpleNamespace
    from dataclasses import replace
    import torch
    from benchmarks.api import Implementation
    native = sub(SimpleNamespace(inputs=(torch.zeros_like(case.inputs[0]),case.inputs[0]), params={'alpha':1}))
    def run(x):
        return native.implementation.run(torch.zeros_like(x), x)
    return replace(native, implementation=Implementation(run), adaptation='Native mul_add with factor=-1 and zero shared input; per-call layout and padding.')


def silu_and_mul(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/activation/swiglu_quant.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.activation.swiglu_quant', 'swiglu_quant', host)
    x, = case.inputs
    size = x.numel() // 2
    tail = -size % 1024
    counts = torch.tensor([(size+tail)//1024], device=x.device, dtype=torch.int32)
    def run(x):
        gate, up = x.chunk(2, dim=-1)
        gate = F.pad(gate.contiguous().view(-1), (0,tail)).view(-1,1024)
        up = F.pad(up.contiguous().view(-1), (0,tail)).view(-1,1024)
        out, _ = fn(torch.cat((gate,up), dim=-1), counts, 1, need_quant=False)
        return out.reshape(-1)[:size].reshape(*x.shape[:-1],x.shape[-1]//2)
    return Candidate(Implementation(run), 'sgl_kernel_npu.activation.swiglu_quant.swiglu_quant(need_quant=False)',
                     'Separate gate/up halves, reshape independent elements into 1024-wide rows, pad/crop tail; disable optional quantization via original API.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host))
