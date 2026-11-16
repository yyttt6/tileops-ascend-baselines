"""Bound independent elementwise rows through layout alone."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, _add_layout, local_function, source_hashes


def sgl_add_cases(case):
    import math
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    _, _, shape, _, _, pack = _add_layout(case)
    size = math.prod(shape)
    hidden = 4096
    tail = -size % hidden
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.moe.mul_add', 'mul_add', host)
    def rows(value):
        return F.pad(pack(value).reshape(-1), (0, tail)).reshape(-1, hidden)
    def run(x, y):
        return fn(rows(y), rows(x), 1).reshape(-1)[:size].reshape(shape)
    return Candidate(Implementation(run), 'sgl_kernel_npu.moe.mul_add.mul_add(y,x,1)',
                     'Materialize broadcasts per call; reshape independent elements into width4096 rows, zero-pad tail if needed and crop. Original kernel/schedule unchanged.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host,
                                   LIBRARIES / 'sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py'))
