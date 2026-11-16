"""Original ops-nn MatMulV3 with complete tiles and its native NT binding."""
from pathlib import Path

from benchmarks.ascend_baselines import WORKSPACE
from benchmarks.ascend_gemm_cases import _layout
from benchmarks.ascend_graph_call import GraphCall
from benchmarks.ascend_nn_graph import _candidate


def opsnn_aligned_gemm(case):
    import torch

    rows, columns, _, pack = _layout(case, (1024, 256, 64))
    binding = GraphCall('ops-nn')

    def run(a, b):
        lhs, rhs_nt = pack(a, b)
        out = torch.empty((lhs.shape[0],rhs_nt.shape[0]),device=a.device,dtype=a.dtype)
        binding.call('MatMulV3',[lhs,rhs_nt,None,None],[out],
                     {'transpose_x1': False, 'transpose_x2': True})
        return out[:rows,:columns].contiguous()

    return _candidate(binding,run,'MatMulV3',
        'Pad independent rows/columns/reduction with zeros to1024x256x64; bind original MatMulV3 transpose_x2=True to contiguous [N,K] right operand; crop output. No native kernel, tiler or schedule edits.',
        'matmul/mat_mul_v3',extra_sources=(Path(__file__),
            Path(__file__).with_name('ascend_gemm_cases.py'),
            WORKSPACE/'.cache/ascend-classic/native-pilot-20261020/ops-nn/source/matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp'))
