"""Complete native stages for the original SSD chunk-prefix and coupling calls."""
import json
from pathlib import Path
from types import SimpleNamespace


def cumsum(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes
    from .softplus import build as softplus
    from .extra_arithmetic import _affine
    from .additional import mul
    p = case.params
    chunk = p['chunk_len']
    if not p['dt_softplus'] or p['dt_min'] != 0 or p['dt_max'] != float('inf'):
        raise UnsupportedCase('This route covers the original softplus and [0,infinity] chunk-prefix cases')
    # Each independent original head becomes a batch entry with native H=1.
    # For the original chunk=256, unchanged BLOCK_T=1024 contains four chunks.
    block = 1 << (((2**18)//chunk)-1).bit_length()
    if block < chunk or case.inputs[0].shape[1] % block:
        raise UnsupportedCase('Original native chunk-prefix scheduling requires complete BLOCK_T tiles')
    dt = case.inputs[0]
    positive = softplus(SimpleNamespace(inputs=(dt,), params={'beta':1.0,'threshold':20.0}))
    affine, hashes = _affine(case)
    product = mul(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/cumsum.py'
    prefix = local_function('sgl-kernel-npu','python/sgl_kernel_npu',
                           'sgl_kernel_npu.fla.cumsum','chunk_local_cumsum',host)
    dtype = getattr(torch, p['out_dtype'])

    def run(dt, A, dt_bias=None):
        b, t, h = dt.shape
        raw = dt.float() if dt_bias is None else affine(dt.float(),dt_bias.float(),1.0)
        activated = positive.implementation.run(raw)
        # Softplus is positive: the declared [0,infinity] clamp is the identity.
        step = activated.transpose(1,2).contiguous().reshape(b,h,t//chunk,chunk)
        decay = product.implementation.run(activated,A.float())
        packed = decay.transpose(1,2).contiguous().reshape(b*h,t,1)
        accumulated = prefix(packed,chunk_size=chunk,output_dtype=torch.float32)
        return step.to(dtype), accumulated.reshape(b,h,t//chunk,chunk)

    return Candidate(Implementation(run),
        'sgl_kernel_npu::fused_gdn_gating_npu + mul_add + fused_scale_shift + chunk_local_cumsum',
        'Original native bias/Softplus/product pipeline in FP32, with lossless independent-head-to-batch layout for the unchanged complete chunk-local cumsum API. Return both declared dt_out storage and FP32 dA_cumsum; decay uses unrounded FP32 steps as in the original oracle. No external prefix arithmetic or native scheduling modification.',
        positive.source_sha256 | product.source_sha256 | hashes | source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def coupling(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from .native_fp32_gemv import bind
    x = case.inputs[0]
    chunk = case.params['chunk_len']
    path = Path(__file__).resolve().parents[4]/'.cache/all-pairs/sgl/ssd_layout'/f'causal_{chunk}.json'
    coordinates = json.loads(path.read_text())
    row = torch.tensor(coordinates['row'],device=x.device,dtype=torch.int64)
    col = torch.tensor(coordinates['column'],device=x.device,dtype=torch.int64)
    mm, hashes = bind(x.device)

    def run(C_mat,B_mat):
        b,t,g,n = C_mat.shape
        count = t//chunk
        left = C_mat.reshape(b,count,chunk,g,n).permute(0,1,3,2,4).contiguous().reshape(-1,chunk,n)
        right = B_mat.reshape(b,count,chunk,g,n).permute(0,1,3,2,4).contiguous().reshape(-1,chunk,n)
        output = torch.zeros((left.shape[0],chunk,chunk),device=C_mat.device,dtype=C_mat.dtype)
        for i in range(left.shape[0]):
            full = mm(left[i],right[i])
            output[i,row,col] = full[row,col].to(C_mat.dtype)
        return output.reshape(b,count,g,chunk,chunk)

    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink complete state-axis coupling',
        'Pack each independent batch/chunk/group into complete original state-axis native SGEMMV products with FP32 accumulators. Copy only source-independent causal coordinates into zero-initialized output, preserving the original output dtype and all rows. No external matmul, causal arithmetic, partial reduction or changed native schedule.',
        hashes | source_hashes(Path(__file__),path))
