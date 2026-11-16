"""Original SGL normalization APIs with manifest layouts and optional affine."""
from pathlib import Path
import math


def _candidate(run, host, api, adaptation):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, source_hashes
    return Candidate(Implementation(run),api,adaptation,source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def layernorm(case):
    import torch
    from benchmarks.ascend_baselines import local_function
    x,weight,bias = case.inputs
    width = math.prod(case.params['normalized_shape'])
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    eps = case.params.get('eps')
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    def run(x,weight,bias):
        w = torch.ones(width, device=x.device, dtype=x.dtype) if weight is None else weight.contiguous().reshape(width)
        b = None if bias is None else bias.contiguous().reshape(width)
        return fn(x.contiguous().reshape(-1,width),w,b,eps)[0].reshape(x.shape)
    return _candidate(run,host,'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd',
                      'Flatten complete normalized dimensions; optional unit affine and native bias. Original centered variance and schedule unchanged.')


def fused_add_rms(case):
    import torch
    from benchmarks.ascend_baselines import local_function
    x,residual,weight = case.inputs
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/norm/add_rmsnorm_bias.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.add_rmsnorm_bias','add_rmsnorm_bias',host)
    # Vendor manual cache omits dtype. Factory-local reset prevents stale binaries
    # in a process that validates multiple dtypes; compilation is unchanged.
    fn.__globals__['kernels'].clear()
    bias = torch.zeros_like(weight)
    def run(x,residual,weight):
        return fn(x.contiguous(),residual.contiguous(),weight.contiguous(),bias,case.params['eps'])
    return _candidate(run,host,'sgl_kernel_npu.norm.add_rmsnorm_bias.add_rmsnorm_bias',
                      'Original fused add and RMS normalization; zero optional bias, quantization disabled, both outputs preserved. Clear dtype-incomplete vendor host cache at factory build.')


def groupnorm(case):
    import torch
    from benchmarks.ascend_baselines import local_function
    x,weight,bias = case.inputs
    n,c,*spatial = x.shape
    size = math.prod(spatial)
    groups = case.params['num_groups']
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    def run(x,weight,bias):
        w = torch.ones(c*size, device=x.device,dtype=x.dtype) if weight is None else weight[:,None].expand(c,size).contiguous().reshape(-1)
        b = None if bias is None else bias[:,None].expand(c,size).contiguous().reshape(-1)
        return fn(x.contiguous().reshape(n,c*size),w,b,case.params['eps'],group_size=c*size//groups)[0].reshape(x.shape)
    return _candidate(run,host,'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd(group_size)',
                      'NCHW contiguous groups become native group_size; repeat supplied per-channel affine over spatial positions inside every run.')


def instancenorm(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase, local_function
    x,rm,rv,weight,bias = case.inputs
    if not case.params.get('use_input_stats',True):
        raise UnsupportedCase('Native layer norm uses input statistics; running-stat inference requires a different arithmetic API.')
    n,c,*spatial = x.shape
    size = math.prod(spatial)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    fn = local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu_simd',host)
    def run(x,rm,rv,weight,bias):
        w = torch.ones(c*size,device=x.device,dtype=x.dtype) if weight is None else weight[:,None].expand(c,size).contiguous().reshape(-1)
        b = None if bias is None else bias[:,None].expand(c,size).contiguous().reshape(-1)
        return fn(x.contiguous().reshape(n,c*size),w,b,case.params['eps'],group_size=size)[0].reshape(x.shape)
    return _candidate(run,host,'sgl_kernel_npu.fla.layernorm_gated.layer_norm_fwd_npu_simd(group_size=spatial)',
                      'One complete spatial group per channel; per-call affine broadcast; output uses original input statistics, supplied running-stat buffers remain protected.')
