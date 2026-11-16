"""Supplied inverse routing plus original complete native weighted-row sum."""
from pathlib import Path


def build(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    from .additional import mul
    from .extra_arithmetic import _affine
    product = mul(case)
    affine, hashes = _affine(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/layernorm_gated.py'
    norm = local_function('sgl-kernel-npu','python/sgl_kernel_npu',
                          'sgl_kernel_npu.fla.layernorm_gated','layer_norm_fwd_npu',host)
    epilogue = case.params.get('epilogue')
    if isinstance(epilogue,dict):
        fields = epilogue if 'routed_scaling_factor' in epilogue else next(iter(epilogue.values()))
        scale = fields['routed_scaling_factor']
    else:
        scale = 1.0 if epilogue is None else epilogue.routed_scaling_factor
    dtype = getattr(torch,case.params.get('out_dtype') or str(case.inputs[0].dtype).split('.')[-1])
    def run(expert_output,topk_weights,inverse_indices):
        tokens,k = topk_weights.shape
        rows = expert_output.reshape(-1,expert_output.shape[-1])
        values = rows[inverse_indices.long()].float().reshape(tokens,k,-1)
        weighted = product.implementation.run(values,topk_weights.float().unsqueeze(-1))
        complete_rows = weighted.transpose(1,2).contiguous().reshape(-1,k)
        mean = norm(complete_rows,torch.ones(k,device=values.device,dtype=torch.float32),None,1e-5)[1]
        summed = affine(mean,torch.zeros_like(mean),k)
        if epilogue is not None:summed = affine(summed,torch.zeros_like(summed),scale)
        return summed.reshape(tokens,rows.shape[-1]).to(dtype)
    return Candidate(Implementation(run),'sgl_kernel_npu::scale_shift + complete-row LayerNorm mean + mul_add',
        'Use the supplied inverse routing only to gather unchanged expert rows. Original native multiplication applies route weights, and one complete native K-route mean plus exact K scaling forms each output element. Optional original native epilogue follows the FP32 sum, then one declared storage cast. No communication/topology or external weighted reduction required by this original call.',
        product.source_sha256 | hashes | source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))
