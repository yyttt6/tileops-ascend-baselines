"""Full probabilities and min-p masks from unchanged original SGL stages."""
from pathlib import Path


def build(case):
    import torch
    from types import SimpleNamespace
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
    from .exp_composed import build as exponential
    from .extra_arithmetic import _affine
    from .additional import mul
    from .comparison import build as compare
    from .conditional import select
    x = case.inputs[0]
    proxy = lambda name:SimpleNamespace(inputs=case.inputs, params={}, _call=SimpleNamespace(signature=SimpleNamespace(name=name)))
    # The exponential factory only needs a single shape-bearing floating input.
    exp_case = SimpleNamespace(inputs=(x,),params={},_call=SimpleNamespace(signature=SimpleNamespace(name='ExpFwdOp')))
    exp = exponential(exp_case)
    product = mul(case)
    affine, hashes = _affine(case)
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py'
    peak = local_function('sgl-kernel-npu','python/sgl_kernel_npu',
                          'sgl_kernel_npu.sample.argmax_softmax_prob','argmax_softmax_prob_fused',host)
    is_mask = case._call.signature.name=='MinPMaskFwdOp'
    compare_ge = compare(proxy('GeFwdOp')) if is_mask else None
    choose = select(proxy('WhereFwdOp')) if is_mask else None
    dim = -1 if is_mask else case.params['dim']
    output_dtype = x.dtype if is_mask else getattr(torch,case.params.get('dtype') or str(x.dtype).split('.')[-1])
    def run(x, minimum_p=None):
        moved = x.movedim(dim,-1)
        rows = moved.float().contiguous().reshape(-1,moved.shape[-1])
        ids, peak_probability = peak(rows)
        max_value = rows.gather(1,ids[:,None])
        shifted = affine(max_value,rows,-1.0)
        relative = exp.implementation.run(shifted)
        if is_mask:
            keep = compare_ge.implementation.run(relative, minimum_p.float().reshape(-1,1))
            result = choose.implementation.run(keep,x,torch.full((),float('-inf'),device=x.device,dtype=x.dtype))
            return result
        result = product.implementation.run(relative,peak_probability[:,None])
        return result.reshape(moved.shape).movedim(-1,dim).to(output_dtype)
    sources = hashes | exp.source_sha256 | product.source_sha256 | source_hashes(Path(__file__), LIBRARIES/'sgl-kernel-npu'/host)
    if is_mask:sources |= compare_ge.source_sha256 | choose.source_sha256
    return Candidate(Implementation(run), 'sgl_kernel_npu::argmax_softmax_prob_fused + full native exp/affine/product'+(' + native threshold/select' if is_mask else ''),
        'Original full-row argmax/peak-softmax probability supplies the exact reduction domain. Original native exp of shifted logits and native product reconstruct every softmax probability. Min-p instead compares native exp(logit-max) with the original per-row threshold, then gathers unchanged original logit bits or -inf, preserving endpoint 0/1 and ties. No external softmax, log, comparison or masking arithmetic.',
        sources,
        limitations=('Original finite-logit manifest domain; near-threshold or FP32 rounding differences must pass the unchanged original comparator. No tolerance relaxation or full-softmax claim from an argmax-only output.',))
