"""Compose existing native activation outputs without adding a kernel."""
from .extended import native, result
from .pointwise import softplus, _scale_native
from .select_extra import extra


def activation(case):
    import torch
    from types import SimpleNamespace
    tanh_case = SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name='TanhFwdOp')), params={}, inputs=case.inputs)
    tanh_candidate = extra(tanh_case)
    soft_candidate = softplus(case) if case._call.signature.name == 'MishFwdOp' else None
    product = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    norm = native('quant.norm_forward_kernel', 'norm_forward')
    def multiply(a, b):
        count = a.numel()
        left = torch.nn.functional.pad(a.float().contiguous().reshape(-1), (0, -count % 256))
        right = torch.nn.functional.pad(b.expand_as(a).float().contiguous().reshape(-1), (0, -count % 256))
        out = torch.empty_like(left).reshape(1, -1)
        product(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:count].reshape_as(a)
    def add(a, b):
        count = a.numel()
        left = torch.nn.functional.pad(a.float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        right = torch.nn.functional.pad(b.expand_as(a).float().contiguous().reshape(-1), (0, -count % 256)).reshape(-1, 256)
        return norm(left, None, 1e-6, residual=right)[2].reshape(-1)[:count].reshape_as(a)
    def run(x):
        if soft_candidate is not None:
            soft = soft_candidate.implementation.run(x.float())
            return multiply(x.float(), tanh_candidate.implementation.run(soft)).to(x.dtype)
        gate, value = x.chunk(2, dim=-1)
        xf = gate.float()
        cube = multiply(multiply(xf, xf), xf)
        argument = _scale_native(torch, product, add(xf, _scale_native(torch, product, cube, 0.044715)), 0.7978845608028654)
        smooth = add(tanh_candidate.implementation.run(argument), torch.ones((), device=x.device, dtype=torch.float32))
        activated = multiply(_scale_native(torch, product, xf, 0.5), smooth).to(x.dtype)
        return multiply(activated, value).to(x.dtype)
    api = 'mhc.pre_split_mixes_kernel.mhc_pre_split_mixes_fwd[post] + mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd + quant.norm_forward_kernel.norm_forward[2]'
    if soft_candidate is not None: api = 'moe.moe_topk_gate_forward_kernel.moe_topk_gate_forward[scores] + ' + api
    candidate = result(run, api,
                  'Mish uses original native softplus score selection, native tanh affine-sigmoid selection and native product. Tanh-GELU uses its prescribed cubic/tanh formula through native products/sums, rounds activation to input dtype, then native gate product. Every stage is counted; no Torch arithmetic.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py', 'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py')
    from dataclasses import replace
    sources = dict(candidate.source_sha256)
    sources.update(tanh_candidate.source_sha256)
    if soft_candidate is not None: sources.update(soft_candidate.source_sha256)
    return replace(candidate, source_sha256=sources)
