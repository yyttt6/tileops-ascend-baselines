"""Full RMS native quant with runtime scale-floor fallback; no padding/crop."""
import torch
import torch_npu


def factory(params):
    dtype = getattr(torch, params['quant_dtype'])
    fp8 = dtype == torch.float8_e4m3fn
    qmax = 448.0 if fp8 else 127.0
    minimum = 1.0 / (448.0 * 512.0) if fp8 else torch.finfo(torch.float32).eps
    # Mathematical constants only: no input/weight/output-dependent cache.
    gamma = torch.ones(params['x'][0][-1], dtype=getattr(torch,params['x'][1]), device='npu')
    divisor = torch.tensor(qmax, dtype=torch.float32, device='npu')
    epsilon, upper = params['eps'], params['scale_ub']

    def run(x, residual, weight):
        normalized, _rstd, summed = torch_npu.npu_add_rms_norm(
            x, residual, gamma, epsilon=epsilon
        )
        # These are the two original low-dtype roundings, before and after gamma.
        weighted = normalized.to(x.dtype) * weight
        if upper is None and torch.isfinite(weighted).all().item():
            q, scale = torch_npu.npu_dynamic_quant(
                weighted.reshape(-1, weighted.shape[-1]), dst_type=dtype
            )
            scale = scale.reshape(*weighted.shape[:-1], 1)
            # Small scales require the original floor and quantized codes.
            # Check actual runtime output scales; never cache this predicate.
            if torch.all(scale >= minimum).item():
                return q.reshape_as(weighted), scale, summed.to(x.dtype)
        # Preserve original upperbound and all nonfinite behavior, including
        # NaN code policy, by executing the original quantization expression.
        y = weighted.float()
        maximum = torch.where(y.isnan(), 0.0, y.abs()).amax(-1, keepdim=True)
        if upper is not None:
            maximum = maximum.clamp(max=upper)
        scale = (maximum / divisor).clamp(min=minimum)
        v = y * (1.0 / scale)
        v = torch.where(v.isnan(), qmax if fp8 else 0.0, v)
        q = v.clamp(-qmax, qmax) if fp8 else v.clamp(-128.0, qmax).round()
        return q.to(dtype), scale, summed.to(x.dtype)

    return run
