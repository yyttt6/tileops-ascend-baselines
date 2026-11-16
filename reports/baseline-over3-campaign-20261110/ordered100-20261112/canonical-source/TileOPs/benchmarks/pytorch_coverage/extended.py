"""Device PyTorch implementations of the extended convolution/GEMM/merge cases.

Only packed convolution's small integer sequence offsets are read on the host
for scheduling. Numerical tensors and all arithmetic stay on their input device.
The optional CPU reference is an independently evaluated verification path; it
is never called by a bound candidate.
"""

import torch
import torch.nn.functional as F


OPS = {
    "CausalConv1dFwdOp",
    "CausalConv1dUpdateFwdOp",
    "GemmINT8W8A8FwdOp",
    "MergeAttentionStatesFwdOp",
}

MUTATED = {"CausalConv1dFwdOp": (2,), "CausalConv1dUpdateFwdOp": (2,)}


def _causal_conv(x, weight, conv_state, cu_seqlens, bias=None,
                 has_initial_state=None, *, activation):
    # cu_seqlens contains B+1 small integer scheduling values, not numerical
    # activations. Read the actual argument rather than assuming manifest data.
    offsets = cu_seqlens.tolist()
    width = weight.shape[-1]
    wf = weight.float()
    output = torch.empty_like(x)
    for sequence, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        history = conv_state[sequence]
        if has_initial_state is None:
            history = torch.zeros_like(history)
        else:
            history = torch.where(has_initial_state[sequence], history, 0)
        # Retain the original input dtype for exact history copies. Chunk only
        # intermediate arithmetic, never the logical sequence or return value.
        tokens = torch.cat((history, x[:, begin:end]), dim=-1)
        for start in range(0, end - begin, 1024):
            stop = min(start + 1024, end - begin)
            value = tokens[:, start:stop].float() * wf[:, 0, None]
            for tap in range(1, width):
                value = value + tokens[:, start + tap:stop + tap].float() * wf[:, tap, None]
            if bias is not None:
                value = value + bias.float()[:, None]
            if activation == "silu":
                value = F.silu(value)
            output[:, begin + start:begin + stop] = value.to(x.dtype)
        conv_state[sequence].copy_(tokens[:, -(width - 1):])
    return output, conv_state


def _causal_update(x, weight, conv_state, bias=None, *, activation):
    state = torch.cat((conv_state[..., 1:], x.unsqueeze(-1)), dim=-1)
    value = (state.float() * weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        value = value + bias.float()
    if activation == "silu":
        value = F.silu(value)
    conv_state.copy_(state)
    return value.to(x.dtype), conv_state


def _int8_dot(a, b):
    """Exact int32 contraction, including the full signed int8 input domain.

    Each 512-term partial has sum(abs(products)) <= 512*128*128 = 2**23.
    Integer operands/products and every possible partial sum are exactly
    representable in FP32 (also when operands use a reduced input mantissa).
    Convert each exact partial to int32 *before* adding it to the accumulator.
    The largest manifest K=28672 bounds the total by 469762048 < 2**31.
    """
    result = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.int32, device=a.device)
    for start in range(0, a.shape[1], 512):
        partial = a[:, start:start + 512].float() @ b[:, start:start + 512].float().T
        result.add_(partial.to(torch.int32))
    return result


def _int8_gemm(a, b, scale_a, scale_b, bias=None, *, out_dtype):
    output = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
    # Bound partial/accumulator allocations while preserving all M,N,K values.
    for start in range(0, a.shape[0], 512):
        stop = min(start + 512, a.shape[0])
        value = _int8_dot(a[start:stop], b).float()
        sa = scale_a if scale_a.shape[0] == 1 else scale_a[start:stop]
        value = value * sa.float() * scale_b.float()
        if bias is not None:
            value = value + bias.float()
        output[start:stop] = value.to(out_dtype)
    return output


def _merge(v_a, s_a, v_b, s_b):
    output = torch.empty_like(v_a)
    lse = torch.empty_like(s_a)
    for start in range(0, v_a.shape[0], 256):
        stop = min(start + 256, v_a.shape[0])
        a, b = s_a[start:stop].float(), s_b[start:stop].float()
        maximum = torch.maximum(a, b)
        empty = torch.isneginf(maximum)
        safe_maximum = torch.where(empty, 0, maximum)
        wa, wb = torch.exp(a - safe_maximum), torch.exp(b - safe_maximum)
        denominator = wa + wb
        # A zero-weight state's values can be nonfinite: mask before multiply.
        va = torch.where((wa != 0).unsqueeze(-1), v_a[start:stop].float(), 0)
        vb = torch.where((wb != 0).unsqueeze(-1), v_b[start:stop].float(), 0)
        value = (va * wa.unsqueeze(-1) + vb * wb.unsqueeze(-1)) / torch.where(
            empty, 1, denominator
        ).unsqueeze(-1)
        output[start:stop] = value.to(v_a.dtype)
        lse[start:stop] = (safe_maximum + torch.log(denominator)).to(s_a.dtype)
    return output, lse


def bind(case):
    name = case._call.signature.name
    params = case._call.params
    if name == "CausalConv1dFwdOp":
        return lambda *inputs: _causal_conv(*inputs, activation=params["activation"])
    if name == "CausalConv1dUpdateFwdOp":
        return lambda *inputs: _causal_update(*inputs, activation=params["activation"])
    if name == "GemmINT8W8A8FwdOp":
        return lambda *inputs: _int8_gemm(*inputs, out_dtype=getattr(torch, params["out_dtype"]))
    if name == "MergeAttentionStatesFwdOp":
        return _merge
    raise KeyError(name)


def cpu_reference(case, inputs):
    """Independent CPU oracle; use exact FP64 BLAS for the large INT8 cases.

    Signed int8 dot products at every manifest K fit in int32, hence are exact
    in FP64 too. This avoids CPU's unoptimized int32 GEMM on trillion-op cases.
    Scale and bias are evaluated in FP64 as in workloads.ascend_extended.
    """
    if any(isinstance(x, torch.Tensor) and x.device.type != "cpu" for x in inputs):
        raise ValueError("cpu_reference requires already captured CPU inputs")
    if case._call.signature.name != "GemmINT8W8A8FwdOp":
        return case.reference(*inputs)
    a, b, scale_a, scale_b, bias = inputs
    dtype = getattr(torch, case._call.params["out_dtype"])
    output = torch.empty((a.shape[0], b.shape[0]), dtype=dtype)
    bd = b.double().T
    for start in range(0, a.shape[0], 512):
        stop = min(start + 512, a.shape[0])
        value = a[start:stop].double() @ bd
        sa = scale_a if scale_a.shape[0] == 1 else scale_a[start:stop]
        value *= sa.double()
        value *= scale_b.double()
        if bias is not None:
            value += bias.double()
        output[start:stop] = value.to(dtype)
    return output
