"""Device PyTorch implementations of DeltaNet and gated linear attention.

Inference uses bounded blocks of FP32 matrix products; the CPU oracle instead
evaluates the token recurrence in FP64.  No candidate invokes a CPU reference,
FLA, a custom operator, or a third-party kernel.  Only sequence-offset integers
may be read on the host (prefer the supplied CPU copy); tensor arithmetic stays
on the input device.  All input tensors are preserved.
"""

import math

import torch
import torch.nn.functional as F

OPS = {"DeltaNetFwdOp", "GLAFwdOp", "DeltaNetChunkFwdOp"}
MUTATED = {}


def _inverse_unit_lower(lower):
    """Finite geometric series by doubling, not a triangular solver.

    Strictly lower C-by-C L is nilpotent: (I+L)^-1=sum_{j=0}^{C-1}(-L)^j.
    This also works when C is not a power of two.
    """
    size = lower.shape[-1]
    power = -lower
    inverse = torch.eye(size, device=lower.device, dtype=lower.dtype) + power
    span = 2
    while span < size:
        power = power @ power
        inverse = inverse + power @ inverse
        span *= 2
    return inverse


def _blocks(x, chunk):
    """BHLD -> BHNCD, padding the final block with zero."""
    length, width = x.shape[-2:]
    padded = F.pad(x, (0, 0, 0, (-length) % chunk))
    return padded.reshape(*x.shape[:-2], -1, chunk, width)


def _delta_blocks(q, k, v, beta, state, chunk, *, training=False):
    """BHLD delta rule, with exact neutral padding and optional saved buffers."""
    length = q.shape[-2]
    qb, kb, vb = (_blocks(x.float(), chunk) for x in (q, k, v))
    bb = _blocks(beta.float().unsqueeze(-1), chunk)
    lower = torch.tril((kb @ kb.transpose(-1, -2)) * bb, diagonal=-1)
    inverse = _inverse_unit_lower(lower)
    w = inverse @ (kb * bb)
    u = inverse @ (vb * bb)
    attention = torch.tril(qb @ kb.transpose(-1, -2))
    outputs = []
    states = [state] if training else None
    for c in range(qb.shape[-3]):
        residual = u[:, :, c] - w[:, :, c] @ state
        outputs.append(qb[:, :, c] @ state + attention[:, :, c] @ residual)
        state = state + kb[:, :, c].transpose(-1, -2) @ residual
        if training:
            states.append(state)
    out = torch.cat(outputs, dim=2)[:, :, :length]
    if training:
        flatten = lambda x: x.flatten(2, 3)[:, :, :length].to(q.dtype).contiguous()
        aw = flatten(inverse)
        return (out.to(q.dtype).contiguous(), torch.stack(states, dim=2).contiguous(),
                aw, aw.clone(), flatten(w), flatten(u))
    return out, state


def _gla_blocks(q, k, v, gates, state, scale):
    """BHLD GLA using short, independently rebased log-gate blocks.

    The manifest gates lie in [-1, 0]. A 32-token block keeps exp(-prefix)
    below exp(32), so a long sequence cannot overflow the factorization.
    """
    length = q.shape[-2]
    qb, kb, vb, gb = (_blocks(x.float(), 32) for x in (q, k, v, gates))
    prefix = gb.cumsum(dim=-2)
    last = prefix[..., -1, :]
    weighted_q = qb * prefix.exp() * scale
    weighted_k = kb * (-prefix).exp()
    local = torch.tril(weighted_q @ weighted_k.transpose(-1, -2)) @ vb
    additions = (kb * (last.unsqueeze(-2) - prefix).exp()).transpose(-1, -2) @ vb
    decay = last.exp().unsqueeze(-1)
    outputs = []
    for c in range(qb.shape[-3]):
        outputs.append(weighted_q[:, :, c] @ state + local[:, :, c])
        state = state * decay[:, :, c] + additions[:, :, c]
    return torch.cat(outputs, dim=2)[:, :, :length], state


def _ranges(q, cu_seqlens, cu_seqlens_cpu):
    if cu_seqlens is None:
        return [(0, q.shape[0], 0, q.shape[1])]
    offsets = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens).tolist()
    return [(i, i + 1, start, end) for i, (start, end) in enumerate(zip(offsets, offsets[1:]))]


def _inference(q, k, v, gate, initial_state, cu_seqlens, cu_seqlens_cpu,
               *, delta, scale, l2norm):
    output = torch.empty_like(v)
    final = []
    packed = cu_seqlens is not None
    for first, end, start, stop in _ranges(q, cu_seqlens, cu_seqlens_cpu):
        batch = slice(0, 1) if packed else slice(first, end)
        sl = (batch, slice(start, stop))
        qq, kk, vv = (x[sl].float().transpose(1, 2).contiguous() for x in (q, k, v))
        if l2norm:
            qq = qq * torch.rsqrt(qq.square().sum(dim=-1, keepdim=True) + 1e-6)
            kk = kk * torch.rsqrt(kk.square().sum(dim=-1, keepdim=True) + 1e-6)
        state = (torch.zeros((*qq.shape[:2], qq.shape[-1], vv.shape[-1]),
                             device=q.device, dtype=torch.float32)
                 if initial_state is None else initial_state[first:end].float().clone())
        gg = gate[sl].float().transpose(1, 2).contiguous()
        if stop == start:
            final.append(state)
            continue
        if stop - start == 1:
            qt, kt, vt = qq[:, :, 0], kk[:, :, 0], vv[:, :, 0]
            if delta:
                residual = gg[:, :, 0, None] * (vt - (kt[..., None] * state).sum(dim=-2))
                state = state + kt[..., None] * residual[..., None, :]
            else:
                state = state * gg[:, :, 0].exp()[..., None] + kt[..., None] * vt[..., None, :]
            out = ((qt[..., None] * state).sum(dim=-2) * scale).unsqueeze(2)
        elif delta:
            out, state = _delta_blocks(qq * scale, kk, vv, gg, state, 64)
        else:
            out, state = _gla_blocks(qq, kk, vv, gg, state, scale)
        output[sl] = out.transpose(1, 2).to(v.dtype)
        final.append(state)
    return output, torch.cat(final, dim=0).contiguous()


def bind(case):
    """Bind only scalar manifest arguments; never materialize case.arguments."""
    op = case._call.signature.name
    if op not in OPS:
        raise NotImplementedError(op)
    params = case._call.arguments({})
    if op == "DeltaNetChunkFwdOp":
        chunk = params["chunk_size"]

        def run(q, k, v, beta):
            state = torch.zeros((*q.shape[:2], q.shape[-1], v.shape[-1]),
                                dtype=torch.float32, device=q.device)
            return _delta_blocks(q, k, v, beta, state, chunk, training=True)

        return run

    def run(q, k, v, gate, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None):
        scale = params["scale"]
        return _inference(q, k, v, gate, initial_state, cu_seqlens, cu_seqlens_cpu,
                          delta=op == "DeltaNetFwdOp",
                          scale=q.shape[-1] ** -0.5 if scale is None else scale,
                          l2norm=params.get("use_qk_l2norm_in_kernel", False))

    return run


@torch.no_grad()
def cpu_reference(case, inputs):
    """Independent FP64 token recurrence (CPU tensors are mandatory).

    The optional training op uses the existing independent FP64 triangular-solve
    oracle, which checks all six public tensors, including both WY inverses.
    """
    if any(isinstance(x, torch.Tensor) and x.device.type != "cpu" for x in inputs):
        raise ValueError("cpu_reference requires captured CPU inputs")
    op = case._call.signature.name
    params = case._call.arguments({})
    if op == "DeltaNetChunkFwdOp":
        from workloads.linear_attention.chunk_cpu import deltanet_chunk_fwd_cpu
        return deltanet_chunk_fwd_cpu(*inputs, params["chunk_size"])
    if op not in OPS:
        raise NotImplementedError(op)
    q, k, v, gate, initial, offsets, _ = inputs
    delta = op == "DeltaNetFwdOp"
    normalize = params.get("use_qk_l2norm_in_kernel", False)
    scale = params["scale"]
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    if offsets is None:
        spans = [(0, q.shape[0], 0, q.shape[1])]
    else:
        bounds = offsets.tolist()
        spans = [(i, i + 1, a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    out = torch.empty_like(v)
    finals = []
    for first, end, start, stop in spans:
        rows = slice(first, end) if offsets is None else slice(0, 1)
        qq, kk, vv, gg = (x[rows, start:stop].double() for x in (q, k, v, gate))
        if normalize:
            qq = qq / (torch.linalg.vector_norm(qq, dim=-1, keepdim=True).square() + 1e-6).sqrt()
            kk = kk / (torch.linalg.vector_norm(kk, dim=-1, keepdim=True).square() + 1e-6).sqrt()
        state = (torch.zeros((qq.shape[0], q.shape[2], q.shape[-1], v.shape[-1]), dtype=torch.float64)
                 if initial is None else initial[first:end].double().clone())
        for t in range(stop - start):
            qt, kt, vt = qq[:, t], kk[:, t], vv[:, t]
            if delta:
                residual = gg[:, t, :, None] * (vt - torch.einsum("bhk,bhkv->bhv", kt, state))
                state = state + torch.einsum("bhk,bhv->bhkv", kt, residual)
            else:
                state = gg[:, t].exp()[..., None] * state + torch.einsum("bhk,bhv->bhkv", kt, vt)
            out[rows, start + t] = (scale * torch.einsum("bhk,bhkv->bhv", qt, state)).to(v.dtype)
        finals.append(state.float())
    return out, torch.cat(finals, dim=0).contiguous()
