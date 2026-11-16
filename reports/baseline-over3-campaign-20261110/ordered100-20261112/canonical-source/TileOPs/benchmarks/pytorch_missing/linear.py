"""Chunked FP32 PyTorch derivatives and a bounded-memory selective scan.

DeltaNet uses the chunk WY equations also used by upstream TileOPs' CUDA
chunk_bwd.py, with explicit differentiation of the triangular system. GLA and
S6 use associative affine scans, avoiding division by tiny prefix products.
All inputs and arithmetic stay on the caller's device; no vendor kernels or
CPU reference functions are invoked.
"""

import torch
import torch.nn.functional as F


def _transpose(x):
    return x.transpose(-1, -2)


def _affine_scan(a, b):
    """Inclusive (a,b) composition along dimension -2: s[t]=a[t]*s[t-1]+b[t]."""
    step = 1
    while step < b.shape[-2]:
        b = torch.cat(
            (b[..., :step, :], b[..., step:, :] + a[..., step:, :] * b[..., :-step, :]), -2
        )
        a = torch.cat((a[..., :step, :], a[..., step:, :] * a[..., :-step, :]), -2)
        step *= 2
    return a, b


def deltanet_chunk_bwd(do, q, k, v, beta, S, Aw, Au, w, u, *, chunk_size=64):
    """BHLD -> dq/dk/dv/dbeta; saved inputs must be forward-consistent.

    Recompute the WY system in FP32 from k/v/beta rather than differentiating
    rounded saved Aw/Au/w/u. S supplies the FP32 chunk boundaries. The strictly
    lower triangular matrix is nilpotent, so the finite doubling inverse is
    exact algebraically and needs no unsupported triangular-solver operator.
    """
    B, H, length, K = q.shape
    if length % chunk_size:
        raise ValueError("Sequence length must be divisible by chunk_size")
    nc, C = length // chunk_size, chunk_size

    def chunks(x):
        return x.float().reshape(B, H, nc, C, x.shape[-1])

    qc, kc, vc, dc = map(chunks, (q, k, v, do))
    bc = beta.float().reshape(B, H, nc, C, 1)
    gram = kc @ _transpose(kc)
    power = -torch.tril(bc * gram, diagonal=-1)
    eye = torch.eye(C, dtype=torch.float32, device=q.device)
    inverse = eye + power
    degree = 2
    while degree < C:
        power = power @ power
        inverse = inverse @ (eye + power)
        degree *= 2
    wc, uc = inverse @ (bc * kc), inverse @ (bc * vc)
    before = S[:, :, :nc].float()
    residual = uc - wc @ before
    attn = torch.tril(qc @ _transpose(kc))
    da = torch.tril(dc @ _transpose(residual))
    dq = dc @ _transpose(before) + da @ kc
    dk = _transpose(da) @ qc
    dr_local = _transpose(attn) @ dc
    drs, dks = [], []
    adj = torch.zeros_like(before[:, :, 0])
    for c in reversed(range(nc)):
        dr = dr_local[:, :, c] + kc[:, :, c] @ adj
        dks.append(dk[:, :, c] + residual[:, :, c] @ _transpose(adj))
        drs.append(dr)
        adj = adj + _transpose(qc[:, :, c]) @ dc[:, :, c] - _transpose(wc[:, :, c]) @ dr
    dr = torch.stack(drs[::-1], 2)
    dk = torch.stack(dks[::-1], 2)
    dw = -dr @ _transpose(before)
    dyw, dyu = _transpose(inverse) @ dw, _transpose(inverse) @ dr
    dl = -torch.tril(dyw @ _transpose(wc) + dyu @ _transpose(uc), diagonal=-1)
    weighted = bc * dl
    dk = dk + bc * dyw + (weighted + _transpose(weighted)) @ kc
    dv = bc * dyu
    db = (dyw * kc).sum(-1) + (dyu * vc).sum(-1) + (dl * gram).sum(-1)
    return (
        dq.reshape_as(q).to(q.dtype),
        dk.reshape_as(k).to(k.dtype),
        dv.reshape_as(v).to(v.dtype),
        db.reshape_as(beta).to(beta.dtype),
    )


def gla_chunk_bwd(q, k, v, g, h, do, dht, *, chunk_size=64, scale=-1.0, has_initial_state=False):
    """BTHK -> four FP32 gradients, including nonzero dht and initial state.

    Each chunk's states are recovered by an affine scan. Reverse scans propagate
    output and final-state adjoints without an autograd tape or a T*K*V allocation.
    """
    B, length, H, K = q.shape
    if length % chunk_size:
        raise ValueError("Sequence length must be divisible by chunk_size")
    sc = K**-0.5 if scale is None or scale <= 0 else scale
    qq, kk, vv, gg, dd = (x.float().transpose(1, 2) for x in (q, k, v, g, do))
    gradients = [torch.empty_like(x) for x in (qq, kk, vv, gg)]
    adj = dht.float()
    for c in reversed(range(length // chunk_size)):
        start, end = c * chunk_size, (c + 1) * chunk_size
        qc, kc, vc, dc = (x[:, :, start:end] for x in (qq, kk, vv, dd))
        decay = gg[:, :, start:end].exp()  # B,H,C,K
        # Move time next to the last dimension for the shared affine scan.
        a = decay.transpose(-1, -2).unsqueeze(-1)  # B,H,K,C,1
        kv = (kc[..., None] * vc[..., None, :]).transpose(-3, -2)
        prefix, summed = _affine_scan(a, kv)
        initial = h[:, c].float()
        state = (summed + prefix * initial.unsqueeze(-2)).transpose(-3, -2)
        previous = torch.cat((initial.unsqueeze(2), state[:, :, :-1]), 2)
        local = (sc * qc[..., None] * dc[..., None, :]).transpose(-3, -2)
        next_decay = torch.cat((decay[:, :, 1:], torch.ones_like(decay[:, :, :1])), 2)
        rev_a = next_decay.transpose(-1, -2).unsqueeze(-1).flip(-2)
        rev_p, rev_b = _affine_scan(rev_a, local.flip(-2))
        adjoints = (rev_b + rev_p * adj.unsqueeze(-2)).flip(-2).transpose(-3, -2)
        values = (
            sc * (state * dc[..., None, :]).sum(-1),
            (adjoints * vc[..., None, :]).sum(-1),
            (adjoints * kc[..., None]).sum(-2),
            (adjoints * previous).sum(-1) * decay,
        )
        for dest, value in zip(gradients, values, strict=True):
            dest[:, :, start:end] = value
        adj = adjoints[:, :, 0] * decay[:, :, 0, :, None]
    return tuple(x.transpose(1, 2).contiguous() for x in gradients)


def selective_scan(
    u,
    delta,
    A,
    B_in,
    C_in,
    D_in=None,
    z=None,
    delta_bias=None,
    seq_idx=None,
    initial_states=None,
    *,
    delta_softplus=True,
):
    batch, channels, length = u.shape
    groups, states = B_in.shape[1:3]
    heads = channels // groups
    source = u.float().reshape(batch, groups, heads, length)
    dt = delta.float().reshape_as(source)
    if delta_bias is not None:
        dt = dt + delta_bias.float().reshape(1, groups, heads, 1)
    if delta_softplus:
        dt = F.softplus(dt)
    decay_rate = A.float().reshape(1, groups, heads, 1, states)
    state = (
        torch.zeros(batch, groups, heads, states, device=u.device)
        if initial_states is None
        else initial_states.float().reshape(batch, groups, heads, states)
    )
    output = torch.empty_like(source)
    # The largest grouped manifest case stays below ~256 MB scratch per scan
    # stage; no full [B,D,L,N] state history is retained.
    for start in range(0, length, 32):
        end = min(start + 32, length)
        steps = dt[..., start:end, None]
        a = torch.exp(steps * decay_rate)
        if seq_idx is not None:
            positions = torch.arange(start, end, device=u.device)
            prior = (positions - 1).clamp_min(0)
            reset = (seq_idx[:, start:end] != seq_idx.index_select(1, prior)) & (positions > 0)
            a = torch.where(reset[:, None, None, :, None], torch.zeros_like(a), a)
        b = (
            steps
            * source[..., start:end, None]
            * B_in[..., start:end].float().transpose(-1, -2)[:, :, None]
        )
        prefix, accumulated = _affine_scan(a, b)
        history = accumulated + prefix * state.unsqueeze(-2)
        cc = C_in[..., start:end].float().transpose(-1, -2)[:, :, None]
        output[..., start:end] = (history * cc).sum(-1)
        state = history[..., -1, :]
    output = output.reshape(batch, channels, length)
    if D_in is not None:
        output = output + u.float() * D_in.float()[None, :, None]
    if z is not None:
        output = output * F.silu(z.float())
    return output.to(u.dtype), state.reshape(batch, channels, states).contiguous()
