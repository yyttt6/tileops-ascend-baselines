"""Device PyTorch Gated DeltaNet, with an independent workload CPU oracle.

All recurrence arithmetic is FP32.  A finite block inverse of the unit lower
triangular chunk system avoids both FP64 and backend-specific triangular solves.
Only packed integer sequence boundaries may be copied to the host for scheduling.
"""

import functools

import torch
import torch.nn.functional as F

OPS = {"GDNFwdOp"}
MUTATED = {}


def _options(case):
    p = case._call.indices
    return dict(
        scale=p["scale"],
        raw_gate=p["use_gate_in_kernel"],
        beta_sigmoid=p["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=p["allow_neg_eigval"],
        l2norm=p["use_qk_l2norm_in_kernel"],
        state_v_first=p["state_v_first"],
    )


def bind(case):
    return functools.partial(gdn, **_options(case))


def cpu_reference(case, inputs):
    from workloads.linear_attention.gdn import gdn_ref

    return gdn_ref(*inputs, **_options(case))


def _inverse(system, block_indices):
    """Exact block recursion: inv([[A,0],[C,D]]) has -inv(D) C inv(A).

    Unlike a truncated Neumann series, this includes every strictly lower
    contribution and remains stable for correlated normalized keys.
    """
    n = system.shape[-1]
    inverse = torch.ones(
        (*system.shape[:-2], n, 1, 1), dtype=system.dtype, device=system.device
    )
    for rows, columns in block_indices:
        a, d = inverse[..., 0::2, :, :], inverse[..., 1::2, :, :]
        c = system[..., rows, columns]
        lower = -(d @ c) @ a
        inverse = torch.cat(
            (torch.cat((a, torch.zeros_like(a)), -1), torch.cat((lower, d), -1)), -2
        )
    return inverse[..., 0, :, :]


def gdn(
    q, k, v, g, beta, initial_state=None, cu_seqlens=None,
    cu_seqlens_cpu=None, A_log=None, dt_bias=None, *, scale=None,
    raw_gate=False, beta_sigmoid=False, allow_neg_eigval=False,
    l2norm=False, state_v_first=False,
):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        rows = list(range(q.shape[0]))
        starts, lengths = [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    else:
        # Host metadata is limited to integer sequence offsets, never tensor values.
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        rows, starts = [0] * (len(bounds) - 1), bounds[:-1]
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:])]
    order = sorted(range(len(lengths)), key=lambda s: -lengths[s])
    longest = max(lengths, default=0)
    device = q.device
    row = torch.tensor([rows[s] for s in order], dtype=torch.long, device=device)
    start = torch.tensor([starts[s] for s in order], dtype=torch.long, device=device)
    length = torch.tensor([lengths[s] for s in order], dtype=torch.long, device=device)
    if initial_state is None:
        state = torch.zeros(
            len(order), v.shape[2], q.shape[-1], v.shape[-1],
            dtype=torch.float32, device=device,
        )
    else:
        state = initial_state[order].float().clone()
        if state_v_first:
            state = state.transpose(-1, -2).contiguous()
    output = torch.empty_like(v)
    chunk = 1 if longest <= 1 else 64
    offset = torch.arange(chunk, device=device)
    inclusive = torch.ones(chunk, chunk, dtype=torch.bool, device=device).tril()
    strict = inclusive.tril(-1)
    block_indices = []
    for half in (1, 2, 4, 8, 16, 32):
        base = torch.arange(0, chunk, 2 * half, device=device)[:, None, None]
        local = torch.arange(half, device=device)
        block_indices.append((base + half + local[None, :, None], base + local[None, None, :]))
    running = len(order)
    for first in range(0, longest, chunk):
        while running and lengths[order[running - 1]] <= first:
            running -= 1
        r = row[:running, None]
        valid = first + offset < length[:running, None]
        t = (start[:running, None] + first + offset).clamp(max=q.shape[1] - 1)
        mask = valid.float()[..., None]
        qc = q[r, t].float().repeat_interleave(group, dim=2)
        kc = k[r, t].float().repeat_interleave(group, dim=2)
        if l2norm:
            qc = qc * torch.rsqrt(qc.square().sum(-1, keepdim=True) + 1e-6)
            kc = kc * torch.rsqrt(kc.square().sum(-1, keepdim=True) + 1e-6)
        gc, bc = g[r, t].float(), beta[r, t].float()
        if raw_gate:
            gc = -A_log.exp() * F.softplus(gc + dt_bias)
        if beta_sigmoid:
            bc = bc.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
        live = state[:running]
        if chunk == 1:
            qt, kt = qc[:, 0] * scale, kc[:, 0]
            decay = gc[:, 0].exp()
            old_value = (live * kt[..., None]).sum(-2)
            written = bc[:, 0, :, None] * (v[r, t][:, 0].float() - decay[..., None] * old_value)
            live = decay[..., None, None] * live + kt[..., None] * written[..., None, :]
            out = (qt[..., None] * live).sum(-2)[:, None]
        else:
            qc = (qc * scale).transpose(1, 2)
            kc = (kc * mask[..., None]).transpose(1, 2)
            vc = (v[r, t].float() * mask[..., None]).transpose(1, 2)
            logs = (gc * mask).transpose(1, 2).cumsum(-1)
            step = (bc * mask).transpose(1, 2)
            gap = logs[..., :, None] - logs[..., None, :]
            decay = logs.exp()[..., None]
            system = step[..., None] * gap.masked_fill(~strict, -torch.inf).exp() * (kc @ kc.transpose(-1, -2))
            rhs = step[..., None] * (vc - decay * (kc @ live))
            written = _inverse(system, block_indices) @ rhs
            attention = gap.masked_fill(~inclusive, -torch.inf).exp() * (qc @ kc.transpose(-1, -2))
            out = (decay * (qc @ live) + attention @ written).transpose(1, 2)
            carry = (logs[..., -1:] - logs).exp()[..., None]
            live = logs[..., -1].exp()[..., None, None] * live + (kc * carry).transpose(-1, -2) @ written
        state[:running] = live
        output[r.expand_as(t)[valid], t[valid]] = out[valid].to(v.dtype)
    final = torch.empty_like(state)
    final[order] = state
    if state_v_first:
        final = final.transpose(-1, -2)
    return output, final
