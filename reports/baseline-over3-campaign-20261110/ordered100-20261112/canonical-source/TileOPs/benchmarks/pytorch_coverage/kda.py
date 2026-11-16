"""Device PyTorch KDA, with bounded FP32 workspaces and an independent FP64 oracle.

Only the short integer packed-sequence offsets may be read on the host. All
activation, gate, attention and recurrent-state arithmetic stays on the input
device. No input is mutated. The chunk inverse uses the finite nilpotent series
of a strictly lower triangular matrix, so no device triangular solver is needed.
"""

import torch
import torch.nn.functional as F

OPS = {"KDAFwdOp"}
MUTATED = {}


def _options(case):
    p = case._call.indices
    return dict(
        scale=p["scale"],
        raw_gate=p["use_gate_in_kernel"],
        lower_bound=p["lower_bound"],
        beta_sigmoid=p["use_beta_sigmoid_in_kernel"],
        allow_neg_eigval=p["allow_neg_eigval"],
        l2norm=p["use_qk_l2norm_in_kernel"],
        state_v_first=p["state_v_first"],
    )


def bind(case):
    options = _options(case)

    def run(*inputs):
        return kda(*inputs, **options)

    return run


def _sequences(q, cu_seqlens, cu_seqlens_cpu):
    if cu_seqlens is None:
        return [(b, 0, q.shape[1]) for b in range(q.shape[0])]
    # Scheduling metadata only; prefer the explicit host copy when provided.
    bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
    return [(0, a, b - a) for a, b in zip(bounds[:-1], bounds[1:])]


def kda(
    q, k, v, g, beta, initial_state=None, cu_seqlens=None,
    cu_seqlens_cpu=None, a_log=None, dt_bias=None, *, scale=None,
    raw_gate=False, lower_bound=None, beta_sigmoid=False,
    allow_neg_eigval=False, l2norm=False, state_v_first=False,
):
    """Exact chunk algebra for the decayed delta recurrence, evaluated in FP32.

    Chunks have at most 16 tokens. Per-channel decay differences are formed one
    time diagonal at a time, never as [batch, heads, chunk, chunk, key_dim].
    In particular, raw gates never require exp(-cumulative_decay), which could
    overflow. Independent equal-length sequences share each state-scan step.
    """
    sequences = _sequences(q, cu_seqlens, cu_seqlens_cpu)
    h, key_dim, value_dim = v.shape[2], q.shape[-1], v.shape[-1]
    group = h // q.shape[2]
    scale = key_dim ** -0.5 if scale is None else scale
    output = torch.empty_like(v)
    if initial_state is None:
        final = torch.zeros(len(sequences), h, key_dim, value_dim,
                            dtype=torch.float32, device=q.device)
    else:
        final = initial_state.float()
        final = (final.transpose(-1, -2) if state_v_first else final).clone()
    by_length = {}
    for index, (_, _, length) in enumerate(sequences):
        by_length.setdefault(length, []).append(index)
    for length, indices in by_length.items():
        if length == 0:
            continue
        count = len(indices)
        chunk = min(16, length)
        # Bound all local fields to roughly 4M FP32 elements at K=128.
        slab_chunks = max(1, min(128, 32768 // (count * h * chunk)))
        live = final[indices].reshape(count * h, key_dim, value_dim)
        eye = torch.eye(chunk, dtype=torch.float32, device=q.device)
        strict = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)

        def gather(tensor, first, end, blocks):
            pieces = [tensor[sequences[i][0], sequences[i][1] + first:
                             sequences[i][1] + end] for i in indices]
            x = torch.stack(pieces).float()
            if x.shape[1] < blocks * chunk:
                pad = x.new_zeros((count, blocks * chunk - x.shape[1], *x.shape[2:]))
                x = torch.cat((x, pad), dim=1)
            return x

        def blocked(x, blocks):
            # chunk-major makes each state-scan operand contiguous.
            return x.reshape(count, blocks, chunk, h, -1).permute(
                1, 0, 3, 2, 4).reshape(blocks, count * h, chunk, -1).contiguous()

        for first in range(0, length, slab_chunks * chunk):
            end = min(length, first + slab_chunks * chunk)
            blocks = (end - first + chunk - 1) // chunk
            qc = gather(q, first, end, blocks).repeat_interleave(group, dim=2)
            kc = gather(k, first, end, blocks).repeat_interleave(group, dim=2)
            if l2norm:
                qc = qc * torch.rsqrt(qc.square().sum(-1, keepdim=True) + 1e-6)
                kc = kc * torch.rsqrt(kc.square().sum(-1, keepdim=True) + 1e-6)
            qc, kc = blocked(qc * scale, blocks), blocked(kc, blocks)
            vc = blocked(gather(v, first, end, blocks), blocks)
            step = gather(beta, first, end, blocks)
            gc = gather(g, first, end, blocks)
            if raw_gate:
                x = gc + (0.0 if dt_bias is None else dt_bias.float().view(h, key_dim))
                rate = a_log.float().exp()[:, None]
                gc = (lower_bound * torch.sigmoid(rate * x) if lower_bound is not None
                      else -rate * F.softplus(x))
            if beta_sigmoid:
                step = torch.sigmoid(step) * (2.0 if allow_neg_eigval else 1.0)
            # Raw transforms must not turn zero padding into real updates/decay.
            if end - first < blocks * chunk:
                gc[:, end - first:] = 0
                step[:, end - first:] = 0
            step = blocked(step[..., None], blocks)
            decay = blocked(gc, blocks).cumsum(-2)
            del gc
            system = qc.new_zeros((blocks, count * h, chunk, chunk))
            attention = torch.empty_like(system)
            attention.zero_()
            for lag in range(chunk):
                # Every used exponent is a forward-time log decay (<= 0).
                weight = (decay[..., lag:, :] - decay[..., :chunk - lag, :]).exp()
                keys = weight * kc[..., :chunk - lag, :]
                system.diagonal(-lag, -2, -1).copy_((kc[..., lag:, :] * keys).sum(-1))
                attention.diagonal(-lag, -2, -1).copy_((qc[..., lag:, :] * keys).sum(-1))
            system = (system * step).masked_fill(~strict, 0)
            # (I + L)^-1 = I - L + L^2 - ...; L^chunk = 0.
            power = -system
            inverse = eye + power
            degree = 2
            while degree < chunk:
                power = power @ power
                inverse = inverse + power @ inverse
                degree *= 2
            d = decay.exp()
            u = inverse @ (step * vc)
            w = inverse @ (step * kc * d)
            carry_keys = kc * (decay[..., -1:, :] - decay).exp()
            local_output = attention @ u
            state_read = qc * d - attention @ w
            slab_output = torch.empty_like(vc)
            for block in range(blocks):
                written = u[block] - torch.bmm(w[block], live)
                slab_output[block] = local_output[block] + torch.bmm(state_read[block], live)
                live = (d[block, :, -1, :, None] * live +
                        torch.bmm(carry_keys[block].transpose(-1, -2), written))
            restored = slab_output.reshape(blocks, count, h, chunk, value_dim).permute(
                1, 0, 3, 2, 4).reshape(count, blocks * chunk, h, value_dim)
            for slot, index in enumerate(indices):
                row, start, _ = sequences[index]
                output[row, start + first:start + end] = restored[slot, :end - first].to(v.dtype)
        final[indices] = live.reshape(count, h, key_dim, value_dim)
    return output, final.transpose(-1, -2) if state_v_first else final


def cpu_reference(case, inputs):
    """Original independent FP64 recurrence; head slices bound its gate workspace.

    Slicing complete query-head groups changes neither recurrence order nor
    sequence length, and retains the original FP64 state throughout each call.
    """
    from workloads.linear_attention.kda import kda_ref

    q, k, v, g, beta, initial, cu, cu_cpu, a_log, bias = inputs
    options = _options(case)
    group = v.shape[2] // q.shape[2]
    q_heads_per_call = max(1, 4 // group)
    result = torch.empty_like(v)
    states = []
    for start in range(0, q.shape[2], q_heads_per_call):
        end = min(q.shape[2], start + q_heads_per_call)
        lo, hi = start * group, end * group
        out, state = kda_ref(
            q[:, :, start:end], k[:, :, start:end], v[:, :, lo:hi],
            g[:, :, lo:hi], beta[:, :, lo:hi],
            None if initial is None else initial[:, lo:hi], cu, cu_cpu,
            None if a_log is None else a_log[lo:hi],
            None if bias is None else bias.view(v.shape[2], -1)[lo:hi].reshape(-1),
            **options,
        )
        result[:, :, lo:hi] = out
        states.append(state)
    return result, torch.cat(states, dim=1)
