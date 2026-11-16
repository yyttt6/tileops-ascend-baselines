"""Complete original KDA entries for KDA, scalar-decay GDN and DeltaNet."""
from pathlib import Path
from .runtime import TRITON_ROOT, candidate, kda_module, module, validate


def linear_attention(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    native, sources = kda_module()
    p, op = case.params, case._call.signature.name
    raw = p.get("use_gate_in_kernel", False)
    beta_sigmoid = p.get("use_beta_sigmoid_in_kernel", False)
    if raw or beta_sigmoid:
        gating, gate_sources = module(TRITON_ROOT + "fused_gdn_gating.py", hardware=True)
        sources += gate_sources
    if p.get("allow_neg_eigval", False) and beta_sigmoid:
        multiply, mul_sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
        sources += mul_sources

    def run(q, k, v, *args):
        decode_op = op == "DeltaNetRecurrentFwdOp"
        if decode_op:
            beta, initial = args
            q, k, v = (t.unsqueeze(1) for t in (q, k, v))
            beta = beta.unsqueeze(1)
            gate, offsets, offsets_cpu, a_log, dt_bias = None, None, None, None, None
        elif op == "DeltaNetFwdOp":
            beta, initial, offsets, offsets_cpu = args
            gate, a_log, dt_bias = None, None, None
        else:
            gate, beta, initial, offsets, offsets_cpu, a_log, dt_bias = args
        b, t, qh, kd = q.shape
        vh, vd = v.shape[2:]
        original_shape = v.shape
        dim = max(kd, vd)
        # Repeat shared Q/K heads as layout materialization, not a new recurrence.
        repeats = vh // qh
        if repeats != 1:
            q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
        # The native output buffer has K's shape; zero features extend K/V to a
        # common width while preserving all dot products and the original scale.
        q, k = (F.pad(x, (0, dim-kd)) for x in (q, k))
        v = F.pad(v, (0, dim-vd))
        if raw and op == "KDAFwdOp":
            gate = native.fused_kda_gate(gate.reshape(b*t, vh*kd).float(), a_log.float(), kd,
                                         g_bias=None if dt_bias is None else dt_bias.float()).reshape(b, t, vh, kd)
        if (raw and op == "GDNFwdOp") or beta_sigmoid:
            a = gate.reshape(b*t, vh).float() if raw and op == "GDNFwdOp" else torch.zeros((b*t, vh), device=q.device)
            alog = a_log.float() if raw and op == "GDNFwdOp" else torch.zeros(vh, device=q.device)
            bias = dt_bias.float() if raw and op == "GDNFwdOp" and dt_bias is not None else torch.zeros(vh, device=q.device)
            computed_gate, computed_beta = gating.fused_gdn_gating_patch(alog, a, beta.reshape(b*t, vh).float(), bias)
            if raw and op == "GDNFwdOp":
                gate = computed_gate.reshape(b, t, vh)
            if beta_sigmoid:
                beta = computed_beta.reshape(b, t, vh)
                if p.get("allow_neg_eigval", False):
                    bp = beta.reshape(-1, 1)
                    beta = multiply.muls_add_triton(bp, torch.zeros_like(bp), 2.0).reshape(b, t, vh)
        if gate is None:
            gate = torch.zeros((b, t, vh, dim), device=q.device, dtype=torch.float32)
        elif op == "GDNFwdOp":
            gate = gate.unsqueeze(-1).expand(b, t, vh, dim).contiguous()
        else:
            gate = F.pad(gate, (0, dim-kd))
        supplied = offsets if offsets is not None else offsets_cpu
        boundaries = (torch.arange(b + 1, device=q.device, dtype=torch.int64) * t if supplied is None
                      else supplied.to(device=q.device, dtype=torch.int64).clone())
        # Clone metadata every call: original tensor_cache is identity based;
        # changed caller offsets must never reuse an input-dependent result.
        sequences = boundaries.numel() - 1
        state_v_first = p.get("state_v_first", False)
        if initial is None:
            state = torch.zeros((sequences, vh, dim, dim), device=q.device, dtype=torch.float32)
        else:
            state = initial if state_v_first else initial.transpose(-1, -2)
            state = F.pad(state.float(), (0, dim-kd, 0, dim-vd)).clone().contiguous()
        # Known native L2 tile requirement is accommodated by independent heads.
        padded_heads = 1 << (vh-1).bit_length() if p.get("use_qk_l2norm_in_kernel", False) else vh
        if padded_heads != vh:
            q, k, v, gate = (F.pad(x, (0, 0, 0, padded_heads-vh)) for x in (q, k, v, gate))
            beta = F.pad(beta, (0, padded_heads-vh))
            state = F.pad(state, (0, 0, 0, 0, 0, padded_heads-vh))
        qp, kp = (x.reshape(1, b*t, padded_heads, dim).contiguous() for x in (q, k))
        vp = v.reshape(1, b*t, padded_heads, dim).clone().contiguous()
        gp = gate.reshape(1, b*t, padded_heads, dim).contiguous()
        bp = beta.reshape(1, b*t, padded_heads).contiguous()
        scale = 1.0 if decode_op else (p.get("scale") if p.get("scale") is not None else kd ** -0.5)
        kwargs = dict(scale=scale, initial_state=state, cu_seqlens=boundaries,
                      use_qk_l2norm_in_kernel=p.get("use_qk_l2norm_in_kernel", False))
        if t == 1 and supplied is None:
            out, final = native.fused_recurrent_kda(qp, kp, vp, gp, bp,
                inplace_final_state=False, sigmoid_beta=False, compute_gate=False, **kwargs)
        else:
            out, final = native.chunk_kda(qp, kp, vp, gp, bp, output_final_state=True, **kwargs)
        out = out.reshape(b, t, padded_heads, dim)[:, :, :vh, :vd].contiguous()
        final = final[:, :vh, :vd, :kd].contiguous()
        if not state_v_first:
            final = final.transpose(-1, -2).contiguous()
        if decode_op:
            return out.squeeze(1), final.to(q.dtype)
        return out.reshape(original_shape), final

    return candidate(run, "kda.chunk_kda / fused_recurrent_kda", "Complete original KDA recurrence; scalar GDN gate broadcast over key channels; zero DeltaNet gate. Replicate shared Q/K heads, zero-pad unequal K/V widths, preserve original scale and state layout, clone mutable value/state/metadata, return every output. Raw gates and sigmoid use original native gating kernels, never Torch arithmetic.", [Path(__file__), *sources], limitations=("Native chunk and recurrent algorithms and launch settings unchanged; full final-state and sequence-boundary validation still required.",), timing_mode="host_completed")
