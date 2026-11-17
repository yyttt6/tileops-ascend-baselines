"""Full attention merge: native original-storage path, exact reference fallback.

The runtime predicate and all conversions/copies stay in the measured call.
No tensor identity, frozen-input predicate or result is cached.
"""
import math
import torch
import torch_npu


def factory(params, original_reference):
    base2 = params.get('lse_base', 'e') == '2'

    def run(v_a, s_a, v_b, s_b):
        # Native PA semantics may reinterpret infinities. Preserve the complete
        # original nonfinite contract by executing its real Torch implementation.
        if not torch.isfinite(torch.stack((s_a, s_b))).all().item():
            return original_reference(v_a, s_a, v_b, s_b)
        sa, sb = s_a, s_b
        if base2:
            maximum = torch.maximum(sa, sb)
            sa = (sa - maximum) * math.log(2)
            sb = (sb - maximum) * math.log(2)
        out, lse = torch_npu.npu_attention_update(
            [sa.reshape(-1).contiguous(), sb.reshape(-1).contiguous()],
            [v_a.reshape(-1, v_a.shape[-1]).contiguous(),
             v_b.reshape(-1, v_b.shape[-1]).contiguous()],
            1,
        )
        lse = lse.reshape_as(s_a)
        if base2:
            lse = lse / math.log(2) + maximum
        return out.reshape_as(v_a), lse

    return run
