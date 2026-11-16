"""Native activation specializations; no Torch arithmetic substitute."""
from pathlib import Path
from .runtime import TRITON_ROOT, candidate, module, validate


def activation(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    op = case._call.signature.name
    if op in ("SiluFwdOp", "SiluAndMulFwdOp"):
        native, sources = module(TRITON_ROOT + "activation/swiglustep.py", hardware=True)

        def run(x):
            shape = x.shape
            if op == "SiluAndMulFwdOp":
                gate, up = x.chunk(2, dim=-1)
                shape = gate.shape
            else:
                gate, up = x, torch.ones_like(x)
            # Elementwise op: independent 16-column rows satisfy alignment and
            # avoid non-power-of-two full FFN width without changing schedule.
            elements = gate.numel()
            pad = (-elements) % 16
            g = F.pad(gate.contiguous().reshape(-1), (0, pad)).reshape(-1, 16)
            u = F.pad(up.contiguous().reshape(-1), (0, pad)).reshape(-1, 16)
            out = native.swiglustep_forward_triton(torch.cat((g, u), -1), limit=float("inf"))
            return out.reshape(-1)[:elements].reshape(shape)

        return candidate(run, "swiglustep_forward_triton(limit=inf)", "Disable optional finite clamp through original limit API; split/repack 16-element independent gate/up rows, pad/crop tails; unit up for SiLU.", [Path(__file__), *sources])

    native, sources = module(TRITON_ROOT + "fused_gdn_gating.py", hardware=True)
    if op == "SoftplusFwdOp":
        negate, neg_sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
        sources += neg_sources

    def run(x):
        xp = x.contiguous().reshape(-1, 1).float()
        zeros = torch.zeros(1, device=x.device, dtype=torch.float32)
        # The selected output is original native sigmoid or negative softplus.
        g, beta = native.fused_gdn_gating_patch(zeros, xp, xp, zeros,
                                               beta=case.params.get("beta", 1.0),
                                               threshold=case.params.get("threshold", 20.0))
        if op == "SigmoidFwdOp":
            out = beta
        else:
            out = negate.muls_add_triton(g.reshape(-1, 1), torch.zeros_like(xp), -1.0)
        return out.to(x.dtype).reshape(x.shape)

    return candidate(run, "fused_gdn_gating_patch" + (" + muls_add_triton" if op == "SoftplusFwdOp" else ""), "Exactly widen input to FP32, select native sigmoid output or negate native -softplus output using original native scalar-add API; restore dtype/layout. No Torch transcendental arithmetic.", [Path(__file__), *sources])
