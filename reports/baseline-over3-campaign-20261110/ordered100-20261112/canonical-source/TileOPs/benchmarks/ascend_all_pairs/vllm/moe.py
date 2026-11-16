"""Expert matmuls and MLPs using only owned GEMM and SiLU kernels."""
from pathlib import Path
from .runtime import TRITON_ROOT, candidate, module, validate


def expert(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    mm, sources = module(TRITON_ROOT + "batch_invariant/matmul.py")
    op = case._call.signature.name
    activation = case.params.get("activation")
    mlp = op in ("MoEExpertMLPFwdOp", "SharedExpertMLPFwdOp")
    if mlp or activation is not None:
        act, act_sources = module(TRITON_ROOT + "activation/swiglustep.py", hardware=True)
        sources += act_sources
    layout = None if op == "SharedExpertMLPFwdOp" else case._call.params["layout"]

    def silu(x):
        g, u = x.chunk(2, dim=-1)
        count = g.numel()
        pad = (-count) % 16
        gp, up = (F.pad(t.contiguous().reshape(-1), (0, pad)).reshape(-1, 16) for t in (g, u))
        return act.swiglustep_forward_triton(torch.cat((gp, up), -1), limit=float("inf")).reshape(-1)[:count].reshape(g.shape)

    def grouped(a, b, metadata, activate=False, out_dtype=None):
        values = metadata.tolist()
        dim_out = b.shape[1] // (2 if activate else 1)
        result = torch.zeros((*a.shape[:-1], dim_out), dtype=out_dtype or a.dtype, device=a.device)
        for group in range(b.shape[0]):
            if layout.kind == "masked":
                index = (group, slice(0, values[group]))
            elif layout.metadata_kind == "per_row":
                # Index selection is data movement; metadata is not cached.
                index = torch.tensor([i for i, g in enumerate(values) if g == group], device=a.device, dtype=torch.int64)
            else:
                start = 0 if group == 0 else values[group-1]
                start = ((start + layout.alignment - 1) // layout.alignment) * layout.alignment
                index = slice(start, values[group])
            packed = a[index].float().contiguous()
            if packed.numel() == 0:
                continue
            out = mm.matmul_persistent(packed, b[group].float().transpose(0, 1).contiguous())
            if activate:
                out = silu(out)
            result[index] = out.to(result.dtype)
        return result

    def run(a, b, *args):
        if op == "SharedExpertMLPFwdOp":
            down, = args
            up = mm.matmul_persistent(a.float().contiguous(), b.float().transpose(0, 1).contiguous())
            return mm.matmul_persistent(silu(up), down.float().transpose(0, 1).contiguous()).to(a.dtype)
        if mlp:
            down, metadata = args
            # Match the manifest's dtype boundary between the two grouped GEMMs.
            up = grouped(a, b, metadata, activate=True)
            return grouped(up, down, metadata)
        metadata, = args
        dtype = case.params.get("out_dtype")
        return grouped(a, b, metadata, activate=activation is not None,
                       out_dtype=None if dtype is None else getattr(torch, dtype))

    return candidate(run, "matmul_persistent + swiglustep_forward_triton" if mlp or activation else "matmul_persistent (expert groups)", "Original GEMM per current expert segment or gathered row group; exact FP32 widening and manifest output dtype boundary. Optional native unbounded SiLU gate/up epilogue; original second GEMM for MLP. Metadata reads and all packing are inside run.", [Path(__file__), *sources], timing_mode="host_completed")
