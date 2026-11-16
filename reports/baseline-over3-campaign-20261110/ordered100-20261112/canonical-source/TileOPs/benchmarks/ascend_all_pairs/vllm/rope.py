"""Manifest table state and original native NeoX/interleaved rotation."""
from pathlib import Path
from .runtime import TRITON_ROOT, WORKSPACE, candidate, module, validate


def rope(case):
    validate(case)
    import torch
    # These helpers construct the cosine/sine API input state only. No reference
    # rotation/output is called. The same approved table route was used by classics.
    from workloads.rope import (rope_frequency_tables, llama31_frequency_tables,
                                yarn_frequency_tables, longrope_frequency_tables)
    native, sources = module(TRITON_ROOT + "rope.py", hardware=True)
    p, op = case.params, case._call.signature.name
    tables = {"RoPEFwdOp": rope_frequency_tables, "RoPELlama31FwdOp": llama31_frequency_tables,
              "YaRNFwdOp": yarn_frequency_tables, "LongRoPEFwdOp": longrope_frequency_tables}
    # LongRoPE factors are a constructor tensor, not a run argument. Keep the
    # original tensor identity only, and rebuild tables from its values each run.
    factors = case.workload.tensors.get("rescale_factors") if "rescale_factors" in case._call.tensors else None

    def run(x, position_ids=None):
        dim = p.get("rotary_dim") or x.shape[-1]
        if op == "RoPENeoxPositionIdsFwdOp":
            cos, sin = rope_frequency_tables(dim, p["max_position"], base=p["base"], dtype=x.dtype, device=x.device)
            private = x.clone().contiguous()
            return native.rope_forward_triton_siso(private,
                cos_sin_cache=torch.cat((cos, sin), -1), positions=position_ids.contiguous(),
                rope_dim=dim, is_neox_style=True)
        batch, seq, heads = (1, x.shape[0], 1) if x.ndim == 2 else x.shape[:3]
        kwargs = {k: v for k, v in case._call.params.items() if k not in ("input_layout", "rope_layout")}
        if factors is not None:
            kwargs["rescale_factors"] = factors
        cos, sin = tables[op](dim, seq, dtype=x.dtype, device=x.device, **kwargs)
        private = x.clone().contiguous().reshape(batch*seq, heads, x.shape[-1])
        result = native.rope_forward_triton_siso(private, cos=cos.repeat(batch, 1), sin=sin.repeat(batch, 1),
                    rope_dim=dim, is_neox_style=p.get("rope_layout", "neox") == "neox")
        return result.reshape(x.shape)

    return candidate(run, "rope_forward_triton_siso", "Manifest-defined frequency tables become original API state; native rotation alone computes every output. Preserve table rounding, layout, pairing, position IDs, partial rotation and constructor rescaling factors. Clone mutated input; table creation and selection are timed each run.", [Path(__file__), WORKSPACE / "TileOPs/workloads/rope.py", *sources], timing_mode="host_completed")
