"""Original tilelang-ascend varlen GQA with padded decode queries."""
import importlib.util
from pathlib import Path
import sys

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _runtime


def tla_gqa(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    runtime = _runtime()
    q, k, v, *extra = case.inputs
    b, sq, hq, dim = q.shape
    sk, hk = k.shape[1:3]
    params = case.params
    if (any(t is not None for t in extra) or sq != 1 or dim != 128
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or k.shape != v.shape or hq % hk or sk % 128
            or params.get("softcap") is not None
            or params.get("sm_scale") not in (None, dim ** -0.5)
            or params.get("pos_encoding_mode", "none") != "none"
            or params.get("window_size_left", -1) != -1
            or params.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This sample is full-context FP16 decode with head dimension128.")
    source = LIBRARIES / "tilelang-ascend/examples/gqa_fwd_varlen/gqa_fwd_varlen.py"
    name = "_classic_tla_original_gqa"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    builder = runtime.jit(
        out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=module.PASS_CONFIGS,
        target="pto",
    )(module.flashattn.__wrapped__)
    # Keep all original tile, pipeline, core-distribution and mask-path defaults.
    kernel = builder(b, hq // hk, hq, dim, 128, sk, False)

    def run(q, k, v, *extra):
        query = F.pad(q.permute(0, 2, 1, 3).contiguous(), (0, 0, 0, 127))
        key, value = k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
        # Bottom-right causal alignment for one query exposes every KV token.
        mask = torch.ones((b, 128, sk), dtype=torch.float32, device=q.device)
        out = kernel(query, key, value, mask)
        return out[:, :, :1, :].permute(0, 2, 1, 3).contiguous()

    return Candidate(
        Implementation(run), "tilelang-ascend::gqa_fwd_varlen.flashattn",
        "Original PTO backend and schedule; BSND->BHSD, zero-pad Q to128, all-visible decode mask, crop first query.",
        source_hashes(Path(__file__), source,
                      Path(__file__).with_name("ascend_tla_baselines.py")),
    )
