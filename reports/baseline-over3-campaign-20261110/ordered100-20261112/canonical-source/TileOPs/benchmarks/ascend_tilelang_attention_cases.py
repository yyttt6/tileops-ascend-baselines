"""Original BF16 fixed128 attention with exact feature padding adapters."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes

def tilelang_gqa_cases(case):
    import hashlib
    import importlib.util
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import WORKSPACE

    q, k, v, *extra = case.inputs
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise UnsupportedCase(
            "Original attention fixes BF16 input and probability storage; "
            "FP16 requires modifying the kernel's internal precision.")
    batch, sq, hq, dim = q.shape
    _, sk, hk, _ = k.shape
    p = case.params
    group = hq // hk
    if (any(t is not None for t in extra)
            or sq != 1 or dim not in (64, 128) or hq % hk or group > 128 or sk % 128
            or p.get("softcap") is not None or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1):
        raise UnsupportedCase("Original full-context BF16 decode; native fixed128 features allow exact zero padding of64; no new causal or softcap algorithm.")
    tilelang = _runtime()
    original = LIBRARIES / "tilelang/examples/ascend/flash_attention/core.py"
    # Preserve the complete original kernel, including probability rounding.
    source = original.read_text()
    dtype = str(q.dtype).split(".")[-1]
    adapted = source
    key = hashlib.sha256(adapted.encode()).hexdigest()
    cached = WORKSPACE / ".cache/ascend-classic" / ("tilelang-gqa-cases-" + key[:20])
    cached.mkdir(parents=True, exist_ok=True)
    path = cached / "core.py"
    if path.exists() and path.read_text() != adapted:
        raise RuntimeError("Cached attention source differs from the original.")
    path.write_text(adapted)
    name = "_classic_tilelang_gqa_" + key[:20]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    program = module.flash_attention_fwd(
        128, sk, head_dim=128, out_dtype=dtype,
        softmax_scale=p.get("sm_scale") if p.get("sm_scale") is not None else dim ** -0.5, num_blocks=1)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)

    def run(q, k, v, *extra):
        out = torch.empty_like(q)
        for b in range(batch):
            for head in range(hk):
                queries = q[b, 0, head * group:(head + 1) * group]
                queries = F.pad(queries, (0, 128 - dim, 0, 128 - group))
                values = kernel(queries, F.pad(k[b, :, head].contiguous(), (0, 128 - dim)),
                                F.pad(v[b, :, head].contiguous(), (0, 128 - dim)))
                out[b, 0, head * group:(head + 1) * group].copy_(values[:group, :dim])
        return out

    return Candidate(
        Implementation(run), "tilelang::flash_attention_fwd (original BF16)",
        "Unchanged original BF16 kernel and internal probability precision; per-batch/KV-head calls, independent query padding, exact feature zero padding, original scale and output crop.",
        source_hashes(Path(__file__), original, path) | _runtime_hashes(tilelang),
        limitations=("Host loop launches each KV-head group separately; full adapter cost belongs in timing.",),
    )
