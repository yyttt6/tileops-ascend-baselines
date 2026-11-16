"""Original ops-transformer APIs with manifest shapes and optional flags."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_math_more import _MathCall


def _candidate(binding, run, directory, adaptation):
    from benchmarks.api import Implementation

    return Candidate(
        Implementation(run), "ops-transformer::" + binding.api, adaptation,
        source_hashes(Path(__file__)) |
        binding.hashes(LIBRARIES / "ops-transformer" / directory),
        limitations=("Requires isolated original vendor build and own device-kernel attribution.",),
        cleanup=binding.cleanup,
    )


def transformer_deltanet(case):
    import torch

    if case.inputs[5] is not None or case.inputs[6] is not None:
        raise UnsupportedCase("This sample uses dense batches.")
    if case.params.get("use_qk_l2norm_in_kernel"):
        raise UnsupportedCase("This sample preserves the manifest's unnormalized Q/K.")
    binding = _MathCall(
        "aclnnChunkKdaFwd",
        [C.c_void_p] * 10 + [C.c_char_p, C.c_double, C.c_int64, C.c_bool,
                             C.c_double, C.c_bool, C.c_bool] + [C.c_void_p] * 11,
        library="ops-transformer",
    )
    scale = case.params.get("scale")

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        b, s, h, dk = q.shape
        dv = v.shape[-1]
        packed = [t.contiguous() for t in (q, k, v)]
        gate = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        final = torch.empty((b, h, dk, dv), device=q.device, dtype=torch.float32)
        # Required intermediate outputs belong to the original public API.
        aqk = torch.empty((b, h, s, 64), dtype=q.dtype, device=q.device)
        akk = torch.empty_like(aqk)
        record = binding.record()
        ptr = lambda t: binding.tensor(t, record)
        binding.call([
            *[ptr(t) for t in packed], ptr(gate), ptr(beta.contiguous()),
            None, None, ptr(initial.contiguous()) if initial is not None else None,
            None, None, b"BSND",
            dk ** -0.5 if scale is None else scale, 64, False, -5.0, False, False,
            ptr(out), ptr(final), None, ptr(aqk), ptr(akk),
            None, None, None, None, None, None,
        ], record, q.device)
        return out, final

    return _candidate(binding, run, "attention/chunk_kda_fwd",
                      "Zero log gate reduces original KDA to DeltaNet; original64 chunk, BSND, FP32 final state.")


def transformer_rms(case):
    import torch

    binding = _MathCall(
        "aclnnNormRopeConcat",
        [C.c_void_p] * 16 + [C.c_int64] * 4 + [C.c_double, C.c_bool] + [C.c_void_p] * 11,
        library="ops-transformer",
    )
    epsilon = case.params["eps"]

    def run(x, weight):
        packed = x.contiguous().reshape(1, -1, 1, x.shape[-1])
        shape = (1, 1, packed.shape[1], x.shape[-1])
        qout, kout, vout = [torch.empty(shape, device=x.device, dtype=x.dtype) for _ in range(3)]
        record = binding.record()
        p = binding.tensor(packed, record)
        gamma = binding.tensor(weight.contiguous(), record)
        binding.call([
            p, p, p, None, None, None, gamma, None, gamma, None,
            None, None, None, None, None, None,
            4, 0, 0, 0, epsilon, False,
            binding.tensor(qout, record), binding.tensor(kout, record),
            binding.tensor(vout, record), *([None] * 8),
        ], record, x.device)
        return qout.reshape(x.shape)

    return _candidate(binding, run, "posembedding/norm_rope_concat",
                      "RMS_NORM_AFFINE, ropeNONE, no encoder; singleton batch/head and native Q-output selection.")


def transformer_prepermute(case):
    import torch

    layout = case.params["layout"].get("contiguous", {})
    if layout != {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}:
        raise UnsupportedCase("This pilot uses complete, tight expert-major routing.")
    binding = _MathCall(
        "aclnnMoeInitRoutingV2",
        [C.c_void_p] * 2 + [C.c_int64] * 5 + [C.c_bool] + [C.c_void_p] * 4,
        library="ops-transformer",
    )
    experts = case.params["num_local_experts"]

    def run(x, ids):
        n, hidden = x.shape
        routes = ids.numel()
        out = torch.empty((routes, hidden), dtype=x.dtype, device=x.device)
        inverse = torch.empty(routes, dtype=torch.int32, device=x.device)
        prefix = torch.empty(experts, dtype=torch.int32, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(x.contiguous(), record), binding.tensor(ids.contiguous(), record),
            routes, 0, experts, 0, 1, False,
            binding.tensor(out, record), binding.tensor(inverse, record),
            binding.tensor(prefix, record), None,
        ], record, x.device)
        # arch35 native expansion indexes token-major input routes.
        return out, prefix, inverse

    return _candidate(binding, run, "moe/moe_init_routing_v2",
                      "Dropless mode, allroutes active, native prefix counts and inverse route indices.")


def transformer_gqa(case):
    import torch

    q, k, v, *extra = case.inputs
    p = case.params
    if (any(t is not None for t in extra) or q.shape[1] != 1
            or p.get("softcap") is not None or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This sample is full-context, single-query decode.")
    binding = _MathCall(
        "aclnnFlashAttentionScore",
        [C.c_void_p] * 8 + [C.c_double] * 2 + [C.c_int64] * 3 + [C.c_char_p]
        + [C.c_int64] * 2 + [C.c_void_p] * 4, library="ops-transformer",
    )
    scale = p.get("sm_scale") or q.shape[-1] ** -0.5

    def run(q, k, v, *extra):
        b, sq, h, dim = q.shape
        out = torch.empty_like(q)
        maximum = torch.empty((b, h, sq, 8), dtype=torch.float32, device=q.device)
        total = torch.empty_like(maximum)
        record = binding.record()
        ptr = lambda t: binding.tensor(t.contiguous(), record)
        # The sole decode query is the final position: all KV are visible.
        binding.call([
            ptr(q), ptr(k), ptr(v), None, None, None, None, None,
            scale, 1.0, 2147483647, 2147483647, h, b"BSND", 0, 0,
            ptr(maximum), ptr(total), None, ptr(out),
        ], record, q.device)
        return out

    return _candidate(binding, run, "attention/flash_attention_score",
                      "Original GQA BSND attention; keepProb1, full-visible bottom-right decode, select attention output.")


def transformer_gemm(case):
    import torch

    binding = _MathCall(
        "aclnnGroupedMatmulAdd", [C.c_void_p] * 4 + [C.c_bool] * 2 + [C.c_int64],
        library="ops-transformer",
    )
    ta, tb = case.params["trans_a"], case.params["trans_b"]

    def run(a, b):
        lhs, rhs = (a.T if ta else a), (b.T if tb else b)
        # Original API requires transposeX=True and one K group.
        source = lhs.T.contiguous()
        group = torch.full((1,), lhs.shape[1], dtype=torch.int64, device=a.device)
        out = torch.zeros((1, lhs.shape[0], rhs.shape[1]), dtype=torch.float32, device=a.device)
        record = binding.record()
        binding.call([
            binding.tensor(source, record), binding.tensor(rhs.contiguous(), record),
            binding.tensor(group, record), binding.tensor(out, record), True, False, 2,
        ], record, a.device)
        return out[0].to(a.dtype)

    return _candidate(binding, run, "gmm/grouped_matmul_add",
                      "Single Kgroup, native FP32 accumulator initialized zero, transposeX layout and BF16 output cast.")


def transformer_fp8(case):
    import torch
    import torch.nn.functional as F

    if case.inputs[0].shape[-1] != 64:
        raise UnsupportedCase("This pilot uses64-value groups padded to128.")
    binding = _MathCall(
        "aclnnIndexerQuantCache",
        [C.c_void_p] * 4 + [C.c_int64, C.c_bool, C.c_float],
        library="ops-transformer",
    )

    def run(x):
        packed = F.pad(x.reshape(-1, 64).contiguous(), (0, 64), value=1e-4)
        rows = packed.shape[0]
        cache = torch.empty((rows, 1, 1, 128), dtype=torch.float8_e4m3fn, device=x.device)
        scales = torch.empty((rows, 1, 1, 1), dtype=torch.float32, device=x.device)
        slots = torch.arange(rows, dtype=torch.int32, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(cache, record), binding.tensor(scales, record),
            binding.tensor(packed, record), binding.tensor(slots, record), 1, False, 1.0,
        ], record, x.device)
        return scales.reshape(x.shape[:-1]), cache.reshape(rows, 128)[:, :64].contiguous().reshape(x.shape)

    return _candidate(
        binding, run, "attention/indexer_quant_cache",
        "Identity cache slots, Normal E4M3 mode, pad with FP16-representable1e-4 then crop. "
        "Tiny-input scale rounding must pass the manifest tolerance.",
    )
