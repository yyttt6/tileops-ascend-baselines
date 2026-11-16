"""Original Transformer graph APIs; only host shape, constants and output views."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_graph_call import GraphCall
from benchmarks.ascend_transform_baselines import _rope_tables


def _candidate(binding, run, op, directory, adaptation):
    from benchmarks.api import Implementation

    files = [Path(__file__), Path(__file__).with_name("ascend_graph_call.py"),
             Path(__file__).with_name("ascend_transform_baselines.py")]
    for root in (binding.vendor, LIBRARIES / "ops-transformer" / directory):
        files.extend(p for p in root.rglob("*") if p.is_file() and
                     p.suffix in (".cpp", ".h", ".py", ".json", ".so", ".o"))
    return Candidate(Implementation(run), "ops-transformer::" + op + " via CANN graph JIT",
                     adaptation, source_hashes(*files),
                     limitations=("Requires own source compilation and device launch attribution.",),
                     cleanup=binding.cleanup)


def transformer_graph_rope(case):
    import torch

    if case.inputs[0].ndim != 2 or case.params["rope_layout"] != "neox":
        raise UnsupportedCase("This pilot uses the 1D half/NeoX layout.")
    binding = GraphCall("ops-transformer")

    def run(x):
        cos, sin = _rope_tables(torch, x, case.params.get("base", 10000.))
        # The manifest rotates in FP32 after rounding its tables to input dtype.
        # Exact widening selects the existing FP32 kernel, avoiding intermediate
        # FP16 products; the original kernel body and schedule remain unchanged.
        inp = x.float().contiguous().view(1, x.shape[0], 1, x.shape[1])
        cos, sin = [torch.cat((t, t), dim=-1).float().view_as(inp) for t in (cos, sin)]
        out = torch.empty_like(inp)
        binding.call("RotaryPositionEmbedding", [inp, cos, sin, None], [out], {"mode": 0})
        return out.view_as(x).to(x.dtype)

    return _candidate(binding, run, "RotaryPositionEmbedding", "posembedding/rotary_position_embedding",
                      "1D->BSND, manifest-rounded tables widened exactly with input to FP32; native NeoX and output cast.")


def transformer_graph_rms(case):
    import torch

    binding = GraphCall("ops-transformer")

    def run(x, weight):
        packed = x.contiguous().reshape(1, -1, 1, x.shape[-1])
        gamma = weight.contiguous()
        encoder = torch.empty((1, 0, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        zero = torch.zeros_like(gamma)
        one = torch.ones_like(gamma)
        rope_zero = torch.zeros((0, x.shape[-1]), device=x.device, dtype=x.dtype)
        outs = [torch.empty((1, 1, packed.shape[1], x.shape[-1]),
                            device=x.device, dtype=x.dtype) for _ in range(3)]
        binding.call("NormRopeConcat",
                     [packed, packed, packed, encoder, encoder, encoder, gamma, zero, gamma,
                      zero, one, zero, one, zero, rope_zero, rope_zero],
                     outs,
                     {"norm_type": 4, "norm_added_type": 0, "rope_type": 0,
                      "concat_order": 0, "eps": float(case.params["eps"]), "is_training": False})
        return outs[0].reshape(x.shape)

    return _candidate(binding, run, "NormRopeConcat", "posembedding/norm_rope_concat",
                      "Original affine RMS mode4, ropeNONE, no encoder; select normalized Q.")


def transformer_graph_gemm(case):
    import torch

    binding = GraphCall("ops-transformer")

    def run(a, b):
        lhs, rhs = (a.T if case.params["trans_a"] else a), (b.T if case.params["trans_b"] else b)
        group = torch.full((1,), lhs.shape[1], device=a.device, dtype=torch.int64)
        out = torch.zeros((1, lhs.shape[0], rhs.shape[1]), device=a.device, dtype=torch.float32)
        binding.call("GroupedMatmulAdd", [lhs.T.contiguous(), rhs.contiguous(), group, out], [out],
                     {"transpose_x": True, "transpose_weight": False, "group_type": 2, "group_list_type": 0})
        return out[0].to(a.dtype)

    return _candidate(binding, run, "GroupedMatmulAdd", "gmm/grouped_matmul_add",
                      "One Kgroup; native transposeX layout, zero FP32 accumulator and BF16 output cast.")


def transformer_graph_prepermute(case):
    import torch

    if case.params["layout"].get("contiguous") != {
            "packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}:
        raise UnsupportedCase("This pilot uses complete tight expert-major routing.")
    binding = GraphCall("ops-transformer")
    experts = case.params["num_local_experts"]

    def run(x, ids):
        routes = ids.numel()
        out = torch.empty((routes, x.shape[1]), device=x.device, dtype=x.dtype)
        inverse = torch.empty(routes, device=x.device, dtype=torch.int32)
        prefix = torch.empty(experts, device=x.device, dtype=torch.int32)
        binding.call("MoeInitRoutingV2", [x.contiguous(), ids.contiguous()], [out, inverse, prefix],
                     {"active_num": routes, "expert_capacity": 0, "expert_num": experts,
                      "drop_pad_mode": 0, "expert_tokens_count_or_cumsum_flag": 1,
                      "expert_tokens_before_capacity_flag": False})
        return out, prefix, inverse

    return _candidate(binding, run, "MoeInitRoutingV2", "moe/moe_init_routing_v2",
                      "Native dropless routing, prefix counts and original inverse indices; all outputs checked.")


def _attention(binding, q, k, v, scale):
    torch = binding.torch
    batch, queries, heads, _ = q.shape
    out = torch.empty_like(q)
    maximum = torch.empty((batch, heads, queries, 8), device=q.device, dtype=torch.float32)
    total = torch.empty_like(maximum)
    unused = torch.empty((0, 0, 0, 0), device=q.device, dtype=q.dtype)
    binding.call("FlashAttentionScore",
                 [q.contiguous(), k.contiguous(), v.contiguous(), *([None] * 16)],
                 [maximum, total, unused, out],
                 {"scale_value": float(scale), "keep_prob": 1.0,
                  "pre_tockens": 2147483647, "next_tockens": 2147483647,
                  "head_num": heads, "input_layout": "BSND", "inner_precise": 0,
                  "sparse_mode": 0})
    return out, maximum


def transformer_graph_gqa(case):
    q, _, _, *extra = case.inputs
    p = case.params
    if (any(t is not None for t in extra) or q.shape[1] != 1 or p.get("softcap") is not None
            or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1):
        raise UnsupportedCase("This pilot is full-context single-query decode.")
    binding = GraphCall("ops-transformer")
    scale = p.get("sm_scale") or q.shape[-1] ** -0.5

    def run(q, k, v, *extra):
        return _attention(binding, q, k, v, scale)[0]

    return _candidate(binding, run, "FlashAttentionScore", "attention/flash_attention_score",
                      "Original BSND grouped attention; final decode query sees all KV, keepProb1.")


def transformer_graph_amax(case):
    import torch
    import torch.nn.functional as F

    if case.inputs[0].ndim != 2 or case.params["dim"] != -1:
        raise UnsupportedCase("This pilot reduces independent complete rows.")
    binding = GraphCall("ops-transformer")

    def run(x):
        rows, width = x.shape
        keys = F.pad(x.reshape(rows, width, 1, 1), (0, 127)).contiguous()
        query = torch.zeros((rows, 1, 1, 128), device=x.device, dtype=x.dtype)
        query[..., 0] = 1
        _, maximum = _attention(binding, query, keys, keys, 1.0)
        out = maximum[:, 0, 0, 0].to(x.dtype)
        return out[:, None] if case.params["keepdim"] else out

    return _candidate(binding, run, "FlashAttentionScore.softmax_max", "attention/flash_attention_score",
                      "Basis query and padded keys give QK=x; select exposed raw maximum. Include all padding costs.")


def transformer_graph_fp8(case):
    import torch
    import torch.nn.functional as F

    if case.inputs[0].shape[-1] != 64:
        raise UnsupportedCase("This pilot pads64-value groups to128.")
    binding = GraphCall("ops-transformer")

    def run(x):
        packed = F.pad(x.reshape(-1, 64).contiguous(), (0, 64), value=1e-4)
        rows = packed.shape[0]
        cache = torch.empty((rows, 1, 1, 128), device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((rows, 1, 1, 1), device=x.device, dtype=torch.float32)
        slots = torch.arange(rows, device=x.device, dtype=torch.int32)
        binding.call("IndexerQuantCache", [cache, scales, packed, slots], [cache, scales],
                     {"quant_mode": 1, "round_scale": False, "x_scale": 1.0})
        return scales.reshape(x.shape[:-1]), cache.reshape(rows, 128)[:, :64].contiguous().reshape(x.shape)

    return _candidate(binding, run, "IndexerQuantCache", "attention/indexer_quant_cache",
                      "Identity slots, original E4M3 mode1, FP16 dummy1e-4 and crop; strict tiny-input scale check required.")


def transformer_graph_deltanet(case):
    import torch

    if any(t is not None for t in case.inputs[5:]) or case.params.get("use_qk_l2norm_in_kernel"):
        raise UnsupportedCase("This pilot uses dense batches and unnormalized Q/K.")
    binding = GraphCall("ops-transformer")

    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        batch, length, heads, key_dim = q.shape
        gate = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)
        final = torch.empty((batch, heads, key_dim, v.shape[-1]), device=q.device, dtype=torch.float32)
        aqk = torch.empty((batch, heads, length, 64), device=q.device, dtype=q.dtype)
        akk = torch.empty_like(aqk)
        # The original ACLNN wrapper transposes beta to BNS even for BSND Q/K/V.
        beta_head = beta.permute(0, 2, 1).contiguous()
        binding.call("ChunkKdaFwd",
                     [q.contiguous(), k.contiguous(), v.contiguous(), gate, beta_head,
                      None, None, initial.contiguous() if initial is not None else None, None, None],
                     [out, final, None, aqk, akk, None, None, None, None, None, None],
                     {"layout": "BSND", "scale": float(case.params.get("scale") or key_dim ** -0.5),
                      "chunk_size": 64, "safe_gate": False, "lower_bound": -5.0,
                      "use_gate_in_kernel": False, "state_v_first": False})
        return out, final

    return _candidate(binding, run, "ChunkKdaFwd", "attention/chunk_kda_fwd",
                      "Zero log gate reduces original KDA to DeltaNet; original chunk64 and FP32 final state.")
