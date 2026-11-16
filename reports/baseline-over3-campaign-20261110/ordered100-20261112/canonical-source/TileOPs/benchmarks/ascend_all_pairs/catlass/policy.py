"""CPU-only semantic screening shared by factories and the coverage generator.

Unsupported means no matching *complete* entry under the allowed transforms in
this inspected source snapshot. It does not assert an operator is impossible to
implement by writing a new CATLASS kernel or changing an existing schedule.
"""
from __future__ import annotations
from math import prod

FLOATS = ("float16", "bfloat16", "float32")
UNARY = {"AbsFwdOp", "NegFwdOp", "ExpFwdOp", "LogFwdOp", "SqrtFwdOp", "ReciprocalFwdOp", "ReluFwdOp"}
BINARY = {"AddFwdOp", "SubFwdOp", "MulFwdOp", "DivFwdOp", "MaximumFwdOp", "MinimumFwdOp"}
BITWISE = {"BitwiseNotFwdOp", "BitwiseAndFwdOp", "BitwiseOrFwdOp", "BitwiseXorFwdOp"}
LOGICAL = {"LogicalNotFwdOp", "LogicalAndFwdOp", "LogicalOrFwdOp"}
ATTENTION = {"GQADenseFwdOp", "GQAVarlenFwdOp", "GQAPagedFwdOp", "MHADecodePagedWithKVCacheFwdOp",
             "GQAPrefillPagedWithKVCacheFwdOp", "NSAVarlenFwdOp"}


def _dtype(params, *names):
    return next((params[n][1] for n in names if n in params and params[n] is not None), None)


def _decision(supported, route, reason, evidence, code=None):
    return {"semantic_status": "supported" if supported else "unsupported", "route": route if supported else None,
            "reason": reason, "reason_code": code or ("matching_original_entry" if supported else "no_equivalent_complete_entry"),
            "source_keys": evidence}


def classify(op, p, family=None):
    yes = lambda route, reason, evidence: _decision(True, route, reason, evidence)
    no = lambda reason, evidence, code=None: _decision(False, None, reason, evidence, code)
    if op in UNARY | BINARY | BITWISE | LOGICAL:
        keys = ["dsl_bitwise"] if op in BITWISE | LOGICAL else ["dsl_unary" if op in UNARY else "dsl_binary"]
        if op == "ReluFwdOp":
            keys = ["dsl_compare"]
        if op == "ReciprocalFwdOp":
            keys = ["dsl_binary"]
        dtype = _dtype(p, "input", "x", "self", "a")
        if op in LOGICAL and dtype != "bool":
            return no("Numeric logical operands require truth normalization; the inspected bitwise kernel only gives the same function for bool representations.", keys, "truth_conversion_not_layout")
        if op in BITWISE and dtype not in ("bool", "int8", "int16", "int32", "int64"):
            return no("Bitwise dtype is outside the inspected integer/boolean representation route.", keys, "dtype_contract")
        if op in UNARY | BINARY and dtype not in FLOATS:
            return no("Inspected arithmetic route requires lossless FP16/BF16/FP32 input representation.", keys, "dtype_contract")
        if p.get("alpha", 1) != 1 or p.get("rounding_mode") is not None:
            return no("Original binary API has no alpha scaling or rounded-division option.", keys, "parameter_contract")
        return yes("vector", "Original DSL entry with operation/shape config; full input packing, lossless dtype representation, chunk padding/crop and complete native output selection implemented.", keys)
    if op == "SumFwdOp":
        if _dtype(p, "x", "input") not in FLOATS or p.get("dtype") not in (None, *FLOATS):
            return no("Original SIMT sum uses FP32 accumulation; integer/FP64 input or output contract cannot be replaced by a lossless FP32 conversion.", ["dsl_sum"], "dtype_contract")
        return yes("sum", "Original SIMT full-row sum; N_ELE input extent specialization preserves the existing128-thread reduction schedule; permute independent rows and keepdim restoration.", ["dsl_sum"])
    if op in ("MaxPool1dFwdOp", "MaxPool2dFwdOp", "MaxPool3dFwdOp", "AdaptiveMaxPool2dFwdOp"):
        if _dtype(p, "input") not in FLOATS:
            return no("Native64-lane max API requires lossless FP32 input representation.", ["dsl_reduce"], "dtype_contract")
        if op == "AdaptiveMaxPool2dFwdOp":
            from .pooling import adaptive_windows
            windows = adaptive_windows(p["input"][0][-2:], p["output_size"])
            width = max(prod(hi - lo for lo, hi in w) for w in windows)
        else:
            rank = int(op[7])
            kernel = p["kernel_size"]
            width = kernel ** rank if isinstance(kernel, int) else prod(kernel)
        if width > 64:
            return no(f"Complete pooling window has up to{width} elements; original max entry reduces only64 lanes. Splitting then re-reducing partials would add a reduction schedule and is not used.", ["dsl_reduce"], "whole_window_exceeds_native_reduce")
        return yes("pooling", "Whole independent pooling windows pack into the original64-lane max entry with negative-infinity padding; preserves stride/dilation/ceil/adaptive floor-ceil boundaries and complete output, no split reduction.", ["dsl_reduce"])
    if op in ("GemmFwdOp", "BmmFwdOp", "GroupedGemmFwdOp", "MoEGroupedGemmFwdOp"):
        if _dtype(p, "a") not in FLOATS:
            return no("Original example43 binding covers FP16/BF16/FP32 operands.", ["gemm"], "dtype_contract")
        if p.get("activation") is not None:
            return no("The full grouped GEMM contract includes a gated activation; example43 returns the complete linear GEMM only. The MX fused example65 additionally quantizes outputs/scales and is not this output contract.", ["gemm", "moe_mx"], "fused_output_contract")
        return yes("gemm", "Original example43 full GEMM; independent batch/group/expert slices and original dtype/output-type selection, with all current metadata reads inside run.", ["gemm", "gemm_kernel"])
    if op in ("Conv1dFwdOp", "Conv2dFwdOp", "Conv3dFwdOp"):
        if _dtype(p, "input") not in FLOATS:
            return no("Convolution packing route requires a supported native GEMM operand type.", ["gemm"], "dtype_contract")
        if "bias" in p and _dtype(p, "input") == "float32":
            return no("Original FP32 bias tile exceeds L1 at compile time. Schedule/tile changes were not applied.", ["gemm_bias", "gemm"], "native_fp32_bias_l1_limit")
        return yes("convolution", "Im2col is implemented using only padding/strided unfold/permute; native complete GEMM and optional native bias per independent convolution group, preserving original filter/channel reduction.", ["gemm", "gemm_bias", "gemm_kernel"])
    if op in ("GemmFP8FwdOp", "BmmFP8FwdOp", "MoEGroupedGemmFP8FwdOp"):
        if _dtype(p, "a") != "float8_e4m3fn":
            return no("Only original E4M3 operand type specialization is bound.", ["fp8", "fp8_epilogue"], "dtype_contract")
        return yes("fp8", "Original example51 public FP8 operand/FP32 accumulator types and native FP32 scale epilogue; scalar/block scale layout adaptation; independent columns for per-row B scale grids; native optional bias.", ["fp8", "fp8_epilogue"])
    if op == "GemmINT8W8A8FwdOp":
        if p.get("out_dtype") != "bfloat16" or "bias" in p:
            return no("The implemented original example57 binding selects BF16 output and no bias.", ["int8"], "dtype_or_bias_contract")
        return yes("int8", "Original example57 INT32 dot product and native token/channel FP32 dequantization; scalar scales broadcast losslessly on every invocation.", ["int8", "int8_epilogue"])
    if op in ATTENTION:
        keys = ["attention", "attention_tiling"]
        dtype = _dtype(p, "q")
        kn = "k_pages" if op == "GQAPagedFwdOp" else "k_new" if op == "GQAPrefillPagedWithKVCacheFwdOp" else "k"
        if dtype not in ("float16", "bfloat16") or _dtype(p, kn) != dtype:
            return no("Original example49 does not accept this scaled FP8/input dtype contract; MX scale encodings elsewhere in CATLASS cannot losslessly encode arbitrary FP32 scales.", keys + ["mx_attention"], "dtype_contract")
        if p["q"][0][-1] > 128:
            return no("Original example49 fixes the PV/output tile width at128; existing example70's fixed192 QK/128 V shape is not the256-wide contract without a prohibited tile/schedule change.", keys + ["attention70"], "native_head_width_limit")
        if p.get("softcap") not in (None, 0, 0.0):
            return no("Native example49 attention lacks score softcapping; layout transforms cannot introduce tanh score math.", keys, "softcap_missing")
        if p.get("fuse_rope") or p.get("pos_encoding_mode", "none") != "none":
            return no("Native attention expects already-rotated operands and provides no fused rotary transform for this contract.", keys, "rope_missing")
        if "sinks" in p or "sink" in p:
            return no("Native attention has no learned sink-logit contribution to the softmax denominator.", keys, "sink_missing")
        if p.get("out_dtype") not in (None, dtype):
            return no("Original example49 output has the operand storage dtype.", keys, "output_dtype_contract")
        if op == "GQAPrefillPagedWithKVCacheFwdOp" and _dtype(p, "k_pages") != dtype:
            return no("Append would require cache quantization/dequantization math; original attention does not implement that transform.", keys, "quantized_cache_append")
        return yes("attention", "Original complete attention; per-run layout/sequence/page/selected-slot packing preserves logical KV order and duplicate selected slots; optional interval crops, full cache outputs for append.", keys)
    if op in ("GemmW4A16FwdOp", "GemmW4A8FwdOp"):
        return no("Manifest uses packed affine integer4 weights with group/zero-point or FP32 channel scales. CATLASS w8a16, w4a8 integer activation and MX FP4/FP8 scale contracts do not match; unpacking/dequantizing/requantizing weights would introduce new arithmetic or lose scale information.", ["w8a16", "w4a8", "mx_w4"])
    if op == "GQABwdOp":
        return no("Manifest supplies base2 LSE; original FAG consumes natural-log LSE in exp(score-LSE), so direct layout/dtype adaptation is not semantically equivalent. Its bound host also selects unmasked FP16 AtlasA2, while current manifest cases require causal gradients. Original low-level mask support exists but does not fix the LSE API mismatch or3510 dispatch absence.", ["fag", "fag_lse", "bwd_contract", "fag_cmake", "backward_design"], "lse_convention_and_arch_contract")
    if op == "MLADecodeWithKVCacheFwdOp":
        return yes("mla", "Original example19 Mla API exactly matches512-latent+64-rope decode; complete per-call page-layout adapter implemented. CPU3510 compilation fails because MmadAtlasA2MLAQK dispatch has no3510 specialization. This is a deployment block, not mathematical non-support.", ["mla", "mla_host", "mla_cmake", "mla_dispatch"])
    if op in ("MLAPagedFwdOp", "MLAVarlenFwdOp", "DSAPagedFwdOp"):
        return no("Manifest requires both attention output and per-query LSE (and some cases additionally require dequantization). Original example19 and example49 exposed host APIs return attention values only; example70 LSE is an internal split-workspace state, not this complete LSE output API. Layout packing cannot supply missing outputs. The950 MLA directory is design-only.", ["mla", "mla_host", "attention", "attention70", "mla_design"])
    if op == "DSADecodeWithKVCacheFwdOp":
        return no("Sparse selected-slot attention uses512/576 score dimensions and variable valid index counts, including dim_tail=0. Example49 cannot handle this full score width, and original MLA entry is specialized for512+64 dense paged decode on AtlasA2; a complete3510 sparse-selected output route is not provided under the allowed transforms.", ["mla", "attention", "mla_cmake", "mla_design"])
    if op == "NSACompressedVarlenFwdOp":
        return no("Manifest returns both attention output and dtype-preserving per-query LSE over completed compressed blocks. Example49 returns only attention values; obtaining missing LSE via torch or a new softmax/reduction schedule is prohibited.", ["attention", "nsa_contract"])
    if op in ("NSATopKVarlenFwdOp", "TopKSelectFwdOp", "FP8LightningIndexerFwdOp"):
        return no("Native attention examples return attention values, not the requested compressed-block ranking / top-k IDs with this score formula. x_attention/rain-fusion masks use different selection semantics and do not expose the complete manifest outputs.", ["x_attention", "rain_attention"])
    if op == "MergeAttentionStatesFwdOp":
        return no("Native online-softmax rescaling is internal to attention. No original standalone API merges two supplied output/LSE states and returns both values and base2 LSE under this contract.", ["attention", "attention70"])
    if op in ("PagedKVCacheGatherFwdOp", "PagedKVCacheWriteFwdOp", "MLAKVCacheWriteFwdOp"):
        return no("CATLASS consumes cache pages inside attention but supplies no standalone full cache gather/write entry with this output/mutation contract. Torch gather/scatter alone would not be a CATLASS implementation.", ["attention", "mla"])
    if op.startswith(("Eq", "NeFwd", "Gt", "GeFwd", "Lt", "LeFwd")) or op in ("WhereFwdOp", "MaskedFillScalarFwdOp", "MaskedFillTensorFwdOp"):
        return no("Inspected compare_mask entry returns where(cmp(a,b),a,b), not boolean comparison output, and does not accept an arbitrary external condition. Casting selected operand values cannot recover the manifest boolean/where/masked-fill function.", ["dsl_compare"])
    if family == "elementwise":
        if any(token in op for token in ("Gelu", "Silu", "Sigmoid", "Tanh", "Relu")):
            return no(f"{op} is not an original standalone complete activation entry. Related CATLASS activation tiles are internal matmul epilogues; converting them to a new GM kernel or injecting an artificial matrix product exceeds layout/API adaptation.", ["activation_silu", "activation_gelu", "dsl_unary"])
        return no(f"{op} is absent from the original unary/binary complete vector operations. SIMD instruction builders and fused demonstration expressions cannot supply this contract without authoring new arithmetic/kernel behavior.", ["dsl_unary", "dsl_binary", "dsl_multi"])
    if family in ("norm", "rope"):
        return no(f"No complete original {op} kernel/API in CATLASS examples, optest, experimental entries or DSL examples. GEMM/attention and internal vector primitives do not provide the required normalization/rotary outputs under layout-only adaptation.", ["api_catalog", "dsl_unary", "attention"])
    if family == "reduction":
        return no(f"{op} requires a different reduction/scan formula or full-row output contract. SIMD reduction example only reduces independent64-lane chunks; SIMT entry implements sum only. Combining chunk results or adding variance/normalization/scan arithmetic changes the algorithm/schedule.", ["dsl_reduce", "dsl_sum"])
    if family == "pool":
        return no(f"No complete original {op} pooling kernel or index-output contract. Convolution/matmul computes dot products, and the inspected64-lane reduction demo is not a full pooling entry with these window/boundary semantics.", ["conv_native", "dsl_reduce"])
    if family == "quantization":
        return no(f"{op} needs standalone quantization/dequantization and its full scale/output contract. CATLASS quantized GEMMs consume scales; example62 generates FP8 scales only after a native matrix product and example65 returns MX fused-gated outputs. Those are not the standalone manifest function.", ["quant_output", "moe_mx", "int8"])
    if family == "linear_attention":
        return no(f"No complete {op} stateful recurrence/chunk backward entry. Dense softmax attention and stateless GEMM do not implement gates, recurrence, normalization, initial state or final-state outputs; composing a new recurrence would change mathematics/schedule.", ["attention", "api_catalog", "ssd_design"])
    if family == "mamba":
        return no(f"No complete original {op} state-space/causal-convolution entry. Example108 is an SSD design document, not executable kernels; ordinary convolution lacks the complete gated/stateful mutation contract.", ["ssd_design", "conv_native"])
    if family == "moe":
        return no(f"No complete {op} routing/permutation/fused-expert API matching all manifest outputs. Original grouped GEMM has materialized input groups; MX gated/finalize-routing examples impose FP8/MX scale, routing and output contracts that cannot be substituted losslessly.", ["moe_mx", "moe_finalize", "grouped_native"])
    if family == "sampling":
        return no(f"No original {op} probability filtering/sampling entry with requested masks, RNG or acceptance outputs. Attention selection masks are not probability-distribution sampling.", ["api_catalog", "x_attention"])
    if family in ("fft", "transform"):
        return no(f"CATLASS planar complex matmul is a matrix product, not {op}. Injecting Fourier/Hadamard coefficients and new arithmetic would be an algorithm implementation, not a layout-only wrapper of an existing transform API.", ["complex_matmul", "api_catalog"])
    if family == "sequence_modeling":
        return no(f"No complete original {op} sequence-modeling/hash/gating/Sinkhorn contract; plain GEMM/convolution entries do not produce its full state/auxiliary outputs.", ["api_catalog", "conv_native", "gemm"])
    return no(f"No matching complete original {op} entry identified in the audited CATLASS source catalog within the allowed transforms.", ["api_catalog"])
