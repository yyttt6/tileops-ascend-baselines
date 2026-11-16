"""Pure metadata decisions shared by factories and CPU coverage generation.

Status is never numerical verification. A supported path may still fail compilation,
execution or the unchanged oracle; those outcomes require a separate device receipt.
"""
from math import prod

SUPPORTED = "supported_unverified"
BLOCKED = "deployment_blocked"
UNSUPPORTED = "unsupported"
FLOATS = {"float16", "bfloat16", "float32"}
NATIVE_BLOCK = ("Original vllm-ascend extension and custom OPP were not built/deployed for this ARM64/CANN9.1 task; expected .cache/all-pairs/vllm/install/vllm_ascend/vllm_ascend_C.abi3.so and install/vendors/custom_transformer/op_api/lib/libcust_opapi.so. NPU execution forbidden.")
GATING_BLOCK = NATIVE_BLOCK + " Snapshot also lacks csrc/moe/moe_gating_top_k/op_kernel/arch35/moe_gating_top_k_regbase.h included by moe_gating_top_k_apt.cpp."

# Every unregistered operator has an explicit semantic review group. These are
# claims about the inspected owned implementation surface, not universal
# impossibility proofs or permanent exclusions of future upstream kernels.
REVIEWS = {}


def review(names, reason, evidence):
    for name in names.split():
        if name in REVIEWS:
            raise RuntimeError("Duplicate semantic review: " + name)
        REVIEWS[name] = {"reason": reason, "evidence": evidence}


review("AbsFwdOp CeilFwdOp CosFwdOp ErfFwdOp ExpFwdOp Expm1FwdOp FloorFwdOp Log1pFwdOp LogFwdOp ReciprocalFwdOp RoundFwdOp RsqrtFwdOp SignFwdOp SinFwdOp SqrtFwdOp TanhFwdOp TruncFwdOp",
       "The owned public elementwise surface exposes scalar multiply-add, specialized gates and fused model operators; it has no standalone {op} output. Internal instructions inside JIT/native functions are not callable APIs. Extracting/replacing their kernel body would exceed layout/API adaptation.",
       ["vllm_ascend/ops/triton/muls_add.py", "vllm_ascend/ops/triton/fused_gdn_gating.py", "csrc/torch_binding.cpp"])
review("BitwiseAndFwdOp BitwiseNotFwdOp BitwiseOrFwdOp BitwiseXorFwdOp LogicalAndFwdOp LogicalNotFwdOp LogicalOrFwdOp EqFwdOp NeFwdOp GeFwdOp GtFwdOp LeFwdOp LtFwdOp IsfiniteFwdOp IsinfFwdOp IsnanFwdOp",
       "No owned general {op} tensor API in the binding/kernel catalog. Sampling-mask code returns packed finite-token bits plus counts, with activity masking; it does not expose the manifest's elementwise boolean result. Expanding packed bits using Torch would compute the requested op outside this library.",
       ["csrc/torch_binding.cpp", "vllm_ascend/ops/triton/v2/sample/sampling_mask.py"])
review("DivFwdOp FloorDivideFwdOp RemainderFwdOp PowFwdOp MulFwdOp MaximumFwdOp MinimumFwdOp LerpScalarFwdOp LerpTensorFwdOp WhereFwdOp MaskedFillScalarFwdOp MaskedFillTensorFwdOp ClampScalarFwdOp ClampTensorFwdOp NanToNumFwdOp",
       "Original muls_add_triton implements scalar*x+y, not the requested {op} tensor arithmetic/selection. The exposed fused gates do not return these intermediate values. No owned equivalent direct API found; using Torch or rewriting a kernel is outside scope.",
       ["vllm_ascend/ops/triton/muls_add.py", "vllm_ascend/ops/triton/activation/swiglustep.py", "csrc/torch_binding.cpp"])
review("EluFwdOp GeluFwdOp GeluAndMulFwdOp GeluTanhAndMulFwdOp HardsigmoidFwdOp HardswishFwdOp HardtanhFwdOp LeakyReluFwdOp MishFwdOp PreluFwdOp ReluFwdOp SeluFwdOp",
       "Owned activation kernel implements SiLU gate/up with optional clipping. GELU/other framework activation paths call torch_npu or Torch; no equivalent owned {op} kernel/API was found. SiLU, fast-GELU, exact GELU and tanh-GELU are not interchangeable.",
       ["vllm_ascend/ops/activation.py", "vllm_ascend/ops/triton/activation/swiglustep.py"])
review("AlibiFwdOp SinusoidalFwdOp DropoutFwdOp",
       "No owned standalone {op} generation/RNG API. RoPE consumes preconstructed trigonometric tables and rotates input; it does not expose an equivalent ALiBi, sinusoidal encoding or dropout result and mask.",
       ["vllm_ascend/ops/triton/rope.py", "csrc/torch_binding.cpp"])
review("AllFwdOp AnyFwdOp ArgmaxFwdOp ArgminFwdOp AminFwdOp CountNonzeroFwdOp CumprodFwdOp LogSoftmaxFwdOp LogSumExpFwdOp ProdFwdOp StdFwdOp VarFwdOp VarMeanFwdOp VectorNormFwdOp",
       "Reviewed owned reduction outputs are mean, normalized RMS/LayerNorm, chunk cumsum, router probabilities/selected IDs, and attention maximum. None directly exposes the complete {op} contract. Torch reduction paths and hidden fused intermediates are not own executable implementations.",
       ["vllm_ascend/ops/triton/batch_invariant/mean.py", "vllm_ascend/ops/triton/batch_invariant/softmax.py", "vllm_ascend/ops/triton/batch_invariant/rmsnorm.py", "csrc/torch_binding.cpp"])
review("AdaLayerNormZeroFwdOp BatchNormFwdOp BatchNormBwdOp FusedAddRMSNormPerTokenQuantFwdOp",
       "The inspected LayerNorm/RMSNorm APIs do not expose the full {op} outputs and auxiliary contracts. AdaLNZero needs a separate arbitrary output gate; BatchNorm needs running/saved statistics or gradients; quantized/fused QK routes have distinct packing/quantization/table-state contracts. No verified layout-only complete binding is available, and component Torch calls are not credited.",
       ["vllm_ascend/ops/triton/layernorm_gated.py", "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py", "csrc/attention/rms_norm_dynamic_quant/op_host/rms_norm_dynamic_quant_def.cpp"])
review("AdaptiveMaxPool2dFwdOp AdaptiveMaxPool2dIndicesFwdOp MaxPool1dFwdOp MaxPool1dIndicesFwdOp MaxPool2dFwdOp MaxPool2dIndicesFwdOp MaxPool3dFwdOp MaxPool3dIndicesFwdOp",
       "No owned spatial {op} API. MeanPooling's independent sequence chunks differ from overlapping/adaptive spatial windows, padding divisors and maxpool index outputs. Building a pooling algorithm or delegating to Torch is outside the inspected layout/API routes.",
       ["vllm_ascend/ops/triton/batch_invariant/mean.py", "csrc/torch_binding.cpp"])
review("BmmFP8FwdOp GemmFP8FwdOp GemmINT8W8A8FwdOp GemmW4A16FwdOp GemmW4A8FwdOp",
       "Owned matmul_persistent has unscaled floating GEMM only. Model quantized linear wrappers dispatch torch_npu/other libraries; owned grouped quantized kernels fuse Swiglu/quantization and use different scale/packing/output contracts. No complete owned {op} API matching manifest scales and result is established without arithmetic outside the target.",
       ["vllm_ascend/ops/triton/batch_invariant/matmul.py", "vllm_ascend/ops/linear.py", "csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h"])
review("FP8QuantFwdOp FP8QuantPerBlockFwdOp INT4QuantPerGroupFwdOp INT8DequantPerBlockFwdOp INT8DequantPerChannelFwdOp INT8DequantPerTensorFwdOp INT8QuantPerBlockFwdOp INT8QuantPerChannelFwdOp INT8QuantPerTensorFwdOp SmoothQuantFwdOp",
       "Owned indexer query quantization fixes 128-channel INT8, rounds scale to FP16 before division and floors it at 2^-24. Other owned quantizers are MXFP4 or fused gate/quant stages. These scale dtype, clamp, group axis, packing and/or output contracts differ from {op}; no Torch quantization/dequantization substitute is registered.",
       ["vllm_ascend/ops/triton/quantize_indexer_query.py", "vllm_ascend/ops/triton/quantize_mxfp4_indexer.py", "vllm_ascend/ops/triton/activation/swiglu_quant.py"])
review("DeltaNetChunkFwdOp DeltaNetChunkBwdOp GLAChunkFwdOp GLAChunkBwdOp GLARecurrentFwdOp GLAFwdOp",
       "Complete KDA entries expose final output/state of a delta-update recurrence. {op} either requires intermediate training tensors/gradients or a GLA recurrence without the delta projection. Original lightning attention uses fixed per-head slope; changing it or exposing internal stages is not a host layout transformation.",
       ["vllm_ascend/ops/triton/kda/kda.py", "vllm_ascend/ops/triton/mamba/lightning_attn.py"])
review("SSDChunkCouplingFwdOp SSDChunkCumsumFwdOp SSDChunkScanFwdOp SSDChunkStateFwdOp SSDRecurrentFwdOp SSDStatePassingFwdOp Mamba2FwdOp SelectiveScanFwdOp",
       "Original lightning attention uses fixed slope decay and KDA uses delta projection; neither exposes {op}'s arbitrary dt/A/selective-SSM staged contract and all state outputs. Model paths importing external FLA/mamba or torch_npu are dependency implementations, not owned kernels.",
       ["vllm_ascend/ops/triton/mamba/lightning_attn.py", "vllm_ascend/ops/triton/kda/kda.py", "vllm_ascend/ops/gdn.py"])
review("TopKMaskFwdOp TopKTopPMaskFwdOp TopPMaskFwdOp MinPMaskFwdOp SamplingFromProbsFwdOp ChainSpeculativeSamplingFwdOp",
       "Sampler masks are Torch/torch_npu probability-based routes; categorical/rejection kernels have their own RNG or block-verification interfaces. They do not supply the manifest's exact {op} logits-mask or externally supplied uniform/random and full returned-token contract. No fallback is registered.",
       ["vllm_ascend/sample/sampler.py", "vllm_ascend/ops/triton/reject_sample.py", "vllm_ascend/ops/triton/v2/sample/categorical_sample.py"])
review("MoEPrePermuteFwdOp MoEPostPermuteFwdOp MoEPermuteAlignFwdOp FusedMoEFwdOp FusedMoEExpertsFwdOp FusedMoESharedExpertFwdOp IndexedExpertMLPFwdOp MoEGroupedGemmFP8FwdOp",
       "Owned fused dispatch/FFN/combine includes routing stages inside a distributed fused kernel, not standalone manifest {op} outputs. Framework routing/permutation/quantized-GEMM calls use torch_npu/dependencies. Current owned grouped matmul/MLP adapters do not implement external route-weight aggregation, inverse maps, distributed outputs or FP8 scale contracts.",
       ["csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h", "vllm_ascend/ops/fused_moe/router/fused_topk_router.py", "csrc/torch_binding.cpp"])
review("GQABwdOp DSAPagedFwdOp MLAPagedFwdOp NSATopKVarlenFwdOp FP8LightningIndexerFwdOp TopKSelectFwdOp",
       "Source contains sparse/decode/model-cache operators, but {op} needs a separate full-contract mapping (packed/paged sequence metadata, cache mutation, selected indices, gradients, quantization and/or auxiliary outputs). The implemented dense GQA route alone is not proof of that contract. No executable adapter is claimed for this route; see per-op evidence and semantic review limitation rather than interpreting it as mathematical impossibility.",
       ["csrc/torch_binding.cpp", "vllm_ascend/ops/triton/pcp_kv_cache.py", "vllm_ascend/attention/sparse_flash_mla.py"])
review("FFTC2CFwdOp HadamardTransformFwdOp EngramDecodeFwdOp EngramGateConvFwdOp EngramGateConvBwdOp MHCPreFwdOp",
       "No matching complete owned {op} API established. FFT/Hadamard transforms are absent; Engram kernels expose hash/lookback/lookup/gating pieces, not full decode or convolution gradients. HcPre uses sqrt(mean(x^2)+eps) and sigmoid offsets, whereas the manifest reference uses BF16 x*x then sqrt(sum)/sqrt(D)+1e-4 and a different full residual-output contract; this cannot be fixed by changing layout or tolerance.",
       ["csrc/torch_binding.cpp", "vllm_ascend/ops/triton/engram_gate.py", "csrc/moe/hc_pre/op_kernel/hc_pre_base_arch35.h"])


# These source candidates require unresolved COMPLETE output/metadata bindings.
# They are deliberately separate from an established mathematical mismatch.
ADAPTER_GAPS = {
}

REVIEWS['FusedAddRMSNormPerTokenQuantFwdOp'] = {'reason': 'Owned RmsNormDynamicQuant has no residual input or residual output, multiplies gamma in FP32 before quantization, and has no scale_ub parameter. Manifest requires FP32 unrounded residual statistics, two input-dtype normalization/weight roundings, residual_out and capped scale. These cannot be selected by changing layouts or API attributes.', 'evidence': ['csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_single_row_kernel.h', 'csrc/torch_binding.cpp']}
REVIEWS['FP8LightningIndexerFwdOp'] = {'reason': 'The manifest returns the full dense [B,Sq,Sk,1] index logits. Owned quant_lightning_indexer_v2 returns selected indices (and selected values in its candidate path), while its normal wrapper rejects return_value !=0. It does not expose the complete unselected logits. Extracting an internal stage or computing scores with Torch would violate scope.', 'evidence': ['csrc/torch_binding.cpp', 'csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h']}
REVIEWS['TopKSelectFwdOp'] = {'reason': 'The standalone manifest selects from very wide raw score rows with per-row starts/ends. Owned gating limits row width to 2048 and selects normalized/grouped routing scores; indexer performs its own QK/weight pipeline. No exposed raw wide-row selection API preserving this contract was found.', 'evidence': ['csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp', 'csrc/torch_binding.cpp']}
REVIEWS['NSATopKVarlenFwdOp'] = {'reason': 'Manifest ranks summed per-head block importance after closed-block softmax normalization, with forced-priority blocks and input-dtype Q scaling. Owned router top-k consumes logits and sparse-attention returns aggregated values/LSE, not these per-block importance scores. No complete original API exposes this result.', 'evidence': ['csrc/torch_binding.cpp', 'vllm_ascend/ops/triton/kda/kda.py']}
REVIEWS['MLAPagedFwdOp'] = {'reason': 'Manifest cache/query use 512 latent +64 RoPE coordinates and require complete LSE. Original sparse-flash-attention supports the separated 512+64 score but returns only max/sum. Original sparse_flash_mla returns LSE but has fixed 512 total score dimension and no separate RoPE input. The complete 576-coordinate score plus LSE cannot be selected from either current API without new math.', 'evidence': ['csrc/torch_binding.cpp', 'csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.h', 'csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp']}
REVIEWS['DSAPagedFwdOp'] = {'reason': 'Manifest consumes a packed uint8 656-byte row and returns both attention and LSE. The reviewed owned FP8 sparse-attention route consumes its distinct C8 packing and returns max/sum; the full-LSE sparse_flash_mla API is BF16-only/fixed512. No original API matches the complete packed-cache/576-score/LSE contract.', 'evidence': ['csrc/torch_binding.cpp', 'csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp', 'csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.h']}

def decide(op, p):
    """p is Case.params, or inventory tensors merged with params; no tensors read."""
    if op in ADAPTER_GAPS:
        return "adapter_blocked", ADAPTER_GAPS[op]
    if op in REVIEWS:
        return UNSUPPORTED, REVIEWS[op]["reason"].format(op=op)
    if op in {"AddFwdOp", "SubFwdOp", "NegFwdOp"}:
        dtype = p["input"][1]
        if dtype not in FLOATS or ("other" in p and p["other"][1] != dtype):
            return UNSUPPORTED, "Native scalar multiply-add uses floating scale promotion; arbitrary integer or mixed-dtype arithmetic is not established without lossy conversion."
    elif op in {"RMSNormFwdOp", "FusedAddRMSNormFwdOp", "MeanFwdOp", "SumFwdOp", "MeanPoolingFwdOp", "CumsumFwdOp", "LayerNormFwdOp", "FusedAddLayerNormFwdOp", "GroupNormFwdOp", "InstanceNormFwdOp", "AdaLayerNormFwdOp"}:
        shape, dtype = p["x"]
        if dtype not in FLOATS:
            return UNSUPPORTED, "Original reviewed floating reduction/norm API has no lossless path for this dtype."
        if op == "InstanceNormFwdOp" and not p.get("use_input_stats", True):
            return UNSUPPORTED, "LayerNorm's fresh input statistics cannot substitute supplied running mean/variance."
        if op == "InstanceNormFwdOp" and "running_mean" in p:
            return UNSUPPORTED, "The original LayerNorm route does not update the manifest's running mean/variance; its normalization tensor alone does not establish the full stateful contract."
        if op in {"LayerNormFwdOp", "FusedAddLayerNormFwdOp", "GroupNormFwdOp", "InstanceNormFwdOp", "AdaLayerNormFwdOp"}:
            width = (prod(shape[1:]) // p["num_groups"] if op == "GroupNormFwdOp" else
                     prod(shape[2:]) if op == "InstanceNormFwdOp" else prod(p.get("normalized_shape", [shape[-1]])))
            if width > 65536 // (4 if dtype == "float32" else 2):
                return UNSUPPORTED, "Original layer_norm_fwd_npu rejects normalized groups above 64KB; splitting a statistics domain is not an equivalent call."
            if width & (width - 1):
                return "adapter_blocked", "Original LayerNorm sums masked-load lanes without an explicit zero value for non-power-of-two domains; equivalence is unresolved until the original source/backend defines those lanes. No kernel rewrite or statistics-domain padding is credited."
    elif op in {"AdaptiveAvgPool2dFwdOp", "AvgPool1dFwdOp", "AvgPool2dFwdOp", "AvgPool3dFwdOp"}:
        if p["input"][1] not in FLOATS:
            return UNSUPPORTED, "Original mean API has no lossless integer accumulation route."
    elif op in {"GemmFwdOp", "BmmFwdOp", "GroupedGemmFwdOp", "MoEGroupedGemmFwdOp", "MoEExpertMLPFwdOp", "SharedExpertMLPFwdOp"}:
        if p.get("activation") not in (None, "silu_and_mul"):
            return UNSUPPORTED, "Owned expert epilogue is SiLU gate/up; requested activation is different."
    elif op in {"Conv1dFwdOp", "Conv2dFwdOp", "Conv3dFwdOp"}:
        if any(n != 1 for n in p["weight"][0][2:]):
            return UNSUPPORTED, "Spatial kernel is not pointwise. Owned GEMM expresses only unit-window convolution by views; causalConv1D is channelwise causal and cannot express cross-channel general spatial convolution via layout changes."
    elif op in {"GDNFwdOp", "KDAFwdOp", "DeltaNetFwdOp", "DeltaNetRecurrentFwdOp"}:
        if op == "KDAFwdOp" and p.get("use_gate_in_kernel") and p.get("lower_bound") is not None:
            return UNSUPPORTED, "Bounded KDA raw gate is implemented only by the vendor Torch helper; native fused_kda_gate computes unbounded -exp(A)*softplus and cannot express this bound."
    elif op in {"RoPEFwdOp", "RoPELlama31FwdOp", "RoPENeoxPositionIdsFwdOp", "LongRoPEFwdOp", "YaRNFwdOp", "SigmoidFwdOp", "SoftplusFwdOp", "SiluFwdOp", "SiluAndMulFwdOp", "MergeAttentionStatesFwdOp"}:
        pass
    elif op in {"PagedKVCacheGatherFwdOp", "PagedKVCacheWriteFwdOp", "MLAKVCacheWriteFwdOp"}:
        if op != "PagedKVCacheGatherFwdOp":
            dt = p["k_pages"][1] if op == "PagedKVCacheWriteFwdOp" else p["kv_cache"][1]
            if dt == "float8_e4m3fn":
                return UNSUPPORTED, "Original state-copy kernel copies values without saturating divide-and-quantize; quantized cache writes are not replaced by Torch arithmetic."
    elif op == "AmaxFwdOp":
        if p["x"][1] not in {"float16", "bfloat16"}:
            return UNSUPPORTED, "Sparse attention OpDef accepts FP16/BF16 only; FP32 input narrowing would be lossy."
        return BLOCKED, NATIVE_BLOCK
    elif op == "GQAPrefillPagedWithKVCacheFwdOp":
        if p.get("softcap") is not None or p["k_pages"][1] == "float8_e4m3fn":
            return UNSUPPORTED, "Original current attention/cache-copy routes lack logit softcap and saturating scale-divide FP8 append writes; these are not replaced by Torch arithmetic."
        if 2*p["q"][0][-1] > 512 or p["q"][0][-2]//p["k_new"][0][-2] > 128:
            return UNSUPPORTED, "Packed original shared-KV score/value width exceeds512 or grouped query heads exceed128."
        return BLOCKED, NATIVE_BLOCK
    elif op == "FusedQKNormRoPEFwdOp":
        pass
    elif op in {"MLAVarlenFwdOp", "NSACompressedVarlenFwdOp"}:
        q, dt = p["q"]
        v = p["v"][0] if op == "MLAVarlenFwdOp" else p["v_cmp"][0]
        if dt != "bfloat16" or q[-1]+v[-1] > 512:
            return UNSUPPORTED, "Original sparse_flash_mla complete-LSE OpDef accepts only BF16 and exactly 512 shared score/value coordinates. FP16 cannot be losslessly narrowed to BF16; packed K+V must fit 512."
        return BLOCKED, NATIVE_BLOCK + " Original sparse_flash_mla_metadata AICPU component must also be built."
    elif op == "NSAVarlenFwdOp":
        if p["q"][1] not in {"float16", "bfloat16"} or p["q"][0][-1]+p["v"][0][-1] > 512:
            return UNSUPPORTED, "Original sparse attention only accepts FP16/BF16 and <=512 packed K+V coordinates."
        return BLOCKED, NATIVE_BLOCK
    elif op in {"MLADecodeWithKVCacheFwdOp", "DSADecodeWithKVCacheFwdOp"}:
        q, dt = p["q"]
        k = p["k"][0] if op == "MLADecodeWithKVCacheFwdOp" else p["kv"][0]
        tail = p["q_pe"][0][-1] if op == "MLADecodeWithKVCacheFwdOp" else p["dim_tail"]
        dim = q[-1] if op == "MLADecodeWithKVCacheFwdOp" else q[-1]-tail
        if dt not in {"float16", "bfloat16"} or dim > 512 or tail > 64 or q[-2]//k[-2] > 128:
            return UNSUPPORTED, "Original sparse attention requires FP16/BF16, at most 512 shared-KV +64 RoPE coordinates and <=128 grouped heads; no lossy narrowing/splitting of softmax is used."
        return BLOCKED, NATIVE_BLOCK
    elif op in {"GQADenseFwdOp", "GQAVarlenFwdOp", "GQAPagedFwdOp", "MHADecodePagedWithKVCacheFwdOp"}:
        q, dt = p["q"]
        k, kdt = p.get("k", p.get("k_pages"))
        v, _ = p.get("v", p.get("v_pages"))
        if p.get("softcap") is not None or "sinks" in p:
            return UNSUPPORTED, "Original sparse-attention API lacks logit softcap / auxiliary sink normalization; slicing cannot supply these score transformations."
        if op != "GQADenseFwdOp" and (dt not in {"float16", "bfloat16"} or kdt != dt or any(n in p for n in ("q_scale", "k_scale", "v_scale"))):
            return UNSUPPORTED, "Packed/page adapter uses the unquantized original native API; scaled FP8 cache inputs need a different native quantized contract, without lossy conversion."
        if dt == "float8_e4m3fn":
            if q[0] != 1 or q[-1] != 128 or v[-1] != 128 or k[1] % 128 or p.get("out_dtype") != "float16" or not p.get("is_causal") or p.get("pos_encoding_mode", "none") != "none":
                return UNSUPPORTED, "Existing owned packed FP8 C API route requires B1/D128, complete 128-token pages, causal mode and FP16 output without RoPE."
            return BLOCKED, "Missing source-tree custom OPP op_api/lib/libcust_opapi.so for existing aclnnKvQuantSparseFlashAttentionVllm binding; no ARM64 build or execution performed."
        if dt not in {"float16", "bfloat16"} or q[-1] + v[-1] > 512 or q[-2] // k[-2] > 128:
            return UNSUPPORTED, "Sparse shared-KV API requires FP16/BF16, K+V feature width <=512 and at most 128 grouped query heads."
        return BLOCKED, NATIVE_BLOCK
    elif op == "SoftmaxFwdOp":
        shape = p["x"][0]
        if shape[p["dim"]] > 2048:
            return UNSUPPORTED, "Owned MoE gating host hard-limits expert/reduction width to 2048; splitting softmax changes normalization. batch_invariant/softmax.py uses only Torch and is not credited."
        return BLOCKED, GATING_BLOCK
    elif op == "FusedTopKFwdOp":
        return BLOCKED, GATING_BLOCK
    elif op == "MHCPostFwdOp":
        return BLOCKED, NATIVE_BLOCK
    elif op in {"CausalConv1dFwdOp", "CausalConv1dUpdateFwdOp"}:
        if p["x"][1] != "float16":
            return UNSUPPORTED, "Owned CausalConv1dV310 OpDef is FP16-only; BF16 cannot be narrowed losslessly. The Python alternative is Torch conv1d and is not credited."
        return BLOCKED, "CausalConv1dV310 is registered only for ascend310p, not the present target, and no ARM64 extension/OPP is built. Layout/state adapter exists; legal target deployment remains unresolved."
    else:
        raise KeyError("Missing explicit semantic decision for " + op)
    return SUPPORTED, "Concrete owned-kernel adapter implemented; import/layout/contract reviewed statically only. No NPU initialization, compilation or numerical validation performed."
