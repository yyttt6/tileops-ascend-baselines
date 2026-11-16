"""CPU-only, incremental coverage report. Never imports torch or a vendor runtime."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang'
VENDOR=ROOT/'ascend_baselines/tilelang'
INVENTORY=OUT.parent/'inventory.json'
sys.path.insert(0,str(ROOT/'TileOPs'))
from benchmarks.ascend_all_pairs.tilelang import PROVIDERS

_CACHE={}
def sha(path):
    path=Path(path)
    if path not in _CACHE:_CACHE[path]=hashlib.sha256(path.read_bytes()).hexdigest()
    return _CACHE[path]
def source(relative,symbol=None):
    p=VENDOR/relative
    line=1
    if symbol and p.suffix=='.py':
        tree=ast.parse(p.read_text())
        found=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==symbol]
        if found:line=found[0].lineno
    return {'file':str(p.relative_to(ROOT)),'line':line,'symbol':symbol,'sha256':sha(p)}

catalog=[]
for directory in ('examples','testing/ascend','testing/python','benchmark','maint'):
    for f in sorted((VENDOR/directory).rglob('*.py')):
        try:tree=ast.parse(f.read_text())
        except (SyntaxError,UnicodeDecodeError):continue
        functions=[]
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                functions.append({'name':n.name,'line':n.lineno,'end_line':n.end_lineno})
        text=f.read_text()
        catalog.append({'file':str(f.relative_to(ROOT)),'sha256':sha(f),'functions':functions,
                        'has_native_dsl':any(s in text for s in ('T.Kernel','T.prim_func','@tilelang.jit','@tl.jit')),
                        'explicit_ascend_vf':('SimtVF' in text or 'SimdVF' in text),
                        'parallel_lines':[i for i,l in enumerate(text.splitlines(),1) if 'T.Parallel(' in l]})
catalog_path=OUT/'source_catalog.json'
catalog_path.write_text(json.dumps({'schema':'tilelang-source-catalog/v1','scope':['examples','testing/ascend','testing/python','benchmark','maint'],
                                    'excludes':'Third-party dependencies are not own-kernel implementations. Torch references are never candidates.',
                                    'files':catalog},indent=2)+'\n')

inventory=json.loads(INVENTORY.read_text())
plans={}
readme=source('examples/ascend/README.md')
checker=source('tilelang/ascend/analysis/vf_checker.py','check_ascend_vf')

def plan(ops,status,reason,refs,adapter=None):
    for op in ops.split():
        plans[op]={'semantic_status':status,'reason':reason,'sources':[source(*r) if isinstance(r,tuple) else source(r) for r in refs],
                   'adapter':str((HERE/adapter).relative_to(ROOT)) if adapter else None}

plan('AddFwdOp GemmFwdOp RMSNormFwdOp FP8QuantFwdOp','supported_unmeasured','Reusable original Ascend API; unchanged existing helper. Per-case verified history is admitted only with an intact receipt and checked source hashes.',
     [('examples/ascend/example_vecadd.py','vector_add'),('examples/ascend/example_gemm.py','gemm'),('examples/ascend/example_rmsnorm.py','rms_norm_fwd'),('examples/ascend/example_per_token_cast_to_fp8.py','per_token_cast_simt')],'__init__.py')
plan('BmmFwdOp Conv1dFwdOp Conv2dFwdOp Conv3dFwdOp GroupedGemmFwdOp MoEGroupedGemmFwdOp SumFwdOp MeanFwdOp','supported_unmeasured','Original GEMM with exact input layout/batch/group/padding/constant-coordinate mapping; no kernel arithmetic or schedule edits. Requires individual manifest validation.',[('examples/ascend/example_gemm.py','gemm')],'adapters.py')
plan('FusedAddRMSNormFwdOp','supported_unmeasured','Original Add plus original RMSNorm; residual rounds to manifest input dtype before normalization; both outputs retained.',[('examples/ascend/example_vecadd.py','vector_add'),('examples/ascend/example_rmsnorm.py','rms_norm_fwd')],'adapters.py')
plan('GQADenseFwdOp GQAPagedFwdOp','supported_unmeasured','Original Ascend per-KV-head full-context attention. Decode/window crops and page gather supported; multi-query causal/window, softcap, RoPE and FP8 scaling need other original APIs.',[('examples/ascend/flash_attention/core.py','flash_attention_fwd')],'attention.py')
plan('BitwiseNotFwdOp BitwiseXorFwdOp','supported_unmeasured','Original fixed64 int32 SIMD kernel is semantically suitable through independent slices and integer bit-preserving views; launch-count capacity must be measured.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel')],'microblocks.py')
plan('ClampScalarFwdOp ClampTensorFwdOp HardtanhFwdOp','supported_unmeasured','Original fixed128 clamp kernel with independent slices, exact dtype widening and scalar/tensor bound broadcasting; no kernel body changes.',[('testing/ascend/language/test_tilelang_ascend_clamp.py','test_clamp_nan_and_inverted_bounds')],'microblocks.py')
plan('AbsFwdOp ExpFwdOp LogFwdOp SinFwdOp CosFwdOp TanhFwdOp SqrtFwdOp RsqrtFwdOp ErfFwdOp FloorFwdOp CeilFwdOp TruncFwdOp RoundFwdOp SigmoidFwdOp IsnanFwdOp IsinfFwdOp PowFwdOp',
     'deployment_blocked_static','Generic original scalar-math kernel exposes the required operation, but its unconditional T.Parallel is outside a VF block. Current Ascend vf_checker rejects it; adding a VF changes the kernel schedule and is not done. Per-op execution is not claimed.',
     [('testing/python/math/test_math_fast_math.py','run_single_arg_mathop_test'),'tilelang/ascend/analysis/vf_checker.py'],'diagnostics.py')
plan('SubFwdOp MulFwdOp DivFwdOp ReciprocalFwdOp','deployment_blocked_static','Original IEEE math test has the matching operation and unchanged 128-thread generic kernel; the enclosing Parallel has no Ascend VF. The native precise-div microtest is fixed64 and is an alternative capacity-limited route, not a numerical pass.',
     [('testing/python/math/test_math_ieee_math.py','run_ieee_math_test'),('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','test_precise_division'),'tilelang/ascend/analysis/vf_checker.py'])
plan('AmaxFwdOp AminFwdOp','deployment_blocked_static','Generic original reductions/topk have equivalent value reductions but use unsupported non-VF scheduling. Ascend reduction regression builders allocate one thread per reduction element, exceeding a VF for these full manifest axes; splitting the reduction would require additional native reduction composition.',
     [('examples/topk/example_topk.py','tl_topk'),('testing/ascend/language/test_tilelang_ascend_reduce.py','make_reduce_kernel'),'tilelang/ascend/analysis/vf_checker.py'],'diagnostics.py')
plan('LayerNormFwdOp FusedAddLayerNormFwdOp GroupNormFwdOp InstanceNormFwdOp','deployment_blocked_static','Original _layernorm_fwd supplies mean/variance/affine normalization; group/instance dimensions can be packed as rows, and Add has an original API. The LayerNorm kernel has unconditional non-VF Parallel/casts; no schedule edits allowed.',
     [('examples/norm/layernorm.py','_layernorm_fwd'),'tilelang/ascend/analysis/vf_checker.py'],'diagnostics.py')
plan('SoftmaxFwdOp','deployment_blocked_static','Original online softmax computes the same full-axis output; original default generic schedule has non-VF Parallel and is rejected on Ascend.',[('examples/online_softmax/online_softmax.py','softmax_kernel'),'tilelang/ascend/analysis/vf_checker.py'],'diagnostics.py')
plan('CumsumFwdOp','deployment_blocked_static','Original scan templates provide cumsum with axis/layout parameters, but their generic shared/fragment execution has no Ascend VF scan lowering; complete original body retained.',[('testing/python/language/test_tilelang_language_scan.py','cumsum_fragment_test')])
plan('HadamardTransformFwdOp','deployment_blocked_static','Original Hadamard kernel exists, uses warp shuffle/CUDA MMA shared layout and generic thread scheduling. Exact manifest factory coverage is tracked from the current inventory.',[('examples/hadamard_transform/example_hadamard.py','hadamard')])
plan('GQAVarlenFwdOp GQABwdOp','deployment_blocked_static','Original generic GQA causal/varlen forward/backward exists; its generic T.Parallel/casts are outside VF. Native Ascend backward is explicitly noncausal and cannot substitute for these causal contracts.',
     [('examples/flash_attention/example_gqa_fwd_varlen.py','flashattn'),('examples/flash_attention/example_gqa_bwd.py','flashattn_bwd_split'),('examples/ascend/flash_attention/core_bwd.py','flash_attention_bwd_fused_dq_atomic_staged')])
plan('MLAVarlenFwdOp MLAPagedFwdOp MLADecodeWithKVCacheFwdOp DSAPagedFwdOp DSADecodeWithKVCacheFwdOp NSAVarlenFwdOp NSACompressedVarlenFwdOp NSATopKVarlenFwdOp FP8LightningIndexerFwdOp',
     'deployment_blocked_static','Repository contains matching MLA/NSA/sparse-logit GPU kernels, but their generic/CUDA-specific scheduling, MMA/TMA/warp intrinsics and non-VF loops require backend kernel changes. Optional cache mutation/outputs still require a full contract adapter; no dependency Torch/Triton reference is credited.',
     ['examples/deepseek_v32/sparse_mla_fwd.py','examples/deepseek_v32/fp8_lighting_indexer.py','examples/deepseek_nsa/example_tilelang_nsa_fwd_varlen.py'])
plan('TopKSelectFwdOp','deployment_blocked_static','Original radix topk_selector has starts/ends and index outputs, unlike value-only TopK; its shared histogram/atomics/thread sync and CUDA-style thread binding have no original Ascend schedule.',
     [('examples/deepseek_v32/topk_selector.py','tl_topk_impl')])
plan('SSDChunkStateFwdOp SSDChunkScanFwdOp','deployment_blocked_static','Original chunk-state/scan kernels exist but require FP16 operands and perform casts and Parallel outside VF. FP32 state/dt/dA contracts cannot be silently narrowed; no such rounding is accepted as lossless.',
     [('examples/linear_attention/example_mamba_chunk_state.py','chunk_state_fwd'),('examples/linear_attention/example_mamba_chunk_scan.py','chunk_scan_fwd')])
plan('GemmFP8FwdOp BmmFP8FwdOp MoEGroupedGemmFP8FwdOp','deployment_blocked_static','Generic FP8 GEMM supports floating block scales but requires CUDA scheduling. Native Ascend deepgemm uses UE8M0 power-of-two packed scales, not arbitrary manifest FP32 scales; silently rounding scales would change the contract.',
     ['examples/gemm_fp8/example_tilelang_gemm_fp8.py',('examples/ascend/deepgemm/api.py','_transform_sf_into_required_layout'),'examples/ascend/deepgemm/kernels/transform_sf.py'])
plan('GemmFP8FwdOp BmmFP8FwdOp','supported_unmeasured','Per-tensor scale route: exact FP8-to-BF16 widening and original native BF16 GEMM runtime alpha, with original FP32 accumulation input for bias; floating per-block scales remain blocked without a different original API.',
     [('examples/ascend/deepgemm/api.py','_make_bf16_gemm'),'examples/ascend/deepgemm/api.py'],'fp8.py')
plan('GemmW4A16FwdOp GemmW4A8FwdOp GemmINT8W8A8FwdOp','deployment_blocked_static','Original low-bit GEMM examples exist, with CUDA packed-integer dequantization/matrix instructions; native Ascend example does not expose those scale/zero-point formats.',
     ['examples/dequantize_gemm/example_dequant_gemm_fine_grained.py','examples/dequantize_gemm/example_dequant_gemm_w4a8.py'])
plan('FusedMoEFwdOp FusedMoEExpertsFwdOp FusedMoESharedExpertFwdOp SharedExpertMLPFwdOp MoEExpertMLPFwdOp IndexedExpertMLPFwdOp',
     'deployment_blocked_static','Original fusedMoE kernels implement gate/up/SiLU/down arithmetic, with original generic GEMM and non-VF epilogues. Host routing uses Torch and cannot be credited as native permutation/topk; generic kernel needs backend changes.',
     [('examples/fusedmoe/example_fusedmoe_tilelang.py','moe_forward_tilelang_shared'),('examples/fusedmoe/example_fusedmoe_tilelang.py','moe_forward_tilelang_routed')])
plan('MHCPreFwdOp MHCPostFwdOp','deployment_blocked_static','Original mHC pre/post computation exists; post can select identity residual mixing, but original non-VF casts and Parallel prevent Ascend deployment without schedule changes.',
     [('examples/deepseek_mhc/example_mhc_pre.py','mhc_pre'),('examples/deepseek_mhc/example_mhc_post.py','mhc_post_tilelang')])
plan('DeltaNetFwdOp DeltaNetChunkFwdOp DeltaNetChunkBwdOp DeltaNetRecurrentFwdOp GDNFwdOp GLAChunkFwdOp GLAChunkBwdOp GLAFwdOp GLARecurrentFwdOp KDAFwdOp',
     'unsupported_within_scope','GDN/KDA examples expose intermediate chunk O/H/WY/triangular stages, not the full arbitrary q/k/v/beta/gate/state manifest API. Completing missing stages with FLA/Triton would misattribute a dependency. safe_gated_kda_decode hardcodes Q/K normalization and a bounded gate, unlike a general precomputed-gate contract.',
     ['examples/gdn/README.md','examples/gdn/example_chunk_delta_h.py','examples/kda/chunk_inter_solve_fused.py',('examples/kda/example_safe_gated_kda_decode.py','safe_gated_kda_decode')])
plan('RoPEFwdOp LongRoPEFwdOp RoPELlama31FwdOp RoPENeoxPositionIdsFwdOp YaRNFwdOp FusedQKNormRoPEFwdOp',
     'unsupported_within_scope','No own complete RoPE frequency/scaling/rotation API in the source catalog. Attention receives already-rotated operands; model-side Torch rotation is not a TileLang kernel and cannot replace this op.',
     ['examples/ascend/flash_attention/core.py','examples/deepseek_v32/inference/model.py'])
plan('MoEPrePermuteFwdOp MoEPostPermuteFwdOp MoEPermuteAlignFwdOp FusedTopKFwdOp',
     'unsupported_within_scope','Original fusedMoE host routes with Torch argsort/bincount/indexing; native kernels do expert GEMM/finalize, not the required standalone routing/count/inverse-map or grouped score/topk contract. No Torch routing calculation is credited.',
     ['examples/fusedmoe/example_fusedmoe_tilelang.py'])
plan('TopKMaskFwdOp TopKTopPMaskFwdOp TopPMaskFwdOp MinPMaskFwdOp SamplingFromProbsFwdOp ChainSpeculativeSamplingFwdOp',
     'unsupported_within_scope','Original TopK returns fixed-k values/indices and removes all equal maxima; it does not return the full tie-preserving threshold mask, cumulative-probability mask, or stochastic sampling/chain outputs. Constructing those calculations is more than a layout change.',
     [('examples/topk/example_topk.py','tl_topk'),'examples/deepseek_v32/inference/generate.py'])
plan('ArgmaxFwdOp ArgminFwdOp','unsupported_within_scope','Native generic TopK selects the highest equal-max index, while manifest argmax/argmin requires the first index; selecting/casting TopK outputs does not preserve ties or supply min. No complete matching original arg reduction found.',[('examples/topk/example_topk.py','tl_topk')])
plan('FP8QuantPerBlockFwdOp INT4QuantPerGroupFwdOp INT8DequantPerBlockFwdOp INT8DequantPerChannelFwdOp INT8DequantPerTensorFwdOp INT8QuantPerBlockFwdOp INT8QuantPerChannelFwdOp INT8QuantPerTensorFwdOp SmoothQuantFwdOp FusedAddRMSNormPerTokenQuantFwdOp',
     'unsupported_within_scope','Original Ascend quantizer fixes groups of128 and FP8 scale floor1e-4. This does not implement 128x128 block reduction, INT8/INT4 formats, zero-points or SmoothQuant. Weight dequantization inside GPU GEMM is not a standalone native output; cannot use Torch quantization references.',
     [('examples/ascend/example_per_token_cast_to_fp8.py','per_token_cast_simt'),'examples/deepseek_v4/act_quant.py','examples/dequantize_gemm/dequantize_utils.py'])
plan('FFTC2CFwdOp','unsupported_within_scope','No complete complex64 FFT API in original examples/tests; constructing FFT/DFT stages from primitive GEMM would introduce the missing algorithm.', ['examples/ascend/README.md'])
plan('EngramDecodeFwdOp EngramGateConvFwdOp EngramGateConvBwdOp','unsupported_within_scope','No complete Engram hashing/gating/convolution forward/backward API in the original source catalog.', ['examples/ascend/README.md'])
plan('CausalConv1dFwdOp CausalConv1dUpdateFwdOp Mamba2FwdOp SelectiveScanFwdOp SSDRecurrentFwdOp SSDStatePassingFwdOp SSDChunkCouplingFwdOp SSDChunkCumsumFwdOp',
     'unsupported_within_scope','Original Mamba examples cover chunk_state and chunk_scan only. They do not expose these complete recurrence/state-passing/causal-cache contracts; GDN local cumsum is a different intermediate API. No state algorithm is synthesized.',
     ['examples/linear_attention/example_mamba_chunk_state.py','examples/linear_attention/example_mamba_chunk_scan.py','examples/gdn/example_cumsum.py'])
plan('MHADecodePagedWithKVCacheFwdOp GQAPrefillPagedWithKVCacheFwdOp MLAKVCacheWriteFwdOp PagedKVCacheGatherFwdOp PagedKVCacheWriteFwdOp MergeAttentionStatesFwdOp',
     'unsupported_within_scope','Read-only original attention kernels do not implement the complete cache update/copy-back or LSE state-merge contract. kpool cache kernels use a different compressed pool format; cannot count plain Torch writes as own native cache operations.',
     ['examples/ascend/flash_attention/core.py','examples/kpool/example_glm53_kpool_decode_tail.py'])

plan('ReluFwdOp MaximumFwdOp MinimumFwdOp','supported_unmeasured','Original NaN-propagating clamp with 0 lower bound, tensor lower bound or tensor upper bound; unchanged native clamp body and exact widening.',[('testing/ascend/language/test_tilelang_ascend_clamp.py','test_clamp_nan_and_inverted_bounds')],'microblocks.py')
plan('AbsFwdOp','supported_unmeasured','Original absolute-difference API with constant-zero second operand; independent fixed64 slices and exact FP32 widening. Full manifest device validation is required.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','test_absolute_difference')],'microblocks.py')
plan('DivFwdOp ReciprocalFwdOp','supported_unmeasured','Original precise SIMD divide with unchanged fast-math-disabled configuration; reciprocal uses constant-one numerator. Rounded quotient modes require a separate generic rounding API.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','test_precise_division')],'microblocks.py')
plan('LogicalNotFwdOp','supported_unmeasured','Original XOR with constant1 computes boolean inversion; nonboolean nonzero testing is not implemented by this route.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel')],'microblocks.py')

plan('AmaxFwdOp AminFwdOp AllFwdOp AnyFwdOp','supported_unmeasured','Original native 2x128 min/max reducer; exact widening, axis packing and neutral padding; repeated original reducer calls combine partial extrema. Boolean all/any select min/max on 0/1. No original schedule or arithmetic changes.',[('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')],'reduce.py')

plan('SiluAndMulFwdOp SiluFwdOp','deployment_blocked_static','Original compile-speed silu_gate exposes complete SiLU*up; unit Up selects SiLU and exact FP32 widening avoids an extra intermediate rounding. Its original T.Parallel is outside Ascend VF. Original fusedMoE has the same backend restriction.',[('benchmark/compile_speed/kernel_zoo.py','silu_gate')])
plan('BitwiseAndFwdOp BitwiseOrFwdOp LogicalAndFwdOp LogicalOrFwdOp','deployment_blocked_static','Original generic bitwise_reduce can reduce two stacked int32 operand columns; integer pair views preserve int64 bits and bool inputs preserve 0/1. Its fragment reduction has no Ascend VF, and native Ascend explicitly rejects bitand/bitor collectives; nonboolean logical conversion is a further missing API.',[('testing/python/math/test_math_bitwise_reduce.py','bitwise_reduce'),('testing/ascend/language/test_tilelang_ascend_reduce.py','test_reducer_v2_rejects_unsupported_bitwise_collectives')])

plan('AvgPool1dFwdOp AvgPool2dFwdOp AvgPool3dFwdOp AdaptiveAvgPool2dFwdOp MeanPoolingFwdOp','supported_unmeasured','Original GEMM constant projection over packed pooling regions; shape-derived per-window/tail divisors, including ceil padding and override; ragged metadata is read every call.',[('examples/ascend/example_gemm.py','gemm')],'pool.py')
plan('MaxPool1dFwdOp MaxPool2dFwdOp MaxPool3dFwdOp AdaptiveMaxPool2dFwdOp','supported_unmeasured','Original Ascend max reducer over padded/dilated/adaptive input regions; original shape parameters keep256 threads. Value-only contracts supported; indices contracts are separate.',[('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')],'pool.py')
plan('NegFwdOp','supported_unmeasured','Original XOR with constant IEEE sign-bit mask and lossless dtype views; native kernel does the operation, no external negation.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel')],'bit_layout.py')
plan('BitwiseAndFwdOp BitwiseOrFwdOp LogicalAndFwdOp LogicalOrFwdOp','supported_unmeasured','Boolean 0/1 operands map exactly to original clamp min/max; nonboolean variants still require blocked generic bitwise reduction or a missing nonzero-conversion API.',[('testing/ascend/language/test_tilelang_ascend_clamp.py','test_clamp_nan_and_inverted_bounds'),('testing/python/math/test_math_bitwise_reduce.py','bitwise_reduce')],'bit_layout.py')
plan('SumFwdOp','supported_unmeasured','FP32 input selects original native sum reducer after the GEMM accumulation order failed one case. Other input dtypes retain the validated original GEMM projection. No tolerance change.',[('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel'),('examples/ascend/example_gemm.py','gemm')],'sum_reduction.py')
plan('GQADenseFwdOp GQAPagedFwdOp','supported_unmeasured','Decode/full-context uses original adapter; multi-query causal/window contexts now have exact per-query crop adapter using the same original native attention. Many distinct context lengths may require substantial compilation; softcap/RoPE/FP8 unsupported by this API.',[('examples/ascend/flash_attention/core.py','flash_attention_fwd')],'attention_crops.py')
plan('ArgmaxFwdOp','deployment_blocked_static','Original generic TopK selects the last equal maximum. Reversing the reduction-axis layout and mapping its index through the reversed coordinate table recovers first-index ties without new kernel math. The original generic TopK still fails Ascend VF validation.',[('examples/topk/example_topk.py','tl_topk')])

plan('CausalConv1dFwdOp CausalConv1dUpdateFwdOp','deployment_blocked_static','Convolution windows can map to original GEMM, including causal initial-state layout. Every current manifest case additionally requires SiLU; the original silu_gate API has a non-VF schedule and cannot deploy unchanged. Complete state mutation/outputs would still need full contract verification.',[('examples/ascend/example_gemm.py','gemm'),('benchmark/compile_speed/kernel_zoo.py','silu_gate')])

plan('MulFwdOp SubFwdOp','supported_unmeasured','Original native SIMD FMA with a zero accumulator for multiplication or current x accumulator and exact -1 multiplier for subtraction; original body/schedule unchanged, exact widening and broadcast packing only.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','test_fused_multiply_add')],'fma.py')

plan('ExpFwdOp LogFwdOp SqrtFwdOp RsqrtFwdOp BitwiseAndFwdOp BitwiseOrFwdOp MulFwdOp SubFwdOp','supported_unmeasured','Original complete Ascend merging_kernel API supersedes generic non-VF restriction. All-true predicate selects original SFU/bitwise/multiply/subtract operation; direct vmul preserves signed zero unlike an FMA-with-zero shortcut; integer pair views preserve int64, FP32 widening is exact, Rsqrt composes original sqrt and exact divide.',[('testing/ascend/language/test_tilelang_ascend_simdvf_merging.py','merging_kernel')],'merge_native.py')
plan('WhereFwdOp MaskedFillScalarFwdOp MaskedFillTensorFwdOp','supported_unmeasured','Original masked XOR with a constant-zero operand performs exact conditional bit copy; current false operand binds Old, true operand binds A. No external selection computation.',[('testing/ascend/language/test_tilelang_ascend_simdvf_merging.py','merging_kernel')],'merge_native.py')

plan('LogicalNotFwdOp LogicalAndFwdOp LogicalOrFwdOp AllFwdOp AnyFwdOp CountNonzeroFwdOp','supported_unmeasured','Original native unsigned-mask comparison converts arbitrary inputs to exact0/1. Native AND removes float sign bits, so either signed zero is false; native OR combines integer limbs and both complex components without FP64 narrowing. Original bitwise/reducer APIs then provide logic, all/any or counts.',[('testing/ascend/language/test_tilelang_ascend_simdvf_merging.py','merging_kernel'),('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')],'truth.py')

plan('MaxPool1dIndicesFwdOp MaxPool2dIndicesFwdOp MaxPool3dIndicesFwdOp AdaptiveMaxPool2dIndicesFwdOp','deployment_blocked_static','Original generic TopK(k=1) returns values and indices for the current finite floating manifests. Packed windows, reverse coordinate ordering for first ties, and a static coordinate-table gather can map both outputs. The original TopK has unconditional non-VF Parallel/reductions and cannot deploy unchanged on Ascend; no executable adapter or numerical pass is claimed.',[('examples/topk/example_topk.py','tl_topk'),'tilelang/ascend/analysis/vf_checker.py'])
plan('ArgminFwdOp','deployment_blocked_static','Current manifests use finite FP16/BF16/FP32. Original native IEEE sign flip plus generic TopK(k=1), reversed reduction-axis layout and coordinate-table mapping can express first-index argmin. The original TopK still has unsupported non-VF scheduling; no kernel change or execution claim.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','bitwise_kernel'),('examples/topk/example_topk.py','tl_topk'),'tilelang/ascend/analysis/vf_checker.py'])

plan('LeakyReluFwdOp','supported_unmeasured','Current manifest slope0.01 can select the original fixed0.1 native LeakyReLU API twice; positive values remain unchanged. Kernel body/globalN/schedule unchanged; original tolerance must validate the composed FP32 rounding.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','relu_kernel')],'leaky.py')
plan('PreluFwdOp','unsupported_within_scope','Original native relu_kernel has a fixed0.2 slope for every lane. Current PReLU manifests provide runtime per-channel weights; substituting those weights inside the original body would change kernel mathematics. The standalone original API does not accept them.',[('testing/ascend/language/simdvf/test_simdvf_arithmetic.py','relu_kernel')])
plan('IsfiniteFwdOp','deployment_blocked_static','Original source contains a complete single-element isfinite PrimFunc, and its generic unary test builder accepts the original T.isfinite API. The bulk unary route has unchanged non-VF Parallel and fails the current Ascend semantic checker before scheduling. The single-element CUDA codegen test is not a validated Ascend bulk route; no absence-of-math claim or execution pass is made.',[('testing/python/language/test_tilelang_language_isfinite_codegen.py','_get_isfinite_source'),('testing/python/math/test_math_fast_math.py','run_single_arg_mathop_test'),'tilelang/ascend/analysis/vf_checker.py'])

plan('HardsigmoidFwdOp HardswishFwdOp LerpScalarFwdOp LerpTensorFwdOp PreluFwdOp SigmoidFwdOp SiluFwdOp SiluAndMulFwdOp','supported_unmeasured','Composition of unchanged native merging_kernel arithmetic/SFU APIs: clamp-affine hard activations, runtime-weight interpolation/PReLU and sigmoid/SiLU. Every input transform and intermediate call occurs inside run; no external arithmetic or native body/schedule edits. Original full-manifest numerical validation remains required.',[('testing/ascend/language/test_tilelang_ascend_simdvf_merging.py','merging_kernel')],'composed.py')

# Explicitly screen remaining standalone contracts against related original APIs.
family_notes={
 'elementwise':('No complete own standalone kernel for this specific operation in the scanned source catalog. Vecadd performs only addition; RMS/quant/attention epilogues do not expose this output. Primitive expression definitions and Torch references are not runnable baselines.', ['examples/ascend/example_vecadd.py','testing/ascend/language/simdvf/test_simdvf_arithmetic.py']),
 'pool':('No original pooling API with the required window/divisor/ceil-mode/index outputs in the catalog. Generic convolution and GEMM are not a ready pooling implementation; introducing pooling reduction/index logic is outside layout-only adaptation.', ['examples/convolution/example_convolution.py','examples/ascend/example_gemm.py']),
 'norm':('No original native API exposes this normalization/statistics contract and all of its outputs. RMSNorm lacks centering/batch running-state/backward statistics; LayerNorm alone does not supply the missing contract.', ['examples/ascend/example_rmsnorm.py','examples/norm/layernorm.py']),
 'reduction':('No original complete reduction API exposes this op with the required statistics, scan/product, normalization or dtype behavior. Existing sum/min/max/absolute reductions do not supply the missing arithmetic/outputs without new algorithm logic.', ['testing/ascend/language/test_tilelang_ascend_reduce.py','testing/python/language/test_tilelang_language_reduce.py']),
}
for op in inventory['operators']:
    name=op['op']
    if name not in plans:
        note,refs=family_notes.get(op['family'],('No complete matching native API found in original source catalog; dependent-library/reference implementations are excluded.',['examples/ascend/README.md']))
        plan(name,'unsupported_within_scope',name+': '+note,refs)

# Preserve history as evidence, and reuse only exactly unchanged existing helper routes.
classic=json.loads((ROOT/'TileOPs/benchmarks/ascend/classic_matrix.json').read_text())
history={(r['op'],r['case_id']):r for r in classic['rows'] if r['library']=='tilelang'}
receipts={}
for path in sorted((OUT/'runs').glob('*.json')):
    r=json.loads(path.read_text())
    if 'op' in r and 'case_id' in r:
        receipts.setdefault((r['op'],r['case_id']),[]).append((path,r))
reusable={'AddFwdOp','GemmFwdOp','RMSNormFwdOp','FP8QuantFwdOp'}
rows=[]
invalidated={receipt:batch for batch in json.loads((OUT/'invalidated_batches.json').read_text()).get('batches',[]) for receipt in batch['receipts']} if (OUT/'invalidated_batches.json').exists() else {}
for c in inventory['cases']:
    op=c['op'];key=op,c['case_id'];p=plans[op]
    row={'op':op,'case_id':c['case_id'],'registered':op in PROVIDERS,'status':p['semantic_status'],'reason':p['reason'],
         'sources':p['sources'],'adapter':p['adapter'],'device_executed':False,'device_receipt':None}
    if op in inventory.get('reference_blocked_ops',[]):
        row['shared_inventory_reference_block']=inventory.get('factory_note','Reference reported unavailable by shared inventory; no numerical test is fabricated.')
    if op=='RMSNormFwdOp':
        dimensions=c['params'].get('normalized_shape',c['tensors'].get('x',next(iter(c['tensors'].values())))[0][-1:])
        hidden=__import__('math').prod(dimensions)
        if hidden*10>216*1024:
            row.update(status='deployment_blocked_static',reason='Source rechecked: original RMSNorm has two FP32 full-row buffers plus weight. At minimum rows=1 the selected FP32 path needs12*hidden bytes; even an original16-bit weight path needs at least10*hidden bytes, exceeding SIMT216KiB UB for these>=32768 widths; generic alternate lacks VF. No row/feature padding removes this bound, and kernel schedule is not changed.')
    if op=='GemmFP8FwdOp' and any(__import__('math').prod(c['tensors'][t][0])!=1 for t in ('scale_a','scale_b')):
        row.update(status='deployment_blocked_static',reason='Generic blockscaled FP8 implementation requires CUDA; native UE8M0 conversion cannot exactly represent arbitrary manifest FP32 block scales. Scalar alpha adaptation does not implement this case.')
    h=history.get(key)
    if h:
        row['historical_classic']={'status':h['status'],'receipt':h.get('device_receipt') or h.get('last_receipt') or h.get('receipt'), 'reason':h.get('reason')}
        hp=row['historical_classic']['receipt']
        if op in reusable and h['status']=='verified' and hp and (ROOT/hp).exists():
            old=json.loads((ROOT/hp).read_text());changes=[]
            for file,value in old.get('source_sha256',{}).items():
                f=ROOT/file
                if not f.exists() or sha(f)!=value:changes.append(file)
            row['historical_classic']['current_hash_mismatches']=changes
            check=old.get('verification',{})
            if not changes and check.get('checked_outputs')==check.get('total_outputs') and check.get('checked_outputs',0)>0:
                row.update(status='verified',reason='Retained original full-output/input-preservation receipt; all recorded source hashes still match; same helper factory.',device_executed=True,device_receipt=hp,evidence_origin='retained_classic')
    attempts=receipts.get(key,[])
    if attempts:
        row['attempts']=[str(path.relative_to(ROOT)) for path,r in attempts]
        path,r=attempts[-1]
        row['device_receipt']=str(path.relative_to(ROOT));row['device_attempted']=True
        row['device_executed']=True if r['status']=='verified' else None
        attr=path.with_suffix('.attribution.json')
        if attr.exists():
            evidence=json.loads(attr.read_text());row['native_attribution']=str(attr.relative_to(ROOT));row['native_successful_calls']=evidence['total_successful_calls']
            row['native_launch_observed']=evidence['total_successful_calls']>0
            row['device_executed']=True if r['status']=='verified' and row['native_launch_observed'] else None
            row['native_call_count_semantics']='Successful original JIT host calls; complete device/output/input validation is established only by a verified receipt.'
            if 'reference_compatibility' in evidence:row['reference_compatibility']=evidence['reference_compatibility']
            if 'input_preservation_compatibility' in evidence:row['input_preservation_compatibility']=evidence['input_preservation_compatibility']
        row['evidence_origin']='new_all_pairs'
        if r['status']=='verified':
            row['status']='verified';row['reason']='Full original manifest verification and input preservation passed; original vendor API/source fingerprints in receipt.'
            row['verification']=r['verification'];row['api']=r['api'];row['adaptation']=r['adaptation']
            row['adapter_sources']=[{'file':file,'sha256':value} for file,value in r.get('source_sha256',{}).items() if 'benchmarks/ascend_all_pairs/tilelang/' in file]
            row['receipt_source_mismatches_at_report']=[file for file,value in r.get('source_sha256',{}).items() if not (ROOT/file).is_file() or sha(ROOT/file)!=value]
            row['verified_revision']='Immutable receipt source hashes and available content-addressed snapshots; current-source mismatches are explicitly listed, not hidden.'
            row['ownership_evidence']={'method':'observed_original_JIT_launches' if 'native_attribution' in row else 'audited_direct_original_API_path','api':r['api'],'source_hashes_in_receipt':True,'external_arithmetic_fallback':False}
            row['adapter_snapshots']=[str((OUT/'source_snapshots'/(value+'.py')).relative_to(ROOT)) for file,value in r.get('source_sha256',{}).items() if 'benchmarks/ascend_all_pairs/tilelang/' in file and (OUT/'source_snapshots'/(value+'.py')).exists()]
        elif r['status']=='unsupported':row.update(status='deployment_blocked_static' if any(v in r.get('reason','').lower() for v in ('deployment-blocked','generic','cuda')) else 'unsupported_within_scope',reason=r.get('reason',''))
        elif 'aclnnIsClose' in r.get('traceback','') and 'DT_COMPLEX' in r.get('traceback','') and '_unchanged' in r.get('traceback',''):
            row.update(status='verifier_blocked',reason='Original zero-tolerance input-preservation comparator cannot call NPU isclose for complex tensors. Candidate native calls occurred, but full output/input verification is not established. See verifier_issues.json.',verifier_issue=str((OUT/'verifier_issues.json').relative_to(ROOT)))
        elif op=='AdaptiveMaxPool2dFwdOp' and r.get('status')=='device_sync_failed' and path.name=='008-035-AdaptiveMaxPool2dFwdOp-spp-6x6-float16.json':
            row.update(status='verifier_blocked',reason='Driver exception attributes SIMT stack overflow to the original CANN AdaptiveMaxPool2d reference kernel, not the asynchronously reported TileLang launch. Full validation remains incomplete.',verifier_issue=str((OUT/'oracle_adaptive_max_fault.json').relative_to(ROOT)))
        elif r.get('stage') in ('verify','edge_check') and r.get('error_type')=='AssertionError':row.update(status='numerical_failed',reason=r.get('error',''))
        elif r.get('error_type')=='TimeoutError':row.update(status='resource_blocked',reason=r.get('error',''))
        elif r.get('error_type') in ('KeyError','TypeError','AttributeError','NameError'):row.update(status='adapter_blocked',reason=r.get('error',''))
        else:row.update(status='deployment_blocked',reason=r.get('error') or r.get('reason') or r['status'])
    if op=='MoEGroupedGemmFwdOp' and c['params'].get('activation')=='silu_and_mul':
        row['adapter']=str((HERE/'moe_activation.py').relative_to(ROOT))
        row['sources']=row['sources']+[source('testing/ascend/language/test_tilelang_ascend_simdvf_merging.py','merging_kernel')]
        if row['status']=='deployment_blocked_static':
            row.update(status='supported_unmeasured',reason='Superseding original-native route now composes FP32 expert GEMM with original native SiLU/gate operations, then final dtype cast. Old unsupported receipt is retained as a prior route attempt; new route requires complete verification.',superseded_route_receipt=row.get('device_receipt'),device_receipt=None,device_executed=False)
    if op in ('GQADenseFwdOp','GQAPagedFwdOp') and c['tensors']['q'][1]=='float16' and row['status']!='unsupported_within_scope':
        prior=row['status']
        if prior=='verified':
            row['withdrawn_verified_receipt']=row.get('device_receipt')
            row['withdrawn_numerical_check']=row.pop('verification',None)
            row['withdrawn_ownership_evidence']=row.pop('ownership_evidence',None)
            row['withdrawn_adaptation']=row.pop('adaptation',None)
        row.update(status='deployment_blocked_static',reason='Original Ascend attention fixes BF16 inputs and BF16 probability storage. Prior FP16 source rewriting changed internal probability rounding and is outside allowed adaptations; its numerical receipts are retained but not credited. Unmodified generic FP16 attention remains non-VF blocked.',
                   scope_audit=str((OUT/'attention_scope_correction.json').relative_to(ROOT)),prior_route_status=prior)
    if row.get('device_receipt') in invalidated:
        row.update(status='resource_blocked',reason='Host npu-guard detected external NPU interference and invalidated the entire batch. Existing output comparisons cannot establish an isolated verification pass.',
                   guard_invalidation=invalidated[row['device_receipt']])
    rows.append(row)
ops=[]
for entry in inventory['operators']:
    p=plans[entry['op']];selected=[r for r in rows if r['op']==entry['op']]
    counts=dict(Counter(r['status'] for r in selected))
    status='missing_workload_factory' if not selected else ('verified' if counts=={'verified':len(selected)} else ('partial_verified' if counts.get('verified') else next(iter(counts)) if len(counts)==1 else 'mixed'))
    ops.append(entry|p|{'registered':entry['op'] in PROVIDERS,'status':status,'case_status_counts':counts,'case_ids':[r['case_id'] for r in selected]})
result={'schema':'tilelang-all-pairs/v1','updated_at':datetime.now(timezone.utc).isoformat(),'library':'tilelang',
        'provider_module':'benchmarks.ascend_all_pairs.tilelang','registered_operators':sorted(PROVIDERS),
        'inventory':str(INVENTORY.relative_to(ROOT)),'inventory_sha256':sha(INVENTORY),'scope':{'operators':len(ops),'cases':len(rows)},
        'status_definitions':{'verified':'Full original manifest outputs and input protection with an original TileLang call; see immutable receipt.',
          'supported_unmeasured':'Semantic adapter route exists; no pass or execution is implied.',
          'deployment_blocked_static':'Original matching computation exists but current backend/source constraints block it; unexecuted cases remain marked false.',
          'deployment_blocked':'An actual build/runtime failure receipt exists.',
          'resource_blocked':'Full-case device attempt exhausted its allowance or the host guard invalidated device isolation; no numerical pass is credited.',
          'adapter_blocked':'Local adapter/API binding requires correction; not mathematical unsupported.',
          'verifier_blocked':'Original reference or full input/output comparison cannot execute on this platform; no numerical pass or candidate mismatch is inferred.',
          'numerical_failed':'Executed complete comparison failed at unchanged tolerances.',
          'unsupported_within_scope':'Inspected original APIs lack required math/output contract within authorized adaptations.',
          'missing_workload_factory':'Inventory has zero cases; exact manifest testing cannot be fabricated; semantic analysis retained.'},
        'kernel_ast_audit':str((OUT/'kernel_ast_audit.json').relative_to(ROOT)),
        'source_revision_audit':str((OUT/'source_revision_audit.json').relative_to(ROOT)),
        'backend_constraints':str((OUT/'backend_constraints.json').relative_to(ROOT)),
        'attention_scope_correction':str((OUT/'attention_scope_correction.json').relative_to(ROOT)),
        'verifier_issues':str((OUT/'verifier_issues.json').relative_to(ROOT)) if (OUT/'verifier_issues.json').exists() else None,
        'source_catalog':str(catalog_path.relative_to(ROOT)),'source_catalog_sha256':sha(catalog_path),'source_files_scanned':len(catalog),
        'summary':{'case_status_counts':dict(Counter(r['status'] for r in rows)),'operator_status_counts':dict(Counter(o['status'] for o in ops)),
                   'new_verified_cases':sum(r['status']=='verified' and r.get('evidence_origin')=='new_all_pairs' for r in rows),
                   'retained_verified_cases':sum(r['status']=='verified' and r.get('evidence_origin')=='retained_classic' for r in rows)},
        'missing_workload_factories':sorted(set(inventory.get('missing_factory_ops',[]))|{o['op'] for o in ops if o['status']=='missing_workload_factory'}),
        'shared_inventory_reference_blocked_ops':inventory.get('reference_blocked_ops',[]),
        'operators':ops,'cases':rows,'performance_measured':False,'compiled_artifact_index':str((OUT/'compiled_artifacts.json').relative_to(ROOT)) if (OUT/'compiled_artifacts.json').exists() else None,
        'limitations':['Static backend failure propagation is not a device test for each case.','Global TileLang interpreter primitives are not credited as existing standalone kernels.','Historical receipts are read-only; new adapter paths do not inherit their numerical status.']}
(OUT/'coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
previous=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {}
(OUT/'progress.json').write_text(json.dumps(previous | {'worker':'c3f3396a47d643ec','phase':previous.get('phase','device_validation'),'updated_at':result['updated_at'],'summary':result['summary'],
                                          'all_205_ops_screened':True,'coverage':'coverage.json'},indent=2)+'\n')
print(json.dumps(result['summary'],indent=2))
