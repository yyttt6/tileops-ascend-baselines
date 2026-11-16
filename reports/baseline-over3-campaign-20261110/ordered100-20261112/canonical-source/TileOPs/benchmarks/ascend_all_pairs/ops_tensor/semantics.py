"""Source-grounded screening; enum names and dependencies are not implementations."""
from .build import REG, SOURCES

PUBLIC = "include/cann_ops_tensor.h"
TYPES = "include/cann_ops_tensor_types.h"
COMPARE = f"{REG}/compare/compare.asc"
REDUCE = f"{REG}/reduction_compute/reduction_compute.asc"
GEMM = SOURCES["gemm"]
GROUPED = "examples/grouped_matmul/grouped_matmul_no_quant/grouped_matmul_no_quant.cpp"
MX = "examples/quant_batch_matmul/quant_batch_matmul_mx/quant_batch_matmul_mx.cpp"
CUBE = "examples/quant_batch_matmul/quant_batch_matmul_cube/quant_batch_matmul_cube.cpp"
MIX = "examples/quant_batch_matmul/quant_batch_matmul_mix/quant_batch_matmul_mix.cpp"
WQ = "examples/weight_quant_batch_matmul/weight_quant_batch_matmul_antiquant/weight_quant_batch_matmul_antiquant.cpp"
W4 = "examples/weight_quant_matmul/quant_batch_matmul_pergroup_a8w4/quant_batch_matmul_pergroup_a8w4.cpp"
SWIGLU = "include/blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
GELU = "include/blaze/epilogue/tile/arch35/gelu.h"
RMS = "include/blaze/epilogue/tile/arch35/rms_softmax.h"
ATTN = "examples/block_attn_res_prepare/block_attn_res_prepare/block_attn_res_prepare.cpp"
FLAT = "examples/flat_quant/flat_quant/flat_quant.cpp"

# Anchors identify actual definitions, not inferred support from documentation.
ANCHORS = {
    PUBLIC: "ACLTENSOR_API acltensorStatus_t acltensorElementwiseBinaryExecute(",
    TYPES: "ACLTENSOR_OP_IDENTITY =",
    COMPARE: "auto dst_reg = asc::te::experimental::select((src0_reg > src1_reg)",
    REDUCE: "constexpr uint32_t input_length = 64;",
    GEMM: "__global__ __aicore__ void matmul_basic_kernel",
    GROUPED: "struct GmmNoQuantTilingData",
    MX: "Both A and B are MX-quantized",
    CUBE: "const bool validTypeAndMode",
    MIX: "__global__ __aicore__ void quant_batch_matmul_mix_kernel",
    WQ: "using Policy = Blaze::Gemm::MatmulWithWeightAntiquant",
    W4: "using AType = fp8_e4m3fn_t;",
    SWIGLU: "Consumes concatenated left/right MMAD results",
    GELU: "class Gelu",
    RMS: "class RmsSoftmax",
    ATTN: "__global__ __aicore__ void BlockAttnResPrepareKernel",
    FLAT: "__global__ __aicore__ void FlatQuantKernel",
    SOURCES["add"]: "__global__ __vector__ void add(",
    SOURCES["trunc"]: "__global__ __vector__ void trunc(",
}


def unsupported_assessment(op, family, params):
    """Explain absent complete mathematics or non-layout changes needed for this exact op."""
    if op in {"GemmFP8FwdOp", "BmmFP8FwdOp", "MoEGroupedGemmFP8FwdOp"}:
        return (
            "FP8 contraction variants inspected: MX uses E8M0 power-of-two K/32 scales, cube uses hifloat8, and MIX uses INT8. "
            "The manifest supplies E4M3FN/E5M2 with arbitrary FP32 tensor/row/block scales. Converting scales to E8M0 is lossy; "
            "a new FP32-scale epilogue or external arithmetic would be required, not shape/layout conversion.", [MX, CUBE, MIX])
    if op == "GemmINT8W8A8FwdOp":
        return (
            "Original MIX is INT8*INT8->FP16 with two FP32 scales, but its whole-problem l0TileShape={m,n,k} "
            "cannot hold manifest K=4096..28672 and output is hardcoded half, whereas all manifest outputs are BF16. "
            "A FP16 intermediate is lossy/overflow-prone. Cube default outputs INT32 without both scale vectors; "
            "fixing either route requires kernel tiling/epilogue changes, outside this task.", [MIX, CUBE])
    if op == "GemmW4A16FwdOp":
        return (
            "Original floating-activation antiquant API consumes INT8 weights with per-tensor/per-channel scale/offset; "
            "manifest is packed unsigned INT4 with independent scale/zero point per K group. The per-group A8W4 example "
            "instead consumes FP8 activations and emits INT8. Neither retains the required arbitrary K-group affine dequantization "
            "and floating output through packing alone; no external dequant or split-K accumulation is substituted.", [WQ, W4])
    if op == "GemmW4A8FwdOp":
        return (
            "Native A8W4 per-group kernel emits quantized INT8 and uses its own antiquant/MSD format. The manifest requires "
            "BF16 output after arbitrary group, channel and token scales. Selecting it would require replacing its epilogue; "
            "MX variants' E8M0/FP4 formats cannot losslessly encode this INT4/FP32-scale contract.", [W4, MX])
    if op in {"MaximumFwdOp", "MinimumFwdOp", "ReluFwdOp", "GtFwdOp", "GeFwdOp", "LtFwdOp", "LeFwdOp", "EqFwdOp", "NeFwdOp", "WhereFwdOp"}:
        return (
            "The compare example stores select(x>y,x,y), or select(x>0,x,y), not a Boolean comparison/independent condition. "
            "It chooses y for x=NaN, violating torch maximum/minimum and NaN-propagating ReLU semantics. "
            "Changing predicate/store/NaN logic is a kernel-math change; this example is not a complete equivalent API.", [COMPARE, PUBLIC])
    if family == "elementwise":
        if op in {"GeluFwdOp", "GeluAndMulFwdOp", "GeluTanhAndMulFwdOp", "ErfFwdOp"}:
            return (
                "The library has UB-only GeluTanh/GeluErf tile methods, but no standalone GM-input global example for this manifest operation. "
                "GELU fusion entries consume matmul fixpipe results or produce MX quantized output; adding a GM/UB scheduling kernel or "
                "extracting internal erf/gating math would exceed adapter-only scope.", [GELU, SWIGLU, PUBLIC])
        if op in {"SiluFwdOp", "SiluAndMulFwdOp"}:
            return (
                "The original SwiGLU epilogue consumes GEMM accumulators and writes quantized MXFP8 plus E8M0 scales, "
                "not the manifest's floating SiLU/SwiGLU tensor. Removing quantization or exposing its intermediate changes native math/schedule.",
                [SWIGLU, PUBLIC])
        return (
            f"No complete original {op} entry in the audited include/examples corpus. The C-style elementwise API is declarations only "
            "(acltensorElementwiseBinaryExecute has no definition); enum members are not executable implementations. "
            "Register-level Tensor API primitives would require a new computation/scheduling kernel for this operation; "
            "the existing add/compare/trunc/reduction examples do not compute this contract.", [PUBLIC, TYPES, SOURCES["add"], COMPARE])
    if family == "reduction":
        if op in {"AmaxFwdOp", "AminFwdOp", "SumFwdOp"}:
            tensor = params.get("x")
            extent = ""
            if tensor:
                from math import prod
                shape = tensor[0]
                dims = params.get("dim")
                dims = list(range(len(shape))) if dims is None else ([dims] if isinstance(dims, int) else dims)
                extent = f" This case reduces {prod(shape[d] for d in dims)} values per result."
            return (
                "The only complete reduction example loads one fixed 64-float register and returns three scalars (sum/max/min). "
                "Manifest axes exceed 64; changing the literal cannot extend that single-register load. A new multi-pass reduction tree "
                "would change the original reduction schedule, not just its layout." + extent, [REDUCE])
        return (
            f"No complete {op} reduction/scan in the original entry points. The 64-element example returns only sum/max/min, "
            "without indices, Boolean/count output, statistics, cumulative outputs or normalization. RmsSoftmax is a UB-only "
            "RMS-normalized attention-score exponential/statistics substage, not a complete generic Softmax/LogSoftmax result.", [REDUCE, RMS])
    if family == "norm":
        return (
            f"No complete {op} norm forward/backward API. BlockAttnResPrepare computes RMS-normalized residual-query scores "
            "then softmax-weighted residual statistics, using fixed epsilon and no manifest affine-weight/norm-output interface. "
            "The internal RmsSoftmax method writes exponentials/max/sum rather than a standalone normalized tensor; "
            "removing/extracting fusion stages would change kernel math/schedule.", [ATTN, RMS])
    if family == "pool":
        return (
            f"No original {op} pooling entry. Dense GEMM/Conv2D and a single-register reduction do not provide the manifest "
            "pooling-window normalization/selection, adaptive boundaries, ceil-mode/padding divisor or index semantics. "
            "A new pooling reduction composition would be additional kernel mathematics, not a layout adapter.", [GEMM, REDUCE])
    if family == "quantization":
        return (
            f"No complete {op} standalone API with the manifest's scales/zero points and output dtype. FlatQuant performs "
            "P1@X@P2 then MX-FP4 E2M1; other native quantizers are fused GEMM epilogues with E8M0 block scales. "
            "They cannot substitute INT4 affine/INT8/FP8 scalar-or-block quant/dequant outputs by a lossless representation change.", [FLAT, MX, SWIGLU])
    if family == "rope":
        return (f"No rotary-position kernel for {op} in the original entry-point corpus. Dense contractions and vector examples "
                "do not compute positional frequencies, layout-specific pair rotations or the model-specific scaling; "
                "implementing those outside the library would replace target mathematics.", [GEMM, PUBLIC])
    if family == "attention":
        return (
            f"No complete {op} path. BlockAttnResPrepare shares a residual tensor as both normalized key and value and returns "
            "unnormalized weighted output/max/sum; it is not independently supplied Q/K/V attention, cache mutation/gather, "
            "top-k indexing or base-2 state merging. Required masks, layouts, outputs and normalization cannot be recovered "
            "from it using only the allowed layout/crop/API changes.", [ATTN, RMS])
    if family == "linear_attention":
        return (f"No original {op} recurrent/chunk forward/backward implementation or state-update API. Dense GEMM "
                "does not supply delta/gated recurrent state transitions, decay/prefix operations and complete final states; "
                "assembling them would introduce new mathematical computation.", [GEMM, ATTN])
    if family == "mamba":
        if op.startswith("CausalConv1d"):
            return (
                "Sliding-window convolution and history copies are layout-expressible, but every manifest case requests SiLU. "
                "The only SwiGLU route quantizes its output; no floating SiLU API completes the convolution plus updated state. "
                "Returning only the convolution or using torch SiLU would violate the complete output contract.", [GEMM, SWIGLU])
        return (f"No complete {op} scan/state-update API. Only SSDChunkCoupling reduces to layout plus independent GEMM; "
                "this operation additionally requires exponential decay, prefix/recurrent accumulation, cross-chunk or sequence-boundary "
                "state semantics absent from the original examples. No external torch computation is substituted.", [GEMM, REDUCE])
    if family == "moe":
        return (
            f"The library exposes dense/grouped GEMM and quantized SwiGLU/finalize-routing fusion, not the complete {op} "
            "floating routing/permutation/top-k/MLP contract. Quantized intermediates are lossy and do not preserve the manifest "
            "floating gate/up/down outputs or required permutation/metadata results.", [GROUPED, SWIGLU])
    if family == "sampling":
        return (f"No original {op} probability-mask/sampling path, rank selection or RNG sequence contract. "
                "Register compare and 64-value min/max have no top-k/top-p cumulative probabilities, categorical draw or rejection logic.",
                [COMPARE, REDUCE])
    if family == "fft":
        return ("No complex C2C FFT entry or butterfly schedule in ops-tensor. Dense real GEMM has no FFT plan, complex layout "
                "or normalization contract; synthesizing a Fourier transform would add an algorithm outside adapter scope.", [GEMM, PUBLIC])
    if family == "transform":
        return ("FlatQuant accepts transform matrices but always emits MXFP4 values/scales; it exposes no floating normalized "
                "Walsh-Hadamard output. A newly constructed Hadamard/GEMM algorithm is not an existing transform API.", [FLAT])
    if family == "sequence_modeling":
        return (f"No complete {op} sequence-model path. Existing GEMM/attention-residual primitives do not supply the "
                "manifest's hash/gather/gating/convolution or hyperconnection/Sinkhorn transforms and all auxiliary outputs.", [ATTN, GEMM])
    raise ValueError(f"Unreviewed family: {family}/{op}")
