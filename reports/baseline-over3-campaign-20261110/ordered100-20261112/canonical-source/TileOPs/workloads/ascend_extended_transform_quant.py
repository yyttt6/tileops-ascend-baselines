"""Transform/quantized-GEMM factories with explicit reference availability.

Hadamard uses independent CPU arithmetic and fixed, source-traced sign matrices.
W4A8 exposes its complete manifest input signature but has no reference until the
project defines its implementation-owned packing and dequantization convention.
No baseline, accelerator runtime, or vendor kernel is imported here.
"""

import math

import torch

from workloads.numerics import Exact, ReferenceInfeasible
from workloads.workload_base import CallWorkload


# Mathematical sign tables, in output-row/input-column order, transcribed from:
# Dao-AILab/fast-hadamard-transform @ e7706faf8d1c3b9f241e36860640ad1dac644ede
# csrc/fast_hadamard_transform_special.h, hadamard_mult_thread_{12,20,28,40}.
# https://github.com/Dao-AILab/fast-hadamard-transform/blob/
# e7706faf8d1c3b9f241e36860640ad1dac644ede/csrc/fast_hadamard_transform_special.h
# These constants specify the convention; no upstream implementation is called.
_HADAMARD_SIGNS = {
    12: (
        "+-++++++++++",
        "--+-+-+-+-+-",
        "+++-++----++",
        "+---+--+-++-",
        "+++++-++----",
        "+-+---+--+-+",
        "++--+++-++--",
        "+--++---+--+",
        "++----+++-++",
        "+--+-++---+-",
        "++++----+++-",
        "+-+--+-++---",
    ),
    20: (
        "+----+----++--++-++-",
        "-+----+---+++---+-++",
        "--+----+---+++-+-+-+",
        "---+----+---+++++-+-",
        "----+----++--++-++-+",
        "-+++++-----+--+++--+",
        "+-+++-+---+-+--+++--",
        "++-++--+---+-+--+++-",
        "+++-+---+---+-+--+++",
        "++++-----++--+-+--++",
        "--++-+-++-+-----++++",
        "---++-+-++-+---+-+++",
        "+---++-+-+--+--++-++",
        "++---++-+----+-+++-+",
        "-++---++-+----+++++-",
        "-+--+--++-+----+----",
        "+-+-----++-+----+---",
        "-+-+-+---+--+----+--",
        "--+-+++------+----+-",
        "+--+--++------+----+",
    ),
    28: (
        "+------++----++-+--+-+--++--",
        "-+-----+++-----+-+--+-+--++-",
        "--+-----+++---+-+-+----+--++",
        "---+-----+++---+-+-+-+--+--+",
        "----+-----+++---+-+-+++--+--",
        "-----+-----++++--+-+--++--+-",
        "------++----++-+--+-+--++--+",
        "--++++-+-------++--+++-+--+-",
        "---++++-+-----+-++--+-+-+--+",
        "+---+++--+----++-++--+-+-+--",
        "++---++---+----++-++--+-+-+-",
        "+++---+----+----++-++--+-+-+",
        "++++--------+-+--++-++--+-+-",
        "-++++--------+++--++--+--+-+",
        "-+-++-++--++--+--------++++-",
        "+-+-++--+--++--+--------++++",
        "-+-+-++--+--++--+----+---+++",
        "+-+-+-++--+--+---+---++---++",
        "++-+-+-++--+------+--+++---+",
        "-++-+-+-++--+------+-++++---",
        "+-++-+---++--+------+-++++--",
        "-++--++-+-++-+++----++------",
        "+-++--++-+-++-+++-----+-----",
        "++-++---+-+-++-+++-----+----",
        "-++-++-+-+-+-+--+++-----+---",
        "--++-++++-+-+----+++-----+--",
        "+--++-+-++-+-+----+++-----+-",
        "++--++-+-++-+-+----++------+",
    ),
    40: (
        "+-------------------+-------------------",
        "++-++----+-+-++++--+++-++----+-+-++++--+",
        "+++-++----+-+-++++--+++-++----+-+-++++--",
        "+-++-++----+-+-++++-+-++-++----+-+-++++-",
        "+--++-++----+-+-+++++--++-++----+-+-++++",
        "++--++-++----+-+-+++++--++-++----+-+-+++",
        "+++--++-++----+-+-+++++--++-++----+-+-++",
        "++++--++-++----+-+-+++++--++-++----+-+-+",
        "+++++--++-++----+-+-+++++--++-++----+-+-",
        "+-++++--++-++----+-++-++++--++-++----+-+",
        "++-++++--++-++----+-++-++++--++-++----+-",
        "+-+-++++--++-++----++-+-++++--++-++----+",
        "++-+-++++--++-++----++-+-++++--++-++----",
        "+-+-+-++++--++-++---+-+-+-++++--++-++---",
        "+--+-+-++++--++-++--+--+-+-++++--++-++--",
        "+---+-+-++++--++-++-+---+-+-++++--++-++-",
        "+----+-+-++++--++-+++----+-+-++++--++-++",
        "++----+-+-++++--++-+++----+-+-++++--++-+",
        "+++----+-+-++++--++-+++----+-+-++++--++-",
        "+-++----+-+-++++--+++-++----+-+-++++--++",
        "+--------------------+++++++++++++++++++",
        "++-++----+-+-++++--+--+--++++-+-+----++-",
        "+++-++----+-+-++++-----+--++++-+-+----++",
        "+-++-++----+-+-++++--+--+--++++-+-+----+",
        "+--++-++----+-+-++++-++--+--++++-+-+----",
        "++--++-++----+-+-+++--++--+--++++-+-+---",
        "+++--++-++----+-+-++---++--+--++++-+-+--",
        "++++--++-++----+-+-+----++--+--++++-+-+-",
        "+++++--++-++----+-+------++--+--++++-+-+",
        "+-++++--++-++----+-+-+----++--+--++++-+-",
        "++-++++--++-++----+---+----++--+--++++-+",
        "+-+-++++--++-++----+-+-+----++--+--++++-",
        "++-+-++++--++-++------+-+----++--+--++++",
        "+-+-+-++++--++-++----+-+-+----++--+--+++",
        "+--+-+-++++--++-++---++-+-+----++--+--++",
        "+---+-+-++++--++-++--+++-+-+----++--+--+",
        "+----+-+-++++--++-++-++++-+-+----++--+--",
        "++----+-+-++++--++-+--++++-+-+----++--+-",
        "+++----+-+-++++--++----++++-+-+----++--+",
        "+-++----+-+-++++--++-+--++++-+-+----++--",
    ),
}


class HadamardTransformCall(CallWorkload):
    """Normalized H_base kron H_radix2 along the complete last dimension.

    For n = base_order * q, the transform is H_base kron H_q. The upstream
    special kernels load q consecutive elements for each base-order component;
    see fast_hadamard_transform_common.h:129-142 and
    fast_hadamard_transform_cuda.cu:189-226 at the revision above. In particular,
    base_order=20 is not a zero-padded power-of-two transform. Other equivalent
    Hadamard matrices with different row/column signs or permutations are not
    interchangeable with these tables.

    All arithmetic and the output dtype conversion occur on CPU. Only the final
    result is copied back to the input device for the shared comparison protocol.
    """

    def arguments(self):
        # Neither manifest entry in this module has constructor tensor arguments.
        return self.call.arguments({})

    def ref_program(self, x):
        base_order = self.call.params["base_order"]
        if base_order not in (1, 12, 20, 28, 40):
            raise ValueError(f"Unsupported Hadamard base_order: {base_order}")
        if x.ndim < 1 or x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Hadamard requires a non-scalar float16/bfloat16 tensor")
        n = x.shape[-1]
        if n <= 0 or n % base_order:
            raise ValueError("Hadamard n must be a positive multiple of base_order")
        q = n // base_order
        if q & (q - 1) or q > 268435456:
            raise ValueError("Hadamard n/base_order must be a power of two <= 2**28")

        source = x.detach().to(device="cpu").contiguous().reshape(-1, n)
        output = torch.empty(source.shape, dtype=x.dtype, device="cpu")
        base_matrix = None
        if base_order != 1:
            base_matrix = torch.tensor(
                [[1 if sign == "+" else -1 for sign in row]
                 for row in _HADAMARD_SIGNS[base_order]],
                dtype=torch.float64,
                device="cpu",
            )

        # Bound FP64 scratch space while retaining every row and every column.
        rows_per_chunk = max(1, (1 << 20) // n)
        normalization = 1.0 / math.sqrt(n)
        for begin in range(0, source.shape[0], rows_per_chunk):
            end = min(begin + rows_per_chunk, source.shape[0])
            rows = end - begin
            values = source[begin:end].to(torch.float64).reshape(rows, base_order, q)
            width = 1
            while width < q:
                pairs = values.reshape(rows, base_order, q // (2 * width), 2, width)
                left, right = pairs[..., 0, :].clone(), pairs[..., 1, :].clone()
                pairs[..., 0, :].copy_(left + right)
                pairs[..., 1, :].copy_(left - right)
                width *= 2
            if base_matrix is not None:
                values = torch.matmul(base_matrix, values)
            output[begin:end].copy_(
                (values.reshape(rows, n) * normalization).to(dtype=x.dtype)
            )
        return output.reshape(x.shape).to(device=x.device)

    def verification(self, *inputs):
        # Keep dtype defaults and additionally guard the complete output's energy.
        return Exact(normalized=1e-3)


W4A8_REFERENCE_BLOCKER = (
    "GemmW4A8FwdOp does not define a portable packed_weight order: gemm.yaml "
    "assigns it to this op's repack, but no GemmW4A8FwdOp/repack implementation "
    "exists in the inspected project sources. The cited vLLM cutlass_w4a8_mm "
    "uses encoded/reordered signed INT4, FP8 lookup-table group scales, and "
    "FP8-rounded dequantized weights; the manifest instead declares uint8 "
    "[N,K/2] weights and BF16 [N,K/group_size] group scales. A documented "
    "byte-to-logical-weight mapping and group-scale/intermediate rounding "
    "contract are required before an independent reference can be supplied."
)


class GemmW4A8Call(CallWorkload):
    """Five physical manifest inputs, with correctness explicitly unestablished.

    This is a factory for case enumeration and an explicit reference blocker,
    not an implemented W4A8 oracle. No layout is inferred from tensor shape or
    from the unrelated W4A16 repacker. No BF16 scale is silently rounded to FP8.
    A consumer must not publish correctness or ratios for this factory.

    Source of the mismatch (read-only inspection, never an execution oracle):
    vllm-project/vllm @ d7ada3aace051fecec553c185225db2259c2ac6a,
    csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu:36-41, 64-77, 380-422;
    tests/kernels/quantization/test_cutlass_w4a8.py:134-145.
    """

    # benchmarks.api.Case supports an absent reference. Returning no reference
    # avoids a fake output and also prevents an unavailable oracle being timed.
    ref_program = None

    def arguments(self):
        return self.call.arguments({})

    def gen_inputs(self):
        tensors = self.call.materialize("cpu")
        # These are opaque physical bytes: generation makes no decoding claim.
        # Cover both nibbles and every code, unlike materialize's default 0..7.
        packed = tensors["packed_weight"]
        tensors["packed_weight"] = torch.randint(
            0, 256, packed.shape, dtype=torch.uint8, device="cpu",
            generator=self.rng("packed_weight"),
        )
        # Positive, varying scales exercise each declared axis without changing
        # its dtype. No output or decoded weight is cached during generation.
        for name in ("weight_group_scale", "weight_channel_scale", "activation_scale"):
            scale = tensors[name]
            tensors[name] = (scale.float().abs() * 0.1 + 0.5).to(dtype=scale.dtype)
        return tuple(tensors[name].to(self.device) for name in self.call.signature.inputs)

    def verification(self, *inputs):
        return ReferenceInfeasible(
            missing="GemmW4A8 repack/dequantization specification",
            reason=W4A8_REFERENCE_BLOCKER,
        )
