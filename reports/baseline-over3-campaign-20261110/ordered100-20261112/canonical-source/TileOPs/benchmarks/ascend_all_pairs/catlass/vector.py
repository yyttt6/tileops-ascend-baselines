"""Call original CATLASS TLA DSL vector examples without rewriting their kernels.

Only original config APIs select operation/dtype/shape. Large inputs are packed
into independent chunks, and padded/cropped; every input transform is in run.
The current source snapshot lacks catlass/_mlir, so these factories fail with a
specific deployment error until the *same-source* DSL toolchain is installed.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import uuid

from .build import CACHE, VENDOR, hashes

DSL = VENDOR / "python/tla_dsl"
EXAMPLES = DSL / "examples/end_to_end/vector_ops"
CHUNK = 1024
UNARY = {"AbsFwdOp": "abs", "NegFwdOp": "neg", "ExpFwdOp": "exp", "LogFwdOp": "log", "SqrtFwdOp": "sqrt"}
BINARY = {"AddFwdOp": "add", "SubFwdOp": "sub", "MulFwdOp": "mul", "DivFwdOp": "div",
          "MaximumFwdOp": "max", "MinimumFwdOp": "min", "ReciprocalFwdOp": "div"}
BITWISE = {"BitwiseNotFwdOp": 4, "BitwiseAndFwdOp": 5, "BitwiseOrFwdOp": 6, "BitwiseXorFwdOp": 7,
           "LogicalNotFwdOp": 7, "LogicalAndFwdOp": 5, "LogicalOrFwdOp": 6}
VECTOR_OPS = tuple(UNARY) + tuple(BINARY) + tuple(BITWISE) + ("ReluFwdOp",)


def _load_original(stem, *, source_file=None):
    if not (DSL / "catlass/_mlir").is_dir():
        raise RuntimeError(f"CATLASS DSL deployment blocked: missing generated bindings {DSL / 'catlass/_mlir'}; no substitute installed package or prebuilt cross-host library is accepted.")
    sys.path.insert(0, str(DSL))
    sys.path.insert(0, str(EXAMPLES))
    try:
        catlass = importlib.import_module("catlass")
        if Path(catlass.__file__).resolve() != (DSL / "catlass/__init__.py").resolve():
            raise RuntimeError("CATLASS resolved outside the assigned vendor source")
        tla = importlib.import_module("catlass.tla")
        harness = importlib.import_module("vector_op_harness")
        if Path(harness.__file__).resolve() != (EXAMPLES / "vector_op_harness.py").resolve():
            raise RuntimeError("vector_op_harness resolved outside the inspected vendor source")
        # Each candidate owns its original module's compile-time config globals.
        name = "_catlass_all_pairs_" + stem + "_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, source_file or EXAMPLES / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    return tla, module


def vector(case, *, op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    values = case.inputs
    p = case.params
    x = values[0]
    if op not in VECTOR_OPS:
        raise UnsupportedCase(f"No inspected vector API for {op}")
    if p.get("alpha", 1) != 1 or p.get("rounding_mode") is not None:
        raise UnsupportedCase("Original binary entry has no non-unit alpha or rounded-division option")
    logical = op.startswith("Logical")
    if logical and any(v.dtype != torch.bool for v in values):
        raise UnsupportedCase("Logical arithmetic on numeric inputs needs a truth-test kernel, not a lossless layout conversion")
    bitwise = op in BITWISE
    int64_bits = bitwise and x.dtype == torch.int64
    if bitwise:
        if any(v.dtype != x.dtype for v in values) or x.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
            raise UnsupportedCase("Bitwise inputs must have equal bool/integer dtype")
        native_dtype, dtype_name = torch.int32, "i32"
    else:
        if any(v.dtype != x.dtype for v in values) or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise UnsupportedCase("This arithmetic entry uses lossless FP16/BF16 -> FP32 widening; no lossy integer/float conversions")
        native_dtype, dtype_name = torch.float32, "f32"
    output_spec = p.get("output")
    shape = tuple(output_spec[0]) if output_spec else tuple(torch.broadcast_shapes(*(v.shape for v in values)))
    count = 1
    for extent in shape:
        count *= extent
    raw_count = count * (2 if int64_bits else 1)
    padded = ((raw_count + CHUNK - 1) // CHUNK) * CHUNK
    if op in UNARY:
        stem, kernel_name, operation, input_count, output_count, selected = "unary_ops", "vector_unary", UNARY[op], 1, 4, 0
    elif op in BINARY:
        stem, kernel_name, operation, input_count, output_count, selected = "binary_op", "binary_op", BINARY[op], 2, 1, 0
    elif bitwise:
        stem, kernel_name, operation, input_count, output_count, selected = "bitwise_ops", "bitwise_ops", "bitwise_ops", 2, 8, BITWISE[op]
        if op == "BitwiseNotFwdOp" and x.dtype == torch.bool:
            selected = 7  # bool NOT = bitwise XOR with representation of true
    else:
        stem, kernel_name, operation, input_count, output_count, selected = "compare_mask", "compare_mask", "vector_scalar_gt", 2, 1, 0
    tla, module = _load_original(stem)
    module._set_kernel_config(operation, dtype_name, (CHUNK,))
    samples = [torch.empty(CHUNK, dtype=native_dtype, device=x.device)
               for _ in range(input_count + output_count)]
    def runtime(t):
        return tla.from_dlpack(t, layout_tag=tla.arch.RowMajor, origin_shape=(CHUNK,))
    old_cache = os.environ.get("CATLASS_DSL_CACHE_DIR")
    os.environ["CATLASS_DSL_CACHE_DIR"] = str(CACHE / "dsl")
    try:
        artifact = tla.compile(getattr(module, kernel_name), *(runtime(t) for t in samples), options="--npu-arch 3510")
    finally:
        if old_cache is None:
            os.environ.pop("CATLASS_DSL_CACHE_DIR", None)
        else:
            os.environ["CATLASS_DSL_CACHE_DIR"] = old_cache
    source_paths = [Path(__file__), EXAMPLES / f"{stem}.py", EXAMPLES / "vector_op_harness.py"]
    source_paths += list((DSL / "catlass").rglob("*.py"))
    source_paths += list((DSL / "catlass").rglob("*.so"))
    binary = Path(artifact.kernel_binary_path).resolve()
    source_sha256 = hashes(source_paths + [binary])

    def pack(value):
        value = value.expand(shape).contiguous()
        value = value.view(torch.int32).flatten() if int64_bits else value.to(native_dtype).flatten()
        if value.numel() != raw_count:
            raise ValueError("Vector input shape changed after candidate construction")
        # Packing never mutates inputs, including when expand returns a view.
        data = torch.zeros(padded, dtype=native_dtype, device=value.device)
        data[:raw_count].copy_(value)
        return data

    def run(*values):
        inputs = [pack(value) for value in values]
        if op == "ReciprocalFwdOp":
            inputs.insert(0, torch.ones(padded, dtype=native_dtype, device=x.device))
        elif op in ("LogicalNotFwdOp", "BitwiseNotFwdOp"):
            inputs.append(torch.ones(padded, dtype=native_dtype, device=x.device))
        elif op == "ReluFwdOp":
            inputs.append(torch.zeros(padded, dtype=native_dtype, device=x.device))
        output = torch.empty(padded, dtype=native_dtype, device=x.device)
        scratch = [torch.empty(CHUNK, dtype=native_dtype, device=x.device) for _ in range(output_count)]
        for start in range(0, padded, CHUNK):
            args = [v[start:start + CHUNK] for v in inputs] + scratch
            artifact(*(runtime(v) for v in args), block_num=1,
                     stream=torch.npu.current_stream(x.device).npu_stream)
            output[start:start + CHUNK].copy_(scratch[selected])
        output = output[:raw_count]
        if int64_bits:
            return output.view(torch.int64).reshape(shape)
        return output.to(x.dtype).reshape(shape)

    return Candidate(Implementation(run), f"CATLASS TLA DSL {stem}.{kernel_name} / {operation} / output {selected}",
                     "Original config API, unchanged kernel/schedule; broadcast/contiguous packing and independent chunks with padding/crop inside run. FP16/BF16 widen exactly to FP32; int64 bitwise uses two raw int32 lanes; full original multi-output kernel is timed.",
                     source_sha256, timing_mode="host_roundtrip", limitations=("No current-host numerical validation; DSL generated bindings absent in current snapshot.", "Chunk launches are correctness-oriented and may be expensive at large manifest sizes.",))
