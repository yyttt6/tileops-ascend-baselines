"""Ascend library resolvers and small adapters for the classic benchmark cases.

Like ``benchmarks.baselines``, a named provider must resolve its own implementation.
No torch fallback is permitted. Runtime imports and allocations belong inside the
host npu-guard reservation. Original library kernels and schedules are unchanged.
"""

from dataclasses import dataclass
import hashlib
import importlib
from math import prod
from pathlib import Path
import sys
from typing import Callable


WORKSPACE = Path(__file__).resolve().parents[2]
LIBRARIES = WORKSPACE / "ascend_baselines"


class UnsupportedCase(ValueError):
    """The inspected original API cannot express this case within the scope."""


@dataclass(frozen=True)
class Candidate:
    implementation: object
    api: str
    adaptation: str
    source_sha256: dict[str, str]
    limitations: tuple[str, ...] = ()
    timing_mode: str = "device_async"
    cleanup: Callable | None = None


def source_hashes(*paths):
    return {
        str(Path(p).relative_to(WORKSPACE)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in paths
    }


def local_function(library, package_root, module, name, source):
    """Import the original package and reject an installed namesake."""
    root = LIBRARIES / library
    sys.path.insert(0, str(root / package_root))
    loaded = importlib.import_module(module)
    expected = (root / source).resolve()
    if Path(loaded.__file__).resolve() != expected:
        raise RuntimeError(f"{module} resolved outside the selected source: {loaded.__file__}")
    fn = getattr(loaded, name)
    if Path(fn.__code__.co_filename).resolve() != expected:
        raise RuntimeError(f"{module}.{name} is not the inspected original implementation")
    return fn


def _add_layout(case):
    import torch

    x, y = case.inputs
    if case.params.get("alpha", 1) != 1:
        raise UnsupportedCase("These residual-add specializations require alpha=1.")
    if x.dtype != y.dtype or x.device != y.device:
        raise UnsupportedCase("The original API requires equal input dtypes/devices.")
    shape = tuple(torch.broadcast_shapes(x.shape, y.shape))
    if not shape or min(shape) < 1:
        raise UnsupportedCase("The inspected row APIs require a nonempty output.")
    hidden = prod(shape[-2:]) if len(shape) > 2 else shape[-1]
    rows = prod(shape) // hidden

    def pack(value):
        # Materialize broadcast/stride adaptations on EVERY invocation. They are
        # part of the candidate's measured work, never a precomputed input.
        return value.expand(shape).contiguous().view(rows, hidden)

    return x, y, shape, rows, hidden, pack


def tilekernels_add(case):
    import torch
    from benchmarks.api import Implementation

    x, y, shape, rows, hidden, pack = _add_layout(case)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase("The Add adapter supports FP16/BF16/FP32 inputs.")
    host = "tile_kernels/quant/norm_forward_kernel.py"
    fn = local_function("TileKernels", ".", "tile_kernels.quant.norm_forward_kernel",
                        "norm_forward", host)
    native_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
    out = torch.empty((rows, hidden), dtype=native_dtype, device=x.device)
    summed = torch.empty_like(out)

    def run(x, y):
        value = fn(pack(x).to(native_dtype), None, 1e-6,
                   residual=pack(y).to(native_dtype), out=out, residual_out=summed)[2]
        return value.to(x.dtype).view(shape)

    return Candidate(
        Implementation(run), "tile_kernels.quant.norm_forward(...)[2]",
        "Broadcast/contiguous/2D view; FP16 widens exactly to native FP32; select residual sum and cast back; full norm is timed.",
        source_hashes(LIBRARIES / "TileKernels" / host,
                      LIBRARIES / "TileKernels/tile_kernels/quant/norm_forward_asc.py"),
    )


def sgl_add(case):
    from benchmarks.api import Implementation

    _, _, shape, _, _, pack = _add_layout(case)
    source = "python/sgl_kernel_npu/sgl_kernel_npu/moe/mul_add.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.moe.mul_add", "mul_add", source)

    def run(x, y):
        return fn(pack(y), pack(x), 1).view(shape)

    return Candidate(
        Implementation(run), "sgl_kernel_npu.moe.mul_add.mul_add(y,x,1)",
        "Broadcast/contiguous/2D view; original a*1+b API; no kernel changes.",
        source_hashes(LIBRARIES / "sgl-kernel-npu" / source,
                      LIBRARIES / "sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py"),
    )


def vllm_native():
    import os
    import torch

    root = WORKSPACE / ".tools/python/vllm-ascend/vllm_ascend"
    # baseline-run's installed set_env.bash points at the source-tree OPP.
    # Check and fingerprint the location actually selected by that environment.
    vendor = LIBRARIES / "vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    configured = [Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p]
    if vendor.resolve() not in configured:
        raise RuntimeError("Use baseline-run vllm-ascend to select this library's custom OPP.")
    libraries = list(root.glob("vllm_ascend_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one deployed vLLM native library, found {len(libraries)}")
    torch.ops.load_library(str(libraries[0]))
    return torch.ops._C_ascend, libraries[0], vendor


def vllm_add(case):
    import torch
    from benchmarks.api import Implementation

    x, _, shape, _, hidden, pack = _add_layout(case)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase("AddRmsNormBias supports FP16/BF16/FP32.")
    ops, binary, vendor = vllm_native()
    gamma = torch.ones(hidden, dtype=x.dtype, device=x.device)

    def run(x, y):
        return ops.npu_add_rms_norm_bias(pack(x), pack(y), gamma, None, 1e-6)[2].view(shape)

    native = LIBRARIES / "vllm-ascend/csrc/moe/add_rms_norm_bias"
    return Candidate(
        Implementation(run), "torch.ops._C_ascend.npu_add_rms_norm_bias(...)[2]",
        "Broadcast/contiguous/2D view; select original residual sum; full fused op is timed.",
        source_hashes(binary, vendor / "op_api/lib/libcust_opapi.so",
                      LIBRARIES / "vllm-ascend/csrc/torch_binding.cpp",
                      native / "add_rms_norm_bias_torch_adpt.h",
                      native / "op_host/add_rms_norm_bias_def.cpp",
                      native / "op_kernel/arch35/add_rms_norm_bias_regbase.h"),
    )


def _rms_layout(case):
    import torch

    x, weight = case.inputs
    hidden = prod(case.params["normalized_shape"])
    shape = tuple(x.shape)
    eps = case.params.get("eps")
    if eps is None:
        # Match the manifest's FP32-reference epsilon, not finfo(x.dtype).
        eps = torch.finfo(torch.float32).eps
    unit_weight = torch.ones(hidden, device=x.device, dtype=x.dtype) if weight is None else None

    def pack(x, weight):
        gamma = unit_weight if weight is None else weight.contiguous().view(hidden)
        return x.contiguous().view(-1, hidden), gamma

    return x, shape, hidden, eps, pack


def tilekernels_rms(case):
    import torch
    from benchmarks.api import Implementation

    x, shape, _, eps, pack = _rms_layout(case)
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise UnsupportedCase("norm_forward supports BF16/FP32; no external floating cast.")
    host = "tile_kernels/quant/norm_forward_kernel.py"
    fn = local_function("TileKernels", ".", "tile_kernels.quant.norm_forward_kernel",
                        "norm_forward", host)

    def run(x, weight):
        x2, gamma = pack(x, weight)
        return fn(x2, gamma, eps)[0].view(shape)

    return Candidate(Implementation(run), "tile_kernels.quant.norm_forward(...)[0]",
                     "Collapse normalized dimensions and batch dimensions; original RMSNorm.",
                     source_hashes(LIBRARIES / "TileKernels" / host,
                                   LIBRARIES / "TileKernels/tile_kernels/quant/norm_forward_asc.py"))


def sgl_rms(case):
    import torch
    from benchmarks.api import Implementation

    x, shape, hidden, eps, pack = _rms_layout(case)
    host = "python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_bias.py"
    fn = local_function("sgl-kernel-npu", "python/sgl_kernel_npu",
                        "sgl_kernel_npu.norm.rmsnorm_bias", "rmsnorm_bias", host)
    bias = torch.zeros(hidden, dtype=x.dtype, device=x.device)

    def run(x, weight):
        x2, gamma = pack(x, weight)
        return fn(x2, gamma, bias, eps).view(shape)

    return Candidate(Implementation(run), "sgl_kernel_npu.norm.rmsnorm_bias.rmsnorm_bias",
                     "2D view and zero bias; quantization disabled through original optional arguments.",
                     source_hashes(LIBRARIES / "sgl-kernel-npu" / host,
                                   LIBRARIES / "sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/utils/triton_utils.py"))


def vllm_rms(case):
    import torch
    from benchmarks.api import Implementation

    x, shape, hidden, eps, pack = _rms_layout(case)
    ops, binary, vendor = vllm_native()
    residual = torch.zeros((x.numel() // hidden, hidden), dtype=x.dtype, device=x.device)

    def run(x, weight):
        x2, gamma = pack(x, weight)
        return ops.npu_add_rms_norm_bias(x2, residual, gamma, None, eps)[0].view(shape)

    native = LIBRARIES / "vllm-ascend/csrc/moe/add_rms_norm_bias"
    return Candidate(
        Implementation(run), "torch.ops._C_ascend.npu_add_rms_norm_bias(...)[0]",
        "2D view; zero residual; use existing Ascend950 fused RMSNorm implementation.",
        source_hashes(binary, vendor / "op_api/lib/libcust_opapi.so",
                      LIBRARIES / "vllm-ascend/csrc/torch_binding.cpp",
                      native / "add_rms_norm_bias_torch_adpt.h",
                      native / "op_kernel/arch35/add_rms_norm_bias_regbase.h"),
    )


def deepgemm_gemm(case):
    import torch
    from benchmarks.api import Implementation

    a, b = case.inputs
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise UnsupportedCase("The original native BF16 GEMM requires BF16 operands.")
    root = LIBRARIES / "DeepGEMM-Ascend"
    build = root / f"build/lib.linux-x86_64-cpython-{sys.version_info.major}{sys.version_info.minor}"
    sys.path.insert(0, str(build))
    module = importlib.import_module("deep_gemm")
    native = Path(module._C.__file__).resolve()
    if not Path(module.__file__).resolve().is_relative_to(build) or not native.is_relative_to(build):
        raise RuntimeError("DeepGEMM resolved outside its own original build.")
    trans_a, trans_b = case.params.get("trans_a", False), case.params.get("trans_b", False)
    suffix = ("t" if trans_a else "n") + ("t" if trans_b else "n")
    fn = getattr(module, "bf16_gemm_" + suffix)
    if fn is not getattr(module._C, "bf16_gemm_" + suffix):
        raise RuntimeError("Public GEMM resolved to a different implementation.")
    m, n = a.shape[1 if trans_a else 0], b.shape[0 if trans_b else 1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def run(a, b):
        fn(a.contiguous(), b.contiguous(), out)
        return out

    return Candidate(
        Implementation(run), "deep_gemm.bf16_gemm_" + suffix,
        "Use the original transpose-aware dense GEMM API; contiguous copies stay in run.",
        source_hashes(native, Path(module.__file__), root / "csrc/apis/gemm.hpp",
                      root / "csrc/jit_kernels/bf16_gemm.hpp",
                      root / "deep_gemm/include/deep_gemm/bf16_gemm.hpp"),
    )


def deepselect_amax(case):
    import torch
    from benchmarks.api import Implementation

    (x,) = case.inputs
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase("Ascend TopK is a 16-bit selector; no numeric dtype cast.")
    root = LIBRARIES / "DeepSelect"
    build = root / f"build/lib.linux-x86_64-cpython-{sys.version_info.major}{sys.version_info.minor}"
    sys.path.insert(0, str(build))
    module = importlib.import_module("deep_select")
    native = Path(module.interface._backend.__file__).resolve()
    if not Path(module.__file__).resolve().is_relative_to(build) or not native.is_relative_to(build):
        raise RuntimeError("DeepSelect resolved outside its own original build.")
    dim = case.params.get("dim")
    axes = tuple(range(x.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
    axes = tuple(a % x.ndim for a in axes)
    kept = tuple(a for a in range(x.ndim) if a not in axes)
    order = kept + axes
    width = prod(x.shape[a] for a in axes)
    rows = x.numel() // width
    shape = tuple(1 if a in axes else x.shape[a] for a in range(x.ndim)) \
        if case.params.get("keepdim", False) else tuple(x.shape[a] for a in kept)
    alignment = module.get_stride_requirement()[0] // x.element_size()
    padded = (width + alignment - 1) // alignment * alignment
    buffer = torch.empty((rows, padded), dtype=x.dtype, device=x.device) if padded != width else None

    def run(x):
        packed = x.permute(order).contiguous().view(rows, width)
        if buffer is not None:
            buffer[:, :width].copy_(packed)
            packed = buffer[:, :width]
        # Both 16-bit sign-magnitude formats have identical finite bit ordering.
        # Original radix selection and its output copies operate on those bits.
        values, _ = module.topk(packed.view(torch.bfloat16), 1, return_value=True,
                                indices_type=torch.int32)
        return values.contiguous().view(x.dtype).view(shape)

    return Candidate(
        Implementation(run), "deep_select.topk(...,topk=1,return_value=True)[0]",
        "Fold reduction axes; align row stride; bit-preserving FP16/BF16 views, no numeric casts.",
        source_hashes(native, root / "deep_select/interface.py", root / "csrc/api.cpp",
                      root / "csrc/ascend_kernels/kernel.asc"),
        limitations=("Finite manifest inputs only: original NaN policy differs from torch.amax.",
                     "TopK orders +0 above -0; zero sign is not bitwise equivalent to all torch reductions."),
    )


def _owned_native(path, symbols):
    """Resolve an ELF's own entry points, rejecting DT_NEEDED/stock aliases."""
    import ctypes as C
    import subprocess

    path = path.resolve(strict=True)
    output = subprocess.check_output(["nm", "-D", "--defined-only", str(path)], text=True)
    defined = {line.split()[-1].split("@")[0] for line in output.splitlines() if line.split()}
    missing = set(symbols) - defined
    if missing:
        raise RuntimeError(f"{path} does not define {sorted(missing)}")
    lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)

    class DlInfo(C.Structure):
        _fields_ = [("fname", C.c_char_p), ("base", C.c_void_p),
                    ("sname", C.c_char_p), ("saddr", C.c_void_p)]

    dl = C.CDLL(None)
    dl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dl.dladdr.restype = C.c_int
    for name in symbols:
        info = DlInfo()
        if not dl.dladdr(C.cast(getattr(lib, name), C.c_void_p), C.byref(info)):
            raise RuntimeError(f"Cannot establish owner of {name}")
        if not info.fname or Path(info.fname.decode()).resolve() != path:
            raise RuntimeError(f"{name} resolved to another library")
    return lib


def opsfft_fft(case):
    import ctypes as C
    import torch
    from benchmarks.api import Implementation

    (x,) = case.inputs
    if x.dtype != torch.complex64:
        raise UnsupportedCase("This deployed complete C2C API supports complex64.")
    n, batch = x.shape[-1], prod(x.shape[:-1])
    if n <= 1 or n > 1 << 27 or n & (n - 1):
        raise UnsupportedCase("This adapter uses the inspected power-of-two C2C path.")
    binary = WORKSPACE / ".tools/baselines/ops-fft/cann/x86_64-linux/lib64/libcann_ops_fft.so.9.0.0"
    lib = _owned_native(binary, ["aclfftPlan1d", "aclfftSetStream", "aclfftExecC2C", "aclfftDestroy"])
    lib.aclfftPlan1d.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_int, C.c_int]
    lib.aclfftSetStream.argtypes = [C.c_void_p, C.c_void_p]
    lib.aclfftExecC2C.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    lib.aclfftDestroy.argtypes = [C.c_void_p]
    for name in ("aclfftPlan1d", "aclfftSetStream", "aclfftExecC2C", "aclfftDestroy"):
        getattr(lib, name).restype = C.c_int

    def check(rc):
        if rc:
            raise RuntimeError(f"ops-fft native status={rc}")

    plan = C.c_void_p()
    check(lib.aclfftPlan1d(C.byref(plan), n, 0x29, batch, 0))  # C2C, horizontal

    def run(x):
        # This original API accepts HOST pointers and internally stages data.
        # Every call repeats both transfers; no cached input values.
        host_in = x.detach().cpu().contiguous()
        host_out = torch.empty_like(host_in)
        check(lib.aclfftSetStream(plan, torch.npu.current_stream().npu_stream))
        check(lib.aclfftExecC2C(plan, host_in.data_ptr(), host_out.data_ptr(), -1))
        return host_out.to(x.device)

    root = LIBRARIES / "ops-fft"
    return Candidate(
        Implementation(run), "aclfftPlan1d(ACLFFT_C2C)+aclfftExecC2C(ACLFFT_FORWARD)",
        "Original plan/execute API; contiguous host staging and transfers inside every call.",
        source_hashes(binary, root / "include/cann_ops_fft.h",
                      root / "src/fft1_d/arch35/fft1_d_exec_api.cpp",
                      root / "src/fft1_d/arch35/fft/fft1_d_c2c.cpp",
                      root / "src/fft1_d/arch35/fft/fft1_d_c2c_kernel.h"),
        timing_mode="host_roundtrip",
        cleanup=lambda: check(lib.aclfftDestroy(plan)),
    )
