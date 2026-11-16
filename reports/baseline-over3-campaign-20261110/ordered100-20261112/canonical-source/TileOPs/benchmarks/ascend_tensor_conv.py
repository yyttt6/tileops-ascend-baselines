"""Original Tensor API Conv2D with shape constants and host tile layout only."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


def opstensor_conv(case):
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, w, *extra = case.inputs
    p = case.params
    pair = lambda value: (value, value) if isinstance(value, int) else tuple(value)
    if (tuple(x.shape) != (2, 64, 56, 56) or tuple(w.shape) != (64, 64, 3, 3)
            or any(t is not None for t in extra) or p.get("groups", 1) != 1
            or pair(p["stride"]) != (1, 1) or pair(p["padding"]) != (1, 1)
            or pair(p["dilation"]) != (1, 1) or x.dtype != torch.float16):
        raise UnsupportedCase("This sample uses the frozen 3x3 FP16 Conv2D case.")
    root = LIBRARIES / "ops-tensor"
    source = root / ("include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/"
                     "matrix_compute/conv2d_forward_tensor_api/conv2d_forward_tensor_api.asc")
    text = source.read_text()
    constants = {"BLOCK_DIM": (4, 1), "N": (4, 1), "C1": (2, 4),
                 "H": (9, 6), "W": (9, 6), "STRIDE_H": (2, 1), "STRIDE_W": (2, 1),
                 "DILATION_H": (2, 1), "DILATION_W": (2, 1),
                 "PAD_LEFT": (2, 0), "PAD_RIGHT": (2, 0), "PAD_TOP": (2, 0),
                 "PAD_BOTTOM": (2, 0), "PAD_VALUE": (1, 0)}
    original = text
    for name, (before, after) in constants.items():
        old = f"constexpr uint32_t {name} = {before};"
        assert text.count(old) == 1, name
        text = text.replace(old, f"constexpr uint32_t {name} = {after};")
    # Erasing only these literal constants must recover identical source.
    erase = lambda t: re.sub(
        r"(constexpr uint32_t (?:" + "|".join(constants) + r") = )\d+(;)", r"\1VALUE\2", t)
    assert erase(original) == erase(text)
    headers = [q for q in (root / "include/tensor_api/include").rglob("*")
               if q.suffix in (".h", ".hpp")]
    hashes = source_hashes(Path(__file__), source, source.with_name("data_utils.h"), *headers)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("Original Conv2D requires bisheng.")
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared",
             "-fPIC", "-DSCENARIO_NUM=1"]
    key = hashlib.sha256(json.dumps([hashes, flags, str(sdk.resolve())],
                                    sort_keys=True).encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / ("opstensor-conv-" + key)
    cache.mkdir(parents=True, exist_ok=True)
    adapted = cache / "original_shape_only.asc"
    adapted.write_text(text)
    bridge = cache / "bridge.cpp"
    bridge.write_text('''#define main classic_original_conv_demo_main
#include "original_shape_only.asc"
#undef main
extern "C" int classic_opstensor_conv(half* patches, half* weights, half* output, void* stream) {
    // Host batching only; each original kernel still has one complete convolution.
    for (int patch = 0; patch < 392; ++patch)
        for (int group = 0; group < 4; ++group)
            conv2d_forward_tensor_api_custom<<<1, 0, stream>>>(
                patches + patch * FMAP_BATCH_SIZE,
                weights + group * WEIGHT_SIZE,
                output + (patch * 4 + group) * OUT_BATCH_SIZE);
    return 0;
}
''')
    binary = cache / "libclassic.so"
    if not binary.exists():
        temporary = cache / "libclassic.building.so"
        command = [compiler, *flags, str(bridge), "-o", str(temporary)]
        for include in [source.parent, root / "include/tensor_api/include",
                        root / "include/tensor_api", sdk / "include",
                        sdk / "x86_64-linux/asc/include", sdk / "asc/impl/c_api"]:
            command.extend(["-I", str(include)])
        command.extend(["-L", str(sdk / "lib64"), "-lascendcl", "-lascendc_runtime", "-lruntime"])
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (cache / "build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Original Conv2D build failed: {cache / 'build.log'}")
        temporary.replace(binary)
    lib = _owned_native(binary, ["classic_opstensor_conv"])
    fn = lib.classic_opstensor_conv
    fn.argtypes = [C.c_void_p] * 4
    fn.restype = C.c_int

    def run(x, weight, *extra):
        padded = F.pad(x, (1, 1, 1, 1))
        patches = (padded.unfold(2, 6, 4).unfold(3, 6, 4)
                   .permute(0, 2, 3, 1, 4, 5).contiguous())
        packed_weight = (weight.reshape(4, 16, 4, 16, 3, 3)
                         .permute(0, 2, 4, 5, 3, 1).contiguous())
        out = torch.empty((2, 14, 14, 4, 16, 4, 4), device=x.device, dtype=x.dtype)
        rc = fn(patches.data_ptr(), packed_weight.data_ptr(), out.data_ptr(),
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"Original Conv2D host status {rc}")
        return out.permute(0, 3, 4, 1, 5, 2, 6).reshape(2, 64, 56, 56).contiguous()

    return Candidate(
        Implementation(run), "ops-tensor::conv2d_forward_tensor_api_custom",
        "Only original shape/stride/padding literals changed. Host packs6x6 patches, "
        "16 output-channel groups and concatenates4x4 outputs; no split-K or kernel schedule change.",
        hashes | source_hashes(adapted, bridge, binary),
        limitations=("1568 original invocations per input; complete wall time includes layout packing.",),
        timing_mode="host_roundtrip",
    )
