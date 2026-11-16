"""Original Tensor API Conv2D with shape constants and host tile layout only."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, _owned_native, source_hashes,
)


def opstensor_spatial_conv_cases(case):
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
    if any(t is not None for t in extra) or x.dtype not in (torch.float16, torch.bfloat16) or w.dtype != x.dtype:
        raise UnsupportedCase('Original spatial Tensor API template uses matching16bit operands and no bias input.')
    batch, channels, height, width = x.shape
    out_channels, channels_per_group, kh, kw = w.shape
    groups = p.get('groups', 1)
    if channels != groups * channels_per_group or out_channels % groups:
        raise UnsupportedCase('Grouped convolution channel extents disagree.')
    sh, sw = pair(p['stride'])
    dh, dw = pair(p['dilation'])
    ph, pw = pair(p['padding'])
    ho = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    output_per_group = out_channels // groups
    ci_blocks = (channels_per_group + 15) // 16
    co_blocks = (output_per_group + 15) // 16
    padded_ci = ci_blocks * 16
    tile_h, tile_w = (ho + 3) // 4, (wo + 3) // 4
    patch_h, patch_w = 3 * sh + dh * (kh - 1) + 1, 3 * sw + dw * (kw - 1) + 1
    total_patches = batch * groups * tile_h * tile_w
    # A complete K axis must fit original L0A/L0B. Target SDK platform config:
    # Ascend950DT_9592.ini: both L0A and L0B are 65536 bytes.
    reduction = padded_ci * kh * kw
    if 16 * reduction * 2 > 65536:
        raise UnsupportedCase('A complete original convolution reduction exceeds L0A/L0B; split-K would require native schedule changes.')
    if (padded_ci * patch_h * patch_w + reduction * 16) * 2 > 524288:
        raise UnsupportedCase('Original whole-patch fmap/weight buffers exceed target L1 capacity.')
    root = LIBRARIES / "ops-tensor"
    source = root / ("include/tensor_api/examples/01_simd_cpp_api/07_tensor_api/"
                     "matrix_compute/conv2d_forward_tensor_api/conv2d_forward_tensor_api.asc")
    text = source.read_text()
    constants = {'BLOCK_DIM': (4, 1), 'N': (4, 1), 'C1': (2, ci_blocks),
                 'H': (9, patch_h), 'W': (9, patch_w), 'KH': (3, kh), 'KW': (3, kw),
                 'STRIDE_H': (2, sh), 'STRIDE_W': (2, sw),
                 'DILATION_H': (2, dh), 'DILATION_W': (2, dw),
                 'PAD_LEFT': (2, 0), 'PAD_RIGHT': (2, 0), 'PAD_TOP': (2, 0),
                 'PAD_BOTTOM': (2, 0), 'PAD_VALUE': (1, 0)}
    original = text
    for name, (before, after) in constants.items():
        old = f"constexpr uint32_t {name} = {before};"
        assert text.count(old) == 1, name
        text = text.replace(old, f"constexpr uint32_t {name} = {after};")
    scalar_type = 'half' if x.dtype == torch.float16 else 'bfloat16_t'
    text = re.sub(r'\bhalf\b', scalar_type, text)
    # PadType describes the padding register, independently of tensor dtype.
    # Native SDK lacks a BF16 overload; zero has identical uint16/BF16 bits.
    if scalar_type == 'bfloat16_t':
        assert constants['PAD_VALUE'][1] == 0
        assert text.count('img2col_params<bfloat16_t> img2colParams;') == 1
        text = text.replace('img2col_params<bfloat16_t> img2colParams;',
                            'img2col_params<uint16_t> img2colParams;')
    # Erasing literal constants and the selected scalar type recovers source.
    erase = lambda t: re.sub(
        r"(constexpr uint32_t (?:" + "|".join(constants) + r") = )\d+(;)", r"\1VALUE\2", t)
    assert re.sub(r'\b(?:half|bfloat16_t)\b', 'SCALAR', erase(original)) == re.sub(r'\b(?:half|bfloat16_t)\b', 'SCALAR', erase(text.replace('img2col_params<uint16_t>', 'img2col_params<bfloat16_t>')))
    headers = [q for q in (root / "include/tensor_api/include").rglob("*")
               if q.suffix in (".h", ".hpp")]
    hashes = source_hashes(Path(__file__), source, source.with_name("data_utils.h"), *headers)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    if scalar_type == 'bfloat16_t':
        hashes.update(source_hashes(
            sdk / 'asc/include/tensor_api/arch/cube/cube_copy_op.h',
            sdk / 'asc/impl/c_api/reg_base_impl/cube_datamove_intf_impl.h'))
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("Original Conv2D requires bisheng.")
    flags = ["-x", "asc", "--npu-arch=dav-3510", "-std=c++17", "-O2", "-shared",
             "-fPIC", "-DSCENARIO_NUM=1"]
    key = hashlib.sha256(json.dumps([hashes, flags, str(sdk.resolve()), constants, scalar_type, batch, groups, co_blocks, tile_h, tile_w],
                                    sort_keys=True).encode()).hexdigest()[:20]
    cache = WORKSPACE / ".cache/ascend-classic" / ("opstensor-conv-cases-" + key)
    cache.mkdir(parents=True, exist_ok=True)
    adapted = cache / "original_shape_only.asc"
    adapted.write_text(text)
    bridge = cache / "bridge.cpp"
    bridge.write_text(f'''#define main classic_original_conv_demo_main
#include "original_shape_only.asc"
#undef main
extern "C" int classic_opstensor_conv({scalar_type}* patches, {scalar_type}* weights, {scalar_type}* output, void* stream) {{
    for (int patch = 0; patch < {total_patches}; ++patch) {{
        const int convolution_group = (patch / {tile_h * tile_w}) % {groups};
        for (int output_block = 0; output_block < {co_blocks}; ++output_block)
            conv2d_forward_tensor_api_custom<<<1, 0, stream>>>(
                patches + patch * FMAP_BATCH_SIZE,
                weights + (convolution_group * {co_blocks} + output_block) * WEIGHT_SIZE,
                output + (patch * {co_blocks} + output_block) * OUT_BATCH_SIZE);
    }}
    return 0;
}}
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
        # Complete independent spatial patches/groups; never split K or sum
        # separate partial convolutions in the wrapper.
        padded_height = (tile_h - 1) * 4 * sh + patch_h
        padded_width = (tile_w - 1) * 4 * sw + patch_w
        grouped = x.reshape(batch, groups, channels_per_group, height, width)
        grouped = F.pad(grouped, (pw, padded_width - width - pw,
                                  ph, padded_height - height - ph,
                                  0, padded_ci - channels_per_group))
        patches = (grouped.unfold(3, patch_h, 4 * sh).unfold(4, patch_w, 4 * sw)
                   .permute(0, 1, 3, 4, 2, 5, 6).contiguous())
        grouped_weight = weight.reshape(groups, output_per_group, channels_per_group, kh, kw)
        grouped_weight = F.pad(grouped_weight, (0, 0, 0, 0, 0, padded_ci - channels_per_group,
                                                0, co_blocks * 16 - output_per_group))
        packed_weight = (grouped_weight.reshape(groups, co_blocks, 16, ci_blocks, 16, kh, kw)
                         .permute(0, 1, 3, 5, 6, 4, 2).contiguous())
        out = torch.empty((batch, groups, tile_h, tile_w, co_blocks, 16, 4, 4),
                          device=x.device, dtype=x.dtype)
        rc = fn(patches.data_ptr(), packed_weight.data_ptr(), out.data_ptr(),
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f'Original Conv2D host status {rc}')
        tiled = out.permute(0, 1, 4, 5, 2, 6, 3, 7).reshape(
            batch, groups, co_blocks * 16, tile_h * 4, tile_w * 4)
        return tiled[:, :, :output_per_group, :ho, :wo].reshape(batch, out_channels, ho, wo).contiguous()

    return Candidate(
        Implementation(run), "ops-tensor::conv2d_forward_tensor_api_custom",
        "Original dimension/stride/dilation/padding literals and FP16/BF16 scalar type only; per-call independent spatial patch and convolution-group packing, padded channels, output crop. No split-K or kernel schedule change.",
        hashes | source_hashes(adapted, bridge, binary),
        limitations=(f"{total_patches * co_blocks} original invocations per input; all per-call layout packing and launches count toward wall time.",),
        timing_mode="host_roundtrip",
    )


def opstensor_conv_cases(case):
    if tuple(case.inputs[1].shape[-2:]) == (1, 1) and case.params.get('groups', 1) == 1:
        from benchmarks.ascend_tensor_gemm_cases import opstensor_pointwise_conv_cases
        return opstensor_pointwise_conv_cases(case)
    return opstensor_spatial_conv_cases(case)
