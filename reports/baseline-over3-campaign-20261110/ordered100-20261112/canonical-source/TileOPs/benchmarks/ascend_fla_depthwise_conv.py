"""Spatial depthwise convolution by layout padding of original causal Conv1D."""
import ast
import importlib.util
from pathlib import Path
import sys

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes


def fla_depthwise_conv(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, weight, *extra = case.inputs
    p = case.params
    pair = lambda value: (value, value) if isinstance(value, int) else tuple(value)
    batch, channels, height, width = x.shape
    kh, kw = weight.shape[-2:]
    if (p['groups'] != channels or tuple(weight.shape[:2]) != (channels, 1)
            or x.dtype != torch.float16 or weight.dtype != x.dtype
            or any(t is not None for t in extra)):
        raise UnsupportedCase('This native causal entry adapts same-channel FP16 depthwise convolution without bias.')
    sh, sw = pair(p['stride'])
    dh, dw = pair(p['dilation'])
    ph, pw = pair(p['padding'])
    padded_h, padded_w = height + 2 * ph, width + 2 * pw
    ho = (padded_h - dh * (kh - 1) - 1) // sh + 1
    wo = (padded_w - dw * (kw - 1) - 1) // sw + 1
    window = dh * (kh - 1) * padded_w + dw * (kw - 1) + 1
    # Native tl.arange(0,W) requires power-of-two width. Leading zero taps do
    # not change the convolution at any selected output location.
    native_window = 1 << (window - 1).bit_length()
    padded_channels = channels + -channels % 256  # Original host's fixed BD.
    source = LIBRARIES / 'flash-linear-attention-npu/fla/ops/triton/triton_core/causal_conv1d.py'
    name = '_classic_fla_original_causal_conv1d'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # Older original FLA source names this Ascend builtin through tl.
    # Bind the current installed API without changing the original JIT body.
    from triton.language.extra.cann.extension import vec_ops
    extract_slice = vec_ops.extract_slice
    if not hasattr(module.tl, 'extract_slice'):
        module.tl.extract_slice = extract_slice
    tree = ast.parse(source.read_text(), filename=str(source))
    host = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'causal_conv1d_fwd_impl_old')
    replaced = set()
    for node in ast.walk(host):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ('BT', 'BD'):
                if key == 'BD':
                    assert ast.literal_eval(node.value) == 256
                else:
                    assert isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    assert node.value.func.id == 'min' and ast.literal_eval(node.value.args[0]) == 32
                node.value = ast.Constant(value=16)
                replaced.add(key)
    assert replaced == {'BT', 'BD'}
    # Only host tile arguments change. The original JIT kernel, device loops,
    # buffers, launch options and synchronization remain exactly as loaded.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[host], type_ignores=[])),
                 str(source), 'exec'), module.__dict__)
    native = module.causal_conv1d_fwd_impl_old
    y_positions = torch.arange(ho, device=x.device) * sh + dh * (kh - 1)
    x_positions = torch.arange(wo, device=x.device) * sw + dw * (kw - 1)
    selected = (y_positions[:, None] * padded_w + x_positions[None, :]).reshape(-1)

    def run(x, weight, bias=None):
        if bias is not None:
            raise UnsupportedCase("Original depthwise candidate has no bias input.")
        padded = F.pad(x, (pw, pw, ph, ph, 0, padded_channels - channels))
        data = padded.permute(0, 2, 3, 1).contiguous().reshape(batch, -1, padded_channels)
        taps = torch.zeros((native_window, padded_channels), device=x.device, dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                index = native_window - window + i * dh * padded_w + j * dw
                taps[index, :channels].copy_(weight[:, 0, i, j])
        values, _ = native(data, taps, bias=None, residual=None, initial_state=None,
                           output_final_state=False, activation=None, cu_seqlens=None)
        return values[:, selected, :channels].reshape(batch, ho, wo, channels).permute(0, 3, 1, 2).contiguous()

    return Candidate(Implementation(run), 'FLA original Triton causal_conv1d_fwd_impl_old',
                     'Only original host tile sizes BT/BD changed to16/16; zero-pad spatial borders/channels; flatten padded rows; place original spatial weights at corresponding 1D tap offsets with zero gaps; select bottom-right window endpoints. All multiplies/reductions remain in unchanged original causal kernel.',
                     source_hashes(Path(__file__), source, Path(vec_ops.__file__)),
                     limitations=('Original host BT/BD tile sizes reduced from32/256 to16/16 after114 UB overflow; device body/loops/stages unchanged. Must compile and verify the expanded tap window.',))
