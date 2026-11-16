"""Native per-row FP8 quantization with manifest group widths and padding."""
from dataclasses import replace
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes


def opsnn_fp8_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_nn_graph import _candidate

    hidden = case.inputs[0].shape[-1]
    if hidden not in (64, 128):
        raise UnsupportedCase('This adapter covers complete manifest groups of 64 or 128 values.')
    block = 2 * hidden
    binding = GraphCall('ops-nn')
    def run(x):
        # Dummy values remain in the same native quantization group. Their
        # exact FP32 magnitude supplies the manifest's max(amax, 1e-4) floor.
        packed = F.pad(x.float().reshape(-1, hidden), (0, hidden), value=1e-4)
        codes = torch.empty(packed.shape, device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((packed.shape[0], 1), device=x.device, dtype=torch.float32)
        binding.call('DynamicBlockQuant', [packed], [codes, scales], {
            'min_scale': 0.0, 'round_mode': 'rint', 'dst_type': 36,
            'row_block_size': 1, 'col_block_size': block,
        })
        return scales.reshape(x.shape[:-1]), codes[:, :hidden].contiguous().reshape(x.shape)
    native = _candidate(binding, run, 'DynamicBlockQuant',
                        'Exact FP32 widening; pad same group with 1e-4 to native block128/256; crop codes, retain original scale.',
                        'quant/dynamic_block_quant')
    return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def tilelang_fp8_cases(case):
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes

    x, = case.inputs
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    if hidden not in (64, 128):
        raise UnsupportedCase('This adapter covers native128 groups and padded manifest64 groups.')
    padded_rows = rows + -rows % 32
    tilelang = _runtime()
    host = 'examples/ascend/example_per_token_cast_to_fp8.py'
    builder = local_function('tilelang', 'examples/ascend', 'example_per_token_cast_to_fp8', 'per_token_cast_to_fp8', host)
    kernel = builder(padded_rows, 128, mode='simt')
    def run(x):
        packed = F.pad(x.float().reshape(rows, hidden), (0, 128 - hidden, 0, padded_rows - rows))
        quant, scale = kernel(packed)
        return scale[:rows].reshape(x.shape[:-1]), quant[:rows, :hidden].contiguous().reshape(x.shape)
    return Candidate(Implementation(run), 'tilelang::per_token_cast_to_fp8(simt)',
                     'Exact FP32 widening; native128 or zero-padded64 group; independent-row padding/crop; original quantizer and schedule.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_tilelang_baselines.py'),
                                   LIBRARIES / 'tilelang' / host) | _runtime_hashes(tilelang))
