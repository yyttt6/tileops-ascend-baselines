"""Affine RMS through an unchanged original RMS+RoPE kernel's table inputs."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, _rms_layout, source_hashes
from benchmarks.ascend_tla_baselines import _example_definitions, _runtime


def tla_weighted_rms_rope(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, shape, hidden, eps, pack = _rms_layout(case)
    rows = x.numel() // hidden
    padded_rows = rows + rows % 2
    source = 'examples/pos_embedding/rms_rope_fused.py'
    original = _example_definitions(source, 'rms_rope_fused')
    # Keep the source's synchronization, memory planning and all kernel bodies.
    # One complete normalization row per vector half reduces only the tile size.
    builder = _runtime().jit(target='pto', out_idx=[-1],
                            pass_configs=original.__wrapped__.__globals__['pass_configs'])(original.__wrapped__)
    kernel = builder(padded_rows, 2, padded_rows, hidden, hidden, 1, eps,
                     dtype=str(x.dtype).split('.')[-1])

    def run(x, weight):
        data, gamma = pack(x, weight)
        data = F.pad(data, (0, 0, 0, padded_rows - rows))
        # With sin=0, the original rotary expression is y_i = normalized_x_i*cos_i.
        # Supplying gamma as cos realizes affine RMS exactly, with no external
        # multiplication or changes to the existing fused kernel's arithmetic.
        cos = gamma.reshape(1, hidden).expand(padded_rows, hidden).contiguous()
        sin = torch.zeros_like(cos)
        return kernel(data, sin, cos)[:rows].reshape(shape)

    return Candidate(Implementation(run), 'tilelang-ascend::rms_rope_fused(sin=0,cos=gamma)',
                     'Flatten normalization rows; zero-pad an odd row count; supply gamma as full-width cosine table and zero sine; original RMS+RoPE arithmetic and all schedules retained, block_M=2 tile parameter only.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_baselines.py'),
                                   Path(__file__).with_name('ascend_tla_baselines.py'),
                                   LIBRARIES / 'tilelang-ascend' / source),
                     limitations=('Distinct original entry from quarantined split_qkv_rmsnorm_mrope. Complete-row native UB capacity and numerical behavior remain unverified; no automatic fallback.',))
