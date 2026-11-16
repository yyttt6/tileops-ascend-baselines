"""An independent original RoPE entry with one frequency row per vector tile."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_tla_baselines import _example_definitions, _runtime


def tla_simple_rope(case):
    import math
    import torch
    from benchmarks.api import Implementation

    x, *extra = case.inputs
    if extra or x.ndim not in (2, 4) or x.shape[-1] % 2:
        raise UnsupportedCase('Original rotary entry needs a complete even feature axis and 1D/2D sequence layout.')
    layout = case.params['rope_layout']
    if layout not in ('neox', 'interleaved'):
        raise UnsupportedCase('Unknown rotary layout.')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise UnsupportedCase('Original rotary entry requires floating-point input.')
    shape = tuple(x.shape)
    dim, rows = shape[-1], math.prod(shape[:-1])
    if rows % 2:
        raise UnsupportedCase('Focused original two-row tile requires an even row count.')
    source = 'examples/pos_embedding/rope.py'
    original = _example_definitions(source, 'rope_kernel_in_place')
    body = original.__wrapped__
    builder = _runtime().jit(target='pto', pass_configs=body.__globals__['pass_configs'])(body)
    # Two vector subcores each own one row. Each flattened input row gets its
    # own frequency row, so original row_sin_cos=row_x/head_num remains valid.
    kernel = builder(rows, 2, rows, dim, dim, 1, dtype=str(x.dtype).split('.')[-1])
    seq = shape[0] if x.ndim == 2 else shape[1]
    base = case.params.get('base', 10000.)

    def run(x):
        inv = 1. / (base ** (torch.arange(dim // 2, dtype=torch.float32, device=x.device) / (dim // 2)))
        angles = torch.arange(seq, dtype=torch.float32, device=x.device)[:, None] * inv[None, :]
        cos, sin = [table.to(x.dtype).repeat_interleave(2, dim=-1)
                    for table in (angles.cos(), angles.sin())]
        if x.ndim == 4:
            cos = cos[None, :, None, :].expand(*shape).contiguous().view(rows, dim)
            sin = sin[None, :, None, :].expand(*shape).contiguous().view(rows, dim)
        if layout == 'neox':
            packed = torch.stack((x[..., :dim // 2], x[..., dim // 2:]), dim=-1).reshape(rows, dim)
        else:
            packed = x.contiguous().view(rows, dim).clone()
        kernel(packed, sin, cos)
        if layout == 'neox':
            return torch.cat((packed[:, 0::2], packed[:, 1::2]), dim=-1).reshape(shape)
        return packed.reshape(shape)

    return Candidate(Implementation(run), 'tilelang-ascend::pos_embedding.rope.rope_kernel_in_place',
                     'Distinct original rotary entry. Change only block_M to2, flatten independent heads, expand frequency rows, permute NeoX/interleaved layout and clone in-place input. Original kernel body and pass settings unchanged; existing PTO backend.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_tla_baselines.py'),
                                   LIBRARIES / 'tilelang-ascend' / source),
                     limitations=('The previous RotaryPositionEmbedding entry remains quarantined. This independent entry needs its own frozen-case device validation before expansion.',))
