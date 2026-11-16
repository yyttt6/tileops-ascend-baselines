"""Original standalone TLA RMS for the manifest case without affine weight."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, _rms_layout, source_hashes
from benchmarks.ascend_tla_baselines import _example_definitions, _runtime


def tla_unweighted_rms_cases(case):
    from benchmarks.api import Implementation

    if case.inputs[1] is not None:
        # The weighted fused entry remains quarantined; do not retry it here.
        raise UnsupportedCase('This original standalone entry has no affine weight input.')
    x, shape, hidden, eps, pack = _rms_layout(case)
    if hidden != 128 or x.numel() // hidden % 32:
        raise UnsupportedCase('This focused alternative is prepared for complete 32-row, 128-wide tiles.')
    host = 'examples/normalization/rms_norm.py'
    original = _example_definitions(host, '_rms_norm')
    builder = _runtime().jit(target='pto', out_idx=[1],
                             pass_configs=original.__wrapped__.__globals__['pass_configs'])(original.__wrapped__)
    kernel = builder(x.numel() // hidden, hidden, 32, 128, eps=eps,
                     dtype=str(x.dtype).split('.')[-1])

    def run(x, weight):
        if weight is not None:
            raise UnsupportedCase('A supplied affine weight must not be silently ignored.')
        data, _ = pack(x, weight)
        return kernel(data).reshape(shape)

    return Candidate(Implementation(run), 'tilelang-ascend::normalization._rms_norm',
                     'Original unweighted forward entry; flatten leading independent rows, original dtype/epsilon/tile parameters, existing PTO backend. Kernel body, ping-pong buffers and schedule unchanged.',
                     source_hashes(Path(__file__), Path(__file__).with_name('ascend_baselines.py'),
                                   Path(__file__).with_name('ascend_tla_baselines.py'),
                                   LIBRARIES / 'tilelang-ascend' / host))
