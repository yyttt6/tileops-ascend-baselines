"""Reuse the original vector GEMV for independent output-column groups."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.ascend_baselines import source_hashes
from benchmarks.ascend_sgl_gemv_pilot import sgl_gemv_pilot


def sgl_gemv_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    a, b = case.inputs
    ta, tb = case.params['trans_a'], case.params['trans_b']
    aa, weights = (a.T if ta else a), (b if tb else b.T)
    columns = weights.shape[0]
    if columns <= 1024 and columns % 16 == 0:
        return sgl_gemv_pilot(case)
    # Every group retains the complete reduction axis K. No partial products
    # are accumulated in Python, and the native vector schedule is unchanged.
    groups = []
    for begin in range(0, columns, 1024):
        end = min(begin + 1024, columns)
        padding = -(end - begin) % 16
        native = sgl_gemv_pilot(SimpleNamespace(
            inputs=(aa, F.pad(weights[begin:end], (0, 0, 0, padding))),
            params={'trans_a': False, 'trans_b': True},
        ))
        groups.append((begin, end, padding, native))
    assert all(native.cleanup is None for _, _, _, native in groups)

    def run(a, b):
        aa, weights = (a.T if ta else a), (b if tb else b.T)
        outputs = [native.implementation.run(
                       aa, F.pad(weights[begin:end], (0, 0, 0, padding)))[:, :end - begin]
                   for begin, end, padding, native in groups]
        return torch.cat(outputs, dim=1)

    hashes = source_hashes(Path(__file__))
    for _, _, _, native in groups:
        hashes.update(native.source_sha256)
    return replace(
        groups[0][3], implementation=Implementation(run),
        adaptation='Partition independent output columns into groups of at most1024, zero-pad the final group to16 for native DMA alignment, invoke the original SGEMMV with full K, crop padding and concatenate columns; original kernel body, fixed tile and buffer count unchanged.',
        source_sha256=hashes,
        limitations=(f'{len(groups)} native calls plus all packing and concatenation count in timing. Expand only after the frozen SGEMMV pilot passes; never calls quarantined SGEMMC.',),
    )
