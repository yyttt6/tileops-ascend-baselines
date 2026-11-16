"""Run the unchanged adaptive-max manifest reference on CPU after a CANN fault.

Only the reference's execution device changes. Candidate inputs, execution,
output comparison, input protection and tolerances remain the original ones.
"""
from pathlib import Path
from dataclasses import replace

_original = None
_metadata = None


def install():
    global _original
    if _original is not None:
        return
    from scripts.ascend_classic_check import require_guard, cpu_reference
    require_guard()
    import torch
    from workloads import numerics
    _original = numerics.verify

    def verify(reference, inputs, **kwargs):
        if _metadata is not None:
            reference = cpu_reference(torch, reference, inputs)
        return _original(reference, inputs, **kwargs)

    numerics.verify = verify


def annotate(candidate, case):
    global _metadata
    _metadata = None
    if case._call.signature.name != 'AdaptiveMaxPool2dFwdOp':
        return candidate
    from benchmarks.ascend_baselines import WORKSPACE, source_hashes
    sources = source_hashes(Path(__file__), WORKSPACE/'TileOPs/workloads/pool.py',
                            WORKSPACE/'TileOPs/scripts/ascend_classic_check.py')
    _metadata = {
        'reference_device': 'cpu',
        'reference_engine': 'Unmodified Case.reference / AdaptiveMaxPool2dWorkload.ref_program',
        'transfer_api': 'scripts.ascend_classic_check.cpu_reference',
        'candidate_device': 'npu',
        'full_output_and_input_checks_unchanged': True,
        'manifest_and_tolerances_unchanged': True,
        'reason': 'CANN AdaptiveMaxPool2d SIMT stack overflow; driver kernel name, symbol and PC identify the original NPU reference.',
        'issue': '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang/oracle_adaptive_max_fault.json',
        'sources': sources,
    }
    candidate.source_sha256.update(sources)
    return replace(candidate, adaptation=candidate.adaptation +
                   ' Verification executes the unmodified manifest AdaptiveMaxPool2d reference on exact CPU copies via the shared checker cpu_reference helper; candidate remains on NPU. See oracle_adaptive_max_fault.json.')
