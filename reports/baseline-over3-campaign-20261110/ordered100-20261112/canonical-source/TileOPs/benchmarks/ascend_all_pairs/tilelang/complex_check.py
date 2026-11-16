"""Platform compatibility for unchanged, zero-tolerance complex input checks.

No candidate arithmetic, manifest reference or output comparison is replaced.
"""
from pathlib import Path
from dataclasses import replace
import json
import os
from datetime import datetime, timezone

_metadata=None


def install():
    global _metadata
    if _metadata is not None:return _metadata
    from scripts.ascend_classic_check import require_guard
    require_guard()
    import torch
    from workloads import numerics
    from benchmarks.ascend_baselines import WORKSPACE,source_hashes
    original=numerics._unchanged
    def unchanged(tensors,snapshot):
        actual=[];expected=[]
        for tensor,source in zip(tensors,snapshot,strict=True):
            if tensor.is_complex() and tensor.device.type=='npu':
                if (tensor.device!=source.device or tensor.dtype!=source.dtype
                        or tensor.shape!=source.shape or tensor.layout!=source.layout):
                    return False
                tensor,source=tensor.cpu(),source.cpu()
            actual.append(tensor);expected.append(source)
        return original(actual,expected)
    results=[]
    for dtype,word,bits in ((torch.complex64,torch.int32,0x3f800000),
                            (torch.complex128,torch.int64,0x3ff0000000000000)):
        probe=torch.tensor([complex(0.,0.),complex(-0.,-0.),complex(1.,-2.),
                            complex(float('inf'),0.),complex(0.,-float('inf')),
                            complex(float('nan'),0.),complex(0.,float('nan'))],dtype=dtype,device='cpu').to('npu')
        snapshot=probe.clone()
        assert unchanged([probe],[snapshot])
        probe.view(word)[0]=bits
        assert not unchanged([probe],[snapshot])
        probe.copy_(snapshot)
        probe.view(word)[1]=bits
        assert not unchanged([probe],[snapshot])
        probe.copy_(snapshot)
        assert unchanged([probe],[snapshot])
        results.append({'dtype':str(dtype),'unchanged_special_values':True,
                        'real_mutation_detected':True,'imaginary_mutation_detected':True})
    torch.npu.synchronize()
    record={'schema':'tilelang-complex-preservation-selfcheck/v1',
            'method':'Original workloads.numerics._unchanged, same rtol0/atol0/equal_nan comparator on exact CPU copies of complex pairs only. Device, dtype, shape and layout are checked before transfer. All other inputs delegate unchanged.',
            'oracle_unchanged':True,'output_checks_unchanged':True,'results':results,
            'sources':source_hashes(Path(__file__),WORKSPACE/'TileOPs/workloads/numerics.py')}
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    path=WORKSPACE/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang'/f'complex-preservation-{now}-{os.getpid()}.json'
    with path.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    numerics._unchanged=unchanged
    _metadata={'selfcheck':str(path.relative_to(WORKSPACE)),**record}
    return _metadata


def annotate(candidate,case):
    if not any(getattr(t,'is_complex',lambda:False)() for t in case.inputs):return candidate
    from benchmarks.ascend_baselines import source_hashes
    metadata=install()
    candidate.source_sha256.update(source_hashes(Path(__file__)))
    return replace(candidate,adaptation=candidate.adaptation+
                   ' Complex input preservation uses the original zero-tolerance comparator on CPU copies; mutation selfcheck: '+metadata['selfcheck'])
