"""Fingerprint helpers for the independent deployment; caches only immutable metadata."""
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def hashes(vendor, directories, adapter):
    from benchmarks.ascend_baselines import WORKSPACE, LIBRARIES, source_hashes
    base=Path(__file__).parent
    files=[Path(adapter),Path(__file__),base/'graph.py',base/'catalog.py',base/'truth.py',base/'tensorlist.py']
    for root in [Path(vendor),*(LIBRARIES/'ops-nn'/d for d in directories)]:
        files.extend(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.py','.json','.so','.o'))
    return source_hashes(*files)


def candidate(binding, run, api, adaptation, *directories):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    if getattr(binding,'_graph_tensorlist',False) and 'Foreach' in api:
        adaptation+=' Foreach uses the original graph JIT kernel with the documented device TensorList pointer-only layout, rebuilt per call.'
    return Candidate(Implementation(run), 'ops-nn::'+api+' via original ACLNN/graph JIT',adaptation,
                     dict(hashes(str(binding.vendor),tuple(directories),str(Path(run.__code__.co_filename).resolve()))),
                     limitations=('Acceptance requires matching own vendor compilation and native device launch logs.',),
                     cleanup=binding.cleanup)


def new_binding():
    from .tensorlist import TensorListGraphCall
    return TensorListGraphCall('ops-nn')


def op_name(case):
    return case._call.signature.name
