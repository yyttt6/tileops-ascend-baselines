"""Original complete GEMV provider with only audited zero-pad bypass candidate."""
from pathlib import Path
import hashlib,sys,types,importlib.util
P=Path(__file__).resolve().parent
HELPER=Path('/opt/tileops-baseline-cache/workspace/TileOPs/benchmarks/ascend_baselines.py')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def factory(op,params,fixed,args,case=None):
 assert op=='MeanFwdOp' and tuple(args[0].shape)==(64,32768)
 helper=load('benchmarks.ascend_baselines',HELPER)
 # Only provenance serialization changes: private provider lives outside original workspace.
 helper.source_hashes=lambda *paths:{str(Path(p)):sha(p) for p in paths}
 pkg=types.ModuleType('private_row_mean');pkg.__path__=[str(P)];sys.modules[pkg.__name__]=pkg
 load('private_row_mean.reduction',P/'reduction.py')
 provider=load('private_row_mean.impl',P/('row_gemv_noop_pad_candidate.py' if fixed else 'row_gemv.py'))
 proxy=types.SimpleNamespace(inputs=args,params=case.params)
 candidate=provider.build(proxy,average=True)
 assert candidate.timing_mode=='device_async'
 sources=dict(candidate.source_sha256);sources[str(HELPER)]=sha(HELPER)
 return candidate.implementation.run,sources
