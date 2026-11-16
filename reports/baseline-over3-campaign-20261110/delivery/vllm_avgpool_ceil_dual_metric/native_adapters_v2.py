"""Guarded factory for one exact fused RMSNorm representative; no device imports at module import."""
import pathlib,sys,types,importlib,importlib.util,hashlib
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
EXPECTED_BASIC='e015aa2096bdc1e2df78340a5702950305273cb84d55290bf94501577fbb5ed3'
def factory(op,params,fixed,args,*,case):
 assert op in ('AvgPool1dFwdOp','AvgPool3dFwdOp')
 folder=ROOT/'TileOPs/benchmarks/ascend_all_pairs/vllm'
 assert hashlib.sha256((folder/'reductions.py').read_bytes()).hexdigest()==EXPECTED_BASIC,'runtime source changed; re-review before measurement'
 sys.path.append(str(ROOT/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'))
 pkg=types.ModuleType('over3_rms_vllm');pkg.__path__=[str(folder)];sys.modules[pkg.__name__]=pkg
 rt=importlib.import_module(pkg.__name__+'.runtime');rt.CACHE=pathlib.Path(__file__).resolve().parent/'jit_binding_cache'
 file=pathlib.Path(__file__).with_name('pool_candidate.py')
 spec=importlib.util.spec_from_file_location(pkg.__name__+'.layout_candidate',file)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 # Capture the complete callable and original source list; measuring runner handles provenance/JIT/raw.
 def capture(run,api,adaptation,sources,**kwargs):
  assert kwargs.get('timing_mode')=='host_completed'
  return run,sources
 mod.candidate=capture
 fn,sources=(mod.pool_fixed if fixed else mod.pool_original)(case)
 paths=[file,*sources,*folder.glob('*.py')]
 proof={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
 return fn,proof
