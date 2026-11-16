"""Guarded factory for one exact fused RMSNorm representative; no device imports at module import."""
import pathlib,sys,types,importlib,importlib.util,hashlib
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
EXPECTED_BASIC='cf9045a5183e6e3d3a5aebfd20155e4d48f9852f38e80d943bfa0920f15c1201'
def factory(op,params,fixed,args,*,case):
 assert op=='SigmoidFwdOp'
 folder=ROOT/'TileOPs/benchmarks/ascend_all_pairs/vllm'
 assert hashlib.sha256((folder/'activation.py').read_bytes()).hexdigest()==EXPECTED_BASIC,'runtime source changed; re-review before measurement'
 sys.path.append(str(ROOT/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'))
 pkg=types.ModuleType('over3_rms_vllm');pkg.__path__=[str(folder)];sys.modules[pkg.__name__]=pkg
 rt=importlib.import_module(pkg.__name__+'.runtime');rt.CACHE=pathlib.Path(__file__).resolve().parent/'jit_binding_cache'
 file=pathlib.Path(__file__).with_name('activation_candidate.py')
 spec=importlib.util.spec_from_file_location(pkg.__name__+'.layout_candidate',file)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 # Capture the complete callable and original source list; measuring runner handles provenance/JIT/raw.
 def capture(run,api,adaptation,sources,**kwargs):return run,sources
 mod.candidate=capture
 fn,sources=(mod.activation_fixed if fixed else mod.activation_original)(case)
 paths=[file,*sources,*folder.glob('*.py')]
 proof={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
 return fn,proof
