"""Original mean_pooling vs uniform batch through original mean_dim; guard required."""
import pathlib,sys,types,importlib,hashlib
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
def factory(op,params,fixed,args,*,case):
 assert op=='MeanPoolingFwdOp' and case.id.startswith('uniform')
 folder=ROOT/'TileOPs/benchmarks/ascend_all_pairs/vllm'
 assert hashlib.sha256((folder/'basic.py').read_bytes()).hexdigest()=='b4f860c870eec0c3991a1f5f0482fd531f9cb91871844c1e8c890d09ebb5aaf2'
 sys.path.append(str(ROOT/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'))
 pkg=types.ModuleType('over3_pool_vllm');pkg.__path__=[str(folder)];sys.modules[pkg.__name__]=pkg
 rt=importlib.import_module(pkg.__name__+'.runtime');rt.CACHE=pathlib.Path(__file__).parent/'jit_binding_cache'
 basic=importlib.import_module(pkg.__name__+'.basic')
 def capture(run,api,adaptation,sources,**kw):
  assert kw['timing_mode']=='host_completed'
  return run,sources
 basic.candidate=capture
 original,sources=basic.mean_pooling(case)
 if fixed:
  from uniform_candidate import build
  native,extra=rt.module(rt.TRITON_ROOT+'batch_invariant/mean.py');sources+=extra
  chunk=case.params['chunk_size'];assert args[0].shape[1]%chunk==0
  fn=build(native,chunk,True)
 else:fn=original
 files=[pathlib.Path(__file__).with_name('uniform_candidate.py'),*sources,*folder.glob('*.py')]
 return fn,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
