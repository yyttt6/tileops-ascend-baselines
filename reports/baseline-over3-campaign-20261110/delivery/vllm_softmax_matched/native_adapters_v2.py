"""Original complete vLLM native Softmax provider; no candidate/math edits."""
import pathlib,sys,importlib,importlib.util,types,hashlib
W=pathlib.Path('/opt/tileops-all-pairs/workspace');AP=W/'TileOPs/benchmarks/ascend_all_pairs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def factory(op,params,fixed,args,case=None):
 assert op=='SoftmaxFwdOp' and not fixed
 helper=W/'TileOPs/benchmarks/ascend_baselines.py'
 if 'benchmarks.ascend_baselines' not in sys.modules:
  spec=importlib.util.spec_from_file_location('benchmarks.ascend_baselines',helper);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 if 'benchmarks.ascend_all_pairs' not in sys.modules:
  package=types.ModuleType('benchmarks.ascend_all_pairs');package.__path__=[str(AP)];sys.modules[package.__name__]=package
 module=importlib.import_module('benchmarks.ascend_all_pairs.vllm.native')
 proxy=types.SimpleNamespace(inputs=args,params=case.params,_call=case._call)
 candidate=module.softmax(proxy);assert candidate.timing_mode=='device_async'
 sources={str(pathlib.Path(k) if pathlib.Path(k).is_absolute() else W/k):v for k,v in candidate.source_sha256.items()};sources[str(helper)]=sha(helper)
 return candidate.implementation.run,sources
