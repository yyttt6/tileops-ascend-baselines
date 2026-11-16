"""Private function globals replace only proven zero-width pad; original native API and timed run unchanged."""
from pathlib import Path
import ast,copy,hashlib,json,types,dataclasses
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1')
def require_zero_padding_shape(m,n,k):
 assert min(m,n,k)>0 and m%1024==0 and n%256==0 and k%64==0,'Only already-aligned zero-pad shapes'
def clone_layout_without_zero_pad(text):
 tree=ast.parse(text);original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_layout');new=copy.deepcopy(original);pack=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='pack');ret=next(n for n in pack.body if isinstance(n,ast.Return));saved=copy.deepcopy(ret.value);assert isinstance(saved,ast.Tuple) and len(saved.elts)==2 and all(isinstance(e,ast.Call) and ast.unparse(e.func)=='F.pad' for e in saved.elts);assert [ast.unparse(e.args[0]) for e in saved.elts]==['lhs','rhs'];ret.value=ast.Tuple(elts=[copy.deepcopy(e.args[0]) for e in saved.elts],ctx=ast.Load());reverse=copy.deepcopy(new);next(n for n in next(x for x in reverse.body if isinstance(x,ast.FunctionDef) and x.name=='pack').body if isinstance(n,ast.Return)).value=saved;assert ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),'Only both F.pad calls may change';return ast.fix_missing_locations(new)
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.ascend_baselines import UnsupportedCase
  from ops_nn_gemm_pad4_support import candidate_factory as original
  assert library=='ops-nn' and case._call.signature.name=='GemmFwdOp' and case.id=='square-1k-float16' and all(x.dtype==torch.float16 for x in case.inputs)
  proof=P/'OPS_NN_GEMM_PAD4_ZERO_PAD_CPU_PROOF.json';assert json.loads(proof.read_text())['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  base=original(case,library)();path=N/'original-ascend_gemm_cases.py';text=path.read_text();assert hashlib.sha256(path.read_bytes()).hexdigest()=='174b54193f55b2b32e7500fe2986abfc1978c8e5e9ce8e72d1367179810a0453';layout=clone_layout_without_zero_pad(text);ns={'UnsupportedCase':UnsupportedCase};exec(compile(ast.Module(body=[layout],type_ignores=[]),str(path),'exec'),ns);m,n,k,pack=ns['_layout'](case,(1024,256,64));require_zero_padding_shape(m,n,k)
  fn=base.implementation.run;assert fn.__closure__ is None and 'pack' in fn.__globals__;globals2=dict(fn.__globals__);globals2['pack']=pack;new=types.FunctionType(fn.__code__,globals2,fn.__name__,fn.__defaults__,fn.__closure__);assert new.__code__ is fn.__code__ and all(globals2[x] is fn.__globals__[x] for x in fn.__globals__ if x!='pack')
  files={**base.source_sha256,str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),str(proof):hashlib.sha256(proof.read_bytes()).hexdigest()};return dataclasses.replace(base,implementation=dataclasses.replace(base.implementation,run=new),source_sha256=files,adaptation=base.adaptation+' Candidate only removes zero-width F.pad clones for this alignedreadonlyfullAPI; original timed run bytecode, outputallocation/crop and nativecall unchanged; no sharedmodule mutation.')
 return build
