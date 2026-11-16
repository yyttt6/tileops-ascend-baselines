"""Complete original opsmath.graph.make; source/hash/vendor relocation only."""
from pathlib import Path
import ast,copy,hashlib
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-div-v1')
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='DivFwdOp' and case.params.get('rounding_mode') is None
  from benchmarks.ascend_all_pairs.ops_math import graph,specs
  source=Path(graph.__file__);assert hashlib.sha256(source.read_bytes()).hexdigest()=='b8a51f71398f1be6e99c3a723c18b41a115dbdf604d2c7ac4bc82e76bb30f360'
  original=ast.parse(source.read_text());namespace=dict(vars(graph));namespace.update(SOURCE=N/'source',VENDOR=N/'graph-private-v1/vendors/allpairs_math_div_math');specs.SOURCE=N/'source'
  namespace['source_hashes']=lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
  for name in ['deployment_hashes','make']:
   node=copy.deepcopy(next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name));before=ast.dump(node,include_attributes=False);saved=[]
   class MetadataOnly(ast.NodeTransformer):
    def visit_ImportFrom(self,n):
     if n.module=='benchmarks.ascend_baselines' and any(a.name=='source_hashes' for a in n.names):
      saved.append(copy.deepcopy(n));n.names=[a for a in n.names if a.name!='source_hashes']
      if not n.names:return ast.copy_location(ast.Pass(),n)
     return n
   node=MetadataOnly().visit(node);assert len(saved)==1
   class Restore(ast.NodeTransformer):
    def visit_Pass(self,n):return copy.deepcopy(saved[0])
    def visit_ImportFrom(self,n):
     if n.module=='benchmarks.ascend_baselines':return copy.deepcopy(saved[0])
     return n
   assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
   exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
  candidate=namespace['make'](case,'DivFwdOp');candidate.source_sha256[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();candidate.source_sha256[str(N/'DIV_PACKAGE.json')]=hashlib.sha256((N/'DIV_PACKAGE.json').read_bytes()).hexdigest();return candidate
 return build
