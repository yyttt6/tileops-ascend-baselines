"""Original reduction factory; metadata-only absolute file paths for taskprivate vendor."""
from pathlib import Path
import ast,copy,hashlib

def candidate_factory(case,library):
 def build():
  assert library=='ops-nn' and case._call.signature.name=='AllFwdOp'
  import torch
  assert case.inputs[0].dtype==torch.bool,'Only original bool route; numeric LpNorm not deployed in this package'
  from benchmarks.ascend_all_pairs.ops_nn import common,reductions
  tree=ast.parse(Path(common.__file__).read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='hashes');before=ast.dump(node,include_attributes=False);saved=[]
  class MetadataOnly(ast.NodeTransformer):
   def visit_ImportFrom(self,n):
    if n.module=='benchmarks.ascend_baselines' and any(x.name=='source_hashes' for x in n.names):
     saved.append(copy.deepcopy(n));n.names=[x for x in n.names if x.name!='source_hashes']
    return n
  node=MetadataOnly().visit(node);assert len(saved)==1;restored=copy.deepcopy(node)
  class Restore(ast.NodeTransformer):
   def visit_ImportFrom(self,n):
    if n.module=='benchmarks.ascend_baselines':return copy.deepcopy(saved[0])
    return n
  assert ast.dump(Restore().visit(restored),include_attributes=False)==before
  namespace=dict(vars(common));namespace['source_hashes']=lambda *paths:{str(Path(f)):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in paths};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),common.__file__,'exec'),namespace);common.hashes=namespace['hashes']
  candidate=reductions.segment(case);candidate.source_sha256[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return candidate
 return build
