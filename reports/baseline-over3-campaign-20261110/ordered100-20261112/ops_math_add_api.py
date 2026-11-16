"""Original opsmath_add function AST; only private source/package metadata paths rebound."""
import ast,hashlib,json,ctypes as C
from pathlib import Path
class OpsMathAddAPI:
 def __init__(self,torch,workspace,manifest):
  self.t=torch;self.W=Path(workspace);self.N=Path(manifest).parent;self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root']);self.src=self.N/'opsmath_add_original_provider.py'
  for k,v in self.m['all_files'].items():assert hashlib.sha256((self.root/k).read_bytes()).hexdigest()==v['sha256'],k
  import benchmarks.ascend_baselines as base
  tree=ast.parse(self.src.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='opsmath_add');self.ast_sha=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
  def source_hashes(*paths):return {str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in paths}
  self.ns={'C':C,'Path':Path,'Candidate':base.Candidate,'LIBRARIES':self.N/'binding-libraries','WORKSPACE':self.N/'binding-workspace','_add_layout':base._add_layout,'_owned_native':base._owned_native,'source_hashes':source_hashes,'__file__':str(self.src)}
  mod=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,str(self.src),'exec'),self.ns)
 def prepare(self,case):return self.ns['opsmath_add'](case)
