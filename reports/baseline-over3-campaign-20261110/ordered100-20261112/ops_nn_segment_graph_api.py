"""Original GraphCall and exact original segment.run AST; no new native math."""
import ast,importlib.util,math,ctypes as C,hashlib,json,os
from pathlib import Path
class SegmentGraphAPI:
 def __init__(self,torch,workspace,manifest,qualification=None):
  self.t=torch;self.W=Path(workspace);self.source=self.W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';self.m=json.loads(Path(manifest).read_text());self.root=Path(self.m['staging_root'])
  if qualification is None:
   g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner'].startswith('root/baseline-native-probe/server1-segment-graph-') and g['child']['pid']==os.getpgrp()
  else:assert json.loads(Path(qualification).read_text())['private_original_device_kernels_verified']
  assert Path(os.environ['OPS_NN_VENDOR']).resolve()==self.root and self.root.resolve() in {Path(p).resolve() for p in os.environ['ASCEND_CUSTOM_OPP_PATH'].split(':') if p}
  path=self.source/'graph.py';s=importlib.util.spec_from_file_location('_original_segment_graph',path);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod);self.binding=mod.GraphCall('ops-nn');assert {'UnsortedSegmentMin','UnsortedSegmentProd'}<=self.binding.owned_ops
  self.tree=ast.parse((self.source/'reductions.py').read_text());outer=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='segment');self.run_ast=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='run');self.ast_sha256=hashlib.sha256(ast.dump(self.run_ast,include_attributes=False).encode()).hexdigest()
 def prepare(self,op,params):
  assert op in ('UnsortedSegmentMin','UnsortedSegmentProd');namespace={'torch':self.t,'C':C,'prod':math.prod,'p':params,'binding':self.binding,'logical':False,'api':op}
  tree=ast.Module(body=[self.run_ast],type_ignores=[]);ast.fix_missing_locations(tree);exec(compile(tree,str(self.source/'reductions.py'),'exec'),namespace);return namespace['run']
 def cleanup(self):self.binding.cleanup()
