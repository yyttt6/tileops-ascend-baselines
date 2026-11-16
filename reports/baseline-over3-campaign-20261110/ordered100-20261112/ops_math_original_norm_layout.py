"""Original affine/row/run function AST retained, native complete original LayerNorm binding injected."""
from pathlib import Path
from math import prod
import ast,copy,hashlib
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1')
def norm_layout_run(case,op,native):
 import torch
 from benchmarks.ascend_baselines import UnsupportedCase
 assert op in {'LayerNormFwdOp','AdaLayerNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp'}
 if op=='InstanceNormFwdOp':assert case.params.get('use_input_stats',True) and all(x is None for x in case.inputs[1:3]),'Original full tracked/eval semantics are unsupported, no silent stats removal'
 f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/layernorm.py';raw=f.read_bytes();make=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='make');nodes=[copy.deepcopy(n) for n in make.body if isinstance(n,ast.FunctionDef) and n.name in {'affine','row','run'}];assert len(nodes)==3;before=[ast.dump(n,include_attributes=False) for n in nodes];assign=[copy.deepcopy(n) for n in make.body if isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id in {'p','eps'} for t in n.targets)];assert len(assign)==3;scope={'torch':torch,'case':case,'op':op,'native':native,'prod':prod,'UnsupportedCase':UnsupportedCase};exec(compile(ast.fix_missing_locations(ast.Module(body=assign+nodes,type_ignores=[])),str(f),'exec'),scope);assert before==[ast.dump(n,include_attributes=False) for n in nodes];return scope['run']
