"""Exact frozen CPU FP64 oracle and numeric interpreter for a restored published catalog case."""
from pathlib import Path
import ast,types,sys,hashlib,importlib.util
P=Path(__file__).resolve().parent;OLD=P/'workspace/TileOPs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(case):
 assert case.id in {'grouped-state-bfloat16','mamba-2.8b-decode-bfloat16','mamba-2.8b-packed-bfloat16','mamba-2.8b-prefill-float16','mamba-2.8b-prefill-bfloat16'} and case._call.signature.name=='SelectiveScanFwdOp'
 import torch
 from workloads.numerics import Custom
 spec=importlib.util.spec_from_file_location('frozen_bare_scan_numerics',OLD/'workloads/numerics.py');num=importlib.util.module_from_spec(spec);sys.modules[spec.name]=num;spec.loader.exec_module(num);source=OLD/'workloads/ascend_extended.py';cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SelectiveScanCall');nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'ref_program','verification'}];assert len(nodes)==2;env={'torch':torch,'F':torch.nn.functional,'Exact':num.Exact};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),env);reference=types.MethodType(env['ref_program'],case.workload);evidence=types.MethodType(env['verification'],case.workload)(*case.inputs)
 def cpu(value):
  if isinstance(value,torch.Tensor):return value.cpu()
  if isinstance(value,tuple):return tuple(cpu(x) for x in value)
  if isinstance(value,list):return [cpu(x) for x in value]
  return value
 def validate(got,expected):num.compare_outputs(cpu(got),cpu(expected),evidence)
 checker=Custom(validate,'Unchanged frozen SelectiveScan Exact(normalized=1e-3), CPU interpreter; both complete outputs')
 return reference,checker,{'frozen_oracle_and_verification_FunctionDef_AST_unmodified':True,'frozen_source':str(source),'frozen_source_sha256':sha(source),'frozen_numerics_source':str(OLD/'workloads/numerics.py'),'frozen_numerics_source_sha256':sha(OLD/'workloads/numerics.py'),'original_normalized_bound':1e-3,'new_framework_adapter_only_delegates_to_original_numeric_interpreter':True,'not_native_qualification':True}
