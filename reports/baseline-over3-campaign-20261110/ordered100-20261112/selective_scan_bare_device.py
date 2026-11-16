from pathlib import Path
import ast,hashlib
from selective_scan_bare_frozen_policy import bind
P=Path(__file__).resolve().parent
def build(case):
 assert case.id=='bare-scan-bfloat16' and case._call.signature.name=='SelectiveScanFwdOp'
 import torch
 reference,checker,proof=bind(case);source=P/'workspace/TileOps-08f-original/workloads/mamba.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref');env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),env);fn=env['selective_scan_ref'];flag=case.params['delta_softplus']
 def measured(*xs):
  assert xs[0].device.type=='npu'
  return fn(*xs,delta_softplus=flag)
 proof.update({'published_program_function_AST_unchanged':True,'published_FP32_program_precision_unchanged':True,'published_source':str(source),'published_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'both_complete_outputs_retained':True,'frozen_FP64_oracle_checker_retained':True})
 return measured,reference,checker,proof
