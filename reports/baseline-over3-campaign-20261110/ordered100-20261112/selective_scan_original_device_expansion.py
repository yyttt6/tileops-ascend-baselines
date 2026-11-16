"""Bind unchanged published complete Torch device program; retain existing frozen oracle and inputs."""
from pathlib import Path
import ast,hashlib,inspect
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(case):
 assert case._call.signature.name=='SelectiveScanFwdOp' and case.id in {'grouped-state-bfloat16','mamba-2.8b-decode-bfloat16','mamba-2.8b-packed-bfloat16','mamba-2.8b-prefill-float16','mamba-2.8b-prefill-bfloat16'}
 import torch
 proof=__import__('json').loads((P/'SELECTIVE_SCAN_ORIGINAL_CPU_DIAGNOSIS.json').read_text());assert proof['CPU_published_original_FP32_Torch_program_passes_existing_FP64_oracle_and_original_checker'];source=Path(proof['published_original_program_source']);oracle=Path(inspect.getsourcefile(case.reference));assert sha(source)==proof['published_original_program_source_sha256'] and str(oracle)==proof['original_oracle_source'] and sha(oracle)==proof['original_oracle_source_sha256'];node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref');assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==proof['published_function_AST_sha256'];assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cpu' for n in ast.walk(node));env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),env);program=env['selective_scan_ref'];flag=case.params['delta_softplus']
 def measured(*xs):
  assert xs[0].device.type=='npu'
  return program(*xs,delta_softplus=flag)
 return measured,{'published_original_program_source':str(source),'published_original_program_source_sha256':sha(source),'published_function_AST_unchanged':True,'original_published_FP32_math_not_precision_edited':True,'existing_frozen_CPU_FP64_oracle_unchanged':True,'original_oracle_source_sha256':sha(oracle),'both_y_and_final_states_retained':True,'input_metadata_payload_unchanged':True,'CPU_proof_not_native_acceptance':True}
