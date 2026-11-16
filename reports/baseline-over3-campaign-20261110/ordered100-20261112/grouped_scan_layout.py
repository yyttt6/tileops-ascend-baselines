"""Same original FP32 recurrence; replace repeated group coefficients with broadcast views."""
from pathlib import Path
import ast,copy,hashlib,json
P=Path(__file__).resolve().parent;SCOPE={'grouped-state-bfloat16','mamba-2.8b-decode-bfloat16','bare-scan-bfloat16','mamba-2.8b-packed-bfloat16','mamba-2.8b-prefill-float16','mamba-2.8b-prefill-bfloat16'}
def construct(torch):
 source=P/'workspace/TileOps-08f-original/workloads/mamba.py';node=copy.deepcopy(next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref'));before=ast.dump(node,include_attributes=False);changes=[]
 def replace(old,text,label):
  new=ast.parse(text).body[0];new._original_layout_node=copy.deepcopy(old);changes.append(label);return new
 class Layout(ast.NodeTransformer):
  def visit_Assign(self,n):
   if len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):return self.generic_visit(n)
   target=n.targets[0].id;value=ast.unparse(n.value)
   if target=='Bf':
    assert value=='B_in.float().repeat_interleave(rep, 1)';return replace(n,'Bf=B_in.float().reshape(batch,groups,1,A.shape[1],seqlen)','B broadcast view')
   if target=='Cf':
    assert value=='C_in.float().repeat_interleave(rep, 1)';return replace(n,'Cf=C_in.float().reshape(batch,groups,1,A.shape[1],seqlen)','C broadcast view')
   if target=='ys':
    assert value=='[]';return replace(n,'if True:\n uf=uf.reshape(batch,groups,rep,seqlen)\n dt=dt.reshape(batch,groups,rep,seqlen)\n h=h.reshape(batch,groups,rep,A.shape[1])\n decay=A.reshape(1,groups,rep,A.shape[1])\n ys=[]','split channel axis into group/head views')
   if target=='h' and value.startswith('torch.where('):return replace(n,'h=torch.where(restart[:,None,None,None],0.0,h)','same reset mask extra singleton')
   if target=='h' and isinstance(n.value,ast.BinOp):
    expected=ast.parse('h=torch.exp(dt[:,:,t,None]*A)*h+(dt[:,:,t]*uf[:,:,t])[...,None]*Bf[...,t]').body[0];assert ast.dump(n,include_attributes=False)==ast.dump(expected,include_attributes=False);return replace(n,'h=torch.exp(dt[...,t,None]*decay)*h+(dt[...,t]*uf[...,t])[...,None]*Bf[...,t]','same FP32 recurrence/order with grouped indexing')
   if target=='y' and value=='torch.stack(ys, -1)':return replace(n,'y=torch.stack(ys,-1).reshape(batch,dim,seqlen)','restore output channel shape')
   if target=='y' and value=='y + uf * D_in.float()[:, None]':return replace(n,'y=y+uf.reshape(batch,dim,seqlen)*D_in.float()[:,None]','restore uf view for same D residual')
   return self.generic_visit(n)
  def visit_Return(self,n):
   assert ast.unparse(n.value)=='(y.to(u.dtype), h)';return replace(n,'return y.to(u.dtype),h.reshape(batch,dim,A.shape[1])','restore complete final state shape')
 class Restore(ast.NodeTransformer):
  def visit(self,n):
   if hasattr(n,'_original_layout_node'):return copy.deepcopy(n._original_layout_node)
   return super().visit(n)
 transformed=Layout().visit(node);assert ast.dump(Restore().visit(copy.deepcopy(transformed)),include_attributes=False)==before;assert len(changes)==8;env={'torch':torch,'F':torch.nn.functional};exec(compile(ast.fix_missing_locations(ast.Module(body=[transformed],type_ignores=[])),str(Path(__file__)),'exec'),env)
 return env['selective_scan_ref'],{'original_source':str(source),'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'full_reverse_AST_identical':True,'changes':changes,'FP32_cast_exp_softplus_silu_reduction_and_multiply_add_order_preserved':True,'both_outputs_reset_initialstate_and_optional_work_retained':True,'only_eliminates_unnecessary_B_C_repeat_materialization':True,'not_native_qualification':True,'transformed_AST_sha256':hashlib.sha256(ast.dump(transformed,include_attributes=False).encode()).hexdigest()}
def build(case):
 assert case._call.signature.name=='SelectiveScanFwdOp' and case.id in SCOPE
 import torch
 function,proof=construct(torch);flag=case.params['delta_softplus']
 def measured(*xs):return function(*xs,delta_softplus=flag)
 return measured,proof
