import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,types,ast,json
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parent));from zero_pad_candidate import clone_factory
P=pathlib.Path(__file__).parent;src=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/tilekernels/dequant.py';node=next(x for x in ast.parse(src.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='int8_dequant')
def native(*unused):
 def mul(left,right,out):out.copy_(left*right)
 return mul
def result(run,*args,**kw):return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run))
env={'native':native,'result':result};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(src),'exec'),env)
module=types.SimpleNamespace(__file__=str(src),native=native,result=result);clone,proof=clone_factory(module)
# clone_factory needs only the native/result globals used by this original body.
count=0
for op in ['INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp']:
 for dtype in ['float16','bfloat16','float32']:
  for shape in [(3,128),(5,131),(17,4099)]:
   torch.manual_seed(0);q=torch.randint(-128,128,shape,dtype=torch.int8)
   scale=torch.randn(()) if 'Tensor' in op else torch.randn(shape[0]) if 'Channel' in op else torch.randn(shape[0],(shape[1]+127)//128)
   case=types.SimpleNamespace(params={'out_dtype':dtype},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
   a=env['int8_dequant'](case).implementation.run(q,scale);b=clone(case).implementation.run(q,scale)
   assert torch.equal(a.view(torch.int16 if a.element_size()==2 else torch.int32),b.view(torch.int16 if b.element_size()==2 else torch.int32));count+=1
(P/'ZERO_PAD_CPU_EQUIVALENCE.json').write_text(json.dumps({'tests':count,'surrogate_only_not_native_device_acceptance':True,'all_bitwise_equal':True,'proof':proof},indent=2));print('CPU_EQUIVALENCE',count)
