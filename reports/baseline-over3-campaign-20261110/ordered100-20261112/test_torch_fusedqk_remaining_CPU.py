import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,sys,pathlib,ast,types,json,hashlib,datetime
P=pathlib.Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112');sys.path.insert(0,str(P));import torch_fusedqk_remaining_reference as h
f=P/'workspace/TileOPs/workloads/ascend_extended.py';source=f.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FusedQKNormRoPECall');fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='ref_program');module=types.ModuleType('qualified_fusedqk_original_CPU');module.__file__=str(f);module.torch=torch;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(f),'exec'),vars(module));Cls=type('FusedQKNormRoPECall',(),{'__module__':module.__name__,'ref_program':module.ref_program});count=0;torch.set_num_threads(8)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for layout in ['neox','interleaved']:
  for width in [8,64]:
   w=Cls();w.call=types.SimpleNamespace(params={'num_heads':4,'num_kv_heads':2,'eps':1e-5,'rope_layout':layout});x=torch.randn(7,(4+4)*width,dtype=dtype);qw=torch.randn(width,dtype=dtype);kw=torch.randn(width,dtype=dtype);cache=torch.randn(128,width,dtype=torch.float32);pos=torch.arange(7,dtype=torch.int64);args=[x,qw,kw,cache,pos];old=[a.clone() for a in args];new,proof=h.bind(w);assert torch.equal(new(*args),w.ref_program(*args));assert all(torch.equal(a,b) for a,b in zip(args,old));count+=1
   args[3]=torch.randn(128,width//2,dtype=torch.float32);old=[a.clone() for a in args];assert torch.equal(new(*args),w.ref_program(*args));assert all(torch.equal(a,b) for a,b in zip(args,old));count+=1
p={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_cases':count,'proof':proof,'original_workload_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(pathlib.Path(h.__file__).read_bytes()).hexdigest(),'scope':list(range(1356,1361)),'boundary':'CPU placement equivalence and AST proof only; no native performance/checker acceptance. Original CPUoracle and verifier untouched.'};(P/'TORCH_FUSEDQK_REMAINING_CPU_PROOF.json').write_text(json.dumps(p,indent=2));print('CPU_CASES',count)
