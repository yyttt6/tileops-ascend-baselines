import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch,torch.nn.functional as F,sys,pathlib,ast,types,json,hashlib,datetime
P=pathlib.Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112');sys.path.insert(0,str(P));import torch_extended_device_reference as h
f=P/'workspace/TileOPs/workloads/ascend_extended.py';source=f.read_text();tree=ast.parse(source);torch.set_num_threads(8);count=0;proofs={}
def eq(x,y):
 if isinstance(x,tuple):return len(x)==len(y) and all(eq(a,b) for a,b in zip(x,y))
 if x.dtype==torch.float8_e4m3fn:return torch.equal(x.view(torch.uint8),y.view(torch.uint8))
 return torch.equal(x,y)
for name in h.CLASS_CALLS:
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name);fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='ref_program');module=types.ModuleType('original_CPU_'+name);module.__file__=str(f);module.torch=torch;module.F=F;sys.modules[module.__name__]=module;exec(compile(ast.Module(body=[fn],type_ignores=[]),str(f),'exec'),vars(module));cls=type(name,(),{'__module__':module.__name__,'ref_program':module.ref_program})
 for dt in [torch.float16,torch.bfloat16,torch.float32]:
  for opt in [False,True]:
   for tail in [8,33]:
    w=cls()
    if name=='CausalConvUpdateCall':
     w.call=types.SimpleNamespace(params={'activation':'silu' if opt else None});args=[torch.randn(2,tail,dtype=dt),torch.randn(tail,4,dtype=dt),torch.randn(2,tail,4,dtype=dt),torch.randn(tail,dtype=dt) if opt else None]
    else:
     w.call=types.SimpleNamespace(params={'eps':1e-5,'scale_ub':0.1 if opt else None,'quant_dtype':'float8_e4m3fn' if opt else 'int8'});args=[torch.randn(3,tail,dtype=dt),torch.randn(3,tail,dtype=dt),torch.randn(tail,dtype=dt)];args[0][0].zero_();args[1][0].zero_()
    old=[a.clone() if a is not None else None for a in args];new,proof=h.bind(w);assert eq(new(*args),w.ref_program(*args));assert all(a is None or torch.equal(a,b) for a,b in zip(args,old));count+=1;proofs[name]=proof
x={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_cases':count,'proofs':proofs,'original_workload_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(pathlib.Path(h.__file__).read_bytes()).hexdigest(),'scope':[1296,1348,1349,1350,1351,1352,1353],'boundary':'CPU oracle placement-only equivalence; original CPUchecker and mathroundings unchanged, NPUfullcheck/15raw needed.'};(P/'TORCH_EXTENDED_DEVICE_CPU_PROOF.json').write_text(json.dumps(x,indent=2));print('CPU_CASES',count)
