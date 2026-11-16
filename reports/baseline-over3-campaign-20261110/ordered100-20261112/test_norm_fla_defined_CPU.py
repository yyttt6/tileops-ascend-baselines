import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,sys,pathlib,types,json,hashlib,datetime,copy,math
import torch
P=pathlib.Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112');sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_norm_fla_defined_api as h
base=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm';ns={};exec(compile((base/'contracts.py').read_text(),str(base/'contracts.py'),'exec'),ns)
contract=types.ModuleType('benchmarks.ascend_all_pairs.vllm.contracts');contract.__dict__.update(ns);sys.modules[contract.__name__]=contract
errmod=types.ModuleType('benchmarks.ascend_baselines');
class UnsupportedCase(Exception):pass
errmod.UnsupportedCase=UnsupportedCase;sys.modules[errmod.__name__]=errmod
proofs=[]
def native(x,w,b,eps,group_size=None,**kw):
 n=group_size or x.shape[-1];xx=x.float().reshape(-1,n);mu=xx.mean(-1,keepdim=True);rr=torch.rsqrt(((xx-mu)**2).mean(-1,keepdim=True)+eps);y=((xx-mu)*rr).reshape(x.shape)*w.float()
 if b is not None:y=y+b.float()
 return y.to(x.dtype),mu,rr
module=types.ModuleType('basic_cpu_only');module.__file__=str(base/'basic.py');module.prod=math.prod;module.Path=pathlib.Path;module.TRITON_ROOT='';module.validate=lambda c:None;module.module=lambda *a,**k:(types.SimpleNamespace(_layer_norm_fwd=native),[]);module.candidate=lambda run,*a:types.SimpleNamespace(implementation=types.SimpleNamespace(run=run))
for name in ['group_norm','normalization']:fn,proof=h.clone_factory(module,name);setattr(module,name,fn);proofs.append(proof)
params=json.loads((P/'VLLM_NORM_FLA_DEFINED_FROZEN_PARAMS.json').read_text());tasks=json.loads((P/'TASKS.json').read_text())
def case(i,p):
 op=tasks[i]['op'];dtype=getattr(torch,p['x'][1]);shape=p['x'][0];x=torch.randn(shape,dtype=dtype);w=torch.randn(p['weight'][0],dtype=dtype) if 'weight'in p else None;b=torch.randn(p['bias'][0],dtype=dtype) if 'bias'in p else None;args=(x,None,None,w,b) if op=='InstanceNormFwdOp' else (x,w,b)
 return types.SimpleNamespace(id=tasks[i]['case'],params=p,inputs=args,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
def check(c):
 x=c.inputs[0];snapshot=[a.clone() for a in c.inputs if isinstance(a,torch.Tensor)];op=c._call.signature.name;factory=module.normalization if op=='LayerNormFwdOp' else module.group_norm;y=factory(c).implementation.run(*c.inputs)
 if op=='LayerNormFwdOp':
  width=math.prod(c.params['normalized_shape']);xx=x.float().reshape(-1,width);z=(xx-xx.mean(-1,keepdim=True))*torch.rsqrt(((xx-xx.mean(-1,keepdim=True))**2).mean(-1,keepdim=True)+c.params['eps']);want=z.to(x.dtype).reshape(x.shape)
 else:
  n,cnt=x.shape[:2];g=cnt if op=='InstanceNormFwdOp' else c.params['num_groups'];xx=x.float().reshape(n,g,-1);mu=xx.mean(-1,keepdim=True);z=((xx-mu)*torch.rsqrt(((xx-mu)**2).mean(-1,keepdim=True)+c.params['eps'])).reshape(x.shape);w,b=c.inputs[-2:];view=(1,cnt)+((1,)*(x.ndim-2))
  if w is not None:z=z*w.float().reshape(view)
  if b is not None:z=z+b.float().reshape(view)
  want=z.to(x.dtype)
 assert torch.equal(y,want),(c.id,x.dtype,(y-want).abs().max());assert all(torch.equal(a,b) for a,b in zip(snapshot,[a for a in c.inputs if isinstance(a,torch.Tensor)]));return True
count=0
for i,p in params.items():check(case(int(i),p));count+=1
for i in [1177,1180,1190,1191,1192,1194,1195,1199]:
 for dt in ['float16','bfloat16','float32']:
  p=copy.deepcopy(params[str(i)]);p['x']=[[2,8,3,5],dt] if tasks[i]['op']!='LayerNormFwdOp' else [[3,15],dt];p['output']=p['x'];p['num_groups']=2
  if tasks[i]['op']=='LayerNormFwdOp':p['normalized_shape']=[15]
  for k in ['weight','bias']:
   if k in p:p[k]=[[8],dt]
  check(case(i,p));count+=1
negative=[]
def reject(name,c):
 try:h.validate_alternate(c)
 except UnsupportedCase:negative.append(name)
 else:raise AssertionError(name)
c=case(1192,params['1192']);d=copy.copy(c);d.id='unreviewed';reject('unreviewed_exact_key',d)
d=copy.copy(c);d.inputs=(c.inputs[0][:,:,:,:-1],*c.inputs[1:]);reject('declared_shape_mismatch',d)
d=copy.copy(c);d.inputs=(c.inputs[0],torch.zeros(64),None,*c.inputs[3:]);reject('running_mean_not_None',d)
p=copy.deepcopy(c.params);p['use_input_stats']=False;reject('supplied_statistics_not_replaced',case(1192,p))
p=copy.deepcopy(c.params);p['x'][0]=[1,64,1,65537];p['output']=p['x'];reject('64KiB_limit',case(1192,p))
d=copy.copy(c);d.inputs=(*c.inputs[:-1],torch.ones(63,dtype=torch.float16));reject('affine_shape',d)
p=copy.deepcopy(c.params);p['running_mean']=[[64],'float16'];reject('other_original_rejection',case(1192,p))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((P/'VLLM_ADALN_FLA_DEFINED_CPU_PROOF.json').read_text());x={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':count,'negative_contract_tests':negative,'factory_AST_proofs':proofs,'helper_sha256':sha(pathlib.Path(h.__file__)),'source_sha256':sha(base/'basic.py'),'alternate_native_sha256':prior['alternate_native_sha256'],'global_contract_sha256_unchanged':sha(base/'contracts.py'),'scope':[1177,1180,1190,1191,1192,1194,1195,1199],'boundary':'CPU surrogate mapping/contract tests only; no original native execution or performance acceptance. No global guard/kernel/compiler/precision change.'};(P/'VLLM_NORM_FLA_DEFINED_CPU_PROOF.json').write_text(json.dumps(x,indent=2));print('CPU_MAPPING_CASES',count,'NEGATIVE_CONTRACT_TESTS',len(negative))
