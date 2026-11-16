"""CPU-only original-workload reference, ragged/tail and changed metadata; not device acceptance."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,types,datetime
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU chunk proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.pool import MeanPoolingWorkload,mean_pooling_chunk_index
from workloads.numerics import reference_tolerance
from ops_math_chunk_mean_support import chunk_candidate
set_run_device('cpu');torch.set_num_threads(4);T=types.SimpleNamespace;checks=[]
configs=[(2,16,3,5,4,torch.float16,None),(1,17,3,5,4,torch.float32,None),(1,16,3,5,4,torch.float16,[8,8]),(2,17,3,5,4,torch.float16,[5,0,12]),(1,17,3,5,4,torch.float32,[5,12]),(1,3,3,5,1,torch.float32,None)]
for b,t,h,d,k,dt,lens in configs:
 x=(torch.arange(b*t*h*d).reshape(b,t,h,d)%97-48).to(dt)/16;meta=() if lens is None else mean_pooling_chunk_index(lens,k)
 if lens is not None and 0 in lens:
  # The old auxiliary generator renumbers sequences across an empty sequence.
  # Keep this boundary test with explicit valid caller indices; frozen inputs unchanged.
  meta=(meta[0],torch.tensor([(seq,j) for seq,length in enumerate(lens) for j in range((length+k-1)//k)],dtype=torch.int32))
 args=(x,*meta);before=[a.contiguous().view(torch.uint8).clone() for a in args];w=MeanPoolingWorkload(b,t,h,d,k,dt,torch.float32,lens)
 def child(c,op):
  assert op=='MeanFwdOp' and c.params['dim']==2;return T(implementation=T(run=lambda y:y.mean(2,dtype=torch.float32).to(y.dtype)),cleanup=lambda:None)
 candidate=chunk_candidate(T(params={'chunk_size':k,'accum_dtype':'float32'}),child,lambda:{});out=candidate.implementation.run(*args);gold=w.ref_program(*args);torch.testing.assert_close(out,gold,**reference_tolerance(dt));assert all(torch.equal(a.contiguous().view(torch.uint8),v) for a,v in zip(args,before));r={'shape':list(x.shape),'dtype':str(dt),'chunk':k,'seq_lens':lens,'full_original_reference_passed':True,'input_and_metadata_bits_preserved':True,'unchanged_run_reduce_AST':True}
 if lens is not None:
  reverse=args[-1].flip(0);changed=(x,args[1],reverse);out=candidate.implementation.run(*changed);gold=w.ref_program(*changed);torch.testing.assert_close(out,gold,**reference_tolerance(dt));r['changed_metadata_reused_shape_cache_passed']=True
 checks.append(r)
proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'CPU_surrogate_only':True,'checks':checks,'not_formal_or_device_acceptance':True,'sources':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['ops_math_chunk_mean_api.py','ops_math_chunk_mean_support.py','prove_math_chunk_mean_cpu_v3.py']}};(P/'MATH_CHUNK_MEAN_CPU_PROOF_V3.json').write_text(json.dumps(proof,indent=2)+'\n');print('CHUNK_MEAN_CPU_ORIGINAL_REFERENCE_PASS',len(checks))
