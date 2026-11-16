"""CPU mathematical diagnostic; original oracle and every device implementation untouched."""
import os,sys,pathlib,importlib.abc,json,pickle,hashlib,datetime,time,gc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();started=time.time();scope=[270,272,273,274,276,277,278,279,280,281,282,284,294,295,296,297,298];rows=[];out=P/'REMAINING_MATH_MATRIX_REFERENCE_CPU_DIAGNOSIS.json';assert not out.exists()
def save():out.write_text(json.dumps({'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':scope,'rows':rows,'CPU_only_no_NPU_import':True,'seconds':time.time()-started,'source_sha256':sha(__file__),'diagnostic_only_no_reference_or_precision_or_checker_repair':True},indent=2))
for i in scope:
 begin=time.time();d=P/'prepared-rollout'/str(i);m=json.loads((d/'prepare.json').read_text());assert m['reference_policy']=='original_cpu';assert sha(d/'payload.pt')==m['payload_sha256'] and sha(d/'case.pkl')==m['descriptor_sha256'];saved=torch.load(d/'payload.pt',weights_only=False);args=saved['inputs'];case=pickle.load((d/'case.pkl').open('rb'));case.__dict__['inputs']=args
 if hasattr(case.workload,'tensors'):
  for name,value in zip(case._call.signature.inputs,args,strict=True):case.workload.tensors[name]=value
 case.__dict__.pop('verification',None);gold=case.reference(*args);tol=0.016 if gold.dtype==torch.bfloat16 else 0.001;badcount=0;maxabs=0.;first=None;p=m['params']
 def check(lhs,rhs,value,where):
  global badcount,maxabs,first
  assert tuple(value.shape)==(lhs.shape[0],rhs.shape[1]);right=rhs.double()
  for left in range(0,lhs.shape[0],128):
   stop=min(left+128,lhs.shape[0]);truth=(lhs[left:stop].double()@right).to(value.dtype);part=value[left:stop];diff=(part.float()-truth.float()).abs();bad=diff>tol+tol*truth.float().abs();badcount+=int(bad.sum());maxabs=max(maxabs,float(diff.max())) if diff.numel() else maxabs
   if bad.any() and first is None:
    xy=torch.nonzero(bad)[0].tolist();first=where|{'row_in_group':left+xy[0],'column':xy[1],'original':float(part[tuple(xy)]),'FP64_rounded':float(truth[tuple(xy)])}
 if i<294:
  a,b=args;check(a.T if p['trans_a'] else a,b.T if p['trans_b'] else b,gold,{})
 else:
  a,b,sizes,offsets=args;start=0
  for group,size in enumerate(sizes.tolist()):
   end=start+int(size)
   if not p['transpose_a']:check(a[start:end],b[group].T if p['transpose_b'] else b[group],gold[start:end],{'group':group,'output_row_offset':start})
   else:check(a[start:end].T,b[:,start:end].T if p['transpose_b'] else b[start:end],gold[group],{'group':group})
   start=end
  assert start==a.shape[0]
 row={'ordinal':i,'params':p,'payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256'],'original_reference_policy':m['reference_policy'],'original_CPU_reference_vs_FP64':{'mismatched':badcount,'elements':gold.numel(),'max_absolute_difference':maxabs,'first_bad':first,'atol':tol,'rtol':tol},'seconds':time.time()-begin};rows.append(row);save();print(json.dumps({'ordinal':i,'mismatched':badcount,'elements':gold.numel(),'seconds':row['seconds']}),flush=True);del case,saved,args,gold;gc.collect()
