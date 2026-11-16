"""Read-only CPU domain proof for the original frozen Softplus payloads."""
import os,sys,pathlib,importlib.abc,json,hashlib,datetime,gc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU domain proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
sys.path[:0]=['/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
assert torch.__version__=='2.13.0+cpu' and 'torch_npu' not in sys.modules
torch.set_num_threads(8)
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'TASKS.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
proofs=[]
for i in range(2065,2069):
 d=P/'prepared-rollout'/str(i);meta=json.loads((d/'prepare.json').read_text());p=meta['params'];row=rows[i]
 assert meta['key']==['tilelang-ascend','SoftplusFwdOp',row['case']]
 assert p.get('beta',1)==1 and p.get('threshold',20)==20
 payload=d/'payload.pt';before=sha(payload);assert before==meta['payload_sha256']
 args=torch.load(payload,weights_only=False)['inputs'];x,=args
 assert x.device.type=='cpu' and torch.isfinite(x).all().item()
 low=float(x.min());high=float(x.max());assert low>=-20 and high<=20
 assert sha(payload)==before
 proofs.append({'ordinal':i,'key':meta['key'],'params':p,'numel':x.numel(),'dtype':str(x.dtype),'min':low,'max':high,'all_finite':True,'beta':1,'threshold':20,'threshold_branch_unreachable_for_this_entire_frozen_payload':True,'payload_sha256':before,'prepare_sha256':sha(d/'prepare.json')})
 del x,args;gc.collect()
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':list(range(2065,2069)),'torch':torch.__version__,'no_NPU_imported':'torch_npu' not in sys.modules,'CPU_payload_domain_only_not_native_acceptance':True,'script_sha256':sha(pathlib.Path(__file__)),'proofs':proofs}
f=P/'SOFTPLUS_CPU_DOMAIN.json';f.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
