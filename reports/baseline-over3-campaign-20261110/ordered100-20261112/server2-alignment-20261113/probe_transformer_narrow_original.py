import os,sys,json,datetime,hashlib
from pathlib import Path
R=Path('/opt/baseline-over3-20261110/ordered100-20261112');N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-narrow-v1')
os.environ['ASCEND_WORK_PATH']=str(N/'probe-jit-work'/os.environ['NARROW_PROBE_TAG']);sys.path[:0]=[str(N),str(R/'workspace/TileOPs'),str(R/'workspace/TileOPs/src')]
import torch,torch_npu
from ops_transformer_narrow_graph_api import GraphCall
binding=GraphCall('ops-transformer');results=[]
for dtype,width in [(torch.float32,17),(torch.float16,17),(torch.bfloat16,17),(torch.float32,1024)]:
 x=torch.linspace(-2,2,3*width,dtype=torch.float32).reshape(3,width).to(dtype).npu();before=x.cpu().view(torch.uint8).clone();y=torch.full_like(x,float('nan'));idx=torch.full(x.shape,-1,dtype=torch.int32,device=x.device);rows=torch.full_like(idx,-1)
 binding.call('MoeGatingTopKSoftmax',[x,None],[y,idx,rows],{'k':width});torch.npu.synchronize();actual=y.cpu().scatter(1,idx.cpu().long(),y.cpu());expected=torch.softmax(x.cpu().float(),dim=-1).to(dtype)
 torch.testing.assert_close(actual,expected,rtol=1e-5 if dtype==torch.float32 else .01,atol=1e-6 if dtype==torch.float32 else .001)
 assert torch.equal(torch.sort(idx.cpu(),dim=1).values,torch.arange(width,dtype=torch.int32).expand(3,width));assert torch.equal(rows.cpu(),torch.arange(3*width,dtype=torch.int32).reshape(width,3).t());assert torch.equal(x.cpu().view(torch.uint8),before)
 results.append({'dtype':str(dtype),'width':width,'all3outputs_and_input_bits_passed':True});binding.cleanup()
# No performance/coverage acceptance; preserve process JIT products before cleanup.
import shutil
work=Path(os.environ['ASCEND_WORK_PATH']);dest=N/(os.environ['NARROW_PROBE_TAG']+'-JIT');dest.mkdir(exist_ok=False);saved={}
for p in work.rglob('*'):
 if p.is_file() and p.suffix in ['.o','.json','.i']:
  target=dest/p.relative_to(work);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);saved[str(target.relative_to(dest))]=hashlib.sha256(target.read_bytes()).hexdigest()
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixture_results':results,'JIT_files':saved,'not_performance_or_coverage':True};(N/(os.environ['NARROW_PROBE_TAG']+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report));torch.npu.synchronize()
