"""Two complete original graph APIs fixtures, full3outputs; not performance acceptance."""
from pathlib import Path
import os,json,hashlib,traceback,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-transformer-selection-v1');g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['child']['pid']==os.getpgrp();tag=g['owner'].rsplit('/',1)[1];assert tag.startswith('server1-transformer-selection-probe-');dest=N/(tag+'.json');assert not dest.exists();m=json.loads((N/'TRANSFORMER_SELECTION_PACKAGE.json').read_text());vendor=Path(m['vendor']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,r in m['files'].items():assert sha(vendor/name)==r['sha256']
out={'status':'initializing','zero_formal_coverage':True,'checks':[],'manifest_sha256':sha(N/'TRANSFORMER_SELECTION_PACKAGE.json'),'source_sha256':sha(__file__)}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from ops_transformer_selection_graph_api import GraphCall
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});p=torch.npu.get_device_properties(0);assert (p.cube_core_num,p.vector_core_num)==(32,64);binding=GraphCall('ops-transformer',vendor_name='selectionpairs_transformer')
 for width in [17,2048]:
  cpu=(torch.arange(3*width,dtype=torch.float32).reshape(3,width)%61-30)/13;cpu=cpu+torch.arange(width,dtype=torch.float32).reshape(1,width)/width/7;x=cpu.to('npu');weights=torch.empty((3,1),dtype=torch.float32,device='npu');ids=torch.empty((3,1),dtype=torch.int32,device='npu');full=torch.empty_like(x);out.update(status='before_full_API',width=width);save();binding.call('MoeGatingTopKSoftmaxV2',[x,None],[weights,ids,full],{'k':1,'renorm':0,'output_softmax_result_flag':True});binding.cleanup();torch.npu.synchronize();gold=cpu.softmax(dim=-1);torch.testing.assert_close(full.cpu(),gold);v,k=gold.topk(1,dim=-1);torch.testing.assert_close(weights.cpu(),v);assert torch.equal(ids.cpu().long(),k);assert torch.equal(x.cpu().view(torch.uint8),cpu.view(torch.uint8));out['checks'].append({'width':width,'all3_outputs_checked':True,'input_bits_preserved':True,'original_complete_API':True});save()
 out['status']='full_original_V2_graph_API_fixtures_passed_pending_private_JIT_attribution'
except BaseException as e:out.update(status='failed_requires_diagnosis_no_retry',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(tag+'.maps.txt')).write_text(maps);out['loaded_library_sha256']={l.split()[-1]:sha(l.split()[-1]) for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1] and Path(l.split()[-1]).is_file()};save()
