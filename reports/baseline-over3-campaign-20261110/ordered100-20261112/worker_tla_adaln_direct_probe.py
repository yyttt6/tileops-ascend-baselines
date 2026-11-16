"""Functional-only direct original LayerNorm probe; no performance acceptance."""
import pathlib,sys,os,json,hashlib,time,fcntl
P=pathlib.Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs'),str(P/'workspace/isolated_wrappers')];tag=sys.argv[1]
f=open('/run/lock/ascend-npu/device-0.lock');
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:pass
else:raise RuntimeError('Whole-process guard required')
import torch,torch_npu
from types import SimpleNamespace
from benchmarks.ascend_all_pairs.tla import adapters,native
import tla_cache_artifact_identity as identity
import tla_direct_norm_selector as selector
identity.install();native.prepare_deployment();torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
out={'status':'started','checks':[],'not_formal_acceptance':True,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'source':{}};dest=P/'direct-norm-probes'/tag;dest.mkdir(parents=True,exist_ok=False)
def put(): (dest/'RESULT.json').write_text(json.dumps(out,indent=2))
try:
 for shape in [(64,1152)]:
  print('PROBE_FACTORY',shape,flush=True);x=torch.randn(shape,dtype=torch.float32,device='npu');snapshot=x.clone();case=SimpleNamespace(params={'x':(shape,'float32'),'output':(shape,'float32'),'normalized_shape':(shape[-1],),'eps':1e-5},inputs=(x,));child=selector.select_direct(adapters)('LayerNormFwdOp',case);out['source'].update(child.source_sha256);print('PROBE_RUN',shape,flush=True);got=child.implementation.run(x);torch.npu.synchronize();expected=torch.nn.functional.layer_norm(x,(shape[-1],),eps=1e-5);torch.testing.assert_close(got,expected,rtol=1e-5,atol=1e-5);assert torch.equal(x,snapshot);out['checks'].append({'shape':shape,'dtype':'float32','eps':1e-5,'all_outputs_checked':True,'inputs_bitwise_unchanged':True,'maxabs':float((got-expected).abs().max().item())});
  if child.cleanup:child.cleanup()
 for p in [pathlib.Path(__file__),pathlib.Path(selector.__file__),pathlib.Path(identity.__file__)]:out['source'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 out['status']='direct_original_API_full_output_and_input_passed';print('PROBE_PASSED',flush=True)
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e));raise
finally:
 put();torch.npu.synchronize()
