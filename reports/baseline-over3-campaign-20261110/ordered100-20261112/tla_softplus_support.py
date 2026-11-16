"""Frozen, fully CPU-domain-checked Softplus through complete original APIs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='SoftplusFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  import tla_hardsigmoid_support as original_scalar
  domainfile=P/'SOFTPLUS_CPU_DOMAIN.json';domain=json.loads(domainfile.read_text())
  assert domain['no_NPU_imported'] and domain['script_sha256']==sha(P/'softplus_cpu_domain.py')
  records=[r for r in domain['proofs'] if r['key']==[library,'SoftplusFwdOp',case.id]]
  assert len(records)==1;record=records[0]
  assert record['all_finite'] and -20<=record['min']<=record['max']<=20
  assert record['beta']==1 and record['threshold']==20
  assert json.loads(json.dumps(case.params))==record['params']
  prep=P/'prepared-rollout'/str(record['ordinal'])/'prepare.json';meta=json.loads(prep.read_text())
  assert sha(prep)==record['prepare_sha256'] and meta['payload_sha256']==record['payload_sha256']
  assert case.params['beta']==1 and case.params['threshold']==20
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16)
  shape=tuple(case.params['output'][0]);assert shape==tuple(x.shape)
  params={'input':(shape,'float32'),'output':(shape,'float32')}
  desc=SimpleNamespace(params=params,inputs=(x,))
  exp=adapters.elementwise('ExpFwdOp',desc)
  plus=original_scalar.scalar_factory(adapters,'adds',1.)('NegFwdOp',desc)
  log=adapters.elementwise('LogFwdOp',SimpleNamespace(params={**params,'output':case.params['output']},inputs=(x,)))
  children=(exp,plus,log)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   result=exp.implementation.run(value)
   result=plus.implementation.run(result)
   return log.implementation.run(result)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(original_scalar.__file__),Path(__file__),domainfile,P/'softplus_cpu_domain.py',prep):
   hashes[str(f)]=sha(f)
  return Candidate(Implementation(run),'TLA original complete exp + original complete adds(1) + original complete ln','Frozen Softplus beta=1/threshold=20 only: entire original payload is CPU-verified finite within [-20,20], binding original payload and descriptor hashes; threshold branch is unreachable on this frozen input. Complete original FP32 exp/adds/ln, every pack/pad/native/crop/intermediate and original output cast timed. No input/oracle/kernel/block/passflags/checker changes. Not generic Softplus or extreme-domain acceptance; original full device checker still required.',hashes,cleanup=cleanup)
 return build
