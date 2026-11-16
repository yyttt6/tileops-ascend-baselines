"""CPU-only exact-input reference repair and full-output certification."""
import os, sys, importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
class NoNPU(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch_npu','acl'):
            raise ModuleNotFoundError('Golden certification forbids '+name, name=name)
sys.meta_path.insert(0, NoNPU())
from pathlib import Path
import json, pickle, hashlib, datetime, time, gc, dataclasses
P = Path(__file__).resolve().parent
T = P/'workspace/TileOPs'
sys.path[:0] = [str(P),str(T),str(T/'src')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
from workloads.numerics import compare_outputs, reference_tolerance
from golden42_reference import SCOPE, mathematical_reference
set_run_device('cpu')
torch.set_num_threads(4)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
O = P/'golden42-repair-20261116'
O.mkdir(exist_ok=True)
report = P/'GOLDEN42_CERTIFICATES.json'
source_sha = sha(P/'golden42_reference.py')
doc = {'started_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'status':'running','scope':sorted(SCOPE),'oracle_source_sha256':source_sha,
       'certifier_sha256':sha(__file__),'torch':torch.__version__,
       'CPU_only_NPU_import_forbidden':True,'threads':4,'rows':[],
       'route1':'full-K FP64 matmul/einsum or centered two-pass variance, original output dtype',
       'route2':'reverse split-K (127) FP64 or chunk-Welford variance, original output dtype',
       'baseline_accepted_by_this_report':False,'new_performance_measurements':0}
if report.exists():
    prior=json.loads(report.read_text())
    assert prior['oracle_source_sha256']==source_sha and prior['certifier_sha256']==sha(__file__)
    doc=prior
def save():
    doc['updated_at_remote']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(doc,indent=2,default=str)+'\n');os.replace(tmp,report)
for ordinal in sorted(SCOPE):
    if any(r['ordinal']==ordinal and r['status']=='certified' for r in doc['rows']):continue
    start=time.time();d=P/'prepared-rollout'/str(ordinal)
    m=json.loads((d/'prepare.json').read_text())
    assert sha(d/'payload.pt')==m['payload_sha256'] and sha(d/'case.pkl')==m['descriptor_sha256']
    inputs=torch.load(d/'payload.pt',weights_only=False)['inputs']
    case=pickle.load((d/'case.pkl').open('rb'));case.__dict__['inputs']=inputs
    if hasattr(case.workload,'tensors'):
        for name,value in zip(case._call.signature.inputs,inputs,strict=True):case.workload.tensors[name]=value
    case.__dict__.pop('verification',None)
    before=[x.clone() if isinstance(x,torch.Tensor) else x for x in inputs]
    original=case.reference(*inputs)
    fixed=mathematical_reference(m['key'][1],m['params'],inputs)
    independent=mathematical_reference(m['key'][1],m['params'],inputs,True)
    assert fixed.shape==independent.shape==original.shape and fixed.dtype==independent.dtype==original.dtype
    check=compare_outputs(fixed,independent,case.verification)
    assert check.checked_outputs==check.total_outputs==1 and not check.unchecked_reason
    for old,new in zip(before,inputs,strict=True):
        if isinstance(old,torch.Tensor):assert torch.equal(old.view(torch.uint8),new.view(torch.uint8))
    path=O/str(ordinal);path.mkdir(exist_ok=False)
    torch.save(original,path/'original_golden.pt');torch.save(fixed,path/'corrected_golden.pt')
    orig=original.flatten().double();new=fixed.flatten().double();alt=independent.flatten().double()
    tolerance=case.verification
    assert tolerance.kind=='exact', 'Must retain the original declared Exact evidence'
    resolved=tolerance.tolerance(reference_tolerance(fixed.dtype))
    atol,rtol=resolved['atol'],resolved['rtol']
    bad=(orig-new).abs()>atol+rtol*new.abs()
    row={name:m[name] for name in ('key','params','payload_sha256','descriptor_sha256')}
    row.update(ordinal=ordinal,status='certified',original_reference_policy=m['reference_policy'],prepare_sha256=sha(d/'prepare.json'),
               evidence=dataclasses.asdict(tolerance),resolved_original_tolerance=resolved,full_output_route_comparison=dataclasses.asdict(check),
               independently_rounded_different_elements=int((new!=alt).sum()),
               original_vs_corrected_exceeding_original_tolerance=int(bad.sum()),output_elements=fixed.numel(),
               original_golden=str((path/'original_golden.pt').relative_to(P)),original_golden_sha256=sha(path/'original_golden.pt'),
               corrected_golden=str((path/'corrected_golden.pt').relative_to(P)),corrected_golden_sha256=sha(path/'corrected_golden.pt'),
               seconds=time.time()-start,certified_at_remote=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert sha(d/'payload.pt')==m['payload_sha256'] and sha(d/'case.pkl')==m['descriptor_sha256']
    doc['rows'].append(row);save();print(json.dumps({'ordinal':ordinal,'status':row['status'],'old_bad':int(bad.sum()),'route_differences':row['independently_rounded_different_elements'],'seconds':row['seconds']}),flush=True)
    del inputs,case,before,original,fixed,independent,orig,new,alt,bad;gc.collect()
assert len(doc['rows'])==42 and {r['ordinal'] for r in doc['rows']}==SCOPE
doc['status']='all42_certified';save()
