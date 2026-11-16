"""Full certificate integrity, live selector and retained native-output recheck.

CPU-only: proves corrected golden/reference integration, not device execution,
input-checker/probe completion, performance validity or qualified coverage.
"""
import os, sys, importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch_npu','acl'):
            raise ModuleNotFoundError('CPU audit forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
from pathlib import Path
import json,hashlib,pickle,datetime,copy,dataclasses,importlib.util
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs'
sys.path[:0]=[str(P),str(T),str(T/'src')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from golden42_reference import SCOPE,corrected_reference_for_case
from golden42_bindings import bind
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
torch.set_num_threads(4);set_run_device('cpu')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=P/'GOLDEN42_CERTIFICATES.json';doc=json.loads(manifest.read_text())
assert doc['status']=='all42_certified' and {r['ordinal'] for r in doc['rows']}==SCOPE
spec=importlib.util.spec_from_file_location('original_canonical',T/'scripts/cache_frozen_pair_check.py')
canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical)
rows=[];native=[];negative=[]
for r in doc['rows']:
    i=r['ordinal'];d=P/'prepared-rollout'/str(i);m=json.loads((d/'prepare.json').read_text())
    for filename,key in [('payload.pt','payload_sha256'),('case.pkl','descriptor_sha256'),('prepare.json','prepare_sha256')]:
        assert sha(d/filename)==r[key]
    for key in ['original_golden','corrected_golden']:assert sha(P/r[key])==r[key+'_sha256']
    assert callable(corrected_reference_for_case(i,m))
    changed=copy.deepcopy(m);changed['payload_sha256']='0'*64
    try:corrected_reference_for_case(i,changed)
    except AssertionError:negative.append(i)
    else:raise AssertionError('Wrong payload was accepted')
    if i in (309,486):
        args=torch.load(d/'payload.pt',weights_only=False)['inputs'];sources={}
        out=P/'golden42-repair-20261116'/'selector-audit'/str(i);out.mkdir(parents=True,exist_ok=False)
        ref=bind(i,m,canonical,torch,args,sources,out)
        got=ref(*args);expected=torch.load(P/r['corrected_golden'],weights_only=False)
        assert torch.equal(got.view(torch.uint8),expected.view(torch.uint8))
        assert all(sha(path)==digest for path,digest in sources.items())
    rows.append({'ordinal':i,'identity_and_golden_hashes_checked':True,'selector_accepts_exact_identity':True,'selector_rejects_wrong_payload':True})
    # The only retained tensors used here are real old device outputs. They
    # remain old failed evidence; this report does not rewrite their run status.
    for diag in (P/'runs').glob('*/'+str(i)+'/CHECKER_DIAGNOSTIC.pt'):
        data=torch.load(diag,weights_only=False);actual=data.get('produced')
        if actual is None:continue
        saved=torch.load(d/'payload.pt',weights_only=False);case=pickle.load((d/'case.pkl').open('rb'));case.__dict__['inputs']=saved['inputs']
        if hasattr(case.workload,'tensors'):
            for name,value in zip(case._call.signature.inputs,saved['inputs'],strict=True):case.workload.tensors[name]=value
        case.__dict__.pop('verification',None)
        fixed=torch.load(P/r['corrected_golden'],weights_only=False)
        try:
            check=compare_outputs(actual,fixed,case.verification)
            assert check.checked_outputs==check.total_outputs==1 and not check.unchecked_reason
            native.append({'ordinal':i,'status':'retained_native_output_passes_corrected_golden','diagnostic':str(diag.relative_to(P)),'diagnostic_sha256':sha(diag),'check':dataclasses.asdict(check)})
        except Exception as error:
            native.append({'ordinal':i,'status':'still_fails_corrected_golden','diagnostic':str(diag.relative_to(P)),'diagnostic_sha256':sha(diag),'error':repr(error)})
try:corrected_reference_for_case(370,{})
except AssertionError:pass
else:raise AssertionError('Outside42 incorrectly accepted')
report={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','manifest_sha256':sha(manifest),'oracle_sha256':sha(P/'golden42_reference.py'),'bindings_sha256':sha(P/'golden42_bindings.py'),'audit_sha256':sha(__file__),'scope':sorted(SCOPE),'rows':rows,'negative_wrong_payload_checks_passed':len(negative),'out_of_scope_rejected':True,'live_canonical_CPU_selector_bitwise_checks':[309,486],'retained_native_output_rechecks':native,'new_qualified_coverage':0,'new_pairs':0,'new_final_closures':0,'new_performance_measurements':0,'original_NPU_golden_not_recomputed_by_CPU_audit':True,'original_golden_files_in_certificates_are_original_reference_recomputed_on_CPU':True}
out=P/'GOLDEN42_ACTIVATION_AUDIT.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'certified':len(rows),'retained_native':native,'manifest_sha256':sha(manifest)}),flush=True)
