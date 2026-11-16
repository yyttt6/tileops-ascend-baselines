"""CPU-only handoff integrity checks, independent of fake tensor execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

from . import PROVIDERS
from . import native

HERE=Path(__file__).resolve().parent


def main():
    checks=[]
    def passed(name,detail):checks.append({'check':name,'status':'pass','detail':detail})
    sources=list(HERE.glob('*.py'))
    for path in sources:compile(path.read_text(),str(path),'exec',dont_inherit=True)
    passed('syntax',f'{len(sources)} own Python modules; in-memory compile only')
    code="""
import importlib.abc,sys,inspect
class ForbidRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torch_npu','tilelang','acl','tvm'}:
            raise RuntimeError('Forbidden eager runtime import: '+fullname)
sys.meta_path.insert(0,ForbidRuntime())
from benchmarks.ascend_all_pairs.tla import PROVIDERS
assert len(PROVIDERS)==84
assert not any(n in sys.modules for n in ('torch','torch_npu','tilelang','acl','tvm'))
for factory in PROVIDERS.values():
    assert list(inspect.signature(factory).parameters)==['case']
print('84 lazy one-argument provider factories; no real runtime imports')
"""
    check=subprocess.run([sys.executable,'-B','-c',code],cwd=native.WORKSPACE/'TileOPs',
                         capture_output=True,text=True,check=True)
    passed('entry_lazy_import_and_factory_signature',check.stdout.strip())
    inventory=json.loads((HERE/'inventory_snapshot.json').read_text())
    coverage=json.loads((HERE/'coverage.json').read_text())
    receipts=json.loads((HERE/'static_case_receipts.json').read_text())['cases']
    static=json.loads((HERE/'static_validation.json').read_text())
    keys=lambda rows:{(r['op'],r['case_id']) for r in rows}
    assert keys(inventory['cases'])==keys(coverage['cases'])
    assert len(coverage['cases'])==len(keys(coverage['cases']))==2062
    assert {r['op'] for r in coverage['operators']}=={r['op'] for r in inventory['operators']}
    assert len(coverage['operators'])==205
    passed('coverage_completeness','205 unique operators; 2062 unique manifest case identities')
    supported=[r for r in coverage['cases'] if r['case_adapter_available']]
    assert keys(supported)==keys(receipts)
    assert {r['op'] for r in supported}==set(PROVIDERS)
    assert static['factory_metadata_pass']==static['run_metadata_pass']==len(supported)==715
    assert static['metadata_failures']==0
    passed('registered_contracts','84 providers and 715 complete metadata-shape/dtype/write-provenance receipts')
    hashes={}
    for row in receipts:
        for path,sha in row['source_sha256'].items():
            if path in hashes:assert hashes[path]==sha,(path,'inconsistent receipt hashes')
            hashes[path]=sha
    for rows in (coverage['operators'],coverage['cases']):
        for row in rows:
            for source in row.get('sources',row.get('source_evidence',[])):
                path=source['file'];sha=source['sha256']
                if path in hashes:assert hashes[path]==sha
                hashes[path]=sha
    for path,sha in hashes.items():
        assert hashlib.sha256((native.WORKSPACE/path).read_bytes()).hexdigest()==sha,path
    passed('source_attribution_integrity',f'{len(hashes)} current adapter/vendor source hashes match receipts and coverage')
    catalog=json.loads((HERE/'source_catalog.json').read_text())
    owned=[row for row in catalog if '/3rdparty/' not in row['file']]
    for row in owned:
        assert hashlib.sha256((native.WORKSPACE/row['file']).read_bytes()).hexdigest()==row['sha256'],row['file']
    passed('vendor_corpus_preserved',f'{len(owned)} owned source catalogue files unchanged')
    assert coverage['summary']['device_runs']==coverage['summary']['verified_cases']==0
    for row in coverage['cases']:
        assert not row['verified'] and row['device_runs']==0
        assert row['numerical_receipt_path'] is None
        assert row['input_protection_device_receipt'] is None
        assert row['native_ownership_device_receipt'] is None
    assert not static['real_runtime_imports'] and not static['numerical_validation']
    passed('no_false_device_claim','All numerical, native ownership and input-protection device receipts are null')
    assert coverage['summary']['case_status_counts']==dict(Counter(r['status'] for r in coverage['cases']))
    assert hashlib.sha256((HERE/'inventory_snapshot.json').read_bytes()).hexdigest()==coverage['inventory_snapshot']['sha256']
    passed('schema_counts_and_snapshot_hash','Semantic and deployment statuses counted separately; snapshot hash matches')
    report={'schema':'tla-final-cpu-check/v1','generated_utc':datetime.now(timezone.utc).isoformat(),
            'npu_used':False,'numerical_validation':False,'checks':checks,
            'result':'pass','coverage_sha256':hashlib.sha256((HERE/'coverage.json').read_bytes()).hexdigest()}
    (HERE/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
