"""Check exact-case continuation receipts without importing a device runtime."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WS=HERE.parents[3]

@lru_cache(None)
def accepted_records():
    manifest=HERE/'accepted_continuation_cases.json'
    if not manifest.is_file():return {}
    data=json.loads(manifest.read_text());digests={};accepted={}
    def sha(p):
        p=Path(p)
        if p not in digests:
            digests[p]=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        return digests[p]
    for record in data['records']:
        receipt=WS/record['receipt'];review=WS/record['review']
        if sha(receipt)!=record['receipt_sha256'] or sha(review)!=record['review_sha256']:continue
        row=json.loads(receipt.read_text());proof=json.loads(review.read_text())
        identity=(record['op'],record['case_id'])
        if (row.get('op'),row.get('case_id'))!=identity or row.get('library')!='ops-nn':continue
        if (proof.get('op'),proof.get('case_id'))!=identity:continue
        if proof.get('receipt')!=record['receipt'] or proof.get('receipt_sha256')!=record['receipt_sha256']:continue
        check=row.get('verification',{})
        if row.get('status')!='verified' or not check.get('checked_outputs') or check['checked_outputs']!=check.get('total_outputs'):continue
        if not proof.get('accepted_by_worker') or not proof.get('full_outputs') or not proof.get('original_seed_matches'):continue
        if not proof.get('expected_apis') or not set(proof['expected_apis'])<=set(proof.get('proven_apis',[])):continue
        native=proof.get('native_evidence',{})
        if not set(proof['expected_apis'])<=native.keys():continue
        binary_ok=True
        for api in proof['expected_apis']:
            artifact=native[api].get('cached_binary',{})
            binary=Path(artifact.get('binary','/nonexistent'))
            expected=artifact.get('binary_sha256')
            alternatives=[binary,WS/'.cache/all-pairs/ops_nn/graph_cache/kernel_cache/Ascend950DT_9592'/binary.name]
            if not expected or not any(sha(p)==expected for p in alternatives):
                binary_ok=False;break
        if not binary_ok:continue
        guard=proof.get('guard') or {}
        if guard.get('state')!='terminal' or guard.get('exit_code')!=0:continue
        if not row.get('source_sha256') or any(sha(WS/name)!=value for name,value in row['source_sha256'].items()):continue
        accepted[identity]={**record,'api':row['api'],'source_sha256':row['source_sha256']}
    return accepted
