"""Validate byte-preserved deployed source against existing complete device receipts."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WS=HERE.parents[3]
EVIDENCE=WS/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_nn'


@lru_cache(None)
def accepted_records():
    manifest=EVIDENCE/'pinned_smoke004_acceptance.json'
    if not manifest.exists():return {}
    data=json.loads(manifest.read_text());digests={}
    def digest(path):
        path=Path(path)
        if path not in digests:
            digests[path]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        return digests[path]
    if digest(EVIDENCE/'inventory_snapshot.json')!=data['inventory_sha256']:return {}
    if any(digest(WS/name)!=sha for name,sha in data['runtime_files'].items()):return {}
    accepted={}
    for record in data['records']:
        receipt=WS/record['receipt'];audit=WS/record['native_audit']
        if digest(receipt)!=record['receipt_sha256'] or digest(audit)!=record['native_audit_sha256']:continue
        row=json.loads(receipt.read_text());proof=json.loads(audit.read_text())
        if (row.get('library')!='ops-nn' or row.get('op')!=record['op'] or row.get('case_id')!=record['case_id'] or
            proof.get('receipt')!=record['receipt'] or proof.get('numerical_receipt_sha256')!=record['receipt_sha256']):continue
        verification=row.get('verification',{})
        if not verification.get('checked_outputs') or verification['checked_outputs']!=verification.get('total_outputs'):continue
        if row.get('status')!='verified' or not proof.get('full_manifest_outputs') or proof.get('missing_apis'):continue
        if not proof.get('expected_apis') or not set(proof['expected_apis'])<=set(proof['proven_apis']):continue
        relocated=record['runtime_source_relocation']
        if any(digest(WS/relocated.get(name,{}).get('relocated_file',name))!=sha
               for name,sha in row['source_sha256'].items()):continue
        accepted[(record['op'],record['case_id'])]=record
    return accepted
