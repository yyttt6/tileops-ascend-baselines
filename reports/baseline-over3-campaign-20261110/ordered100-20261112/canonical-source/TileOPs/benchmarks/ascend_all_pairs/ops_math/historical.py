#!/usr/bin/env python3
"""Read-only revalidation of unchanged Add provider and its existing accepted receipts."""
import hashlib,json
from specs import ROOT,EVIDENCE
matrix=json.loads((ROOT/'TileOPs/benchmarks/ascend/classic_matrix.json').read_text());out=[]
for r in matrix['rows']:
 if r['library']!='ops-math' or r['op']!='AddFwdOp' or r['status']!='verified':continue
 receipt=ROOT/r['device_receipt'];d=json.loads(receipt.read_text());review=ROOT/r['root_review']
 mismatches=[]
 for name,h in d['source_sha256'].items():
  f=ROOT/name
  if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:mismatches.append(name)
 v=d.get('verification',{});edge=d.get('edge_check',{})
 accepted=not mismatches and d['status']=='verified' and v.get('checked_outputs')==v.get('total_outputs')==1 and edge.get('signed_zero') and review.exists()
 out.append({'op':r['op'],'case_id':r['case_id'],'accepted':bool(accepted),'receipt':r['device_receipt'],'root_review':r['root_review'],'mismatches':mismatches,'source_sha256':d['source_sha256'],'reused_historical_device_evidence':True})
(EVIDENCE/'historical_add_revalidation.json').write_text(json.dumps({'schema':'ops-math-historical-source-revalidation/v1','cases':out},indent=2));print(json.dumps({'accepted':sum(r['accepted'] for r in out),'cases':len(out)}))
